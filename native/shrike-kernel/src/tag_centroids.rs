//! Tag-centroid vectors (#178/#179): each curated tag becomes a
//! content-grounded concept vector — the **renormalized mean of its member
//! notes' TEXT vectors** — stored in the engine's `tag.text` space, keyed by
//! a stable hash of the tag string. Membership is exact (one pass over
//! `notes.tags`), with hierarchy rolled up by prefix aggregation
//! (`a::b::c` contributes to `a::b::c`, `a::b`, and `a`).
//!
//! Layout decision (#178, recorded in docs/decisions.md): the tag space lives
//! in the SAME engine as the note-item spaces (same model/dim/metric — a tag
//! centroid is only meaningful in the notes' space) under a distinct space
//! name, so note searches scoped to [`crate::NOTE_MODALITIES`] can never
//! surface a tag key structurally. The key→tag map is in-memory only and
//! rebuilt with the centroids (cheap); the persisted `index.tag.text.usearch`
//! survives restarts but the signal stays off until the first recompute —
//! which boot triggers — so a stale map can never mislabel a key.
//!
//! Consistency: centroids are a pure function of the note TEXT vectors
//! already in the engine plus the membership map, so they recompute (whole
//! set — typically hundreds of tags) at the tail of every index-changing op.
//! No separate watermark, no sidecar.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest};

use shrike_ffi::{NativeError, NativeResult};
use shrike_store_api::VectorIndex;

use crate::TAG_TEXT_SPACE;

/// Hygiene knobs (#179: "a curation surface — make them configurable"). The
/// defaults live here; the harness threads overrides through
/// [`TagCentroidConfig`].
pub const DEFAULT_MIN_MEMBERS: usize = 2;
pub const DEFAULT_MAX_COVERAGE: f64 = 0.5;
pub const DEFAULT_BLOCKLIST: &[&str] = &["leech", "marked"];

/// Per-tag ceiling on members scored during query expansion (#445): a huge
/// tag — approaching the whole collection at 100k notes — would otherwise
/// pay a vector read + dot product per member on every query it activates
/// for. Tags over the ceiling are stride-sampled (deterministic, spread
/// across the member range); a tag that big is weakly informative as a
/// concept anyway, and the expansion feeds a 50-slot cap.
pub const MEMBER_SCORE_CEILING: usize = 4096;

/// The hygiene filter configuration: which tags are *concepts* worth a vector.
#[derive(Debug, Clone)]
pub struct TagCentroidConfig {
    /// Tags with fewer members are skipped (a 1-member centroid IS the note).
    pub min_members: usize,
    /// Tags covering more than this fraction of all notes are structural
    /// (deck-org, source markers), not conceptual.
    pub max_coverage: f64,
    /// Known meta-tags, matched case-insensitively against each `::` segment
    /// (so `foo::leech` is excluded too).
    pub blocklist: Vec<String>,
}

impl Default for TagCentroidConfig {
    fn default() -> Self {
        Self {
            min_members: DEFAULT_MIN_MEMBERS,
            max_coverage: DEFAULT_MAX_COVERAGE,
            blocklist: DEFAULT_BLOCKLIST.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl TagCentroidConfig {
    fn blocked(&self, tag: &str) -> bool {
        tag.split("::")
            .any(|seg| self.blocklist.iter().any(|b| seg.eq_ignore_ascii_case(b)))
    }
}

/// Stable i64 key for a tag string (blake2b-8; masked positive so keys can
/// never collide with the sign conventions of note ids). Fixed-size
/// `Blake2b<U8>` — the same digest-length parameter block (so the same
/// bytes) as `Blake2bVar::new(8)`, without the per-call fallible
/// construction (#382).
pub fn tag_key(tag: &str) -> i64 {
    let out = Blake2b::<U8>::digest(tag.as_bytes());
    (i64::from_be_bytes(out.into())) & i64::MAX
}

/// Hierarchy roll-up: every `::` prefix of every leaf tag, including the leaf.
fn expand_hierarchy(leaf: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut prefix = String::new();
    for segment in leaf.split("::") {
        if !prefix.is_empty() {
            prefix.push_str("::");
        }
        prefix.push_str(segment);
        out.push(prefix.clone());
    }
    out
}

/// Membership map from the one-pass `(note_id, leaf tags)` rows, hierarchy
/// rolled up; each tag's member list is deduped.
pub fn membership(rows: &[(i64, Vec<String>)]) -> BTreeMap<String, Vec<i64>> {
    let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for (note_id, leaves) in rows {
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for leaf in leaves {
            for tag in expand_hierarchy(leaf) {
                if seen.insert(tag.clone()) {
                    map.entry(tag).or_default().push(*note_id);
                }
            }
        }
    }
    map
}

/// The live tag state for the centroids currently in the engine's tag space:
/// key→tag names plus key→member note ids (the retrieval signal's expansion
/// set). Empty until the first recompute (the persisted space is searched
/// only through this state, so a stale file can't mislabel keys or expand to
/// stale members).
#[derive(Default)]
pub struct TagKeyMap {
    inner: RwLock<TagState>,
}

#[derive(Default)]
struct TagState {
    names: BTreeMap<i64, String>,
    members: BTreeMap<i64, Vec<i64>>,
}

impl TagKeyMap {
    pub fn lookup(&self, key: i64) -> Option<String> {
        self.inner
            .read()
            .expect("tag keys poisoned")
            .names
            .get(&key)
            .cloned()
    }

    /// The member note ids behind a centroid key (empty if unknown).
    pub fn members(&self, key: i64) -> Vec<i64> {
        self.inner
            .read()
            .expect("tag keys poisoned")
            .members
            .get(&key)
            .cloned()
            .unwrap_or_default()
    }

    /// The not-yet-`seen` members behind a centroid key, stride-sampled down
    /// to `ceiling` under the read lock (#445) — the expansion's bounded
    /// working set, never the full member clone `members()` hands out. The
    /// stride is deterministic and spreads the sample across the member
    /// range rather than biasing toward the lowest note ids.
    fn fresh_members(
        &self,
        key: i64,
        seen: &std::collections::BTreeSet<i64>,
        ceiling: usize,
    ) -> Vec<i64> {
        let state = self.inner.read().expect("tag keys poisoned");
        let Some(members) = state.members.get(&key) else {
            return Vec::new();
        };
        let fresh = members.iter().filter(|nid| !seen.contains(nid));
        let n = fresh.clone().count();
        if n <= ceiling {
            return fresh.copied().collect();
        }
        // ceil-stride keeps the sample at or under the ceiling.
        fresh.copied().step_by(n.div_ceil(ceiling)).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.read().expect("tag keys poisoned").names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether any of `ids` is currently a member of any tag — the in-memory
    /// half of the op-tail relevance probe (#445): a delete (or an update
    /// that removed tags) changes membership only if the note was IN it.
    /// One pass over the member lists (~sub-ms even at 100k members).
    pub fn any_member_of(&self, ids: &[i64]) -> bool {
        if ids.is_empty() {
            return false;
        }
        let set: std::collections::BTreeSet<i64> = ids.iter().copied().collect();
        let state = self.inner.read().expect("tag keys poisoned");
        state
            .members
            .values()
            .flatten()
            .any(|nid| set.contains(nid))
    }

    fn replace(&self, names: BTreeMap<i64, String>, members: BTreeMap<i64, Vec<i64>>) {
        *self.inner.write().expect("tag keys poisoned") = TagState { names, members };
    }
}

/// Coalescing background refresher (#445): write-op tails previously ran the
/// full centroid recompute INLINE — O(tagged-notes) on every upsert/delete,
/// serialized on the op. `request` returns immediately: the first request
/// spawns a refresh right away (an isolated op's centroids land as fast as
/// the inline call produced them, just off the tail), and requests arriving
/// while one runs coalesce into ONE follow-up after `window` — so a burst of
/// N ops costs a handful of recomputes, not N. A refresh is a pure function
/// of current collection + engine state, so the coalesced run sees
/// everything the skipped ones would have.
pub struct TagRefresher {
    collection: Arc<crate::SerializedCollection>,
    engine: Arc<dyn VectorIndex>,
    keys: Arc<TagKeyMap>,
    config: TagCentroidConfig,
    saver: Arc<crate::index_orchestrator::DebouncedSaver>,
    embed: Arc<std::sync::RwLock<crate::EmbedSpaces>>,
    window: std::time::Duration,
    state: std::sync::Mutex<RefreshState>,
}

#[derive(Default)]
struct RefreshState {
    running: bool,
    dirty: bool,
    /// The in-flight task — aborted on shutdown so a sleeping follow-up
    /// never outlives the kernel's collection actor.
    task: Option<tokio::task::AbortHandle>,
}

/// Pacing between coalesced follow-up refreshes under sustained write
/// traffic (the first refresh of a quiet period runs immediately).
pub const TAG_REFRESH_WINDOW: f64 = 2.0;

impl TagRefresher {
    pub fn new(
        collection: Arc<crate::SerializedCollection>,
        engine: Arc<dyn VectorIndex>,
        keys: Arc<TagKeyMap>,
        config: TagCentroidConfig,
        saver: Arc<crate::index_orchestrator::DebouncedSaver>,
        embed: Arc<std::sync::RwLock<crate::EmbedSpaces>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            collection,
            engine,
            keys,
            config,
            saver,
            embed,
            window: std::time::Duration::from_secs_f64(TAG_REFRESH_WINDOW),
            state: std::sync::Mutex::new(RefreshState::default()),
        })
    }

    /// Note a membership-relevant change. Never blocks, never errors: the
    /// refresh is best-effort by contract (the tag layer is
    /// conditionally-present and must not fail the op it rides on).
    pub fn request(self: &Arc<Self>) {
        {
            let mut st = self.state.lock().expect("tag refresher poisoned");
            if st.running {
                st.dirty = true;
                return;
            }
            st.running = true;
            st.dirty = false;
        }
        let this = Arc::clone(self);
        let task = crate::runtime::handle().spawn(async move {
            loop {
                this.run_once().await;
                {
                    let mut st = this.state.lock().expect("tag refresher poisoned");
                    if !st.dirty {
                        st.running = false;
                        st.task = None;
                        break;
                    }
                    st.dirty = false;
                }
                tokio::time::sleep(this.window).await;
            }
        });
        // The task may already have finished (and cleared itself); storing a
        // finished handle is harmless — abort on a completed task is a no-op.
        self.state.lock().expect("tag refresher poisoned").task = Some(task.abort_handle());
    }

    /// Abort any in-flight/scheduled refresh (kernel close): the collection
    /// actor is about to drain, and a late follow-up has nothing to read.
    pub fn shutdown(&self) {
        let mut st = self.state.lock().expect("tag refresher poisoned");
        st.dirty = false;
        st.running = false;
        if let Some(task) = st.task.take() {
            task.abort();
        }
    }

    async fn run_once(&self) {
        if self.embed.read().expect("embed slot poisoned").is_empty() {
            return; // no embedder → no text vectors to mean over
        }
        let result: NativeResult<()> = async {
            let (rows, total) = self
                .collection
                .run(|core| -> NativeResult<_> { Ok((core.note_tag_rows()?, core.note_count()?)) })
                .await??;
            // `recompute` is O(collection) CPU + engine reads/writes (a vector
            // per distinct tagged note, then a wholesale tag-space rebuild). It
            // MUST NOT run on a runtime worker (#445: blocking/compute work rides
            // `spawn_blocking`, never a worker thread) — at 100k notes it would
            // stall the worker for the whole recompute. Hand it to the blocking
            // pool with owned/Arc captures.
            let engine = Arc::clone(&self.engine);
            let keys = Arc::clone(&self.keys);
            let config = self.config.clone();
            tokio::task::spawn_blocking(move || recompute(&*engine, &rows, total, &config, &keys))
                .await
                .map_err(|e| NativeError::internal(format!("tag recompute task: {e}")))??;
            Ok(())
        }
        .await;
        match result {
            Ok(()) => self.saver.request_save(),
            Err(e) => tracing::warn!(error = ?e, "tag centroid refresh failed"),
        }
    }
}

/// Recompute every tag centroid from the engine's note TEXT vectors and
/// replace the `tag.text` space wholesale. Returns the centroid count.
/// Centroid = renormalized mean (the mean of unit vectors is not unit-norm).
pub fn recompute(
    engine: &dyn VectorIndex,
    rows: &[(i64, Vec<String>)],
    total_notes: usize,
    config: &TagCentroidConfig,
    keys: &TagKeyMap,
) -> NativeResult<usize> {
    let members_by_tag = membership(rows);
    // Each distinct member's text vector is fetched ONCE (#445): a note in N
    // tags (and every `::` prefix) previously paid N engine lock + copy round
    // trips — ~200k fetches per recompute at 100k notes with hierarchy.
    let mut vec_cache: std::collections::HashMap<i64, Option<Vec<f32>>> =
        std::collections::HashMap::new();
    let mut tag_keys: BTreeMap<i64, String> = BTreeMap::new();
    let mut tag_members: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut centroid_keys: Vec<i64> = Vec::new();
    let mut centroids: Vec<Vec<f32>> = Vec::new();

    for (tag, members) in &members_by_tag {
        if members.len() < config.min_members || config.blocked(tag) {
            continue;
        }
        if total_notes > 0 && (members.len() as f64 / total_notes as f64) > config.max_coverage {
            continue;
        }
        // Mean over the members' text vectors that are actually indexed —
        // an unembedded member (mid-build) just doesn't contribute.
        let mut sum: Vec<f32> = Vec::new();
        let mut count = 0usize;
        for note_id in members {
            let v = vec_cache.entry(*note_id).or_insert_with(|| {
                engine
                    .modality_get("text", *note_id)
                    .and_then(|vectors| vectors.into_iter().next())
            });
            let Some(v) = v else { continue };
            if sum.is_empty() {
                sum = vec![0.0; v.len()];
            }
            if v.len() != sum.len() {
                continue;
            }
            for (acc, x) in sum.iter_mut().zip(v.iter()) {
                *acc += *x;
            }
            count += 1;
        }
        if count < config.min_members {
            continue;
        }
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            continue;
        }
        let centroid: Vec<f32> = sum.iter().map(|x| x / norm).collect();
        let key = tag_key(tag);
        if tag_keys.contains_key(&key) {
            // A blake2b-8 collision between live tags: vanishingly unlikely;
            // skip the later one rather than mislabel the earlier.
            tracing::warn!(tag, "tag key collision; centroid skipped");
            continue;
        }
        tag_keys.insert(key, tag.clone());
        tag_members.insert(key, members.clone());
        centroid_keys.push(key);
        centroids.push(centroid);
    }

    // Replace the space wholesale: drop + re-add (the set is small).
    engine.drop_modality(TAG_TEXT_SPACE);
    if let Some(first) = centroids.first() {
        engine.ensure(TAG_TEXT_SPACE, first.len())?;
        engine.add(TAG_TEXT_SPACE, &centroid_keys, &centroids)?;
    }
    let built = centroid_keys.len();
    keys.replace(tag_keys, tag_members);
    Ok(built)
}

/// Retrieval knobs (#179): how many top tags may activate per query, and the
/// note-ranking cap so one giant tag can't flood the fusion.
pub const TAG_TOP_TAGS: usize = 3;
pub const TAG_RANK_CAP: usize = 50;

/// The tag activation floor — deliberately BELOW the semantic note threshold:
/// a centroid is a MEAN over members, so dilution systematically lowers its
/// attainable cosine against any single query (a perfectly on-topic tag with
/// one off-topic member already sits well under the best member's score).
/// Offline calibration against the tag space (the #201b approach) is the
/// future refinement; this fixed floor is the v1 knob.
pub const TAG_ACTIVATION: f64 = 0.35;

/// The tag retrieval signal (#179): rank the `tag.text` space with the query
/// vector, activate tags whose centroid cosine clears `threshold` (the same
/// floor the semantic note ranking uses — centroids live in the text space,
/// so the scales are commensurable), cap activation at `top_tags`, and expand
/// to member notes: best tag first, members within a tag ordered by their own
/// text-vector cosine to the query, deduped across tags (first appearance
/// wins), capped at `cap`. Conditionally present: no tags / no activation /
/// empty state → an empty ranking, which contributes nothing to RRF.
pub fn tag_ranking(
    engine: &dyn VectorIndex,
    keys: &TagKeyMap,
    query: &[f32],
    threshold: f64,
    top_tags: usize,
    cap: usize,
) -> Vec<i64> {
    if keys.is_empty() {
        return Vec::new();
    }
    let Ok(rankings) = engine.search_by_modality(
        &[query.to_vec()],
        top_tags,
        Some(&[TAG_TEXT_SPACE.to_string()]),
    ) else {
        return Vec::new();
    };
    let mut out: Vec<i64> = Vec::new();
    let mut seen: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    let Some(per_query) = rankings.first() else {
        return Vec::new();
    };
    let Some((ranked_keys, distances)) = per_query.get(TAG_TEXT_SPACE) else {
        return Vec::new();
    };
    for (key, dist) in ranked_keys.iter().zip(distances) {
        // cos = 1 - distance; the ranking is distance-ascending, so the first
        // miss ends activation.
        if f64::from(1.0 - dist) < threshold {
            break;
        }
        // Fresh-first (#445): members an earlier (better) tag already ranked
        // would only be skipped *after* scoring — filtering them up front
        // means hierarchy overlap is never re-scored, and pushing the fresh
        // subset in score order is exactly what the full sort + skip did.
        let fresh = keys.fresh_members(*key, &seen, MEMBER_SCORE_CEILING);
        if fresh.is_empty() {
            continue;
        }
        let mut scored = engine.dot_scores("text", &fresh, query);
        // Only the top `needed` ever leave this loop iteration — partition
        // them out, then order just that slice (#445: a huge tag previously
        // paid a full O(m log m) sort to fill a 50-slot cap).
        let needed = cap - out.len();
        if scored.len() > needed {
            scored.select_nth_unstable_by(needed - 1, |a, b| b.1.total_cmp(&a.1));
            scored.truncate(needed);
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (nid, _) in scored {
            seen.insert(nid);
            out.push(nid);
        }
        if out.len() >= cap {
            return out;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrike_index::MultiModalIndex;

    #[test]
    fn hierarchy_expands_prefixes() {
        assert_eq!(
            expand_hierarchy("a::b::c"),
            vec!["a".to_string(), "a::b".to_string(), "a::b::c".to_string()]
        );
        assert_eq!(expand_hierarchy("plain"), vec!["plain".to_string()]);
    }

    #[test]
    fn membership_rolls_up_and_dedupes() {
        let rows = vec![
            (1, vec!["sci::phys".to_string(), "sci::chem".to_string()]),
            (2, vec!["sci::phys".to_string()]),
        ];
        let m = membership(&rows);
        assert_eq!(m["sci"], vec![1, 2]); // deduped roll-up
        assert_eq!(m["sci::phys"], vec![1, 2]);
        assert_eq!(m["sci::chem"], vec![1]);
    }

    #[test]
    fn any_member_of_checks_current_membership() {
        let keys = TagKeyMap::default();
        assert!(!keys.any_member_of(&[1]), "empty state has no members");
        keys.replace(
            BTreeMap::from([(7, "t".to_string())]),
            BTreeMap::from([(7, vec![1, 2, 3])]),
        );
        assert!(keys.any_member_of(&[3, 99]));
        assert!(!keys.any_member_of(&[98, 99]));
        assert!(!keys.any_member_of(&[]));
    }

    #[test]
    fn blocklist_matches_segments_case_insensitively() {
        let config = TagCentroidConfig::default();
        assert!(config.blocked("leech"));
        assert!(config.blocked("deck::Leech"));
        assert!(!config.blocked("leeches")); // segment match, not substring
    }

    #[test]
    fn tag_keys_are_stable_and_positive() {
        assert_eq!(tag_key("cardio"), tag_key("cardio"));
        assert_ne!(tag_key("cardio"), tag_key("cardiology"));
        assert!(tag_key("anything") >= 0);
    }

    #[test]
    fn tag_ranking_activates_orders_and_degrades() {
        let engine = MultiModalIndex::new(vec![
            "text".to_string(),
            "image".to_string(),
            TAG_TEXT_SPACE.to_string(),
        ])
        .unwrap();
        engine.ensure("text", 2).unwrap();
        // Two members near the x-axis (one closer), plus off-topic notes so
        // coverage hygiene passes.
        engine
            .add(
                "text",
                &[1, 2, 3, 4, 5],
                &[
                    vec![1.0, 0.0],
                    vec![0.9, (1.0f32 - 0.81).sqrt()],
                    vec![0.0, 1.0],
                    vec![0.0, 1.0],
                    vec![0.0, 1.0],
                ],
            )
            .unwrap();
        let rows = vec![
            (1, vec!["topic".to_string()]),
            (2, vec!["topic".to_string()]),
        ];
        let keys = TagKeyMap::default();
        recompute(&engine, &rows, 5, &TagCentroidConfig::default(), &keys).unwrap();
        assert_eq!(keys.members(tag_key("topic")), vec![1, 2]);

        // An on-axis query activates `topic`; members come back in their own
        // cosine order (1 before 2).
        let ranking = tag_ranking(&engine, &keys, &[1.0, 0.0], 0.5, 3, 50);
        assert_eq!(ranking, vec![1, 2]);

        // An orthogonal query clears no activation → empty (degrades).
        let off = tag_ranking(&engine, &keys, &[0.0, 1.0], 0.5, 3, 50);
        assert!(off.is_empty());

        // The cap bounds the expansion.
        assert_eq!(tag_ranking(&engine, &keys, &[1.0, 0.0], 0.5, 3, 1).len(), 1);
    }

    #[test]
    fn tag_ranking_dedupes_across_overlapping_tags() {
        // Hierarchy overlap: `a::b`'s member is also `a`'s (roll-up). The
        // fresh-first expansion (#445) must still surface each note once,
        // best tag first.
        let engine = MultiModalIndex::new(vec![
            "text".to_string(),
            "image".to_string(),
            TAG_TEXT_SPACE.to_string(),
        ])
        .unwrap();
        engine.ensure("text", 2).unwrap();
        engine
            .add(
                "text",
                &[1, 2],
                &[vec![1.0, 0.0], vec![0.9, (1.0f32 - 0.81).sqrt()]],
            )
            .unwrap();
        let rows = vec![(1, vec!["a::b".to_string()]), (2, vec!["a".to_string()])];
        let keys = TagKeyMap::default();
        let config = TagCentroidConfig {
            min_members: 1,
            max_coverage: 1.0,
            ..TagCentroidConfig::default()
        };
        recompute(&engine, &rows, 2, &config, &keys).unwrap();
        assert_eq!(keys.members(tag_key("a")), vec![1, 2]);
        assert_eq!(keys.members(tag_key("a::b")), vec![1]);

        // `a::b`'s centroid is note 1 exactly → activates first; `a` then
        // contributes only its fresh member.
        let ranking = tag_ranking(&engine, &keys, &[1.0, 0.0], 0.5, 3, 50);
        assert_eq!(ranking, vec![1, 2]);
    }

    #[test]
    fn fresh_members_filters_seen_and_bounds_to_ceiling() {
        let keys = TagKeyMap::default();
        let members: Vec<i64> = (1..=10).collect();
        keys.replace(
            BTreeMap::from([(7, "t".to_string())]),
            BTreeMap::from([(7, members)]),
        );
        let seen: std::collections::BTreeSet<i64> = [2, 4].into_iter().collect();

        // Under the ceiling: every fresh member, in member order.
        assert_eq!(
            keys.fresh_members(7, &seen, 100),
            vec![1, 3, 5, 6, 7, 8, 9, 10]
        );
        // Over the ceiling: bounded, all fresh, deterministic.
        let sampled = keys.fresh_members(7, &seen, 3);
        assert_eq!(sampled.len(), 3);
        assert!(sampled.iter().all(|n| !seen.contains(n)));
        assert_eq!(sampled, keys.fresh_members(7, &seen, 3));
        // Unknown key: empty.
        assert!(keys.fresh_members(99, &seen, 3).is_empty());
    }

    #[test]
    fn recompute_builds_renormalized_means_and_filters() {
        let engine = MultiModalIndex::new(vec![
            "text".to_string(),
            "image".to_string(),
            TAG_TEXT_SPACE.to_string(),
        ])
        .unwrap();
        engine.ensure("text", 2).unwrap();
        // Three notes on two axes.
        engine
            .add(
                "text",
                &[1, 2, 3],
                &[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]],
            )
            .unwrap();
        let rows = vec![
            (1, vec!["topic".to_string(), "leech".to_string()]),
            (2, vec!["topic".to_string()]),
            (3, vec!["solo".to_string()]), // 1 member → filtered
        ];
        let keys = TagKeyMap::default();
        // total_notes=8: `topic` covers 2/8 (under the structural-coverage cap).
        let built = recompute(&engine, &rows, 8, &TagCentroidConfig::default(), &keys).unwrap();
        assert_eq!(built, 1, "only `topic` survives hygiene");
        assert_eq!(keys.lookup(tag_key("topic")).as_deref(), Some("topic"));

        // The centroid is the renormalized mean of [1,0] and [0,1].
        let centroid = engine
            .modality_get(TAG_TEXT_SPACE, tag_key("topic"))
            .unwrap()
            .remove(0);
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!((centroid[0] - expected).abs() < 1e-5);
        assert!((centroid[1] - expected).abs() < 1e-5);

        // A note search scoped to NOTE_MODALITIES never sees the tag key.
        let hits = engine
            .search_by_modality(&[vec![1.0, 0.0]], 10, Some(&["text".to_string()]))
            .unwrap();
        let ids = &hits[0]["text"].0;
        assert!(!ids.contains(&tag_key("topic")));
    }
}
