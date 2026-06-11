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
use std::sync::RwLock;

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

use shrike_ffi::NativeResult;
use shrike_index::MultiModalIndex;

use crate::TAG_TEXT_SPACE;

/// Hygiene knobs (#179: "a curation surface — make them configurable"). The
/// defaults live here; the harness threads overrides through
/// [`TagCentroidConfig`].
pub const DEFAULT_MIN_MEMBERS: usize = 2;
pub const DEFAULT_MAX_COVERAGE: f64 = 0.5;
pub const DEFAULT_BLOCKLIST: &[&str] = &["leech", "marked"];

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
/// never collide with the sign conventions of note ids).
pub fn tag_key(tag: &str) -> i64 {
    let mut hasher = Blake2bVar::new(8).expect("8-byte blake2b");
    hasher.update(tag.as_bytes());
    let mut out = [0u8; 8];
    hasher.finalize_variable(&mut out).expect("8-byte output");
    (i64::from_be_bytes(out)) & i64::MAX
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

    pub fn len(&self) -> usize {
        self.inner.read().expect("tag keys poisoned").names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn replace(&self, names: BTreeMap<i64, String>, members: BTreeMap<i64, Vec<i64>>) {
        *self.inner.write().expect("tag keys poisoned") = TagState { names, members };
    }
}

/// Recompute every tag centroid from the engine's note TEXT vectors and
/// replace the `tag.text` space wholesale. Returns the centroid count.
/// Centroid = renormalized mean (the mean of unit vectors is not unit-norm).
pub fn recompute(
    engine: &MultiModalIndex,
    rows: &[(i64, Vec<String>)],
    total_notes: usize,
    config: &TagCentroidConfig,
    keys: &TagKeyMap,
) -> NativeResult<usize> {
    let members_by_tag = membership(rows);
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
            let Some(vectors) = engine.modality_get("text", *note_id) else {
                continue;
            };
            let Some(v) = vectors.first() else { continue };
            if sum.is_empty() {
                sum = vec![0.0; v.len()];
            }
            if v.len() != sum.len() {
                continue;
            }
            for (acc, x) in sum.iter_mut().zip(v) {
                *acc += x;
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The tag retrieval signal (#179): rank the `tag.text` space with the query
/// vector, activate tags whose centroid cosine clears `threshold` (the same
/// floor the semantic note ranking uses — centroids live in the text space,
/// so the scales are commensurable), cap activation at `top_tags`, and expand
/// to member notes: best tag first, members within a tag ordered by their own
/// text-vector cosine to the query, deduped across tags (first appearance
/// wins), capped at `cap`. Conditionally present: no tags / no activation /
/// empty state → an empty ranking, which contributes nothing to RRF.
pub fn tag_ranking(
    engine: &MultiModalIndex,
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
        let mut members: Vec<(i64, f32)> = keys
            .members(*key)
            .iter()
            .filter_map(|nid| {
                let vectors = engine.modality_get("text", *nid)?;
                let v = vectors.first()?;
                (v.len() == query.len()).then(|| (*nid, dot(query, v)))
            })
            .collect();
        members.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (nid, _) in members {
            if seen.insert(nid) {
                out.push(nid);
                if out.len() >= cap {
                    return out;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
