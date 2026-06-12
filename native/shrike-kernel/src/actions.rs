//! The action core (#331, kernel inversion S2) — slice 1: the read surface.
//!
//! Each action is the *whole* tool body: parameter normalization, the
//! collection-core call, and validation into the Rust-canonical response type
//! (#330) — so the typed contract guards the internal wire, not just the MCP
//! edge. Python's `actions.py` shrinks to a binding per re-homed action:
//! typed signature (FastMCP's inputSchema source) + context assembly + the
//! completion-log fragment.
//!
//! Actions are synchronous over `&CollectionCore`: the transitional harness
//! invokes them on its collection worker thread through the shrike-py
//! per-action bindings (the same serialization every collection op rides);
//! the kernel's async layer (S3, #332) will drive the same bodies through
//! [`crate::SerializedCollection`]. No threading, no runtime assumption here
//! (#308/#310).

use serde::de::DeserializeOwned;

use shrike_collection::CollectionCore;
use shrike_ffi::{NativeError, NativeResult};
use shrike_schemas::{CollectionInfo, ListNotesResponse};

/// The actions this module has re-homed (the registry seam: the Python
/// binding asserts its forwarding list against this, so the two sides can't
/// drift silently).
pub const REHOMED_ACTIONS: &[&str] = &[
    "collection_info",
    "list_notes",
    "collection_query",
    "search_notes",
];

/// Parse a core-emitted JSON payload into its canonical response type.
///
/// A parse failure here is a *bug* (the core and the schema disagree), not
/// caller input — surfaced as an internal error with the type named.
fn validate<T: DeserializeOwned>(name: &str, json: &str) -> NativeResult<T> {
    serde_json::from_str(json).map_err(|e| {
        NativeError::internal(format!(
            "{name}: core payload does not match the schema: {e}"
        ))
    })
}

/// `collection_info` — sectioned collection structure/stats.
///
/// `include` mirrors the tool param (empty = summary, `"all"` expands);
/// `note_type_details` selects which note types carry their full definition.
pub fn collection_info(
    core: &CollectionCore,
    include: &[String],
    note_type_details: &[String],
) -> NativeResult<CollectionInfo> {
    let raw = core.collection_info(include, note_type_details)?;
    validate("CollectionInfo", &raw)
}

/// Structured filters for [`list_notes`]. `modified_since_epoch` is an
/// epoch-seconds cutoff — ISO-8601 parsing stays host-side (a deliberate
/// divergence recorded on the core's `list_notes`).
#[derive(Debug, Clone, Default)]
pub struct ListNotesParams {
    pub ids: Option<Vec<i64>>,
    pub deck: Option<String>,
    pub tags: Option<Vec<String>>,
    pub note_type: Option<String>,
    pub modified_since_epoch: Option<i64>,
    pub with_fields: bool,
    pub limit: usize,
}

/// `list_notes` — filter/retrieve notes (filters ANDed; at least one given,
/// enforced by the core as invalid input).
pub fn list_notes(
    core: &CollectionCore,
    params: &ListNotesParams,
) -> NativeResult<ListNotesResponse> {
    let raw = core.list_notes(
        params.ids.as_deref(),
        params.deck.as_deref(),
        params.tags.as_deref(),
        params.note_type.as_deref(),
        params.modified_since_epoch,
        params.with_fields,
        params.limit,
    )?;
    validate("ListNotesResponse", &raw)
}

/// `collection_query` — a raw Anki search expression (the read-only escape
/// hatch, #97). A malformed expression is invalid input (isolation marks
/// already stripped by the core's error decoding).
pub fn collection_query(
    core: &CollectionCore,
    query: &str,
    with_fields: bool,
    limit: usize,
) -> NativeResult<ListNotesResponse> {
    let raw = core.query(query, with_fields, limit)?;
    validate("ListNotesResponse", &raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn temp_collection() -> (std::path::PathBuf, CollectionCore) {
        // Process id + a process-wide counter: parallel test threads can land
        // on the same nanosecond stamp (observed under the Bazel sandbox), and
        // two cores on one path fail with "Anki already open".
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-kernel-actions-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.anki2");
        let core = CollectionCore::open(path.to_str().unwrap()).unwrap();
        (dir, core)
    }

    pub(super) fn add_note(core: &CollectionCore, front: &str, back: &str) -> i64 {
        let req = serde_json::json!([
            {"note_type": "Basic", "deck": "D", "fields": {"Front": front, "Back": back}}
        ]);
        let out = core.upsert_notes(&req.to_string(), "allow", false).unwrap();
        let results: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(results[0]["status"], "created", "{results}");
        results[0]["id"].as_i64().unwrap()
    }

    #[test]
    fn collection_info_returns_typed_sections() {
        let (_dir, core) = temp_collection();
        add_note(&core, "Q", "A");
        let info = collection_info(&core, &["summary".into(), "decks".into()], &[]).unwrap();
        let summary = info.summary.expect("summary requested");
        assert_eq!(summary.notes, 1);
        assert!(info.decks.is_some());
        assert!(info.stats.is_none()); // not requested
        core.close().unwrap();
    }

    #[test]
    fn list_notes_filters_and_validates() {
        let (_dir, core) = temp_collection();
        let id = add_note(&core, "mitochondria", "powerhouse");
        add_note(&core, "momentum", "mass times velocity");
        let resp = list_notes(
            &core,
            &ListNotesParams {
                deck: Some("D".into()),
                with_fields: true,
                limit: 50,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(resp.total, 2);
        assert!(resp.notes.iter().any(|n| n.id == id));
        // with_fields=true → content present on every note.
        assert!(resp.notes.iter().all(|n| n.content.is_some()));
        core.close().unwrap();
    }

    #[test]
    fn list_notes_without_filters_is_invalid_input() {
        let (_dir, core) = temp_collection();
        let err = list_notes(
            &core,
            &ListNotesParams {
                limit: 50,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("input"),
            "{err:?}"
        );
        core.close().unwrap();
    }

    #[test]
    fn collection_query_runs_raw_expressions() {
        let (_dir, core) = temp_collection();
        add_note(&core, "the cell", "biology");
        let resp = collection_query(&core, "deck:D", false, 10).unwrap();
        assert_eq!(resp.total, 1);
        assert!(resp.notes[0].content.is_none()); // meta mode
        assert!(collection_query(&core, "prop:bogus(((", false, 10).is_err());
        core.close().unwrap();
    }

    #[test]
    fn rehomed_registry_names_the_slice() {
        assert_eq!(
            REHOMED_ACTIONS,
            &[
                "collection_info",
                "list_notes",
                "collection_query",
                "search_notes"
            ]
        );
    }
}

// ── search_notes (#331, S2: the assembly re-home) ────────────────────────────
// The whole fused-search body: per-modality semantic ranking over query
// vectors (embedded host-side — a handful of query vectors crossing the FFI is
// the recorded design point on #331), substring + fuzzy lexical candidates
// from the derived store (with the find_notes fallback), RRF fusion with the
// exact-match priority tier, and annotation/provenance assembly — validated
// into the canonical SearchResultGroup. Orchestrator state (semantic
// availability, the #201b image activation floor, the index size for the
// over-fetch clamp) is injected per call until S3 internalizes it.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use shrike_derived::{DerivedEngine, MIN_TRIGRAM};
use shrike_index::MultiModalIndex;
use shrike_schemas::SearchResultGroup;

/// One source's per-modality semantic rankings (`search_by_modality`'s row).
type ModalityHits = std::collections::BTreeMap<String, (Vec<i64>, Vec<f32>)>;

/// Anki search-language specials, escaped for the `"*text*"` wildcard
/// pre-filter (mirrors `collection._escape_anki_text`).
fn escape_anki_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if matches!(ch, '\\' | '"' | '*' | '_' | ':') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

/// Locate a case-insensitive literal substring in a note's fields — the
/// authority for exact-match evidence (mirrors `collection.substring_info`,
/// code-point index math included: find runs over the lowered text, the
/// snippet slices the original by those indices).
pub fn substring_info(content: Option<&Value>, text: &str) -> Value {
    let needle: Vec<char> = text.to_lowercase().chars().collect();
    let mut matched: Vec<&str> = Vec::new();
    let mut snippet: Option<String> = None;
    if let Some(Value::Object(fields)) = content {
        for (name, value) in fields {
            let v = value.as_str().unwrap_or("");
            let chars: Vec<char> = v.chars().collect();
            let lowered: Vec<char> = v.to_lowercase().chars().collect();
            let idx = match find_subsequence(&lowered, &needle) {
                Some(i) => i,
                None => continue,
            };
            matched.push(name);
            if snippet.is_none() {
                let start = idx.saturating_sub(30);
                let end = (idx + needle.len() + 30).min(chars.len());
                // Python slices the ORIGINAL with lowered-string indices; on
                // length-changing lowercasings the window drifts identically.
                let end = end.min(chars.len());
                let mut frag: String = chars[start.min(chars.len())..end].iter().collect();
                if start > 0 {
                    frag = format!("…{frag}");
                }
                if idx + needle.len() + 30 < chars.len() {
                    frag.push('…');
                }
                snippet = Some(frag);
            }
        }
    }
    if matched.is_empty() {
        Value::Null
    } else {
        json!({"matched_fields": matched, "snippet": snippet})
    }
}

fn find_subsequence(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Python's `round(x, 3)` (round-half-even on the binary double).
fn round3(x: f64) -> f64 {
    (x * 1000.0).round_ties_even() / 1000.0
}

/// One search source: a query string (semantic + lexical) or an id anchor
/// (semantic only).
#[derive(Debug, Clone)]
pub struct SearchSource {
    pub label: String,
    pub text: String,
    pub is_query: bool,
}

/// The per-call arguments (orchestrator state injected by the harness).
#[derive(Debug, Clone, Default)]
pub struct SearchArgs {
    pub top_k: usize,
    pub threshold: f64,
    /// The RESOLVED deck name (semantic candidates filter on exact equality;
    /// the find_notes fallback uses `deck:` which includes children — the
    /// Python original's behaviour, ported faithfully).
    pub deck: Option<String>,
    pub tags: Vec<String>,
    pub exclude: Vec<i64>,
    /// The #201b activation floor for the image modality (None = no gating).
    pub image_floor: Option<f64>,
    /// Per-signal RRF weights (the harness passes SEARCH_WEIGHTS).
    pub weights: std::collections::BTreeMap<String, f64>,
    /// Whether semantic ranking runs (index ready + backend up); `vectors`
    /// carries one query vector per source when true.
    pub semantic: bool,
    /// The index's current vector count, for the over-fetch clamp.
    pub index_size: usize,
}

/// Insertion-ordered candidate cache: Python dict iteration order is part of
/// the exact ranking's contract, so the order log rides along.
struct NoteData {
    map: HashMap<i64, Value>,
    order: Vec<i64>,
}

impl NoteData {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            order: Vec::new(),
        }
    }
    fn contains(&self, nid: i64) -> bool {
        self.map.contains_key(&nid)
    }
    fn insert(&mut self, nid: i64, data: Value) {
        if self.map.insert(nid, data).is_none() {
            self.order.push(nid);
        }
    }
}

fn in_scope(data: &Value, deck: Option<&str>, tags: &[String]) -> bool {
    if let Some(d) = deck {
        if data.get("deck").and_then(Value::as_str) != Some(d) {
            return false;
        }
    }
    if !tags.is_empty() {
        let note_tags: HashSet<&str> = data
            .get("tags")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if !tags.iter().all(|t| note_tags.contains(t.as_str())) {
            return false;
        }
    }
    true
}

/// Read one note's full dict, skipping unreadable notes (the Python original
/// catches and debug-logs per note).
fn read_note(core: &CollectionCore, nid: i64) -> Option<Value> {
    match core.note_dicts(&[nid], true) {
        Ok(mut v) if !v.is_empty() => Some(v.remove(0)),
        Ok(_) => None,
        Err(e) => {
            tracing::debug!(nid, error = ?e, "search: skipping unreadable note");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rank_modality(
    core: &CollectionCore,
    hits_keys: &[i64],
    hits_distances: &[f32],
    note_data: &mut NoteData,
    sem_score: &mut HashMap<i64, f64>,
    exclude: &HashSet<i64>,
    args: &SearchArgs,
    thresholded: bool,
) -> Vec<i64> {
    let mut ranking: Vec<i64> = Vec::new();
    for (nid, dist) in hits_keys.iter().zip(hits_distances.iter()) {
        let nid = *nid;
        if exclude.contains(&nid) {
            continue;
        }
        let score = round3(1.0 - f64::from(*dist));
        if thresholded && score < args.threshold {
            break; // distance-ascending → the rest are below threshold
        }
        if !note_data.contains(nid) {
            let data = match read_note(core, nid) {
                Some(d) => d,
                None => continue,
            };
            if !in_scope(&data, args.deck.as_deref(), &args.tags) {
                continue; // out of scope — keep it out of note_data entirely
            }
            note_data.insert(nid, data);
        }
        ranking.push(nid);
        let entry = sem_score.entry(nid).or_insert(score);
        if score > *entry {
            *entry = score;
        }
        if ranking.len() >= args.top_k {
            break;
        }
    }
    ranking
}

fn collect_substring_candidates(
    core: &CollectionCore,
    derived: Option<&DerivedEngine>,
    text: &str,
    note_data: &mut NoteData,
    exclude: &HashSet<i64>,
    args: &SearchArgs,
    scope: Option<&[i64]>,
) -> NativeResult<()> {
    // The store serves scoped queries too (#177 retirement of the wildcard
    // scan): the scope id set — from anki's INDEXED deck:/tag: search — is
    // pushed into the FTS5 query, so a scoped literal search reads no note
    // text outside the store. The wildcard `*text*` fallback (a full field
    // scan) survives only for the cases FTS5 can't serve: a sub-trigram
    // query (<3 chars) or a missing/unbuilt store.
    let lex = if let Some(d) = derived {
        match d.search_substring(text, (args.top_k + exclude.len()) as i64, scope) {
            Ok(rows) => rows,
            Err(e) => {
                tracing::debug!(error = ?e, "FTS5 substring query failed; falling back");
                None
            }
        }
    } else {
        None
    };

    let Some(rows) = lex else {
        // Fallback: Anki's "*text*" wildcard as a fast pre-filter, scope in
        // the query; substring_info confirms + annotates each candidate.
        if text.trim().is_empty() {
            return Ok(());
        }
        let mut parts = vec![format!("\"*{}*\"", escape_anki_text(text))];
        if let Some(d) = &args.deck {
            parts.push(format!("\"deck:{d}\""));
        }
        for tag in &args.tags {
            parts.push(format!("\"tag:{tag}\""));
        }
        let candidates: Vec<i64> = core
            .find_notes(&parts.join(" "))?
            .into_iter()
            .filter(|nid| !exclude.contains(nid))
            .collect();
        let mut added = 0usize;
        for nid in candidates {
            if note_data.contains(nid) {
                continue;
            }
            let mut data = match read_note(core, nid) {
                Some(d) => d,
                None => continue,
            };
            let info = substring_info(data.get("content"), text);
            if info.is_null() {
                continue; // Anki matched across markup/normalization; not literal
            }
            data["substring"] = info;
            note_data.insert(nid, data);
            added += 1;
            if added >= args.top_k {
                break;
            }
        }
        return Ok(());
    };

    let mut added = 0usize;
    for (nid, _source, _ref, _snippet) in rows {
        if exclude.contains(&nid) || note_data.contains(nid) {
            continue; // store may return a row per field
        }
        let data = match read_note(core, nid) {
            Some(d) => d,
            None => continue,
        };
        if !in_scope(&data, args.deck.as_deref(), &args.tags) {
            continue;
        }
        note_data.insert(nid, data);
        added += 1;
        if added >= args.top_k {
            break;
        }
    }
    Ok(())
}

type FuzzyEvidence = HashMap<i64, (String, String, Option<String>)>;

fn collect_fuzzy(
    core: &CollectionCore,
    derived: Option<&DerivedEngine>,
    text: &str,
    note_data: &mut NoteData,
    exclude: &HashSet<i64>,
    args: &SearchArgs,
    scope: Option<&[i64]>,
) -> (Vec<i64>, FuzzyEvidence) {
    let Some(d) = derived else {
        return (Vec::new(), HashMap::new());
    };
    let hits = match d.search_fuzzy(text, args.top_k as i64, scope) {
        Ok(h) => h,
        Err(e) => {
            tracing::debug!(error = ?e, "FTS5 fuzzy query failed");
            return (Vec::new(), HashMap::new());
        }
    };
    let mut ranking: Vec<i64> = Vec::new();
    let mut evidence: FuzzyEvidence = HashMap::new();
    for (nid, source, r, snippet) in hits {
        if exclude.contains(&nid) || evidence.contains_key(&nid) {
            continue;
        }
        if !note_data.contains(nid) {
            let data = match read_note(core, nid) {
                Some(d2) => d2,
                None => continue,
            };
            if !in_scope(&data, args.deck.as_deref(), &args.tags) {
                continue;
            }
            note_data.insert(nid, data);
        }
        ranking.push(nid);
        evidence.insert(nid, (source, r, snippet));
        if ranking.len() >= args.top_k {
            break;
        }
    }
    (ranking, evidence)
}

/// The fused search assembly (see module-section comment). `vectors` carries
/// one query vector per source when `args.semantic`.
pub fn search_notes(
    core: &CollectionCore,
    index: Option<&MultiModalIndex>,
    derived: Option<&DerivedEngine>,
    tag_keys: Option<&crate::tag_centroids::TagKeyMap>,
    sources: &[SearchSource],
    vectors: &[Vec<f32>],
    args: &SearchArgs,
) -> NativeResult<Vec<SearchResultGroup>> {
    let exclude: HashSet<i64> = args.exclude.iter().copied().collect();

    // Semantic pass (batched), per modality — over-fetch to cover exclusions
    // and post-hoc scope/substring filtering.
    let mut sem_by_source: Vec<ModalityHits> = Vec::new();
    if args.semantic {
        let index = index.ok_or_else(|| {
            NativeError::invalid_input("semantic search requested without an index engine")
        })?;
        let mut fetch_k = args.top_k + exclude.len();
        if args.deck.is_some() || !args.tags.is_empty() {
            fetch_k = fetch_k.max(args.top_k * 10);
            if args.index_size > 0 {
                fetch_k = fetch_k.min(args.index_size);
            }
        }
        // Scoped to the NOTE-item spaces: tag-centroid spaces (#178) share
        // the engine but must never surface a tag key from a note search.
        let note_spaces: Vec<String> = crate::NOTE_MODALITIES
            .iter()
            .map(|m| m.to_string())
            .collect();
        sem_by_source = index.search_by_modality(vectors, fetch_k, Some(&note_spaces))?;
    }

    // The lexical scope set (#177): one INDEXED anki query (deck:/tag: —
    // never a field-text scan) shared by both lexical collectors, pushed
    // into the FTS5 queries so scoped literal/fuzzy search keeps exact
    // recall without over-fetch. None = unscoped.
    let lex_scope: Option<Vec<i64>> = if args.deck.is_some() || !args.tags.is_empty() {
        let mut parts: Vec<String> = Vec::new();
        if let Some(d) = &args.deck {
            parts.push(format!("\"deck:{d}\""));
        }
        for tag in &args.tags {
            parts.push(format!("\"tag:{tag}\""));
        }
        Some(core.find_notes(&parts.join(" "))?)
    } else {
        None
    };

    let mut results: Vec<Value> = Vec::new();
    for (i, source) in sources.iter().enumerate() {
        let mut note_data = NoteData::new();

        // Literal-substring candidates (query sources only): a fast pre-filter;
        // substring_info below is the authority that confirms + annotates.
        if source.is_query {
            collect_substring_candidates(
                core,
                derived,
                &source.text,
                &mut note_data,
                &exclude,
                args,
                lex_scope.as_deref(),
            )?;
        }

        // Per-modality semantic rankings. Text is thresholded; image is not
        // (the gap makes the text-calibrated cosine threshold meaningless —
        // flooring image hits is the #201b activation gate's job below).
        let empty = ModalityHits::new();
        let modality_hits = sem_by_source.get(i).unwrap_or(&empty);
        let mut sem_score: HashMap<i64, f64> = HashMap::new();
        let (tk, td) = modality_hits
            .get("text")
            .map(|(k, d)| (k.as_slice(), d.as_slice()))
            .unwrap_or((&[], &[]));
        let ranking_text = rank_modality(
            core,
            tk,
            td,
            &mut note_data,
            &mut sem_score,
            &exclude,
            args,
            true,
        );
        // Image modality into a scratch score first: the gate is judged on the
        // best hit that SURVIVES exclusion + scope, not the raw rank-1.
        let mut image_score: HashMap<i64, f64> = HashMap::new();
        let (ik, idists) = modality_hits
            .get("image")
            .map(|(k, d)| (k.as_slice(), d.as_slice()))
            .unwrap_or((&[], &[]));
        let mut ranking_image = rank_modality(
            core,
            ik,
            idists,
            &mut note_data,
            &mut image_score,
            &exclude,
            args,
            false,
        );
        if let (Some(first), Some(floor)) = (ranking_image.first(), args.image_floor) {
            if image_score[first] <= floor {
                ranking_image.clear(); // no good-enough surviving image match
                image_score.clear();
            }
        }
        // Tag-centroid signal (#179): conditionally present — activated tags
        // expand to member notes through the SAME scope/exclusion machinery
        // (synthetic order-preserving distances into a scratch score map, so
        // tag evidence never masquerades as a semantic `score`).
        let mut ranking_tag: Vec<i64> = Vec::new();
        if args.semantic {
            if let (Some(keys), Some(engine), Some(qvec)) = (tag_keys, index, vectors.get(i)) {
                let member_ids = crate::tag_centroids::tag_ranking(
                    engine,
                    keys,
                    qvec,
                    crate::tag_centroids::TAG_ACTIVATION,
                    crate::tag_centroids::TAG_TOP_TAGS,
                    crate::tag_centroids::TAG_RANK_CAP,
                );
                let synth: Vec<f32> = (0..member_ids.len()).map(|r| r as f32 * 1e-4).collect();
                let mut scratch: HashMap<i64, f64> = HashMap::new();
                ranking_tag = rank_modality(
                    core,
                    &member_ids,
                    &synth,
                    &mut note_data,
                    &mut scratch,
                    &exclude,
                    args,
                    false,
                );
            }
        }

        for (nid, isim) in &image_score {
            let entry = sem_score.entry(*nid).or_insert(*isim);
            if *isim > *entry {
                *entry = *isim;
            }
        }

        // Fuzzy ranking + evidence (query sources only), before the exact loop
        // so a fuzzy candidate that also literally matches joins the exact tier.
        let (mut ranking_fuzzy, mut fuzzy_evidence) = if source.is_query {
            collect_fuzzy(
                core,
                derived,
                &source.text,
                &mut note_data,
                &exclude,
                args,
                lex_scope.as_deref(),
            )
        } else {
            (Vec::new(), HashMap::new())
        };

        // Exact ranking = every candidate whose content literally contains the
        // query (annotation ⟺ floated), in note_data insertion order.
        let mut exact_ids: Vec<i64> = Vec::new();
        if source.is_query {
            for nid in &note_data.order {
                let data = note_data.map.get_mut(nid).expect("ordered key present");
                if data.get("substring").is_none() {
                    data["substring"] = substring_info(data.get("content"), &source.text);
                }
                if !data["substring"].is_null() {
                    exact_ids.push(*nid);
                }
            }
        }

        // An exact note is trivially also a fuzzy match — drop it from the
        // fuzzy signal so `fuzzy` means the DISTINGUISHING lexical signal.
        if !exact_ids.is_empty() && !ranking_fuzzy.is_empty() {
            let exact_set: HashSet<i64> = exact_ids.iter().copied().collect();
            ranking_fuzzy.retain(|nid| !exact_set.contains(nid));
            fuzzy_evidence.retain(|nid, _| !exact_set.contains(nid));
        }

        let rankings: Vec<(String, Vec<i64>)> = vec![
            ("text".into(), ranking_text),
            ("image".into(), ranking_image),
            ("tag".into(), ranking_tag),
            ("exact".into(), exact_ids),
            ("fuzzy".into(), ranking_fuzzy),
        ];
        let priority: HashSet<String> = std::iter::once("exact".to_owned()).collect();
        let fused =
            crate::fusion::rrf_fuse(&rankings, &args.weights, crate::fusion::RRF_K, &priority);

        // Provenance (#182): best (lowest) rank first, ties by signal name.
        let mut matches: Vec<Value> = Vec::new();
        for (nid, _score, signals) in fused.into_iter().take(args.top_k) {
            let mut m = note_data
                .map
                .get(&nid)
                .expect("fused hit was a candidate")
                .clone();
            m["score"] = match sem_score.get(&nid) {
                Some(s) => json!(s),
                None => Value::Null,
            };
            let mut prov: Vec<(String, i64)> = signals;
            prov.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            m["provenance"] = Value::Array(
                prov.into_iter()
                    .map(|(sig, rank)| json!({"signal": sig, "rank": rank}))
                    .collect(),
            );
            if let Some((src, r, snippet)) = fuzzy_evidence.get(&nid) {
                m["fuzzy"] = json!({"source": src, "ref": r, "snippet": snippet});
            }
            matches.push(m);
        }
        results.push(json!({"source": source.label, "matches": matches}));
    }

    serde_json::from_value(Value::Array(results)).map_err(|e| {
        NativeError::internal(format!(
            "SearchResultGroup: assembly does not match the schema: {e}"
        ))
    })
}

/// The minimum query length the derived store's trigram index can serve —
/// re-exported so the harness's availability checks agree with the engine.
pub const SUBSTRING_MIN_QUERY: usize = MIN_TRIGRAM;

#[cfg(test)]
mod search_tests {
    use super::*;
    use crate::actions::tests::{add_note, temp_collection};

    fn derived_for(core: &CollectionCore, dir: &std::path::Path) -> DerivedEngine {
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        let ids = core.find_notes("deck:*").unwrap();
        let rows = core.derived_field_rows(&ids).unwrap();
        e.build(&rows, 1).unwrap();
        e
    }

    fn args(top_k: usize) -> SearchArgs {
        SearchArgs {
            top_k,
            threshold: 0.5,
            weights: [
                ("text".to_owned(), 1.0),
                ("image".to_owned(), 1.0),
                ("exact".to_owned(), 1.0),
                ("fuzzy".to_owned(), 0.5),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        }
    }

    fn query(text: &str) -> SearchSource {
        SearchSource {
            label: text.to_owned(),
            text: text.to_owned(),
            is_query: true,
        }
    }

    #[test]
    fn scoped_lexical_search_serves_from_the_store() {
        // #177 (scan retirement): a deck-scoped literal/fuzzy search rides
        // the FTS5 store with the scope id set pushed into the query — exact
        // recall inside the scope, zero leakage outside it, and the wildcard
        // `*text*` field scan is never consulted (the store served Some).
        let (dir, core) = temp_collection();
        let results: Vec<Value> = serde_json::from_str(
            &core
                .upsert_notes(
                    r#"[
                      {"note_type": "Basic", "deck": "Scoped",
                       "fields": {"Front": "the krebs cycle in scope", "Back": "b"}},
                      {"note_type": "Basic", "deck": "Other",
                       "fields": {"Front": "the krebs cycle out of scope", "Back": "b"}}
                    ]"#,
                    "error",
                    false,
                )
                .unwrap(),
        )
        .unwrap();
        let inside = results[0]["id"].as_i64().unwrap();
        let derived = derived_for(&core, &dir);

        let mut scoped_args = args(10);
        scoped_args.deck = Some("Scoped".to_owned());
        let groups = search_notes(
            &core,
            None,
            Some(&derived),
            None,
            &[query("krebs"), query("kreps cycle")], // literal + typo
            &[],
            &scoped_args,
        )
        .unwrap();

        let exact_ids: Vec<i64> = groups[0].matches.iter().map(|m| m.note.id).collect();
        assert_eq!(exact_ids, vec![inside], "literal: in-scope only");
        assert!(groups[0].matches[0].substring.is_some());

        let fuzzy_ids: Vec<i64> = groups[1].matches.iter().map(|m| m.note.id).collect();
        assert_eq!(fuzzy_ids, vec![inside], "fuzzy: in-scope only");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn lexical_only_search_exact_and_fuzzy() {
        // Semantic off (no index): exact substring hits + a fuzzy typo hit.
        let (dir, core) = temp_collection();
        let mito = add_note(&core, "the mitochondria is the powerhouse", "of the cell");
        add_note(&core, "momentum", "mass times velocity");
        let derived = derived_for(&core, &dir);

        let groups = search_notes(
            &core,
            None,
            Some(&derived),
            None,
            &[query("mitochondria"), query("mitochondira")], // exact + transposition
            &[],
            &args(10),
        )
        .unwrap();

        assert_eq!(groups.len(), 2);
        let exact = &groups[0].matches;
        assert_eq!(exact[0].note.id, mito);
        assert!(
            exact[0].substring.is_some(),
            "literal hit carries the annotation"
        );
        assert!(exact[0].score.is_none(), "no semantic ranking ran");
        assert!(exact[0].provenance.iter().any(|p| p.signal == "exact"));

        let fuzzy = &groups[1].matches;
        assert_eq!(
            fuzzy[0].note.id, mito,
            "typo recovered via the fuzzy signal"
        );
        assert!(fuzzy[0].fuzzy.is_some(), "fuzzy evidence carried");
        assert!(fuzzy[0].provenance.iter().any(|p| p.signal == "fuzzy"));
        core.close().unwrap();
    }

    #[test]
    fn semantic_ranking_thresholds_and_excludes() {
        let (_dir, core) = temp_collection();
        let a = add_note(&core, "alpha", "first");
        let b = add_note(&core, "beta", "second");
        let index = MultiModalIndex::new(vec!["text".to_owned()]).unwrap();
        // Hand-planted unit vectors: a ≈ query, b orthogonal (below threshold).
        index
            .add("text", &[a, b], &[vec![1.0, 0.0], vec![0.0, 1.0]])
            .unwrap();

        let mut a1 = args(10);
        a1.semantic = true;
        a1.index_size = 2;
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("nothing-literal-matches-this")],
            &[vec![1.0, 0.0]],
            &a1,
        )
        .unwrap();
        let matches = &groups[0].matches;
        assert_eq!(
            matches.len(),
            1,
            "orthogonal note is under the 0.5 threshold"
        );
        assert_eq!(matches[0].note.id, a);
        assert_eq!(matches[0].score, Some(1.0));
        assert!(matches[0].provenance.iter().any(|p| p.signal == "text"));

        // The anchor-exclusion path: excluding `a` leaves nothing above threshold.
        let mut a2 = a1.clone();
        a2.exclude = vec![a];
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("x-y-z-no-literal")],
            &[vec![1.0, 0.0]],
            &a2,
        )
        .unwrap();
        assert!(groups[0].matches.is_empty());
        core.close().unwrap();
    }

    #[test]
    fn exact_tier_outranks_semantic() {
        // A literal hit must float above a semantically-closer non-literal one.
        let (dir, core) = temp_collection();
        let literal = add_note(&core, "ATP synthase enzyme", "biology");
        let semantic = add_note(&core, "energy of motion", "physics");
        let derived = derived_for(&core, &dir);
        let index = MultiModalIndex::new(vec!["text".to_owned()]).unwrap();
        index
            .add(
                "text",
                &[literal, semantic],
                &[vec![0.0, 1.0], vec![1.0, 0.0]],
            )
            .unwrap();

        let mut a1 = args(10);
        a1.semantic = true;
        a1.index_size = 2;
        a1.threshold = 0.0;
        let groups = search_notes(
            &core,
            Some(&index),
            Some(&derived),
            None,
            &[query("ATP synthase")],
            &[vec![1.0, 0.0]], // semantically closest to the NON-literal note
            &a1,
        )
        .unwrap();
        let matches = &groups[0].matches;
        assert_eq!(
            matches[0].note.id, literal,
            "exact tier floats the literal hit"
        );
        assert_eq!(matches[1].note.id, semantic);
        core.close().unwrap();
    }

    #[test]
    fn deck_scope_filters_semantic_candidates() {
        let (_dir, core) = temp_collection();
        let a = add_note(&core, "alpha", "first"); // deck D
        let req = serde_json::json!([
            {"note_type": "Basic", "deck": "Other", "fields": {"Front": "beta", "Back": "x"}}
        ]);
        let out = core.upsert_notes(&req.to_string(), "allow", false).unwrap();
        let b = serde_json::from_str::<serde_json::Value>(&out).unwrap()[0]["id"]
            .as_i64()
            .unwrap();
        let index = MultiModalIndex::new(vec!["text".to_owned()]).unwrap();
        index
            .add("text", &[a, b], &[vec![1.0, 0.0], vec![0.9, 0.1]])
            .unwrap();

        let mut a1 = args(10);
        a1.semantic = true;
        a1.index_size = 2;
        a1.threshold = 0.0;
        a1.deck = Some("Other".to_owned());
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("zz-nothing-literal")],
            &[vec![1.0, 0.0]],
            &a1,
        )
        .unwrap();
        let matches = &groups[0].matches;
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].note.id, b, "only the in-deck candidate survives");
        core.close().unwrap();
    }

    #[test]
    fn substring_info_authority_and_snippet_window() {
        let content = serde_json::json!({"Front": "x".repeat(50) + "NEEDLE" + &"y".repeat(50)});
        let info = substring_info(Some(&content), "needle");
        assert!(!info.is_null());
        assert_eq!(info["matched_fields"], serde_json::json!(["Front"]));
        let snippet = info["snippet"].as_str().unwrap();
        assert!(snippet.starts_with('…') && snippet.ends_with('…'));
        assert!(snippet.contains("NEEDLE"));
        assert!(substring_info(Some(&content), "absent").is_null());
        assert!(substring_info(None, "x").is_null());
    }
}
