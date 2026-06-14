//! The action core (#331, kernel inversion S2) — slice 1: the read surface.
//!
//! Each action is the *whole* tool body: parameter normalization, the
//! collection-core call, and the Rust-canonical response type (#330) — typed
//! end-to-end since #391 phase 2 (the read surface returns `shrike-schemas`
//! types straight from the core; serialization happens once, at the host
//! edge). Python's `actions.py` shrinks to a binding per re-homed action:
//! typed signature (FastMCP's inputSchema source) + context assembly + the
//! completion-log fragment.
//!
//! Actions are synchronous over `&dyn Collection` (#389): the transitional harness
//! invokes them on its collection worker thread through the shrike-py
//! per-action bindings (the same serialization every collection op rides);
//! the kernel's async layer (S3, #332) will drive the same bodies through
//! [`crate::SerializedCollection`]. No threading, no runtime assumption here
//! (#308/#310).

use serde::de::DeserializeOwned;

use shrike_ffi::{NativeError, NativeResult};
use shrike_schemas::{CollectionInfo, ListNotesResponse};
use shrike_store_api::Collection;

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
///
/// The read surface no longer needs this (#391 phase 2: the core returns the
/// typed values directly); it stays for the unconverted modules — the
/// media/write/note-type re-homes still ride core-emitted JSON.
#[allow(dead_code)]
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
/// Typed end-to-end (#391 phase 2): the core builds the canonical type, the
/// action forwards it, and serialization happens once, at the host edge.
pub fn collection_info(
    core: &dyn Collection,
    include: &[String],
    note_type_details: &[String],
) -> NativeResult<CollectionInfo> {
    core.collection_info(include, note_type_details)
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
    core: &dyn Collection,
    params: &ListNotesParams,
) -> NativeResult<ListNotesResponse> {
    core.list_notes(
        params.ids.as_deref(),
        params.deck.as_deref(),
        params.tags.as_deref(),
        params.note_type.as_deref(),
        params.modified_since_epoch,
        params.with_fields,
        params.limit,
    )
}

/// `collection_query` — a raw Anki search expression (the read-only escape
/// hatch, #97). A malformed expression is invalid input (isolation marks
/// already stripped by the core's error decoding).
pub fn collection_query(
    core: &dyn Collection,
    query: &str,
    with_fields: bool,
    limit: usize,
) -> NativeResult<ListNotesResponse> {
    core.query(query, with_fields, limit)
}

// ── upsert dedup neighbors (#391 phase 1, re-homed from actions.py) ─────────

/// Dedup lexical-overlap cheapness gate (#206): the trigram OR-query grows
/// with text length, so texts are char-truncated before the fuzzy lookup and
/// the verify — plenty for near-verbatim detection, bounded for the
/// high-volume dedup path.
pub const DEDUP_LEXICAL_QUERY_CHARS: usize = 200;

/// The propose-verify floor (#206): the trigram index PROPOSES candidates
/// (recall — any shared trigrams), then whole-text similarity VERIFIES
/// near-verbatim before a candidate becomes a neighbor (precision — a shared
/// question stem is real overlap but not a near-duplicate).
pub const DEDUP_LEXICAL_MIN_RATIO: f64 = 0.6;

/// First `n` CHARS of `s` (Python's `s[:n]` — code points, never mid-char).
fn truncate_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// The lexical verifier (#206): whole-text similarity over the candidate's
/// PREFETCHED embedding text vs the (pre-stripped/lowered/truncated) draft —
/// exactly the near-verbatim question. A candidate absent from the prefetch
/// map (unreadable, deleted) verifies false, as the per-id read did.
fn near_verbatim(verify_texts: &HashMap<i64, String>, neighbor_id: i64, draft: &str) -> bool {
    let Some(candidate) = verify_texts.get(&neighbor_id) else {
        return false;
    };
    if candidate.is_empty() {
        return false;
    }
    crate::textsim::sequence_ratio(draft, candidate) >= DEDUP_LEXICAL_MIN_RATIO
}

struct NeighborCandidate {
    id: i64,
    score: Option<f64>,
    provenance: Vec<(String, i64)>,
}

/// Attach near-duplicate candidates to each upsert draft (#204; the policy
/// re-homed in #391 phase 1 — byte-faithful to the retired Python).
///
/// Two complementary signals per draft (#206): the semantic match catches
/// paraphrase dupes (cosine-thresholded, `vectors` host-embedded like the
/// search action's), and the trigram lexical-overlap propose-verify catches
/// near-verbatim restatements the embedding threshold misses — each hit
/// carrying `{signal, rank}` provenance (#208). `best` per draft is the
/// calibration sample (#207) the host's dedup-stats recorder consumes.
#[allow(clippy::too_many_arguments)]
pub fn attach_neighbors(
    core: &dyn Collection,
    index: Option<&dyn VectorIndex>,
    derived: Option<&dyn DerivedStore>,
    texts: &[String],
    vectors: &[Vec<f32>],
    exclude: &[i64],
    top_k: usize,
    threshold: f64,
) -> NativeResult<Vec<shrike_schemas::UpsertNeighbors>> {
    let exclude_set: HashSet<i64> = exclude.iter().copied().collect();

    // Semantic pass: one batched per-modality search over the text space
    // (neighbors are text-space candidates, mirroring the host's view).
    let sem: Vec<ModalityHits> = match index {
        Some(engine) if !vectors.is_empty() => {
            let spaces = vec!["text".to_string()];
            engine.search_by_modality(vectors, top_k + exclude_set.len(), Some(&spaces))?
        }
        _ => vec![ModalityHits::new(); texts.len()],
    };
    if sem.len() != texts.len() {
        return Err(NativeError::internal(format!(
            "attach_neighbors: {} rankings for {} drafts",
            sem.len(),
            texts.len()
        )));
    }

    // Lexical proposals per draft, gathered up front (#445 follow-up): the
    // per-(draft, proposal) verification needs each proposed candidate's
    // embedding text, and the ratio itself is an in-memory comparison — so
    // ONE batched note_texts over the proposal union replaces a per-proposal
    // singleton read (which paid an SQL query + notetype lookup each).
    let fuzzy_rows: Vec<Vec<i64>> = texts
        .iter()
        .map(|text| match derived {
            Some(d) if !text.trim().is_empty() => {
                match d.search_fuzzy(
                    truncate_chars(text, DEDUP_LEXICAL_QUERY_CHARS),
                    (top_k + exclude_set.len()) as i64,
                    None,
                    // The upsert-neighbor dedup signal is over note CONTENT;
                    // no lexical-visibility hiding applies here (#485).
                    &[],
                ) {
                    Ok(rows) => rows.into_iter().map(|(fid, ..)| fid).collect(),
                    Err(e) => {
                        tracing::debug!(error = ?e, "dedup lexical overlap unavailable");
                        Vec::new()
                    }
                }
            }
            _ => Vec::new(),
        })
        .collect();
    let verify_ids: Vec<i64> = fuzzy_rows
        .iter()
        .flatten()
        .copied()
        .filter(|fid| !exclude_set.contains(fid))
        .collect::<std::collections::BTreeSet<i64>>()
        .into_iter()
        .collect();
    let verify_texts: HashMap<i64, String> = if verify_ids.is_empty() {
        HashMap::new()
    } else {
        match core.note_texts(&verify_ids) {
            Ok(rendered) => verify_ids
                .iter()
                .zip(rendered)
                .map(|(id, t)| {
                    let lowered = t.trim().to_lowercase();
                    (
                        *id,
                        truncate_chars(&lowered, DEDUP_LEXICAL_QUERY_CHARS).to_string(),
                    )
                })
                .collect(),
            Err(e) => {
                // Unreadable verifies false — the singleton read's behavior.
                tracing::debug!(error = ?e, "dedup verify prefetch failed; proposals unverified");
                HashMap::new()
            }
        }
    };

    // Per-draft candidate assembly (no collection reads in this loop).
    let mut staged: Vec<(Vec<NeighborCandidate>, Option<f64>)> = Vec::with_capacity(texts.len());
    for ((text, per_query), proposals) in texts.iter().zip(sem.iter()).zip(fuzzy_rows.iter()) {
        // Insertion-ordered candidates: the final sort is stable, so ties
        // keep discovery order exactly like the Python dict did.
        let mut candidates: Vec<NeighborCandidate> = Vec::new();

        if let Some((ids, dists)) = per_query.get("text") {
            let mut sem_rank: i64 = 0;
            for (nid, dist) in ids.iter().zip(dists.iter()) {
                let score = round3(1.0 - *dist as f64);
                if score < threshold {
                    break; // distance-ascending: nothing further clears it
                }
                if exclude_set.contains(nid) {
                    continue;
                }
                sem_rank += 1;
                candidates.push(NeighborCandidate {
                    id: *nid,
                    score: Some(score),
                    provenance: vec![("text".to_string(), sem_rank)],
                });
                if sem_rank >= top_k as i64 {
                    break;
                }
            }
        }

        // Lexical overlap (#206): near-verbatim dupes the cosine gate
        // missed. Propose-verify; no cosine to report → score stays None.
        if !proposals.is_empty() {
            let draft_verify =
                truncate_chars(&text.trim().to_lowercase(), DEDUP_LEXICAL_QUERY_CHARS).to_string();
            let mut fuzzy_rank: i64 = 0;
            for fid in proposals {
                let fid = *fid;
                if exclude_set.contains(&fid) {
                    continue;
                }
                let existing = candidates.iter_mut().find(|c| c.id == fid);
                if existing.is_none() && !near_verbatim(&verify_texts, fid, &draft_verify) {
                    continue; // proposed but not verified — overlap, not a dupe
                }
                fuzzy_rank += 1;
                match existing {
                    Some(entry) => entry.provenance.push(("fuzzy".to_string(), fuzzy_rank)),
                    None => candidates.push(NeighborCandidate {
                        id: fid,
                        score: None,
                        provenance: vec![("fuzzy".to_string(), fuzzy_rank)],
                    }),
                }
                if fuzzy_rank >= top_k as i64 {
                    break;
                }
            }
        }

        // The calibration sample (#207): the draft's best SEMANTIC match, or
        // a no-match tick — dedup's own traffic, never the #201 calibration.
        let best = candidates
            .iter()
            .filter_map(|c| c.score)
            .fold(None, |acc: Option<f64>, s| {
                Some(acc.map_or(s, |a| if s > a { s } else { a }))
            });

        // Semantically-scored candidates first (descending score),
        // lexical-only after (by their first-signal rank); stable.
        candidates.sort_by(|x, y| {
            let kx = -(x.score.unwrap_or(-1.0));
            let ky = -(y.score.unwrap_or(-1.0));
            kx.total_cmp(&ky)
                .then(x.provenance[0].1.cmp(&y.provenance[0].1))
        });
        staged.push((candidates, best));
    }

    // ONE meta read for every candidate across the batch (#445 follow-up:
    // the per-candidate note_dicts singleton paid two SQL queries plus a
    // FULL deck_names enumeration each — the very N+1 #454 removed from the
    // search action). A whole-batch failure skips all neighbors with a
    // debug log, the same trade read_notes_batch makes; an id absent from
    // the map is the per-note skip-unreadable the singleton path had.
    let all_ids: Vec<i64> = staged
        .iter()
        .flat_map(|(cands, _)| cands.iter().map(|c| c.id))
        .collect::<std::collections::BTreeSet<i64>>()
        .into_iter()
        .collect();
    let meta_by_id: HashMap<i64, Value> = if all_ids.is_empty() {
        HashMap::new()
    } else {
        match core.note_dicts(&all_ids, false) {
            Ok(dicts) => dicts
                .into_iter()
                .filter_map(|d| d.get("id").and_then(Value::as_i64).map(|id| (id, d)))
                .collect(),
            Err(e) => {
                tracing::debug!(error = ?e, "neighbor meta batch failed; neighbors skipped");
                HashMap::new()
            }
        }
    };

    let mut out: Vec<shrike_schemas::UpsertNeighbors> = Vec::with_capacity(staged.len());
    for (candidates, best) in staged {
        // Metadata per surviving candidate (skip-unreadable, cap at top_k).
        let mut neighbors: Vec<shrike_schemas::Neighbor> = Vec::new();
        for cand in candidates {
            if neighbors.len() >= top_k {
                break;
            }
            let Some(meta) = meta_by_id.get(&cand.id) else {
                continue;
            };
            let tags: Vec<String> = meta
                .get("tags")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            neighbors.push(shrike_schemas::Neighbor {
                id: cand.id,
                score: cand.score,
                tags,
                provenance: cand
                    .provenance
                    .into_iter()
                    .map(|(signal, rank)| shrike_schemas::SignalContribution { signal, rank })
                    .collect(),
            });
        }
        out.push(shrike_schemas::UpsertNeighbors { neighbors, best });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrike_collection::CollectionCore;

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

    pub(super) fn add_note(core: &dyn Collection, front: &str, back: &str) -> i64 {
        let req = serde_json::json!([
            {"note_type": "Basic", "deck": "D", "fields": {"Front": front, "Back": back}}
        ]);
        let notes: Vec<shrike_schemas::NoteInput> = serde_json::from_value(req).unwrap();
        let results = serde_json::to_value(
            core.upsert_notes(&notes, shrike_collection::DuplicatePolicy::Allow, false)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(results[0]["status"], "created", "{results}");
        results[0]["id"].as_i64().unwrap()
    }

    #[test]
    fn collection_info_returns_typed_sections() {
        let (_dir, core) = temp_collection();
        add_note(&core, "Q", "A");
        let info: CollectionInfo =
            collection_info(&core, &["summary".into(), "decks".into()], &[]).unwrap();
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
        let resp: ListNotesResponse = list_notes(
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
        let resp: ListNotesResponse = collection_query(&core, "deck:D", false, 10).unwrap();
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

use shrike_derived::MIN_TRIGRAM;
use shrike_schemas::SearchResultGroup;
use shrike_store_api::{DerivedStore, VectorIndex};

/// One source's per-modality semantic rankings (`search_by_modality`'s row).
type ModalityHits = std::collections::BTreeMap<String, (Vec<i64>, Vec<f32>)>;

/// One SECONDARY embedding space's already-embedded + already-searched semantic
/// results, fed into cross-space fusion (#234). The PRIMARY text space's hits
/// ride the existing `vectors`/`index` path (unchanged, host-supplied); each
/// secondary text-capable space embeds the query with ITS OWN model and
/// searches ITS OWN engine at the kernel level, then hands the per-source rows
/// here as data — so `search_notes` stays the pure fusion assembly and never
/// holds N engines.
///
/// Empty `cross_space` (the N=1 / single-space case) → the rankings vector fed
/// to `rrf_fuse` is EXACTLY the per-modality set today, so the fused output is
/// byte-identical.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpaceSemantic {
    /// The space's CONTENT fingerprint — surfaced in per-space provenance
    /// (#182) only when N≥2 (vacuous/absent at N=1).
    pub space_key: String,
    /// One entry per search source, in `sources` order.
    pub per_source: Vec<SpaceSourceHits>,
    /// This space's OWN #201b intra-modal image activation floor
    /// (`mean + margin·std` of its image modality's typical best match),
    /// calibrated on its OWN index stats — NOT the primary's `image_floor`,
    /// which is calibrated on a different index (#576). `None` when the space
    /// is uncalibrated (text-only collection, too few samples), in which case
    /// the intra-modal floor is a no-op and only the relative gate applies.
    /// The kernel fills this from the space's orchestrator at fan-out time.
    #[serde(default)]
    pub image_floor: Option<f64>,
}

/// The cross-space fusion variant. `FloorAdmit` is the SHIPPED PRODUCTION
/// default since #580: the relative winner-take-all gate (#234) is RETIRED from
/// the production path — a secondary image space is admitted on its OWN
/// calibrated intra-modal floor (`image_best > z_floor`), independent of how the
/// text space did, so a strong on-topic CLIP hit reaches RRF and corroborates a
/// card even when the text space "won" on a spurious filename/lexical match (the
/// #580 corroboration win, measured on the real MiniLM+CLIP corpus:
/// `eval/search_quality/RESULTS_580.md`). The floor margin is the precision/
/// recall dial (`search.cross_space_fusion.margin`, threaded into calibration).
///
/// Retiring the relative gate is sound because >1 image-embedding space is a
/// config error (`profiles.resolve_profile`): with at most ONE image space
/// there is no multiplicity to guard, which was the relative gate's sole job
/// (the N≥2 flood — `cross_space_ungated_regresses_text_negative_control` +
/// `floor_admit_alone_floods_n2_but_budget_holds_it` document the
/// impossible-by-construction behaviour at the kernel level).
///
/// The other modes are EVAL-ONLY (`SHRIKE_CROSS_SPACE_FUSION_MODE`), kept to
/// reproduce the historical #576/#580 decision tables — they NEVER select in
/// production: the relative family (`Relative`, `RelativeFloor`, `SoftRelative`,
/// `SoftCalibrated`) reproduces the pre-#580 gate; `SoftFloorAdmit*` reproduces
/// the dominated soft variant (#580 §5: zero recall upside, re-opens the
/// over-return leak with τ); the `*Budget` modes reproduce the N≥2 multiplicity
/// measurement that justifies the single-image-space invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossSpaceFusionMode {
    /// PRODUCTION (#580) — FLOOR-ADMIT (binary): the relative gate is gone; admit
    /// a vision space iff its best surviving image cosine clears its OWN
    /// calibrated floor (`image_best > z_floor`), at full weight 1.0. An
    /// uncalibrated space (no floor) is admitted (the floor is a no-op). The
    /// single-image-space invariant (a config error otherwise) means the N≥2
    /// flood the relative gate used to guard cannot occur.
    #[default]
    FloorAdmit,
    /// V0+floor (eval) — the pre-#580 production default: relative gate AND a
    /// per-space calibrated intra-modal floor. Kept to reproduce the #576 table.
    RelativeFloor,
    /// V0 (eval) — binary relative gate only (`clip_best >= text_best`). The
    /// pre-#576 behaviour; leaks weak image cards when the primary's best cosine
    /// → 0. Kept to measure the leak the floor closes.
    Relative,
    /// V1 (eval) — soft-relative: weight `w = σ((clip_best − text_best)/τ)`
    /// folded into the `image#<key>` RRF weight. Calibration-free CONTROL — it
    /// still leaks (proves the leak is intra-modal, not relative).
    SoftRelative,
    /// V2 (eval) — soft-calibrated: weight `w = σ((z_s − z0)/τ)`, composed with
    /// the relative gate. The soft alternative to the hard floor.
    SoftCalibrated,
    /// #580 (eval) — FLOOR-ADMIT + WEIGHT BUDGET (binary): admit on the absolute
    /// floor, but bound the TOTAL vision RRF weight when N≥2 spaces fire by
    /// splitting a budget `B` (default 1.0, `cross_space_budget`) equally across
    /// the admitted spaces (each gets `B/N`). N=1 keeps full weight `B`. The
    /// budget held the N≥2 negative control without relative suppression — the
    /// MEASURED RATIONALE for the single-image-space invariant (moot in
    /// production, where N≥2 is a config error).
    FloorAdmitBudget,
    /// #580 (eval) — SOFT floor-admit (NO budget): drop the relative gate; weight
    /// each admitted space `w = σ((image_best − z_floor)/τ)`. DOMINATED (#580
    /// §5): no recall upside over binary, and re-opens the over-return leak as τ
    /// grows. Kept only to reproduce that finding.
    SoftFloorAdmit,
    /// #580 (eval) — SOFT floor-admit + WEIGHT BUDGET: the soft variant with the
    /// N≥2 budget (sum-scaled to `B`). Also dominated; kept for completeness.
    SoftFloorAdmitBudget,
}

/// One secondary space's per-source search result: its per-modality hits plus
/// the raw best query→match cosine the RELATIVE activation gate (#234) reads
/// BEFORE RRF strips magnitude.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SpaceSourceHits {
    /// This space's `search_by_modality` row for the source (its own engine,
    /// its own query embedding).
    pub modality_hits: ModalityHits,
    /// The space's best query→match cosine for this source (`1 - rank-1
    /// distance`, over the NOTE-item modalities). `None` = the space returned
    /// no hits. The relative gate fires this space only when this clears the
    /// PRIMARY text space's best query cosine for the same source.
    pub best_query_cosine: Option<f64>,
}

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
///
/// `Some(&Value::Null)` behaves exactly like `None` (the `Object` match arm
/// is the only productive path): note dicts serialize with plain serde since
/// the #391 to_wire retirement, so a meta-mode note carries an explicit
/// `"content": null` and `data.get("content")` yields `Some(Null)`, not
/// `None` — both mean "no content", and both return `Null` here.
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
    /// Per-signal RRF weights. EMPTY means the canonical set
    /// (`fusion::search_weights`) — the sentinel a future `--search-*`
    /// override knob must preserve; a non-empty map is used verbatim.
    pub weights: std::collections::BTreeMap<String, f64>,
    /// Whether semantic ranking runs (index ready + backend up); `vectors`
    /// carries one query vector per source when true.
    pub semantic: bool,
    /// The index's current vector count, for the over-fetch clamp.
    pub index_size: usize,
    /// Derived `source` strings hidden from the lexical (substring/fuzzy)
    /// surfaces (#485): a VectorOnly recognition source (VLM describe) is
    /// stored for provenance + reconcile but never surfaced on a lexical
    /// query. EMPTY (the default) = nothing hidden, the pre-#485 behaviour.
    pub hidden_lexical_sources: Vec<String>,
    /// SECONDARY embedding spaces' semantic results for cross-space fusion
    /// (#234), one per space, each already embedded + searched at the kernel
    /// level. EMPTY (the default) is the N=1 / single-space case — the rankings
    /// fed to `rrf_fuse` are then EXACTLY today's per-modality set, so the fused
    /// output is byte-identical. Non-empty appends each space's gated `image`
    /// ranking (the relative activation gate, below).
    pub cross_space: Vec<SpaceSemantic>,
    /// Disable the cross-space relative activation gate (#234) — the NEGATIVE
    /// CONTROL only. `false` (the default) keeps the mandatory gate on; `true`
    /// fires every secondary space's image ranking ungated, which the eval
    /// showed floods text queries and regresses text recall (the load-bearing
    /// 0.08-vs-1.00 separation a test pins). Never set in production.
    pub disable_cross_space_gate: bool,
    /// The cross-space fusion variant (#576). `Relative` (the default) is
    /// today's binary relative gate — production behaviour is unchanged until
    /// the experiment's data selects a winner. The floor/soft modes are
    /// eval-selectable measurement variants.
    pub cross_space_fusion_mode: CrossSpaceFusionMode,
    /// The temperature τ for the soft variants (#576): smaller τ → a sharper
    /// taper (τ→0 is the binary floor limit). Ignored by the binary modes.
    pub cross_space_tau: f64,
    /// The total vision-WEIGHT BUDGET `B` for the #580 `*Budget` floor-admission
    /// modes: when N≥2 vision spaces clear their floor, their combined RRF weight
    /// is bounded to `B` (split equally in the binary mode, sum-scaled in the
    /// soft mode). N=1 keeps full weight `B`. `<= 0.0` (the `Default`) is read as
    /// the canonical `1.0` so a space's default weight is unchanged — the budget
    /// modes only ever DIVIDE down from there. Ignored by the non-budget modes.
    pub cross_space_budget: f64,
}

impl SearchArgs {
    /// The effective vision-weight budget — `<= 0.0` reads as the canonical
    /// `1.0` (a single fired space then keeps full weight, matching the
    /// non-budget modes; `Default::default()` zeroes the field, so this keeps
    /// the eval's `..Default::default()` construction honest).
    fn effective_budget(&self) -> f64 {
        if self.cross_space_budget > 0.0 {
            self.cross_space_budget
        } else {
            1.0
        }
    }
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

/// Batch-hydrate candidate dicts (#445): ONE `note_dicts` call per ranking
/// replaces the old one-call-per-candidate shape (each singleton paid two
/// DB-proxy queries plus a full `deck_names` RPC plus a full notetype proto —
/// per candidate, hundreds of times per search). Returns `nid -> dict` for
/// ids not already hydrated; a missing/unreadable id is simply absent (the
/// per-note skip the singleton path had).
fn read_notes_batch(
    core: &dyn Collection,
    note_data: &NoteData,
    ids: &[i64],
) -> HashMap<i64, Value> {
    let missing: Vec<i64> = ids
        .iter()
        .copied()
        .filter(|nid| !note_data.contains(*nid))
        .collect();
    if missing.is_empty() {
        return HashMap::new();
    }
    match core.note_dicts(&missing, true) {
        Ok(dicts) => dicts
            .into_iter()
            .filter_map(|d| d.get("id").and_then(Value::as_i64).map(|id| (id, d)))
            .collect(),
        Err(e) => {
            tracing::debug!(error = ?e, "search: batch hydrate failed; candidates skipped");
            HashMap::new()
        }
    }
}

/// The fusion signal name for a SECONDARY vision space's image ranking (#234):
/// `image#<space-key>`. Distinct per space so provenance identifies which
/// vision space surfaced a note and each space fuses as its own RRF signal
/// (the canonical `search_weights` has no entry → `rrf_fuse` defaults its weight
/// to 1.0, the eval's equal weighting). Never collides with the primary's plain
/// `image` signal, so N=1 (no secondary) emits exactly today's signal set.
pub fn cross_space_signal(space_key: &str) -> String {
    format!("image#{space_key}")
}

/// The best (highest) query→match cosine across a space's NOTE-item modalities
/// for one source — `1 - rank-1 distance`, maxed over `text`/`image` (#234). The
/// relative cross-space activation gate compares a vision space's value to the
/// dedicated text space's value for the same query. `None` when the space
/// returned no note-item hits. The rank-1 distance is the smallest (the engine
/// returns distance-ascending), so this reads `[0]`.
///
/// Public so the kernel's cross-space fan-out captures each SECONDARY space's
/// value as it searches (the gate input rides into `SpaceSourceHits`).
pub fn best_query_cosine_of(
    hits: &std::collections::BTreeMap<String, (Vec<i64>, Vec<f32>)>,
) -> Option<f64> {
    crate::NOTE_MODALITIES
        .iter()
        .filter_map(|m| hits.get(*m).and_then(|(_, dists)| dists.first()))
        .map(|d| 1.0 - f64::from(*d))
        .fold(None, |acc, c| Some(acc.map_or(c, |a: f64| a.max(c))))
}

/// The #201b intra-modal activation floor from `(mean, std)` of a modality's
/// typical best match (`mean + margin·std`), the kernel mirror of
/// `shrike.index.activation_floor`. The single source of the floor formula —
/// the harness-side secondary calibration (#576) routes through it. `None`
/// (uncalibrated — too few samples) → no floor, the gate disabled there.
pub fn activation_floor(stats: Option<(f64, f64)>, margin: f64) -> Option<f64> {
    stats.map(|(mean, std)| mean + margin * std)
}

/// The host-side `ACTIVATION_MARGIN` (#201b) mirrored for the kernel-computed
/// cross-space floor — kept in lockstep with `shrike.actions.ACTIVATION_MARGIN`.
pub const ACTIVATION_MARGIN: f64 = 1.0;

/// Logistic squash `σ(x) = 1/(1+e^-x)` for the #576 soft-weight variants.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The PER-NOTE image activation floor (#582): retain in `ranking` only the
/// notes whose OWN image cosine (`score[id]`) clears `floor`, and prune the
/// dropped notes from `score` so the displayed-`score` fold and the `image_best`
/// read stay consistent. The pre-#582 behaviour was a per-SPACE gate (admit the
/// WHOLE ranking iff the rank-1 cosine cleared the floor); this is the correct
/// per-note granularity — a below-floor tail card no longer rides in on a
/// strong rank-1's coat-tails, so it carries no spurious image signal/provenance.
///
/// `floor = None` (an uncalibrated space) is a no-op (the floor can't judge).
/// It can only TIGHTEN: every kept note cleared the floor, and the rank-1 (if
/// it cleared) is unchanged — so a genuine cross-modal find (above-floor by
/// construction) is preserved, while an ∅-gold ranking whose best is sub-floor
/// is emptied exactly as the per-space gate did.
fn apply_image_floor(ranking: &mut Vec<i64>, score: &mut HashMap<i64, f64>, floor: Option<f64>) {
    let Some(floor) = floor else {
        return;
    };
    ranking.retain(|nid| score.get(nid).is_some_and(|&c| c > floor));
    score.retain(|_, &mut c| c > floor);
}

#[allow(clippy::too_many_arguments)]
fn rank_modality(
    core: &dyn Collection,
    hits_keys: &[i64],
    hits_distances: &[f32],
    note_data: &mut NoteData,
    sem_score: &mut HashMap<i64, f64>,
    exclude: &HashSet<i64>,
    args: &SearchArgs,
    thresholded: bool,
) -> Vec<i64> {
    // Prospective candidates (exclude/threshold pass) hydrate in ONE batch;
    // the loop below then filters scope and ranks exactly as before.
    let prospective: Vec<i64> = hits_keys
        .iter()
        .zip(hits_distances.iter())
        .filter(|(nid, _)| !exclude.contains(nid))
        .take_while(|(_, dist)| !thresholded || round3(1.0 - f64::from(**dist)) >= args.threshold)
        .map(|(nid, _)| *nid)
        .collect();
    let mut hydrated = read_notes_batch(core, note_data, &prospective);
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
            let data = match hydrated.remove(&nid) {
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
    core: &dyn Collection,
    derived: Option<&dyn DerivedStore>,
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
    let hidden: Vec<&str> = args
        .hidden_lexical_sources
        .iter()
        .map(String::as_str)
        .collect();
    // `Ok(None)` = the store can't serve this query (a sub-trigram query, or no
    // store at all) → the `find_notes` field-text fallback below is correct.
    // `Err` = a REAL derived-read failure (e.g. a transient SQLITE_BUSY that
    // outlived the busy-retry). It must NOT silently fall back to `find_notes`
    // (#644): OCR/ASR text lives ONLY in the derived store, never in an anki
    // field, so the field-text fallback structurally cannot serve it — a silent
    // fallback would drop the OCR/ASR `exact`/substring signal with no error.
    // Surface it instead (the store's own contract: "a MATCH error is a real
    // error; the caller decides whether to degrade" — and degrading to a
    // fallback that can't serve derived sources is not a valid degradation).
    let lex = if let Some(d) = derived {
        d.search_substring(text, (args.top_k + exclude.len()) as i64, scope, &hidden)?
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
        let mut hydrated = read_notes_batch(core, note_data, &candidates);
        let mut added = 0usize;
        for nid in candidates.iter().copied() {
            if note_data.contains(nid) {
                continue;
            }
            let mut data = match hydrated.remove(&nid) {
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

    let row_ids: Vec<i64> = rows
        .iter()
        .map(|(nid, ..)| *nid)
        .filter(|nid| !exclude.contains(nid))
        .collect();
    let mut hydrated = read_notes_batch(core, note_data, &row_ids);
    let mut added = 0usize;
    for (nid, source, reference, snippet) in rows {
        if exclude.contains(&nid) || note_data.contains(nid) {
            continue; // store may return a row per field
        }
        let mut data = match hydrated.remove(&nid) {
            Some(d) => d,
            None => continue,
        };
        if !in_scope(&data, args.deck.as_deref(), &args.tags) {
            continue;
        }
        // A derived-source row is its own authority (#199/#388): FTS5
        // matched the stored text's literal trigrams, and the field-content
        // re-check below would wrongly reject a literal living only in an
        // OCR/ASR row. Provenance carries the source + ref so the result can
        // say where it hit; field rows stay with the substring_info
        // authority over rendered content.
        if source != crate::FIELD_SOURCE {
            data["substring"] = json!({
                "matched_fields": Vec::<String>::new(),
                "snippet": snippet,
                "source": source,
                "ref": reference,
            });
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
    core: &dyn Collection,
    derived: Option<&dyn DerivedStore>,
    text: &str,
    note_data: &mut NoteData,
    exclude: &HashSet<i64>,
    args: &SearchArgs,
    scope: Option<&[i64]>,
) -> NativeResult<(Vec<i64>, FuzzyEvidence)> {
    let Some(d) = derived else {
        return Ok((Vec::new(), HashMap::new()));
    };
    let hidden: Vec<&str> = args
        .hidden_lexical_sources
        .iter()
        .map(String::as_str)
        .collect();
    // A real derived-read failure surfaces (#644): the fuzzy signal has no
    // anki-field fallback at all (no `find_notes` path here), so silently
    // returning empty would drop OCR/ASR fuzzy matches with no error. Propagate.
    let hits = d.search_fuzzy(text, args.top_k as i64, scope, &hidden)?;
    let hit_ids: Vec<i64> = hits
        .iter()
        .map(|(nid, ..)| *nid)
        .filter(|nid| !exclude.contains(nid))
        .collect();
    let mut hydrated = read_notes_batch(core, note_data, &hit_ids);
    let mut ranking: Vec<i64> = Vec::new();
    let mut evidence: FuzzyEvidence = HashMap::new();
    for (nid, source, r, snippet) in hits {
        if exclude.contains(&nid) || evidence.contains_key(&nid) {
            continue;
        }
        if !note_data.contains(nid) {
            let data = match hydrated.remove(&nid) {
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
    Ok((ranking, evidence))
}

/// The fused search assembly (see module-section comment). `vectors` carries
/// one query vector per source when `args.semantic`.
pub fn search_notes(
    core: &dyn Collection,
    index: Option<&dyn VectorIndex>,
    derived: Option<&dyn DerivedStore>,
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
        // #582: per-NOTE image floor — keep only the cards whose own image
        // cosine clears the floor (was a per-space all-or-nothing gate on the
        // rank-1). A below-floor tail card no longer rides in on the rank-1's
        // coat-tails. When the rank-1 itself is sub-floor the whole ranking
        // empties (the old behaviour falls out as the rank-1 case).
        apply_image_floor(&mut ranking_image, &mut image_score, args.image_floor);
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
            )?
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

        let mut rankings: Vec<(String, Vec<i64>)> = vec![
            ("text".into(), ranking_text),
            ("image".into(), ranking_image),
            ("tag".into(), ranking_tag),
            ("exact".into(), exact_ids),
            ("fuzzy".into(), ranking_fuzzy),
        ];

        // ── Cross-space fusion (#234 / #576 / #580) ──────────────────────────
        // Each SECONDARY image space contributes its own `image` ranking. EMPTY
        // cross_space (N=1 — the production-common case) → this whole block is a
        // no-op and the rankings vector above is byte-identical, so `rrf_fuse`
        // gets identical inputs.
        //
        // PRODUCTION (#580): FLOOR-ADMIT. The relative winner-take-all gate is
        // RETIRED — a secondary image space is admitted on its OWN calibrated
        // intra-modal floor (`image_best > z_floor`), independent of how the text
        // space did, so a strong on-topic CLIP hit corroborates the card even
        // when text "won" on a spurious filename lexical match (the #580 win).
        // Sound because >1 image space is a config error (`profiles`): with at
        // most one image space there is no multiplicity, which was the relative
        // gate's only job.
        //
        // EVAL-ONLY (`SHRIKE_CROSS_SPACE_FUSION_MODE`): the relative family
        // (`Relative*`/`*Floor` non-admit) reproduces the pre-#580 gate; the
        // `Soft*`/`*Budget` modes reproduce the dominated soft variant + the N≥2
        // multiplicity measurement. `uses_relative_gate` keeps the gate for the
        // relative family ONLY; the floor-admit family skips it.
        //
        // Per-space soft/budget weights (eval modes) collected here, applied to
        // the `image#<key>` signal's RRF weight after the canonical weights
        // resolve. Two passes: (1) admit + raw weight per space; (2) the budget
        // normalization across the admitted set (a no-op for the production
        // FloorAdmit mode), then fold/push.
        let mut cross_space_weights: std::collections::BTreeMap<String, f64> =
            std::collections::BTreeMap::new();
        if !args.cross_space.is_empty() {
            let mode = args.cross_space_fusion_mode;
            let uses_relative_gate = matches!(
                mode,
                CrossSpaceFusionMode::Relative
                    | CrossSpaceFusionMode::RelativeFloor
                    | CrossSpaceFusionMode::SoftRelative
                    | CrossSpaceFusionMode::SoftCalibrated
            );
            let is_budget = matches!(
                mode,
                CrossSpaceFusionMode::FloorAdmitBudget | CrossSpaceFusionMode::SoftFloorAdmitBudget
            );
            // The primary text space's best query cosine (the relative gate's
            // reference). The primary's hits are `modality_hits` (this row).
            let primary_best = best_query_cosine_of(modality_hits);
            // Pass 1: collect each admitted space's ranking, its per-note image
            // scores (for the displayed-score fold), and its RAW pre-budget
            // weight.
            struct Admitted {
                signal: String,
                ranking: Vec<i64>,
                space_score: HashMap<i64, f64>,
                weight: f64,
            }
            let mut admitted: Vec<Admitted> = Vec::new();
            for space in &args.cross_space {
                let Some(shits) = space.per_source.get(i) else {
                    continue;
                };
                // The relative gate (#234) — applied ONLY for the relative
                // family. With the gate disabled (the negative control), every
                // space fires. The floor-admit family skips it entirely (the
                // floor below is the sole admission test).
                if uses_relative_gate {
                    let gate_open = args.disable_cross_space_gate
                        || match (shits.best_query_cosine, primary_best) {
                            (Some(v), Some(p)) => v >= p,
                            // No primary text reference → nothing to gate
                            // against; admit (degenerate, lexical-only primary).
                            (Some(_), None) => true,
                            // The space itself returned no hits → nothing to add.
                            (None, _) => false,
                        };
                    if !gate_open {
                        continue;
                    }
                }
                let (sk, sd) = shits
                    .modality_hits
                    .get("image")
                    .map(|(k, d)| (k.as_slice(), d.as_slice()))
                    .unwrap_or((&[], &[]));
                if sk.is_empty() {
                    continue;
                }
                let mut space_score: HashMap<i64, f64> = HashMap::new();
                let mut ranking_space_image = rank_modality(
                    core,
                    sk,
                    sd,
                    &mut note_data,
                    &mut space_score,
                    &exclude,
                    args,
                    false,
                );
                if ranking_space_image.is_empty() {
                    continue;
                }
                // #582 (PRODUCTION FloorAdmit only): a PER-NOTE floor — keep only
                // the cards whose own image cosine clears this space's floor, so
                // a below-floor tail card carries no `image#clip`. The eval modes
                // keep the historical per-SPACE rule below (on `image_best`) so
                // they reproduce the decision tables unchanged. An uncalibrated
                // space (no floor) is a no-op either way.
                if matches!(
                    mode,
                    CrossSpaceFusionMode::FloorAdmit | CrossSpaceFusionMode::FloorAdmitBudget
                ) {
                    apply_image_floor(
                        &mut ranking_space_image,
                        &mut space_score,
                        space.image_floor,
                    );
                    if ranking_space_image.is_empty() {
                        continue; // every card fell below the floor → nothing to add
                    }
                }
                // The best surviving image cosine — the value the eval modes'
                // per-space floor judges (exactly the primary's old rank-1 rule).
                let image_best = ranking_space_image
                    .first()
                    .and_then(|nid| space_score.get(nid).copied());

                // Apply the fusion mode's admission + raw weight. `None` drops
                // the space; `Some(w)` admits it at raw weight `w` (the budget
                // pass may scale it down for the `*Budget` modes).
                let weight = match mode {
                    // V0 — relative only (today): contribute at weight 1.0.
                    CrossSpaceFusionMode::Relative => Some(1.0),
                    // V0+floor — relative AND z_s > z_floor (per-space). Drop
                    // the space when its best surviving image cosine clears no
                    // floor; an uncalibrated space (no floor) is admitted (the
                    // floor is a no-op, the relative gate alone governs).
                    CrossSpaceFusionMode::RelativeFloor => match (image_best, space.image_floor) {
                        (Some(b), Some(floor)) if b <= floor => None,
                        _ => Some(1.0),
                    },
                    // V1 — soft-relative: w = σ((clip_best − text_best)/τ). The
                    // calibration-free CONTROL (expected to still leak).
                    CrossSpaceFusionMode::SoftRelative => {
                        match (shits.best_query_cosine, primary_best, args.cross_space_tau) {
                            (Some(v), Some(p), tau) if tau > 0.0 => Some(sigmoid((v - p) / tau)),
                            // No reference / τ≤0 → fall back to the binary
                            // contribution (the σ→step limit).
                            _ => Some(1.0),
                        }
                    }
                    // V2 — soft-calibrated: w = σ((z_s − z0)/τ), composed with
                    // the relative gate (already applied above). The ren
                    // proposal — taper near the per-space floor instead of a
                    // hard drop. An uncalibrated space falls back to weight 1.0.
                    CrossSpaceFusionMode::SoftCalibrated => {
                        match (image_best, space.image_floor, args.cross_space_tau) {
                            (Some(b), Some(floor), tau) if tau > 0.0 => {
                                Some(sigmoid((b - floor) / tau))
                            }
                            _ => Some(1.0),
                        }
                    }
                    // #580/#582 FloorAdmit / FloorAdmitBudget — admit on the
                    // absolute floor alone (relative gate already skipped). The
                    // PER-NOTE floor filter above already dropped sub-floor cards
                    // and `continue`d if none survived, so any surviving ranking
                    // has cleared the floor → raw weight 1.0 (the budget pass
                    // divides it down for the budget mode).
                    CrossSpaceFusionMode::FloorAdmit | CrossSpaceFusionMode::FloorAdmitBudget => {
                        Some(1.0)
                    }
                    // #580 SoftFloorAdmit / SoftFloorAdmitBudget — soft admission
                    // `w = σ((image_best − z_floor)/τ)`, no relative composition.
                    // An uncalibrated space / τ≤0 falls back to weight 1.0. There
                    // is no hard drop: a sub-floor hit gets a near-zero weight
                    // (negligible RRF mass), which the soft taper is for.
                    CrossSpaceFusionMode::SoftFloorAdmit
                    | CrossSpaceFusionMode::SoftFloorAdmitBudget => {
                        match (image_best, space.image_floor, args.cross_space_tau) {
                            (Some(b), Some(floor), tau) if tau > 0.0 => {
                                Some(sigmoid((b - floor) / tau))
                            }
                            _ => Some(1.0),
                        }
                    }
                };
                let Some(weight) = weight else {
                    continue; // the floor dropped this space's image ranking
                };
                admitted.push(Admitted {
                    signal: cross_space_signal(&space.space_key),
                    ranking: ranking_space_image,
                    space_score,
                    weight,
                });
            }

            // Pass 2: the #580 vision-weight BUDGET. When N≥2 admitted spaces
            // share a bounded total weight `B`, no flood of always-confident
            // off-topic spaces can out-fuse the text answer (the negative
            // control the relative gate used to guard). N=1 keeps full weight
            // (the production-common case is unpenalized). Two scalings:
            //   - binary budget (FloorAdmitBudget): every weight is 1.0, so each
            //     becomes B/N (the equal split).
            //   - soft budget (SoftFloorAdmitBudget): only scale DOWN, and only
            //     when the soft weights already sum above B (`B / Σ raw`); a
            //     confident N=1 hit keeps its near-1.0 weight.
            if is_budget && admitted.len() >= 2 {
                let budget = args.effective_budget();
                let total: f64 = admitted.iter().map(|a| a.weight).sum();
                if total > budget && total > 0.0 {
                    let scale = budget / total;
                    for a in &mut admitted {
                        a.weight *= scale;
                    }
                }
            }

            // Fold + push each admitted space.
            for a in admitted {
                // Fold the space's image cosines into the displayed semantic
                // `score` (max-over-items, exactly like the primary image).
                for (nid, isim) in &a.space_score {
                    let entry = sem_score.entry(*nid).or_insert(*isim);
                    if *isim > *entry {
                        *entry = *isim;
                    }
                }
                // A DISTINCT signal name per space so provenance (#182)
                // identifies which vision space surfaced the note, and each
                // space fuses as its own RRF signal (weight defaults to 1.0 —
                // the eval's equal weighting; the soft/budget modes override it).
                // `image#<key>` reads as "the image modality of space <key>".
                if (a.weight - 1.0).abs() > f64::EPSILON {
                    cross_space_weights.insert(a.signal.clone(), a.weight);
                }
                rankings.push((a.signal, a.ranking));
            }
        }

        // Host-supplied weights override; empty means the kernel's canonical
        // set (#388 — the one source of truth in `fusion`).
        let mut weights = if args.weights.is_empty() {
            crate::fusion::search_weights()
        } else {
            args.weights.clone()
        };
        // #576 soft variants: the per-space `image#<key>` weight (a per-query
        // taper) overrides the canonical default of 1.0 for that signal.
        weights.extend(cross_space_weights);
        let priority: HashSet<String> =
            std::iter::once(crate::fusion::PRIORITY_SIGNAL.to_owned()).collect();
        let fused = crate::fusion::rrf_fuse(&rankings, &weights, crate::fusion::RRF_K, &priority);

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
    use shrike_derived::DerivedEngine;
    use shrike_index::MultiModalIndex;

    fn derived_for(core: &dyn Collection, dir: &std::path::Path) -> DerivedEngine {
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

    /// Unit vector at exact cosine `1 - d` against the `[1, 0]` query (the
    /// cross-space tests plant primary text vectors at known cosines).
    fn at_distance(d: f32) -> Vec<f32> {
        let sim = 1.0 - d;
        vec![sim, (1.0 - sim * sim).max(0.0).sqrt()]
    }

    #[test]
    fn scoped_lexical_search_serves_from_the_store() {
        // #177 (scan retirement): a deck-scoped literal/fuzzy search rides
        // the FTS5 store with the scope id set pushed into the query — exact
        // recall inside the scope, zero leakage outside it, and the wildcard
        // `*text*` field scan is never consulted (the store served Some).
        let (dir, core) = temp_collection();
        let scoped_notes: Vec<shrike_schemas::NoteInput> = serde_json::from_str(
            r#"[
              {"note_type": "Basic", "deck": "Scoped",
               "fields": {"Front": "the krebs cycle in scope", "Back": "b"}},
              {"note_type": "Basic", "deck": "Other",
               "fields": {"Front": "the krebs cycle out of scope", "Back": "b"}}
            ]"#,
        )
        .unwrap();
        let results: Vec<Value> = serde_json::from_str(
            &serde_json::to_string(
                &core
                    .upsert_notes(
                        &scoped_notes,
                        shrike_collection::DuplicatePolicy::Error,
                        false,
                    )
                    .unwrap(),
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

    /// A `DerivedStore` that delegates to a real engine but forces the two
    /// lexical reads (`search_substring`/`search_fuzzy`) to return `Err` — the
    /// shape of a derived-read failure that survives the busy-retry (#644). All
    /// other methods delegate so build/ingest still work.
    struct FlakyDerived {
        inner: DerivedEngine,
        fail_lexical: bool,
    }
    impl shrike_store_api::DerivedStore for FlakyDerived {
        fn build(&self, rows: &[(i64, String, String, String)], col_mod: i64) -> NativeResult<()> {
            self.inner.build(rows, col_mod)
        }
        fn ingest(&self, n: i64, s: &str, r: &[(String, String)]) -> NativeResult<()> {
            self.inner.ingest(n, s, r)
        }
        fn ingest_many(
            &self,
            notes: &[(i64, Vec<(String, String)>)],
            source: &str,
        ) -> NativeResult<()> {
            self.inner.ingest_many(notes, source)
        }
        fn remove(&self, ids: &[i64], source: Option<&str>) -> NativeResult<()> {
            self.inner.remove(ids, source)
        }
        fn count(&self) -> NativeResult<i64> {
            self.inner.count()
        }
        fn get_col_mod(&self) -> Option<i64> {
            self.inner.get_col_mod()
        }
        fn set_col_mod(&self, v: i64) -> NativeResult<()> {
            self.inner.set_col_mod(v)
        }
        fn meta_get(&self, k: &str) -> NativeResult<Option<String>> {
            self.inner.meta_get(k)
        }
        fn meta_set(&self, k: &str, v: &str) -> NativeResult<()> {
            self.inner.meta_set(k, v)
        }
        fn refs_for_source(&self, s: &str) -> NativeResult<Vec<(i64, String)>> {
            self.inner.refs_for_source(s)
        }
        fn texts_for_source(&self, s: &str) -> NativeResult<Vec<(i64, String, String)>> {
            self.inner.texts_for_source(s)
        }
        fn texts_for_source_for_notes(
            &self,
            s: &str,
            ids: &[i64],
        ) -> NativeResult<Vec<(i64, String, String)>> {
            self.inner.texts_for_source_for_notes(s, ids)
        }
        fn mark_gated(&self, s: &str, pairs: &[(i64, String)]) -> NativeResult<()> {
            self.inner.mark_gated(s, pairs)
        }
        fn gated_refs_for_source(&self, s: &str) -> NativeResult<Vec<(i64, String)>> {
            self.inner.gated_refs_for_source(s)
        }
        fn clear_gated(&self, s: &str) -> NativeResult<()> {
            self.inner.clear_gated(s)
        }
        fn put_segments(&self, n: i64, s: &str, r: &str, j: &str) -> NativeResult<()> {
            self.inner.put_segments(n, s, r, j)
        }
        fn get_segments(&self, n: i64, s: &str, r: &str) -> NativeResult<Option<String>> {
            self.inner.get_segments(n, s, r)
        }
        fn match_rows(
            &self,
            expr: &str,
            limit: i64,
            with_text: bool,
            scope: Option<&[i64]>,
            exclude: &[&str],
        ) -> NativeResult<Vec<shrike_store_api::MatchRow>> {
            self.inner
                .match_rows(expr, limit, with_text, scope, exclude)
        }
        fn search_substring(
            &self,
            q: &str,
            limit: i64,
            scope: Option<&[i64]>,
            exclude: &[&str],
        ) -> NativeResult<Option<Vec<shrike_store_api::LexicalRow>>> {
            if self.fail_lexical {
                return Err(NativeError::unavailable("derived busy (simulated)"));
            }
            self.inner.search_substring(q, limit, scope, exclude)
        }
        fn search_fuzzy(
            &self,
            q: &str,
            top_k: i64,
            scope: Option<&[i64]>,
            exclude: &[&str],
        ) -> NativeResult<Vec<shrike_store_api::LexicalRow>> {
            if self.fail_lexical {
                return Err(NativeError::unavailable("derived busy (simulated)"));
            }
            self.inner.search_fuzzy(q, top_k, scope, exclude)
        }
    }

    #[test]
    fn derived_read_error_surfaces_never_silently_field_falls_back() {
        // #644 (the correctness fix): a REAL derived-read failure (a busy that
        // outlived the retry) must SURFACE from search_notes, not silently fall
        // back to the `find_notes("*text*")` field scan. The field scan can't
        // serve OCR/ASR text (it lives only in the derived store, never in an
        // anki field), so a silent fallback would drop the OCR `exact`/`fuzzy`
        // signal with NO error — exactly the silent degradation #644 is about.
        let (dir, core) = temp_collection();
        // Ingest an OCR row whose text is NOT in any anki field (the note's
        // field text is unrelated), so only the derived store can serve it.
        let nid = add_note(&core, "unrelated field text", "back");
        let derived = derived_for(&core, &dir);
        derived
            .ingest(
                nid,
                "ocr",
                &[("photo.png".into(), "chlorophyll spectrum".into())],
            )
            .unwrap();
        // Sanity: with a healthy store the OCR text IS found with an `exact`
        // signal (the very thing the flake intermittently lost).
        let ok = search_notes(
            &core,
            None,
            Some(&derived),
            None,
            &[query("chlorophyll spectrum")],
            &[],
            &args(10),
        )
        .unwrap();
        let ok_hit = ok[0].matches.iter().find(|m| m.note.id == nid).unwrap();
        assert!(
            ok_hit.provenance.iter().any(|p| p.signal == "exact"),
            "healthy store: OCR text carries the exact signal"
        );

        // Now the lexical reads fail (a surviving busy). search_notes must Err —
        // NOT return a degraded result that silently dropped the OCR `exact`.
        let flaky = FlakyDerived {
            inner: derived,
            fail_lexical: true,
        };
        let err = search_notes(
            &core,
            None,
            Some(&flaky),
            None,
            &[query("chlorophyll spectrum")],
            &[],
            &args(10),
        );
        assert!(
            err.is_err(),
            "a derived-read failure must surface, never silently field-fall-back \
             (the field scan can't serve OCR text)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn derived_unavailable_none_still_field_falls_back() {
        // The other side of #644: `Ok(None)` (a sub-trigram query, or no store)
        // is NOT an error — the `find_notes` field-text fallback is correct for
        // FIELD text and must still run. A 2-char query (< MIN_TRIGRAM) makes
        // the store return None; the literal must still hit via the field scan.
        let (dir, core) = temp_collection();
        let ab = add_note(&core, "ab cd ef", "back"); // contains the 2-char literal "ab"
        let derived = derived_for(&core, &dir);
        // The test relies on a 2-char query being sub-trigram (compile-time pin).
        const { assert!(MIN_TRIGRAM > 2) };
        let groups = search_notes(
            &core,
            None,
            Some(&derived),
            None,
            &[query("ab")],
            &[],
            &args(10),
        )
        .unwrap();
        assert!(
            groups[0].matches.iter().any(|m| m.note.id == ab),
            "a sub-trigram literal still hits via the field-text fallback"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
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
        let notes: Vec<shrike_schemas::NoteInput> = serde_json::from_value(req).unwrap();
        let out = serde_json::to_value(
            core.upsert_notes(&notes, shrike_collection::DuplicatePolicy::Allow, false)
                .unwrap(),
        )
        .unwrap();
        let b = out[0]["id"].as_i64().unwrap();
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
        // An explicit-null content (a meta-mode note dict under plain serde,
        // #391 to_wire retirement) is treated exactly like absent.
        assert!(substring_info(Some(&serde_json::Value::Null), "x").is_null());
    }

    // ── Cross-space fusion + the relative activation gate (#234) ─────────────
    //
    // These productionize the #231 eval's load-bearing findings against
    // `search_notes` DIRECTLY (no host wiring needed): the gate PRESERVES
    // text-target ranking, the negative control (gate OFF) REGRESSES it, and a
    // gated vision space DELIVERS image recall. The two spaces are a primary
    // text engine (planted vectors) + a secondary space's pre-searched
    // `SpaceSemantic` rows — exactly the shape `build_cross_space` produces.

    /// One secondary space's `SpaceSemantic` carrying image-modality hits for a
    /// single source, with the best query cosine the relative gate reads. No
    /// intra-modal floor (`image_floor: None`) — `vision_space_floored` adds one.
    fn vision_space(key: &str, image_keys: &[i64], image_dists: &[f32]) -> SpaceSemantic {
        vision_space_floored(key, image_keys, image_dists, None)
    }

    /// `vision_space` with an explicit per-space intra-modal image floor (#576).
    fn vision_space_floored(
        key: &str,
        image_keys: &[i64],
        image_dists: &[f32],
        image_floor: Option<f64>,
    ) -> SpaceSemantic {
        let best = image_dists
            .iter()
            .copied()
            .fold(None, |acc: Option<f64>, d| {
                let c = 1.0 - f64::from(d);
                Some(acc.map_or(c, |a| a.max(c)))
            });
        let mut hits = ModalityHits::new();
        hits.insert(
            "image".to_owned(),
            (image_keys.to_vec(), image_dists.to_vec()),
        );
        SpaceSemantic {
            space_key: key.to_owned(),
            per_source: vec![SpaceSourceHits {
                modality_hits: hits,
                best_query_cosine: best,
            }],
            image_floor,
        }
    }

    /// args with semantic on, a 0.0 threshold (so the planted text vector always
    /// ranks), and the canonical (empty → kernel-default) weights so the
    /// cross-space `image#<key>` signal gets weight 1.0 like the eval.
    fn cross_args(top_k: usize) -> SearchArgs {
        SearchArgs {
            top_k,
            threshold: 0.0,
            semantic: true,
            index_size: 8,
            weights: std::collections::BTreeMap::new(),
            ..Default::default()
        }
    }

    #[test]
    fn cross_space_gated_preserves_text_target() {
        // (a) Documents the now-EVAL-ONLY relative gate (retired from production
        // by #580): under `Relative`, an OFF-TOPIC vision space (its best image
        // cosine BELOW the primary text space's best) does NOT fire — the
        // text-target note stays rank-1. Kept as a kernel-level record of the
        // gate's behaviour now that floor-admission is the default. (Under the
        // production `FloorAdmit` the off-topic space here would be dropped by
        // the floor instead; this test exercises the relative path explicitly.)
        let (dir, core) = temp_collection();
        let text_target = add_note(&core, "the krebs cycle oxidizes acetyl coa", "biology");
        let image_note = add_note(&core, "unrelated filler card", "misc");

        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        // The query matches the text-target strongly (cos 0.9) in the primary.
        index
            .add("text", &[text_target], &[at_distance(0.1)])
            .unwrap();

        // BASELINE: text-only (no cross-space) — the text note is rank-1.
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle energy metabolism")],
            &[vec![1.0, 0.0]],
            &cross_args(10),
        )
        .unwrap();
        assert_eq!(groups[0].matches[0].note.id, text_target, "baseline rank-1");

        // GATED cross-space (eval `Relative`): the vision space's best image
        // cosine (0.30) is far BELOW the primary text space's best (0.90) → the
        // relative gate keeps it OUT. Text-target ranking is byte-for-byte the
        // baseline.
        let mut a = cross_args(10);
        a.cross_space_fusion_mode = CrossSpaceFusionMode::Relative;
        a.cross_space = vec![vision_space("clip", &[image_note], &[0.70])]; // cos 0.30
        let gated = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle energy metabolism")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        assert_eq!(
            gated[0].matches[0].note.id, text_target,
            "the gate preserved the text-target rank-1 (vision < text → closed)"
        );
        // The image note never entered (the gate dropped the whole vision space).
        assert!(
            !gated[0].matches.iter().any(|m| m.note.id == image_note),
            "off-topic image note stayed out"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cross_space_ungated_regresses_text_negative_control() {
        // (b) THE NEGATIVE CONTROL — the eval's load-bearing finding (text R@1
        // collapse, the 0.08-vs-1.00 separation). It documents WHY the relative
        // gate existed: N=2 off-topic vision spaces flood a text query. #580
        // RETIRED the gate because >1 image space is a config error
        // (`profiles.resolve_profile`), so this N=2 shape is IMPOSSIBLE in
        // production — this test is a kernel-level record of the now-eval-only
        // relative gate (selected explicitly via `Relative`), not a production
        // path. The query is ON-TOPIC for text; the off-topic image's vision
        // cosine is BELOW the text-target's, so the relative gate keeps it OUT;
        // ungated (or under floor-admit with no floor) it floods.
        let (dir, core) = temp_collection();
        let text_target = add_note(&core, "the krebs cycle oxidizes acetyl coa", "biology");
        let off_topic_image = add_note(
            &core,
            "an unrelated diagram whose content is in its image",
            "img",
        );

        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        // Primary: the text-target matches STRONGLY (cos 0.8) — the correct
        // rank-1 for this text query.
        index
            .add("text", &[text_target], &[at_distance(0.2)])
            .unwrap();

        // Two always-on vision spaces, each surfacing the OFF-TOPIC image note
        // at a vision cosine (0.7) BELOW the text-target's (0.8) — so the
        // relative gate would close BOTH. Ungated, the image note accumulates
        // across two `image#…` signals (2× RRF mass) and out-fuses the
        // text-target's single `text` signal.
        let secondaries = vec![
            vision_space("clip-a", &[off_topic_image], &[0.30]), // cos 0.70 < 0.80
            vision_space("clip-b", &[off_topic_image], &[0.30]),
        ];
        let mut ungated_args = cross_args(10);
        ungated_args.disable_cross_space_gate = true; // the control: gate OFF
        ungated_args.cross_space = secondaries.clone();
        let ungated = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle energy metabolism")],
            &[vec![1.0, 0.0]],
            &ungated_args,
        )
        .unwrap();
        // The regression: the off-topic image note tops the text-target (text
        // R@1 collapses) — the always-on image signal flooded the fusion.
        assert_eq!(
            ungated[0].matches[0].note.id, off_topic_image,
            "ungated: the always-on vision spaces flood and demote the text-target"
        );
        assert!(
            ungated[0]
                .matches
                .iter()
                .position(|m| m.note.id == text_target)
                .map(|r| r > 0)
                .unwrap_or(true),
            "ungated: the text-target is no longer rank-1 (the regression)"
        );

        // THE SAME inputs WITH the relative gate ON (eval `Relative`): both
        // vision spaces close (vision 0.70 < text 0.80), the off-topic image
        // note is excluded, and the text-target is restored to rank-1. This is
        // the load-bearing contrast: the gate IS what prevented (b) — the reason
        // it was safe to retire is that the N=2 input is now a config error.
        let mut gated_args = ungated_args.clone();
        gated_args.disable_cross_space_gate = false;
        gated_args.cross_space_fusion_mode = CrossSpaceFusionMode::Relative;
        let gated = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle energy metabolism")],
            &[vec![1.0, 0.0]],
            &gated_args,
        )
        .unwrap();
        assert_eq!(
            gated[0].matches[0].note.id, text_target,
            "gated: the text-target is rank-1 again (the gate closed the off-topic vision spaces)"
        );
        assert!(
            !gated[0]
                .matches
                .iter()
                .any(|m| m.note.id == off_topic_image),
            "gated: the off-topic image note is kept out (vision < text → closed)"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cross_space_gate_open_delivers_image_recall() {
        // (c) The payoff: a text query whose answer lives in a card's IMAGE
        // surfaces it through the gated vision space — the vision space's best
        // image cosine (0.92) CLEARS the primary text space's best (0.55), so
        // the relative gate OPENS and the image-bearing note joins the fusion
        // via its `image#clip` signal. (Without cross-space, the text-only
        // primary never sees it.)
        let (dir, core) = temp_collection();
        let weak_text = add_note(&core, "a loosely related text card", "text");
        let image_note = add_note(&core, "card whose answer is only in the picture", "img");

        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        // Primary text: only a WEAK match (cos 0.55), and the image note has no
        // text vector at all → text-only would surface only weak_text.
        index
            .add("text", &[weak_text], &[at_distance(0.45)])
            .unwrap();

        // Without cross-space: the image note is absent.
        let baseline = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("the content shown in the diagram")],
            &[vec![1.0, 0.0]],
            &cross_args(10),
        )
        .unwrap();
        assert!(
            !baseline[0].matches.iter().any(|m| m.note.id == image_note),
            "text-only never surfaces the image-only note"
        );

        // With the gated vision space (cos 0.92 ≥ 0.55 → gate OPEN): the image
        // note surfaces, tagged with the per-space provenance signal.
        let mut a = cross_args(10);
        a.cross_space = vec![vision_space("clip", &[image_note], &[0.08])]; // cos 0.92
        let crossed = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("the content shown in the diagram")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        let img = crossed[0]
            .matches
            .iter()
            .find(|m| m.note.id == image_note)
            .expect("the image-bearing note is delivered via the gated vision space");
        // Per-space provenance (#182): the surfacing signal is `image#clip`.
        assert!(
            img.provenance.iter().any(|p| p.signal == "image#clip"),
            "the match carries its vision space's per-space provenance"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    // ── The #576 cross-space intra-modal floor (the over-return leak) ────────
    //
    // These pin the variant mechanics against `search_notes` DIRECTLY. The
    // failure they guard: when the PRIMARY text space's best cosine → 0 (an
    // ∅-gold query nothing answers), the relative gate `v >= p` is trivially
    // satisfied by ANY positive CLIP cosine, so a weak image card (cos ~0.2)
    // leaks in at full RRF weight. The intra-modal floor (V0+floor / V2) is the
    // backstop the relative gate is structurally blind to.

    /// An ∅-gold scenario: the primary text space has a vector that the query
    /// matches only WEAKLY (cos ≈ 0), and one secondary vision space surfaces a
    /// weak off-topic image (cos ≈ 0.2, well below its own floor). Returns the
    /// (dir, core, index, weak_text, image_note) so each variant runs the same
    /// inputs. The query embeds to `[1,0]`; the primary text vector sits at the
    /// given cosine to it.
    fn empty_primary_scenario(
        primary_text_cos: f32,
    ) -> (
        std::path::PathBuf,
        shrike_collection::CollectionCore,
        MultiModalIndex,
        i64,
        i64,
    ) {
        let (dir, core) = temp_collection();
        let weak_text = add_note(&core, "a totally unrelated text card", "text");
        let image_note = add_note(&core, "card whose picture is off-topic", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[weak_text], &[at_distance(1.0 - primary_text_cos)])
            .unwrap();
        (dir, core, index, weak_text, image_note)
    }

    #[test]
    fn over_return_v0_leaks_weak_image_on_empty_primary() {
        // V0 (today): the primary's best cosine ≈ 0, so the relative gate
        // `v(0.2) >= p(0.0)` is trivially OPEN and the weak off-topic image
        // leaks into the results. This is the leak #576 closes — pinned here so
        // the floor variants below have a positive baseline to beat.
        let (dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
        let mut a = cross_args(10);
        // The space carries its own floor (0.5), but V0 never consults it.
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.80],
            Some(0.5),
        )]; // cos 0.20
        a.cross_space_fusion_mode = CrossSpaceFusionMode::Relative;
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("purple elephant playing chess")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        assert!(
            groups[0].matches.iter().any(|m| m.note.id == image_note),
            "V0 leaks the weak off-topic image (the relative gate inverts under an empty primary)"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn over_return_v0_floor_closes_the_leak() {
        // V0+floor: the space's best surviving image cosine (0.20) is BELOW its
        // OWN intra-modal floor (0.5) → the floor hard-drops the space's image
        // ranking. The weak off-topic image never enters. The relative gate is
        // still satisfied (empty primary) — the floor is the backstop.
        let (dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
        let mut a = cross_args(10);
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.80],
            Some(0.5),
        )]; // cos 0.20
        a.cross_space_fusion_mode = CrossSpaceFusionMode::RelativeFloor;
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("purple elephant playing chess")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        assert!(
            !groups[0].matches.iter().any(|m| m.note.id == image_note),
            "V0+floor closes the leak: cos 0.20 ≤ floor 0.50 → the vision space is dropped"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn over_return_v0_floor_keeps_an_above_floor_image() {
        // The floor must NOT over-suppress: a vision space whose best image
        // cosine (0.92) CLEARS its own floor (0.5) still contributes, even
        // under an empty primary — the floor drops only the genuinely-weak best.
        let (dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
        let mut a = cross_args(10);
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.08],
            Some(0.5),
        )]; // cos 0.92
        a.cross_space_fusion_mode = CrossSpaceFusionMode::RelativeFloor;
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("the content shown in the diagram")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        assert!(
            groups[0].matches.iter().any(|m| m.note.id == image_note),
            "V0+floor keeps an above-floor image (0.92 > 0.50): no over-suppression"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn over_return_v2_soft_calibrated_tapers_the_weak_image() {
        // V2 (soft-calibrated): the weak image (cos 0.20, floor 0.5) gets a
        // near-zero weight `σ((0.20-0.50)/τ)` at a small τ, so even if it stays
        // in the ranking its RRF mass is negligible — it does not out-rank the
        // primary's own (weak) text hit. With a tiny τ this approaches the hard
        // floor; here we assert the weak image is demoted below the text card.
        let (dir, core, index, weak_text, image_note) = empty_primary_scenario(0.30);
        let mut a = cross_args(10);
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.80],
            Some(0.5),
        )]; // cos 0.20
        a.cross_space_fusion_mode = CrossSpaceFusionMode::SoftCalibrated;
        a.cross_space_tau = 0.05;
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        let text_rank = groups[0]
            .matches
            .iter()
            .position(|m| m.note.id == weak_text);
        let image_rank = groups[0]
            .matches
            .iter()
            .position(|m| m.note.id == image_note);
        assert!(
            matches!((text_rank, image_rank), (Some(t), Some(i)) if t < i) || image_rank.is_none(),
            "V2 tapers the weak image's weight to near-zero → it ranks below the text card"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn over_return_v1_soft_relative_still_leaks() {
        // V1 (soft-relative, calibration-free): the weight is
        // `σ((clip_best − text_best)/τ)`. On an empty primary `text_best ≈ 0`
        // and `clip_best = 0.20`, so the argument is POSITIVE → weight ≈ 1.
        // V1 is the CONTROL: it does NOT consult the intra-modal floor, so it
        // STILL leaks the weak image. This proves the leak is intra-modal, not
        // relative.
        let (dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
        let mut a = cross_args(10);
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.80],
            Some(0.5),
        )]; // cos 0.20
        a.cross_space_fusion_mode = CrossSpaceFusionMode::SoftRelative;
        a.cross_space_tau = 0.05;
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("purple elephant playing chess")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        assert!(
            groups[0].matches.iter().any(|m| m.note.id == image_note),
            "V1 (soft-relative) still leaks: it never consults the intra-modal floor"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn floor_holds_the_negative_control_at_n2() {
        // The negative control MUST hold under V0+floor: a text-target query
        // with N=2 off-topic-but-intra-modally-confident vision spaces. Each
        // space's image best (0.70) CLEARS its own floor (0.5) — so the floor
        // does NOT drop them — but the RELATIVE gate closes them (vision 0.70 <
        // text 0.80). The composition (floor AND relative) keeps the text-target
        // at rank-1: the floor never re-opens what the relative gate closed.
        let (dir, core) = temp_collection();
        let text_target = add_note(&core, "the krebs cycle oxidizes acetyl coa", "biology");
        let off_topic_image = add_note(&core, "an off-topic but confident diagram", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[text_target], &[at_distance(0.2)])
            .unwrap(); // cos 0.80

        let mut a = cross_args(10);
        a.cross_space_fusion_mode = CrossSpaceFusionMode::RelativeFloor;
        // Two intra-modally-CONFIDENT spaces (image best 0.70 > floor 0.50) but
        // OFF-TOPIC relative to the text answer (0.70 < text 0.80).
        a.cross_space = vec![
            vision_space_floored("clip-a", &[off_topic_image], &[0.30], Some(0.5)), // cos 0.70
            vision_space_floored("clip-b", &[off_topic_image], &[0.30], Some(0.5)),
        ];
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle energy metabolism")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        assert_eq!(
            groups[0].matches[0].note.id, text_target,
            "the negative control holds under V0+floor: the relative gate closes both off-topic spaces"
        );
        assert!(
            !groups[0]
                .matches
                .iter()
                .any(|m| m.note.id == off_topic_image),
            "the off-topic image stays out (relative gate closed, floor did not re-open it)"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    // ── #580 floor-based admission (drop the relative gate) ──────────────────
    //
    // These pin the floor-admission mechanics against `search_notes` DIRECTLY.
    // The win they prove: a strong on-topic CLIP hit reaches RRF even when the
    // text space "won" (the relative gate would close it) — corroboration, not
    // winner-take-all. The guard they prove: the absolute floor still rejects a
    // genuinely-weak (spurious-filename) image. The multiplicity tension they
    // measure: FloorAdmit alone floods the N≥2 negative control (the rationale
    // for the "no two image spaces" config assertion); the budget holds it.

    #[test]
    fn floor_admit_corroborates_when_text_wins() {
        // THE #580 WIN — the filename-collision case at the unit level. A text
        // query whose answer is in a card's image, where the card's FILENAME
        // also lexically wins the primary text space (so the relative gate would
        // shut CLIP out). Here the primary text matches the image card STRONGLY
        // (cos 0.95 — the "filename won" proxy) while the vision space's image
        // best (0.85) is BELOW it → the relative gate CLOSES. But 0.85 clears the
        // floor (0.50), so floor-admission ADMITS the CLIP hit: the card carries
        // its `image#clip` provenance (the corroborating vote the relative gate
        // discarded).
        let (dir, core) = temp_collection();
        let image_card = add_note(&core, "card whose answer is in heart.png", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        // Primary text wins on the filename token (cos 0.95).
        index
            .add("text", &[image_card], &[at_distance(0.05)])
            .unwrap();

        // BASELINE — RelativeFloor (production): vision 0.85 < text 0.95 → the
        // relative gate CLOSES, so no `image#clip` reaches the card.
        let mut rel = cross_args(10);
        rel.cross_space_fusion_mode = CrossSpaceFusionMode::RelativeFloor;
        rel.cross_space = vec![vision_space_floored(
            "clip",
            &[image_card],
            &[0.15],
            Some(0.5),
        )]; // cos 0.85
        let baseline = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("a labelled diagram of the human heart")],
            &[vec![1.0, 0.0]],
            &rel,
        )
        .unwrap();
        let rel_match = baseline[0]
            .matches
            .iter()
            .find(|m| m.note.id == image_card)
            .expect("the card still surfaces via its text/filename hit");
        assert!(
            !rel_match.provenance.iter().any(|p| p.signal == "image#clip"),
            "RelativeFloor BASELINE: the relative gate (vision 0.85 < text 0.95) discards the CLIP vote"
        );

        // FLOOR-ADMIT: same inputs, the relative gate dropped. 0.85 > floor 0.50
        // → CLIP is admitted; the card now carries `image#clip` provenance.
        let mut fa = rel.clone();
        fa.cross_space_fusion_mode = CrossSpaceFusionMode::FloorAdmit;
        let admitted = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("a labelled diagram of the human heart")],
            &[vec![1.0, 0.0]],
            &fa,
        )
        .unwrap();
        let fa_match = admitted[0]
            .matches
            .iter()
            .find(|m| m.note.id == image_card)
            .expect("the card is present");
        assert!(
            fa_match.provenance.iter().any(|p| p.signal == "image#clip"),
            "FloorAdmit: the on-topic CLIP hit corroborates the card even though text won"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn floor_admit_rejects_spurious_filename_image() {
        // THE #580 PRECISION GUARD — the homonym/lying-filename case at the unit
        // level. A card's filename lexically wins the text space, but its IMAGE
        // is OFF-TOPIC for the query, so the vision space's image best (0.30)
        // falls BELOW the floor (0.50). Floor-admission must REJECT the CLIP hit:
        // the card may still surface via its filename text hit, but it carries NO
        // `image#clip` (the floor is the SOLE discriminator now that the relative
        // gate is gone — this is the load-bearing test for the thesis).
        let (dir, core) = temp_collection();
        let lying_card = add_note(&core, "card with jaguar.png but the image is a car", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[lying_card], &[at_distance(0.05)])
            .unwrap(); // text wins on "jaguar"

        let mut fa = cross_args(10);
        fa.cross_space_fusion_mode = CrossSpaceFusionMode::FloorAdmit;
        // The image is off-topic for "the spotted big cat" → cos 0.30 < floor.
        fa.cross_space = vec![vision_space_floored(
            "clip",
            &[lying_card],
            &[0.70],
            Some(0.5),
        )];
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("a jaguar, the spotted big cat")],
            &[vec![1.0, 0.0]],
            &fa,
        )
        .unwrap();
        let m = groups[0]
            .matches
            .iter()
            .find(|m| m.note.id == lying_card)
            .expect("the card surfaces via its filename text hit");
        assert!(
            !m.provenance.iter().any(|p| p.signal == "image#clip"),
            "FloorAdmit: the floor REJECTS the off-topic image (cos 0.30 ≤ floor 0.50) — \
             the lying filename does not summon a spurious CLIP vote"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn floor_admit_keeps_the_over_return_leak_closed() {
        // The #576 over-return leak must STAY closed under floor-admission: an
        // ∅-gold query, empty primary, one weak off-topic image (cos 0.20 < floor
        // 0.50). With no relative gate the floor is the only guard — and it holds
        // (0.20 ≤ 0.50 → dropped). Same outcome as RelativeFloor, different path.
        let (dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
        let mut a = cross_args(10);
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.80],
            Some(0.5),
        )]; // cos 0.20
        a.cross_space_fusion_mode = CrossSpaceFusionMode::FloorAdmit;
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("purple elephant playing chess")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        assert!(
            !groups[0].matches.iter().any(|m| m.note.id == image_note),
            "FloorAdmit closes the over-return leak: cos 0.20 ≤ floor 0.50 → dropped"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn floor_admit_alone_floods_n2_but_budget_holds_it() {
        // THE MEASURED RATIONALE for the "no two image spaces" config assertion
        // (#580). The N=2 negative control's protection came ENTIRELY from the
        // relative gate (see `floor_holds_the_negative_control_at_n2`): both
        // off-topic spaces clear their floor, so dropping the relative gate lets
        // them flood. FloorAdmit (no budget) REGRESSES text R@1; FloorAdmitBudget
        // (B=1.0 split 0.5/0.5) restores it. In production this scenario is
        // IMPOSSIBLE (only one image space exists), so the budget is moot there —
        // this test documents WHY the relative gate can be dropped: not because
        // floor-admission handles N≥2, but because N≥2 image spaces never occur.
        let (dir, core) = temp_collection();
        let text_target = add_note(&core, "the krebs cycle oxidizes acetyl coa", "biology");
        let off_topic_image = add_note(&core, "an off-topic but confident diagram", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[text_target], &[at_distance(0.2)])
            .unwrap(); // text cos 0.80
                       // Two intra-modally-confident but off-topic spaces (image best 0.70 >
                       // floor 0.50) — exactly the shape the relative gate used to close.
        let spaces = vec![
            vision_space_floored("clip-a", &[off_topic_image], &[0.30], Some(0.5)), // cos 0.70
            vision_space_floored("clip-b", &[off_topic_image], &[0.30], Some(0.5)),
        ];

        // FloorAdmit (no budget): both spaces clear the floor and flood — the
        // off-topic image accumulates 2× RRF mass and out-fuses the text target.
        let mut flood = cross_args(10);
        flood.cross_space_fusion_mode = CrossSpaceFusionMode::FloorAdmit;
        flood.cross_space = spaces.clone();
        let flooded = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle energy metabolism")],
            &[vec![1.0, 0.0]],
            &flood,
        )
        .unwrap();
        assert_eq!(
            flooded[0].matches[0].note.id, off_topic_image,
            "FloorAdmit (no budget) floods at N=2: the off-topic image out-fuses the text target"
        );

        // FloorAdmitBudget (B=1.0): the two spaces share the budget (0.5 each),
        // so their combined RRF mass cannot out-fuse the full-weight text signal
        // → the text target is restored to rank-1.
        let mut budgeted = flood.clone();
        budgeted.cross_space_fusion_mode = CrossSpaceFusionMode::FloorAdmitBudget;
        budgeted.cross_space_budget = 1.0;
        let held = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("krebs cycle energy metabolism")],
            &[vec![1.0, 0.0]],
            &budgeted,
        )
        .unwrap();
        assert_eq!(
            held[0].matches[0].note.id, text_target,
            "FloorAdmitBudget holds the N=2 negative control: the split budget can't out-fuse text"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn floor_admit_n1_unaffected_by_budget() {
        // The production-common case: a SINGLE image space. The budget must not
        // penalize it — N=1 keeps full weight under FloorAdmitBudget (the budget
        // only divides when N≥2). The on-topic CLIP hit corroborates at full
        // strength exactly like FloorAdmit.
        let (dir, core) = temp_collection();
        let weak_text = add_note(&core, "a loosely related text card", "text");
        let image_note = add_note(&core, "card whose answer is only in the picture", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[weak_text], &[at_distance(0.45)])
            .unwrap();

        let mut a = cross_args(10);
        a.cross_space_fusion_mode = CrossSpaceFusionMode::FloorAdmitBudget;
        a.cross_space_budget = 1.0;
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.08],
            Some(0.5),
        )]; // cos 0.92
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("the content shown in the diagram")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        let m = groups[0]
            .matches
            .iter()
            .find(|m| m.note.id == image_note)
            .expect("the single image space surfaces the image-only card");
        assert!(
            m.provenance.iter().any(|p| p.signal == "image#clip"),
            "N=1 FloorAdmitBudget: the lone image space keeps full weight (no budget penalty)"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn soft_floor_admit_tapers_near_floor() {
        // SoftFloorAdmit: a borderline-above-floor image (cos 0.55, floor 0.50)
        // gets a tapered weight `σ((0.55-0.50)/τ)` < 1, while a confident one
        // (cos 0.92) gets ≈1. Here we assert the confident hit is admitted with
        // its provenance (the graceful form still corroborates), and a clearly
        // sub-floor hit (cos 0.20) gets a near-zero weight so it cannot out-rank
        // a real text card — the soft analogue of the hard floor's drop.
        let (dir, core) = temp_collection();
        let weak_text = add_note(&core, "a loosely related text card", "text");
        let image_note = add_note(&core, "card whose answer is only in the picture", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[weak_text], &[at_distance(0.45)])
            .unwrap();

        let mut a = cross_args(10);
        a.cross_space_fusion_mode = CrossSpaceFusionMode::SoftFloorAdmit;
        a.cross_space_tau = 0.05;
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[image_note],
            &[0.08],
            Some(0.5),
        )]; // cos 0.92
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("the content shown in the diagram")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        let m = groups[0]
            .matches
            .iter()
            .find(|m| m.note.id == image_note)
            .expect("SoftFloorAdmit admits the confident image hit");
        assert!(
            m.provenance.iter().any(|p| p.signal == "image#clip"),
            "SoftFloorAdmit: a confident hit (cos 0.92 ≫ floor) corroborates at ≈full weight"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    // ── #582 per-note image floor (drop the below-floor tail) ────────────────

    #[test]
    fn floor_admit_secondary_drops_below_floor_tail() {
        // #582: the per-NOTE floor on the SECONDARY cross-space image ranking. A
        // vision space surfaces TWO image cards: one ABOVE the floor (cos 0.92)
        // and one BELOW it (cos 0.20, floor 0.50). Floor-admission's per-note
        // filter keeps the above-floor card's `image#clip` and DROPS the
        // below-floor one — the latter no longer rides in on the rank-1's
        // coat-tails (the pre-#582 per-space gate admitted the whole ranking).
        let (dir, core) = temp_collection();
        let weak_text = add_note(&core, "a loosely related text card", "text");
        let above = add_note(&core, "card whose image strongly matches", "img");
        let below = add_note(&core, "card whose image barely matches", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[weak_text], &[at_distance(0.45)])
            .unwrap();

        let mut a = cross_args(10);
        a.cross_space_fusion_mode = CrossSpaceFusionMode::FloorAdmit;
        // One space, two image hits: above (cos 0.92) + below (cos 0.20), floor 0.5.
        a.cross_space = vec![vision_space_floored(
            "clip",
            &[above, below],
            &[0.08, 0.80],
            Some(0.5),
        )];
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("the content shown in the diagram")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        let above_m = groups[0].matches.iter().find(|m| m.note.id == above);
        let below_m = groups[0].matches.iter().find(|m| m.note.id == below);
        assert!(
            above_m.is_some_and(|m| m.provenance.iter().any(|p| p.signal == "image#clip")),
            "the above-floor card keeps its image#clip"
        );
        // The below-floor card carries NO image#clip (it may be absent entirely
        // if it has no other signal).
        assert!(
            below_m.is_none_or(|m| !m.provenance.iter().any(|p| p.signal == "image#clip")),
            "#582: the below-floor tail card carries no image#clip (per-note floor dropped it)"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn primary_image_floor_is_per_note() {
        // #582: the per-NOTE floor on the PRIMARY image ranking (the #201b core,
        // used by omni/single-space deployments). The primary image modality
        // returns two cards: one above the floor (cos 0.90) and one below (cos
        // 0.20, floor 0.50). The above-floor card surfaces via `image`; the
        // below-floor one does NOT (was: the whole ranking admitted iff the
        // rank-1 cleared the floor).
        let (dir, core) = temp_collection();
        let above = add_note(&core, "card whose image strongly matches", "imgA");
        let below = add_note(&core, "card whose image barely matches", "imgB");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add(
                "image",
                &[above, below],
                &[at_distance(0.10), at_distance(0.80)],
            )
            .unwrap(); // cos 0.90 (above) + cos 0.20 (below)

        let mut a = cross_args(10);
        a.image_floor = Some(0.5);
        let groups = search_notes(
            &core,
            Some(&index),
            None,
            None,
            &[query("the diagram content")],
            &[vec![1.0, 0.0]],
            &a,
        )
        .unwrap();
        let above_m = groups[0].matches.iter().find(|m| m.note.id == above);
        let below_m = groups[0].matches.iter().find(|m| m.note.id == below);
        assert!(
            above_m.is_some_and(|m| m.provenance.iter().any(|p| p.signal == "image")),
            "the above-floor card surfaces via the primary image signal"
        );
        assert!(
            below_m.is_none_or(|m| !m.provenance.iter().any(|p| p.signal == "image")),
            "#582: the below-floor card carries no primary image signal (per-note floor)"
        );
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }
}

#[cfg(test)]
mod neighbor_tests {
    use super::*;
    use crate::actions::tests::{add_note, temp_collection};
    use shrike_derived::DerivedEngine;
    use shrike_index::MultiModalIndex;

    fn derived_for(core: &dyn Collection, dir: &std::path::Path) -> DerivedEngine {
        let e = DerivedEngine::open(dir.join("shrike.db").to_str().unwrap(), 1).unwrap();
        let ids = core.find_notes("deck:*").unwrap();
        let rows = core.derived_field_rows(&ids).unwrap();
        e.build(&rows, 1).unwrap();
        e
    }

    /// Unit vector at exact cosine `1 - d` against the `[1, 0]` query.
    fn at_distance(d: f32) -> Vec<f32> {
        let sim = 1.0 - d;
        vec![sim, (1.0 - sim * sim).max(0.0).sqrt()]
    }

    #[test]
    fn semantic_and_lexical_signals_merge_with_provenance() {
        let (dir, core) = temp_collection();
        // A paraphrase dupe (semantic), a near-verbatim restatement (lexical),
        // and an unrelated note sharing a question stem (proposed, must be
        // verified OUT).
        let paraphrase = add_note(&core, "mitochondria are the cell's power plants", "energy");
        let verbatim = add_note(&core, "what is the powerhouse of the cell", "atp");
        let stem_only = add_note(&core, "what is the capital of france", "paris");
        let derived = derived_for(&core, &dir);

        let index = MultiModalIndex::new(vec!["text".to_owned()]).unwrap();
        index
            .add("text", &[paraphrase], &[at_distance(0.2)])
            .unwrap();
        index
            .add("text", &[verbatim], &[at_distance(0.9)]) // below the gate
            .unwrap();
        index
            .add("text", &[stem_only], &[at_distance(0.95)])
            .unwrap();

        let draft = "what is the powerhouse of the cell?".to_string();
        let out = attach_neighbors(
            &core,
            Some(&index),
            Some(&derived),
            &[draft],
            &[vec![1.0, 0.0]],
            &[],
            5,
            0.6,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        let entry = &out[0];

        // The paraphrase arrives semantically (cosine 0.8 ≥ 0.6) with text
        // provenance; the verbatim restatement arrives lexically (score None,
        // fuzzy provenance) despite its weak cosine; the shared stem is
        // proposed by trigrams but verified out.
        let ids: Vec<i64> = entry.neighbors.iter().map(|n| n.id).collect();
        assert!(ids.contains(&paraphrase), "semantic dupe attached: {ids:?}");
        assert!(ids.contains(&verbatim), "lexical dupe attached: {ids:?}");
        assert!(
            !ids.contains(&stem_only),
            "stem overlap verified out: {ids:?}"
        );

        let sem = entry.neighbors.iter().find(|n| n.id == paraphrase).unwrap();
        assert_eq!(sem.score, Some(0.8));
        assert_eq!(sem.provenance[0].signal, "text");
        let lex = entry.neighbors.iter().find(|n| n.id == verbatim).unwrap();
        assert_eq!(lex.score, None);
        assert_eq!(lex.provenance[0].signal, "fuzzy");

        // Scored candidates order before lexical-only ones.
        assert!(
            ids.iter().position(|&i| i == paraphrase) < ids.iter().position(|&i| i == verbatim)
        );
        // The calibration sample is the best semantic cosine.
        assert_eq!(entry.best, Some(0.8));
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn exclusion_and_no_match_sample() {
        let (dir, core) = temp_collection();
        let just_written = add_note(&core, "exactly this draft text", "b");
        let derived = derived_for(&core, &dir);
        let index = MultiModalIndex::new(vec!["text".to_owned()]).unwrap();
        index
            .add("text", &[just_written], &[at_distance(0.05)])
            .unwrap();

        // The just-written note is excluded from BOTH signals; with nothing
        // else in range the draft records a no-match tick (best = None).
        let out = attach_neighbors(
            &core,
            Some(&index),
            Some(&derived),
            &["exactly this draft text".to_string()],
            &[vec![1.0, 0.0]],
            &[just_written],
            5,
            0.6,
        )
        .unwrap();
        assert!(out[0].neighbors.is_empty());
        assert_eq!(out[0].best, None);

        // A second draft sees it both ways: semantic + fuzzy provenance on
        // ONE candidate (the merge path).
        let out = attach_neighbors(
            &core,
            Some(&index),
            Some(&derived),
            &["exactly this draft text".to_string()],
            &[vec![1.0, 0.0]],
            &[],
            5,
            0.6,
        )
        .unwrap();
        let n = &out[0].neighbors[0];
        assert_eq!(n.id, just_written);
        assert_eq!(n.score, Some(0.95));
        let signals: Vec<&str> = n.provenance.iter().map(|c| c.signal.as_str()).collect();
        assert_eq!(signals, vec!["text", "fuzzy"]);
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }
}
