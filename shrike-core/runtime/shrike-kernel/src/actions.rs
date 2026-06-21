//! The action core: the read surface.
//!
//! Each action is the *whole* tool body: parameter normalization, the
//! collection-core call, and the Rust-canonical response type — typed
//! end-to-end (the read surface returns `shrike-schemas` types straight from
//! the core; serialization happens once, at the host edge). Python's
//! `actions.py` is a binding per action: typed signature (FastMCP's
//! inputSchema source) + context assembly + the completion-log fragment.
//!
//! Actions are synchronous over `&dyn Collection`: the harness invokes them on
//! its collection worker thread through the shrike-pyo3 per-action bindings (the
//! same serialization every collection op rides); the kernel's async layer
//! drives the same bodies through [`crate::SerializedCollection`]. No
//! threading, no runtime assumption here.

use serde::de::DeserializeOwned;

use shrike_collection::Collection;
use shrike_error::{NativeError, NativeResult};
use shrike_schemas::{CollectionInfo, ListNotesResponse};

/// The actions this module implements (the registry seam: the Python
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
/// The read surface returns typed values directly; this stays for the modules
/// that still ride core-emitted JSON (media/write/note-type).
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
/// Typed end-to-end: the core builds the canonical type, the
/// action forwards it, and serialization happens once, at the host edge.
///
/// # Errors
///
/// Returns an error if the collection core fails to assemble the requested
/// sections.
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
    /// Restrict to these note IDs.
    pub ids: Option<Vec<i64>>,
    /// Restrict to notes in this deck.
    pub deck: Option<String>,
    /// Restrict to notes carrying all of these tags.
    pub tags: Option<Vec<String>>,
    /// Restrict to notes of this note type.
    pub note_type: Option<String>,
    /// Restrict to notes modified at or after this epoch-seconds cutoff.
    pub modified_since_epoch: Option<i64>,
    /// Whether to include field bodies in the result.
    pub with_fields: bool,
    /// Maximum notes to return.
    pub limit: usize,
}

/// `list_notes` — filter/retrieve notes (filters ANDed; at least one given,
/// enforced by the core as invalid input).
///
/// # Errors
///
/// Returns an error if no filter is given (invalid input) or the core read
/// fails.
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
/// hatch). A malformed expression is invalid input (isolation marks
/// already stripped by the core's error decoding).
///
/// # Errors
///
/// Returns an error if the search expression is malformed (invalid input) or
/// the core read fails.
pub fn collection_query(
    core: &dyn Collection,
    query: &str,
    with_fields: bool,
    limit: usize,
) -> NativeResult<ListNotesResponse> {
    core.query(query, with_fields, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrike_collection::CollectionCore;

    pub(super) fn temp_collection() -> (crate::test_support::ScratchDir, CollectionCore) {
        let dir = crate::test_support::ScratchDir::new("shrike-kernel-actions");
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

// ── search_notes: the assembly re-home ───────────────────────────────────────
// The whole fused-search body: per-modality semantic ranking over query
// vectors (embedded host-side — a handful of query vectors crossing the FFI is
// the recorded design point), substring + fuzzy lexical candidates
// from the derived store (with the find_notes fallback), RRF fusion with the
// exact-match priority tier, and annotation/provenance assembly — validated
// into the canonical SearchResultGroup. Orchestrator state (semantic
// availability, the image activation floor, the index size for the
// over-fetch clamp) is injected per call until the kernel internalizes it.

use std::collections::{BTreeMap, HashSet};

use shrike_derived::{FxI64Map, MIN_TRIGRAM};
use shrike_schemas::{
    FuzzyMatch, Note, SearchMatch, SearchResultGroup, SignalContribution, SubstringInfo,
};
use shrike_store::{DerivedStore, LexicalRow, VectorIndex};

/// A note's field map (`Note.content`): the "full" projection's fields, absent
/// in "meta" mode. The literal-substring authority reads it.
type NoteContent = BTreeMap<String, String>;

/// One source's per-modality semantic rankings (`search_by_modality`'s row).
type ModalityHits = std::collections::BTreeMap<String, (Vec<i64>, Vec<f32>)>;

/// One SECONDARY embedding space's already-embedded + already-searched semantic
/// results, fed into cross-space fusion. The PRIMARY text space's hits ride the
/// host-supplied `vectors`/`index` path; each secondary text-capable space
/// embeds the query with ITS OWN model and searches ITS OWN engine at the
/// kernel level, then hands the per-source rows here as data — so `search_notes`
/// stays the pure fusion assembly and never holds N engines.
///
/// Empty `cross_space` (the N=1 / single-space case) → the rankings vector fed
/// to `rrf_fuse` is EXACTLY the per-modality set, so the fused output is
/// unchanged from single-space search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpaceSemantic {
    /// The space's CONTENT fingerprint — surfaced in per-space provenance
    /// only when N≥2 (vacuous/absent at N=1).
    pub space_key: String,
    /// One entry per search source, in `sources` order.
    pub per_source: Vec<SpaceSourceHits>,
    /// This space's OWN intra-modal image activation floor
    /// (`mean + margin·std` of its image modality's typical best match),
    /// calibrated on its OWN index stats — NOT the primary's `image_floor`,
    /// which is calibrated on a different index. `None` when the space
    /// is uncalibrated (text-only collection, too few samples), in which case
    /// the intra-modal floor is a no-op and only the relative gate applies.
    /// The kernel fills this from the space's orchestrator at fan-out time.
    #[serde(default)]
    pub image_floor: Option<f64>,
}

/// The cross-space fusion variant. `FloorAdmit` is the PRODUCTION default: no
/// relative winner-take-all gate — a secondary image space is admitted on its
/// OWN calibrated intra-modal floor (`image_best > z_floor`), independent of how
/// the text space did, so a strong on-topic CLIP hit reaches RRF and
/// corroborates a card even when the text space "won" on a spurious
/// filename/lexical match (the corroboration win, measured on the real
/// MiniLM+CLIP corpus: `eval/search_quality/RESULTS_580.md`). The floor margin
/// is the precision/recall dial (`search.cross_space_fusion.margin`, threaded
/// into calibration).
///
/// No relative gate is sound because >1 image-embedding space is a config error
/// (`profiles.resolve_profile`): with at most ONE image space there is no
/// multiplicity to guard, which was the relative gate's sole job (the N≥2 flood
/// — `cross_space_ungated_regresses_text_negative_control` +
/// `floor_admit_alone_floods_n2_but_budget_holds_it` document the
/// impossible-by-construction behaviour at the kernel level).
///
/// The other modes are EVAL-ONLY (`SHRIKE_CROSS_SPACE_FUSION_MODE`), kept to
/// reproduce the decision tables — they NEVER select in production: the relative
/// family (`Relative`, `RelativeFloor`, `SoftRelative`, `SoftCalibrated`) is the
/// relative gate; `SoftFloorAdmit*` is the dominated soft variant (zero recall
/// upside, re-opens the over-return leak with τ); the `*Budget` modes reproduce
/// the N≥2 multiplicity measurement that justifies the single-image-space
/// invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossSpaceFusionMode {
    /// PRODUCTION — FLOOR-ADMIT (binary): the relative gate is gone; admit
    /// a vision space iff its best surviving image cosine clears its OWN
    /// calibrated floor (`image_best > z_floor`), at full weight 1.0. An
    /// uncalibrated space (no floor) is admitted (the floor is a no-op). The
    /// single-image-space invariant (a config error otherwise) means the N≥2
    /// flood the relative gate guards against cannot occur.
    #[default]
    FloorAdmit,
    /// V0+floor (eval) — relative gate AND a per-space calibrated intra-modal
    /// floor. Kept to reproduce the table.
    RelativeFloor,
    /// V0 (eval) — binary relative gate only (`clip_best >= text_best`); leaks
    /// weak image cards when the primary's best cosine → 0. Kept to measure the
    /// leak the floor closes.
    Relative,
    /// V1 (eval) — soft-relative: weight `w = σ((clip_best − text_best)/τ)`
    /// folded into the `image#<key>` RRF weight. Calibration-free CONTROL — it
    /// still leaks (proves the leak is intra-modal, not relative).
    SoftRelative,
    /// V2 (eval) — soft-calibrated: weight `w = σ((z_s − z0)/τ)`, composed with
    /// the relative gate. The soft alternative to the hard floor.
    SoftCalibrated,
    /// (eval) — FLOOR-ADMIT + WEIGHT BUDGET (binary): admit on the absolute
    /// floor, but bound the TOTAL vision RRF weight when N≥2 spaces fire by
    /// splitting a budget `B` (default 1.0, `cross_space_budget`) equally across
    /// the admitted spaces (each gets `B/N`). N=1 keeps full weight `B`. The
    /// budget held the N≥2 negative control without relative suppression — the
    /// MEASURED RATIONALE for the single-image-space invariant (moot in
    /// production, where N≥2 is a config error).
    FloorAdmitBudget,
    /// (eval) — SOFT floor-admit (NO budget): drop the relative gate; weight
    /// each admitted space `w = σ((image_best − z_floor)/τ)`. DOMINATED:
    /// no recall upside over binary, and re-opens the over-return leak as τ
    /// grows. Kept only to reproduce that finding.
    SoftFloorAdmit,
    /// (eval) — SOFT floor-admit + WEIGHT BUDGET: the soft variant with the
    /// N≥2 budget (sum-scaled to `B`). Also dominated; kept for completeness.
    SoftFloorAdmitBudget,
}

/// One secondary space's per-source search result: its per-modality hits plus
/// the raw best query→match cosine the RELATIVE activation gate reads
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
/// `None` content (a meta-mode note dict carries no fields) yields `None`, the
/// "no literal match" answer.
pub fn substring_info(content: Option<&NoteContent>, text: &str) -> Option<SubstringInfo> {
    // NFC-normalize BOTH the needle and the field content, so confirmation is
    // canonical-form-agnostic regardless of Anki's `NormalizeNoteText` config (when
    // it's off, fields stay NFD; normalizing only the needle would then reject a
    // match). The snippet slices the normalized field, so its indices stay aligned.
    let needle: Vec<char> = shrike_derived::nfc(text).to_lowercase().chars().collect();
    let mut matched: Vec<String> = Vec::new();
    let mut snippet: Option<String> = None;
    let fields = content?;
    for (name, value) in fields {
        let value = shrike_derived::nfc(value);
        let chars: Vec<char> = value.chars().collect();
        let lowered: Vec<char> = value.to_lowercase().chars().collect();
        let idx = match find_subsequence(&lowered, &needle) {
            Some(i) => i,
            None => continue,
        };
        matched.push(name.clone());
        if snippet.is_none() {
            let start = idx.saturating_sub(30);
            let end = (idx + needle.len() + 30).min(chars.len());
            // `chars` and `lowered` derive from the SAME normalized value, so a
            // length-changing lowercasing drifts both identically and indices align.
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
    if matched.is_empty() {
        None
    } else {
        Some(SubstringInfo {
            matched_fields: matched,
            snippet,
            // A field hit's source/ref are the schema defaults (`source:
            // "field"`, `ref: None`).
            source: crate::FIELD_SOURCE.to_owned(),
            r#ref: None,
        })
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
    /// The result label this source's hits are annotated with.
    pub label: String,
    /// The query text (or the anchor note's text).
    pub text: String,
    /// Whether this is a query string (semantic + lexical) vs. an id anchor
    /// (semantic only).
    pub is_query: bool,
}

/// The per-call arguments (orchestrator state injected by the harness).
#[derive(Debug, Clone, Default)]
pub struct SearchArgs {
    /// Maximum fused results to return.
    pub top_k: usize,
    /// The cosine floor for the text-calibrated semantic signal.
    pub threshold: f64,
    /// The RESOLVED deck name (semantic candidates filter on exact equality;
    /// the find_notes fallback uses `deck:` which includes children — the
    /// Python original's behaviour, ported faithfully).
    pub deck: Option<String>,
    /// Restrict candidates to notes carrying all of these tags.
    pub tags: Vec<String>,
    /// Note IDs to exclude from results (e.g. the anchor note itself).
    pub exclude: Vec<i64>,
    /// The activation floor for the image modality (None = no gating).
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
    /// surfaces: a VectorOnly recognition source (VLM describe) is
    /// stored for provenance + reconcile but never surfaced on a lexical
    /// query. EMPTY (the default) = nothing hidden.
    pub hidden_lexical_sources: Vec<String>,
    /// SECONDARY embedding spaces' semantic results for cross-space fusion,
    /// one per space, each already embedded + searched at the kernel
    /// level. EMPTY (the default) is the N=1 / single-space case — the rankings
    /// fed to `rrf_fuse` are then EXACTLY the per-modality set, so the fused
    /// output is unchanged. Non-empty appends each space's gated `image`
    /// ranking (the relative activation gate, below).
    pub cross_space: Vec<SpaceSemantic>,
    /// Disable the cross-space relative activation gate — the NEGATIVE
    /// CONTROL only. `false` (the default) keeps the mandatory gate on; `true`
    /// fires every secondary space's image ranking ungated, which the eval
    /// showed floods text queries and regresses text recall (the load-bearing
    /// 0.08-vs-1.00 separation a test pins). Never set in production.
    pub disable_cross_space_gate: bool,
    /// The cross-space fusion variant. The floor/soft modes are eval-selectable
    /// measurement variants; production uses the [`CrossSpaceFusionMode`]
    /// default.
    pub cross_space_fusion_mode: CrossSpaceFusionMode,
    /// The temperature τ for the soft variants: smaller τ → a sharper
    /// taper (τ→0 is the binary floor limit). Ignored by the binary modes.
    pub cross_space_tau: f64,
    /// The total vision-WEIGHT BUDGET `B` for the `*Budget` floor-admission
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

/// One search candidate: the typed note record plus the working substring
/// annotation accumulated as the candidate is ranked. The score / provenance /
/// fuzzy evidence are NOT held here — they are assembled once, at the edge,
/// into a [`SearchMatch`] (this is the working model; that is the wire shape).
///
/// `substring` is `Some` once a literal-substring authority has annotated the
/// candidate (a field hit, or a derived OCR/ASR-row hit), `None` until then.
struct Candidate {
    note: Note,
    substring: Option<SubstringInfo>,
}

impl Candidate {
    fn new(note: Note) -> Self {
        Self {
            note,
            substring: None,
        }
    }
}

/// Insertion-ordered candidate cache: Python dict iteration order is part of
/// the exact ranking's contract, so the order log rides along.
struct NoteData {
    map: FxI64Map<Candidate>,
    order: Vec<i64>,
}

impl NoteData {
    fn new() -> Self {
        Self {
            map: FxI64Map::default(),
            order: Vec::new(),
        }
    }
    fn contains(&self, nid: i64) -> bool {
        self.map.contains_key(&nid)
    }
    fn insert(&mut self, nid: i64, candidate: Candidate) {
        if self.map.insert(nid, candidate).is_none() {
            self.order.push(nid);
        }
    }
}

fn in_scope(note: &Note, deck: Option<&str>, tags: &[String]) -> bool {
    if let Some(d) = deck {
        if note.deck != d {
            return false;
        }
    }
    if !tags.is_empty() {
        let note_tags: HashSet<&str> = note.tags.iter().map(String::as_str).collect();
        if !tags.iter().all(|t| note_tags.contains(t.as_str())) {
            return false;
        }
    }
    true
}

/// Hydrate candidates from the cross-source `prefetch` map, falling back to ONE
/// `note_dicts` call for any id not prefetched. `search_notes` prefetches every
/// candidate id across ALL sources in a single batched read (the union of the
/// semantic + lexical hits), so the common case here is a cheap clone out of
/// `prefetch` with no collection-actor round-trip — instead of a `note_dicts` per
/// ranking per source (each paying two DB-proxy queries plus a `deck_names` RPC
/// plus a notetype proto). The fallback keeps it correct when the prefetch is
/// incomplete — tag-member ids (not known until the loop) and secondary
/// cross-space image ids both hydrate this way. Each wire dict is a serialized
/// `Note`; a missing/unreadable id is simply absent (skipped per note).
fn read_notes_batch(
    core: &dyn Collection,
    prefetch: &FxI64Map<Note>,
    note_data: &NoteData,
    ids: &[i64],
) -> FxI64Map<Candidate> {
    let mut out: FxI64Map<Candidate> = FxI64Map::default();
    let mut need_fetch: Vec<i64> = Vec::new();
    for &nid in ids {
        if note_data.contains(nid) {
            continue;
        }
        match prefetch.get(&nid) {
            Some(note) => {
                out.insert(nid, Candidate::new(note.clone()));
            }
            None => need_fetch.push(nid),
        }
    }
    if need_fetch.is_empty() {
        return out;
    }
    match core.note_dicts(&need_fetch, true) {
        Ok(dicts) => {
            for d in dicts {
                match serde_json::from_value::<Note>(d) {
                    // A dict that won't parse as a Note is a core/schema
                    // disagreement (a bug), but a search must degrade rather
                    // than fail — skip the candidate, exactly like a missing id.
                    Ok(note) => {
                        out.insert(note.id, Candidate::new(note));
                    }
                    Err(e) => {
                        tracing::debug!(error = ?e, "search: candidate dict is not a Note; skipped");
                    }
                }
            }
        }
        Err(e) => {
            tracing::debug!(error = ?e, "search: batch hydrate failed; candidates skipped");
        }
    }
    out
}

/// The fusion signal name for a SECONDARY vision space's image ranking:
/// `image#<space-key>`. Distinct per space so provenance identifies which
/// vision space surfaced a note and each space fuses as its own RRF signal
/// (the canonical `search_weights` has no entry → `rrf_fuse` defaults its weight
/// to 1.0, equal weighting). Never collides with the primary's plain `image`
/// signal, so N=1 (no secondary) emits exactly the single-space signal set.
pub fn cross_space_signal(space_key: &str) -> String {
    format!("image#{space_key}")
}

/// The best (highest) query→match cosine across a space's NOTE-item modalities
/// for one source — `1 - rank-1 distance`, maxed over `text`/`image`. The
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

/// The intra-modal activation floor from `(mean, std)` of a modality's
/// typical best match (`mean + margin·std`), the kernel mirror of
/// `shrike.index.activation_floor`. The single source of the floor formula —
/// the harness-side secondary calibration routes through it. `None`
/// (uncalibrated — too few samples) → no floor, the gate disabled there.
pub fn activation_floor(stats: Option<(f64, f64)>, margin: f64) -> Option<f64> {
    stats.map(|(mean, std)| mean + margin * std)
}

/// The host-side `ACTIVATION_MARGIN` mirrored for the kernel-computed
/// cross-space floor — kept in lockstep with `shrike.actions.ACTIVATION_MARGIN`.
pub const ACTIVATION_MARGIN: f64 = 1.0;

/// Logistic squash `σ(x) = 1/(1+e^-x)` for the soft-weight variants.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The PER-NOTE image activation floor: retain in `ranking` only the
/// notes whose OWN image cosine (`score[id]`) clears `floor`, and prune the
/// dropped notes from `score` so the displayed-`score` fold and the `image_best`
/// read stay consistent. Per-note granularity (not a per-SPACE gate that admits
/// the whole ranking on the rank-1 cosine): a below-floor tail card no longer
/// rides in on a strong rank-1's coat-tails, so it carries no spurious image
/// signal/provenance.
///
/// `floor = None` (an uncalibrated space) is a no-op (the floor can't judge).
/// It can only TIGHTEN: every kept note cleared the floor, and the rank-1 (if
/// it cleared) is unchanged — so a genuine cross-modal find (above-floor by
/// construction) is preserved, while an ∅-gold ranking whose best is sub-floor
/// is emptied.
fn apply_image_floor(ranking: &mut Vec<i64>, score: &mut FxI64Map<f64>, floor: Option<f64>) {
    let Some(floor) = floor else {
        return;
    };
    ranking.retain(|nid| score.get(nid).is_some_and(|&c| c > floor));
    score.retain(|_, &mut c| c > floor);
}

/// The read-only context every ranking/collection helper threads identically:
/// the collection handle, the exclusion set, and the per-call args. Grouping
/// these into one struct keeps the helper signatures under the
/// `too_many_arguments` clippy threshold.
#[derive(Clone, Copy)]
struct SearchCtx<'a> {
    core: &'a dyn Collection,
    /// Cross-source prefetched notes (built once before the per-source loop), so
    /// hydration is a clone, not a per-source `note_dicts`. See [`read_notes_batch`].
    prefetch: &'a FxI64Map<Note>,
    exclude: &'a HashSet<i64>,
    args: &'a SearchArgs,
}

/// One modality's `search_by_modality` row as a borrowed pair (note keys +
/// distance-ascending distances), the unit `rank_modality` ranks.
#[derive(Clone, Copy)]
struct ModalitySlice<'a> {
    keys: &'a [i64],
    distances: &'a [f32],
}

impl<'a> ModalitySlice<'a> {
    fn empty() -> Self {
        Self {
            keys: &[],
            distances: &[],
        }
    }
}

/// Look up modality `name`'s `(keys, distances)` row as a [`ModalitySlice`],
/// the empty slice when the modality is absent.
fn modality_slice<'a>(hits: &'a ModalityHits, name: &str) -> ModalitySlice<'a> {
    hits.get(name)
        .map_or_else(ModalitySlice::empty, |(k, d)| ModalitySlice {
            keys: k,
            distances: d,
        })
}

fn rank_modality(
    ctx: SearchCtx,
    slice: ModalitySlice,
    note_data: &mut NoteData,
    sem_score: &mut FxI64Map<f64>,
    thresholded: bool,
) -> Vec<i64> {
    let SearchCtx {
        core,
        prefetch,
        exclude,
        args,
    } = ctx;
    // Prospective candidates (exclude/threshold pass) hydrate in ONE batch;
    // the loop below then filters scope and ranks.
    let prospective: Vec<i64> = slice
        .keys
        .iter()
        .zip(slice.distances.iter())
        .filter(|(nid, _)| !exclude.contains(nid))
        .take_while(|(_, dist)| !thresholded || round3(1.0 - f64::from(**dist)) >= args.threshold)
        .map(|(nid, _)| *nid)
        .collect();
    let mut hydrated = read_notes_batch(core, prefetch, note_data, &prospective);
    let mut ranking: Vec<i64> = Vec::new();
    for (nid, dist) in slice.keys.iter().zip(slice.distances.iter()) {
        let nid = *nid;
        if exclude.contains(&nid) {
            continue;
        }
        let score = round3(1.0 - f64::from(*dist));
        if thresholded && score < args.threshold {
            break; // distance-ascending → the rest are below threshold
        }
        if !note_data.contains(nid) {
            let candidate = match hydrated.remove(&nid) {
                Some(c) => c,
                None => continue,
            };
            if !in_scope(&candidate.note, args.deck.as_deref(), &args.tags) {
                continue; // out of scope — keep it out of note_data entirely
            }
            note_data.insert(nid, candidate);
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
    ctx: SearchCtx,
    lex: Option<Vec<LexicalRow>>,
    text: &str,
    note_data: &mut NoteData,
) -> NativeResult<()> {
    let SearchCtx {
        core,
        prefetch,
        exclude,
        args,
    } = ctx;
    // `lex` is this source's pre-fetched store result, read in one batched FTS5
    // pass shared with every other query source (the store pushes the deck/tag
    // scope id set into the MATCH query, so a scoped literal search reads no note
    // text outside the store). `None` = the store couldn't serve this query (a
    // sub-trigram query, or no store at all) → the wildcard `*text*` field-text
    // fallback below is correct. A REAL derived-read failure (e.g. a SQLITE_BUSY
    // that outlived the retry) already surfaced as an `Err` from the batched read
    // — it is never silently turned into a fallback, because OCR/ASR text lives
    // ONLY in the derived store and the field scan structurally cannot serve it.
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
        let mut hydrated = read_notes_batch(core, prefetch, note_data, &candidates);
        let mut added = 0usize;
        for nid in candidates.iter().copied() {
            if note_data.contains(nid) {
                continue;
            }
            let mut candidate = match hydrated.remove(&nid) {
                Some(c) => c,
                None => continue,
            };
            let Some(info) = substring_info(candidate.note.content.as_ref(), text) else {
                continue; // Anki matched across markup/normalization; not literal
            };
            candidate.substring = Some(info);
            note_data.insert(nid, candidate);
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
    let mut hydrated = read_notes_batch(core, prefetch, note_data, &row_ids);
    let mut added = 0usize;
    for (nid, source, reference, snippet) in rows {
        if exclude.contains(&nid) || note_data.contains(nid) {
            continue; // store may return a row per field
        }
        let mut candidate = match hydrated.remove(&nid) {
            Some(c) => c,
            None => continue,
        };
        if !in_scope(&candidate.note, args.deck.as_deref(), &args.tags) {
            continue;
        }
        // A derived-source row is its own authority: FTS5
        // matched the stored text's literal trigrams, and the field-content
        // re-check below would wrongly reject a literal living only in an
        // OCR/ASR row. Provenance carries the source + ref so the result can
        // say where it hit; field rows stay with the substring_info
        // authority over rendered content (their `substring` is computed in
        // the exact loop, so it stays `None` here).
        if source != crate::FIELD_SOURCE {
            candidate.substring = Some(SubstringInfo {
                matched_fields: Vec::new(),
                snippet,
                source,
                r#ref: Some(reference),
            });
        }
        note_data.insert(nid, candidate);
        added += 1;
        if added >= args.top_k {
            break;
        }
    }
    Ok(())
}

/// A fuzzy ranking plus its per-note evidence (the `fuzzy` annotation each
/// surfaced note carries, keyed by note id). The evidence is the wire
/// [`FuzzyMatch`] directly — no intermediate tuple to misread.
struct FuzzyRanking {
    ranking: Vec<i64>,
    evidence: FxI64Map<FuzzyMatch>,
}

impl FuzzyRanking {
    fn empty() -> Self {
        Self {
            ranking: Vec::new(),
            evidence: FxI64Map::default(),
        }
    }
}

fn collect_fuzzy(
    ctx: SearchCtx,
    hits: Vec<LexicalRow>,
    note_data: &mut NoteData,
) -> NativeResult<FuzzyRanking> {
    let SearchCtx {
        core,
        prefetch,
        exclude,
        args,
    } = ctx;
    // `hits` is this source's pre-fetched fuzzy rows, read in the batched FTS5
    // pass. The fuzzy signal has no anki-field fallback (it lives only in the
    // derived store), so a real derived-read failure already surfaced as an `Err`
    // from the batched read rather than silently returning empty.
    let hit_ids: Vec<i64> = hits
        .iter()
        .map(|(nid, ..)| *nid)
        .filter(|nid| !exclude.contains(nid))
        .collect();
    let mut hydrated = read_notes_batch(core, prefetch, note_data, &hit_ids);
    let mut out = FuzzyRanking::empty();
    for (nid, source, r#ref, snippet) in hits {
        if exclude.contains(&nid) || out.evidence.contains_key(&nid) {
            continue;
        }
        if !note_data.contains(nid) {
            let candidate = match hydrated.remove(&nid) {
                Some(c) => c,
                None => continue,
            };
            if !in_scope(&candidate.note, args.deck.as_deref(), &args.tags) {
                continue;
            }
            note_data.insert(nid, candidate);
        }
        out.ranking.push(nid);
        out.evidence.insert(
            nid,
            FuzzyMatch {
                source,
                r#ref,
                snippet,
            },
        );
        if out.ranking.len() >= args.top_k {
            break;
        }
    }
    Ok(out)
}

/// The inputs to one source's cross-space fusion pass: the shared search
/// context, the source's position (to pick each space's per-source row), and
/// the PRIMARY text space's modality hits (the relative gate's reference).
struct CrossSpaceInput<'a> {
    ctx: SearchCtx<'a>,
    source_index: usize,
    primary_hits: &'a ModalityHits,
}

/// What a source's cross-space fusion contributes: the per-space `image#<key>`
/// rankings to append to `rankings`, and any non-default per-space RRF weights
/// (the eval soft/budget modes) to fold into the weight map. Empty/empty for
/// the N=1 production-common case (no secondary spaces), so the fused inputs
/// match the no-cross-space path.
struct CrossSpaceContribution {
    rankings: Vec<(String, Vec<i64>)>,
    weights: BTreeMap<String, f64>,
}

/// One admitted secondary space's pre-budget contribution.
struct AdmittedSpace {
    signal: String,
    ranking: Vec<i64>,
    space_score: FxI64Map<f64>,
    weight: f64,
}

/// Cross-space fusion for one source. Each SECONDARY image
/// space contributes its own `image#<key>` ranking, folding its per-note image
/// cosines into `sem_score` (the displayed-score max-over-items) and hydrating
/// its candidates into `note_data`. EMPTY `cross_space` (N=1 — the
/// production-common case) returns an empty contribution, so the caller's
/// rankings/weights match the no-cross-space path.
///
/// PRODUCTION: FLOOR-ADMIT. No relative winner-take-all gate — a secondary
/// image space is admitted on its OWN calibrated intra-modal floor
/// (`image_best > z_floor`), independent of how the text space did, so a
/// strong on-topic CLIP hit corroborates the card even when text "won" on a
/// spurious filename lexical match (the win). Sound because >1 image space
/// is a config error (`profiles`): with at most one image space there is no
/// multiplicity, which was the relative gate's only job.
///
/// EVAL-ONLY (`SHRIKE_CROSS_SPACE_FUSION_MODE`): the relative family
/// (`Relative*`/`*Floor` non-admit) is the relative gate; the
/// `Soft*`/`*Budget` modes reproduce the dominated soft variant + the N≥2
/// multiplicity measurement. `uses_relative_gate` keeps the gate for the
/// relative family ONLY; the floor-admit family skips it.
///
/// Two passes: (1) admit + raw weight per space; (2) the budget normalization
/// across the admitted set (a no-op for the production FloorAdmit mode), then
/// fold/push.
fn fuse_cross_spaces(
    input: CrossSpaceInput,
    note_data: &mut NoteData,
    sem_score: &mut FxI64Map<f64>,
) -> CrossSpaceContribution {
    let CrossSpaceInput {
        ctx,
        source_index: i,
        primary_hits,
    } = input;
    let args = ctx.args;
    let mut contribution = CrossSpaceContribution {
        rankings: Vec::new(),
        weights: BTreeMap::new(),
    };
    if args.cross_space.is_empty() {
        return contribution;
    }
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
    // reference). The primary's hits are `primary_hits` (this source's row).
    let primary_best = best_query_cosine_of(primary_hits);
    // Pass 1: collect each admitted space's ranking, its per-note image
    // scores (for the displayed-score fold), and its RAW pre-budget weight.
    let mut admitted: Vec<AdmittedSpace> = Vec::new();
    for space in &args.cross_space {
        let Some(shits) = space.per_source.get(i) else {
            continue;
        };
        // The relative gate — applied ONLY for the relative family.
        // With the gate disabled (the negative control), every space fires.
        // The floor-admit family skips it entirely (the floor below is the sole
        // admission test).
        if uses_relative_gate {
            let gate_open = args.disable_cross_space_gate
                || match (shits.best_query_cosine, primary_best) {
                    (Some(v), Some(p)) => v >= p,
                    // No primary text reference → nothing to gate against;
                    // admit (degenerate, lexical-only primary).
                    (Some(_), None) => true,
                    // The space itself returned no hits → nothing to add.
                    (None, _) => false,
                };
            if !gate_open {
                continue;
            }
        }
        let slice = modality_slice(&shits.modality_hits, "image");
        if slice.keys.is_empty() {
            continue;
        }
        let mut space_score: FxI64Map<f64> = FxI64Map::default();
        let mut ranking_space_image = rank_modality(ctx, slice, note_data, &mut space_score, false);
        if ranking_space_image.is_empty() {
            continue;
        }
        // PRODUCTION FloorAdmit only: a PER-NOTE floor — keep only the
        // cards whose own image cosine clears this space's floor, so a
        // below-floor tail card carries no `image#clip`. The eval modes keep
        // the per-SPACE rule below (on `image_best`) to reproduce the decision
        // tables. An uncalibrated space (no floor) is a no-op either way.
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
        // The best surviving image cosine — the value the eval modes' per-space
        // floor judges (the rank-1 rule).
        let image_best = ranking_space_image
            .first()
            .and_then(|nid| space_score.get(nid).copied());

        // Apply the fusion mode's admission + raw weight. `None` drops the
        // space; `Some(w)` admits it at raw weight `w` (the budget pass may
        // scale it down for the `*Budget` modes).
        let weight = match mode {
            // V0 — relative only: contribute at weight 1.0.
            CrossSpaceFusionMode::Relative => Some(1.0),
            // V0+floor — relative AND z_s > z_floor (per-space). Drop the space
            // when its best surviving image cosine clears no floor; an
            // uncalibrated space (no floor) is admitted (the floor is a no-op,
            // the relative gate alone governs).
            CrossSpaceFusionMode::RelativeFloor => match (image_best, space.image_floor) {
                (Some(b), Some(floor)) if b <= floor => None,
                _ => Some(1.0),
            },
            // V1 — soft-relative: w = σ((clip_best − text_best)/τ). The
            // calibration-free CONTROL (expected to still leak).
            CrossSpaceFusionMode::SoftRelative => {
                match (shits.best_query_cosine, primary_best, args.cross_space_tau) {
                    (Some(v), Some(p), tau) if tau > 0.0 => Some(sigmoid((v - p) / tau)),
                    // No reference / τ≤0 → fall back to the binary contribution
                    // (the σ→step limit).
                    _ => Some(1.0),
                }
            }
            // V2 — soft-calibrated: w = σ((z_s − z0)/τ), composed with the
            // relative gate (already applied above). Tapers near the per-space
            // floor instead of a hard drop. An uncalibrated space falls back to
            // weight 1.0.
            CrossSpaceFusionMode::SoftCalibrated => {
                match (image_best, space.image_floor, args.cross_space_tau) {
                    (Some(b), Some(floor), tau) if tau > 0.0 => Some(sigmoid((b - floor) / tau)),
                    _ => Some(1.0),
                }
            }
            // FloorAdmit / FloorAdmitBudget — admit on the absolute
            // floor alone (relative gate already skipped). The PER-NOTE floor
            // filter above already dropped sub-floor cards and `continue`d if
            // none survived, so any surviving ranking has cleared the floor →
            // raw weight 1.0 (the budget pass divides it down for the budget
            // mode).
            CrossSpaceFusionMode::FloorAdmit | CrossSpaceFusionMode::FloorAdmitBudget => Some(1.0),
            // SoftFloorAdmit / SoftFloorAdmitBudget — soft admission
            // `w = σ((image_best − z_floor)/τ)`, no relative composition. An
            // uncalibrated space / τ≤0 falls back to weight 1.0. There is no
            // hard drop: a sub-floor hit gets a near-zero weight (negligible RRF
            // mass), which the soft taper is for.
            CrossSpaceFusionMode::SoftFloorAdmit | CrossSpaceFusionMode::SoftFloorAdmitBudget => {
                match (image_best, space.image_floor, args.cross_space_tau) {
                    (Some(b), Some(floor), tau) if tau > 0.0 => Some(sigmoid((b - floor) / tau)),
                    _ => Some(1.0),
                }
            }
        };
        let Some(weight) = weight else {
            continue; // the floor dropped this space's image ranking
        };
        admitted.push(AdmittedSpace {
            signal: cross_space_signal(&space.space_key),
            ranking: ranking_space_image,
            space_score,
            weight,
        });
    }

    // Pass 2: the vision-weight BUDGET. When N≥2 admitted spaces share a
    // bounded total weight `B`, no flood of always-confident off-topic spaces
    // can out-fuse the text answer (the negative control the relative gate
    // guards against). N=1 keeps full weight (the production-common case is
    // unpenalized). Two scalings:
    //   - binary budget (FloorAdmitBudget): every weight is 1.0, so each becomes
    //     B/N (the equal split).
    //   - soft budget (SoftFloorAdmitBudget): only scale DOWN, and only when the
    //     soft weights already sum above B (`B / Σ raw`); a confident N=1 hit
    //     keeps its near-1.0 weight.
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
        // Fold the space's image cosines into the displayed semantic `score`
        // (max-over-items, exactly like the primary image).
        for (nid, isim) in &a.space_score {
            let entry = sem_score.entry(*nid).or_insert(*isim);
            if *isim > *entry {
                *entry = *isim;
            }
        }
        // A DISTINCT signal name per space so provenance identifies which
        // vision space surfaced the note, and each space fuses as its own RRF
        // signal (weight defaults to 1.0 — equal weighting; the soft/budget
        // modes override it). `image#<key>` reads as "the image modality of
        // space <key>".
        if (a.weight - 1.0).abs() > f64::EPSILON {
            contribution.weights.insert(a.signal.clone(), a.weight);
        }
        contribution.rankings.push((a.signal, a.ranking));
    }
    contribution
}

/// The substring + fuzzy FTS5 batch results [`search_lexical`] returns: one
/// substring entry (`None` = unservable) and one fuzzy entry per QUERY source.
type LexicalBatches = (Vec<Option<Vec<LexicalRow>>>, Vec<Vec<LexicalRow>>);

/// The thread-domain DISCOVERY outputs — the index + derived reads, computed
/// WITHOUT touching `core`, so they can run off the collection actor (on the
/// compute pool) while `core` serves other work. [`search_assemble`] consumes
/// these to build the fused result from the prefetched notes.
pub(crate) struct Discovery {
    pub(crate) sem_by_source: Vec<ModalityHits>,
    pub(crate) substr_batch: Vec<Option<Vec<LexicalRow>>>,
    pub(crate) fuzzy_batch: Vec<Vec<LexicalRow>>,
}

/// The deck/tag lexical scope id set: one INDEXED anki query (deck:/tag: — never
/// a field-text scan) shared by both lexical collectors, pushed into the FTS5
/// queries so scoped literal/fuzzy search keeps exact recall without over-fetch.
/// `None` = unscoped. The only collection read the discovery phase needs, so it
/// is computed on the collection actor and handed to [`search_lexical`].
pub(crate) fn compute_lex_scope(
    core: &dyn Collection,
    args: &SearchArgs,
) -> NativeResult<Option<Vec<i64>>> {
    if args.deck.is_some() || !args.tags.is_empty() {
        let mut parts: Vec<String> = Vec::new();
        if let Some(d) = &args.deck {
            parts.push(format!("\"deck:{d}\""));
        }
        for tag in &args.tags {
            parts.push(format!("\"tag:{tag}\""));
        }
        Ok(Some(core.find_notes(&parts.join(" "))?))
    } else {
        Ok(None)
    }
}

/// The semantic DISCOVERY read: per-modality `search_by_modality` rankings, one
/// `ModalityHits` per source. No `core`. Over-fetches to cover exclusions and the
/// post-hoc scope/substring filtering the assembly applies.
pub(crate) fn search_semantic(
    index: &dyn VectorIndex,
    vectors: &[Vec<f32>],
    args: &SearchArgs,
) -> NativeResult<Vec<ModalityHits>> {
    let exclude: HashSet<i64> = args.exclude.iter().copied().collect();
    let mut fetch_k = args.top_k + exclude.len();
    if args.deck.is_some() || !args.tags.is_empty() {
        fetch_k = fetch_k.max(args.top_k * 10);
        if args.index_size > 0 {
            fetch_k = fetch_k.min(args.index_size);
        }
    }
    // Scoped to the NOTE-item spaces: tag-centroid spaces share the engine but
    // must never surface a tag key from a note search.
    let note_spaces: Vec<String> = crate::NOTE_MODALITIES
        .iter()
        .map(|m| m.to_string())
        .collect();
    index.search_by_modality(vectors, fetch_k, Some(&note_spaces))
}

/// The query texts (in source order) and the hidden-source exclusion list shared
/// by both lexical reads. Borrows from `sources`/`args` — no allocation beyond the
/// two `&str` vectors.
fn lexical_query_inputs<'a>(
    sources: &'a [SearchSource],
    args: &'a SearchArgs,
) -> (Vec<&'a str>, Vec<&'a str>) {
    let query_texts = sources
        .iter()
        .filter(|s| s.is_query)
        .map(|s| s.text.as_str())
        .collect();
    let hidden = args
        .hidden_lexical_sources
        .iter()
        .map(String::as_str)
        .collect();
    (query_texts, hidden)
}

/// The substring lexical read over a slice of query texts (already extracted from
/// the query sources). One entry per query text, in order; a `None` means "the
/// store couldn't serve this query" → the per-source fallback in the assembly.
/// Empty input yields an empty result.
///
/// [`crate::Kernel::search_fused`] dispatches one of these per query-batch CHUNK
/// so the per-query substring reads spread across the compute pool (each chunk
/// checks out its own derived read connection); the sync [`search_lexical`] path
/// calls it once over the whole batch. The over-fetch is `top_k + |distinct
/// exclude|`, so the post-filter still has `top_k` survivors after the excluded
/// ids drop out. A REAL derived-read failure surfaces as `Err` and must never
/// become a silent field-text fallback — OCR/ASR text lives only in the derived
/// store and the field scan can't serve it.
pub(crate) fn search_substring_chunk(
    derived: &dyn DerivedStore,
    query_texts: &[&str],
    hidden: &[&str],
    lex_scope: Option<&[i64]>,
    args: &SearchArgs,
) -> NativeResult<Vec<Option<Vec<LexicalRow>>>> {
    if query_texts.is_empty() {
        return Ok(Vec::new());
    }
    let exclude: HashSet<i64> = args.exclude.iter().copied().collect();
    derived.search_substring_batch(
        query_texts,
        (args.top_k + exclude.len()) as i64,
        lex_scope,
        hidden,
    )
}

/// The fuzzy lexical read over a slice of query texts — the overlap-ranked
/// counterpart to [`search_substring_chunk`] (see it for the chunking rationale).
/// One entry per query text, in order; empty input yields an empty result.
///
/// Chunking trades a little of `search_fuzzy_batch`'s cross-query trigram-posting
/// sharing (a trigram shared across chunks is read once per chunk) for the
/// per-query parallelism — for a bank of distinct queries the pruned trigrams are
/// mostly query-specific, so the lost sharing is small.
pub(crate) fn search_fuzzy_chunk(
    derived: &dyn DerivedStore,
    query_texts: &[&str],
    hidden: &[&str],
    lex_scope: Option<&[i64]>,
    args: &SearchArgs,
) -> NativeResult<Vec<Vec<LexicalRow>>> {
    if query_texts.is_empty() {
        return Ok(Vec::new());
    }
    derived.search_fuzzy_batch(query_texts, args.top_k as i64, lex_scope, hidden)
}

/// The lexical DISCOVERY reads (substring + fuzzy), sequentially over the whole
/// query batch on the caller's thread — the sync composition the binding
/// `search_notes` path uses. [`crate::Kernel::search_fused`] instead chunks each
/// read across the compute pool (see [`search_substring_chunk`]).
pub(crate) fn search_lexical(
    derived: &dyn DerivedStore,
    sources: &[SearchSource],
    lex_scope: Option<&[i64]>,
    args: &SearchArgs,
) -> NativeResult<LexicalBatches> {
    let (query_texts, hidden) = lexical_query_inputs(sources, args);
    let sub = search_substring_chunk(derived, &query_texts, &hidden, lex_scope, args)?;
    let fz = search_fuzzy_chunk(derived, &query_texts, &hidden, lex_scope, args)?;
    Ok((sub, fz))
}

/// Sequential fused search: discover (index + derived reads) then assemble, all
/// on the caller's thread. The `action_search_notes` binding rides this;
/// [`crate::Kernel::search_fused`] drives the same phases across thread domains.
///
/// # Errors
///
/// Returns an error if semantic ranking is requested without an index, or any
/// index/derived/collection read in the discovery or assembly fails.
pub fn search_notes(
    core: &dyn Collection,
    index: Option<&dyn VectorIndex>,
    derived: Option<&dyn DerivedStore>,
    tag_keys: Option<&crate::tag_centroids::TagKeyMap>,
    sources: &[SearchSource],
    vectors: &[Vec<f32>],
    args: &SearchArgs,
) -> NativeResult<Vec<SearchResultGroup>> {
    // Semantic first (its `index` precondition is checked before any read), then
    // the lexical scope — the original monolith's order, so a degenerate
    // `semantic && index == None` surfaces the precondition error rather than a
    // later find_notes failure.
    let sem_by_source = if args.semantic {
        let index = index.ok_or_else(|| {
            NativeError::invalid_input("semantic search requested without an index engine")
        })?;
        search_semantic(index, vectors, args)?
    } else {
        Vec::new()
    };
    let lex_scope = compute_lex_scope(core, args)?;
    let (substr_batch, fuzzy_batch) = match derived {
        Some(d) => search_lexical(d, sources, lex_scope.as_deref(), args)?,
        None => (Vec::new(), Vec::new()),
    };
    search_assemble(
        core,
        index,
        tag_keys,
        sources,
        vectors,
        args,
        Discovery {
            sem_by_source,
            substr_batch,
            fuzzy_batch,
        },
    )
}

/// The ASSEMBLY phase: hydrate every candidate id across all sources in ONE
/// batched `note_dicts`, then build each source's fused result from the
/// prefetched notes + the discovery rankings. Touches `core` for the prefetch and
/// the per-source `read_notes_batch` fallback (tag members + secondary
/// cross-space image ids, discovered in-loop).
///
/// # Errors
///
/// Returns an error if a collection read (prefetch / per-source hydration) fails.
///
/// # Panics
///
/// Panics only on an internal invariant violation — a candidate key collected
/// from the assembled note map must still be present when scored
/// (`expect("ordered key present")`).
pub(crate) fn search_assemble(
    core: &dyn Collection,
    index: Option<&dyn VectorIndex>,
    tag_keys: Option<&crate::tag_centroids::TagKeyMap>,
    sources: &[SearchSource],
    vectors: &[Vec<f32>],
    args: &SearchArgs,
    discovery: Discovery,
) -> NativeResult<Vec<SearchResultGroup>> {
    let exclude: HashSet<i64> = args.exclude.iter().copied().collect();
    let Discovery {
        sem_by_source,
        substr_batch,
        fuzzy_batch,
    } = discovery;

    // Collapse hydration: gather every candidate note id across ALL sources — the
    // semantic keys (per modality) plus the lexical (substring + fuzzy) row ids —
    // and hydrate them in ONE batched `note_dicts`, shared across the per-source
    // assembly via `read_notes_batch`. This replaces a `note_dicts` per ranking per
    // source (the house rule: discover the id set, then ONE batched read). Excluded
    // ids are never looked up, so drop them; ids not known until the loop
    // (tag members, secondary cross-space image hits) hydrate via
    // `read_notes_batch`'s per-source fallback.
    let prefetch: FxI64Map<Note> = {
        let mut cand_ids: HashSet<i64> = HashSet::new();
        for hits in &sem_by_source {
            for (keys, _) in hits.values() {
                cand_ids.extend(keys.iter().copied());
            }
        }
        for rows in substr_batch.iter().flatten() {
            cand_ids.extend(rows.iter().map(|(nid, ..)| *nid));
        }
        for rows in &fuzzy_batch {
            cand_ids.extend(rows.iter().map(|(nid, ..)| *nid));
        }
        cand_ids.retain(|nid| !exclude.contains(nid));
        if cand_ids.is_empty() {
            FxI64Map::default()
        } else {
            let ids: Vec<i64> = cand_ids.into_iter().collect();
            match core.note_dicts(&ids, true) {
                Ok(dicts) => dicts
                    .into_iter()
                    .filter_map(|d| serde_json::from_value::<Note>(d).ok().map(|n| (n.id, n)))
                    .collect(),
                Err(e) => {
                    tracing::debug!(error = ?e, "search: cross-source prefetch failed; per-source fallback");
                    FxI64Map::default()
                }
            }
        }
    };

    // Per-source consumers: advanced once per QUERY source, in source order, so
    // each query source draws its own pre-fetched rows.
    let mut substr_batch = substr_batch.into_iter();
    let mut fuzzy_batch = fuzzy_batch.into_iter();

    let mut results: Vec<SearchResultGroup> = Vec::new();
    for (i, source) in sources.iter().enumerate() {
        let mut note_data = NoteData::new();
        // The read-only context every helper threads (collection + exclusion +
        // args), grouped so the helpers take it as one param.
        let ctx = SearchCtx {
            core,
            prefetch: &prefetch,
            exclude: &exclude,
            args,
        };

        // Literal-substring candidates (query sources only): a fast pre-filter;
        // substring_info below is the authority that confirms + annotates. The
        // FTS5 rows were read in the batched pass above; `flatten` maps an
        // exhausted iterator (no store) or an unservable query to `None` → the
        // per-source field-text fallback.
        if source.is_query {
            let lex = substr_batch.next().flatten();
            collect_substring_candidates(ctx, lex, &source.text, &mut note_data)?;
        }

        // Per-modality semantic rankings. Text is thresholded; image is not
        // (the gap makes the text-calibrated cosine threshold meaningless —
        // flooring image hits is the activation gate's job below).
        let empty = ModalityHits::new();
        let modality_hits = sem_by_source.get(i).unwrap_or(&empty);
        let mut sem_score: FxI64Map<f64> = FxI64Map::default();
        let ranking_text = rank_modality(
            ctx,
            modality_slice(modality_hits, "text"),
            &mut note_data,
            &mut sem_score,
            true,
        );
        // Image modality into a scratch score first: the gate is judged on the
        // best hit that SURVIVES exclusion + scope, not the raw rank-1.
        let mut image_score: FxI64Map<f64> = FxI64Map::default();
        let mut ranking_image = rank_modality(
            ctx,
            modality_slice(modality_hits, "image"),
            &mut note_data,
            &mut image_score,
            false,
        );
        // Per-NOTE image floor — keep only the cards whose own image cosine
        // clears the floor (not a per-space all-or-nothing gate on the rank-1).
        // A below-floor tail card no longer rides in on the rank-1's coat-tails.
        // When the rank-1 itself is sub-floor the whole ranking empties.
        apply_image_floor(&mut ranking_image, &mut image_score, args.image_floor);
        // Tag-centroid signal: conditionally present — activated tags
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
                let mut scratch: FxI64Map<f64> = FxI64Map::default();
                ranking_tag = rank_modality(
                    ctx,
                    ModalitySlice {
                        keys: &member_ids,
                        distances: &synth,
                    },
                    &mut note_data,
                    &mut scratch,
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
        // The fuzzy rows came from the batched pass; an exhausted iterator (no
        // store) yields no fuzzy signal.
        let mut fuzzy = if source.is_query {
            collect_fuzzy(ctx, fuzzy_batch.next().unwrap_or_default(), &mut note_data)?
        } else {
            FuzzyRanking::empty()
        };

        // Exact ranking = every candidate whose content literally contains the
        // query (annotation ⟺ floated), in note_data insertion order.
        let mut exact_ids: Vec<i64> = Vec::new();
        if source.is_query {
            for nid in &note_data.order {
                let candidate = note_data.map.get_mut(nid).expect("ordered key present");
                if candidate.substring.is_none() {
                    candidate.substring =
                        substring_info(candidate.note.content.as_ref(), &source.text);
                }
                if candidate.substring.is_some() {
                    exact_ids.push(*nid);
                }
            }
        }

        // An exact note is trivially also a fuzzy match — drop it from the
        // fuzzy signal so `fuzzy` means the DISTINGUISHING lexical signal.
        if !exact_ids.is_empty() && !fuzzy.ranking.is_empty() {
            let exact_set: HashSet<i64> = exact_ids.iter().copied().collect();
            fuzzy.ranking.retain(|nid| !exact_set.contains(nid));
            fuzzy.evidence.retain(|nid, _| !exact_set.contains(nid));
        }

        let mut rankings: Vec<(String, Vec<i64>)> = vec![
            ("text".into(), ranking_text),
            ("image".into(), ranking_image),
            ("tag".into(), ranking_tag),
            ("exact".into(), exact_ids),
            ("fuzzy".into(), fuzzy.ranking),
        ];

        // Cross-space fusion — each SECONDARY image space
        // contributes its own `image#<key>` ranking + (eval) weight; empty for
        // the N=1 production-common case, so the inputs are unchanged.
        let cross = fuse_cross_spaces(
            CrossSpaceInput {
                ctx,
                source_index: i,
                primary_hits: modality_hits,
            },
            &mut note_data,
            &mut sem_score,
        );
        rankings.extend(cross.rankings);

        // Host-supplied weights override; empty means the kernel's canonical
        // set (the one source of truth in `fusion`).
        let mut weights = if args.weights.is_empty() {
            crate::fusion::search_weights()
        } else {
            args.weights.clone()
        };
        // Soft variants: the per-space `image#<key>` weight (a per-query
        // taper) overrides the canonical default of 1.0 for that signal.
        weights.extend(cross.weights);
        let priority: HashSet<String> =
            std::iter::once(crate::fusion::PRIORITY_SIGNAL.to_owned()).collect();
        let fused = crate::fusion::rrf_fuse(&rankings, &weights, crate::fusion::RRF_K, &priority);

        // Provenance: best (lowest) rank first, ties by signal name.
        // Assemble the typed wire match from the candidate + the per-note score,
        // provenance, and fuzzy evidence (serialization happens once, at the
        // host edge).
        let mut matches: Vec<SearchMatch> = Vec::new();
        for (nid, _score, signals) in fused.into_iter().take(args.top_k) {
            let candidate = note_data
                .map
                .remove(&nid)
                .expect("fused hit was a candidate");
            let mut prov: Vec<(String, i64)> = signals;
            prov.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            matches.push(SearchMatch {
                note: candidate.note,
                score: sem_score.get(&nid).copied(),
                substring: candidate.substring,
                fuzzy: fuzzy.evidence.remove(&nid),
                provenance: prov
                    .into_iter()
                    .map(|(signal, rank)| SignalContribution { signal, rank })
                    .collect(),
            });
        }
        results.push(SearchResultGroup {
            source: source.label.clone(),
            matches,
        });
    }

    Ok(results)
}

/// The read-time freshness stamps a search brackets its read with, so the
/// returned `stale` describes the SNAPSHOT THE RESULT WAS TAKEN AGAINST — not a
/// guess sampled before the read.
///
/// A search runs as ONE collection-actor job (the host dispatches it through
/// `wrapper.run` → `kernel.run_job`), so the entire read — the collection scan,
/// the Arc-shared vector-index and derived reads, the per-note renders — sits
/// inside one span. A *collection* write can't interleave (the actor is FIFO),
/// but the **ingest actor** writes the index/derived this read reads off a
/// separate task; `settled` (the ingest `outstanding == 0` gauge) is how that lag
/// is seen. Capture both stamps at the read's entry and exit and feed them to
/// [`is_stale_read`].
#[derive(Debug, Clone, Copy)]
pub struct FreshnessStamp {
    /// `core.col_mod()` at this edge of the read.
    pub col_mod: i64,
    /// The ingest `outstanding == 0` gauge at this edge (`true` when no
    /// committed-but-unindexed work is in flight). Hosts with no ingest actor (a
    /// facade/test) pass `true` — then only the `col_mod` stability check bites.
    pub settled: bool,
}

/// The read-time staleness verdict from the [`FreshnessStamp`]s bracketing a
/// search's read. Fresh ONLY if the read saw a stable (`col_mod` unchanged) and
/// settled (`outstanding == 0` at both edges) snapshot; any doubt → stale:
///
/// - `col_mod` changed across the read → the snapshot shifted under it (defends a
///   multi-job span or a reopen mid-read; stable within a single collection job
///   by construction);
/// - not settled at *either* edge → there are committed-but-unindexed writes, so
///   the index lags `col_mod` and a just-written note can be missing from the
///   semantic ranking.
pub fn is_stale_read(start: FreshnessStamp, end: FreshnessStamp) -> bool {
    start.col_mod != end.col_mod || !start.settled || !end.settled
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
        e.build(&rows, &ids, 1).unwrap();
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
    fn chunked_lexical_matches_whole_batch_at_every_width() {
        // `search_fused` splits the query batch into ~compute_width chunks per read
        // and reassembles BY INDEX. The chunked result must equal the whole-batch
        // `search_lexical` at EVERY width — the recall/ranking gate: chunking
        // changes only how the reads are batched, never the results.
        let (dir, core) = temp_collection();
        let notes: Vec<shrike_schemas::NoteInput> = serde_json::from_str(
            r#"[
              {"note_type": "Basic", "deck": "D",
               "fields": {"Front": "the mitochondria is the powerhouse of the cell", "Back": "b"}},
              {"note_type": "Basic", "deck": "D",
               "fields": {"Front": "the krebs cycle in the mitochondria", "Back": "b"}},
              {"note_type": "Basic", "deck": "D",
               "fields": {"Front": "mitochondrial dna replication", "Back": "b"}}
            ]"#,
        )
        .unwrap();
        core.upsert_notes(&notes, shrike_collection::DuplicatePolicy::Error, false)
            .unwrap();
        let e = derived_for(&core, &dir);

        // exact hits, a transposition typo (fuzzy), a no-match, and singletons.
        let bank = [
            "mitochondria",
            "mitochondira",
            "krebs",
            "zzznope",
            "powerhouse",
            "cell",
            "dna",
        ];
        let sources: Vec<SearchSource> = bank.iter().map(|t| query(t)).collect();
        let a = args(5);
        let scope: Option<&[i64]> = None;
        let (whole_sub, whole_fz) = search_lexical(&e, &sources, scope, &a).unwrap();

        for width in [1usize, 2, 3, 4, 8, 16] {
            // Mirror search_fused's chunking exactly: chunk_size = div_ceil(width),
            // contiguous ranges, in-order concat.
            let (qt, hidden) = lexical_query_inputs(&sources, &a);
            let chunk_size = qt.len().div_ceil(width.max(1)).max(1);
            let (mut sub, mut fz) = (Vec::new(), Vec::new());
            let mut start = 0;
            while start < qt.len() {
                let end = (start + chunk_size).min(qt.len());
                sub.extend(
                    search_substring_chunk(&e, &qt[start..end], &hidden, scope, &a).unwrap(),
                );
                fz.extend(search_fuzzy_chunk(&e, &qt[start..end], &hidden, scope, &a).unwrap());
                start = end;
            }
            assert_eq!(
                sub, whole_sub,
                "substring chunked@width={width} != whole batch"
            );
            assert_eq!(fz, whole_fz, "fuzzy chunked@width={width} != whole batch");
        }
    }

    #[test]
    fn scoped_lexical_search_serves_from_the_store() {
        // A deck-scoped literal/fuzzy search rides the FTS5 store with the scope
        // id set pushed into the query — exact recall inside the scope, zero
        // leakage outside it, and the wildcard `*text*` field scan is never
        // consulted (the store served Some).
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
        let results: Vec<serde_json::Value> = serde_json::from_str(
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
    /// shape of a derived-read failure that survives the busy-retry. All
    /// other methods delegate so build/ingest still work.
    struct FlakyDerived {
        inner: DerivedEngine,
        fail_lexical: bool,
    }
    impl shrike_store::DerivedStore for FlakyDerived {
        fn build(
            &self,
            rows: &[(i64, String, String, String)],
            live_notes: &[i64],
            col_mod: i64,
        ) -> NativeResult<()> {
            self.inner.build(rows, live_notes, col_mod)
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
        ) -> NativeResult<Vec<shrike_store::MatchRow>> {
            self.inner
                .match_rows(expr, limit, with_text, scope, exclude)
        }
        fn search_substring(
            &self,
            q: &str,
            limit: i64,
            scope: Option<&[i64]>,
            exclude: &[&str],
        ) -> NativeResult<Option<Vec<shrike_store::LexicalRow>>> {
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
        ) -> NativeResult<Vec<shrike_store::LexicalRow>> {
            if self.fail_lexical {
                return Err(NativeError::unavailable("derived busy (simulated)"));
            }
            self.inner.search_fuzzy(q, top_k, scope, exclude)
        }
    }

    #[test]
    fn derived_read_error_surfaces_never_silently_field_falls_back() {
        // A REAL derived-read failure (a busy that outlived the retry) must
        // SURFACE from search_notes, not silently fall back to the
        // `find_notes("*text*")` field scan. The field scan can't
        // serve OCR/ASR text (it lives only in the derived store, never in an
        // anki field), so a silent fallback would drop the OCR `exact`/`fuzzy`
        // signal with NO error — exactly the silent degradation in question.
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
    }

    #[test]
    fn derived_unavailable_none_still_field_falls_back() {
        // The other side of the contract: `Ok(None)` (a sub-trigram query, or no store)
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
        let content: NoteContent = BTreeMap::from([(
            "Front".to_owned(),
            "x".repeat(50) + "NEEDLE" + &"y".repeat(50),
        )]);
        let info = substring_info(Some(&content), "needle").expect("literal hit");
        assert_eq!(info.matched_fields, vec!["Front".to_owned()]);
        // A field hit carries the schema defaults (source "field", no ref).
        assert_eq!(info.source, crate::FIELD_SOURCE);
        assert!(info.r#ref.is_none());
        let snippet = info.snippet.as_deref().unwrap();
        assert!(snippet.starts_with('…') && snippet.ends_with('…'));
        assert!(snippet.contains("NEEDLE"));
        assert!(substring_info(Some(&content), "absent").is_none());
        // Absent content (a meta-mode note dict carries no fields) is no match.
        assert!(substring_info(None, "x").is_none());
    }

    #[test]
    fn substring_info_normalizes_both_needle_and_field() {
        // Confirmation is canonical-form-agnostic on BOTH sides, so it holds
        // regardless of Anki's NormalizeNoteText config.
        let nfd_needle = "cafe\u{0301}"; // c a f e + combining acute
        let nfc_needle = "café"; // precomposed é (U+00E9)
                                 // (a) NFC field (NormalizeNoteText on), NFD query.
        let nfc_field: NoteContent =
            BTreeMap::from([("Front".to_owned(), "le café du coin".to_owned())]);
        assert_eq!(
            substring_info(Some(&nfc_field), nfd_needle)
                .expect("NFD needle confirms NFC field")
                .matched_fields,
            vec!["Front".to_owned()]
        );
        // (b) NFD field (NormalizeNoteText off), NFC query — the field is
        // normalized too, so it still confirms (would regress if only the needle were).
        let nfd_field: NoteContent =
            BTreeMap::from([("Front".to_owned(), "le cafe\u{0301} du coin".to_owned())]);
        assert_eq!(
            substring_info(Some(&nfd_field), nfc_needle)
                .expect("NFC needle confirms NFD field")
                .matched_fields,
            vec!["Front".to_owned()]
        );
    }

    // ── Cross-space fusion + the relative activation gate ────────────────────
    //
    // These pin the eval's load-bearing findings against `search_notes`
    // DIRECTLY (no host wiring needed): the gate PRESERVES text-target ranking,
    // the negative control (gate OFF) REGRESSES it, and a gated vision space
    // DELIVERS image recall. The two spaces are a primary text engine (planted
    // vectors) + a secondary space's pre-searched `SpaceSemantic` rows — exactly
    // the shape `build_cross_space` produces.

    /// One secondary space's `SpaceSemantic` carrying image-modality hits for a
    /// single source, with the best query cosine the relative gate reads. No
    /// intra-modal floor (`image_floor: None`) — `vision_space_floored` adds one.
    fn vision_space(key: &str, image_keys: &[i64], image_dists: &[f32]) -> SpaceSemantic {
        vision_space_floored(key, image_keys, image_dists, None)
    }

    /// `vision_space` with an explicit per-space intra-modal image floor.
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
        // (a) The EVAL-ONLY relative gate: under `Relative`, an OFF-TOPIC vision
        // space (its best image cosine BELOW the primary text space's best) does
        // NOT fire — the text-target note stays rank-1. A kernel-level record of
        // the gate's behaviour; the production `FloorAdmit` would instead drop
        // the off-topic space by the floor, but this test exercises the relative
        // path explicitly.
        let (_dir, core) = temp_collection();
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
    }

    #[test]
    fn cross_space_ungated_regresses_text_negative_control() {
        // (b) THE NEGATIVE CONTROL — the eval's load-bearing finding (text R@1
        // collapse, the 0.08-vs-1.00 separation). Documents WHY the relative
        // gate exists: N=2 off-topic vision spaces flood a text query. The gate
        // is eval-only because >1 image space is a config error
        // (`profiles.resolve_profile`), so this N=2 shape is IMPOSSIBLE in
        // production — a kernel-level record of the relative gate (selected
        // explicitly via `Relative`), not a production path. The query is
        // ON-TOPIC for text; the off-topic image's vision cosine is BELOW the
        // text-target's, so the relative gate keeps it OUT; ungated (or under
        // floor-admit with no floor) it floods.
        let (_dir, core) = temp_collection();
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
        // note is excluded, and the text-target is restored to rank-1. The
        // load-bearing contrast: the gate IS what prevents (b) — and dropping it
        // is safe only because the N=2 input is a config error.
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
    }

    #[test]
    fn cross_space_gate_open_delivers_image_recall() {
        // (c) The payoff: a text query whose answer lives in a card's IMAGE
        // surfaces it through the gated vision space — the vision space's best
        // image cosine (0.92) CLEARS the primary text space's best (0.55), so
        // the relative gate OPENS and the image-bearing note joins the fusion
        // via its `image#clip` signal. (Without cross-space, the text-only
        // primary never sees it.)
        let (_dir, core) = temp_collection();
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
        // Per-space provenance: the surfacing signal is `image#clip`.
        assert!(
            img.provenance.iter().any(|p| p.signal == "image#clip"),
            "the match carries its vision space's per-space provenance"
        );
        core.close().unwrap();
    }

    // ── The cross-space intra-modal floor (the over-return leak) ─────────────
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
        crate::test_support::ScratchDir,
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
        // V0: the primary's best cosine ≈ 0, so the relative gate
        // `v(0.2) >= p(0.0)` is trivially OPEN and the weak off-topic image
        // leaks into the results. The leak the floor closes — pinned here so the
        // floor variants below have a positive baseline to beat.
        let (_dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
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
    }

    #[test]
    fn over_return_v0_floor_closes_the_leak() {
        // V0+floor: the space's best surviving image cosine (0.20) is BELOW its
        // OWN intra-modal floor (0.5) → the floor hard-drops the space's image
        // ranking. The weak off-topic image never enters. The relative gate is
        // still satisfied (empty primary) — the floor is the backstop.
        let (_dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
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
    }

    #[test]
    fn over_return_v0_floor_keeps_an_above_floor_image() {
        // The floor must NOT over-suppress: a vision space whose best image
        // cosine (0.92) CLEARS its own floor (0.5) still contributes, even
        // under an empty primary — the floor drops only the genuinely-weak best.
        let (_dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
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
    }

    #[test]
    fn over_return_v2_soft_calibrated_tapers_the_weak_image() {
        // V2 (soft-calibrated): the weak image (cos 0.20, floor 0.5) gets a
        // near-zero weight `σ((0.20-0.50)/τ)` at a small τ, so even if it stays
        // in the ranking its RRF mass is negligible — it does not out-rank the
        // primary's own (weak) text hit. With a tiny τ this approaches the hard
        // floor; here we assert the weak image is demoted below the text card.
        let (_dir, core, index, weak_text, image_note) = empty_primary_scenario(0.30);
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
    }

    #[test]
    fn over_return_v1_soft_relative_still_leaks() {
        // V1 (soft-relative, calibration-free): the weight is
        // `σ((clip_best − text_best)/τ)`. On an empty primary `text_best ≈ 0`
        // and `clip_best = 0.20`, so the argument is POSITIVE → weight ≈ 1.
        // V1 is the CONTROL: it does NOT consult the intra-modal floor, so it
        // STILL leaks the weak image. This proves the leak is intra-modal, not
        // relative.
        let (_dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
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
    }

    #[test]
    fn floor_holds_the_negative_control_at_n2() {
        // The negative control MUST hold under V0+floor: a text-target query
        // with N=2 off-topic-but-intra-modally-confident vision spaces. Each
        // space's image best (0.70) CLEARS its own floor (0.5) — so the floor
        // does NOT drop them — but the RELATIVE gate closes them (vision 0.70 <
        // text 0.80). The composition (floor AND relative) keeps the text-target
        // at rank-1: the floor never re-opens what the relative gate closed.
        let (_dir, core) = temp_collection();
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
    }

    // ── Floor-based admission (drop the relative gate) ───────────────────────
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
        // THE WIN — the filename-collision case at the unit level. A text
        // query whose answer is in a card's image, where the card's FILENAME
        // also lexically wins the primary text space (so the relative gate would
        // shut CLIP out). Here the primary text matches the image card STRONGLY
        // (cos 0.95 — the "filename won" proxy) while the vision space's image
        // best (0.85) is BELOW it → the relative gate CLOSES. But 0.85 clears the
        // floor (0.50), so floor-admission ADMITS the CLIP hit: the card carries
        // its `image#clip` provenance (the corroborating vote the relative gate
        // discarded).
        let (_dir, core) = temp_collection();
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
    }

    #[test]
    fn floor_admit_rejects_spurious_filename_image() {
        // THE PRECISION GUARD — the homonym/lying-filename case at the unit
        // level. A card's filename lexically wins the text space, but its IMAGE
        // is OFF-TOPIC for the query, so the vision space's image best (0.30)
        // falls BELOW the floor (0.50). Floor-admission must REJECT the CLIP hit:
        // the card may still surface via its filename text hit, but it carries NO
        // `image#clip` (the floor is the SOLE discriminator now that the relative
        // gate is gone — this is the load-bearing test for the thesis).
        let (_dir, core) = temp_collection();
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
    }

    #[test]
    fn floor_admit_keeps_the_over_return_leak_closed() {
        // The over-return leak must STAY closed under floor-admission: an
        // ∅-gold query, empty primary, one weak off-topic image (cos 0.20 < floor
        // 0.50). With no relative gate the floor is the only guard — and it holds
        // (0.20 ≤ 0.50 → dropped). Same outcome as RelativeFloor, different path.
        let (_dir, core, index, _weak_text, image_note) = empty_primary_scenario(0.0);
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
    }

    #[test]
    fn floor_admit_alone_floods_n2_but_budget_holds_it() {
        // THE MEASURED RATIONALE for the "no two image spaces" config
        // assertion. The N=2 negative control's protection comes ENTIRELY from
        // the relative gate (see `floor_holds_the_negative_control_at_n2`): both
        // off-topic spaces clear their floor, so dropping the relative gate lets
        // them flood. FloorAdmit (no budget) REGRESSES text R@1; FloorAdmitBudget
        // (B=1.0 split 0.5/0.5) restores it. In production this scenario is
        // IMPOSSIBLE (only one image space exists), so the budget is moot there —
        // WHY the relative gate can be dropped: not because floor-admission
        // handles N≥2, but because N≥2 image spaces never occur.
        let (_dir, core) = temp_collection();
        let text_target = add_note(&core, "the krebs cycle oxidizes acetyl coa", "biology");
        let off_topic_image = add_note(&core, "an off-topic but confident diagram", "img");
        let index = MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap();
        index
            .add("text", &[text_target], &[at_distance(0.2)])
            .unwrap(); // text cos 0.80
                       // Two intra-modally-confident but off-topic spaces (image best 0.70 >
                       // floor 0.50) — exactly the shape the relative gate closes.
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
    }

    #[test]
    fn floor_admit_n1_unaffected_by_budget() {
        // The production-common case: a SINGLE image space. The budget must not
        // penalize it — N=1 keeps full weight under FloorAdmitBudget (the budget
        // only divides when N≥2). The on-topic CLIP hit corroborates at full
        // strength exactly like FloorAdmit.
        let (_dir, core) = temp_collection();
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
    }

    #[test]
    fn soft_floor_admit_tapers_near_floor() {
        // SoftFloorAdmit: a borderline-above-floor image (cos 0.55, floor 0.50)
        // gets a tapered weight `σ((0.55-0.50)/τ)` < 1, while a confident one
        // (cos 0.92) gets ≈1. Here we assert the confident hit is admitted with
        // its provenance (the graceful form still corroborates), and a clearly
        // sub-floor hit (cos 0.20) gets a near-zero weight so it cannot out-rank
        // a real text card — the soft analogue of the hard floor's drop.
        let (_dir, core) = temp_collection();
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
    }

    // ── Per-note image floor (drop the below-floor tail) ─────────────────────

    #[test]
    fn floor_admit_secondary_drops_below_floor_tail() {
        // The per-NOTE floor on the SECONDARY cross-space image ranking. A
        // vision space surfaces TWO image cards: one ABOVE the floor (cos 0.92)
        // and one BELOW it (cos 0.20, floor 0.50). Floor-admission's per-note
        // filter keeps the above-floor card's `image#clip` and DROPS the
        // below-floor one — the latter no longer rides in on the rank-1's
        // coat-tails (a per-space gate would admit the whole ranking).
        let (_dir, core) = temp_collection();
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
    }

    #[test]
    fn primary_image_floor_is_per_note() {
        // The per-NOTE floor on the PRIMARY image ranking (the core,
        // used by omni/single-space deployments). The primary image modality
        // returns two cards: one above the floor (cos 0.90) and one below (cos
        // 0.20, floor 0.50). The above-floor card surfaces via `image`; the
        // below-floor one does NOT (a per-space gate would admit the whole
        // ranking iff the rank-1 cleared the floor).
        let (_dir, core) = temp_collection();
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
    }

    #[test]
    fn is_stale_read_is_the_conservative_or() {
        let fresh = FreshnessStamp {
            col_mod: 7,
            settled: true,
        };
        // Stable + settled at both edges → fresh.
        assert!(!is_stale_read(fresh, fresh));
        // Not settled at the start (a write was draining as the read began).
        assert!(is_stale_read(
            FreshnessStamp {
                col_mod: 7,
                settled: false
            },
            fresh
        ));
        // Not settled at the end (work arrived/was in flight by read's end).
        assert!(is_stale_read(
            fresh,
            FreshnessStamp {
                col_mod: 7,
                settled: false
            }
        ));
        // col_mod shifted across the read (the snapshot moved under it).
        assert!(is_stale_read(
            fresh,
            FreshnessStamp {
                col_mod: 8,
                settled: true
            }
        ));
    }
}
