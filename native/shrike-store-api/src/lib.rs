//! The store contract (#389) — the missing half of #342's plugin
//! architecture. Engines (embed/recognize) became pluggable there; the
//! kernel's three STORES stayed concrete. These traits make the deployment
//! ladder composition: a beefy server runs all-local impls, mobile (#226)
//! runs local stores + platform engines, a wasm thin client substitutes
//! remote impls for stores it can't host.
//!
//! Shape rules (the engine contract's, restated for stores):
//! - **Sync traits.** Scheduling is the KERNEL's concern — collection ops
//!   serialize through its task-actor, index/derived calls ride the op
//!   bodies or the blocking pool. An impl that talks to a network does its
//!   blocking I/O inside the method (the actor/pool absorbs it), exactly
//!   like sync compute engines behind `Blocking<E>`.
//! - **`Send + Sync`, object-safe.** The kernel holds `Arc<dyn …>`.
//! - **Typed vocabulary lives here.** A trait method's types cannot live in
//!   an impl crate (the dependency points the other way), so the row/stat
//!   aliases are canonical here and re-exported by the impls.
//!
//! [`VectorIndex`] and [`DerivedStore`] landed first (PR A); [`Collection`]
//! followed once its surface was typed (#391).

mod collection;

pub use collection::{
    Collection, CreateOutcome, DuplicatePolicy, ExportOutcome, ExportRequest, ExportScope,
    ImportOptions, ImportSummary, ImportUpdateCondition, OwnedFieldRow, PackageFormat,
    PreparedMedia, PreparedMediaSource, ServiceNote,
};

use shrike_ffi::NativeResult;
use std::collections::BTreeMap;

/// The byte-source media size cap — ONE policy value: the collection's write
/// tail and the kernel's fetch/decode caps must agree, so it lives where
/// both can see it.
pub const MEDIA_MAX_BYTES: usize = 64 * 1024 * 1024;

/// One modality's ranked hits for one query: `(keys, scores)`, best-first.
pub type ModalityRanking = (Vec<i64>, Vec<f32>);

/// Per-(non-text-)modality activation stats: `(modality, n, mean, std)`.
pub type ActivationStats = Vec<(String, f64, f64, f64)>;

/// One derived-text MATCH row:
/// `(note_id, source, ref, text?, snippet?)`.
pub type MatchRow = (i64, String, String, Option<String>, Option<String>);

/// One lexical (substring/fuzzy) hit: `(note_id, source, ref, snippet?)`.
pub type LexicalRow = (i64, String, String, Option<String>);

/// The per-modality vector store the kernel's index orchestration maintains
/// and the search paths rank against (#201a's sub-index layout is the
/// canonical impl: `shrike-index`'s usearch engine).
///
/// `save`/`restore` speak directory paths — persistence is the impl's
/// business; a store with no local durability may treat `save` as a no-op
/// and `restore` as "nothing to load" (`false`).
pub trait VectorIndex: Send + Sync {
    fn size(&self) -> usize;
    fn ndim(&self) -> Option<usize>;
    fn modality_sizes(&self) -> Vec<(String, usize)>;
    /// Per-modality `(name, size, ndim)` — the status breakdown (#684). Unlike
    /// [`modality_sizes`](Self::modality_sizes) it carries each sub-index's own
    /// dimensionality (text 768-dim, image 512-dim under CLIP), which the single
    /// top-level [`ndim`](Self::ndim) (the text modality's) can't express.
    /// `ndim` is `None` for a modality whose sub-index has no vectors yet.
    fn modality_stats(&self) -> Vec<(String, usize, Option<usize>)>;
    fn modality_names(&self) -> Vec<String>;
    /// Create/verify a modality's sub-index at `ndim` (idempotent; a
    /// dimension mismatch is an error).
    fn ensure(&self, modality: &str, ndim: usize) -> NativeResult<()>;
    /// `clear`/`drop_modality` are infallible for the local in-memory impl;
    /// widen to `NativeResult` when a fallible (remote) impl lands.
    fn clear(&self);
    fn drop_modality(&self, modality: &str);
    /// Load persisted state from `dir`; `candidates` bounds key discovery
    /// where the format can't enumerate keys. False = nothing restored.
    fn restore(&self, dir: &str, candidates: Option<&[i64]>) -> bool;
    /// Persist state under `dir`.
    ///
    /// **`save` must not block `search`** (#588): a save runs on the kernel's
    /// blocking pool concurrently with the search paths (the debounced/burst
    /// flush every 60s/100 changes, and `close()`), so an impl must not hold a
    /// lock across its on-disk write that `search_by_modality` also needs —
    /// otherwise every concurrent search stalls for the full save window. An
    /// impl with internal mutation locking should snapshot/serialize under the
    /// lock and write outside it (the #445 "never hold a lock across file
    /// writes" rule).
    fn save(&self, dir: &str) -> NativeResult<()>;
    fn add(&self, modality: &str, keys: &[i64], vectors: &[Vec<f32>]) -> NativeResult<()>;
    /// Remove every vector under each key, across modalities; returns the
    /// number of vectors REMOVED from the text modality (the note count —
    /// one text vector per note). NOT the number remaining: the canonical
    /// impl and the kernel consumer both read this as the removed count, and
    /// the kernel reports it as such (#608).
    fn remove(&self, keys: &[i64]) -> NativeResult<usize>;
    /// Rank each query against each (selected) modality separately — the
    /// per-modality RRF signals (#201a).
    fn search_by_modality(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        modalities: Option<&[String]>,
    ) -> NativeResult<Vec<BTreeMap<String, ModalityRanking>>>;
    fn contains(&self, key: i64) -> bool;
    fn keys(&self) -> Vec<i64>;
    fn get(&self, key: i64) -> Option<Vec<Vec<f32>>>;
    fn modality_contains(&self, modality: &str, key: i64) -> bool;
    fn modality_keys(&self, modality: &str) -> Vec<i64>;
    fn modality_get(&self, modality: &str, key: i64) -> Option<Vec<Vec<f32>>>;
    /// Dot products of `query` against the listed keys' vectors in one
    /// modality (the tag-centroid scorer's read).
    fn dot_scores(&self, modality: &str, keys: &[i64], query: &[f32]) -> Vec<(i64, f32)>;
    /// The intra-modal activation calibration (#201b): sample stored text
    /// vectors as pseudo-queries against each non-text modality.
    fn calibrate_activation(
        &self,
        sample_size: usize,
        k: usize,
        min_count: usize,
    ) -> NativeResult<ActivationStats>;
}

/// The derived-text store (#98) — the FTS5 trigram sidecar's contract:
/// `(note_id, source, ref)`-keyed rows backing the lexical search signals,
/// plus the recognition bookkeeping (segments, below-gate markers, the
/// fingerprint meta) and the `col_mod` watermark.
pub trait DerivedStore: Send + Sync {
    /// Full (re)build from `(note_id, source, ref, text)` rows, committing
    /// under the `col_mod` snapshot.
    fn build(&self, rows: &[(i64, String, String, String)], col_mod: i64) -> NativeResult<()>;
    /// Replace one note's rows for `source` with `(ref, text)` pairs.
    fn ingest(
        &self,
        note_id: i64,
        source: &str,
        refs_text: &[(String, String)],
    ) -> NativeResult<()>;
    /// One transaction over many notes (#445): the batch ingest.
    fn ingest_many(&self, notes: &[(i64, Vec<(String, String)>)], source: &str)
        -> NativeResult<()>;
    /// Drop rows by note — all sources, or one.
    fn remove(&self, note_ids: &[i64], source: Option<&str>) -> NativeResult<()>;
    fn count(&self) -> NativeResult<i64>;
    /// The stored drift watermark (the `col.mod` the store was last reconciled
    /// to), or `None` before the first build.
    fn get_col_mod(&self) -> Option<i64>;
    /// Stamp the drift watermark. INVARIANT (#585): set `value` ONLY after the
    /// rows for every write up to `value`'s `col.mod` are durably committed —
    /// the watermark is the sole drift signal, so over-stamping it silently
    /// hides an un-ingested note from substring/fuzzy search forever. A
    /// failed/partial ingest must leave the watermark behind for the next drift
    /// rebuild to heal.
    fn set_col_mod(&self, value: i64) -> NativeResult<()>;
    fn meta_get(&self, key: &str) -> NativeResult<Option<String>>;
    fn meta_set(&self, key: &str, value: &str) -> NativeResult<()>;
    fn refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>>;
    fn texts_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String, String)>>;
    fn texts_for_source_for_notes(
        &self,
        source: &str,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, String)>>;
    /// Below-gate markers (#416): judged-once bookkeeping for items the
    /// recognition gate dropped.
    fn mark_gated(&self, source: &str, pairs: &[(i64, String)]) -> NativeResult<()>;
    fn gated_refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>>;
    fn clear_gated(&self, source: &str) -> NativeResult<()>;
    /// Per-segment recognition structure (#228), JSON per (note, ref). The
    /// read half has no production caller yet — seamed for #230 (occlusion),
    /// which reads the boxes back.
    fn put_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
        json: &str,
    ) -> NativeResult<()>;
    fn get_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
    ) -> NativeResult<Option<String>>;
    /// Raw FTS5 MATCH (the expression is the impl's syntax), scoped.
    /// `exclude_sources` drops rows whose `source` is in the set BEFORE
    /// ranking/limiting (#485): a VectorOnly recognition source (VLM
    /// describe) is stored for provenance + reconcile but must never surface
    /// on a lexical query — an empty slice is the historical behaviour.
    fn match_rows(
        &self,
        expr: &str,
        limit: i64,
        with_text: bool,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<MatchRow>>;
    /// Fast substring candidates; `None` = the query can't be served (too
    /// short for the tokenizer) and the caller falls back. `exclude_sources`
    /// hides VectorOnly sources (#485) — see [`Self::match_rows`].
    fn search_substring(
        &self,
        query: &str,
        limit: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Option<Vec<LexicalRow>>>;
    /// Trigram/typo ranking — the `fuzzy` RRF signal (#98). `exclude_sources`
    /// hides VectorOnly sources (#485) — see [`Self::match_rows`].
    fn search_fuzzy(
        &self,
        query: &str,
        top_k: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<LexicalRow>>;
}
