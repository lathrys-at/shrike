//! The store contract — the missing half of the plugin
//! architecture. Engines (embed/recognize) are pluggable; this makes the
//! kernel's STORES pluggable too. These traits make the deployment ladder
//! composition: a beefy server runs all-local impls, mobile runs local
//! stores + platform engines, a wasm thin client substitutes remote impls for
//! stores it can't host.
//!
//! The `Collection` trait + its vocabulary live in `shrike-collection` (its
//! sole implementer — homing the trait beside its only impl removes the edge a
//! separate contract crate forced). What stays here are the two traits with
//! *two* impl crates each over disjoint backends — [`VectorIndex`]
//! (`shrike-index`) and [`DerivedStore`] (`shrike-derived`) — which therefore
//! CANNOT live in either impl crate (the dependency points the other way). (The
//! `MEDIA_MAX_BYTES` policy value lives in `shrike-media`, since both the
//! collection write tail and the media fetch/decode caps depend on that crate.)
//!
//! Shape rules (the engine contract's, restated for stores):
//! - **Sync traits.** Scheduling is the KERNEL's concern — index/derived calls
//!   ride the op bodies or the blocking pool. An impl that talks to a network
//!   does its blocking I/O inside the method (the pool absorbs it), exactly
//!   like sync compute engines behind `Blocking<E>`.
//! - **`Send + Sync`, object-safe.** The kernel holds `Arc<dyn …>`.
//! - **Typed vocabulary lives here.** A trait method's types cannot live in
//!   an impl crate (the dependency points the other way), so the row/stat
//!   aliases are canonical here and re-exported by the impls.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use shrike_error::NativeResult;
use std::collections::BTreeMap;

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
/// and the search paths rank against (the per-modality sub-index layout is the
/// canonical impl: `shrike-index`'s usearch engine).
///
/// `save`/`restore` speak directory paths — persistence is the impl's
/// business; a store with no local durability may treat `save` as a no-op
/// and `restore` as "nothing to load" (`false`).
pub trait VectorIndex: Send + Sync {
    /// Total vectors in the text modality (the note count).
    fn size(&self) -> usize;
    /// The text modality's dimensionality, or `None` before any vectors land.
    fn ndim(&self) -> Option<usize>;
    /// Per-modality `(name, size)` vector counts.
    fn modality_sizes(&self) -> Vec<(String, usize)>;
    /// Per-modality `(name, size, ndim)` — the status breakdown. Unlike
    /// [`modality_sizes`](Self::modality_sizes) it carries each sub-index's own
    /// dimensionality (text 768-dim, image 512-dim under CLIP), which the single
    /// top-level [`ndim`](Self::ndim) (the text modality's) can't express.
    /// `ndim` is `None` for a modality whose sub-index has no vectors yet.
    fn modality_stats(&self) -> Vec<(String, usize, Option<usize>)>;
    /// The names of the modalities with a sub-index.
    fn modality_names(&self) -> Vec<String>;
    /// Create/verify a modality's sub-index at `ndim` (idempotent; a
    /// dimension mismatch is an error).
    ///
    /// # Errors
    ///
    /// Returns an error if `ndim` conflicts with an existing sub-index for
    /// `modality`, or the backend cannot create it.
    fn ensure(&self, modality: &str, ndim: usize) -> NativeResult<()>;
    /// `clear`/`drop_modality` are infallible for the local in-memory impl;
    /// widen to `NativeResult` when a fallible (remote) impl lands.
    fn clear(&self);
    /// Drop a modality's sub-index entirely.
    fn drop_modality(&self, modality: &str);
    /// Load persisted state from `dir`; `candidates` bounds key discovery
    /// where the format can't enumerate keys. False = nothing restored.
    fn restore(&self, dir: &str, candidates: Option<&[i64]>) -> bool;
    /// Persist state under `dir`.
    ///
    /// **`save` must not block `search`**: a save runs on the kernel's
    /// blocking pool concurrently with the search paths (the debounced/burst
    /// flush every 60s/100 changes, and `close()`), so an impl must not hold a
    /// lock across its on-disk write that `search_by_modality` also needs —
    /// otherwise every concurrent search stalls for the full save window. An
    /// impl with internal mutation locking should snapshot/serialize under the
    /// lock and write outside it (the "never hold a lock across file
    /// writes" rule).
    ///
    /// # Errors
    ///
    /// Returns an error if the on-disk write fails.
    fn save(&self, dir: &str) -> NativeResult<()>;
    /// Add `vectors` under `keys` in `modality` (one vector per key).
    ///
    /// # Errors
    ///
    /// Returns an error if `keys`/`vectors` lengths disagree, a vector's
    /// dimension conflicts with the modality, or the backend rejects the write.
    fn add(&self, modality: &str, keys: &[i64], vectors: &[Vec<f32>]) -> NativeResult<()>;
    /// Remove every vector under each key, across modalities; returns the
    /// number of vectors REMOVED from the text modality (the note count —
    /// one text vector per note). NOT the number remaining: the canonical
    /// impl and the kernel consumer both read this as the removed count, and
    /// the kernel reports it as such.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend rejects the removal.
    fn remove(&self, keys: &[i64]) -> NativeResult<usize>;
    /// Rank each query against each (selected) modality separately — the
    /// per-modality RRF signals.
    ///
    /// # Errors
    ///
    /// Returns an error if a query's dimension conflicts with a searched
    /// modality, or the backend search fails.
    fn search_by_modality(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        modalities: Option<&[String]>,
    ) -> NativeResult<Vec<BTreeMap<String, ModalityRanking>>>;
    /// Whether any vector is stored under `key` (any modality).
    fn contains(&self, key: i64) -> bool;
    /// Every key with a vector in the text modality.
    fn keys(&self) -> Vec<i64>;
    /// The vectors stored under `key` across modalities, or `None`.
    fn get(&self, key: i64) -> Option<Vec<Vec<f32>>>;
    /// Whether `key` has a vector in `modality`.
    fn modality_contains(&self, modality: &str, key: i64) -> bool;
    /// Every key with a vector in `modality`.
    fn modality_keys(&self, modality: &str) -> Vec<i64>;
    /// The vectors stored under `key` in `modality`, or `None`.
    fn modality_get(&self, modality: &str, key: i64) -> Option<Vec<Vec<f32>>>;
    /// Dot products of `query` against the listed keys' vectors in one
    /// modality (the tag-centroid scorer's read).
    fn dot_scores(&self, modality: &str, keys: &[i64], query: &[f32]) -> Vec<(i64, f32)>;
    /// The intra-modal activation calibration: sample stored text
    /// vectors as pseudo-queries against each non-text modality.
    ///
    /// # Errors
    ///
    /// Returns an error if the sampling search against a non-text modality
    /// fails.
    fn calibrate_activation(
        &self,
        sample_size: usize,
        k: usize,
        min_count: usize,
    ) -> NativeResult<ActivationStats>;
}

/// The derived-text store — the FTS5 trigram sidecar's contract:
/// `(note_id, source, ref)`-keyed rows backing the lexical search signals,
/// plus the recognition bookkeeping (segments, below-gate markers, the
/// fingerprint meta) and the `col_mod` watermark.
pub trait DerivedStore: Send + Sync {
    /// Full (re)build from `(note_id, source, ref, text)` rows, committing
    /// under the `col_mod` snapshot. `live_notes` is the authoritative set of
    /// note ids currently in the collection: recognition rows are pruned only
    /// for notes absent from it, never merely for notes absent from `rows` (a
    /// note can be live yet contribute no field rows — all-blank fields, or a
    /// snapshot taken before the note was written).
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the rebuild transaction.
    fn build(
        &self,
        rows: &[(i64, String, String, String)],
        live_notes: &[i64],
        col_mod: i64,
    ) -> NativeResult<()>;
    /// Replace one note's rows for `source` with `(ref, text)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the write.
    fn ingest(
        &self,
        note_id: i64,
        source: &str,
        refs_text: &[(String, String)],
    ) -> NativeResult<()>;
    /// One transaction over many notes: the batch ingest.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the batch transaction.
    fn ingest_many(&self, notes: &[(i64, Vec<(String, String)>)], source: &str)
        -> NativeResult<()>;
    /// Drop rows by note — all sources, or one.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing store rejects the delete.
    fn remove(&self, note_ids: &[i64], source: Option<&str>) -> NativeResult<()>;
    /// The total row count.
    ///
    /// # Errors
    ///
    /// Returns an error if the count query fails.
    fn count(&self) -> NativeResult<i64>;
    /// The stored drift watermark (the `col.mod` the store was last reconciled
    /// to), or `None` before the first build.
    fn get_col_mod(&self) -> Option<i64>;
    /// Stamp the drift watermark. INVARIANT: set `value` ONLY after the
    /// rows for every write up to `value`'s `col.mod` are durably committed —
    /// the watermark is the sole drift signal, so over-stamping it silently
    /// hides an un-ingested note from substring/fuzzy search forever. A
    /// failed/partial ingest must leave the watermark behind for the next drift
    /// rebuild to heal.
    ///
    /// # Errors
    ///
    /// Returns an error if the watermark write fails.
    fn set_col_mod(&self, value: i64) -> NativeResult<()>;
    /// Read a meta value by key, or `None` if unset.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn meta_get(&self, key: &str) -> NativeResult<Option<String>>;
    /// Write a meta key/value.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn meta_set(&self, key: &str, value: &str) -> NativeResult<()>;
    /// All `(note_id, ref)` rows for `source`.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>>;
    /// All `(note_id, ref, text)` rows for `source`.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn texts_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String, String)>>;
    /// `(note_id, ref, text)` rows for `source`, scoped to `note_ids`.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn texts_for_source_for_notes(
        &self,
        source: &str,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, String)>>;
    /// Below-gate markers: judged-once bookkeeping for items the
    /// recognition gate dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the marker write fails.
    fn mark_gated(&self, source: &str, pairs: &[(i64, String)]) -> NativeResult<()>;
    /// The below-gate `(note_id, ref)` markers for `source`.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn gated_refs_for_source(&self, source: &str) -> NativeResult<Vec<(i64, String)>>;
    /// Clear the below-gate markers for `source`.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    fn clear_gated(&self, source: &str) -> NativeResult<()>;
    /// Per-segment recognition structure, JSON per (note, ref). The
    /// read half has no production caller yet — seamed for occlusion,
    /// which reads the boxes back.
    ///
    /// # Errors
    ///
    /// Returns an error if the segment write fails.
    fn put_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
        json: &str,
    ) -> NativeResult<()>;
    /// Read the stored per-segment JSON for one `(note, source, ref)`, or `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn get_segments(
        &self,
        note_id: i64,
        source: &str,
        reference: &str,
    ) -> NativeResult<Option<String>>;
    /// Raw FTS5 MATCH (the expression is the impl's syntax), scoped.
    /// `exclude_sources` drops rows whose `source` is in the set BEFORE
    /// ranking/limiting: a VectorOnly recognition source (VLM
    /// describe) is stored for provenance + reconcile but must never surface
    /// on a lexical query — an empty slice is the historical behaviour.
    ///
    /// # Errors
    ///
    /// Returns an error if the MATCH query fails.
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
    /// hides VectorOnly sources — see [`Self::match_rows`].
    ///
    /// # Errors
    ///
    /// Returns an error if the search query fails.
    fn search_substring(
        &self,
        query: &str,
        limit: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Option<Vec<LexicalRow>>>;
    /// Trigram/typo ranking — the `fuzzy` RRF signal. `exclude_sources`
    /// hides VectorOnly sources — see [`Self::match_rows`].
    ///
    /// # Errors
    ///
    /// Returns an error if the search query fails.
    fn search_fuzzy(
        &self,
        query: &str,
        top_k: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<LexicalRow>>;
}
