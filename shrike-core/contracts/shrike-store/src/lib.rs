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

/// A fast non-cryptographic hasher for the search path's `i64`-keyed maps (rowids,
/// note-ids). `std`'s default is SipHash — DoS-resistant, but slow on small integer
/// keys; these maps are internal (keys are our own index rowids/note-ids, never
/// adversarial input), so the FxHash rotate-xor-multiply is the right trade. The
/// hash cost shows up directly in the search profile (the fuzzy overlap maps, the
/// kernel's RRF score maps, the vector index's scope set), so it lives here — the
/// only crate the index, derived, and kernel impls all depend on.
#[derive(Default)]
pub struct FxI64Hasher(u64);

impl FxI64Hasher {
    const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    #[inline]
    fn add(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(Self::K);
    }
}

impl std::hash::Hasher for FxI64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    // Required, but the i64/u64-keyed maps never reach it; hash byte-wise as a
    // correct fallback so the type stays a general `Hasher`.
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }
}

/// `HashMap` keyed on `i64` with [`FxI64Hasher`] — the search path's integer-keyed
/// map type (fuzzy overlap, RRF score maps).
pub type FxI64Map<V> =
    std::collections::HashMap<i64, V, std::hash::BuildHasherDefault<FxI64Hasher>>;

/// `HashSet` keyed on `i64` with [`FxI64Hasher`] — the search path's integer-keyed
/// set type (the deck/tag scope membership set).
pub type FxI64Set = std::collections::HashSet<i64, std::hash::BuildHasherDefault<FxI64Hasher>>;

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
    /// `scope`, when set, is the note-id set a deck/tag-scoped search restricts to:
    /// it is pushed into the index walk as a per-candidate predicate (so the walk
    /// returns only in-scope neighbours, no over-fetch-then-filter), NOT applied
    /// afterwards. `None` searches the whole index.
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
        scope: Option<&[i64]>,
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
    /// Streaming [`Self::build`]: pull `(note_id, source, ref, text)` row chunks
    /// via `next` (`None` ends the stream) and ingest them within ONE
    /// transaction, so peak memory is O(chunk), not O(collection). `next` blocks
    /// the calling thread; a producer reads the collection in chunks on another
    /// thread, so reads overlap the FTS5 inserts. Returns the total rows seen.
    /// The default materializes the whole stream and calls [`Self::build`].
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk read or the rebuild transaction fails.
    #[allow(clippy::type_complexity)]
    fn build_streamed(
        &self,
        next: &mut dyn FnMut() -> Option<NativeResult<Vec<(i64, String, String, String)>>>,
        live_notes: &[i64],
        col_mod: i64,
    ) -> NativeResult<usize> {
        let mut rows: Vec<(i64, String, String, String)> = Vec::new();
        while let Some(chunk) = next() {
            rows.extend(chunk?);
        }
        let n = rows.len();
        self.build(&rows, live_notes, col_mod)?;
        Ok(n)
    }
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
    /// Re-materialize any per-write-stable derived snapshot the read path caches —
    /// currently the trigram document-frequency table the fuzzy prune ranks on. A
    /// full (re)build refreshes it inline; this is the incremental-write companion,
    /// meant to be called (debounced) after a write batch settles so the snapshot
    /// tracks `ingest_many`/`remove` between rebuilds. Idempotent; cheap to over-call.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot rewrite fails.
    fn refresh_derived_snapshots(&self) -> NativeResult<()>;
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
    /// [`Self::search_substring`] over a batch of queries, one result per query
    /// in `queries` order. A fused search runs the lexical signals once per query
    /// string; batching lets an impl pay the fixed per-call cost (connection
    /// lock, scope staging, statement compile) ONCE for the whole set instead of
    /// once per query. The default loops the singular method (correct, unbatched);
    /// the local engine overrides it with a single-lock, single-prepare pass.
    ///
    /// # Errors
    ///
    /// Returns an error if any query's search fails (the batch is abandoned at
    /// the first failure, exactly as the singular reads surface one).
    fn search_substring_batch(
        &self,
        queries: &[&str],
        limit: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Option<Vec<LexicalRow>>>> {
        queries
            .iter()
            .map(|q| self.search_substring(q, limit, scope, exclude_sources))
            .collect()
    }
    /// [`Self::search_fuzzy`] over a batch of queries, one result per query in
    /// `queries` order — the fuzzy counterpart to
    /// [`Self::search_substring_batch`].
    ///
    /// # Errors
    ///
    /// Returns an error if any query's search fails (the batch is abandoned at
    /// the first failure, exactly as the singular reads surface one).
    fn search_fuzzy_batch(
        &self,
        queries: &[&str],
        top_k: i64,
        scope: Option<&[i64]>,
        exclude_sources: &[&str],
    ) -> NativeResult<Vec<Vec<LexicalRow>>> {
        queries
            .iter()
            .map(|q| self.search_fuzzy(q, top_k, scope, exclude_sources))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
    use std::sync::Mutex;

    fn hash_one_i64(key: i64) -> u64 {
        let mut h = FxI64Hasher::default();
        h.write_i64(key);
        h.finish()
    }

    // ---- FxI64Hasher ----------------------------------------------------

    #[test]
    fn fxhasher_single_key_write_is_a_bijection() {
        // With an empty starting state, `write_i64(k)` reduces to
        // `(0.rotate_left(5) ^ k as u64) * K` = `(k as u64) * K`, and K is odd,
        // so single-key hashing is a bijection over u64 — distinct keys NEVER
        // collide. This is load-bearing: rowids/note-ids are sequential, and a
        // hasher that collided on sequential small integers would silently
        // degrade every search-path map (`FxI64Map`/`FxI64Set`). Pin it across
        // sequential keys, both signs, the boundaries, and a random sweep.
        let mut seen = std::collections::HashSet::new();
        let push = |k: i64, seen: &mut std::collections::HashSet<u64>| {
            assert!(
                seen.insert(hash_one_i64(k)),
                "single-key hash collided at key {k}"
            );
        };
        for k in -5_000_i64..=5_000 {
            push(k, &mut seen);
        }
        for k in [i64::MIN, i64::MAX, i64::MIN + 1, i64::MAX - 1] {
            assert!(seen.insert(hash_one_i64(k)), "boundary key {k} collided");
        }
    }

    proptest! {
        /// The bijection, stated directly: equal hashes imply equal keys, over
        /// the full i64 range. This is the actual injectivity property the
        /// sequential sweep above only samples — proptest shrinks any collision
        /// to a minimal `(a, b)` witness.
        #[test]
        fn fxhasher_hash_equality_mirrors_key_equality(a: i64, b: i64) {
            prop_assert_eq!(hash_one_i64(a) == hash_one_i64(b), a == b);
        }

        /// The impl casts `i as u64` in `write_i64`; the two entry points must
        /// agree for the same 64-bit pattern, or a map keyed via one path and
        /// probed via the other would miss.
        #[test]
        fn fxhasher_write_i64_and_u64_agree_on_bit_pattern(bits: u64) {
            let mut hi = FxI64Hasher::default();
            hi.write_i64(bits as i64);
            let mut hu = FxI64Hasher::default();
            hu.write_u64(bits);
            prop_assert_eq!(hi.finish(), hu.finish());
        }
    }

    #[test]
    fn fxhasher_is_deterministic_and_empty_is_zero() {
        assert_eq!(hash_one_i64(42), hash_one_i64(42));
        // No writes → no state mutation → finish() is the zero seed.
        assert_eq!(FxI64Hasher::default().finish(), 0);
        // The byte fallback on empty input is likewise a no-op.
        let mut h = FxI64Hasher::default();
        h.write(&[]);
        assert_eq!(h.finish(), 0);
    }

    #[test]
    fn fxhasher_byte_fallback_is_deterministic_and_position_sensitive() {
        let h = |bytes: &[u8]| {
            let mut h = FxI64Hasher::default();
            h.write(bytes);
            h.finish()
        };
        assert_eq!(h(b"abc"), h(b"abc"));
        // rotate-xor-multiply folds position in, so a transposition changes the
        // digest (not a guarantee for all inputs, but it must for this one).
        assert_ne!(h(b"abc"), h(b"acb"));
        assert_ne!(h(b"ab"), h(b"ba"));
    }

    /// One step in a randomized map/set trace. Keys are drawn from a small band
    /// (`KEY_BAND`) so a generated trace collides and overwrites instead of only
    /// ever inserting fresh keys.
    #[derive(Debug, Clone)]
    enum MapOp {
        Insert(i64, i64),
        Remove(i64),
    }

    const KEY_BAND: std::ops::Range<i64> = -512..512;

    fn map_ops() -> impl Strategy<Value = Vec<MapOp>> {
        let op = prop_oneof![
            (KEY_BAND, any::<i64>()).prop_map(|(k, v)| MapOp::Insert(k, v)),
            KEY_BAND.prop_map(MapOp::Remove),
        ];
        prop::collection::vec(op, 0..2_000)
    }

    proptest! {
        /// The Fx-hashed map must behave exactly like the std map it replaces:
        /// replay a generated insert/overwrite/remove trace against `BTreeMap`
        /// as the oracle, asserting the return value AND length after every
        /// step, then the full contents at the end. proptest shrinks a
        /// divergence to the shortest failing trace.
        #[test]
        fn fxi64map_matches_std_hashmap_semantics(ops in map_ops()) {
            let mut fx: FxI64Map<i64> = FxI64Map::default();
            let mut oracle: BTreeMap<i64, i64> = BTreeMap::new();
            for op in ops {
                match op {
                    MapOp::Insert(k, v) => {
                        prop_assert_eq!(fx.insert(k, v), oracle.insert(k, v));
                    }
                    MapOp::Remove(k) => {
                        prop_assert_eq!(fx.remove(&k), oracle.remove(&k));
                        prop_assert_eq!(fx.get(&k).copied(), oracle.get(&k).copied());
                    }
                }
                prop_assert_eq!(fx.len(), oracle.len());
            }
            let mut fx_pairs: Vec<_> = fx.into_iter().collect();
            fx_pairs.sort_unstable();
            let oracle_pairs: Vec<_> = oracle.into_iter().collect();
            prop_assert_eq!(fx_pairs, oracle_pairs);
        }

        /// The Fx-hashed set against `BTreeSet`, same model-based shape.
        #[test]
        fn fxi64set_matches_std_set_semantics(ops in map_ops()) {
            let mut fx: FxI64Set = FxI64Set::default();
            let mut oracle: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for op in ops {
                let key = match op {
                    MapOp::Insert(k, _) => {
                        prop_assert_eq!(fx.insert(k), oracle.insert(k));
                        k
                    }
                    MapOp::Remove(k) => {
                        prop_assert_eq!(fx.remove(&k), oracle.remove(&k));
                        k
                    }
                };
                prop_assert_eq!(fx.contains(&key), oracle.contains(&key));
                prop_assert_eq!(fx.len(), oracle.len());
            }
        }
    }

    #[test]
    fn fxi64_buildhasher_keys_route_through_hasher_for_arbitrary_type() {
        // BuildHasherDefault must drive FxI64Hasher for any Hash key, exercising
        // the byte fallback (a tuple hashes field-wise, not via write_i64).
        let bh = BuildHasherDefault::<FxI64Hasher>::default();
        let one = |v: &(i64, i64)| bh.hash_one(v);
        assert_eq!(one(&(1, 2)), one(&(1, 2)));
        assert_ne!(one(&(1, 2)), one(&(2, 1)));
    }

    // ---- DerivedStore default methods -----------------------------------

    type Row = (i64, String, String, String);

    /// A mock `DerivedStore` recording the calls the default methods make, so
    /// the streaming/batching CONTRACT (not a backend) is what's under test.
    /// Every non-default method is `unimplemented!()` except the few a default
    /// delegates to.
    /// One recorded `build` call: `(rows, live_notes, col_mod)`.
    type BuildCall = (Vec<Row>, Vec<i64>, i64);

    #[derive(Default)]
    struct MockStore {
        built: Mutex<Vec<BuildCall>>,
        // queries the singular search methods saw, in call order
        substr_seen: Mutex<Vec<String>>,
        fuzzy_seen: Mutex<Vec<String>>,
        // a query string that makes the singular methods return Err
        fail_query: Option<String>,
    }

    macro_rules! unused {
        ($($sig:item)*) => { $($sig)* };
    }

    impl DerivedStore for MockStore {
        fn build(&self, rows: &[Row], live_notes: &[i64], col_mod: i64) -> NativeResult<()> {
            self.built
                .lock()
                .unwrap()
                .push((rows.to_vec(), live_notes.to_vec(), col_mod));
            Ok(())
        }
        fn search_substring(
            &self,
            query: &str,
            _limit: i64,
            _scope: Option<&[i64]>,
            _exclude: &[&str],
        ) -> NativeResult<Option<Vec<LexicalRow>>> {
            self.substr_seen.lock().unwrap().push(query.to_owned());
            if self.fail_query.as_deref() == Some(query) {
                return Err(shrike_error::NativeError::internal("boom"));
            }
            // Encode the query into the row so order is verifiable.
            Ok(Some(vec![(
                query.len() as i64,
                query.to_owned(),
                String::new(),
                None,
            )]))
        }
        fn search_fuzzy(
            &self,
            query: &str,
            _top_k: i64,
            _scope: Option<&[i64]>,
            _exclude: &[&str],
        ) -> NativeResult<Vec<LexicalRow>> {
            self.fuzzy_seen.lock().unwrap().push(query.to_owned());
            if self.fail_query.as_deref() == Some(query) {
                return Err(shrike_error::NativeError::internal("boom"));
            }
            Ok(vec![(
                query.len() as i64,
                query.to_owned(),
                String::new(),
                None,
            )])
        }

        unused! {
            fn ingest(&self, _n: i64, _s: &str, _r: &[(String, String)]) -> NativeResult<()> { unimplemented!() }
            fn ingest_many(&self, _n: &[(i64, Vec<(String, String)>)], _s: &str) -> NativeResult<()> { unimplemented!() }
            fn refresh_derived_snapshots(&self) -> NativeResult<()> { unimplemented!() }
            fn remove(&self, _n: &[i64], _s: Option<&str>) -> NativeResult<()> { unimplemented!() }
            fn count(&self) -> NativeResult<i64> { unimplemented!() }
            fn get_col_mod(&self) -> Option<i64> { unimplemented!() }
            fn set_col_mod(&self, _v: i64) -> NativeResult<()> { unimplemented!() }
            fn meta_get(&self, _k: &str) -> NativeResult<Option<String>> { unimplemented!() }
            fn meta_set(&self, _k: &str, _v: &str) -> NativeResult<()> { unimplemented!() }
            fn refs_for_source(&self, _s: &str) -> NativeResult<Vec<(i64, String)>> { unimplemented!() }
            fn texts_for_source(&self, _s: &str) -> NativeResult<Vec<(i64, String, String)>> { unimplemented!() }
            fn texts_for_source_for_notes(&self, _s: &str, _n: &[i64]) -> NativeResult<Vec<(i64, String, String)>> { unimplemented!() }
            fn mark_gated(&self, _s: &str, _p: &[(i64, String)]) -> NativeResult<()> { unimplemented!() }
            fn gated_refs_for_source(&self, _s: &str) -> NativeResult<Vec<(i64, String)>> { unimplemented!() }
            fn clear_gated(&self, _s: &str) -> NativeResult<()> { unimplemented!() }
            fn put_segments(&self, _n: i64, _s: &str, _r: &str, _j: &str) -> NativeResult<()> { unimplemented!() }
            fn get_segments(&self, _n: i64, _s: &str, _r: &str) -> NativeResult<Option<String>> { unimplemented!() }
            fn match_rows(&self, _e: &str, _l: i64, _t: bool, _sc: Option<&[i64]>, _ex: &[&str]) -> NativeResult<Vec<MatchRow>> { unimplemented!() }
        }
    }

    fn row(id: i64) -> Row {
        (id, "field".into(), format!("r{id}"), format!("t{id}"))
    }

    #[test]
    fn build_streamed_concatenates_chunks_in_order_and_counts() {
        let store = MockStore::default();
        let chunks = vec![
            vec![row(1), row(2)],
            vec![], // an empty chunk in the middle must not end the stream
            vec![row(3)],
            vec![row(4), row(5)],
        ];
        let mut it = chunks.into_iter();
        let mut next = move || it.next().map(Ok);
        let total = store
            .build_streamed(&mut next, &[1, 2, 3, 4, 5], 777)
            .unwrap();
        assert_eq!(total, 5, "returns total rows seen across all chunks");
        let built = store.built.lock().unwrap();
        assert_eq!(built.len(), 1, "exactly one build() for the whole stream");
        let (rows, live, col_mod) = &built[0];
        assert_eq!(col_mod, &777);
        assert_eq!(live, &vec![1, 2, 3, 4, 5]);
        let ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4, 5],
            "row order preserved across chunks"
        );
    }

    #[test]
    fn build_streamed_empty_stream_builds_nothing_and_counts_zero() {
        let store = MockStore::default();
        let mut next = || None;
        let total = store.build_streamed(&mut next, &[], 0).unwrap();
        assert_eq!(total, 0);
        let built = store.built.lock().unwrap();
        assert_eq!(built.len(), 1);
        assert!(built[0].0.is_empty(), "build() called with no rows");
    }

    #[test]
    fn build_streamed_aborts_on_chunk_error_without_building() {
        let store = MockStore::default();
        let mut state = 0;
        let mut next = move || {
            state += 1;
            match state {
                1 => Some(Ok(vec![row(1)])),
                2 => Some(Err(shrike_error::NativeError::internal("read failed"))),
                _ => panic!("must stop pulling after an Err chunk"),
            }
        };
        let result = store.build_streamed(&mut next, &[1], 5);
        assert!(result.is_err(), "a chunk read error aborts the rebuild");
        assert!(
            store.built.lock().unwrap().is_empty(),
            "build() must NOT run on a partial/failed stream — the watermark \
             stays behind for a later drift rebuild to heal"
        );
    }

    #[test]
    fn search_substring_batch_default_is_one_result_per_query_in_order() {
        let store = MockStore::default();
        let queries = ["alpha", "be", "gamma"];
        let out = store
            .search_substring_batch(&queries, 10, None, &[])
            .unwrap();
        assert_eq!(out.len(), 3);
        // Each result carries its query back (encoded by the mock), in order.
        let echoed: Vec<&str> = out
            .iter()
            .map(|r| r.as_ref().unwrap()[0].1.as_str())
            .collect();
        assert_eq!(echoed, queries);
        assert_eq!(*store.substr_seen.lock().unwrap(), queries);
    }

    #[test]
    fn search_fuzzy_batch_default_is_one_result_per_query_in_order() {
        let store = MockStore::default();
        let queries = ["x", "yy", "zzz", "x"];
        let out = store.search_fuzzy_batch(&queries, 5, None, &[]).unwrap();
        let echoed: Vec<&str> = out.iter().map(|r| r[0].1.as_str()).collect();
        assert_eq!(
            echoed, queries,
            "duplicate queries each get their own result"
        );
    }

    #[test]
    fn search_batches_abandon_at_first_failure() {
        // The doc contract: "the batch is abandoned at the first failure". The
        // default's `.collect()` into `NativeResult<Vec<_>>` must short-circuit.
        let store = MockStore {
            fail_query: Some("bad".into()),
            ..Default::default()
        };
        assert!(store
            .search_substring_batch(&["ok", "bad", "after"], 1, None, &[])
            .is_err());
        // The query AFTER the failure is never issued (short-circuit).
        let seen = store.substr_seen.lock().unwrap();
        assert_eq!(*seen, vec!["ok".to_string(), "bad".to_string()]);
        drop(seen);

        let store2 = MockStore {
            fail_query: Some("bad".into()),
            ..Default::default()
        };
        assert!(store2
            .search_fuzzy_batch(&["ok", "bad", "after"], 1, None, &[])
            .is_err());
        assert_eq!(
            *store2.fuzzy_seen.lock().unwrap(),
            vec!["ok".to_string(), "bad".to_string()]
        );
    }
}
