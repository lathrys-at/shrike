//! The native index engine: per-modality usearch sub-indexes behind the
//! frozen `IndexEngine` surface.
//!
//! Implements exactly what `shrike.index_engine.UsearchIndexEngine` does —
//! per-modality sub-index files (`index.usearch` / `index.<m>.usearch`),
//! max-sim-per-note dedup, the empty-index phantom-hit guard (inert under this
//! binding, kept as part of the frozen contract), stale-file deletion on save,
//! and the activation calibration — instance-per-space, no global state.
//!
//! One binding gap shapes this module: usearch's Rust crate (2.25.3) exposes no
//! key-enumeration API (the Python binding's `Index.keys`), so the engine
//! tracks per-modality `{key: vector_count}` maps itself. On `restore` of a
//! Python-written file the map is reconstructed by probing `count(key)` for the
//! caller-provided candidate keys (the orchestrator's `index.hashes.json` —
//! every indexed note id). A multimodal index restored with no candidates fails
//! the restore (one-time drift rebuild — convention 7's worst acceptable
//! outcome); a text-only one loads with keys unknown, which only calibration
//! would need (and text-only never calibrates).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};
use usearch::Index;

use crate::new_index;

/// Mirrors `shrike.index_engine.SEARCH_OVERFETCH`.
const SEARCH_OVERFETCH: usize = 4;

// Canonical docs live in shrike-store; re-exported here so existing import
// paths keep working.
pub use shrike_store::{ActivationStats, ModalityRanking};

struct Sub {
    /// `Arc` so [`MultiModalIndex::save`] can clone a cheap handle under the
    /// state lock and serialize+write it *outside* the lock. usearch's `Index`
    /// is `Send + Sync` with its own internal locking; the `save_mutation` guard
    /// excludes the in-place mutators
    /// ([`add`](MultiModalIndex::add)/[`remove`](MultiModalIndex::remove)) for
    /// the duration of the serialize, while leaving searches concurrent — so a
    /// save and a concurrent search share the same `Index` without the state
    /// mutex held across the file write. (`add`/`remove`/`reserve` are also
    /// `&self` on usearch's `Index`, so `&self` alone does not imply read-only —
    /// the mutual exclusion comes from the guard, not the receiver.)
    index: Arc<Index>,
    /// key → vector count (the Rust binding has no key enumeration).
    counts: BTreeMap<i64, u32>,
    /// False when restored without candidates (text-only path) — keys unknown.
    keys_known: bool,
}

impl Sub {
    fn new(index: Index) -> Self {
        Self {
            index: Arc::new(index),
            counts: BTreeMap::new(),
            keys_known: true,
        }
    }
}

struct State {
    indexes: BTreeMap<String, Sub>,
    ndim: Option<usize>,
}

/// The per-modality USearch vector store: a text sub-index plus one
/// sub-index per non-text modality, each ranked separately.
pub struct MultiModalIndex {
    /// Known modalities in load order (TEXT first — it keeps the original
    /// index.usearch filename). Mirrors `_INDEX_MODALITIES`.
    modalities: Vec<String>,
    text: String,
    state: Mutex<State>,
    /// Serializes [`save`](Self::save) against the in-place byte mutators
    /// ([`add`](Self::add)/[`remove`](Self::remove)) **without** gating
    /// [`search_by_modality`](Self::search_by_modality). `save` clones a
    /// cheap `Arc` to each sub-index under the `state` lock, releases that lock,
    /// then serializes+writes the file while holding only this guard — so a
    /// concurrent search (which takes only the `state` lock) is never blocked
    /// behind the file write, while a concurrent `add`/`remove` (which can't
    /// mutate an `Index` usearch is mid-serialization on) is excluded. Always
    /// taken *before* the `state` lock to keep one lock order.
    save_mutation: Mutex<()>,
}

fn file_name(text_modality: &str, modality: &str) -> String {
    if modality == text_modality {
        "index.usearch".to_string()
    } else {
        format!("index.{modality}.usearch")
    }
}

/// `<path>.tmp` — the same-directory staging name for an atomic save.
fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

fn ensure_capacity(index: &Index, extra: usize) -> NativeResult<()> {
    let needed = index.size() + extra;
    if needed > index.capacity() {
        // Grow 1.5× (or straight to the request when larger): amortizes like
        // next_power_of_two doubling without over-allocating up to
        // ~2× a large collection's footprint on the last step.
        let target = needed.max(index.capacity() + index.capacity() / 2).max(64);
        index
            .reserve(target)
            .context(ErrorKind::Internal, "usearch reserve")?;
    }
    Ok(())
}

impl MultiModalIndex {
    /// `modalities[0]` is the text modality (mandatory, owns index.usearch).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `modalities` is empty.
    pub fn new(modalities: Vec<String>) -> NativeResult<Self> {
        let text = modalities
            .first()
            .cloned()
            .ok_or_else(|| NativeError::invalid_input("at least one modality required"))?;
        Ok(Self {
            modalities,
            text,
            state: Mutex::new(State {
                indexes: BTreeMap::new(),
                ndim: None,
            }),
            save_mutation: Mutex::new(()),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("index state lock poisoned")
    }

    fn lock_save_mutation(&self) -> std::sync::MutexGuard<'_, ()> {
        self.save_mutation
            .lock()
            .expect("index save-mutation guard poisoned")
    }

    /// Total vectors across all sub-indexes.
    pub fn size(&self) -> usize {
        self.lock().indexes.values().map(|s| s.index.size()).sum()
    }

    /// The index dimensionality, or `None` before any vectors land.
    pub fn ndim(&self) -> Option<usize> {
        self.lock().ndim
    }

    /// Per-modality `(name, vector count)`.
    pub fn modality_sizes(&self) -> Vec<(String, usize)> {
        self.lock()
            .indexes
            .iter()
            .map(|(m, s)| (m.clone(), s.index.size()))
            .collect()
    }

    /// Per-modality `(name, size, ndim)`: each sub-index reports its OWN
    /// dimensionality (the text and image sub-indexes differ under CLIP). `ndim`
    /// is `None` for an empty sub-index (usearch reports 0 dimensions before the
    /// first vector sets the width; surface that as "unknown", not 0).
    pub fn modality_stats(&self) -> Vec<(String, usize, Option<usize>)> {
        self.lock()
            .indexes
            .iter()
            .map(|(m, s)| {
                let size = s.index.size();
                let dim = s.index.dimensions();
                (m.clone(), size, (dim > 0).then_some(dim))
            })
            .collect()
    }

    /// The names of the loaded modalities.
    pub fn modality_names(&self) -> Vec<String> {
        self.lock().indexes.keys().cloned().collect()
    }

    /// Create/verify `modality`'s sub-index at `ndim`: idempotent at the same
    /// width, an error at a conflicting one.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `modality` already has a sub-index
    /// at a different dimension (a silent no-op would let the caller believe the
    /// modality is the width it asked for while it is another, so the next
    /// add/search at the assumed width mismatches far from the cause), or an
    /// error if a new sub-index cannot be created. usearch reports a sub-index's
    /// configured width via `dimensions()` from creation, before any vector.
    pub fn ensure(&self, modality: &str, ndim: usize) -> NativeResult<()> {
        let mut state = self.lock();
        match state.indexes.get(modality) {
            Some(sub) => {
                let existing = sub.index.dimensions();
                if existing != ndim {
                    return Err(NativeError::invalid_input(format!(
                        "modality '{modality}' already exists at dimension {existing}, \
                         cannot ensure at {ndim}"
                    )));
                }
            }
            None => {
                state
                    .indexes
                    .insert(modality.to_string(), Sub::new(new_index(ndim)?));
                state.ndim = Some(ndim);
            }
        }
        Ok(())
    }

    /// Drop every sub-index.
    pub fn clear(&self) {
        let mut state = self.lock();
        state.indexes.clear();
        state.ndim = None;
    }

    /// Drop one modality's sub-index.
    pub fn drop_modality(&self, modality: &str) {
        self.lock().indexes.remove(modality);
    }

    /// Load the per-modality sub-index files under `dir`.
    ///
    /// `candidates` are the note ids that may be in the index (the
    /// orchestrator's hashes sidecar) — used to reconstruct the per-key count
    /// maps the Rust binding can't enumerate. Returns false (and clears) on a
    /// corrupt present file, a candidate set that doesn't account for every
    /// stored vector, or a multimodal index restored without candidates — all
    /// of which the caller answers with the standard drift rebuild.
    pub fn restore(&self, dir: &str, candidates: Option<&[i64]>) -> bool {
        let base = Path::new(dir);
        let mut loaded: BTreeMap<String, Sub> = BTreeMap::new();
        for modality in &self.modalities {
            let file: PathBuf = base.join(file_name(&self.text, modality));
            if !file.is_file() {
                continue;
            }
            // The placeholder dimension is adopted from the file on load.
            let Ok(index) = new_index(8) else {
                self.clear();
                return false;
            };
            if index.load(&file.to_string_lossy()).is_err() {
                self.clear();
                return false;
            }

            let mut sub = Sub::new(index);
            match candidates {
                Some(keys) => {
                    let mut accounted = 0usize;
                    for key in keys {
                        let n = sub.index.count(*key as u64);
                        if n > 0 {
                            sub.counts.insert(*key, n as u32);
                            accounted += n;
                        }
                    }
                    if accounted != sub.index.size() {
                        // Vectors exist under keys the candidates don't name —
                        // the key map would be wrong; rebuild instead.
                        self.clear();
                        return false;
                    }
                }
                None if modality == &self.text => {
                    // Text-only restore without candidates: keys unknown, which
                    // only calibration would need (and text-only never does).
                    sub.keys_known = false;
                }
                None => {
                    self.clear();
                    return false;
                }
            }
            loaded.insert(modality.clone(), sub);
        }
        let mut state = self.lock();
        state.ndim = loaded.get(&self.text).map(|s| s.index.dimensions());
        state.indexes = loaded;
        true
    }

    /// Persist every loaded sub-index under `dir`; delete a known modality's
    /// file when that modality is no longer loaded (no phantom reload).
    ///
    /// Each file lands atomically: usearch's `save` truncates in place, so it
    /// writes a same-directory `.tmp` first and a rename replaces the
    /// canonical file — a crash mid-save leaves the old file complete, never
    /// a truncation.
    ///
    /// **The state lock is NOT held across the file write (the
    /// "never hold a lock across file writes" rule one layer below
    /// `IndexOrchestrator::save`).** Under the `state` lock only a cheap `Arc`
    /// handle to each sub-index (plus the loaded-modality set) is snapshotted;
    /// the `state` guard is dropped and every `index.save(tmp)`+rename runs
    /// holding only the dedicated `save_mutation` guard, so a concurrent
    /// [`search_by_modality`](Self::search_by_modality) (which takes only the
    /// `state` lock) is never serialized behind the whole multi-file usearch
    /// write.
    ///
    /// `add`/`remove` ARE excluded during the write: `save` serializes the whole
    /// structure while those mutate it in place, so the `save_mutation` guard
    /// serializes save against them (it does NOT gate searches). usearch
    /// documents concurrent search+update but is silent on save-vs-mutate, so
    /// the guard is the conservative, self-evidently-correct exclusion rather
    /// than a reliance on an undocumented upstream guarantee. Concurrent
    /// save+search is read+read on the shared `Index` and safe.
    ///
    /// # Errors
    ///
    /// Returns an error if creating `dir` or writing/renaming a sub-index file fails.
    pub fn save(&self, dir: &str) -> NativeResult<()> {
        // Exclude the in-place byte mutators for the whole serialize+write, but
        // not searches (which take only the `state` lock). Taken before the
        // `state` lock to keep a single lock order.
        let _mutation = self.lock_save_mutation();
        let base = Path::new(dir);
        std::fs::create_dir_all(base)
            .with_context(ErrorKind::Internal, || format!("mkdir {dir}"))?;
        // Snapshot under the state lock, then release it before any I/O.
        let (to_write, loaded): (Vec<(String, Arc<Index>)>, std::collections::HashSet<String>) = {
            let state = self.lock();
            let to_write = state
                .indexes
                .iter()
                .map(|(modality, sub)| (modality.clone(), Arc::clone(&sub.index)))
                .collect();
            let loaded = state.indexes.keys().cloned().collect();
            (to_write, loaded)
        };
        for (modality, index) in &to_write {
            let file = base.join(file_name(&self.text, modality));
            let tmp = tmp_path(&file);
            index
                .save(&tmp.to_string_lossy())
                .context(ErrorKind::Internal, "usearch save")?;
            std::fs::rename(&tmp, &file).context(ErrorKind::Internal, "usearch save rename")?;
        }
        for modality in &self.modalities {
            if !loaded.contains(modality) {
                let stale = base.join(file_name(&self.text, modality));
                if stale.exists() {
                    let _ = std::fs::remove_file(&stale);
                }
                // A crashed save may have stranded a staging file — sweep it
                // (best-effort; restore reads exact names and ignores it).
                let tmp = tmp_path(&stale);
                if tmp.exists() {
                    let _ = std::fs::remove_file(&tmp);
                }
            }
        }
        Ok(())
    }

    /// Pure add of vectors under i64 keys (replace semantics are the caller's,
    /// via `remove`). Keys may repeat (a note's several images).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `keys`/`vectors` lengths disagree,
    /// or an error if the backend rejects a vector (e.g. dimension mismatch).
    ///
    /// # Panics
    ///
    /// Panics if the internal state mutex is poisoned (a prior holder panicked).
    pub fn add(&self, modality: &str, keys: &[i64], vectors: &[Vec<f32>]) -> NativeResult<()> {
        if keys.len() != vectors.len() {
            return Err(NativeError::invalid_input(format!(
                "keys ({}) and vectors ({}) must align",
                keys.len(),
                vectors.len()
            )));
        }
        if keys.is_empty() {
            return Ok(());
        }
        let ndim = vectors[0].len();
        // Exclude a concurrent save's serialization read of this Index.
        let _mutation = self.lock_save_mutation();
        let mut state = self.lock();
        if !state.indexes.contains_key(modality) {
            state
                .indexes
                .insert(modality.to_string(), Sub::new(new_index(ndim)?));
            state.ndim = Some(ndim);
        }
        let sub = state.indexes.get_mut(modality).expect("just ensured");
        ensure_capacity(&sub.index, keys.len())?;
        for (key, vector) in keys.iter().zip(vectors) {
            sub.index
                .add(*key as u64, vector)
                .context(ErrorKind::InvalidInput, "usearch add")?;
            *sub.counts.entry(*key).or_insert(0) += 1;
        }
        Ok(())
    }

    /// Remove the keys' vectors from every sub-index; returns the count removed
    /// from the *text* sub-index (one text vector per note — the note count).
    ///
    /// # Errors
    ///
    /// Returns an error if the backend rejects a removal.
    pub fn remove(&self, keys: &[i64]) -> NativeResult<usize> {
        // Exclude a concurrent save's serialization read of this Index.
        let _mutation = self.lock_save_mutation();
        let mut state = self.lock();
        if keys.is_empty() || state.indexes.is_empty() {
            return Ok(0);
        }
        let mut removed = 0usize;
        for (modality, sub) in state.indexes.iter_mut() {
            let mut count = 0usize;
            for key in keys {
                let n = sub
                    .index
                    .remove(*key as u64)
                    .context(ErrorKind::Internal, "usearch remove")?;
                count += n;
                if n > 0 {
                    sub.counts.remove(key);
                }
            }
            if modality == &self.text {
                removed = count;
            }
        }
        Ok(removed)
    }

    /// Per-query, per-modality max-sim-per-note rankings (best-first, deduped,
    /// truncated to k). `modalities = None` searches every loaded sub-index.
    ///
    /// # Errors
    ///
    /// Returns an error if a backend search fails.
    pub fn search_by_modality(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        modalities: Option<&[String]>,
        scope: Option<&[i64]>,
    ) -> NativeResult<Vec<BTreeMap<String, ModalityRanking>>> {
        let span = tracing::debug_span!("index.search", queries = queries.len(), k);
        let _enter = span.enter();
        let state = self.lock();
        // A deck/tag scope rides the index walk as a per-candidate predicate (the
        // keys ARE note ids), so a scoped search returns in-scope neighbours
        // directly instead of over-fetching the whole index and dropping the rest.
        // The FxHash set keeps the per-candidate `contains` off SipHash.
        let scope_set: Option<shrike_store::FxI64Set> =
            scope.map(|ids| ids.iter().copied().collect());
        let mut out: Vec<BTreeMap<String, ModalityRanking>> =
            (0..queries.len()).map(|_| BTreeMap::new()).collect();
        for (modality, sub) in &state.indexes {
            if let Some(filter) = modalities {
                if !filter.iter().any(|m| m == modality) {
                    continue;
                }
            }
            // Also subsumes the Python binding's empty-index phantom-(0, 0)
            // guard: an empty sub-index never reaches the hit loop, so no
            // per-hit phantom check is needed (frozen-contract parity).
            if sub.index.size() == 0 {
                continue;
            }
            // The over-fetch covers the per-note dedup of a MULTI-vector modality (a
            // note's several images deduped to k distinct notes); a single-vector
            // modality (`size == note count`, e.g. text) has no dupes, so it fetches
            // exactly k. Then clamp to what THIS sub-index can return: usearch's
            // `search(query, count)` reserves AND zero-fills `count` result slots up
            // front (lib.cpp search_) before the graph walk, so an unclamped `fetch`
            // allocates work proportional to `count`, not to the hits — a `limit=0`
            // (k = size) query would be a ~400k-slot zero-fill at 100k notes. A
            // sub-index can never yield more than `size()` hits, so the clamp is
            // lossless; skipping the over-fetch where there is nothing to dedup is the
            // rest of the win.
            let multi_vector = sub.index.size() != sub.counts.len();
            let base = if multi_vector {
                (k * SEARCH_OVERFETCH).max(k)
            } else {
                k
            };
            let fetch = base.min(sub.index.size());
            for (qi, query) in queries.iter().enumerate() {
                let hits = match &scope_set {
                    Some(set) => sub
                        .index
                        .filtered_search(query, fetch, |key| set.contains(&(key as i64)))
                        .context(ErrorKind::Internal, "usearch filtered_search")?,
                    None => sub
                        .index
                        .search(query, fetch)
                        .context(ErrorKind::Internal, "usearch search")?,
                };
                let mut keys: Vec<i64> = Vec::new();
                let mut distances: Vec<f32> = Vec::new();
                let mut seen = shrike_store::FxI64Set::default();
                for (key, dist) in hits.keys.iter().zip(hits.distances.iter()) {
                    let nid = *key as i64;
                    if seen.contains(&nid) {
                        continue;
                    }
                    seen.insert(nid);
                    keys.push(nid);
                    distances.push(*dist);
                    if keys.len() >= k {
                        break;
                    }
                }
                if !keys.is_empty() {
                    out[qi].insert(modality.clone(), (keys, distances));
                }
            }
        }
        Ok(out)
    }

    /// Whether the text sub-index holds a vector for `key`.
    pub fn contains(&self, key: i64) -> bool {
        self.modality_contains(&self.text, key)
    }

    /// Distinct note ids in the text sub-index, sorted.
    pub fn keys(&self) -> Vec<i64> {
        let state = self.lock();
        state
            .indexes
            .get(&self.text)
            .map(|s| s.counts.keys().copied().collect())
            .unwrap_or_default()
    }

    /// A note's stored text vector(s), 2D row-major (multi keys give >1 row).
    pub fn get(&self, key: i64) -> Option<Vec<Vec<f32>>> {
        self.modality_get(&self.text, key)
    }

    /// Whether `modality`'s sub-index holds a vector for `key`.
    pub fn modality_contains(&self, modality: &str, key: i64) -> bool {
        let state = self.lock();
        state
            .indexes
            .get(modality)
            .map(|s| s.index.contains(key as u64))
            .unwrap_or(false)
    }

    /// All keys in a modality's sub-index, one entry per *vector* (multi keys
    /// repeat) — mirroring the Python binding's `Index.keys` view.
    pub fn modality_keys(&self, modality: &str) -> Vec<i64> {
        let state = self.lock();
        let Some(sub) = state.indexes.get(modality) else {
            return Vec::new();
        };
        let mut keys: Vec<i64> = Vec::new();
        for (key, count) in &sub.counts {
            keys.extend(std::iter::repeat_n(*key, *count as usize));
        }
        keys
    }

    /// Dot-score each key's FIRST stored vector against `query` in one lock
    /// hold: the tag expansion otherwise pays a mutex acquire plus a full
    /// per-vector heap clone per member via `modality_get`. Each key's vectors
    /// are read into one reused buffer and only `(key, dot)` pairs come back;
    /// missing keys are skipped, a query/ndim mismatch returns empty. Callers
    /// bound `keys` (the expansion's member ceiling), so the single hold stays
    /// in low-millisecond territory — unlike calibration, which holds per-search
    /// because its total runs much longer.
    pub fn dot_scores(&self, modality: &str, keys: &[i64], query: &[f32]) -> Vec<(i64, f32)> {
        let state = self.lock();
        let Some(sub) = state.indexes.get(modality) else {
            return Vec::new();
        };
        let ndim = sub.index.dimensions();
        if ndim != query.len() {
            return Vec::new();
        }
        let mut buf: Vec<f32> = Vec::new();
        keys.iter()
            .filter_map(|key| {
                let count = sub.index.count(*key as u64);
                if count == 0 {
                    return None;
                }
                buf.resize(count * ndim, 0.0);
                let copied = sub.index.get(*key as u64, &mut buf).ok()?;
                if copied == 0 {
                    return None;
                }
                let dot = buf[..ndim].iter().zip(query).map(|(x, y)| x * y).sum();
                Some((*key, dot))
            })
            .collect()
    }

    /// A key's vector(s) in `modality`, 2D row-major, or `None`.
    pub fn modality_get(&self, modality: &str, key: i64) -> Option<Vec<Vec<f32>>> {
        let state = self.lock();
        let sub = state.indexes.get(modality)?;
        Self::vectors_of(&sub.index, key as u64)
    }

    fn vectors_of(index: &Index, key: u64) -> Option<Vec<Vec<f32>>> {
        let count = index.count(key);
        if count == 0 {
            return None;
        }
        let ndim = index.dimensions();
        let mut buf = vec![0.0f32; count * ndim];
        let copied = index.get(key, &mut buf).ok()?;
        // count() sized the buffer; a short read would mean the index
        // mutated under us (the lock forbids it) or a usearch bug.
        debug_assert_eq!(
            copied, count,
            "usearch get returned fewer vectors than count"
        );
        Some(
            (0..copied)
                .map(|i| buf[i * ndim..(i + 1) * ndim].to_vec())
                .collect(),
        )
    }

    /// Per-(non-text-)modality best-match stats for the activation gate:
    /// sample stored text vectors as pseudo-queries (deterministic),
    /// search each non-text modality, record the best non-self match.
    ///
    /// Lock discipline: the sample keys and their query vectors are snapshotted
    /// under ONE short hold, then each search takes its own brief hold — so a
    /// calibration over hundreds of samples never stalls the whole index behind
    /// a single multi-hundred-millisecond lock. Fully lock-free searching is
    /// deliberately NOT used: writers may interleave, and the engine's safety
    /// model trusts usearch's `Sync` only for read-vs-read — every usearch call
    /// stays under the mutex. The stats are statistical, so an interleaved write
    /// skewing one sample is fine.
    ///
    /// # Errors
    ///
    /// Returns an error if a sampling search fails.
    pub fn calibrate_activation(
        &self,
        sample_size: usize,
        k: usize,
        min_count: usize,
    ) -> NativeResult<ActivationStats> {
        // Phase 1 — one short hold: pick the sample and copy out its query
        // vectors; list the live non-text modalities.
        let (queries, modalities) = {
            let state = self.lock();
            let Some(text_sub) = state.indexes.get(&self.text) else {
                return Ok(Vec::new());
            };
            if text_sub.index.size() == 0 {
                return Ok(Vec::new());
            }
            let modalities: Vec<String> = state
                .indexes
                .iter()
                .filter(|(m, s)| *m != &self.text && s.index.size() > 0)
                .map(|(m, _)| m.clone())
                .collect();
            if modalities.is_empty() {
                return Ok(Vec::new());
            }

            // Deterministic sample: a PARTIAL LCG Fisher-Yates over the sorted
            // keys — only the first `sample_size` slots are drawn, so a large
            // collection isn't fully shuffled to take 256. Stable across runs of
            // this engine; deliberately NOT numpy's sampler — the stats are
            // statistical, never byte-pinned.
            let mut keys: Vec<i64> = text_sub.counts.keys().copied().collect();
            let n = keys.len();
            let take = sample_size.min(n);
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
            for i in 0..take {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = i + ((rng >> 33) as usize % (n - i));
                keys.swap(i, j);
            }
            keys.truncate(take);

            let queries: Vec<(i64, Vec<f32>)> = keys
                .iter()
                .filter_map(|key| {
                    Self::vectors_of(&text_sub.index, *key as u64)
                        .and_then(|v| v.into_iter().next().map(|q| (*key, q)))
                })
                .collect();
            (queries, modalities)
        };

        // Phase 2 — one brief hold per search; writers interleave between
        // samples instead of queueing behind the whole calibration.
        let mut stats: ActivationStats = Vec::new();
        for modality in &modalities {
            let mut best_sims: Vec<f64> = Vec::new();
            for (key, qvec) in &queries {
                let best = {
                    let state = self.lock();
                    let Some(sub) = state.indexes.get(modality) else {
                        break; // modality vanished mid-run (clear/rebuild)
                    };
                    if sub.index.size() == 0 {
                        break;
                    }
                    // k > 1 so a pseudo-query whose own image is the nearest
                    // hit still has a non-self hit to record.
                    let hits = sub
                        .index
                        .search(qvec, k.min(sub.index.size()))
                        .context(ErrorKind::Internal, "usearch search")?;
                    hits.keys
                        .iter()
                        .zip(hits.distances.iter())
                        .find(|(hk, _)| **hk as i64 != *key)
                        .map(|(_, dist)| 1.0 - *dist as f64)
                };
                if let Some(sim) = best {
                    best_sims.push(sim);
                }
            }
            if best_sims.len() >= min_count {
                let count = best_sims.len() as f64;
                let mean = best_sims.iter().sum::<f64>() / count;
                let var = best_sims
                    .iter()
                    .map(|s| (s - mean) * (s - mean))
                    .sum::<f64>()
                    / count;
                stats.push((modality.clone(), count, mean, var.sqrt()));
            }
        }
        Ok(stats)
    }
}

/// The store contract: every method forwards to the inherent impl, so
/// the concrete engine keeps its full API while the kernel consumes
/// `Arc<dyn VectorIndex>`.
impl shrike_store::VectorIndex for MultiModalIndex {
    fn size(&self) -> usize {
        Self::size(self)
    }
    fn ndim(&self) -> Option<usize> {
        Self::ndim(self)
    }
    fn modality_sizes(&self) -> Vec<(String, usize)> {
        Self::modality_sizes(self)
    }
    fn modality_stats(&self) -> Vec<(String, usize, Option<usize>)> {
        Self::modality_stats(self)
    }
    fn modality_names(&self) -> Vec<String> {
        Self::modality_names(self)
    }
    fn ensure(&self, modality: &str, ndim: usize) -> NativeResult<()> {
        Self::ensure(self, modality, ndim)
    }
    fn clear(&self) {
        Self::clear(self)
    }
    fn drop_modality(&self, modality: &str) {
        Self::drop_modality(self, modality)
    }
    fn restore(&self, dir: &str, candidates: Option<&[i64]>) -> bool {
        Self::restore(self, dir, candidates)
    }
    fn save(&self, dir: &str) -> NativeResult<()> {
        Self::save(self, dir)
    }
    fn add(&self, modality: &str, keys: &[i64], vectors: &[Vec<f32>]) -> NativeResult<()> {
        Self::add(self, modality, keys, vectors)
    }
    fn remove(&self, keys: &[i64]) -> NativeResult<usize> {
        Self::remove(self, keys)
    }
    fn search_by_modality(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        modalities: Option<&[String]>,
        scope: Option<&[i64]>,
    ) -> NativeResult<Vec<std::collections::BTreeMap<String, ModalityRanking>>> {
        Self::search_by_modality(self, queries, k, modalities, scope)
    }
    fn contains(&self, key: i64) -> bool {
        Self::contains(self, key)
    }
    fn keys(&self) -> Vec<i64> {
        Self::keys(self)
    }
    fn get(&self, key: i64) -> Option<Vec<Vec<f32>>> {
        Self::get(self, key)
    }
    fn modality_contains(&self, modality: &str, key: i64) -> bool {
        Self::modality_contains(self, modality, key)
    }
    fn modality_keys(&self, modality: &str) -> Vec<i64> {
        Self::modality_keys(self, modality)
    }
    fn modality_get(&self, modality: &str, key: i64) -> Option<Vec<Vec<f32>>> {
        Self::modality_get(self, modality, key)
    }
    fn dot_scores(&self, modality: &str, keys: &[i64], query: &[f32]) -> Vec<(i64, f32)> {
        Self::dot_scores(self, modality, keys, query)
    }
    fn calibrate_activation(
        &self,
        sample_size: usize,
        k: usize,
        min_count: usize,
    ) -> NativeResult<ActivationStats> {
        Self::calibrate_activation(self, sample_size, k, min_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn unit(seed: u64, ndim: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(99);
        let mut v: Vec<f32> = (0..ndim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    fn engine() -> MultiModalIndex {
        MultiModalIndex::new(vec!["text".into(), "image".into()]).unwrap()
    }

    #[test]
    fn add_remove_contains_keys() {
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        e.add("image", &[1, 1], &[unit(3, 8), unit(4, 8)]).unwrap();
        assert_eq!(e.size(), 4);
        assert_eq!(e.ndim(), Some(8));
        assert!(e.contains(1));
        assert_eq!(e.keys(), vec![1, 2]);
        assert_eq!(e.modality_keys("image"), vec![1, 1]);
        assert_eq!(e.remove(&[1, 99]).unwrap(), 1); // text count, images incidental
        assert_eq!(e.size(), 1);
    }

    #[test]
    fn search_dedups_multi_to_best_hit() {
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        e.add("image", &[1, 1], &[unit(1, 8), unit(9, 8)]).unwrap();
        let out = e.search_by_modality(&[unit(1, 8)], 5, None, None).unwrap();
        let (keys, dists) = &out[0]["image"];
        assert_eq!(keys, &vec![1]);
        assert!(dists[0].abs() < 1e-5);
    }

    #[test]
    fn search_k_far_exceeding_size_is_lossless_and_bounded() {
        // The `limit=0` over-fetch path: the caller passes
        // `k = index.size`, the engine over-fetches `SEARCH_OVERFETCH * k`, and
        // the per-sub-index clamp keeps usearch's `search(query, count)` from
        // reserving+zero-filling a buffer far larger than the sub-index. A `k`
        // (and thus `fetch`) orders of magnitude past the 3 stored text vectors
        // must still return exactly those 3, best-first, without error or hang.
        let e = engine();
        e.add("text", &[1, 2, 3], &[unit(1, 8), unit(2, 8), unit(3, 8)])
            .unwrap();
        let out = e
            .search_by_modality(&[unit(1, 8)], 1_000_000, Some(&["text".to_string()]), None)
            .unwrap();
        let (keys, _dists) = &out[0]["text"];
        assert_eq!(keys.len(), 3, "all stored vectors returned, never more");
        assert_eq!(keys[0], 1, "the self-hit ranks first");
    }

    #[test]
    fn search_by_modality_scope_predicate_returns_only_in_scope_keys() {
        // A scope set rides into the walk as a `filtered_search` predicate (the keys
        // ARE note ids), so the result is the in-scope nearest neighbours directly —
        // never an out-of-scope key, no over-fetch-then-drop.
        let e = engine();
        e.add(
            "text",
            &[1, 2, 3, 4],
            &[unit(1, 8), unit(2, 8), unit(3, 8), unit(4, 8)],
        )
        .unwrap();
        let scope = vec![2i64, 4];
        let out = e
            .search_by_modality(&[unit(1, 8)], 10, Some(&["text".to_string()]), Some(&scope))
            .unwrap();
        let (keys, _) = &out[0]["text"];
        let got: std::collections::HashSet<i64> = keys.iter().copied().collect();
        assert_eq!(
            got,
            scope
                .iter()
                .copied()
                .collect::<std::collections::HashSet<i64>>(),
            "filtered_search must return exactly the in-scope keys, got {keys:?}"
        );
    }

    #[test]
    fn restore_with_candidates_round_trip() {
        let dir = std::env::temp_dir().join(format!("shrike-engine-rt-{}", std::process::id()));
        let dirs = dir.to_str().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        e.add("image", &[1, 1], &[unit(3, 8), unit(4, 8)]).unwrap();
        e.save(dirs).unwrap();

        let fresh = engine();
        assert!(fresh.restore(dirs, Some(&[1, 2])));
        assert_eq!(fresh.size(), 4);
        assert_eq!(fresh.keys(), vec![1, 2]);
        assert_eq!(fresh.modality_keys("image"), vec![1, 1]);

        // Multimodal restore without candidates → rebuild signal.
        let blind = engine();
        assert!(!blind.restore(dirs, None));
        // Candidates that don't account for every vector → rebuild signal.
        let partial = engine();
        assert!(!partial.restore(dirs, Some(&[1])));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_deletes_stale_modality_file() {
        let dir = std::env::temp_dir().join(format!("shrike-engine-stale-{}", std::process::id()));
        let dirs = dir.to_str().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        e.add("image", &[1], &[unit(2, 8)]).unwrap();
        e.save(dirs).unwrap();
        assert!(dir.join("index.image.usearch").exists());
        e.clear();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        // A crashed save's stranded staging file is swept with the stale file.
        std::fs::write(dir.join("index.image.usearch.tmp"), b"stranded").unwrap();
        e.save(dirs).unwrap();
        assert!(!dir.join("index.image.usearch").exists());
        assert!(!dir.join("index.image.usearch.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_replaces_torn_files_atomically() {
        let dir = std::env::temp_dir().join(format!("shrike-engine-atomic-{}", std::process::id()));
        let dirs = dir.to_str().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        e.add("image", &[1], &[unit(2, 8)]).unwrap();
        e.save(dirs).unwrap();
        // A completed save leaves no staging files behind.
        assert!(!dir.join("index.usearch.tmp").exists());
        assert!(!dir.join("index.image.usearch.tmp").exists());

        // Tear the canonical file (a truncate-in-place crash) and strand a
        // staging file (a crashed atomic save) — the next save must replace
        // both wholesale, not append or trip over them.
        std::fs::write(dir.join("index.usearch"), b"torn").unwrap();
        std::fs::write(dir.join("index.usearch.tmp"), b"stale").unwrap();
        e.add("text", &[2], &[unit(3, 8)]).unwrap();
        e.save(dirs).unwrap();
        assert!(!dir.join("index.usearch.tmp").exists());
        let fresh = engine();
        assert!(fresh.restore(dirs, Some(&[1, 2])));
        assert_eq!(fresh.size(), 3);
        assert_eq!(fresh.keys(), vec![1, 2]);

        // A stranded staging file alone never disturbs a restore (exact
        // canonical names only — nothing globs `.tmp`).
        std::fs::write(dir.join("index.usearch.tmp"), b"stale").unwrap();
        let again = engine();
        assert!(again.restore(dirs, Some(&[1, 2])));
        assert_eq!(again.size(), 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn calibration_produces_stats_with_enough_pairs() {
        let e = engine();
        let n = 40i64;
        let keys: Vec<i64> = (1..=n).collect();
        let tvecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, 8)).collect();
        e.add("text", &keys, &tvecs).unwrap();
        let ivecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64 + 1000, 8)).collect();
        e.add("image", &keys, &ivecs).unwrap();
        let stats = e.calibrate_activation(256, 5, 30).unwrap();
        assert_eq!(stats.len(), 1);
        let (modality, count, _mean, std) = &stats[0];
        assert_eq!(modality, "image");
        assert!(*count >= 30.0);
        assert!(*std >= 0.0);
    }

    #[test]
    fn dot_scores_matches_get_and_skips_misses() {
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        let q = unit(7, 8);
        let scores = e.dot_scores("text", &[1, 2, 99], &q);
        assert_eq!(scores.iter().map(|(k, _)| *k).collect::<Vec<_>>(), [1, 2]);
        for (k, s) in &scores {
            let v = &e.modality_get("text", *k).unwrap()[0];
            let expect: f32 = v.iter().zip(&q).map(|(x, y)| x * y).sum();
            assert!((s - expect).abs() < 1e-6);
        }
        // A multi-vector key scores its FIRST vector (the expansion's
        // existing semantics).
        e.add("text", &[3, 3], &[unit(8, 8), unit(9, 8)]).unwrap();
        let first = &e.modality_get("text", 3).unwrap()[0];
        let expect: f32 = first.iter().zip(&q).map(|(x, y)| x * y).sum();
        let scored = e.dot_scores("text", &[3], &q);
        assert!((scored[0].1 - expect).abs() < 1e-6);
        // Dim mismatch and unknown modality degrade to empty, never panic.
        assert!(e.dot_scores("text", &[1], &[1.0]).is_empty());
        assert!(e.dot_scores("nope", &[1], &q).is_empty());
    }

    #[test]
    fn calibration_survives_self_hit_heavy_samples() {
        // Every note's image vector IS its text vector, so each
        // pseudo-query's nearest image hit is its own note. At k=1 the
        // self-hit exclusion then records nothing — the sample silently
        // shrinks to zero and the gate disables. k=2 (the kernel's CALIB_K)
        // must still calibrate the full sample.
        let e = engine();
        let keys: Vec<i64> = (1..=40).collect();
        let vecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, 8)).collect();
        e.add("text", &keys, &vecs).unwrap();
        e.add("image", &keys, &vecs).unwrap();
        assert!(e.calibrate_activation(256, 1, 30).unwrap().is_empty());
        let stats = e.calibrate_activation(256, 2, 30).unwrap();
        assert_eq!(stats.len(), 1);
        let (modality, count, _mean, _std) = &stats[0];
        assert_eq!(modality, "image");
        assert_eq!(*count, 40.0);
    }

    /// A unit-normalized `ndim`-vector strategy: draws each component from
    /// `[-1, 1)` and normalizes. usearch's IP/cosine metrics need unit vectors;
    /// a generated raw `Vec<f32>` would otherwise have an arbitrary magnitude.
    fn unit_vec(ndim: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-1.0f32..1.0, ndim).prop_map(|mut v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            // A zero vector can't be unit-normalized; nudge it (vanishingly rare).
            let norm = if norm > 0.0 { norm } else { 1.0 };
            for x in &mut v {
                *x /= norm;
            }
            v
        })
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static C: AtomicU64 = AtomicU64::new(0);
        let d = std::env::temp_dir().join(format!(
            "shrike-index-adv-{tag}-{}-{}",
            std::process::id(),
            C.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    // ---- Dimension-mismatch errors ----------------------------------------

    #[test]
    fn add_second_vector_of_wrong_dim_to_a_modality_is_error() {
        // The modality's width is fixed by its first vector. A later add of a
        // mismatched-width vector must be rejected (usearch's dimension guard
        // surfaced as InvalidInput), not silently stored at the wrong width —
        // a wrong-width vector in the graph would corrupt every later search.
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        let err = e.add("text", &[2], &[unit(2, 16)]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // The good vector is still the only one stored.
        assert_eq!(e.size(), 1);
        assert!(e.contains(1));
        assert!(!e.contains(2));
    }

    #[test]
    fn search_query_of_wrong_dim_is_error_not_garbage() {
        // A query whose width disagrees with the searched sub-index must fail
        // loudly (the contract: "a query's dimension conflicts ... is an error"),
        // never return distances computed over a truncated/padded buffer.
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        let bad = unit(1, 16);
        let err = e
            .search_by_modality(&[bad], 5, Some(&["text".to_string()]), None)
            .expect_err("wrong-dim query must error, not return garbage distances");
        // The search path surfaces a usearch failure as Internal (the add path
        // maps the same dimension class to InvalidInput — an asymmetry, but both
        // are hard errors, never a silent truncated/padded result).
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    #[test]
    fn search_query_of_wrong_dim_under_scope_predicate_is_error() {
        // Same guard on the filtered (scoped) walk — the predicate path must not
        // bypass usearch's dimension check.
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        let bad = unit(1, 4);
        let err = e
            .search_by_modality(&[bad], 5, Some(&["text".to_string()]), Some(&[1i64, 2]))
            .expect_err("wrong-dim scoped query must error");
        // The filtered-search path must not bypass usearch's dimension check;
        // it surfaces the same Internal failure as the unscoped walk.
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    #[test]
    fn ensure_at_a_conflicting_ndim_is_error() {
        // The contract on `ensure` is explicit: idempotent at the SAME ndim, an
        // error at a DIFFERENT ndim. A silent no-op lets a caller believe the
        // modality is the width it asked for while it is actually another — the
        // next add/search at the assumed width then mismatches far from the
        // ensure() that caused it.
        let e = engine();
        e.ensure("text", 8).unwrap();
        e.ensure("text", 8).unwrap(); // idempotent at the same width
        let err = e.ensure("text", 16).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    // ---- Length disagreement ----------------------------------------------

    #[test]
    fn add_with_misaligned_keys_and_vectors_is_error() {
        // The explicit guard: keys/vectors lengths must align (the contract
        // stresses it). A mismatch is InvalidInput and writes nothing.
        let e = engine();
        let err = e
            .add("text", &[1, 2, 3], &[unit(1, 8), unit(2, 8)])
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(e.size(), 0);
        // Reverse imbalance too (more vectors than keys).
        let err = e.add("text", &[1], &[unit(1, 8), unit(2, 8)]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(e.size(), 0);
    }

    #[test]
    fn add_empty_is_a_noop_not_an_error() {
        // Empty/empty aligns trivially; the add short-circuits to Ok without
        // creating a sub-index (so ndim stays unknown).
        let e = engine();
        e.add("text", &[], &[]).unwrap();
        assert_eq!(e.size(), 0);
        assert_eq!(e.ndim(), None);
        assert!(e.modality_names().is_empty());
    }

    // ---- Empty / degenerate states ----------------------------------------

    #[test]
    fn search_on_empty_index_returns_empty_rankings_no_panic() {
        // No sub-index loaded at all: every query maps to an empty per-modality
        // map (the engine returns one entry per query, never panics).
        let e = engine();
        let out = e
            .search_by_modality(&[unit(1, 8), unit(2, 8)], 5, None, None)
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty() && out[1].is_empty());
    }

    #[test]
    fn search_with_no_queries_returns_no_rows() {
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        let out = e.search_by_modality(&[], 5, None, None).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn search_k_zero_yields_no_kept_hits() {
        // k=0 means "keep zero per modality": the dedup loop breaks before
        // pushing any key, so the modality contributes no ranking entry. Must
        // not panic or under/over-fetch into a usearch reservation.
        let e = engine();
        e.add("text", &[1, 2, 3], &[unit(1, 8), unit(2, 8), unit(3, 8)])
            .unwrap();
        let out = e
            .search_by_modality(&[unit(1, 8)], 0, Some(&["text".to_string()]), None)
            .unwrap();
        assert!(out[0].is_empty(), "k=0 keeps nothing, got {:?}", out[0]);
    }

    #[test]
    fn search_modality_filter_to_an_unloaded_modality_returns_empty() {
        // Filtering to a modality with no sub-index yields no rows for it.
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        let out = e
            .search_by_modality(&[unit(1, 8)], 5, Some(&["image".to_string()]), None)
            .unwrap();
        assert!(out[0].is_empty());
    }

    #[test]
    fn calibration_on_empty_index_is_empty_no_panic() {
        let e = engine();
        assert!(e.calibrate_activation(256, 2, 1).unwrap().is_empty());
    }

    #[test]
    fn calibration_with_no_nontext_modality_is_empty() {
        // Text-only: nothing to calibrate against (calibration ranks text
        // pseudo-queries against NON-text modalities only).
        let e = engine();
        let keys: Vec<i64> = (1..=10).collect();
        let vecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, 8)).collect();
        e.add("text", &keys, &vecs).unwrap();
        assert!(e.calibrate_activation(256, 2, 1).unwrap().is_empty());
    }

    #[test]
    fn calibration_sample_size_zero_yields_no_stats() {
        // sample_size=0 draws no pseudo-queries, so no modality reaches
        // min_count and the stats are empty (the gate stays disabled).
        let e = engine();
        let keys: Vec<i64> = (1..=20).collect();
        let tvecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, 8)).collect();
        let ivecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64 + 1000, 8)).collect();
        e.add("text", &keys, &tvecs).unwrap();
        e.add("image", &keys, &ivecs).unwrap();
        assert!(e.calibrate_activation(0, 2, 1).unwrap().is_empty());
    }

    #[test]
    fn calibration_min_count_above_sample_disables_the_gate() {
        // min_count larger than every modality's collected pairs → no stats
        // (the gate disables rather than reporting an under-powered estimate).
        let e = engine();
        let keys: Vec<i64> = (1..=20).collect();
        let tvecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, 8)).collect();
        let ivecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64 + 1000, 8)).collect();
        e.add("text", &keys, &tvecs).unwrap();
        e.add("image", &keys, &ivecs).unwrap();
        assert!(e.calibrate_activation(256, 2, 10_000).unwrap().is_empty());
    }

    #[test]
    fn calibration_sample_size_and_k_exceeding_size_are_clamped() {
        // sample_size and k far past the stored count must clamp, not over-read:
        // the sample is the whole text set, k clamps to the image sub-index size,
        // and the stats still come back for the one non-text modality.
        let e = engine();
        let keys: Vec<i64> = (1..=12).collect();
        let tvecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, 8)).collect();
        let ivecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64 + 1000, 8)).collect();
        e.add("text", &keys, &tvecs).unwrap();
        e.add("image", &keys, &ivecs).unwrap();
        let stats = e.calibrate_activation(10_000, 10_000, 1).unwrap();
        assert_eq!(stats.len(), 1);
        let (modality, count, _mean, _std) = &stats[0];
        assert_eq!(modality, "image");
        assert!(*count >= 1.0 && *count <= 12.0);
    }

    // ---- remove: count semantics ------------------------------------------

    #[test]
    fn remove_returns_text_modality_count_not_other_modalities() {
        // The contract stresses: remove returns the count removed from the TEXT
        // modality (one text vector per note), NOT the total across modalities
        // and NOT the remaining count. Here key 1 has 1 text + 3 image vectors;
        // removing it must report 1 (text), with all 4 actually gone.
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        e.add("image", &[1, 1, 1], &[unit(3, 8), unit(4, 8), unit(5, 8)])
            .unwrap();
        assert_eq!(e.size(), 5);
        let removed = e.remove(&[1]).unwrap();
        assert_eq!(removed, 1, "text-modality removed count, not 4 (total)");
        assert!(!e.contains(1));
        assert_eq!(e.modality_keys("image"), Vec::<i64>::new());
        assert_eq!(e.size(), 1); // only key 2's text vector remains
    }

    #[test]
    fn remove_of_image_only_key_reports_zero_text_removed() {
        // A key that exists ONLY in a non-text modality removes >0 image
        // vectors but 0 text vectors — remove still reports 0 (text count),
        // and the image vectors are gone.
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        e.add("image", &[2, 2], &[unit(2, 8), unit(3, 8)]).unwrap();
        assert_eq!(e.remove(&[2]).unwrap(), 0, "no text vector under key 2");
        assert!(!e.modality_contains("image", 2));
        assert!(e.contains(1));
    }

    #[test]
    fn remove_missing_key_is_noop_returning_zero() {
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        assert_eq!(e.remove(&[999]).unwrap(), 0);
        assert_eq!(e.size(), 1);
        // Empty key list and empty index both short-circuit to 0.
        assert_eq!(e.remove(&[]).unwrap(), 0);
        let empty = engine();
        assert_eq!(empty.remove(&[1, 2, 3]).unwrap(), 0);
    }

    #[test]
    fn readd_same_key_appends_under_multi_semantics() {
        // The index is multi=true: re-adding an existing key APPENDS a vector
        // (the caller does replace via remove-then-add). Pin the count growth so
        // a future "update in place" change can't silently drop the old vector.
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        e.add("text", &[1], &[unit(2, 8)]).unwrap();
        assert_eq!(e.size(), 2);
        assert_eq!(e.get(1).unwrap().len(), 2, "both vectors retained");
        assert_eq!(e.keys(), vec![1], "still one distinct key");
        assert_eq!(e.modality_keys("text"), vec![1, 1], "one entry per vector");
    }

    // ---- search correctness oracle ----------------------------------------

    #[test]
    fn search_ranks_self_vector_first_and_orders_by_similarity() {
        // A query equal to a stored vector ranks that key first at ~zero IP
        // distance; on a tiny single-vector modality the dedup keeps best-first
        // order. This is the recall floor the RRF fusion builds on.
        let e = engine();
        e.add(
            "text",
            &[10, 20, 30],
            &[unit(10, 8), unit(20, 8), unit(30, 8)],
        )
        .unwrap();
        let out = e
            .search_by_modality(&[unit(20, 8)], 3, Some(&["text".to_string()]), None)
            .unwrap();
        let (keys, dists) = &out[0]["text"];
        assert_eq!(keys[0], 20, "the self-vector ranks first");
        assert!(dists[0].abs() < 1e-5, "self IP distance ~0");
        // Distances are non-decreasing (best-first).
        for w in dists.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-6,
                "rankings must be best-first: {dists:?}"
            );
        }
    }

    #[test]
    fn search_multiple_modalities_returns_a_ranking_per_modality() {
        // modalities=None searches every loaded sub-index; each contributes its
        // own ranking entry keyed by modality name.
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        e.add("image", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        let out = e.search_by_modality(&[unit(1, 8)], 5, None, None).unwrap();
        assert!(out[0].contains_key("text"));
        assert!(out[0].contains_key("image"));
        assert_eq!(out[0]["text"].0[0], 1);
        assert_eq!(out[0]["image"].0[0], 1);
    }

    #[test]
    fn scope_with_no_in_scope_keys_returns_empty_ranking() {
        // A scope set disjoint from every stored key yields no in-scope
        // neighbours — the predicate rejects all candidates, so the modality
        // contributes nothing (no over-fetch-then-filter leak of out-of-scope
        // keys).
        let e = engine();
        e.add("text", &[1, 2, 3], &[unit(1, 8), unit(2, 8), unit(3, 8)])
            .unwrap();
        let out = e
            .search_by_modality(
                &[unit(1, 8)],
                5,
                Some(&["text".to_string()]),
                Some(&[404i64, 505]),
            )
            .unwrap();
        assert!(out[0].is_empty(), "no key is in scope, got {:?}", out[0]);
    }

    // ---- multi-modality with differing ndim -------------------------------

    #[test]
    fn modalities_with_different_ndim_coexist_and_stats_report_each_own_width() {
        // CLIP-shaped: text 8-dim, image 4-dim. Each sub-index keeps its OWN
        // width; modality_stats surfaces it per modality (the single top-level
        // ndim — the text width — can't express the image width).
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        e.add("image", &[1], &[unit(3, 4)]).unwrap();
        let stats: std::collections::BTreeMap<String, (usize, Option<usize>)> = e
            .modality_stats()
            .into_iter()
            .map(|(m, sz, d)| (m, (sz, d)))
            .collect();
        assert_eq!(stats["text"], (2, Some(8)));
        assert_eq!(stats["image"], (1, Some(4)));
        // The top-level ndim is the LAST-set width; both sub-indexes keep theirs
        // for their own searches (an 8-dim text query and a 4-dim image query
        // each succeed against their own modality).
        assert!(e
            .search_by_modality(&[unit(9, 8)], 2, Some(&["text".to_string()]), None)
            .is_ok());
        assert!(e
            .search_by_modality(&[unit(9, 4)], 2, Some(&["image".to_string()]), None)
            .is_ok());
    }

    #[test]
    fn drop_modality_removes_one_and_leaves_the_other() {
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        e.add("image", &[1], &[unit(2, 8)]).unwrap();
        e.drop_modality("image");
        assert_eq!(e.modality_names(), vec!["text".to_string()]);
        assert!(e.contains(1));
        assert!(!e.modality_contains("image", 1));
        // Dropping a never-loaded modality is a harmless no-op.
        e.drop_modality("audio");
        assert_eq!(e.modality_names(), vec!["text".to_string()]);
    }

    #[test]
    fn modality_stats_reports_explicit_width_of_an_ensured_empty_subindex() {
        // `ensure(m, ndim)` builds the sub-index via `new_index(ndim)`, which
        // sets the usearch dimension up front — so an ensured-but-empty
        // sub-index already reports `Some(ndim)` (size 0, width known). The
        // `None` branch in `modality_stats` is reserved for the restore
        // placeholder path (a sub-index whose width is not yet set from a file).
        let e = engine();
        e.ensure("text", 8).unwrap();
        let stats = e.modality_stats();
        assert_eq!(stats.len(), 1);
        let (modality, size, dim) = &stats[0];
        assert_eq!(modality, "text");
        assert_eq!(*size, 0);
        assert_eq!(
            *dim,
            Some(8),
            "ensure() fixes the width even with no vectors"
        );
    }

    #[test]
    fn clear_drops_every_subindex_and_resets_ndim() {
        let e = engine();
        e.add("text", &[1], &[unit(1, 8)]).unwrap();
        e.add("image", &[1], &[unit(2, 8)]).unwrap();
        e.clear();
        assert_eq!(e.size(), 0);
        assert_eq!(e.ndim(), None);
        assert!(e.modality_names().is_empty());
        assert!(!e.contains(1));
    }

    // ---- restore corners --------------------------------------------------

    #[test]
    fn restore_from_empty_dir_loads_nothing_and_leaves_index_empty() {
        // No sub-index files present: restore finds nothing to load. It returns
        // true (no corruption signalled) but the index stays empty — distinct
        // from a corrupt-file false.
        let dir = tmp("empty");
        let dirs = dir.to_str().unwrap();
        let e = engine();
        assert!(e.restore(dirs, Some(&[1, 2, 3])));
        assert_eq!(e.size(), 0);
        assert_eq!(e.ndim(), None);
        assert!(e.modality_names().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restore_from_nonexistent_dir_loads_nothing() {
        // A directory that does not exist behaves like an empty one (every file
        // probe is is_file()==false): nothing loaded, no panic.
        let missing = std::env::temp_dir().join(format!(
            "shrike-index-adv-nope-{}-does-not-exist",
            std::process::id()
        ));
        let e = engine();
        assert!(e.restore(missing.to_str().unwrap(), Some(&[1])));
        assert_eq!(e.size(), 0);
    }

    #[test]
    fn restore_over_a_populated_index_replaces_its_state() {
        // A restore from an empty dir into a populated engine replaces state
        // wholesale — the prior vectors must not survive as a phantom.
        let dir = tmp("replace");
        let dirs = dir.to_str().unwrap();
        let e = engine();
        e.add("text", &[1, 2], &[unit(1, 8), unit(2, 8)]).unwrap();
        assert!(e.restore(dirs, Some(&[1, 2])));
        assert_eq!(e.size(), 0, "restore from empty dir cleared prior state");
        std::fs::remove_dir_all(&dir).ok();
    }

    proptest! {
        /// The reconcile-rebuild path saves then restores; size/keys/get must be
        /// bit-stable across the round trip (get returns the stored vectors, so
        /// an exact compare pins the on-disk fidelity for the dedup/centroid
        /// reads that follow a restore).
        #[test]
        fn save_then_restore_round_trips_get_and_keys_for_every_key(
            tvecs in prop::collection::vec(unit_vec(6), 15),
            ivecs in prop::collection::vec(unit_vec(6), 15),
        ) {
            let dir = tmp("rt-get");
            let dirs = dir.to_str().unwrap();
            let keys: Vec<i64> = (1..=15).collect();

            let e = engine();
            e.add("text", &keys, &tvecs).unwrap();
            e.add("image", &keys, &ivecs).unwrap();
            let before_text: Vec<Vec<Vec<f32>>> = keys.iter().map(|k| e.get(*k).unwrap()).collect();
            e.save(dirs).unwrap();

            let fresh = engine();
            prop_assert!(fresh.restore(dirs, Some(&keys)));
            prop_assert_eq!(fresh.size(), e.size());
            prop_assert_eq!(fresh.keys(), e.keys());
            for (k, before) in keys.iter().zip(&before_text) {
                let got = fresh.get(*k);
                prop_assert_eq!(got.as_ref(), Some(before), "text get drift on key {}", k);
                prop_assert!(fresh.modality_contains("image", *k));
            }
            std::fs::remove_dir_all(&dir).ok();
        }

        // ---- generative round-trip property ---------------------------------

        /// Over a generated distinct-key/vector set: every added key is
        /// contained, its text get() returns exactly the stored vector, and a
        /// modality_get for image likewise — the invariant the whole read surface
        /// (dedup, centroid, calibration) relies on.
        #[test]
        fn property_add_get_contains_round_trip_over_random_keys(
            // A set of 1..=40 distinct keys, each holding exactly one vector.
            keys in prop::collection::hash_set(0i64..10_000, 1..=40),
        ) {
            let ndim = 5usize;
            let keys: Vec<i64> = keys.into_iter().collect();
            // One unit vector per key, drawn deterministically from each key so
            // the per-key get() compare below has an exact expected value.
            let tvecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, ndim)).collect();
            let ivecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64 + 7919, ndim)).collect();
            let e = engine();
            e.add("text", &keys, &tvecs).unwrap();
            e.add("image", &keys, &ivecs).unwrap();

            // size() sums every sub-index (text + image), pinned by
            // `add_remove_contains_keys`.
            prop_assert_eq!(e.size(), keys.len() * 2);
            for (i, k) in keys.iter().enumerate() {
                prop_assert!(e.contains(*k), "missing key {}", k);
                prop_assert_eq!(e.get(*k), Some(vec![tvecs[i].clone()]), "text get on {}", k);
                prop_assert_eq!(
                    e.modality_get("image", *k),
                    Some(vec![ivecs[i].clone()]),
                    "image get on {}",
                    k
                );
            }
            // get on an absent key is None, never a panic.
            let present: std::collections::HashSet<i64> = keys.iter().copied().collect();
            let absent = (0..i64::MAX).find(|k| !present.contains(k)).unwrap();
            prop_assert_eq!(e.get(absent), None);
            prop_assert!(!e.contains(absent));
        }

        /// Remove a generated subset; every removed key vanishes from BOTH
        /// modalities, the survivors stay, and the reported count equals the
        /// number of those keys that had a text vector (here all do — one text
        /// vector each). The mask picks each of keys 1..=50 in or out.
        #[test]
        fn property_remove_drops_all_modalities_and_reports_text_count(
            mask in prop::collection::vec(any::<bool>(), 50),
        ) {
            let ndim = 4usize;
            let e = engine();
            let keys: Vec<i64> = (1..=50).collect();
            let tvecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64, ndim)).collect();
            let ivecs: Vec<Vec<f32>> = keys.iter().map(|k| unit(*k as u64 + 7919, ndim)).collect();
            e.add("text", &keys, &tvecs).unwrap();
            e.add("image", &keys, &ivecs).unwrap();

            let to_remove: Vec<i64> = keys
                .iter()
                .copied()
                .zip(&mask)
                .filter_map(|(k, &m)| m.then_some(k))
                .collect();
            let expected = to_remove.len();
            prop_assert_eq!(e.remove(&to_remove).unwrap(), expected);
            for k in &to_remove {
                prop_assert!(!e.contains(*k), "removed key {} still in text", k);
                prop_assert!(
                    !e.modality_contains("image", *k),
                    "removed key {} still in image",
                    k
                );
            }
            for k in keys.iter().filter(|k| !to_remove.contains(k)) {
                prop_assert!(e.contains(*k), "survivor key {} dropped", k);
                prop_assert!(
                    e.modality_contains("image", *k),
                    "survivor image {} dropped",
                    k
                );
            }
            // size() sums both modalities; the survivors keep one text + one image
            // vector each.
            prop_assert_eq!(e.size(), (keys.len() - expected) * 2);
        }
    }
}
