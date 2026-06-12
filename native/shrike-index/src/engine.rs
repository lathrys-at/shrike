//! The native index engine (#273): per-modality usearch sub-indexes behind the
//! frozen `IndexEngine` surface (#267).
//!
//! Implements exactly what `shrike.index_engine.UsearchIndexEngine` does —
//! per-modality sub-index files (`index.usearch` / `index.<m>.usearch`),
//! max-sim-per-note dedup, the empty-index phantom-hit guard (inert under this
//! binding, kept as part of the frozen contract), stale-file deletion on save,
//! and the #201b activation calibration — instance-per-space, no global state
//! (#232's multi-space manager is "make N engines").
//!
//! One binding gap shapes this module: usearch's Rust crate (2.25.3) exposes
//! no key-enumeration API (the Python binding's `Index.keys`), so the engine
//! tracks per-modality `{key: vector_count}` maps itself. On `restore` of a
//! file written by the *Python* engine the map is reconstructed by probing
//! `count(key)` for the caller-provided candidate keys (the orchestrator's
//! `index.hashes.json` — every indexed note id). A multimodal index restored
//! with no candidates fails the restore (one-time drift rebuild — convention
//! 7's worst acceptable outcome); a text-only one loads with keys unknown,
//! which only calibration would need (and text-only never calibrates).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use shrike_ffi::{NativeError, NativeResult};
use usearch::Index;

use crate::new_index;

/// Mirrors `shrike.index_engine.SEARCH_OVERFETCH`.
const SEARCH_OVERFETCH: usize = 4;

/// One modality's ranking for one query: parallel (note_ids, distances),
/// best-first, deduped to distinct notes.
pub type ModalityRanking = (Vec<i64>, Vec<f32>);

/// Per-(non-text-)modality activation stats: (modality, n, mean, std).
pub type ActivationStats = Vec<(String, f64, f64, f64)>;

struct Sub {
    index: Index,
    /// key → vector count (the Rust binding has no key enumeration).
    counts: BTreeMap<i64, u32>,
    /// False when restored without candidates (text-only path) — keys unknown.
    keys_known: bool,
}

impl Sub {
    fn new(index: Index) -> Self {
        Self {
            index,
            counts: BTreeMap::new(),
            keys_known: true,
        }
    }
}

struct State {
    indexes: BTreeMap<String, Sub>,
    ndim: Option<usize>,
}

pub struct MultiModalIndex {
    /// Known modalities in load order (TEXT first — it keeps the original
    /// index.usearch filename). Mirrors `_INDEX_MODALITIES`.
    modalities: Vec<String>,
    text: String,
    state: Mutex<State>,
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
        // the old next_power_of_two doubling without over-allocating up to
        // ~2× a large collection's footprint on the last step (#382).
        let target = needed.max(index.capacity() + index.capacity() / 2).max(64);
        index
            .reserve(target)
            .map_err(|e| NativeError::internal(format!("usearch reserve: {e}")))?;
    }
    Ok(())
}

impl MultiModalIndex {
    /// `modalities[0]` is the text modality (mandatory, owns index.usearch).
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
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("index state lock poisoned")
    }

    pub fn size(&self) -> usize {
        self.lock().indexes.values().map(|s| s.index.size()).sum()
    }

    pub fn ndim(&self) -> Option<usize> {
        self.lock().ndim
    }

    pub fn modality_sizes(&self) -> Vec<(String, usize)> {
        self.lock()
            .indexes
            .iter()
            .map(|(m, s)| (m.clone(), s.index.size()))
            .collect()
    }

    pub fn modality_names(&self) -> Vec<String> {
        self.lock().indexes.keys().cloned().collect()
    }

    pub fn ensure(&self, modality: &str, ndim: usize) -> NativeResult<()> {
        let mut state = self.lock();
        if !state.indexes.contains_key(modality) {
            state
                .indexes
                .insert(modality.to_string(), Sub::new(new_index(ndim)?));
            state.ndim = Some(ndim);
        }
        Ok(())
    }

    pub fn clear(&self) {
        let mut state = self.lock();
        state.indexes.clear();
        state.ndim = None;
    }

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
    /// a truncation (#381).
    pub fn save(&self, dir: &str) -> NativeResult<()> {
        let base = Path::new(dir);
        std::fs::create_dir_all(base)
            .map_err(|e| NativeError::internal(format!("mkdir {dir}: {e}")))?;
        let state = self.lock();
        for (modality, sub) in &state.indexes {
            let file = base.join(file_name(&self.text, modality));
            let tmp = tmp_path(&file);
            sub.index
                .save(&tmp.to_string_lossy())
                .map_err(|e| NativeError::internal(format!("usearch save: {e}")))?;
            std::fs::rename(&tmp, &file)
                .map_err(|e| NativeError::internal(format!("usearch save rename: {e}")))?;
        }
        for modality in &self.modalities {
            if !state.indexes.contains_key(modality) {
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
                .map_err(|e| NativeError::invalid_input(format!("usearch add: {e}")))?;
            *sub.counts.entry(*key).or_insert(0) += 1;
        }
        Ok(())
    }

    /// Remove the keys' vectors from every sub-index; returns the count removed
    /// from the *text* sub-index (one text vector per note — the note count).
    pub fn remove(&self, keys: &[i64]) -> NativeResult<usize> {
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
                    .map_err(|e| NativeError::internal(format!("usearch remove: {e}")))?;
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
    pub fn search_by_modality(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        modalities: Option<&[String]>,
    ) -> NativeResult<Vec<BTreeMap<String, ModalityRanking>>> {
        let span = tracing::debug_span!("index.search", queries = queries.len(), k);
        let _enter = span.enter();
        let state = self.lock();
        let fetch = (k * SEARCH_OVERFETCH).max(k);
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
            // per-hit phantom check is needed (frozen-contract parity, #382).
            if sub.index.size() == 0 {
                continue;
            }
            for (qi, query) in queries.iter().enumerate() {
                let hits = sub
                    .index
                    .search(query, fetch)
                    .map_err(|e| NativeError::internal(format!("usearch search: {e}")))?;
                let mut keys: Vec<i64> = Vec::new();
                let mut distances: Vec<f32> = Vec::new();
                let mut seen = std::collections::HashSet::new();
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
        // mutated under us (the lock forbids it) or a usearch bug (#382).
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

    /// Per-(non-text-)modality best-match stats for the activation gate
    /// (#201b): sample stored text vectors as pseudo-queries (deterministic),
    /// search each non-text modality, record the best non-self match.
    pub fn calibrate_activation(
        &self,
        sample_size: usize,
        k: usize,
        min_count: usize,
    ) -> NativeResult<ActivationStats> {
        let state = self.lock();
        let Some(text_sub) = state.indexes.get(&self.text) else {
            return Ok(Vec::new());
        };
        if text_sub.index.size() == 0 {
            return Ok(Vec::new());
        }
        let non_text: Vec<(&String, &Sub)> = state
            .indexes
            .iter()
            .filter(|(m, s)| *m != &self.text && s.index.size() > 0)
            .collect();
        if non_text.is_empty() {
            return Ok(Vec::new());
        }

        // Deterministic sample: an LCG Fisher-Yates over the sorted keys.
        // Stable across runs of this engine; deliberately NOT numpy's sampler —
        // the stats are statistical, never byte-pinned across engines.
        let mut sample: Vec<i64> = text_sub.counts.keys().copied().collect();
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let n = sample.len();
        for i in (1..n).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            sample.swap(i, j);
        }
        sample.truncate(sample_size);

        let mut stats: ActivationStats = Vec::new();
        for (modality, sub) in non_text {
            let mut best_sims: Vec<f64> = Vec::new();
            for key in &sample {
                let Some(vectors) = Self::vectors_of(&text_sub.index, *key as u64) else {
                    continue;
                };
                // k > 1 so a pseudo-query whose own image is the nearest hit
                // still has a non-self hit to record.
                let hits = sub
                    .index
                    .search(&vectors[0], k.min(sub.index.size()))
                    .map_err(|e| NativeError::internal(format!("usearch search: {e}")))?;
                for (hk, dist) in hits.keys.iter().zip(hits.distances.iter()) {
                    if *hk as i64 == *key {
                        continue; // exclude the pseudo-query's own note
                    }
                    best_sims.push(1.0 - *dist as f64);
                    break; // nearest non-self hit = this query's best match
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let out = e.search_by_modality(&[unit(1, 8)], 5, None).unwrap();
        let (keys, dists) = &out[0]["image"];
        assert_eq!(keys, &vec![1]);
        assert!(dists[0].abs() < 1e-5);
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
}
