//! Native vector index engine (#273) over the usearch Rust crate.
//!
//! Phase 2a (#272) first: the `spike` test module below answers the epic's
//! biggest unknown — whether the official usearch Rust crate (same native C++
//! core as usearch-python, pinned to the same 2.25.3) satisfies Shrike's exact
//! index contract, including on-disk compatibility with files written by the
//! Python binding. The Phase-2b engine is built on the verdict.

mod engine;

use shrike_ffi::{NativeError, NativeResult};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub use engine::{ActivationStats, ModalityRanking, MultiModalIndex};

/// Build a usearch index matching `UsearchIndexEngine._ensure`'s configuration:
/// cosine metric, f32 scalars, multi=true (several vectors per note_id key).
pub fn new_index(ndim: usize) -> NativeResult<Index> {
    let options = IndexOptions {
        dimensions: ndim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        multi: true,
        ..Default::default()
    };
    let index =
        Index::new(&options).map_err(|e| NativeError::internal(format!("usearch new: {e}")))?;
    index
        .reserve(64)
        .map_err(|e| NativeError::internal(format!("usearch reserve: {e}")))?;
    Ok(index)
}

#[cfg(test)]
mod spike {
    //! The #272 checklist, as executable evidence. Cross-binding (Python ↔ Rust)
    //! on-disk compatibility is exercised end-to-end by the Phase-2b facade
    //! tests (a Python-written index loaded through the native engine); here we
    //! pin the crate-side semantics the engine relies on.

    use super::*;

    fn unit(seed: u64, ndim: usize) -> Vec<f32> {
        // Deterministic pseudo-random unit vector.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut v: Vec<f32> = (0..ndim)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn multi_vectors_per_key_add_search_remove() {
        // Checklist: `multi` support — N vectors under one key; remove drops all.
        let index = new_index(8).unwrap();
        index.add(1, &unit(1, 8)).unwrap();
        index.add(1, &unit(2, 8)).unwrap();
        index.add(2, &unit(3, 8)).unwrap();
        assert_eq!(index.size(), 3);
        assert_eq!(index.count(1), 2);

        // Search returns per-vector hits (key repeats), dedupable per note.
        let hits = index.search(&unit(1, 8), 10).unwrap();
        assert!(hits.keys.iter().filter(|&&k| k == 1).count() >= 1);

        // remove(key) drops ALL vectors under the key and reports the count.
        let removed = index.remove(1).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(index.size(), 1);
    }

    #[test]
    fn remove_missing_key_counts_zero() {
        // Checklist: batch remove-by-key semantics — missing keys remove 0,
        // matching the Python binding's count contract.
        let index = new_index(8).unwrap();
        index.add(1, &unit(1, 8)).unwrap();
        assert_eq!(index.remove(999).unwrap(), 0);
        assert_eq!(index.remove(1).unwrap(), 1);
        assert_eq!(index.remove(1).unwrap(), 0);
    }

    #[test]
    fn cosine_is_scale_invariant() {
        // Checklist: cos/f32/i64 keys + the scale-invariance the engine's
        // "normalization is not vector-affecting" stance relies on.
        let index = new_index(8).unwrap();
        let v = unit(7, 8);
        index.add(10, &v).unwrap();
        let scaled: Vec<f32> = v.iter().map(|x| x * 42.0).collect();
        let hits = index.search(&scaled, 1).unwrap();
        assert_eq!(hits.keys[0], 10);
        assert!(hits.distances[0].abs() < 1e-5);
    }

    #[test]
    fn empty_index_search_is_empty() {
        // Checklist: the Python binding's phantom (0, 0) hit on an empty index
        // does NOT reproduce under the Rust crate — searches come back empty.
        // The engine-side dedup guard stays (it's part of the frozen contract
        // and harmless when it never trips).
        let index = new_index(8).unwrap();
        let hits = index.search(&unit(1, 8), 5).unwrap();
        assert_eq!(hits.keys.len(), 0);
    }

    #[test]
    fn slot_reuse_no_tombstone_bloat() {
        // Checklist: re-verify the 2.25.3 slot-reuse property under the Rust
        // binding — repeated remove+add of the same key must not grow the file.
        let dir = std::env::temp_dir().join(format!("shrike-usearch-spike-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("reuse.usearch");

        let index = new_index(8).unwrap();
        for key in 0..50u64 {
            index.add(key, &unit(key, 8)).unwrap();
        }
        index.save(path.to_str().unwrap()).unwrap();
        let baseline = std::fs::metadata(&path).unwrap().len();

        for round in 0..20u64 {
            for key in 0..50u64 {
                index.remove(key).unwrap();
                index.add(key, &unit(key + round, 8)).unwrap();
            }
        }
        index.save(path.to_str().unwrap()).unwrap();
        let churned = std::fs::metadata(&path).unwrap().len();
        assert!(
            churned <= baseline + baseline / 10,
            "tombstone bloat: {baseline} -> {churned}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn save_load_round_trip_same_binding() {
        let dir = std::env::temp_dir().join(format!("shrike-usearch-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.usearch");

        let index = new_index(16).unwrap();
        index.add(1, &unit(1, 16)).unwrap();
        index.add(1, &unit(2, 16)).unwrap();
        index.add(2, &unit(3, 16)).unwrap();
        index.save(path.to_str().unwrap()).unwrap();

        let loaded = new_index(16).unwrap();
        loaded.load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.size(), 3);
        assert_eq!(loaded.count(1), 2);
        assert_eq!(loaded.dimensions(), 16);
        std::fs::remove_file(&path).ok();
    }
}
