//! Regression: `MultiModalIndex::save` must not hold the engine state
//! lock across the on-disk write, so a search issued *during* a save is not
//! serialized behind the whole multi-file usearch write.
//!
//! The guarded-against bug: the save held `Mutex<State>` across every
//! `sub.index.save(tmp)` + rename, the same lock `search_by_modality` takes.
//! The absolute stall ratio is build-dependent, so this asserts the
//! *structural* property — a concurrent search completes in well under the
//! full save window — not a brittle multiplier. The collection is sized so a
//! single search is far cheaper than a full save (the gap the bug closed),
//! while the index build stays quick enough for the unit suite.

use std::sync::Arc;
use std::time::{Duration, Instant};

use shrike_index::MultiModalIndex;

/// Deterministic unit vector for a seed (no rand dependency).
fn unit(seed: u64, ndim: usize) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..ndim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[test]
fn search_is_not_serialized_behind_a_concurrent_save() {
    let ndim = 128usize;
    let n = 3_000u64;
    let engine =
        Arc::new(MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap());
    let vectors: Vec<Vec<f32>> = (0..n).map(|k| unit(k, ndim)).collect();
    let keys: Vec<i64> = (0..n as i64).collect();
    engine.add("text", &keys, &vectors).unwrap();

    let dir = std::env::temp_dir().join(format!("s588-save-block-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dirs = dir.to_str().unwrap().to_owned();
    let q = unit(123_456, ndim);

    // Warm the file cache + measure an uncontended save (the window a search
    // must not be trapped behind).
    let t_save0 = Instant::now();
    engine.save(&dirs).unwrap();
    let save_alone = t_save0.elapsed();

    // A single (uncontended) search is the latency floor.
    let t0 = Instant::now();
    let _ = engine
        .search_by_modality(std::slice::from_ref(&q), 10, None)
        .unwrap();
    let uncontended = t0.elapsed();

    // The save must be meaningfully longer than a search, or the test proves
    // nothing — guard the precondition so a too-fast machine fails loud rather
    // than passing vacuously.
    assert!(
        save_alone > uncontended * 4,
        "precondition: save ({save_alone:?}) should dwarf a search ({uncontended:?})"
    );

    // Start a save on another thread, then race a search against it. The tiny
    // head start lands the search well inside the save window.
    let e2 = Arc::clone(&engine);
    let dirs2 = dirs.clone();
    let saver = std::thread::spawn(move || {
        let t = Instant::now();
        e2.save(&dirs2).unwrap();
        t.elapsed()
    });
    std::thread::sleep(Duration::from_millis(1));

    let t_search0 = Instant::now();
    let _ = engine.search_by_modality(&[q], 10, None).unwrap();
    let contended_search = t_search0.elapsed();

    let _ = saver.join().unwrap();
    std::fs::remove_dir_all(&dir).ok();

    // Structural property: the search overlapping the save was NOT blocked for
    // ~the save duration. Before the fix the contended search ≈ save_alone (the
    // state lock was held across the write); after it, it stays close to the
    // uncontended floor and far below the full save window.
    assert!(
        contended_search < save_alone / 2,
        "search during a save was serialized behind the write \
         (contended {contended_search:?}, save_alone {save_alone:?}, \
         uncontended {uncontended:?}): engine holds its state lock across the file write"
    );
}
