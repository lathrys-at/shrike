//! S7-2 repro (preserved by lead; rev-S7 worktree reaped).
//! The engine's Mutex<State> is held across the on-disk save → a concurrent
//! search stalls for the full save window. RED at fa54f8c: 303× stall
//! (uncontended 158.9µs, save alone 48.1ms, contended search 48.2ms).
//! Place at native/shrike-index/tests/s7_save_blocks_search.rs (existing deps).
//! Run: cd native && CARGO_TARGET_DIR=$HOME/.cache/shrike-review-target/s7 cargo test -p shrike-index --test s7_save_blocks_search -- --nocapture

use std::sync::Arc;
use std::time::{Duration, Instant};
use shrike_index::MultiModalIndex;

fn unit(seed: u64, ndim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..ndim).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }).collect();
    let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
    for x in &mut v { *x /= norm; } v
}

#[test]
fn search_blocks_for_the_full_duration_of_a_concurrent_save() {
    let ndim = 256usize; let n = 60_000u64;
    let engine = Arc::new(MultiModalIndex::new(vec!["text".to_owned(), "image".to_owned()]).unwrap());
    let vectors: Vec<Vec<f32>> = (0..n).map(|k| unit(k, ndim)).collect();
    let keys: Vec<i64> = (0..n as i64).collect();
    engine.add("text", &keys, &vectors).unwrap();
    let dir = std::env::temp_dir().join(format!("s7-save-block-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dirs = dir.to_str().unwrap().to_owned();
    let q = unit(123_456, ndim);
    let t0 = Instant::now(); let _ = engine.search_by_modality(&[q.clone()], 10, None).unwrap();
    let uncontended = t0.elapsed();
    let t_save0 = Instant::now(); engine.save(&dirs).unwrap(); let save_alone = t_save0.elapsed();
    let e2 = Arc::clone(&engine); let dirs2 = dirs.clone();
    let saver = std::thread::spawn(move || { let t = Instant::now(); e2.save(&dirs2).unwrap(); t.elapsed() });
    std::thread::sleep(Duration::from_millis(5));
    let t_search0 = Instant::now(); let _ = engine.search_by_modality(&[q], 10, None).unwrap();
    let contended_search = t_search0.elapsed();
    let _ = saver.join().unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert!(contended_search < save_alone / 2,
        "REPRO: a search during a save was blocked for ~the save duration (contended {contended_search:?} vs save {save_alone:?}); engine holds its state lock across the file write");
}
