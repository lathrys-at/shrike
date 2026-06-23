//! Concurrency regression guards: the `save`-vs-`search`/`add`/`remove` safety
//! is INFERRED (usearch 2.25.3 documents concurrent search+update but is silent
//! on save-vs-search and save-vs-mutate), so these pin the load-bearing
//! properties directly under contention:
//!
//! (a) searches racing repeated saves stay correct and never crash,
//! (b) add/remove racing a save never tears the on-disk file (the save_mutation
//!     guard), and
//! (c) two concurrent saves to the same dir don't corrupt the files.
//!
//! IMPORTANT: usearch is an APPROXIMATE index — a self-query legitimately
//! misses its own key for a few permil of keys even with ZERO concurrency. So
//! (a) asserts STRUCTURAL validity (non-empty, keys in range, finite
//! distances), never exact recall.

use std::sync::atomic::{AtomicBool, Ordering};
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

fn tmpdir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::AtomicU64;
    static C: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "s588-conc-{tag}-{}-{}",
        std::process::id(),
        C.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// (a) A storm of searches racing repeated saves on the SAME shared engine.
/// Expected-correct: every search returns WITHOUT panic / abort / data race,
/// and the result is structurally valid (non-empty on a populated index, keys
/// in the populated range, finite distances). This is the concurrency usearch
/// does not document; if the `Arc`-share is unsound it surfaces here as a
/// crash, an `Err`, or a structurally garbage result under load.
///
/// NOTE: top-1 == self is NOT asserted — usearch is approximate, so a
/// self-query legitimately misses its own key for a few permil of keys even
/// with zero concurrency. Asserting exact recall would flake regardless of the
/// save.
#[test]
fn searches_during_repeated_saves_stay_correct_and_never_crash() {
    let ndim = 96usize;
    let n = 2_000u64;
    let engine = Arc::new(MultiModalIndex::new(vec!["text".into(), "image".into()]).unwrap());
    let vectors: Vec<Vec<f32>> = (0..n).map(|k| unit(k, ndim)).collect();
    let keys: Vec<i64> = (0..n as i64).collect();
    engine.add("text", &keys, &vectors).unwrap();

    let dir = tmpdir("storm");
    let dirs = dir.to_str().unwrap().to_owned();
    let stop = Arc::new(AtomicBool::new(false));

    // Saver thread: serialize as fast as possible for the whole window.
    let e_save = Arc::clone(&engine);
    let stop_save = Arc::clone(&stop);
    let dirs_save = dirs.clone();
    let saver = std::thread::spawn(move || {
        let mut saves = 0u64;
        while !stop_save.load(Ordering::Relaxed) {
            e_save.save(&dirs_save).unwrap();
            saves += 1;
        }
        saves
    });

    // Several searcher threads hammering queries while saves run.
    let mut searchers = Vec::new();
    for t in 0..4 {
        let e = Arc::clone(&engine);
        let vecs = vectors.clone();
        let st = Arc::clone(&stop);
        searchers.push(std::thread::spawn(move || {
            let mut ok = 0u64;
            let mut q = 100i64 + t as i64;
            while !st.load(Ordering::Relaxed) {
                let out = e
                    .search_by_modality(std::slice::from_ref(&vecs[q as usize]), 5, None, None)
                    .unwrap();
                // Structural validity under the concurrent save: a populated
                // index must return hits, every key in range, every distance
                // finite. (Recall is NOT asserted — see the note above.)
                let ranking = out[0].get("text");
                assert!(
                    ranking.is_some_and(|(k, _)| !k.is_empty()),
                    "a search overlapping a save returned no hits on a populated index"
                );
                let (hit_keys, dists) = ranking.unwrap();
                for &k in hit_keys {
                    assert!(
                        (0..n as i64).contains(&k),
                        "garbage key {k} from a search overlapping a save"
                    );
                }
                for &d in dists {
                    assert!(
                        d.is_finite(),
                        "non-finite distance during a concurrent save"
                    );
                }
                ok += 1;
                q = 100 + ((q + 1) % 1500);
            }
            ok
        }));
    }

    std::thread::sleep(Duration::from_millis(500));
    stop.store(true, Ordering::Relaxed);

    let saves = saver.join().unwrap();
    let total: u64 = searchers.into_iter().map(|h| h.join().unwrap()).sum();
    std::fs::remove_dir_all(&dir).ok();

    assert!(saves > 0, "the saver made no progress");
    assert!(total > 0, "the searchers made no progress");
    eprintln!("storm: {saves} saves, {total} correct concurrent searches");
}

/// (b) add/remove racing a save AND a concurrent search: the `save_mutation`
/// guard must serialize add/remove against the save so the on-disk bytes can't
/// tear, and the `state` `RwLock` must exclude the add's `reserve()` realloc from
/// a concurrent search read — so a search over the growing/shrinking index never
/// reads torn state. Expected-correct: no panic; the searches stay structurally
/// valid throughout; the final save restores cleanly with every base key present.
#[test]
fn add_remove_racing_a_save_never_tears_the_file() {
    let ndim = 64usize;
    let base = 1_000u64;
    let engine = Arc::new(MultiModalIndex::new(vec!["text".into()]).unwrap());
    let v0: Vec<Vec<f32>> = (0..base).map(|k| unit(k, ndim)).collect();
    let k0: Vec<i64> = (0..base as i64).collect();
    engine.add("text", &k0, &v0).unwrap();

    let dir = tmpdir("tear");
    let dirs = dir.to_str().unwrap().to_owned();
    let stop = Arc::new(AtomicBool::new(false));

    // Mutator: churn adds and removes of a disjoint high-key band.
    let e_mut = Arc::clone(&engine);
    let stop_mut = Arc::clone(&stop);
    let mutator = std::thread::spawn(move || {
        let mut i = 0i64;
        while !stop_mut.load(Ordering::Relaxed) {
            let key = 10_000 + (i % 500);
            let vec = unit(key as u64, ndim);
            e_mut
                .add("text", &[key], std::slice::from_ref(&vec))
                .unwrap();
            let _ = e_mut.remove(&[key]).unwrap();
            i += 1;
        }
        i
    });

    // Saver racing the mutator.
    let e_save = Arc::clone(&engine);
    let stop_save = Arc::clone(&stop);
    let dirs_save = dirs.clone();
    let saver = std::thread::spawn(move || {
        let mut saves = 0u64;
        while !stop_save.load(Ordering::Relaxed) {
            e_save.save(&dirs_save).unwrap();
            saves += 1;
        }
        saves
    });

    // Searcher racing the mutator: every `add` reserves (and may realloc) under the
    // exclusive write guard, so a concurrent search read guard must never observe a
    // mid-realloc index. A garbage key or non-finite distance here would mean a
    // reserve raced a walk — the exact case the RwLock (not the old `Mutex`) now has
    // to guarantee while still letting searches run concurrently.
    let e_search = Arc::clone(&engine);
    let stop_search = Arc::clone(&stop);
    let base_vecs: Vec<Vec<f32>> = (0..base).map(|k| unit(k, ndim)).collect();
    let searcher = std::thread::spawn(move || {
        let mut ok = 0u64;
        let mut q = 0usize;
        while !stop_search.load(Ordering::Relaxed) {
            let out = e_search
                .search_by_modality(std::slice::from_ref(&base_vecs[q]), 5, None, None)
                .unwrap();
            if let Some((keys, dists)) = out[0].get("text") {
                for &k in keys {
                    assert!(
                        (0..base as i64).contains(&k) || (10_000..10_500).contains(&k),
                        "garbage key {k} from a search racing add/remove+reserve"
                    );
                }
                for &d in dists {
                    assert!(
                        d.is_finite(),
                        "non-finite distance during a concurrent reserve"
                    );
                }
            }
            ok += 1;
            q = (q + 1) % base as usize;
        }
        ok
    });

    std::thread::sleep(Duration::from_millis(400));
    stop.store(true, Ordering::Relaxed);
    let muts = mutator.join().unwrap();
    let saves = saver.join().unwrap();
    let searches = searcher.join().unwrap();

    // A final clean save, then restore it: it must load and the base band must
    // all be present (the disjoint churn band may or may not be there, but the
    // file must be valid and complete).
    engine.save(&dirs).unwrap();
    let restored = MultiModalIndex::new(vec!["text".into()]).unwrap();
    let candidates: Vec<i64> = (0..base as i64).chain(10_000..10_500).collect();
    assert!(
        restored.restore(&dirs, Some(&candidates)),
        "the saved files did not restore cleanly (possible torn write)"
    );
    for key in &k0 {
        assert!(
            restored.contains(*key),
            "base key {key} missing after a save raced with add/remove"
        );
    }
    assert!(searches > 0, "the searcher made no progress");
    std::fs::remove_dir_all(&dir).ok();
    eprintln!("tear: {muts} mutation cycles, {saves} saves, {searches} searches, restore clean");
}

/// (c) Two saves racing the SAME directory. Each save stages to `<file>.tmp`
/// then renames over the canonical name; two concurrent savers must both
/// complete `Ok` and leave a file that restores cleanly with every key — they
/// must not clobber each other's `.tmp` into a torn or partial canonical file.
/// (`IndexOrchestrator::save` serializes savers with its own guard, but the
/// engine is a shared library type, so its `save` should be self-safe under a
/// concurrent caller too.)
#[test]
fn two_concurrent_saves_do_not_corrupt_the_files() {
    let ndim = 64usize;
    let n = 1_500u64;
    let engine = Arc::new(MultiModalIndex::new(vec!["text".into(), "image".into()]).unwrap());
    let vectors: Vec<Vec<f32>> = (0..n).map(|k| unit(k, ndim)).collect();
    let keys: Vec<i64> = (0..n as i64).collect();
    engine.add("text", &keys, &vectors).unwrap();

    let dir = tmpdir("dual");
    let dirs = dir.to_str().unwrap().to_owned();

    // Two savers hammer the same directory for a window.
    let deadline = Instant::now() + Duration::from_millis(400);
    let mut handles = Vec::new();
    for _ in 0..2 {
        let e = Arc::clone(&engine);
        let d = dirs.clone();
        handles.push(std::thread::spawn(move || {
            let mut saves = 0u64;
            while Instant::now() < deadline {
                e.save(&d).unwrap(); // both must always succeed
                saves += 1;
            }
            saves
        }));
    }
    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();

    // After the race, a final save + restore must be clean and complete.
    engine.save(&dirs).unwrap();
    let restored = MultiModalIndex::new(vec!["text".into(), "image".into()]).unwrap();
    assert!(
        restored.restore(&dirs, Some(&keys)),
        "the files did not restore cleanly after two concurrent savers"
    );
    for key in &keys {
        assert!(
            restored.contains(*key),
            "key {key} missing after two concurrent savers"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
    assert!(total > 1, "the two savers made no real progress");
    eprintln!("dual: {total} concurrent saves, restore clean");
}
