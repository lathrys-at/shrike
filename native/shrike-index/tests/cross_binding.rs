//! Cross-binding on-disk compatibility (#272): load an index written by
//! usearch-python 2.25.3, verify contents, write one back for Python to read.
//! Driven by SHRIKE_USEARCH_COMPAT_DIR (the spike runner sets it); skipped
//! silently when unset so `cargo test` stays hermetic.

use shrike_index::new_index;

#[test]
fn loads_python_written_index_and_writes_back() {
    let Ok(dir) = std::env::var("SHRIKE_USEARCH_COMPAT_DIR") else {
        eprintln!("SHRIKE_USEARCH_COMPAT_DIR unset; skipping cross-binding check");
        return;
    };
    let index = new_index(8).unwrap();
    index.load(&format!("{dir}/py-written.usearch")).unwrap();
    assert_eq!(index.size(), 4, "python wrote 4 vectors (multi key 1 has 2)");
    assert_eq!(index.count(1), 2);
    assert_eq!(index.count(2), 1);
    assert_eq!(index.dimensions(), 8);

    // Search must hit key 2 for its own vector at ~zero distance.
    let mut v2 = vec![0.0f32; 8];
    let copied = index.get(2, &mut v2).unwrap();
    assert_eq!(copied, 1);
    let hits = index.search(&v2, 1).unwrap();
    assert_eq!(hits.keys[0], 2);
    assert!(hits.distances[0].abs() < 1e-5);

    // Mutate (remove a multi key, add a new note) and write back.
    assert_eq!(index.remove(1).unwrap(), 2);
    let v_new: Vec<f32> = (0..8).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    index.add(7, &v_new).unwrap();
    index.save(&format!("{dir}/rs-written.usearch")).unwrap();
}
