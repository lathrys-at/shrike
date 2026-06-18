//! A full open → upsert → search → close flow on the driven `current_thread`
//! runtime, via the shared test fixture (1 io + 1 sync + N compute). A separate
//! integration binary on purpose: the runtime seam is process-global, so this
//! proof owns its own process.
//!
//! The collection actor routes every job through `dispatch_sync`, which enqueues
//! onto the committed `drive_sync` thread — a plain OS thread, never a runtime
//! context. That is the sync-never-on-a-runtime-worker invariant made
//! structural: a sync anki call (which `block_on`s) can never land on a runtime
//! worker. So this test asserts the collection job ran on a DIFFERENT thread
//! than the one submitting the flow. The full driven threading model (every
//! committed thread, shutdown + join) is proved by `driven_mode.rs`.

use std::sync::Arc;

use shrike_kernel::runtime::testing;
use shrike_kernel::Kernel;

#[test]
fn full_flow_on_the_driven_runtime() {
    let dir = std::env::temp_dir().join(format!("shrike-current-thread-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("collection.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();

    let submit_thread = std::thread::current().id();
    testing::run_with_sync(async move {
        let kernel = Arc::new(Kernel::open(&collection, &cache).await.unwrap());

        // The actor's jobs run on the committed `drive_sync` thread, never the
        // submitting (or any runtime) thread — the sync offload, structural.
        let kernel2 = Arc::clone(&kernel);
        let job_thread = kernel2
            .collection()
            .run(move |_core| std::thread::current().id())
            .await
            .unwrap();
        assert_ne!(
            job_thread, submit_thread,
            "the collection job ran off the submitting thread (the sync offload, \
             structural)"
        );

        // A lexical end-to-end slice: upsert → search → close. No embedder, so
        // search runs on the lexical signal alone.
        let basic = kernel
            .collection()
            .run(|core| core.notetype_id("Basic"))
            .await
            .unwrap()
            .unwrap();
        let outcomes = kernel
            .upsert_notes(
                vec![shrike_kernel::NoteSpec {
                    notetype_id: basic,
                    deck_id: 1,
                    fields: vec!["single threaded mitochondria".into(), "powerhouse".into()],
                    tags: vec![],
                }],
                shrike_collection_policy(),
            )
            .await
            .unwrap();
        assert_eq!(outcomes.len(), 1);
        let hits = kernel.search("mitochondria", 5).await.unwrap();
        assert!(!hits.is_empty(), "lexical search found the note");
        kernel.close().await.unwrap();
    });
    let _ = std::fs::remove_dir_all(dir);
}

fn shrike_collection_policy() -> shrike_collection::DuplicatePolicy {
    shrike_collection::DuplicatePolicy::parse("error").unwrap()
}
