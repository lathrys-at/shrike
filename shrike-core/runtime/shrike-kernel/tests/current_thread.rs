//! A `current_thread` runtime, end-to-end: install one through the default init
//! seam and run a full open → upsert → search → close flow — the kernel's async
//! side (ops, the collection actor's dispatch loop, completions) on one thread
//! driving its own asynchrony. A separate integration binary on purpose: the
//! runtime seam is process-global, so this proof must not share a process with
//! the multi-thread suite.
//!
//! The collection actor routes every job through `dispatch_sync`, which in
//! default mode is `spawn_blocking` — so a sync job runs on a blocking-pool
//! thread, OFF the runtime worker, even when the runtime is `current_thread`.
//! That is the sync-never-on-a-runtime-worker invariant made structural: a sync
//! anki call (which `block_on`s) can never land on a runtime worker. So this
//! test asserts the collection job ran on a DIFFERENT thread than the `block_on`
//! driver. The full harness-driven model is proved by `driven_mode.rs`.

use std::sync::Arc;

use shrike_kernel::{init_runtime, Kernel};

#[test]
fn full_flow_on_a_current_thread_runtime() {
    init_runtime(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    )
    .unwrap_or_else(|_| panic!("this binary owns the runtime seam"));

    let dir = std::env::temp_dir().join(format!("shrike-current-thread-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("collection.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();

    let main_thread = std::thread::current().id();
    shrike_kernel::block_on(async move {
        let kernel = Arc::new(Kernel::open(&collection, &cache).await.unwrap());

        // The actor's jobs run off the block_on driver thread: `dispatch_sync`
        // is `spawn_blocking` in default mode, so a sync job rides the blocking
        // pool, never the runtime worker.
        let kernel2 = Arc::clone(&kernel);
        let job_thread = kernel2
            .collection()
            .run(move |_core| std::thread::current().id())
            .await
            .unwrap();
        assert_ne!(
            job_thread, main_thread,
            "the collection job ran off the runtime driver thread (the sync \
             offload, structural)"
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
