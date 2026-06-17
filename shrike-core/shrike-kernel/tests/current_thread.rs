//! The degenerate single-thread proof (#374 design 3): install a
//! `current_thread` runtime through the init seam and run a full
//! open → upsert → search → close flow — the entire kernel (ops, the
//! collection actor, completions) on ONE thread driving its own asynchrony,
//! zero extra threads. A separate integration binary on purpose: the runtime
//! seam is process-global, so this proof must not share a process with the
//! multi-thread suite.

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

        // The actor's jobs run on THIS thread (the one driving block_on).
        let kernel2 = Arc::clone(&kernel);
        let job_thread = kernel2
            .collection()
            .run(move |_core| std::thread::current().id())
            .await
            .unwrap();
        assert_eq!(
            job_thread, main_thread,
            "current_thread mode: the collection job ran on the driving thread"
        );

        // A lexical end-to-end slice: upsert → search → close (no embedder —
        // the degenerate kernel is lexical-only, exactly the no-extra-threads
        // case the design preserves).
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
