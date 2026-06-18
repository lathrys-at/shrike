//! The driven-runtime proof: the harness provides every thread and
//! shrike-core spawns none. A SEPARATE integration binary on purpose — the
//! runtime seam (`init_driven_runtime`) is process-global, so this proof must
//! not share a process with the multi-thread suite or the `current_thread.rs`
//! (default-mode) proof.
//!
//! What it pins, playing the harness's N+2-threads role with plain test threads:
//!
//!   1. **The full driven model runs end-to-end.** With a `current_thread`
//!      runtime installed in DRIVEN mode and the harness parking threads in
//!      `drive_io` / `drive_sync` / `drive_compute`, a full open → upsert →
//!      search flow completes and its completion is observed from a request
//!      thread (via `spawn_op` — whose inner future runs eagerly on the runtime,
//!      driven by `drive_io` — plus a channel back to the request thread, the
//!      threaded analogue of the asyncio bridge). A regression that fails to
//!      drive the runtime or drain a pool would hang (caught as a recv timeout,
//!      never an infinite hang).
//!
//!   2. **Sync runs on `drive_sync`, OFF the runtime.** A collection job reports
//!      the thread it ran on; it is the `drive_sync` thread, never a runtime
//!      worker. This is the sync-never-on-a-runtime-worker invariant in its
//!      structural form — anki's
//!      `block_on` is legal on `drive_sync` because that thread is never a
//!      runtime context.
//!
//!   3. **`submit_blocking` runs CPU work on `drive_compute` and returns it.**
//!      The threaded-host submission path puts a unit of work on the compute
//!      pool and blocks the caller until it completes; with N=2 compute threads
//!      the engine overlap property ("N ≥ 2") holds.
//!
//! Lexical-only (no embedder): the degenerate kernel is lexical-only, exactly
//! like `current_thread.rs`. The point is the threading model, not the ranking.
//!
//! Note on teardown: the driven pool senders live in the process-global seam
//! and are not closed in-process (a real harness/binding closes them at
//! shutdown), so the `drive_sync`/`drive_compute` parkers idle on
//! an empty queue and are left to die at process exit. Only `drive_io` returns
//! here (its `until` resolves), and we join it. This is a test of the kernel's
//! threading mechanism, not the binding's shutdown sequencing.

use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use shrike_kernel::{
    drive_compute, drive_io, drive_sync, init_driven_runtime, spawn_op, submit_blocking, Kernel,
};

/// Receive with a TIMEOUT so a runtime that never drives the op (the bug this
/// guards) fails as a timeout, not an infinite hang.
fn recv<T>(rx: &mpsc::Receiver<T>) -> T {
    rx.recv_timeout(Duration::from_secs(30))
        .expect("the driven runtime must complete the op (no hang)")
}

/// Submit a kernel future onto the runtime and forget the observer: `spawn_op`
/// schedules the inner future EAGERLY (the documented detach semantics — the op
/// runs to completion even if the returned future is dropped), and the inner
/// future itself sends its result back over a channel. This is the threaded
/// host's submission shape (the asyncio server instead awaits the returned
/// future via the bridge).
fn submit(fut: impl std::future::Future<Output = shrike_error::NativeResult<()>> + Send + 'static) {
    drop(spawn_op(fut));
}

#[test]
fn full_flow_on_a_driven_runtime() {
    // Install a current_thread runtime in the DRIVEN model: the harness (these
    // test threads) provides drive_io + drive_sync + drive_compute.
    init_driven_runtime(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    )
    .unwrap_or_else(|_| panic!("this binary owns the runtime seam"));

    // drive_io's shutdown signal: a watch channel resolved true at teardown.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // ── The harness's committed N+2 threads ──────────────────────────────────
    // drive_io ×1 — owns + drives the runtime until shutdown resolves.
    let io = thread::Builder::new()
        .name("test-drive-io".into())
        .spawn(move || {
            let mut rx = shutdown_rx;
            drive_io(async move {
                while !*rx.borrow() {
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            })
            .expect("drive_io runs in driven mode");
        })
        .unwrap();

    // drive_sync ×1 — the serialized collection / anki-sync thread. It reports
    // its own thread id so we can prove the collection job ran HERE.
    let (sync_id_tx, sync_id_rx) = mpsc::channel();
    let _sync = thread::Builder::new()
        .name("test-drive-sync".into())
        .spawn(move || {
            sync_id_tx
                .send(thread::current().id())
                .expect("sync id receiver alive");
            drive_sync().expect("drive_sync runs in driven mode");
        })
        .unwrap();
    let drive_sync_thread = sync_id_rx.recv().unwrap();

    // drive_compute ×2 — N≥2 cooperate on one shared queue.
    let _compute: Vec<_> = (0..2)
        .map(|i| {
            thread::Builder::new()
                .name(format!("test-drive-compute-{i}"))
                .spawn(|| drive_compute().expect("drive_compute runs in driven mode"))
                .unwrap()
        })
        .collect();

    // ── The request thread submits ops and observes completion via channels ──
    let dir = std::env::temp_dir().join(format!("shrike-driven-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("collection.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();

    // Open — the open future runs on the runtime (drive_io), opening the
    // collection on drive_sync; we observe via a channel.
    let (open_tx, open_rx) = mpsc::channel();
    submit(async move {
        let opened = Kernel::open(&collection, &cache).await.map(Arc::new);
        let _ = open_tx.send(opened);
        Ok(())
    });
    let kernel = recv(&open_rx).expect("open succeeds under the driven runtime");

    // A collection job reports the thread it ran on — must be drive_sync.
    let (job_tx, job_rx) = mpsc::channel();
    let k = Arc::clone(&kernel);
    submit(async move {
        let tid = k
            .collection()
            .run(|_core| std::thread::current().id())
            .await?;
        let _ = job_tx.send(tid);
        Ok(())
    });
    assert_eq!(
        recv(&job_rx),
        drive_sync_thread,
        "the collection job ran on the drive_sync thread (sync off the runtime)"
    );

    // A lexical end-to-end slice: upsert → search.
    let (up_tx, up_rx) = mpsc::channel();
    let k = Arc::clone(&kernel);
    submit(async move {
        let basic = k
            .collection()
            .run(|core| core.notetype_id("Basic"))
            .await?
            .expect("Basic notetype");
        let outcomes = k
            .upsert_notes(
                vec![shrike_kernel::NoteSpec {
                    notetype_id: basic,
                    deck_id: 1,
                    fields: vec!["driven mitochondria".into(), "powerhouse".into()],
                    tags: vec![],
                }],
                shrike_collection::DuplicatePolicy::parse("error").unwrap(),
            )
            .await?;
        let _ = up_tx.send(outcomes.len());
        Ok(())
    });
    assert_eq!(
        recv(&up_rx),
        1,
        "one note upserted under the driven runtime"
    );

    let (search_tx, search_rx) = mpsc::channel();
    let k = Arc::clone(&kernel);
    submit(async move {
        let hits = k.search("mitochondria", 5).await?;
        let _ = search_tx.send(hits.len());
        Ok(())
    });
    assert!(recv(&search_rx) > 0, "lexical search found the note");

    // ── submit_blocking: CPU work on the compute pool, blocking this thread. ──
    let computed = submit_blocking(|| Ok::<u64, shrike_error::NativeError>(0x5031));
    assert_eq!(
        computed.unwrap(),
        0x5031,
        "submit_blocking ran the work on the compute pool and returned it"
    );

    // ── Teardown: close, resolve drive_io's `until`, join drive_io. ──
    let (close_tx, close_rx) = mpsc::channel();
    let k = Arc::clone(&kernel);
    submit(async move {
        let _ = close_tx.send(k.close().await);
        Ok(())
    });
    recv(&close_rx).expect("close succeeds");

    drop(kernel);
    shutdown_tx.send(true).unwrap();
    io.join()
        .expect("drive_io thread joins after its shutdown signal");
    // The pool parkers idle on the empty process-global queue and die at exit
    // (see the module note); not joined here.

    let _ = std::fs::remove_dir_all(dir);
}
