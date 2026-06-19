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
//!      `drive_io` / `drive_collection` / `drive_compute`, a full open → upsert →
//!      search flow completes and its completion is observed from a request
//!      thread (via `spawn_op` — whose inner future runs eagerly on the runtime,
//!      driven by `drive_io` — plus a channel back to the request thread, the
//!      threaded analogue of the asyncio bridge). A regression that fails to
//!      drive the runtime or drain a pool hangs, which the bazel per-target
//!      `test_timeout` turns into a real failure.
//!
//!   2. **Collection work runs on `drive_collection`, OFF the runtime.** A
//!      collection job reports the thread it ran on; it is the `drive_collection`
//!      thread, never a runtime worker. This is the
//!      collection-never-on-a-runtime-worker invariant in its structural form —
//!      anki's `block_on` is legal on `drive_collection` because that thread is
//!      never a runtime context.
//!
//!   3. **`submit_blocking` runs CPU work on `drive_compute` and returns it.**
//!      The threaded-host submission path puts a unit of work on the compute
//!      pool and blocks the caller until it completes; with N=2 compute threads
//!      the engine overlap property ("N ≥ 2") holds.
//!
//!   4. **The full N+2 shutdown joins.** `shutdown_driven_pools` closes the
//!      pool queues (so `drive_collection`/`drive_compute` see `recv == None` and
//!      return) and trips the `drive_io_until_shutdown` signal, so all N+2
//!      committed threads return and are joined — the binding's shutdown
//!      sequencing in its kernel form. A regression that fails to close a queue
//!      or wake the IO thread hangs a join, which the bazel per-target
//!      `test_timeout` turns into a real failure.
//!
//! Lexical-only (no embedder): the degenerate kernel is lexical-only, exactly
//! like `current_thread.rs`. The point is the threading model, not the ranking.

use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use shrike_error::NativeResult;
use shrike_kernel::{
    drive_collection, drive_compute, drive_io_until_shutdown, init_driven_runtime,
    shutdown_driven_pools, spawn_op, submit_blocking, submit_compute, Kernel,
};

/// Receive the op's REAL completion, unbounded — no in-test wall-clock budget,
/// which would flake a slow-but-not-hung run. A runtime that never drives the op
/// hangs and Bazel's per-target `test_timeout` (the single global hang guard)
/// fails it.
fn recv<T>(rx: &mpsc::Receiver<T>) -> T {
    rx.recv().expect("the driven runtime must complete the op")
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
    // test threads) provides drive_io + drive_collection + drive_compute.
    init_driven_runtime(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    )
    .unwrap_or_else(|_| panic!("this binary owns the runtime seam"));

    // ── The harness's committed N+2 threads ──────────────────────────────────
    // drive_io ×1 — owns + drives the runtime until shutdown_driven_pools trips
    // the built-in signal at teardown.
    let io = thread::Builder::new()
        .name("shrike-io".into())
        .spawn(move || {
            drive_io_until_shutdown().expect("drive_io runs in driven mode");
        })
        .unwrap();

    // drive_collection ×1 — the serialized collection thread. It reports its own
    // thread id so we can prove the collection job ran HERE.
    let (collection_id_tx, collection_id_rx) = mpsc::channel();
    let collection_thread = thread::Builder::new()
        .name("shrike-collection".into())
        .spawn(move || {
            collection_id_tx
                .send(thread::current().id())
                .expect("collection id receiver alive");
            drive_collection().expect("drive_collection runs in driven mode");
        })
        .unwrap();
    let drive_collection_thread = collection_id_rx.recv().unwrap();

    // drive_compute ×2 — N≥2 cooperate on one shared queue. Each reports its
    // thread id (before parking) so the submit_blocking assertion can prove the
    // work ran on a compute thread, not the request thread.
    let (compute_id_tx, compute_id_rx) = mpsc::channel();
    let compute: Vec<_> = (0..2)
        .map(|i| {
            let id_tx = compute_id_tx.clone();
            thread::Builder::new()
                .name(format!("shrike-work-{i}"))
                .spawn(move || {
                    id_tx
                        .send(thread::current().id())
                        .expect("compute id receiver alive");
                    // Drop the id sender BEFORE parking — the thread never
                    // returns from `drive_compute`, so a retained sender would
                    // hang any iter()-style drain of the id channel.
                    drop(id_tx);
                    drive_compute().expect("drive_compute runs in driven mode");
                })
                .unwrap()
        })
        .collect();
    drop(compute_id_tx);
    // Receive EXACTLY the two ids (don't iter()-drain — the threads park, and
    // their senders drop just before parking, but recv-by-count is unambiguous).
    let compute_threads: Vec<_> = (0..2)
        .map(|_| {
            compute_id_rx
                .recv()
                .expect("a compute thread registered its id")
        })
        .collect();

    // ── The request thread submits ops and observes completion via channels ──
    // Root at Bazel's per-action $TEST_TMPDIR (unique per process/run) when
    // present, so a recycled PID can't reopen a prior run's lingering dir; fall
    // back to $TMPDIR for a bare `cargo test`.
    let root = std::env::var_os("TEST_TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = root.join(format!(
        "shrike-driven-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("collection.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();

    // Open — the open future runs on the runtime (drive_io), opening the
    // collection on drive_collection; we observe via a channel.
    let (open_tx, open_rx) = mpsc::channel();
    submit(async move {
        let opened = Kernel::open(&collection, &cache).await.map(Arc::new);
        let _ = open_tx.send(opened);
        Ok(())
    });
    let kernel = recv(&open_rx).expect("open succeeds under the driven runtime");

    // A collection job reports the thread it ran on — must be drive_collection.
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
        drive_collection_thread,
        "the collection job ran on the drive_collection thread (off the runtime)"
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
        // Upsert indexes via the async ingest drain (the derived/FTS write rides the
        // compute pool), so a search immediately after an upsert is eventual-consistency
        // racy — settle the drain before the lexical search.
        k.settle().await;
        let hits = k.search("mitochondria", 5).await?;
        let _ = search_tx.send(hits.len());
        Ok(())
    });
    assert!(recv(&search_rx) > 0, "lexical search found the note");

    // ── submit_blocking: CPU work on the compute pool, blocking this thread. ──
    // The work returns the THREAD it ran on, so we pin "ran on drive_compute"
    // (not just the return value) — matching how the drive_collection claim is pinned.
    let request_thread = thread::current().id();
    let (value, ran_on) = submit_blocking(|| {
        Ok::<(u64, thread::ThreadId), shrike_error::NativeError>((0x5031, thread::current().id()))
    })
    .unwrap();
    assert_eq!(value, 0x5031, "submit_blocking returned the work's value");
    assert_ne!(
        ran_on, request_thread,
        "submit_blocking offloaded off the request thread"
    );
    assert!(
        compute_threads.contains(&ran_on),
        "submit_blocking ran on a drive_compute thread (the CPU pool), not elsewhere"
    );

    // ── N≥2 overlap: two compute jobs that each rendezvous at a shared barrier
    // can only BOTH arrive if they run concurrently on different threads. With
    // N=1 the second job would never start until the first returned, so the
    // first would block at the barrier forever (caught by the join timeout). The
    // submitter threads block on submit_blocking, so run them off-thread. ──
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let overlap: Vec<_> = (0..2)
        .map(|_| {
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                submit_blocking(move || {
                    // The barrier is the structural proof both jobs are on the
                    // pool simultaneously: each blocks until the other arrives. If
                    // the pool can't run both at once the rendezvous never
                    // completes and the bazel `test_timeout` catches it.
                    b.wait();
                    Ok::<thread::ThreadId, shrike_error::NativeError>(thread::current().id())
                })
            })
        })
        .collect();
    let overlap_threads: Vec<thread::ThreadId> = overlap
        .into_iter()
        .map(|h| {
            h.join()
                .expect("overlap submitter thread joins")
                .expect("overlap job ran")
        })
        .collect();
    assert_ne!(
        overlap_threads[0], overlap_threads[1],
        "the two barrier-rendezvous jobs ran on DIFFERENT compute threads (N≥2 overlap)"
    );

    // ── Panic resilience: a panicking pool job must lose ONLY
    // that job (a clean Err to its caller) while the pool thread SURVIVES — the
    // same isolation default-mode `spawn_blocking` already gives. Without it, a
    // panic on the single drive_collection thread (or a drive_compute thread) would
    // kill the OS thread and wedge the pool. ──

    // Compute pool: a panicking submit_blocking → Err, then the NEXT job still
    // runs (the compute thread survived).
    let panicked = submit_blocking(|| -> NativeResult<()> { panic!("boom in a compute job") });
    assert!(
        panicked.is_err(),
        "a panicking compute job returns a clean Err, not an unwind"
    );
    let after = submit_blocking(|| Ok::<u64, shrike_error::NativeError>(42)).unwrap();
    assert_eq!(
        after, 42,
        "the compute pool survived the panic — the next job ran (not a timeout/wedge)"
    );

    // ── submit_compute: a fire-and-forget job runs on the compute pool. ──
    // This is the seam the engine `Blocking` adapter's injected dispatcher calls
    // (it owns its own result channel; here the job reports the thread it ran on
    // so we pin "ran on drive_compute", not the request thread).
    let (sc_tx, sc_rx) = mpsc::channel();
    submit_compute(Box::new(move || {
        let _ = sc_tx.send(thread::current().id());
    }));
    let sc_ran_on = sc_rx
        .recv()
        .expect("submit_compute scheduled the job on the compute pool");
    assert_ne!(
        sc_ran_on, request_thread,
        "submit_compute offloaded off the request thread"
    );
    assert!(
        compute_threads.contains(&sc_ran_on),
        "submit_compute ran on a drive_compute thread (the CPU pool)"
    );

    // Collection pool (the single drive_collection thread — the wedge risk is sharpest
    // here): a panicking collection job → Err, then the next collection op succeeds.
    let (pj_tx, pj_rx) = mpsc::channel();
    let k = Arc::clone(&kernel);
    submit(async move {
        let r = k
            .collection()
            .run(|_core| panic!("boom in a collection job"))
            .await;
        let _ = pj_tx.send(r.is_err());
        Ok(())
    });
    assert!(
        recv(&pj_rx),
        "a panicking collection job returns a clean Err (drive_collection did not die)"
    );
    let (ok_tx, ok_rx) = mpsc::channel();
    let k = Arc::clone(&kernel);
    submit(async move {
        let v = k.collection().run(|_core| 7_i64).await?;
        let _ = ok_tx.send(v);
        Ok(())
    });
    assert_eq!(
        recv(&ok_rx),
        7,
        "the drive_collection thread survived the panic — the next collection op ran (not a wedge)"
    );

    // ── Teardown: close (drains the actor), then shutdown_driven_pools, then
    // join ALL N+2 committed threads — the binding's shutdown sequence. ──
    let (close_tx, close_rx) = mpsc::channel();
    let k = Arc::clone(&kernel);
    submit(async move {
        let _ = close_tx.send(k.close().await);
        Ok(())
    });
    recv(&close_rx).expect("close succeeds");
    drop(kernel);

    // The kernel is quiesced, so closing the pool queues (and tripping the IO
    // signal) lets every committed thread return. Joining proves the shutdown is
    // clean — a queue left open or an un-woken IO thread hangs a join, and Bazel's
    // per-target `test_timeout` (the single global hang guard) fails it. The joins
    // are direct and unbounded — no in-test wall-clock budget to flake a slow run.
    shutdown_driven_pools();

    for t in std::iter::once(io)
        .chain(std::iter::once(collection_thread))
        .chain(compute)
    {
        t.join().expect("a committed drive thread joins cleanly");
    }

    let _ = std::fs::remove_dir_all(dir);
}
