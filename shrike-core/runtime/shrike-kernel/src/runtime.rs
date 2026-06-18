//! The kernel's owned tokio runtime: one runtime per process, owned
//! HERE — the walk-back of the injected-executor model. The kernel is
//! idiomatic async Rust; hosts adapt the *action exchange* (an op in, a
//! completion-backed future out via [`spawn_op`]) and never supply
//! scheduling.
//!
//! **anki retains its own runtime; the kernel pins sync work off the runtime
//! worker.** anki's rslib owns an internal lazy tokio runtime whose
//! sole consumers are the sync/AnkiWeb/AnkiHub services. Today none of those
//! is dispatched on Shrike's call paths (pinned structurally in
//! shrike-collection's `runtime_singularity` test), so anki's runtime is not
//! instantiated and the kernel's is the only one alive. Once client sync
//! lands, anki's runtime DOES come up — two runtimes per process.
//! That is fine, because the invariant the kernel actually guarantees is not
//! "one runtime" but **"a sync op never executes on a runtime worker
//! thread"**. Two facts and the discipline they imply:
//!
//! anki's sync paths call `block_on`, and [`tokio::runtime::Handle::block_on`]
//! PANICS when invoked from inside any runtime context (a worker thread is such
//! a context — see the panic-repro test below). Which runtime owns the worker
//! is irrelevant; the guard keys on the calling thread, not on runtime
//! identity.
//!
//! [`SerializedCollection`](crate::SerializedCollection) runs every collection
//! job inline as a sync closure on whichever runtime worker polls the actor. So
//! a sync anki call invoked *directly* in such a job would land on a runtime
//! worker and panic.
//!
//! The discipline that makes sync safe is therefore: **kernel-side sync ops
//! that may `block_on` (anki's sync services) MUST dispatch via
//! `spawn_blocking`** — a blocking-pool thread is NOT a runtime context, so
//! `block_on` is legal there. This is the same pattern the Python capture seams
//! already use (`py_embedder.rs` / `py_recognizer.rs`: `spawn_blocking` + GIL
//! attach), and it composes with the release-run-reopen orchestration (the
//! actor releases, the sync op rides the blocking pool, the reopen reclaims).
//!
//! The panic-repro test below pins this structurally: it demonstrates that
//! `Handle::block_on` panics on a runtime worker and succeeds on a
//! `spawn_blocking` pool thread — so a sync call cannot quietly land on a
//! runtime worker without the test catching the dispatch-site regression. (See
//! `docs/dev/decisions.md` § "anki retains its sync runtime" for why a runtime
//! handle-injection patch to anki was rejected in favour of this discipline.)
//!
//! The default is a multi-thread runtime; [`init_runtime`] lets a host (or
//! the degenerate-mode proof test) install a custom-built one — e.g.
//! `current_thread`, where the entire kernel runs on a single thread driving
//! its own asynchrony via [`block_on`]. Nothing in the kernel uses
//! `block_in_place` or assumes worker threads, which is what keeps that mode
//! honest.

use std::future::Future;
use std::sync::OnceLock;

use shrike_error::{NativeError, NativeResult};

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// Install a custom-built runtime before the first kernel op (the
/// degenerate single-thread proof; a host with tuned pools).
///
/// # Errors
///
/// Returns `Err` carrying the supplied runtime back if one is already
/// installed (the seam is set-once — the first [`handle`]/[`block_on`] call
/// installs the multi-thread default if no host did).
pub fn init_runtime(runtime: tokio::runtime::Runtime) -> Result<(), tokio::runtime::Runtime> {
    RUNTIME.set(runtime)
}

/// The kernel runtime's handle (installing the multi-thread default on
/// first use). Only the handle escapes — never the `Runtime` — so nothing
/// can drop or block the runtime from inside it.
pub(crate) fn handle() -> &'static tokio::runtime::Handle {
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .thread_name("shrike-kernel")
                .enable_all()
                .build()
                .expect("the kernel runtime must build")
        })
        .handle()
}

/// Drive a future to completion on the kernel runtime from a non-async
/// context (tests; a synchronous embedded host). Installs the multi-thread
/// default on first use if no host called [`init_runtime`]. Async callers
/// just `.await` instead.
///
/// # Panics
///
/// Panics if called from inside the runtime — a runtime worker thread is a
/// runtime context, and `tokio`'s nested-`block_on` guard refuses there. This
/// is the same guard the module's sync-dispatch discipline relies on: a
/// kernel-side sync op that may `block_on` MUST ride `spawn_blocking` (a
/// blocking-pool thread is not a runtime context), pinned by the
/// `sync_dispatch_pin` test.
pub fn block_on<F: Future>(future: F) -> F::Output {
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .thread_name("shrike-kernel")
                .enable_all()
                .build()
                .expect("the kernel runtime must build")
        })
        .block_on(future)
}

/// The action-exchange edge: spawn an op onto the kernel runtime and
/// hand back a small Send future — a oneshot receiver, pollable from ANY
/// context with no tokio dependency on the caller's side. **Dropping the
/// returned future detaches observation; the op runs to completion** (a
/// half-applied collection write from an abort would be far worse than a
/// wasted compute).
/// A oneshot that closes without a value means the op task panicked.
///
/// # Errors
///
/// The returned future yields the op's own `NativeResult`, plus an internal
/// error if the op task vanished without producing one (the oneshot closed
/// empty — i.e. the spawned task panicked).
pub fn spawn_op<T: Send + 'static>(
    future: impl Future<Output = NativeResult<T>> + Send + 'static,
) -> impl Future<Output = NativeResult<T>> + Send + 'static {
    let (tx, rx) = tokio::sync::oneshot::channel();
    handle().spawn(async move {
        let _ = tx.send(future.await);
    });
    async move {
        rx.await
            .map_err(|_| NativeError::internal("kernel op task dropped without a result"))?
    }
}

#[cfg(test)]
mod sync_dispatch_pin {
    //! The acceptance gate: pin the SYNC-OP DISPATCH PATH structurally,
    //! not by luck.
    //!
    //! anki keeps its own runtime for client sync, so two runtimes will live
    //! in the process; the invariant the kernel guarantees is **"a sync op
    //! that may `block_on` never executes on a runtime worker thread"**. This
    //! test demonstrates the two facts that make the `spawn_blocking`
    //! discipline (not luck, not a one-off no-panic run) the thing that
    //! enforces it:
    //!
    //!   1. `Handle::block_on` PANICS when called on a runtime worker thread
    //!      (the way a sync anki call would land if dispatched *directly* from
    //!      a `SerializedCollection` job — which runs inline on a worker).
    //!   2. The SAME `block_on`, against the SAME `Handle` and the SAME inner
    //!      future, SUCCEEDS on a `spawn_blocking` pool thread — which is the
    //!      mandated dispatch site (a blocking-pool thread is not a runtime
    //!      context).
    //!
    //! The only variable between the two halves is *which thread* runs
    //! `block_on`, so a regression that lets a sync call run on a runtime
    //! worker (dropping the `spawn_blocking` hop) flips half 2 from pass to
    //! panic — the test catches the dispatch-site change, not a probabilistic
    //! symptom. Self-contained: no anki source, a locally-built runtime so the
    //! process-global seam is untouched.

    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Stand-in for "a synchronous call that bottoms out in `block_on`" —
    /// exactly the shape of anki's sync service paths, minus the anki
    /// dependency. Returns a sentinel so the success half can assert the
    /// call actually ran to completion (not merely that it didn't panic).
    fn sync_call_that_blocks_on(handle: &tokio::runtime::Handle) -> u64 {
        handle.block_on(async { 0x5031_u64 })
    }

    #[test]
    fn block_on_panics_on_a_runtime_worker_but_rides_the_blocking_pool() {
        // A dedicated multi-thread runtime — never the process-global seam,
        // which other tests in this binary share.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("test runtime builds");
        let handle = rt.handle().clone();

        rt.block_on(async {
            // ── Half 1: a runtime worker thread is a runtime context, so
            // `block_on` MUST panic. Calling it directly inside this async
            // block runs it on the worker polling us. Catch the unwind so the
            // worker survives for half 2.
            let inner = handle.clone();
            let worker_result = catch_unwind(AssertUnwindSafe(|| sync_call_that_blocks_on(&inner)));
            assert!(
                worker_result.is_err(),
                "Handle::block_on must panic on a runtime worker thread — if it \
                 stopped panicking, the dispatch invariant can no longer be \
                 pinned this way (revisit #503)"
            );

            // ── Half 2: the SAME call on a `spawn_blocking` pool thread — the
            // mandated dispatch site — succeeds and returns the sentinel.
            let pooled = handle.clone();
            let value = tokio::task::spawn_blocking(move || sync_call_that_blocks_on(&pooled))
                .await
                .expect("the spawn_blocking task itself must not fail");
            assert_eq!(
                value, 0x5031,
                "block_on on a blocking-pool thread must run the sync call to \
                 completion"
            );
        });
    }
}
