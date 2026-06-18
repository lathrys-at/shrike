//! The kernel's owned tokio runtime: one runtime per process, owned here. The
//! kernel is idiomatic async Rust; hosts adapt the *action exchange* (an op in,
//! a completion-backed future out via [`spawn_op`]) and never supply scheduling.
//!
//! # Two thread-provisioning models
//!
//! The runtime runs in one of two **modes**, fixed for the process at the seam:
//!
//! - **Default — the lazily-built multi-thread runtime** ([`init_runtime`] with a
//!   multi-thread runtime, or no `init_*` at all). tokio owns and spawns its own
//!   worker + blocking-pool threads; nothing here drives it. This is the model
//!   the PyO3 server and cabi run under. The `dispatch_sync`/`dispatch_compute`
//!   enqueue helpers fall through to `Handle::spawn_blocking`, so shrike-core
//!   spawns no thread of its own beyond tokio's pool.
//!
//! - **Driven — a harness-driven `current_thread` runtime** ([`init_driven_runtime`]).
//!   The harness commits **N + 2** threads to the kernel and shrike-core spawns
//!   none of its own: [`drive_io`] (×1, owns + drives tokio's IO/timer drivers
//!   and the async executor — actor dispatch, the debounced saver's timers),
//!   [`drive_sync`] (×1, the serialized collection / anki-sync execution thread —
//!   a consequence of anki's single-writer collection), and [`drive_compute`]
//!   (×N, CPU-bound engine compute + blocking-fs leaves; the only place real
//!   parallelism lives, so the overlap property is "N ≥ 2"). Submission is either
//!   the asyncio bridge (the server keeps it) or [`submit_blocking`] (a request
//!   thread submits a unit of work and blocks on its completion).
//!
//! The mode is decided once, by which `init_*` ran (or the lazy default), and is
//! published before any `handle`/[`block_on`]/dispatch call observes it.
//!
//! # The sync-dispatch invariant
//!
//! anki retains its own runtime for client sync, so two runtimes can live in the
//! process. The invariant the kernel guarantees is not "one runtime" but **"a
//! sync op never executes on a runtime worker thread"**: anki's sync paths call
//! `block_on`, and [`tokio::runtime::Handle::block_on`] PANICS from inside any
//! runtime context (a worker thread is such a context — see the panic-repro test
//! below). The discipline that makes sync safe:
//!
//! - **Default mode:** kernel-side sync ops that may `block_on` ride
//!   `dispatch_sync`, which is `spawn_blocking` — a blocking-pool thread is NOT a
//!   runtime context, so `block_on` is legal there.
//! - **Driven mode:** `dispatch_sync` enqueues onto the [`drive_sync`] thread, a
//!   plain OS thread (never a runtime context) — so `block_on` is legal there by
//!   construction, not by a `spawn_blocking` discipline.
//!
//! Either way the [`SerializedCollection`](crate::SerializedCollection) actor
//! routes every job through `dispatch_sync`, so a sync anki call can never land
//! on a runtime worker. The `sync_dispatch_pin` test below pins this
//! structurally. `docs/dev/decisions.md` records why a runtime handle-injection
//! patch to anki was rejected in favour of this discipline.
//!
//! # The deadlock leaf-invariant
//!
//! Every pool job ([`drive_sync`] or [`drive_compute`]) is a **leaf**: the
//! read→compute→write orchestration fans out and awaits on the async side
//! ([`drive_io`]), and a pool job never enqueues-and-awaits further pool work
//! (the "discover ids → one batched read → compute" pattern keeps compute
//! collection-free), so a fixed pool can't exhaust itself. A debug-build tripwire
//! (a thread-local set inside a running pool job) asserts it — re-entering
//! `dispatch_sync`/`dispatch_compute` from within a pool job is the deadlock
//! shape and fires the assert.

use std::cell::Cell;
use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};

use shrike_error::{NativeError, NativeResult};
use tokio::sync::{mpsc, oneshot, Notify};

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// Which thread-provisioning model the process runs under. Set once,
/// alongside the runtime, before any dispatch observes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RuntimeMode {
    /// tokio owns its worker + blocking-pool threads; dispatch falls through to
    /// `spawn_blocking`.
    Default,
    /// The harness drives a `current_thread` runtime and consumes the
    /// [`drive_sync`]/[`drive_compute`] queues; dispatch enqueues onto them.
    Driven,
}

static MODE: OnceLock<RuntimeMode> = OnceLock::new();

/// A pool job: a boxed sync closure that carries its own completion channel, so
/// finishing it wakes the awaiting async task. Unbounded queues, like the
/// collection actor's channel — backpressure is the harness's committed pool
/// size, not the channel.
type PoolJob = Box<dyn FnOnce() + Send + 'static>;

/// The driven-mode work queues + their receivers, installed by
/// [`init_driven_runtime`]. The sync receiver is single-consumer (one
/// [`drive_sync`] thread); the compute receiver is shared across the N
/// [`drive_compute`] threads behind a mutex (each `recv` hands a job to one
/// waiter). `None` in default mode.
struct DrivenPools {
    /// The senders are `Option` so [`shutdown_driven_pools`] can `.take()` them:
    /// dropping every sender closes the queue, so the [`drive_sync`] /
    /// [`drive_compute`] parkers see `recv() == None` and return for the harness
    /// to join (the same end-the-loop-by-dropping-the-sender shape the
    /// collection actor uses). Held under a mutex because the static is shared
    /// and the take races concurrent enqueues. A `None` here is treated like a
    /// gone pool — the awaiting future sees the closed receiver.
    sync_tx: Mutex<Option<mpsc::UnboundedSender<PoolJob>>>,
    compute_tx: Mutex<Option<mpsc::UnboundedSender<PoolJob>>>,
    /// Taken once by the [`drive_sync`] thread.
    sync_rx: Mutex<Option<mpsc::UnboundedReceiver<PoolJob>>>,
    /// Shared by every [`drive_compute`] thread (cloned Arc; the mutex is the
    /// dequeue lock — held only to pop, never across a running job).
    compute_rx: Arc<Mutex<mpsc::UnboundedReceiver<PoolJob>>>,
    /// Tripped by [`shutdown_driven_pools`] to resolve a [`drive_io`] parked on
    /// [`drive_io_until_shutdown`] (the binding's IO thread), so all N + 2
    /// committed threads return from one shutdown call.
    shutdown: Notify,
}

impl DrivenPools {
    /// A live clone of the sync sender, or `None` once
    /// [`shutdown_driven_pools`] has taken it.
    fn sync_sender(&self) -> Option<mpsc::UnboundedSender<PoolJob>> {
        self.sync_tx
            .lock()
            .expect("driven sync sender poisoned")
            .clone()
    }

    /// A live clone of the compute sender, or `None` post-shutdown.
    fn compute_sender(&self) -> Option<mpsc::UnboundedSender<PoolJob>> {
        self.compute_tx
            .lock()
            .expect("driven compute sender poisoned")
            .clone()
    }
}

static DRIVEN: OnceLock<DrivenPools> = OnceLock::new();

thread_local! {
    /// Set while a [`drive_sync`]/[`drive_compute`] job runs, so the dispatch
    /// helpers can assert the leaf-invariant (a pool job must never enqueue-and-
    /// await further pool work). Drives a `debug_assert` only.
    static IN_POOL_JOB: Cell<bool> = const { Cell::new(false) };
}

/// Build the multi-thread default runtime (the lazy-init body, shared by
/// `handle` and [`block_on`]).
fn build_default_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .thread_name("shrike-kernel")
        .enable_all()
        .build()
        .expect("the kernel runtime must build")
}

/// The runtime, installing the multi-thread default (and `RuntimeMode::Default`)
/// on first use if no host installed one.
fn runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| {
        // First touch with no host install ⇒ the default model. A Driven host
        // always installs its runtime + mode before any lazy touch, so this
        // set only ever publishes Default (and OnceLock keeps the first either
        // way).
        let _ = MODE.set(RuntimeMode::Default);
        build_default_runtime()
    })
}

/// Install a custom-built runtime before the first kernel op (a host with tuned
/// pools; the degenerate single-thread proof). Runs in `RuntimeMode::Default`
/// — tokio drives the installed runtime itself, so dispatch keeps falling
/// through to `spawn_blocking`. A host that wants the harness-driven model uses
/// [`init_driven_runtime`] instead.
///
/// # Errors
///
/// Returns `Err` carrying the supplied runtime back if one is already installed
/// (the seam is set-once — the first `handle`/[`block_on`] call installs the
/// multi-thread default if no host did).
pub fn init_runtime(runtime: tokio::runtime::Runtime) -> Result<(), tokio::runtime::Runtime> {
    RUNTIME.set(runtime)?;
    // We won the install ⇒ we own the mode (a competing init also goes through
    // RUNTIME.set, which we just won).
    let _ = MODE.set(RuntimeMode::Default);
    Ok(())
}

/// Install a `current_thread` runtime in the **driven model**: the
/// harness will park threads in [`drive_io`]/[`drive_sync`]/[`drive_compute`] to
/// provide every thread the kernel uses. shrike-core spawns none of its own —
/// the `dispatch_sync`/`dispatch_compute` helpers enqueue onto the driven
/// queues instead of `spawn_blocking`.
///
/// The supplied runtime SHOULD be a `current_thread` runtime (that is the whole
/// point — one async thread the harness drives via [`drive_io`]). The driven
/// queues are created here, so a job submitted before [`drive_sync`] /
/// [`drive_compute`] is parked simply waits in the channel.
///
/// # Errors
///
/// Returns `Err` carrying the supplied runtime back if one is already installed.
pub fn init_driven_runtime(
    runtime: tokio::runtime::Runtime,
) -> Result<(), tokio::runtime::Runtime> {
    RUNTIME.set(runtime)?;
    // We won the runtime install, so the pools/mode below are uncontended.
    let (sync_tx, sync_rx) = mpsc::unbounded_channel::<PoolJob>();
    let (compute_tx, compute_rx) = mpsc::unbounded_channel::<PoolJob>();
    let _ = DRIVEN.set(DrivenPools {
        sync_tx: Mutex::new(Some(sync_tx)),
        compute_tx: Mutex::new(Some(compute_tx)),
        sync_rx: Mutex::new(Some(sync_rx)),
        compute_rx: Arc::new(Mutex::new(compute_rx)),
        shutdown: Notify::new(),
    });
    let _ = MODE.set(RuntimeMode::Driven);
    Ok(())
}

/// The kernel runtime's handle (installing the multi-thread default on
/// first use). Only the handle escapes — never the `Runtime` — so nothing
/// can drop or block the runtime from inside it.
pub(crate) fn handle() -> &'static tokio::runtime::Handle {
    runtime().handle()
}

/// Drive a future to completion on the kernel runtime from a non-async
/// context (tests; a synchronous embedded host; the cabi driver thread).
/// Installs the multi-thread default on first use if no host called
/// [`init_runtime`]/[`init_driven_runtime`]. Async callers just `.await`.
///
/// # Panics
///
/// Panics if called from inside the runtime — a runtime worker thread is a
/// runtime context, and `tokio`'s nested-`block_on` guard refuses there. This
/// is the same guard the module's sync-dispatch discipline relies on (pinned by
/// the `sync_dispatch_pin` test).
pub fn block_on<F: Future>(future: F) -> F::Output {
    runtime().block_on(future)
}

/// **Driven mode: own + drive the runtime until `until` resolves.** The
/// harness's one IO/timer-driver thread. The first call to the
/// runtime's `block_on` takes ownership of tokio's IO + timer drivers and the
/// async executor; every spawned task (the collection actor's dispatch loop, the
/// debounced saver's timer, the tag refresher) is polled here, and timers fire.
/// Other [`drive_sync`]/[`drive_compute`] threads do not own the drivers — they
/// hook into this one.
///
/// `until` is the harness's shutdown signal (resolved once shutdown begins and
/// in-flight work has drained); when it resolves, `drive_io` returns and the
/// harness joins it before interpreter finalization.
///
/// In **default mode** this is a misuse (tokio self-drives) and returns an
/// error — nothing in the default model calls it.
///
/// # Errors
///
/// Returns an error if called in default mode.
pub fn drive_io<F: Future<Output = ()> + Send + 'static>(until: F) -> NativeResult<()> {
    if mode() != RuntimeMode::Driven {
        return Err(NativeError::internal(
            "drive_io is a driven-mode entry point; the default runtime self-drives",
        ));
    }
    block_on(until);
    Ok(())
}

/// **Driven mode: drive the runtime until [`shutdown_driven_pools`] is called.**
/// The same as [`drive_io`] but parked on the pools' built-in shutdown signal,
/// so a host with no shutdown future of its own (the binding) gets one
/// `drive_io` thread whose `until` and the pool-queue close are tripped by one
/// call. The signal is registered BEFORE the wait so a shutdown that races this
/// thread's start is not missed.
///
/// # Errors
///
/// Returns an error if called in default mode, or if no driven pools are
/// installed.
pub fn drive_io_until_shutdown() -> NativeResult<()> {
    if mode() != RuntimeMode::Driven {
        return Err(NativeError::internal(
            "drive_io_until_shutdown is a driven-mode entry point; the default runtime self-drives",
        ));
    }
    let pools = DRIVEN.get().ok_or_else(driven_missing)?;
    block_on(async {
        // `notified()` registers the waiter before the await, so a `notify_one`
        // (or the permit `notify_waiters` sets) that happens after this point is
        // observed — no lost wakeup against a racing shutdown.
        pools.shutdown.notified().await;
    });
    Ok(())
}

/// **Driven mode: the serialized collection / anki-sync execution thread**
/// A plain OS thread blocking on the sync work queue and running each
/// job to completion. One thread is a *consequence* of anki's single-writer
/// collection (reads and writes serialize; anki forbids concurrent access), not
/// a tuning choice. Because this thread is never inside the kernel runtime
/// context, anki's own `block_on` is legal here — the structural form of the
/// sync-never-on-a-runtime-worker invariant.
///
/// Parks until the queue is closed (every sender dropped — i.e. harness
/// shutdown), then returns so the harness can join it.
///
/// # Errors
///
/// Returns an error if called in default mode, or if the sync queue was already
/// claimed by a prior `drive_sync` (exactly one thread drives it).
///
/// # Panics
///
/// Panics if the sync-receiver mutex is poisoned (a prior holder panicked).
pub fn drive_sync() -> NativeResult<()> {
    if mode() != RuntimeMode::Driven {
        return Err(NativeError::internal(
            "drive_sync is a driven-mode entry point; the default runtime self-drives",
        ));
    }
    let pools = DRIVEN.get().ok_or_else(driven_missing)?;
    let mut rx = pools
        .sync_rx
        .lock()
        .expect("driven sync receiver poisoned")
        .take()
        .ok_or_else(|| NativeError::internal("drive_sync was already claimed by another thread"))?;
    while let Some(job) = rx.blocking_recv() {
        job();
    }
    Ok(())
}

/// **Driven mode: a CPU-bound engine-compute (and blocking-fs leaf) worker**
/// The harness spawns N of these; each blocks on the shared compute
/// queue and runs each job to completion. This is the only place real
/// parallelism lives (independent batches), so the engine search/batch overlap
/// property becomes "N ≥ 2", sized by the harness to its cores. Dispatch target
/// for the `Blocking<E>` adapter, the tag-centroid recompute, the
/// index file save, the derived FTS5 rebuild, and the store-media decode.
///
/// Parks until the queue is closed, then returns. Multiple `drive_compute`
/// threads share one queue — each `recv` hands a job to exactly one waiter, so N
/// parkers cooperate.
///
/// # Errors
///
/// Returns an error if called in default mode.
///
/// # Panics
///
/// Panics if the shared compute-receiver mutex is poisoned (a prior holder
/// panicked).
pub fn drive_compute() -> NativeResult<()> {
    if mode() != RuntimeMode::Driven {
        return Err(NativeError::internal(
            "drive_compute is a driven-mode entry point; the default runtime self-drives",
        ));
    }
    let pools = DRIVEN.get().ok_or_else(driven_missing)?;
    let rx = Arc::clone(&pools.compute_rx);
    loop {
        // Hold the dequeue lock only to pop; run the job lock-free so N workers
        // genuinely overlap.
        let job = {
            let mut guard = rx.lock().expect("driven compute receiver poisoned");
            guard.blocking_recv()
        };
        match job {
            Some(job) => job(),
            None => break, // every sender dropped ⇒ shutdown
        }
    }
    Ok(())
}

/// **Driven mode: signal every committed thread to return so the harness can
/// join them.** Drops the pool senders (closing the [`drive_sync`] /
/// [`drive_compute`] queues, so their `recv` yields `None` and they return) and
/// trips the [`drive_io_until_shutdown`] signal. Call it once, AFTER kernel work
/// has quiesced (the collection actor drained), so no in-flight enqueue is
/// holding a transient sender clone — then the queues close promptly and the
/// joins are immediate. Idempotent: a second call finds the senders already
/// taken and only re-trips the signal.
///
/// In **default mode** there are no committed threads to signal, so this is a
/// no-op (tokio owns its own pool).
///
/// # Panics
///
/// Panics if a sender-slot mutex is poisoned (a prior holder panicked).
pub fn shutdown_driven_pools() {
    let Some(pools) = DRIVEN.get() else {
        return; // default mode (or no driven install) — nothing to signal
    };
    // Drop the senders: this is what closes the queues. Each parker returns
    // once its receiver sees every sender gone — the transient clones held by
    // an in-flight enqueue drop as that call returns, so a quiesced kernel
    // closes immediately.
    drop(
        pools
            .sync_tx
            .lock()
            .expect("driven sync sender poisoned")
            .take(),
    );
    drop(
        pools
            .compute_tx
            .lock()
            .expect("driven compute sender poisoned")
            .take(),
    );
    // Wake the IO thread. `notify_one` stores a permit if no waiter is parked
    // yet, so a shutdown racing `drive_io_until_shutdown`'s start is not lost.
    pools.shutdown.notify_one();
}

/// **Submit a unit of (possibly batched) blocking work and block until it
/// completes** — the submission path for THREADED harnesses (cabi, a
/// synchronous host, tests). Submits onto the driven compute pool and blocks the
/// CALLING (request) thread on a completion channel; it must never run on an
/// async executor thread (it blocks). The asyncio server does NOT use this — it
/// keeps the bridge (`spawn_op` + an awaited `asyncio.Future`).
///
/// In **default mode** there is no driven pool, so the work runs inline on the
/// calling thread — correct for a synchronous host with no committed pool, and
/// it preserves the "block until complete, then return the value" contract.
///
/// # Errors
///
/// Propagates the work's own `NativeResult`; an internal error if the driven
/// compute worker vanished without producing one (the pool shut down mid-flight).
pub fn submit_blocking<T: Send + 'static>(
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> NativeResult<T> {
    debug_assert!(
        !IN_POOL_JOB.with(Cell::get),
        "leaf-invariant: a pool job must not submit-and-block on further pool work (submit_blocking)"
    );
    match mode() {
        RuntimeMode::Driven => {
            let (tx, rx) = std::sync::mpsc::channel();
            let job: PoolJob = Box::new(move || {
                let _ = tx.send(run_in_pool_job(work));
            });
            DRIVEN
                .get()
                .ok_or_else(driven_missing)?
                .compute_sender()
                .ok_or_else(|| NativeError::internal("the compute pool is gone"))?
                .send(job)
                .map_err(|_| NativeError::internal("the compute pool is gone"))?;
            rx.recv()
                .map_err(|_| NativeError::internal("the compute worker dropped a job"))?
        }
        RuntimeMode::Default => run_in_pool_job(work),
    }
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
    let (tx, rx) = oneshot::channel();
    handle().spawn(async move {
        let _ = tx.send(future.await);
    });
    async move {
        rx.await
            .map_err(|_| NativeError::internal("kernel op task dropped without a result"))?
    }
}

// ── dispatch: route blocking work to the right pool, mode-agnostically ───────

/// Run a unit of **anki-collection / sync** blocking work, returning an eagerly-
/// scheduled future of its result. The collection actor and every kernel-side
/// sync op route through here so a sync `block_on` can never land on a runtime
/// worker:
///
/// - **Driven mode:** enqueue onto the [`drive_sync`] thread (a non-runtime
///   context — anki `block_on` legal by construction).
/// - **Default mode:** `tokio::task::spawn_blocking` (a blocking-pool thread —
///   not a runtime context).
///
/// **Eager by contract** (like the engine `Blocking` adapter): the work is
/// scheduled inside this call, before the returned future is first polled.
///
/// # Errors
///
/// The future yields the work's own `NativeResult`, or an internal error if the
/// executing thread/pool vanished without producing one.
pub(crate) fn dispatch_sync<T: Send + 'static>(
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> impl Future<Output = NativeResult<T>> + Send + 'static {
    debug_assert!(
        !IN_POOL_JOB.with(Cell::get),
        "leaf-invariant: a pool job must not enqueue-and-await further pool work (dispatch_sync)"
    );
    enqueue(QueueKind::Sync, work)
}

/// Run a unit of **CPU-bound compute / blocking-fs** work, returning an eagerly-
/// scheduled future of its result. The engine `Blocking` adapter, the
/// tag-centroid recompute, the index file save, the derived FTS5 rebuild, and
/// the store-media decode route through here.
///
/// - **Driven mode:** enqueue onto the [`drive_compute`] pool (N threads).
/// - **Default mode:** `Handle::spawn_blocking`.
///
/// **Eager by contract**: the work is scheduled inside this call.
///
/// # Errors
///
/// The future yields the work's own `NativeResult`, or an internal error if the
/// executing pool vanished without producing one.
pub(crate) fn dispatch_compute<T: Send + 'static>(
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> impl Future<Output = NativeResult<T>> + Send + 'static {
    debug_assert!(
        !IN_POOL_JOB.with(Cell::get),
        "leaf-invariant: a pool job must not enqueue-and-await further pool work (dispatch_compute)"
    );
    enqueue(QueueKind::Compute, work)
}

/// **Schedule a fire-and-forget compute job on the blocking pool, eagerly** —
/// the seam the engine `Blocking` adapter's injected dispatcher calls. The job
/// is the type-erased closure that owns its own result channel (engine-api wraps
/// the engine compute so the awaiting future learns the outcome), so this only
/// has to run it on the right pool:
///
/// - **Driven mode:** enqueue onto the [`drive_compute`] pool (N threads, the
///   N ≥ 2 engine overlap).
/// - **Default mode:** `Handle::spawn_blocking` (tokio's blocking pool).
///
/// **Eager by contract**: the job is queued/scheduled inside this call, before
/// control returns to the adapter — what keeps the engine future in flight
/// before its first poll (the search/add overlap property).
///
/// The job runs through [`run_in_pool_job`] for the same panic containment and
/// leaf-invariant tripwire every kernel pool job gets: a panicking engine job
/// loses only itself (its result channel drops, the awaiting future gets a clean
/// error), the pool thread survives. If the pool is gone (shutdown) the job is
/// dropped; its result channel closes and the awaiting future sees the error.
pub fn submit_compute(job: Box<dyn FnOnce() + Send + 'static>) {
    debug_assert!(
        !IN_POOL_JOB.with(Cell::get),
        "leaf-invariant: a pool job must not submit further pool work (submit_compute)"
    );
    let contained: PoolJob = Box::new(move || {
        let _ = run_in_pool_job(move || {
            job();
            Ok::<(), NativeError>(())
        });
    });
    match mode() {
        RuntimeMode::Driven => {
            if let Some(sender) = DRIVEN.get().and_then(DrivenPools::compute_sender) {
                let _ = sender.send(contained);
            }
        }
        RuntimeMode::Default => {
            handle().spawn_blocking(contained);
        }
    }
}

/// Which driven queue an enqueue targets.
#[derive(Clone, Copy)]
enum QueueKind {
    Sync,
    Compute,
}

/// The shared body of `dispatch_sync`/`dispatch_compute`: branch on mode,
/// enqueue (driven) or `spawn_blocking` (default), and return an eager future of
/// the result. Boxed so the two arms unify to one return type.
fn enqueue<T: Send + 'static>(
    kind: QueueKind,
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> futures::future::BoxFuture<'static, NativeResult<T>> {
    match mode() {
        RuntimeMode::Driven => {
            let (tx, rx) = oneshot::channel();
            let job: PoolJob = Box::new(move || {
                let _ = tx.send(run_in_pool_job(work));
            });
            // If the queue is gone (shutdown took the sender), the receiver
            // closes empty → the internal error below.
            if let Some(pools) = DRIVEN.get() {
                let sender = match kind {
                    QueueKind::Sync => pools.sync_sender(),
                    QueueKind::Compute => pools.compute_sender(),
                };
                if let Some(sender) = sender {
                    let _ = sender.send(job);
                }
            }
            Box::pin(async move {
                rx.await
                    .map_err(|_| NativeError::internal("the driven pool dropped a job"))?
            })
        }
        RuntimeMode::Default => {
            // `Handle::spawn_blocking` (not the free `tokio::task::spawn_blocking`)
            // so dispatch works from a non-async context too — the debounced
            // saver's burst-cap path calls in synchronously.
            let join = handle().spawn_blocking(move || run_in_pool_job(work));
            Box::pin(async move {
                join.await
                    .map_err(|_| NativeError::internal("the blocking task failed"))?
            })
        }
    }
}

/// The current mode (defaulting via the lazy runtime if neither `init_*` ran).
fn mode() -> RuntimeMode {
    if let Some(m) = MODE.get() {
        return *m;
    }
    // No mode set yet ⇒ touch the runtime, which sets Default on lazy init.
    let _ = runtime();
    *MODE.get().unwrap_or(&RuntimeMode::Default)
}

/// Run a pool job body with the leaf-invariant tripwire armed AND its panic
/// contained — the one place both modes converge, so resilience is uniform
/// (default==driven) and DRY.
///
/// - **Tripwire**: sets the `IN_POOL_JOB` thread-local for the duration, so a
///   `dispatch_*` called from within asserts (debug builds). An RAII guard
///   clears it even on unwind, so a panicking job can't leave it armed on a
///   reused blocking-pool thread (default mode) and spuriously trip a later
///   legitimate dispatch.
/// - **Panic containment**: `work` is run under `catch_unwind` and a caught
///   panic becomes `Err(Internal)` rather than unwinding out. In DRIVEN mode the
///   pool runs jobs on a long-lived OS thread with no per-job isolation, so an
///   uncaught panic would KILL that thread — and the single `drive_sync` thread
///   dying would wedge every future collection op (its receiver was taken with
///   no replacement). Default mode's `spawn_blocking` already isolates a panic
///   (tokio turns it into a `JoinError`) and reuses the thread; catching here
///   gives DRIVEN the SAME "a panic loses only that one job, the pool survives,
///   the caller gets a clean Err" resilience — the spec's default==driven parity.
fn run_in_pool_job<T>(work: impl FnOnce() -> NativeResult<T>) -> NativeResult<T> {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            IN_POOL_JOB.with(|f| f.set(false));
        }
    }
    IN_POOL_JOB.with(|f| f.set(true));
    let _disarm = Disarm;
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(result) => result,
        Err(payload) => {
            // Recover a human-readable message from the panic payload for the
            // log + the returned Err (the most common payload shapes).
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            tracing::error!(panic = %what, "a pool job panicked; the pool thread survives");
            Err(NativeError::internal(format!("pool job panicked: {what}")))
        }
    }
}

fn driven_missing() -> NativeError {
    NativeError::internal("driven mode has no pools installed")
}

#[cfg(test)]
mod sync_dispatch_pin {
    //! The acceptance gate: pin the sync-op dispatch path structurally, not by
    //! luck.
    //!
    //! anki keeps its own runtime for client sync, so two runtimes will live in
    //! the process; the invariant the kernel guarantees is **"a sync op that may
    //! `block_on` never executes on a runtime worker thread"**.
    //!
    //! - **Half 1** demonstrates the hazard is real: `Handle::block_on` PANICS
    //!   on a runtime worker thread (the way a sync anki call would land if
    //!   dispatched *directly* from a `SerializedCollection` job inline on a
    //!   worker).
    //! - **Half 2** demonstrates the mandated dispatch site is safe: the SAME
    //!   call on a `spawn_blocking` pool thread — the DEFAULT-mode dispatch
    //!   target — succeeds and returns the sentinel.
    //!
    //! The only variable between the halves is *which thread* runs `block_on`, so
    //! a regression that lets a sync call run on a runtime worker flips Half 2
    //! from pass to panic. The DRIVEN-mode structural form of the invariant
    //! (sync runs on the non-runtime `drive_sync` thread) is pinned end-to-end
    //! by the `driven_mode.rs` integration binary, which can install the
    //! process-global driven seam without colliding with this in-process suite.
    //! Self-contained here: a locally-built runtime so the seam is untouched.

    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Stand-in for "a synchronous call that bottoms out in `block_on`" —
    /// exactly the shape of anki's sync service paths, minus the anki
    /// dependency. Returns a sentinel so the success half can assert the call
    /// actually ran to completion (not merely that it didn't panic).
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
                 pinned this way"
            );

            // ── Half 2: the SAME call on a `spawn_blocking` pool thread — the
            // DEFAULT-mode dispatch target — succeeds and returns the sentinel.
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

#[cfg(test)]
mod leaf_invariant {
    //! The deadlock leaf-invariant tripwire: a pool job must never
    //! enqueue-and-await further pool work. These run in the multi-thread suite
    //! (default mode), where `dispatch_compute` is `spawn_blocking`.

    use super::*;

    /// A WELL-FORMED op — `dispatch_compute` called from the async side, its job
    /// a pure leaf — completes and never trips the assert.
    #[test]
    fn well_formed_dispatch_passes() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let out: NativeResult<u64> =
            rt.block_on(async { dispatch_compute(|| Ok(0x5031_u64)).await });
        assert_eq!(out.unwrap(), 0x5031);
    }

    /// A leaf job that RE-ENTERS dispatch (the deadlock shape) trips the
    /// debug-build tripwire. The nested dispatch happens INSIDE a running pool
    /// job (the tripwire's thread-local is set), so the `debug_assert!` in
    /// `dispatch_compute` panics. In default mode the outer job runs on a
    /// `spawn_blocking` thread, so tokio converts that panic into a `JoinError`
    /// — surfacing as the future's `Err`, NOT an unwind to the caller. The
    /// debug-build outcome is therefore the outer op resolving `Err`; release
    /// builds compile the assert out and the op resolves `Ok`.
    #[test]
    fn nested_pool_dispatch_trips_the_tripwire_in_debug() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let handle = rt.handle().clone();
        let out: NativeResult<()> = rt.block_on(async {
            // The OUTER pool job runs `run_in_pool_job` (sets the flag), and
            // from inside it synchronously builds a nested dispatch — exactly the
            // forbidden enqueue-from-a-pool-job shape.
            let h2 = handle.clone();
            dispatch_compute(move || {
                // Inside a pool job now (flag set). Building a nested dispatch
                // future trips the debug_assert. Enter the runtime so the
                // default-mode `spawn_blocking` path can schedule.
                let _enter = h2.enter();
                let _nested = dispatch_compute(|| Ok::<(), NativeError>(()));
                Ok::<(), NativeError>(())
            })
            .await
        });
        if cfg!(debug_assertions) {
            assert!(
                out.is_err(),
                "a pool job re-entering dispatch must trip the leaf-invariant \
                 debug_assert — the outer job panics, surfacing as the blocking \
                 task's JoinError"
            );
        } else {
            assert!(out.is_ok(), "release: the assert is compiled out");
        }
    }

    /// `submit_compute` (default mode) schedules the fire-and-forget job on the
    /// blocking pool — it runs to completion off the calling thread. The job
    /// carries its own result channel (the engine `Blocking` adapter's shape), so
    /// we observe completion + the thread it ran on through one.
    #[test]
    fn submit_compute_runs_off_thread_in_default_mode() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let _guard = rt.enter();
        let caller = std::thread::current().id();
        let (tx, rx) = std::sync::mpsc::channel();
        submit_compute(Box::new(move || {
            let _ = tx.send(std::thread::current().id());
        }));
        let ran_on = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("submit_compute scheduled the job on the blocking pool");
        assert_ne!(
            ran_on, caller,
            "submit_compute ran the job off the calling thread (on the blocking pool)"
        );
    }
}
