//! The kernel's owned tokio runtime: one runtime per process, owned here. The
//! kernel is idiomatic async Rust; hosts adapt the *action exchange* (an op in,
//! a completion-backed future out via [`spawn_op`]) and never supply scheduling.
//!
//! # The harness-driven runtime — the sole thread-provisioning model
//!
//! The kernel runs a single `current_thread` tokio runtime ([`init_driven_runtime`])
//! and spawns **no threads of its own**: the harness commits **N + 2** threads
//! and drives every one. [`drive_io`] (×1) owns + drives tokio's IO/timer drivers
//! and the async executor (actor dispatch, the debounced saver's timers);
//! [`drive_collection`] (×1) is the serialized collection execution thread —
//! every anki-collection op runs here, a consequence of anki's single-writer
//! collection; [`drive_compute`] (×N) runs
//! CPU-bound engine compute + blocking-fs leaves — the only place real
//! parallelism lives, so the overlap property is "N ≥ 2". Submission is either
//! the asyncio bridge (the server) or [`submit_blocking`] (a request thread
//! submits a unit of work and blocks on its completion).
//!
//! The runtime MUST be installed via [`init_driven_runtime`] before any kernel
//! op: there is no lazy fallback, so [`handle`]/[`block_on`]/dispatch from an
//! uninstalled runtime panics (a setup error — the harness installs first).
//!
//! # The collection-dispatch invariant
//!
//! anki retains its own runtime for client sync, so two runtimes can live in the
//! process. The invariant the kernel guarantees is not "one runtime" but **"a
//! collection op never executes on a runtime worker thread"**: anki's sync paths
//! call `block_on`, and [`tokio::runtime::Handle::block_on`] PANICS from inside
//! any runtime context (a worker thread is such a context — see the panic-repro
//! test below). `dispatch_collection` enqueues onto the [`drive_collection`]
//! thread, a plain OS thread (never a runtime context) — so anki's `block_on` is
//! legal there by construction. The
//! [`SerializedCollection`](crate::SerializedCollection) actor routes every job
//! through `dispatch_collection`, so a collection op (including anki's sync
//! `block_on` when client sync lands) can never run on a runtime worker. The
//! `collection_dispatch_pin` test below pins this structurally.
//! `docs/dev/decisions.md` records why a runtime handle-injection patch to anki
//! was rejected in favour of this discipline.
//!
//! # The deadlock leaf-invariant
//!
//! Every pool job ([`drive_collection`] or [`drive_compute`]) is a **leaf**: the
//! read→compute→write orchestration fans out and awaits on the async side
//! ([`drive_io`]), and a pool job never enqueues-and-awaits further pool work
//! (the "discover ids → one batched read → compute" pattern keeps compute
//! collection-free), so a fixed pool can't exhaust itself. A debug-build tripwire
//! (a thread-local set inside a running pool job) asserts it — re-entering
//! `dispatch_collection`/`dispatch_compute` from within a pool job is the deadlock
//! shape and fires the assert.

use std::cell::Cell;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crossbeam_deque::{Injector, Stealer, Worker as Deque};
use metrics::{
    counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram, Counter,
    Gauge, Histogram, Unit,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use shrike_error::{NativeError, NativeResult};
use tokio::sync::{mpsc, oneshot, Notify};

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// A pool job: a boxed sync closure that carries its own completion channel, so
/// finishing it wakes the awaiting async task, plus the enqueue instant so the
/// worker that runs it records the queue-wait latency. Unbounded queues, like the
/// collection actor's channel — backpressure is the harness's committed pool
/// size, not the channel.
struct PoolJob {
    work: Box<dyn FnOnce() + Send + 'static>,
    queued_at: Instant,
}

impl PoolJob {
    fn new(work: Box<dyn FnOnce() + Send + 'static>) -> Self {
        Self {
            work,
            queued_at: Instant::now(),
        }
    }
}

/// The installed Prometheus pull exporter handle. [`render_prometheus`] renders
/// it for the control-plane /metrics body (appended after the Python registry).
static PROMETHEUS: OnceLock<PrometheusHandle> = OnceLock::new();

/// Install the Prometheus recorder and describe the kernel instruments — once per
/// process, from [`init_driven_runtime`] after the runtime install wins.
/// Recommended naming lets the exporter derive `_total`/`_seconds`/`le`, so the
/// instrument macros never hand-format a Prometheus suffix. All aggregation and
/// snapshotting lives in the exporter (valid histograms by construction).
fn install_prometheus() {
    let _ = PROMETHEUS.get_or_init(|| {
        let handle = PrometheusBuilder::new()
            .with_recommended_naming(true)
            .install_recorder()
            .expect("the Prometheus recorder installs exactly once per process");
        describe_kernel_metrics();
        gauge!("shrike_index_saver_pending").set(0.0);
        handle
    });
}

/// Render the kernel's Prometheus registry. Empty before the exporter installs
/// (the cabi / test / minimal-core paths that never call [`init_driven_runtime`]).
pub fn render_prometheus() -> String {
    PROMETHEUS
        .get()
        .map(PrometheusHandle::render)
        .unwrap_or_default()
}

/// HELP/TYPE/unit metadata for every kernel instrument. Units (`Unit::Seconds`)
/// drive the exporter's `_seconds` suffix; the bare counter names gain `_total`.
fn describe_kernel_metrics() {
    describe_gauge!(
        "shrike_runtime_pool_workers",
        "Live workers driving a pool."
    );
    describe_gauge!(
        "shrike_runtime_pool_active_jobs",
        "Jobs currently executing in a pool."
    );
    describe_gauge!(
        "shrike_runtime_pool_queue_depth",
        "Jobs waiting in a pool's queue."
    );
    describe_counter!("shrike_runtime_pool_jobs", "Jobs completed by a pool.");
    describe_histogram!(
        "shrike_runtime_pool_queue_wait",
        Unit::Seconds,
        "Time jobs waited in queue before execution."
    );
    describe_histogram!(
        "shrike_runtime_pool_job_duration",
        Unit::Seconds,
        "Time spent executing pool jobs."
    );
    describe_gauge!(
        "shrike_runtime_io_alive",
        "Whether the drive_io thread is live."
    );
    describe_counter!(
        "shrike_embedding_batches",
        "Embedding batches by space, modality, operation, and result."
    );
    describe_counter!(
        "shrike_embedding_items",
        "Items submitted for embedding by space, modality, operation, and result."
    );
    describe_histogram!(
        "shrike_embedding_duration",
        Unit::Seconds,
        "Embedding batch latency."
    );
    describe_counter!(
        "shrike_index_saver_runs",
        "Debounced index saver flushes by result."
    );
    describe_counter!(
        "shrike_index_saver_requests",
        "Debounced index saver save-requests."
    );
    describe_gauge!(
        "shrike_index_saver_pending",
        "Unsaved index changes awaiting the debounced saver."
    );
    describe_histogram!(
        "shrike_index_saver_duration",
        Unit::Seconds,
        "Debounced index flush latency."
    );
}

/// The cached per-pool runtime instrument handles, resolved once at install with
/// the `pool` label baked in. The hot worker path emits through these via a
/// direct atomic rather than re-resolving a metric key per job.
struct PoolInstruments {
    workers: Gauge,
    active: Gauge,
    queue_depth: Gauge,
    jobs: Counter,
    queue_wait: Histogram,
    job_duration: Histogram,
}

impl PoolInstruments {
    fn for_pool(pool: &'static str) -> Self {
        Self {
            workers: gauge!("shrike_runtime_pool_workers", "pool" => pool),
            active: gauge!("shrike_runtime_pool_active_jobs", "pool" => pool),
            queue_depth: gauge!("shrike_runtime_pool_queue_depth", "pool" => pool),
            jobs: counter!("shrike_runtime_pool_jobs", "pool" => pool),
            queue_wait: histogram!("shrike_runtime_pool_queue_wait", "pool" => pool),
            job_duration: histogram!("shrike_runtime_pool_job_duration", "pool" => pool),
        }
    }
}

/// RAII worker-liveness accounting: holds the `workers` gauge up for the lifetime
/// of a [`drive_collection`]/[`drive_compute`] thread, decrementing on drop — so a
/// worker that unwinds on a poisoned-mutex `.expect()` still decrements and the
/// registry never strands a dead worker as live.
struct WorkerGuard {
    workers: Gauge,
}

impl WorkerGuard {
    fn enter(workers: &Gauge) -> Self {
        workers.increment(1.0);
        Self {
            workers: workers.clone(),
        }
    }
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        self.workers.decrement(1.0);
    }
}

/// The embedding-call context the kernel op sets around an embed `.await`, so the
/// observed embedder (which sees only the batch) can attribute the batch to its
/// space and query-vs-index operation. `space` is the embedding-space key;
/// `operation` is `"query"` (search-time) or `"index"` (write/build-time).
#[derive(Clone)]
pub struct EmbedContext {
    /// The embedding-space key the batch ran against.
    pub space: String,
    /// `"query"` (search-time) or `"index"` (write/build-time).
    pub operation: &'static str,
}

tokio::task_local! {
    static EMBED_CONTEXT: EmbedContext;
}

/// Scope `fut` (an embed call) with its embedding [`EmbedContext`], so a
/// [`record_embedding`] emitted while the future resolves carries the right
/// `space`/`operation` labels. The observed embedder records the batch; the op
/// boundary names what kind of embed it was.
pub fn embed_scope<F: Future>(
    space: String,
    operation: &'static str,
    fut: F,
) -> impl Future<Output = F::Output> {
    use tracing::Instrument as _;
    let span = tracing::debug_span!("kernel.embed", operation);
    EMBED_CONTEXT.scope(EmbedContext { space, operation }, fut.instrument(span))
}

/// Record one production embedding batch, reading the `space`/`operation` from
/// the ambient [`EmbedContext`] (defaulting to `unknown` outside an
/// [`embed_scope`], e.g. a direct embed with no op boundary). Success and failure
/// are independent counter series (`result="ok"` / `result="error"`), never a
/// subtraction — a concurrent scrape can't observe a non-monotonic derived count.
pub fn record_embedding(modality: &str, items: usize, elapsed: Duration, success: bool) {
    if items == 0 {
        // An empty batch does no work — counting it inflates the batch/items
        // totals and biases the latency histogram low.
        return;
    }
    let (space, operation) = EMBED_CONTEXT
        .try_with(|c| (c.space.clone(), c.operation))
        .unwrap_or_else(|_| ("unknown".to_owned(), "unknown"));
    let result = if success { "ok" } else { "error" };
    counter!(
        "shrike_embedding_batches",
        "space" => space.clone(),
        "modality" => modality.to_owned(),
        "operation" => operation,
        "result" => result,
    )
    .increment(1);
    counter!(
        "shrike_embedding_items",
        "space" => space.clone(),
        "modality" => modality.to_owned(),
        "operation" => operation,
        "result" => result,
    )
    .increment(items as u64);
    histogram!(
        "shrike_embedding_duration",
        "space" => space,
        "modality" => modality.to_owned(),
        "operation" => operation,
    )
    .record(elapsed.as_secs_f64());
}

/// Record a debounced index-saver flush — exactly once per physical save, with
/// `result` as independent ok/error series. `pending` is the live unsaved-change
/// count read from the saver at the call site, so the gauge tracks the real
/// backlog rather than a mirror that drifts.
pub(crate) fn record_saver_run(success: bool, elapsed: Duration, pending: u64) {
    let result = if success { "ok" } else { "error" };
    counter!("shrike_index_saver_runs", "result" => result).increment(1);
    histogram!("shrike_index_saver_duration", "result" => result).record(elapsed.as_secs_f64());
    gauge!("shrike_index_saver_pending").set(pending as f64);
}

/// Record a debounced index-saver save-request, refreshing the pending gauge from
/// the saver's live count.
pub(crate) fn record_saver_request(pending: u64) {
    counter!("shrike_index_saver_requests").increment(1);
    gauge!("shrike_index_saver_pending").set(pending as f64);
}

/// The driven pools, installed by [`init_driven_runtime`]. The collection queue
/// is a single-consumer mpsc (one [`drive_collection`] thread pops it); the
/// compute pool is a [`ComputePool`] — a work-stealing pool over the N
/// [`drive_compute`] threads. Present once the harness has installed the driven
/// runtime.
struct DrivenPools {
    /// The sender is `Option` so [`shutdown_driven_pools`] can `.take()` it:
    /// dropping it closes the queue, so [`drive_collection`] sees `recv() == None`
    /// and returns for the harness to join. Held under a mutex because the static
    /// is shared and the take races concurrent enqueues.
    collection_tx: Mutex<Option<mpsc::UnboundedSender<PoolJob>>>,
    /// Taken once by the [`drive_collection`] thread.
    collection_rx: Mutex<Option<mpsc::UnboundedReceiver<PoolJob>>>,
    /// The work-stealing pool the N [`drive_compute`] threads run.
    compute: ComputePool,
    /// Tripped by [`shutdown_driven_pools`] to resolve a [`drive_io`] parked on
    /// [`drive_io_until_shutdown`] (the binding's IO thread), so all N + 2
    /// committed threads return from one shutdown call.
    shutdown: Notify,
    /// Cached per-pool instrument handles (resolved at install) — the hot worker
    /// path emits through these without re-resolving a metric key per job.
    collection_metrics: PoolInstruments,
    compute_metrics: PoolInstruments,
}

impl DrivenPools {
    /// A live clone of the collection sender, or `None` once
    /// [`shutdown_driven_pools`] has taken it.
    fn collection_sender(&self) -> Option<mpsc::UnboundedSender<PoolJob>> {
        self.collection_tx
            .lock()
            .expect("driven collection sender poisoned")
            .clone()
    }

    /// The cached instruments for `kind`.
    fn metrics(&self, kind: QueueKind) -> &PoolInstruments {
        match kind {
            QueueKind::Collection => &self.collection_metrics,
            QueueKind::Compute => &self.compute_metrics,
        }
    }

    /// Run one job on the calling worker, recording the queue-wait, active-jobs,
    /// duration, and completed-jobs instruments for `kind`. The job body is
    /// panic-contained ([`run_in_pool_job`] runs under `catch_unwind`), so the
    /// active gauge cannot strand between increment and decrement.
    fn run_job(&self, kind: QueueKind, job: PoolJob) {
        let m = self.metrics(kind);
        m.queue_depth.decrement(1.0);
        m.queue_wait.record(job.queued_at.elapsed().as_secs_f64());
        m.active.increment(1.0);
        let started = Instant::now();
        (job.work)();
        m.job_duration.record(started.elapsed().as_secs_f64());
        m.active.decrement(1.0);
        m.jobs.increment(1);
    }
}

/// A **work-stealing thread pool** over the N committed [`drive_compute`]
/// workers. A lock-free global [`Injector`] is the submission queue; each worker
/// owns a local deque and, when it runs dry, steals a batch from the injector or
/// a single task from a peer (see [`find_task`]). The hot path — a busy worker
/// popping its own deque — touches no shared lock; only a worker that finds *no*
/// work parks on a condvar. This replaces the earlier single `Mutex<Receiver>`
/// every worker locked to pop one job, which serialized the whole pool on that
/// one lock under load.
struct ComputePool {
    /// The submission queue. [`dispatch_compute`] / [`submit_compute`] /
    /// [`submit_blocking`] push here; workers steal batches out.
    injector: Injector<PoolJob>,
    /// One stealer per worker deque, built at construction. Read-only; a worker
    /// steals from peers through it.
    stealers: Vec<Stealer<PoolJob>>,
    /// The worker deques, built in [`ComputePool::new`] (one per committed
    /// worker) and claimed one-each by the [`drive_compute`] threads at startup.
    locals: Mutex<Vec<Deque<PoolJob>>>,
    /// Idle-worker parking. A worker that finds no work waits on `unpark` under
    /// `park`; a submission or shutdown notifies under the same lock, so the
    /// check-then-park and the wake can never interleave into a lost wakeup.
    park: Mutex<()>,
    unpark: Condvar,
    /// Set by [`shutdown_driven_pools`]; a worker that drains to empty under it
    /// returns for the harness to join.
    shutdown: AtomicBool,
}

impl ComputePool {
    /// Build the pool with `workers` local deques — one per committed
    /// [`drive_compute`] thread — and their stealers.
    fn new(workers: usize) -> Self {
        let deques: Vec<Deque<PoolJob>> = (0..workers).map(|_| Deque::new_fifo()).collect();
        let stealers = deques.iter().map(Deque::stealer).collect();
        Self {
            injector: Injector::new(),
            stealers,
            locals: Mutex::new(deques),
            park: Mutex::new(()),
            unpark: Condvar::new(),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Push a job and wake one idle worker. Returns `false` (dropping the job, so
    /// its result channel closes and the awaiting future errors) when the pool is
    /// shutting down. The notify is taken under `park` so a worker mid
    /// check-then-park either observes this job in its re-check or is woken by it.
    fn push(&self, job: PoolJob) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        self.injector.push(job);
        let _g = self.park.lock().expect("compute park poisoned");
        self.unpark.notify_one();
        true
    }

    /// Whether the injector or any peer deque holds work — the re-check a worker
    /// runs under `park` before committing to a wait.
    fn has_work(&self) -> bool {
        !self.injector.is_empty() || self.stealers.iter().any(|s| !s.is_empty())
    }

    /// Signal shutdown and wake every parked worker so it can drain and return.
    fn begin_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        let _g = self.park.lock().expect("compute park poisoned");
        self.unpark.notify_all();
    }
}

/// Find the next job for a worker: pop its own deque, else steal a batch from the
/// global injector or a single task from a peer, retrying while a steal reports a
/// transient conflict. `None` only once every queue is genuinely empty.
fn find_task(
    local: &Deque<PoolJob>,
    injector: &Injector<PoolJob>,
    stealers: &[Stealer<PoolJob>],
) -> Option<PoolJob> {
    local.pop().or_else(|| {
        std::iter::repeat_with(|| {
            injector
                .steal_batch_and_pop(local)
                .or_else(|| stealers.iter().map(Stealer::steal).collect())
        })
        .find(|s| !s.is_retry())
        .and_then(|s| s.success())
    })
}

static DRIVEN: OnceLock<DrivenPools> = OnceLock::new();

/// The committed `drive_compute` worker count, set by [`init_driven_runtime`] so
/// [`compute_width`] can size parallel fan-out to the real pool width. 0 until set.
static COMPUTE_WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The compute-pool width for sizing parallel fan-out: the committed
/// [`drive_compute`] worker count (from [`init_driven_runtime`]), or a machine
/// estimate (`available_parallelism`) when no driven runtime was installed (the C
/// ABI, a direct embed, a test). Never 0.
pub fn compute_width() -> usize {
    match COMPUTE_WIDTH.load(std::sync::atomic::Ordering::Relaxed) {
        0 => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        n => n,
    }
}

thread_local! {
    /// Set while a [`drive_collection`]/[`drive_compute`] job runs, so the dispatch
    /// helpers can assert the leaf-invariant (a pool job must never enqueue-and-
    /// await further pool work). Drives a `debug_assert` only.
    static IN_POOL_JOB: Cell<bool> = const { Cell::new(false) };
}

/// The installed runtime. Panics if [`init_driven_runtime`] has not run: there
/// is no lazy fallback — the harness installs the driven runtime before any
/// kernel op (a missing install is a setup error, not a runtime condition).
fn runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME
        .get()
        .expect("the driven runtime must be installed via init_driven_runtime before any kernel op")
}

/// Install a `current_thread` runtime: the harness parks threads in
/// [`drive_io`]/[`drive_collection`]/[`drive_compute`] to provide every thread the
/// kernel uses; shrike-core spawns none of its own. The `dispatch_collection` /
/// `dispatch_compute` helpers enqueue onto the driven queues.
///
/// `compute_workers` is the number of [`drive_compute`] threads the harness will
/// park — the committed compute-pool width. The pool is built here with exactly
/// that many local deques (one per worker, claimed at startup), and the width is
/// recorded for [`compute_width`] to size parallel fan-out. The harness must park
/// the same count of `drive_compute` threads.
///
/// The supplied runtime MUST be a `current_thread` runtime — one async thread
/// the harness drives via [`drive_io`]. The driven queues are created here, so a
/// job submitted before [`drive_collection`] / [`drive_compute`] is parked simply
/// waits in the queue.
///
/// # Errors
///
/// Returns `Err` carrying the supplied runtime back if one is already installed
/// (the seam is set-once).
pub fn init_driven_runtime(
    runtime: tokio::runtime::Runtime,
    compute_workers: usize,
) -> Result<(), tokio::runtime::Runtime> {
    RUNTIME.set(runtime)?;
    // We won the runtime install, so the pools below are uncontended. Install the
    // Prometheus recorder first so the instrument handles below resolve against a
    // live recorder (the metrics facade hands out no-ops before one is installed).
    COMPUTE_WIDTH.store(compute_workers, Ordering::Relaxed);
    install_prometheus();
    let (collection_tx, collection_rx) = mpsc::unbounded_channel::<PoolJob>();
    let _ = DRIVEN.set(DrivenPools {
        collection_tx: Mutex::new(Some(collection_tx)),
        collection_rx: Mutex::new(Some(collection_rx)),
        compute: ComputePool::new(compute_workers),
        shutdown: Notify::new(),
        collection_metrics: PoolInstruments::for_pool("collection"),
        compute_metrics: PoolInstruments::for_pool("compute"),
    });
    Ok(())
}

/// The kernel runtime's handle. Only the handle escapes — never the `Runtime` —
/// so nothing can drop or block the runtime from inside it.
///
/// # Panics
///
/// Panics if the driven runtime is not installed (see [`init_driven_runtime`]).
pub(crate) fn handle() -> &'static tokio::runtime::Handle {
    runtime().handle()
}

/// Drive a future to completion on the kernel runtime from a non-async
/// context (the [`drive_io`] driver thread; a synchronous embedded host).
/// Async callers just `.await`.
///
/// # Panics
///
/// Panics if the driven runtime is not installed (see [`init_driven_runtime`]).
/// Panics if called from inside the runtime — a runtime worker thread is a
/// runtime context, and `tokio`'s nested-`block_on` guard refuses there. This
/// is the same guard the module's collection-dispatch discipline relies on
/// (pinned by the `collection_dispatch_pin` test).
pub fn block_on<F: Future>(future: F) -> F::Output {
    runtime().block_on(future)
}

/// **Own + drive the runtime until `until` resolves.** The harness's one
/// IO/timer-driver thread. The first call to the runtime's `block_on` takes
/// ownership of tokio's IO + timer drivers and the async executor; every spawned
/// task (the collection actor's dispatch loop, the debounced saver's timer, the
/// tag refresher) is polled here, and timers fire. Other
/// [`drive_collection`]/[`drive_compute`] threads do not own the drivers — they hook
/// into this one.
///
/// `until` is the harness's shutdown signal (resolved once shutdown begins and
/// in-flight work has drained); when it resolves, `drive_io` returns and the
/// harness joins it before interpreter finalization.
///
/// # Errors
///
/// Returns an error if the driven runtime is not installed.
pub fn drive_io<F: Future<Output = ()> + Send + 'static>(until: F) -> NativeResult<()> {
    let _ = DRIVEN.get().ok_or_else(driven_missing)?;
    block_on(until);
    Ok(())
}

/// **Drive the runtime until [`shutdown_driven_pools`] is called.** The same as
/// [`drive_io`] but parked on the pools' built-in shutdown signal, so a host
/// with no shutdown future of its own (the binding) gets one `drive_io` thread
/// whose `until` and the pool-queue close are tripped by one call. No lost
/// wakeup against a racing shutdown: `shutdown_driven_pools` uses `notify_one`,
/// which stores a permit when no waiter is parked yet, so a shutdown that fires
/// before this thread reaches the await is consumed by it immediately.
///
/// # Errors
///
/// Returns an error if the driven runtime is not installed.
pub fn drive_io_until_shutdown() -> NativeResult<()> {
    let pools = DRIVEN.get().ok_or_else(driven_missing)?;
    gauge!("shrike_runtime_io_alive").set(1.0);
    block_on(pools.shutdown.notified());
    gauge!("shrike_runtime_io_alive").set(0.0);
    Ok(())
}

/// **The serialized collection execution thread.** A plain OS thread blocking on
/// the collection work queue and running each job to completion — every
/// anki-collection op (read and write, including the journal fsync) runs here.
/// One thread is a *consequence* of anki's single-writer collection (reads and
/// writes serialize; anki forbids concurrent access), not a tuning choice.
/// Because this thread is never inside the kernel runtime context, anki's own
/// `block_on` is legal here — the structural form of the
/// collection-never-on-a-runtime-worker invariant.
///
/// Parks until the queue is closed (every sender dropped — i.e. harness
/// shutdown), then returns so the harness can join it.
///
/// # Errors
///
/// Returns an error if the driven runtime is not installed, or if the collection
/// queue was already claimed by a prior `drive_collection` (exactly one thread
/// drives it).
///
/// # Panics
///
/// Panics if the collection-receiver mutex is poisoned (a prior holder panicked).
pub fn drive_collection() -> NativeResult<()> {
    let pools = DRIVEN.get().ok_or_else(driven_missing)?;
    let mut rx = pools
        .collection_rx
        .lock()
        .expect("driven collection receiver poisoned")
        .take()
        .ok_or_else(|| {
            NativeError::internal("drive_collection was already claimed by another thread")
        })?;
    let _worker = WorkerGuard::enter(&pools.collection_metrics.workers);
    while let Some(job) = rx.blocking_recv() {
        pools.run_job(QueueKind::Collection, job);
    }
    Ok(())
}

/// **Driven mode: a CPU-bound engine-compute (and blocking-fs leaf) worker.**
/// The harness spawns N of these; each runs one of the [`ComputePool`]'s
/// work-stealing workers. This is the only place real parallelism lives
/// (independent batches), so the engine search/batch overlap property becomes
/// "N ≥ 2", sized by the harness to its cores. Dispatch target for the
/// `Blocking<E>` adapter, the tag-centroid recompute, the index file save, the
/// derived FTS5 rebuild, and the store-media decode.
///
/// Claims a local deque, then loops: run every job [`find_task`] can pop or steal
/// (lock-free), and park on the pool's condvar when the queues run dry. Returns
/// once shutdown is signalled and the queues have drained.
///
/// # Errors
///
/// Returns an error if the driven runtime is not installed.
///
/// # Panics
///
/// Panics if the pool's `locals` or `park` mutex is poisoned (a prior holder
/// panicked).
pub fn drive_compute() -> NativeResult<()> {
    let pools = DRIVEN.get().ok_or_else(driven_missing)?;
    let pool = &pools.compute;
    let _worker = WorkerGuard::enter(&pools.compute_metrics.workers);
    // Claim a local deque. init_driven_runtime provisions exactly N (the committed
    // worker count) before the harness spawns the workers; a worker beyond N — not
    // expected — falls back to a fresh unregistered deque (still correct, just not
    // steal-balanced).
    let local = pool
        .locals
        .lock()
        .expect("compute locals poisoned")
        .pop()
        .unwrap_or_else(Deque::new_fifo);
    loop {
        // Hot path: run every job we can find, lock-free.
        if let Some(job) = find_task(&local, &pool.injector, &pool.stealers) {
            pools.run_job(QueueKind::Compute, job);
            continue;
        }
        // Dry. Re-check under `park` (closing the lost-wakeup window): take a job
        // that raced in, return on shutdown, else park until a submission wakes us.
        let guard = pool.park.lock().expect("compute park poisoned");
        if pool.has_work() {
            continue; // a job landed after find_task — drop the guard and take it
        }
        if pool.shutdown.load(Ordering::Acquire) {
            return Ok(()); // drained and closing
        }
        drop(pool.unpark.wait(guard).expect("compute park poisoned"));
    }
}

/// **Signal every committed thread to return so the harness can join them.**
/// Closes the collection queue (drops its sender, so [`drive_collection`]'s `recv`
/// yields `None`), trips the compute pool's shutdown flag and wakes its parked
/// workers, and trips the [`drive_io_until_shutdown`] signal. Call it once, AFTER
/// kernel work has quiesced (the collection actor drained), so no in-flight
/// enqueue is racing — then the threads return and the joins are immediate.
/// Idempotent: a second call finds the sender already taken and only re-signals.
/// A no-op if the driven runtime was never installed.
///
/// # Panics
///
/// Panics if a sender-slot mutex is poisoned (a prior holder panicked).
pub fn shutdown_driven_pools() {
    let Some(pools) = DRIVEN.get() else {
        return; // no driven install — nothing to signal
    };
    // Close the collection queue by dropping its sender: drive_collection returns
    // once its receiver sees the sender gone.
    drop(
        pools
            .collection_tx
            .lock()
            .expect("driven collection sender poisoned")
            .take(),
    );
    // Signal the compute workers: each drains its deques to empty, then returns.
    pools.compute.begin_shutdown();
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
/// # Errors
///
/// Propagates the work's own `NativeResult`; an internal error if the driven
/// runtime is not installed, or if the compute worker vanished without producing
/// a result (the pool shut down mid-flight).
pub fn submit_blocking<T: Send + 'static>(
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> NativeResult<T> {
    debug_assert!(
        !IN_POOL_JOB.with(Cell::get),
        "leaf-invariant: a pool job must not submit-and-block on further pool work (submit_blocking)"
    );
    let (tx, rx) = std::sync::mpsc::channel();
    let job = PoolJob::new(Box::new(move || {
        let _ = tx.send(run_in_pool_job(work));
    }));
    let pools = DRIVEN.get().ok_or_else(driven_missing)?;
    pools.compute_metrics.queue_depth.increment(1.0);
    if !pools.compute.push(job) {
        // The pool refused the job (shutting down): it never reaches run_job, so
        // undo the queue-depth bump here.
        pools.compute_metrics.queue_depth.decrement(1.0);
        return Err(NativeError::internal("the compute pool is gone"));
    }
    rx.recv()
        .map_err(|_| NativeError::internal("the compute worker dropped a job"))?
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

// ── dispatch: route blocking work onto the committed pools ───────────────────

/// Run a unit of **anki-collection** blocking work, returning an eagerly-
/// scheduled future of its result. The collection actor routes every collection
/// op through here so anki's sync `block_on` (when client sync lands) can never
/// land on a runtime worker: the work enqueues onto the [`drive_collection`]
/// thread (a non-runtime context — anki `block_on` legal by construction).
///
/// **Eager by contract** (like the engine `Blocking` adapter): the work is
/// scheduled inside this call, before the returned future is first polled.
///
/// # Errors
///
/// The future yields the work's own `NativeResult`, or an internal error if the
/// executing thread/pool vanished without producing one.
pub(crate) fn dispatch_collection<T: Send + 'static>(
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> impl Future<Output = NativeResult<T>> + Send + 'static {
    debug_assert!(
        !IN_POOL_JOB.with(Cell::get),
        "leaf-invariant: a pool job must not enqueue-and-await further pool work (dispatch_collection)"
    );
    enqueue(QueueKind::Collection, work)
}

/// Run a unit of **CPU-bound compute / blocking-fs** work, returning an eagerly-
/// scheduled future of its result. The engine `Blocking` adapter, the
/// tag-centroid recompute, the index file save, the derived FTS5 rebuild, and
/// the store-media decode route through here, enqueuing onto the
/// [`drive_compute`] pool (N threads).
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

/// **Schedule a fire-and-forget compute job on the [`drive_compute`] pool,
/// eagerly** — the seam the engine `Blocking` adapter's injected dispatcher
/// calls. The job is the type-erased closure that owns its own result channel
/// (engine-api wraps the engine compute so the awaiting future learns the
/// outcome), so this only has to run it on the pool (N threads, the N ≥ 2 engine
/// overlap).
///
/// **Eager by contract**: the job is queued/scheduled inside this call, before
/// control returns to the adapter — what keeps the engine future in flight
/// before its first poll (the search/add overlap property).
///
/// The job runs through [`run_in_pool_job`] for the panic containment and
/// leaf-invariant tripwire every kernel pool job gets: a panicking engine job
/// loses only itself (its result channel drops, the awaiting future gets a clean
/// error), the pool thread survives. If the pool is gone (shutdown, or no driven
/// runtime installed) the job is dropped; its result channel closes and the
/// awaiting future sees the error.
pub fn submit_compute(job: Box<dyn FnOnce() + Send + 'static>) {
    debug_assert!(
        !IN_POOL_JOB.with(Cell::get),
        "leaf-invariant: a pool job must not submit further pool work (submit_compute)"
    );
    let contained = PoolJob::new(Box::new(move || {
        let _ = run_in_pool_job(move || {
            job();
            Ok::<(), NativeError>(())
        });
    }));
    if let Some(pools) = DRIVEN.get() {
        pools.compute_metrics.queue_depth.increment(1.0);
        if !pools.compute.push(contained) {
            pools.compute_metrics.queue_depth.decrement(1.0);
        }
    }
}

/// Which driven queue an enqueue targets.
#[derive(Clone, Copy)]
enum QueueKind {
    Collection,
    Compute,
}

/// The shared body of `dispatch_collection`/`dispatch_compute`: enqueue onto the
/// targeted pool and return an eager future of the result. Boxed so the two
/// callers unify to one return type.
fn enqueue<T: Send + 'static>(
    kind: QueueKind,
    work: impl FnOnce() -> NativeResult<T> + Send + 'static,
) -> futures::future::BoxFuture<'static, NativeResult<T>> {
    let (tx, rx) = oneshot::channel();
    let job = PoolJob::new(Box::new(move || {
        let _ = tx.send(run_in_pool_job(work));
    }));
    // If the queue is gone (shutdown took the sender, or no driven runtime
    // installed), the receiver closes empty → the internal error below.
    if let Some(pools) = DRIVEN.get() {
        pools.metrics(kind).queue_depth.increment(1.0);
        let delivered = match kind {
            QueueKind::Collection => match pools.collection_sender() {
                Some(sender) => sender.send(job).is_ok(),
                None => false,
            },
            QueueKind::Compute => pools.compute.push(job),
        };
        if !delivered {
            // The job never reaches run_job (queue closed / pool shutting down) —
            // undo the queue-depth bump that run_job would otherwise decrement.
            pools.metrics(kind).queue_depth.decrement(1.0);
        }
    }
    Box::pin(async move {
        rx.await
            .map_err(|_| NativeError::internal("the driven pool dropped a job"))?
    })
}

/// Whether the driven runtime has been installed. The binding checks this right
/// after [`init_driven_runtime`] so a lost install (the set-once seam was already
/// taken) is a loud error, not threads that quietly fail to drive.
pub fn is_driven() -> bool {
    DRIVEN.get().is_some()
}

/// Run a pool job body with the leaf-invariant tripwire armed AND its panic
/// contained — the one place every pool job converges, so resilience is uniform
/// and DRY.
///
/// - **Tripwire**: sets the `IN_POOL_JOB` thread-local for the duration, so a
///   `dispatch_*` called from within asserts (debug builds). An RAII guard
///   clears it even on unwind, so a panicking job can't leave it armed and
///   spuriously trip a later legitimate dispatch.
/// - **Panic containment**: `work` is run under `catch_unwind` and a caught
///   panic becomes `Err(Internal)` rather than unwinding out. The pool runs jobs
///   on a long-lived OS thread with no per-job isolation, so an uncaught panic
///   would KILL that thread — and the single `drive_collection` thread dying would
///   wedge every future collection op (its receiver was taken with no
///   replacement). Catching here keeps "a panic loses only that one job, the
///   pool survives, the caller gets a clean Err".
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
    NativeError::internal("the driven runtime is not installed (call init_driven_runtime first)")
}

/// Test-support: install + drive the runtime so a test can run kernel futures.
///
/// **Not part of the stable API** — these are test fixtures for driving the
/// kernel runtime in tests (the kernel's own integration binaries, and the cabi
/// in-crate smoke), `pub` only so those external test crates can reach them.
/// They compile only where the kernel does: the kernel is the `anki-core`
/// capability (an optional, `anki-core`-gated dependency of the bindings), so a
/// compute-only / minimal-core build pulls in neither the kernel nor this module.
///
/// The runtime is harness-driven with no lazy fallback, so a test process must
/// commit the driver threads itself — the maintainer's "1 + N injected fixture".
/// This module donates them, once per process (the seam is set-once and the
/// committed threads outlive any kernel, exactly as in production):
///
/// - [`run`] installs a `current_thread` runtime and parks **1 `drive_io` + N
///   `drive_compute`** threads (no collection thread), then runs a future via the
///   submit + completion-channel shape — the threaded-host submission path.
/// - [`run_with_collection`] adds the **`drive_collection`** thread for a test
///   that opens a collection / exercises anki ops (the serialized-collection
///   actor routes through `dispatch_collection`). Most kernel tests want this; a
///   test that touches no collection uses [`run`].
///
/// The startup barrier is honored (`drive_io` first, [`spawn_op`]-probe, then the
/// rest), so the IO thread owns tokio's drivers before any other thread parks.
/// The threads are never joined: a test process keeps them parked for its life.
pub mod testing {
    use std::sync::OnceLock;
    use std::thread;

    use super::*;

    /// Number of `drive_compute` threads the fixture commits — two so independent
    /// engine batches overlap (the "N ≥ 2" property), enough for any test.
    const COMPUTE_THREADS: usize = 2;

    /// One-time install + IO/compute thread spawn (the barrier-honored core,
    /// shared by [`run`] and [`run_with_collection`]).
    static STARTED: OnceLock<()> = OnceLock::new();
    /// One-time `drive_collection` spawn, lazily added the first time a test
    /// opens a collection.
    static COLLECTION_STARTED: OnceLock<()> = OnceLock::new();

    /// Park a committed driver thread in `entry`, naming it per the kernel's
    /// thread-name scheme.
    fn spawn(name: &str, entry: impl FnOnce() + Send + 'static) {
        thread::Builder::new()
            .name(name.to_string())
            .spawn(entry)
            .expect("a fixture driver thread spawns");
    }

    /// Install the driven runtime and park `drive_io` + N `drive_compute`, once
    /// per process. Honors the startup barrier: spawn `drive_io`, probe until it
    /// is driving, then spawn the compute workers.
    fn ensure_started() {
        STARTED.get_or_init(|| {
            // Mirror production: commit the pool width here so the pool provisions
            // exactly COMPUTE_THREADS deques and search_fused chunks the lexical
            // reads across exactly the workers parked below (not at
            // available_parallelism(), a different count than prod).
            init_driven_runtime(
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("the fixture current_thread runtime builds"),
                COMPUTE_THREADS,
            )
            .unwrap_or_else(|_| panic!("the test process owns the driven runtime seam"));

            spawn("shrike-io", || {
                let _ = drive_io_until_shutdown();
            });
            // The barrier: block until the IO thread owns the drivers before the
            // rest park (a spawned op completes only once a thread is driving).
            let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
            drop(spawn_op(async move {
                let _ = tx.send(());
                Ok(())
            }));
            // Block on the REAL "IO thread is driving" event, no wall-clock
            // budget — see submit_and_wait. A never-starting IO thread hangs and
            // is caught by Bazel's per-test timeout, not an in-test guess.
            rx.recv()
                .expect("the IO thread drives the runtime (the startup barrier)");

            for i in 0..COMPUTE_THREADS {
                spawn(&format!("shrike-work-{i}"), || {
                    let _ = drive_compute();
                });
            }
        });
    }

    /// Add the `drive_collection` thread, once per process. The collection queue
    /// already exists (created by `init_driven_runtime`), so parking the thread
    /// late is fine — jobs enqueued before it parks simply wait.
    fn ensure_collection() {
        ensure_started();
        COLLECTION_STARTED.get_or_init(|| {
            spawn("shrike-collection", || {
                let _ = drive_collection();
            });
        });
    }

    /// Submit `fut` onto the driven runtime and block the calling test thread on
    /// its completion (the threaded-host submission shape — the request thread
    /// must not `block_on` the runtime the IO thread owns).
    fn submit_and_wait<T: Send + 'static>(fut: impl Future<Output = T> + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::sync_channel::<T>(1);
        drop(spawn_op(async move {
            let _ = tx.send(fut.await);
            Ok(())
        }));
        // Block on the REAL completion event (the future finished and sent), with
        // NO wall-clock budget: pass/fail can't hinge on a timeout-vs-load race, so
        // an oversubscribed host just waits rather than flaking. `recv` errors only
        // if the sender dropped without sending — the future panicked — which
        // surfaces here as the test failure it is. A genuine deadlock hangs and is
        // caught by Bazel's per-test timeout (the outer bound), not an in-test
        // wall-clock guess.
        rx.recv()
            .expect("the driven runtime ran the test future to completion")
    }

    /// Run a kernel future on the driven runtime with **1 io + N compute**
    /// threads (no collection thread). For a test that does not open a collection.
    pub fn run<T: Send + 'static>(fut: impl Future<Output = T> + Send + 'static) -> T {
        ensure_started();
        submit_and_wait(fut)
    }

    /// Run a kernel future on the driven runtime with **1 io + 1 collection + N
    /// compute** threads — for a test that opens a collection / exercises anki
    /// ops (the serialized-collection actor routes through the collection thread).
    pub fn run_with_collection<T: Send + 'static>(
        fut: impl Future<Output = T> + Send + 'static,
    ) -> T {
        ensure_collection();
        submit_and_wait(fut)
    }
}

#[cfg(test)]
mod collection_dispatch_pin {
    //! The acceptance gate: pin the collection-dispatch path structurally, not by
    //! luck.
    //!
    //! anki keeps its own runtime for client sync, so two runtimes will live in
    //! the process; the invariant the kernel guarantees is **"a collection op (and
    //! anki's sync `block_on`, when client sync lands) never executes on a runtime
    //! worker thread"**.
    //!
    //! - **Half 1** demonstrates the hazard is real: `Handle::block_on` PANICS
    //!   on a runtime worker thread (the way anki's sync `block_on` would land if
    //!   dispatched *directly* from a `SerializedCollection` job inline on a
    //!   worker).
    //! - **Half 2** demonstrates the mandated dispatch site is safe: the SAME
    //!   call on a plain OS thread — the structural form of the
    //!   [`drive_collection`] dispatch target, never a runtime context — succeeds
    //!   and returns the sentinel.
    //!
    //! The only variable between the halves is *which thread* runs `block_on`, so
    //! a regression that lets a collection op run on a runtime worker flips Half 2
    //! from pass to panic. The full driven-mode form of the invariant (collection
    //! work runs on the committed `drive_collection` thread) is pinned end-to-end
    //! by the `driven_mode.rs` integration binary. Self-contained here: a
    //! locally-built `current_thread` runtime so the process-global seam is
    //! untouched.

    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Stand-in for "a synchronous call that bottoms out in `block_on`" —
    /// exactly the shape of anki's sync service paths, minus the anki
    /// dependency. Returns a sentinel so the success half can assert the call
    /// actually ran to completion (not merely that it didn't panic).
    fn collection_call_that_blocks_on(handle: &tokio::runtime::Handle) -> u64 {
        handle.block_on(async { 0x5031_u64 })
    }

    #[test]
    fn block_on_panics_on_a_runtime_worker_but_rides_a_plain_thread() {
        // A dedicated local current_thread runtime — never the process-global
        // seam, which other tests in this binary share. Its block_on caller IS a
        // runtime context, which is all Half 1 needs.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        let handle = rt.handle().clone();

        // ── Half 1: the runtime's own block_on driver thread is a runtime
        // context, so a nested `block_on` MUST panic. Calling it inside the
        // runtime's block_on runs it on that context. Catch the unwind so the
        // thread survives for half 2.
        let inner = handle.clone();
        let worker_result = rt.block_on(async move {
            catch_unwind(AssertUnwindSafe(|| collection_call_that_blocks_on(&inner)))
        });
        assert!(
            worker_result.is_err(),
            "Handle::block_on must panic on a runtime context — if it stopped \
             panicking, the dispatch invariant can no longer be pinned this way"
        );

        // ── Half 2: the SAME call on a plain OS thread — the structural form of
        // the `drive_collection` dispatch target, never a runtime context — succeeds
        // and returns the sentinel.
        let pooled = handle.clone();
        let value = std::thread::spawn(move || collection_call_that_blocks_on(&pooled))
            .join()
            .expect("the plain thread must not fail");
        assert_eq!(
            value, 0x5031,
            "block_on on a plain (non-runtime) thread must run the sync call to \
             completion"
        );
    }
}

#[cfg(test)]
mod leaf_invariant {
    //! The deadlock leaf-invariant tripwire: a pool job must never
    //! enqueue-and-await further pool work. These run on the driven runtime (the
    //! shared test fixture), where `dispatch_compute` enqueues onto the committed
    //! `drive_compute` pool.

    use super::*;

    /// A WELL-FORMED op — `dispatch_compute` called from the async side, its job
    /// a pure leaf — completes and never trips the assert.
    #[test]
    fn well_formed_dispatch_passes() {
        let out: NativeResult<u64> =
            testing::run(async { dispatch_compute(|| Ok(0x5031_u64)).await });
        assert_eq!(out.unwrap(), 0x5031);
    }

    /// A leaf job that RE-ENTERS dispatch (the deadlock shape) trips the
    /// debug-build tripwire. The nested dispatch happens INSIDE a running pool
    /// job (the tripwire's thread-local is set), so the `debug_assert!` in
    /// `dispatch_compute` panics. The outer job runs on a `drive_compute` thread
    /// under `run_in_pool_job`, which catches that panic and turns it into the
    /// outer job's `Err` — never an unwind to the caller. The debug-build outcome
    /// is therefore the outer op resolving `Err`; release builds compile the
    /// assert out and the op resolves `Ok`.
    #[test]
    fn nested_pool_dispatch_trips_the_tripwire_in_debug() {
        let out: NativeResult<()> = testing::run(async {
            // The OUTER pool job runs `run_in_pool_job` (sets the flag), and
            // from inside it synchronously builds a nested dispatch — exactly the
            // forbidden enqueue-from-a-pool-job shape.
            dispatch_compute(move || {
                // Inside a pool job now (flag set). Building a nested dispatch
                // future trips the debug_assert.
                let _nested = dispatch_compute(|| Ok::<(), NativeError>(()));
                Ok::<(), NativeError>(())
            })
            .await
        });
        if cfg!(debug_assertions) {
            assert!(
                out.is_err(),
                "a pool job re-entering dispatch must trip the leaf-invariant \
                 debug_assert — the outer job panics, caught by run_in_pool_job \
                 and surfaced as the job's Err"
            );
        } else {
            assert!(out.is_ok(), "release: the assert is compiled out");
        }
    }

    /// `submit_compute` schedules the fire-and-forget job on the `drive_compute`
    /// pool — it runs to completion off the calling thread. The job carries its
    /// own result channel (the engine `Blocking` adapter's shape), so we observe
    /// completion + the thread it ran on through one.
    #[test]
    fn submit_compute_runs_off_thread() {
        // Ensure the driven fixture is up (parks the drive_compute threads).
        testing::run(async {});
        let caller = std::thread::current().id();
        let (tx, rx) = std::sync::mpsc::channel();
        submit_compute(Box::new(move || {
            let _ = tx.send(std::thread::current().id());
        }));
        // Block on the REAL "the job ran" event (it sends its thread id), no
        // wall-clock budget: a starved compute pool just waits instead of flaking;
        // a never-scheduled job hangs and Bazel's per-test timeout catches it.
        let ran_on = rx
            .recv()
            .expect("submit_compute scheduled the job on the compute pool");
        assert_ne!(
            ran_on, caller,
            "submit_compute ran the job off the calling thread (on the compute pool)"
        );
    }
}

#[cfg(test)]
mod compute_pool {
    //! The work-stealing compute pool. The shared fixture parks COMPUTE_THREADS
    //! workers; these tests flood them to pin the pool's load-bearing property.

    use super::*;

    /// Burst far more jobs than workers onto the pool and prove every one runs
    /// exactly once: each job returns its index and the summed result is exact.
    /// A lost job (a dropped queue entry) drops a term and fails the sum; a lost
    /// wakeup (a parked worker never woken for queued work) hangs and is caught by
    /// the test timeout. Eager dispatch queues all N before the first await, so
    /// they flood at once — exercising `steal_batch_and_pop` and the park/wake
    /// cycle under contention.
    #[test]
    fn burst_runs_every_job_exactly_once() {
        let n: u64 = 2_000;
        let sum: u64 = testing::run(async move {
            let futs: Vec<_> = (0..n)
                .map(|i| dispatch_compute(move || Ok::<u64, NativeError>(i)))
                .collect();
            let mut total = 0u64;
            for f in futs {
                total += f.await.expect("every burst job resolves");
            }
            total
        });
        assert_eq!(sum, (0..n).sum::<u64>(), "every job ran exactly once");
    }

    /// The exporter renders the kernel instruments with the recommended-naming
    /// suffixes the macros never hand-format (`_total` on counters, `_seconds` on
    /// the second-unit histograms), carrying the embedding `operation`/`space`
    /// labels supplied by [`embed_scope`]. Pins the suffix derivation + label
    /// contract end to end (the install lives in the shared `testing::run` fixture).
    #[test]
    fn render_prometheus_exposes_suffixed_instruments() {
        testing::run(async {
            // A compute job runs through run_job → the pool instruments record.
            dispatch_compute(|| Ok::<(), NativeError>(()))
                .await
                .expect("the probe job resolves");
            // An embed batch tagged query/primary records the embedding instruments.
            embed_scope("primary".to_owned(), "query", async {
                record_embedding("text", 3, Duration::from_millis(5), true);
            })
            .await;
            record_saver_run(true, Duration::from_millis(2), 0);
        });
        let rendered = render_prometheus();
        // Counters carry `_total`; the duration histograms carry `_seconds`.
        assert!(
            rendered.contains("shrike_embedding_items_total"),
            "embedding items counter missing _total:\n{rendered}"
        );
        assert!(
            rendered.contains("shrike_embedding_duration_seconds"),
            "embedding duration histogram missing _seconds:\n{rendered}"
        );
        assert!(
            rendered.contains("shrike_runtime_pool_jobs_total"),
            "pool jobs counter missing _total:\n{rendered}"
        );
        // The embed labels supplied by the op boundary, and independent ok/error.
        assert!(rendered.contains("operation=\"query\""), "{rendered}");
        assert!(rendered.contains("space=\"primary\""), "{rendered}");
        assert!(rendered.contains("result=\"ok\""), "{rendered}");
        assert!(
            rendered
                .lines()
                .any(|line| line.starts_with("shrike_embedding_items_total")
                    && line.contains("result=\"ok\"")),
            "embedding item counter must carry result label:\n{rendered}"
        );
        assert!(
            rendered.lines().any(
                |line| line.starts_with("shrike_index_saver_duration_seconds")
                    && line.contains("result=\"ok\"")
            ),
            "saver duration histogram must carry result label:\n{rendered}"
        );
    }
}
