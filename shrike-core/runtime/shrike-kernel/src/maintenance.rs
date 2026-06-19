//! One structured-maintenance primitive: a coalescing single-flight background
//! job with a uniform lifecycle — request / coalesce / settle / shutdown —
//! behind one type, replacing the hand-rolled `DebouncedSaver` and
//! `TagRefresher` coordinators that each invented their own.
//!
//! Under the single-user assumption (write-vs-write bursts are rare) the
//! coalescing logic is minimal, but the primitive keeps the two pacing knobs
//! the residual jobs genuinely need — they are NOT both pure burst-coalescers:
//!
//! - **`delay` — a RE-ARMING debounce.** Each [`Maintenance::request`] during
//!   the window restarts it, so the run fires `delay` after the *last* request.
//!   This batches a stream of SPACED writes into one run — the index saver's
//!   reason for being: one large file write per quiet period, not one per note.
//!   A spaced interactive add-N-notes session is the case this serves; it is not
//!   reducible to burst-coalescing.
//! - **`threshold` — a burst cap.** Run immediately once this many requests
//!   accumulate without a run, so a flood doesn't sit behind the debounce.
//! - **`window` — coalesced re-run pacing.** A request that lands while a run is
//!   executing marks the job dirty; it re-runs once, `window` later.
//!
//! A job with `delay = 0`, `threshold = 0` is the pure coalesce-loop (run
//! immediately, coalesce concurrent requests into one re-run) the tag refresh
//! uses. The primitive also exposes [`Maintenance::pending`] (requests since the
//! last run was handed off — the saver's status counter) and
//! [`Maintenance::cancel`] (synchronously disarm, for a host's own flush path).
//!
//! All scheduling rides the kernel runtime ([`crate::runtime`]); the `run`
//! closure does its own pool dispatch for blocking/compute work (the recompute
//! and the file write both ride `drive_compute`, never a runtime worker).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::BoxFuture;

/// The maintenance work: a fresh future per run (so a re-run re-reads state).
type RunFn = Box<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>;

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum Phase {
    /// No run armed or executing.
    #[default]
    Idle,
    /// A debounce timer is sleeping before the run (re-armed on each request).
    Delaying,
    /// The run loop is executing.
    Running,
}

#[derive(Default)]
struct State {
    phase: Phase,
    /// A request landed while `Running` → re-run once, `window` later.
    dirty: bool,
    /// The armed timer / run task — aborted on re-arm and on shutdown, so a
    /// sleeping follow-up never outlives the kernel's collection actor.
    task: Option<tokio::task::AbortHandle>,
}

/// A coalescing single-flight maintenance job. Held behind an `Arc` so the
/// spawned run tasks clone it.
pub struct Maintenance {
    run: RunFn,
    delay: Duration,
    window: Duration,
    threshold: u64,
    /// Requests since the last run was handed off (the saver's status counter).
    pending: AtomicU64,
    state: Mutex<State>,
}

impl Maintenance {
    /// Build a job over `run` with the re-arming `delay`, the coalesced re-run
    /// `window`, and the burst `threshold` (`0` = no burst cap). `delay == 0`
    /// makes it the pure immediate coalesce-loop.
    pub fn new(run: RunFn, delay: Duration, window: Duration, threshold: u64) -> Arc<Self> {
        Arc::new(Self {
            run,
            delay,
            window,
            threshold,
            pending: AtomicU64::new(0),
            state: Mutex::new(State::default()),
        })
    }

    /// Note a unit of work. Never blocks, never errors. Coalesces: a request
    /// while a run is in flight marks it dirty (one re-run); a request while
    /// arming re-arms the debounce; the burst cap fires an immediate run.
    ///
    /// # Panics
    ///
    /// Panics if the internal state mutex is poisoned (a prior holder panicked).
    pub fn request(self: &Arc<Self>) {
        let n = self.pending.fetch_add(1, Ordering::SeqCst) + 1;
        let mut st = self.state.lock().expect("maintenance poisoned");
        if st.phase == Phase::Running {
            // The run loop will pick this up via `dirty` and re-run once.
            st.dirty = true;
            return;
        }
        // Idle or Delaying: (re-)arm. Cancel any sleeping debounce timer so the
        // run fires `delay` after THIS request (the re-arming batch window).
        if let Some(task) = st.task.take() {
            task.abort();
        }
        st.phase = Phase::Delaying;
        let burst = self.threshold > 0 && n >= self.threshold;
        if burst {
            // Hand the run off NOW: reset the counter synchronously (the caller
            // observes `pending() == 0` immediately, like the old saver's
            // burst-cap reset) and skip the debounce.
            self.pending.store(0, Ordering::SeqCst);
        }
        let this = Arc::clone(self);
        let task = crate::runtime::handle().spawn(async move {
            if !burst {
                tokio::time::sleep(this.delay).await;
            }
            this.drive().await;
        });
        st.task = Some(task.abort_handle());
    }

    /// The run loop: run, then re-run once `window` later if a request coalesced
    /// in mid-run. A re-arm (or [`cancel`](Self::cancel)) between this timer's
    /// wake and the lock claims `Delaying`, so a superseded timer no-ops.
    async fn drive(self: &Arc<Self>) {
        {
            let mut st = self.state.lock().expect("maintenance poisoned");
            if st.phase != Phase::Delaying {
                return; // re-armed or cancelled out from under this timer
            }
            st.phase = Phase::Running;
            st.dirty = false;
        }
        loop {
            // The run is now handed off — the counter measures work that landed
            // AFTER it started (so `pending()` never under-reports a re-run).
            self.pending.store(0, Ordering::SeqCst);
            (self.run)().await;
            {
                let mut st = self.state.lock().expect("maintenance poisoned");
                if !st.dirty {
                    st.phase = Phase::Idle;
                    st.task = None;
                    return;
                }
                st.dirty = false;
            }
            tokio::time::sleep(self.window).await;
        }
    }

    /// Requests since the last run was handed off (status surface).
    pub fn pending(&self) -> u64 {
        self.pending.load(Ordering::SeqCst)
    }

    /// Synchronously disarm: abort any armed/running task, drop to `Idle`, and
    /// zero the counter — so a host's own flush path can take over the work
    /// (the saver's synchronous shutdown write) with the counter meaning "not
    /// yet handed to a flush" regardless of where that write runs.
    ///
    /// # Panics
    ///
    /// Panics if the internal state mutex is poisoned (a prior holder panicked).
    pub fn cancel(&self) {
        let mut st = self.state.lock().expect("maintenance poisoned");
        st.dirty = false;
        st.phase = Phase::Idle;
        if let Some(task) = st.task.take() {
            // Possibly our own handle (a run task aborting itself) — an abort
            // after the work is a no-op.
            task.abort();
        }
        self.pending.store(0, Ordering::SeqCst);
    }

    /// Abort any in-flight/scheduled run (kernel close). Identical to
    /// [`cancel`](Self::cancel) — named for the lifecycle call site.
    pub fn shutdown(&self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;

    /// A run closure incrementing `counter`, optionally sleeping `run_ms` so a
    /// request can land mid-run (to exercise coalescing deterministically).
    fn counting_run(counter: &Arc<AtomicU64>, run_ms: u64) -> RunFn {
        let counter = Arc::clone(counter);
        Box::new(move || {
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                if run_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(run_ms)).await;
                }
            })
        })
    }

    fn wait_until(deadline_secs: u64, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(deadline_secs);
        while !done() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn burst_threshold_runs_now_and_resets_pending_synchronously() {
        // The burst-cap path is synchronous and deterministic (the saver's
        // contract): below the cap requests sit armed behind the long debounce;
        // at the cap the run is handed off and the counter zeroes immediately.
        crate::runtime::testing::run(async {});
        let counter = Arc::new(AtomicU64::new(0));
        let job = Maintenance::new(
            counting_run(&counter, 0),
            Duration::from_secs(60),
            Duration::from_secs(60),
            3,
        );
        job.request();
        job.request();
        assert_eq!(job.pending(), 2); // armed, debounce not yet elapsed
        job.request(); // hits the burst cap
        assert_eq!(job.pending(), 0); // handed off synchronously
    }

    #[test]
    fn immediate_first_run_with_zero_delay() {
        // delay == 0 is the pure coalesce-loop: the first request runs right
        // away (the tag refresh's contract), observed via the counter.
        crate::runtime::testing::run(async {});
        let counter = Arc::new(AtomicU64::new(0));
        let job = Maintenance::new(
            counting_run(&counter, 0),
            Duration::ZERO,
            Duration::from_millis(20),
            0,
        );
        job.request();
        wait_until(5, || counter.load(Ordering::SeqCst) >= 1);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "first run fired immediately"
        );
    }

    #[test]
    fn concurrent_requests_coalesce_into_one_rerun() {
        // Requests landing while a run executes collapse into exactly ONE
        // re-run — not one run per request. The run sleeps so both extra
        // requests land mid-run; the 200ms settle catches a spurious third run.
        crate::runtime::testing::run(async {});
        let counter = Arc::new(AtomicU64::new(0));
        let job = Maintenance::new(
            counting_run(&counter, 50),
            Duration::ZERO,
            Duration::ZERO,
            0,
        );
        job.request(); // run 1 starts, then sleeps 50ms
        std::thread::sleep(Duration::from_millis(10));
        job.request(); // coalesces → one re-run
        job.request(); // coalesces again → still one re-run
        wait_until(5, || counter.load(Ordering::SeqCst) >= 2);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(counter.load(Ordering::SeqCst), 2, "one coalesced re-run");
    }

    #[test]
    fn cancel_disarms_a_pending_run() {
        // cancel() aborts the armed debounce timer and zeroes the counter, so a
        // host's own flush path can take over (the saver's synchronous
        // shutdown). The long-debounced run never fires.
        crate::runtime::testing::run(async {});
        let counter = Arc::new(AtomicU64::new(0));
        let job = Maintenance::new(
            counting_run(&counter, 0),
            Duration::from_secs(60),
            Duration::from_secs(60),
            0,
        );
        job.request();
        assert_eq!(job.pending(), 1);
        job.cancel();
        assert_eq!(job.pending(), 0);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "the disarmed run never fired"
        );
    }
}
