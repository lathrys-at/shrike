//! The shared "test plays the host" harness for the cabi lifecycle tests.
//!
//! shrike-cabi spawns no threads — it exposes the kernel's blocking drive
//! entries, and the HOST commits + joins the OS threads. Each lifecycle test
//! plays that host: it installs the driven runtime, parks the committed threads
//! in `shrike_drive_io`/`shrike_drive_sync`/`shrike_drive_compute`, runs its
//! flow, calls `shrike_runtime_shutdown`, and joins its own threads — the exact
//! contract a Swift/Kotlin app follows.
//!
//! The startup ORDERING is load-bearing (it is itself under test): the IO thread
//! must enter the runtime's `block_on` FIRST so it owns tokio's IO/timer
//! drivers. [`Host::start`] enforces it with `shrike_runtime_probe` — a barrier
//! that blocks until the IO thread is driving — before parking the sync/compute
//! threads. A regression that parks them too early (or drops the probe) races
//! driver ownership and starves timers/IO.
//!
//! Lives in `tests/host/mod.rs` (a subdirectory module, NOT `tests/host.rs`) so
//! cargo/bazel don't compile it as its own test binary — it has no `#[test]`,
//! and each lifecycle binary pulls it in with `mod host;`.

// Each lifecycle binary uses a different subset of this harness; silence the
// per-binary dead-code warnings for the parts a given binary doesn't touch.
#![allow(dead_code)]

use std::thread::JoinHandle;
use std::time::Duration;

use shrike_cabi::{
    shrike_drive_compute, shrike_drive_io, shrike_drive_sync, shrike_runtime_init,
    shrike_runtime_probe, shrike_runtime_shutdown,
};

/// The committed driver threads the test (as host) owns. Two compute threads
/// give the "N >= 2" engine overlap; the lifecycle tests don't exercise an
/// engine, but the count proves the host picks N (cabi spawns nothing).
const COMPUTE_THREADS: usize = 2;

/// How long to wait for a committed thread to return at teardown before failing
/// the test (a hang here is the regression the join guards — it must surface as
/// a bounded failure, not an infinite hang).
const JOIN_TIMEOUT: Duration = Duration::from_secs(10);

/// The test's committed driver threads, joined on [`Host::shutdown`].
pub struct Host {
    threads: Vec<JoinHandle<bool>>,
}

impl Host {
    /// Install the driven runtime and park the committed N + 2 threads in the C
    /// drive entries, honoring the startup barrier: spawn the IO thread, PROBE
    /// until it is driving, then spawn sync + N compute. Asserts the driven
    /// install and the probe so a misuse fails loudly rather than hanging later.
    pub fn start() -> Self {
        assert!(
            shrike_runtime_init(),
            "shrike_runtime_init installs the driven runtime first in this process"
        );

        let mut threads = Vec::with_capacity(2 + COMPUTE_THREADS);
        // 1. The IO thread FIRST — it must win tokio's first-block_on driver
        //    ownership.
        threads.push(spawn("shrike-io", shrike_drive_io));
        // 2. The barrier: block until the IO thread is in its block_on driving
        //    the executor. Only then is it safe to park the rest.
        assert!(
            shrike_runtime_probe(),
            "shrike_runtime_probe confirms the IO thread is driving before the rest park"
        );
        // 3. Now sync + N compute — they hook into the IO thread's drivers.
        threads.push(spawn("shrike-sync", shrike_drive_sync));
        for i in 0..COMPUTE_THREADS {
            threads.push(spawn(&format!("shrike-work-{i}"), shrike_drive_compute));
        }
        Host { threads }
    }

    /// Shut the runtime down (cabi drains in-flight ops + closes the pools) and
    /// JOIN the committed threads (the host owns them). Each drive entry returns
    /// `true` on a clean return; a join that times out fails the test (a hung
    /// drive thread is the regression the bounded join guards).
    pub fn shutdown(self) {
        shrike_runtime_shutdown();
        for thread in self.threads {
            join_within(thread, JOIN_TIMEOUT);
        }
    }

    /// Shut the runtime down WITHOUT joining — for a test that wants to assert
    /// post-shutdown behaviour (the runtime is undriven once the drive threads
    /// have returned) and join afterward via [`Host::join`].
    pub fn shutdown_no_join(&mut self) {
        shrike_runtime_shutdown();
    }

    /// Join the committed threads (paired with [`Host::shutdown_no_join`]).
    pub fn join(self) {
        for thread in self.threads {
            join_within(thread, JOIN_TIMEOUT);
        }
    }
}

/// Spawn one committed driver thread parking in `entry` (a C drive entry).
fn spawn(name: &str, entry: extern "C" fn() -> bool) -> JoinHandle<bool> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || entry())
        .expect("a host driver thread spawns")
}

/// Join `thread`, failing the test if it doesn't return within `timeout`.
/// `std::thread::JoinHandle` has no timed join, so poll `is_finished` — the
/// committed threads return promptly once `shutdown_driven_pools` closes the
/// queues, so this loop spins only on a regression (then fails bounded).
fn join_within(thread: JoinHandle<bool>, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while !thread.is_finished() {
        assert!(
            std::time::Instant::now() < deadline,
            "a committed drive thread did not return within {timeout:?} of shutdown — \
             shutdown_driven_pools must close the queues so the drive entries return"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let clean = thread.join().expect("the drive thread did not panic");
    assert!(clean, "a drive entry returned false (a driven-mode misuse)");
}
