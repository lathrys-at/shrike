//! The interpreter-finalization gate (#435).
//!
//! The kernel runtime's threads outlive the Python interpreter by design
//! (#374 D) — and on CPython 3.12 under abi3 that leaves one genuine hazard:
//! a foreign thread whose `PyGILState_Ensure`..`Release` window is still open
//! when `Py_Finalize` starts. The interpreter zaps foreign thread states
//! during finalization, and the thread's `PyGILState_Release` then aborts the
//! process (`Fatal Python error: PyGILState_Release: thread state ... must be
//! current when releasing`, observed as the intermittent SIGABRT in
//! `test_exit_without_close_is_clean` on the Linux lanes). The window doesn't
//! have to *start* late: the bridge waker's `call_soon_threadsafe` releases
//! the GIL inside the call (the self-pipe socket write), so the loop thread
//! can run the result, finish `main()`, and reach `Py_Finalize` while the
//! waker thread is descheduled mid-window. pyo3 cannot guard this for us:
//! its finalizing check rides `Py_IsFinalizing`, a 3.13+ limited-API call
//! that is compiled out under `abi3-py312`.
//!
//! So the binding closes the window itself, from the Python side of the
//! boundary: a process-global gate that every foreign-thread `Python::attach`
//! site claims a [`Permit`] from, plus an `atexit` hook (registered at module
//! init) that **closes the gate and drains in-flight permits before
//! finalization proper begins** — `atexit` handlers run while the interpreter
//! is still fully alive, ahead of the finalizing flip. The SeqCst handshake
//! makes the close airtight: a claimer either sees the closed flag (and never
//! attaches) or its increment is visible to the drain (which waits for it).
//! Loop-thread attaches need no permit — they hold the GIL, so finalization
//! cannot be concurrent with them.
//!
//! Refusal semantics per site: a refused wake is dropped (the loop that
//! could observe the result is necessarily gone at exit); a refused backend
//! dispatch yields the `Unavailable` error tier. The drain is bounded
//! ([`DRAIN_DEADLINE`]) so a wedged Python backend degrades to today's
//! behavior instead of hanging exit.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use pyo3::prelude::*;

/// How long the exit hook waits for in-flight attach windows to close. The
/// normal case drains in microseconds (a waker finishing
/// `call_soon_threadsafe`); the bound only matters for a Python backend call
/// wedged mid-embed, where giving up reverts to the pre-gate exit behavior.
const DRAIN_DEADLINE: Duration = Duration::from_secs(5);
const DRAIN_TICK: Duration = Duration::from_millis(1);

/// The gate's state, separable from the process-global instance so the
/// one-way close is unit-testable.
pub(crate) struct Gate {
    finalizing: AtomicBool,
    in_flight: AtomicUsize,
}

/// A claimed attach window: holds the in-flight count up for exactly as long
/// as the holder may touch Python. Claim it *around* `Python::attach`, never
/// inside (the drain must be able to observe the window before it opens).
pub(crate) struct Permit<'a> {
    gate: &'a Gate,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.gate.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Gate {
    const fn new() -> Self {
        Self {
            finalizing: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
        }
    }

    /// Claim an attach window, or `None` once the interpreter is exiting.
    /// Increment-then-check pairs with the close's store-then-wait (both
    /// SeqCst): either this claimer sees the flag, or the drain sees this
    /// claimer.
    pub(crate) fn permit(&self) -> Option<Permit<'_>> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        if self.finalizing.load(Ordering::SeqCst) {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(Permit { gate: self })
    }

    /// Close the gate (one-way) and wait — bounded — for in-flight attach
    /// windows to finish. Runs with the GIL *released* so the windows it
    /// waits on can complete.
    fn close_and_drain(&self, deadline: Duration) {
        self.finalizing.store(true, Ordering::SeqCst);
        let until = Instant::now() + deadline;
        while self.in_flight.load(Ordering::SeqCst) > 0 && Instant::now() < until {
            std::thread::sleep(DRAIN_TICK);
        }
    }
}

static GATE: Gate = Gate::new();

/// Claim an attach window on the process gate — every foreign-thread
/// `Python::attach` in this crate goes through here.
pub(crate) fn permit() -> Option<Permit<'static>> {
    GATE.permit()
}

/// The `atexit` hook: close the gate and drain before finalization begins.
/// Also callable from tests via the module surface — idempotent, but one-way
/// for the process (only meaningful in a subprocess).
#[pyfunction]
pub(crate) fn finalize_gate_close(py: Python<'_>) {
    py.detach(|| GATE.close_and_drain(DRAIN_DEADLINE));
}

/// Register [`finalize_gate_close`] with `atexit` (called from module init,
/// so the gate is armed in every process that imports the extension).
pub(crate) fn register_exit_hook(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let hook = m.getattr("finalize_gate_close")?;
    m.py().import("atexit")?.call_method1("register", (hook,))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permit_refused_after_close() {
        let gate = Gate::new();
        gate.close_and_drain(Duration::ZERO);
        assert!(gate.permit().is_none());
        assert_eq!(gate.in_flight.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn drain_waits_for_inflight_permit() {
        let gate = std::sync::Arc::new(Gate::new());
        let permit = gate.permit().expect("gate open");
        let closer = {
            let gate = std::sync::Arc::clone(&gate);
            std::thread::spawn(move || {
                let start = Instant::now();
                gate.close_and_drain(Duration::from_secs(10));
                start.elapsed()
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        drop(permit);
        let waited = closer.join().expect("closer thread");
        assert!(waited >= Duration::from_millis(40), "drain returned early");
    }

    #[test]
    fn drain_deadline_bounds_a_stuck_permit() {
        let gate = Gate::new();
        let _stuck = gate.permit().expect("gate open");
        let start = Instant::now();
        gate.close_and_drain(Duration::from_millis(30));
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "drain never gave up"
        );
        assert!(gate.permit().is_none(), "gate must stay closed");
    }
}
