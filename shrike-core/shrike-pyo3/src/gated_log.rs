//! The finalize-gated `log::Log` wrapper (#450, sibling of #435).
//!
//! pyo3-log's `Logger::log` runs `Python::attach` on whatever thread emits
//! the record — for native `tracing` events (bridged via `log-always`) that
//! is routinely a kernel-runtime thread (engine compute, the actor, timers).
//! That attach window is exactly the #435 hazard: left ungated, a late
//! emission racing interpreter exit can straddle `Py_Finalize` and abort in
//! `PyGILState_Release`. The gate's other claim sites live in this crate's
//! own code; pyo3-log's lives inside the pyo3-log crate — so instead of
//! editing it, `init_logging` installs the `pyo3_log::Logger` wrapped in
//! this thin adapter, which claims a [`Gate`] permit for the duration of
//! each `log()`/`flush()` and **drops the record silently on refusal** (at
//! finalization there is no Python `logging` left to deliver to — a lost log
//! line at exit is the correct degradation).
//!
//! `enabled()` is gated the same way (refusal ⇒ `false`): pyo3-log 0.13's
//! `enabled` is cache-only (no attach), but the permit costs two atomics and
//! keeps the wrapper's contract independent of the inner impl's details.

use crate::finalize_gate::Gate;
use log::{Log, Metadata, Record};

/// A `log::Log` that forwards to `inner` only while `gate` grants a permit;
/// once the gate closes (interpreter exiting), every record is dropped
/// without touching Python.
pub(crate) struct GatedLog<L> {
    gate: &'static Gate,
    inner: L,
}

impl<L: Log> GatedLog<L> {
    pub(crate) fn new(gate: &'static Gate, inner: L) -> Self {
        Self { gate, inner }
    }
}

impl<L: Log> Log for GatedLog<L> {
    fn enabled(&self, metadata: &Metadata) -> bool {
        match self.gate.permit() {
            Some(_permit) => self.inner.enabled(metadata),
            None => false,
        }
    }

    fn log(&self, record: &Record) {
        let Some(_permit) = self.gate.permit() else {
            return;
        };
        self.inner.log(record);
    }

    fn flush(&self) {
        let Some(_permit) = self.gate.permit() else {
            return;
        };
        self.inner.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// Counts every call that reaches it — the stand-in for pyo3-log's
    /// attaching logger (which a closed gate must never reach).
    struct Probe(Arc<AtomicUsize>);

    impl Log for Probe {
        fn enabled(&self, _metadata: &Metadata) -> bool {
            self.0.fetch_add(1, Ordering::SeqCst);
            true
        }
        fn log(&self, _record: &Record) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn flush(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn leaked_gate() -> &'static Gate {
        Box::leak(Box::new(Gate::new()))
    }

    fn emit_all(logger: &dyn Log) {
        let metadata = Metadata::builder()
            .level(log::Level::Warn)
            .target("shrike_kernel::test")
            .build();
        let _ = logger.enabled(&metadata);
        logger.log(
            &Record::builder()
                .metadata(metadata)
                .args(format_args!("late emission"))
                .build(),
        );
        logger.flush();
    }

    #[test]
    fn closed_gate_drops_records_without_reaching_inner() {
        let calls = Arc::new(AtomicUsize::new(0));
        let gate = leaked_gate();
        gate.close_and_drain(Duration::ZERO);
        let logger = GatedLog::new(gate, Probe(Arc::clone(&calls)));
        emit_all(&logger);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a closed gate must never reach the inner (attaching) logger"
        );
        let metadata = Metadata::builder().level(log::Level::Error).build();
        assert!(!logger.enabled(&metadata), "refusal reads as disabled");
    }

    #[test]
    fn open_gate_forwards_and_releases_permits() {
        let calls = Arc::new(AtomicUsize::new(0));
        let gate = leaked_gate();
        let logger = GatedLog::new(gate, Probe(Arc::clone(&calls)));
        emit_all(&logger);
        let forwarded = calls.load(Ordering::SeqCst);
        assert_eq!(forwarded, 3, "an open gate forwards enabled/log/flush");
        // Every permit was released: the drain returns immediately.
        gate.close_and_drain(Duration::from_secs(5));
        emit_all(&logger);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            forwarded,
            "post-close emissions are dropped"
        );
    }
}
