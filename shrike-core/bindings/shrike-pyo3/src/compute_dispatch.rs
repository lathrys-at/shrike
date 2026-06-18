//! Wiring the engine `Blocking` adapter to the kernel's compute pool.
//!
//! A route-1 sync engine (onnx text/CLIP, Apple Vision OCR) runs behind the
//! [`Blocking`] adapter, which schedules its chunk loop on an injected blocking
//! pool. Here the binding injects the kernel's committed compute pool
//! ([`shrike_kernel::submit_compute`]) so engine compute lands there rather than
//! tokio's default blocking pool. A compute-only build (no `anki-core`, so no
//! kernel) keeps the engines constructible by falling back to the adapter's
//! default pool.

use std::sync::Arc;

use shrike_engine_api::Blocking;

/// The kernel's compute pool as a [`BlockingDispatch`]: each job is handed to
/// [`shrike_kernel::submit_compute`], which schedules it eagerly on
/// `drive_compute` (or the blocking pool in default mode).
#[cfg(feature = "anki-core")]
struct KernelDispatch;

#[cfg(feature = "anki-core")]
impl shrike_engine_api::BlockingDispatch for KernelDispatch {
    fn submit(&self, job: Box<dyn FnOnce() + Send + 'static>) {
        shrike_kernel::submit_compute(job);
    }
}

/// Adapt a route-1 sync engine over the kernel's compute pool.
#[cfg(feature = "anki-core")]
pub(crate) fn blocking<E>(engine: Arc<E>) -> Blocking<E> {
    Blocking::with_dispatch(engine, Arc::new(KernelDispatch))
}

/// Adapt a route-1 sync engine over the adapter's default pool — the
/// compute-only build with no kernel to inject.
#[cfg(not(feature = "anki-core"))]
pub(crate) fn blocking<E>(engine: Arc<E>) -> Blocking<E> {
    Blocking::new(engine)
}
