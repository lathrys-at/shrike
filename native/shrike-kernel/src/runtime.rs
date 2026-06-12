//! The kernel's owned tokio runtime (#374): one runtime per process, owned
//! HERE — the walk-back of the injected-executor model. The kernel is
//! idiomatic async Rust; hosts adapt the *action exchange* (an op in, a
//! completion-backed future out via [`spawn_op`]) and never supply
//! scheduling.
//!
//! anki's own lazy runtime is never instantiated on Shrike's call paths
//! (its sole consumers are the AnkiWeb/AnkiHub sync services, which Shrike
//! never invokes — pinned in shrike-collection), so this is the only tokio
//! runtime that exists in the process.
//!
//! The default is a multi-thread runtime; [`init_runtime`] lets a host (or
//! the degenerate-mode proof test) install a custom-built one — e.g.
//! `current_thread`, where the entire kernel runs on a single thread driving
//! its own asynchrony via [`block_on`]. Nothing in the kernel uses
//! `block_in_place` or assumes worker threads, which is what keeps that mode
//! honest.

use std::future::Future;
use std::sync::OnceLock;

use shrike_ffi::{NativeError, NativeResult};

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// Install a custom-built runtime before the first kernel op (the
/// degenerate single-thread proof; a host with tuned pools). Errors with
/// the runtime handed back if one is already installed.
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
/// context (tests; a synchronous embedded host). Panics if called from
/// inside the runtime (tokio's nested-block_on guard) — async callers just
/// `.await`.
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

/// The action-exchange edge (#374): spawn an op onto the kernel runtime and
/// hand back a small Send future — a oneshot receiver, pollable from ANY
/// context with no tokio dependency on the caller's side. **Dropping the
/// returned future detaches observation; the op runs to completion** (a
/// half-applied collection write from an abort would be far worse than a
/// wasted compute) — byte-identical to the pre-#374 cancellation semantics.
/// A oneshot that closes without a value means the op task panicked.
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
