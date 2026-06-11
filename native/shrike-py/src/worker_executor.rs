//! The harness-owned worker executor (#332, S3b).
//!
//! The kernel's `SerialExecutor` contract says scheduling is *injected*: the
//! kernel owns no threads. This implementation keeps that literal — the
//! **Python harness** creates the worker (`threading.Thread(target=ex.worker_loop)`)
//! and the executor is just a FIFO channel between `submit` (called from
//! kernel futures, any thread) and `worker_loop` (the harness's thread, GIL
//! released for its whole life). Collection jobs therefore run **off the
//! asyncio loop**, unlike S3a's inline `MutexExecutor` (which runs them
//! inside the poll callback) — the loop only ever polls.
//!
//! Shutdown: `shutdown()` closes the channel; `worker_loop` drains what's
//! queued and returns, so `thread.join()` on the harness side is clean.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::BoxFuture;
use pyo3::prelude::*;

use shrike_kernel::SerialExecutor;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Inner {
    sender: Mutex<Option<mpsc::Sender<Job>>>,
    receiver: Mutex<Option<mpsc::Receiver<Job>>>,
}

/// A `SerialExecutor` whose serialization is one harness-owned worker thread.
#[pyclass]
pub(crate) struct WorkerExecutor {
    inner: Arc<Inner>,
}

/// The Rust-facing handle the kernel composes over (cheap to clone; the
/// pyclass stays the Python-facing lifecycle surface).
pub(crate) struct WorkerHandle {
    inner: Arc<Inner>,
}

impl SerialExecutor for WorkerHandle {
    fn submit(&self, job: Job) -> BoxFuture<'static, ()> {
        let (tx, rx) = oneshot::channel::<()>();
        let wrapped: Job = Box::new(move || {
            job();
            let _ = tx.send(());
        });
        let sent = {
            let guard = self.inner.sender.lock().expect("executor poisoned");
            match guard.as_ref() {
                Some(sender) => sender.send(wrapped).is_ok(),
                None => false,
            }
        };
        Box::pin(async move {
            if sent {
                // A dropped worker (shutdown mid-job) resolves the future too —
                // the kernel's oneshot result channel then reports the drop.
                let _ = rx.await;
            }
        })
    }
}

#[pymethods]
impl WorkerExecutor {
    #[new]
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        Self {
            inner: Arc::new(Inner {
                sender: Mutex::new(Some(sender)),
                receiver: Mutex::new(Some(receiver)),
            }),
        }
    }

    /// The worker body — the harness runs this on a thread IT owns
    /// (`threading.Thread(target=executor.worker_loop, daemon=True)`).
    /// Blocks (GIL released) until `shutdown()`; runs jobs FIFO.
    fn worker_loop(&self, py: Python<'_>) -> PyResult<()> {
        let receiver = self
            .inner
            .receiver
            .lock()
            .expect("executor poisoned")
            .take()
            .ok_or_else(|| {
                crate::NativeInternalError::new_err("worker_loop already running (or ran)")
            })?;
        py.detach(move || {
            while let Ok(job) = receiver.recv() {
                job();
            }
        });
        Ok(())
    }

    /// Close the queue: `worker_loop` drains and returns; later submits become
    /// no-op futures whose collection ops report the executor as gone.
    fn shutdown(&self) {
        self.inner.sender.lock().expect("executor poisoned").take();
    }
}

impl WorkerExecutor {
    pub(crate) fn handle(&self) -> WorkerHandle {
        WorkerHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}
