//! Runtime-less Rust-future → asyncio bridge (#332, S3a).
//!
//! The kernel's ops are runtime-agnostic futures (#310); the Python harness's
//! executor of choice is its **asyncio loop** — so this bridge makes the loop
//! itself drive them, honoring the no-owned-runtimes constraint end to end
//! (pyo3-async-runtimes was rejected for S3 because it polls Rust futures on
//! a tokio runtime owned by the binding; verdict recorded on #332):
//!
//! - `future_into_py` wraps a kernel future in an `asyncio.Future` and
//!   schedules a poll callback on the running loop.
//! - Each poll runs as a plain loop callback (GIL held, loop thread); the
//!   future's waker re-schedules the next poll via
//!   `loop.call_soon_threadsafe`, so a wake from any thread (an executor
//!   worker completing a collection job) lands back on the loop.
//! - Cancellation is cooperative: a cancelled `asyncio.Future` drops the Rust
//!   future at its next poll.
//!
//! No threads, no runtime, ~one screen of glue: the loop is the executor.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use pyo3::prelude::*;
use pyo3::BoundObject;

use shrike_ffi::NativeResult;

use crate::to_py_err;

/// The type-erased pending work: a future resolving to a GIL-deferred
/// conversion (the output is turned into a Python object *inside the poll
/// callback*, where the GIL is already held).
type Conversion = Box<dyn FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send>;
type ErasedFuture = Pin<Box<dyn Future<Output = Conversion> + Send>>;

/// Wakes by scheduling the poll callback back onto the asyncio loop — safe
/// from any thread (`call_soon_threadsafe` is asyncio's cross-thread door).
struct LoopWake {
    event_loop: Py<PyAny>,
    poll_cb: Py<PollCallback>,
}

impl Wake for LoopWake {
    fn wake(self: Arc<Self>) {
        Python::attach(|py| {
            let _ = self.event_loop.call_method1(
                py,
                "call_soon_threadsafe",
                (self.poll_cb.clone_ref(py),),
            );
        });
    }
}

/// The loop callback that advances the bridged future by one poll.
#[pyclass]
pub(crate) struct PollCallback {
    future: Arc<Mutex<Option<ErasedFuture>>>,
    py_future: Py<PyAny>,
    event_loop: Py<PyAny>,
}

#[pymethods]
impl PollCallback {
    fn __call__(self_: Py<PollCallback>, py: Python<'_>) -> PyResult<()> {
        let this = self_.borrow(py);
        // A cancelled asyncio.Future drops the Rust future (cooperative).
        if this
            .py_future
            .bind(py)
            .call_method0("cancelled")?
            .is_truthy()?
        {
            this.future.lock().expect("bridge poisoned").take();
            return Ok(());
        }
        let mut slot = this.future.lock().expect("bridge poisoned");
        let Some(fut) = slot.as_mut() else {
            return Ok(()); // already completed (a late wake) — nothing to do
        };
        let waker = Waker::from(Arc::new(LoopWake {
            event_loop: this.event_loop.clone_ref(py),
            poll_cb: self_.clone_ref(py),
        }));
        let mut cx = Context::from_waker(&waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Pending => Ok(()),
            Poll::Ready(convert) => {
                *slot = None;
                drop(slot);
                let py_fut = this.py_future.bind(py);
                if py_fut.call_method0("done")?.is_truthy()? {
                    return Ok(()); // raced a cancellation — result has nowhere to go
                }
                match convert(py) {
                    Ok(value) => py_fut.call_method1("set_result", (value,))?,
                    Err(err) => py_fut.call_method1("set_exception", (err,))?,
                };
                Ok(())
            }
        }
    }
}

/// Bridge a kernel future into an `asyncio.Future` awaitable on the running
/// loop. Must be called from a coroutine context (the loop must be running).
pub(crate) fn future_into_py<'py, F, T>(py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = NativeResult<T>> + Send + 'static,
    T: for<'p> IntoPyObject<'p> + Send + 'static,
{
    let asyncio = py.import("asyncio")?;
    let event_loop = asyncio.call_method0("get_running_loop")?;
    let py_future = event_loop.call_method0("create_future")?;

    let erased: ErasedFuture = Box::pin(async move {
        let out = fut.await;
        Box::new(move |py: Python<'_>| match out {
            Ok(value) => value
                .into_pyobject(py)
                .map(|bound| bound.into_any().unbind())
                .map_err(Into::into),
            Err(err) => Err(to_py_err(err)),
        }) as Conversion
    });

    let poll_cb = Py::new(
        py,
        PollCallback {
            future: Arc::new(Mutex::new(Some(erased))),
            py_future: py_future.clone().unbind(),
            event_loop: event_loop.clone().unbind(),
        },
    )?;
    // Kick off the first poll as an ordinary loop callback.
    event_loop.call_method1("call_soon", (poll_cb,))?;
    Ok(py_future)
}
