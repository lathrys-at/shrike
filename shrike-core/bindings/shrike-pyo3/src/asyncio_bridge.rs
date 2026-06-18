//! Runtime-less Rust-future → asyncio bridge.
//!
//! The kernel's ops are runtime-agnostic futures; the Python harness's
//! executor of choice is its **asyncio loop** — so this bridge makes the loop
//! itself drive them, honoring the no-owned-runtimes constraint end to end
//! (pyo3-async-runtimes was rejected because it polls Rust futures on
//! a tokio runtime owned by the binding):
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyWeakrefMethods, PyWeakrefReference};
use pyo3::BoundObject;

use shrike_error::NativeResult;

use crate::to_py_err;

/// The type-erased pending work: a future resolving to a GIL-deferred
/// conversion (the output is turned into a Python object *inside the poll
/// callback*, where the GIL is already held).
type Conversion = Box<dyn FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send>;
type ErasedFuture = Pin<Box<dyn Future<Output = Conversion> + Send>>;

/// Live [`PollCallback`] count — the bridge's leak tripwire. Counted
/// Rust-side (construction vs `Drop`) so the tests can assert release without
/// trusting Python's GC to even *see* the objects.
static LIVE_POLL_CALLBACKS: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn live_poll_callbacks() -> usize {
    LIVE_POLL_CALLBACKS.load(Ordering::SeqCst)
}

/// Wakes by scheduling the poll callback back onto the asyncio loop — safe
/// from any thread (`call_soon_threadsafe` is asyncio's cross-thread door).
struct LoopWake {
    event_loop: Py<PyAny>,
    /// WEAK by design: a pending future stores its waker, so a strong
    /// reference here would close a cycle through Rust (future → waker →
    /// callback → future slot) that Python's GC cannot traverse — a loop
    /// abandoned mid-op would leak the op's whole bridge state for the life
    /// of the interpreter. The reference that keeps the callback alive while
    /// the op is observable is the asyncio future's done-callback
    /// registration (see [`schedule_first_poll`]).
    poll_cb: Py<PyWeakrefReference>,
}

impl Wake for LoopWake {
    fn wake(self: Arc<Self>) {
        // Interpreter exiting ⇒ drop the wake: nobody can await the
        // result once the loops are gone, and attaching from this foreign
        // thread concurrent with Py_Finalize aborts the process. The permit
        // also covers the window where `call_soon_threadsafe` releases the
        // GIL mid-call (its self-pipe write) — the atexit drain waits for
        // this wake to finish before finalization proper begins.
        let Some(_permit) = crate::finalize_gate::permit() else {
            return;
        };
        Python::attach(|py| {
            // Callback gone ⇒ the asyncio future died with its loop and
            // nobody can observe the result — the wake has nowhere to land.
            let Some(poll_cb) = self.poll_cb.bind(py).upgrade() else {
                return;
            };
            let _ = self
                .event_loop
                .call_method1(py, "call_soon_threadsafe", (poll_cb,));
        });
    }
}

/// The loop callback that advances the bridged future by one poll. Also
/// registered as the asyncio future's done callback: that registration
/// is the strong reference keeping the bridge state alive while the op is
/// observable (the waker's is weak), and it makes cancellation cleanup prompt
/// — the cancelled branch below runs on the cancellation itself instead of
/// waiting for a wake that may never come.
///
/// `weakref` because [`LoopWake`] holds it weakly; `__traverse__`/`__clear__`
/// because the remaining `callback ↔ asyncio future` loop is a pure Python
/// reference cycle the GC must be able to see through this class.
#[pyclass(weakref)]
pub(crate) struct PollCallback {
    future: Arc<Mutex<Option<ErasedFuture>>>,
    py_future: Py<PyAny>,
    event_loop: Py<PyAny>,
}

impl PollCallback {
    fn new(future: ErasedFuture, py_future: Py<PyAny>, event_loop: Py<PyAny>) -> Self {
        LIVE_POLL_CALLBACKS.fetch_add(1, Ordering::SeqCst);
        Self {
            future: Arc::new(Mutex::new(Some(future))),
            py_future,
            event_loop,
        }
    }
}

impl Drop for PollCallback {
    fn drop(&mut self) {
        LIVE_POLL_CALLBACKS.fetch_sub(1, Ordering::SeqCst);
    }
}

#[pymethods]
impl PollCallback {
    // The optional argument is the done-callback shape: asyncio invokes
    // `cb(future)`; the loop's own `call_soon` polls invoke `cb()`.
    #[pyo3(signature = (_fut = None))]
    fn __call__(
        self_: Py<PollCallback>,
        py: Python<'_>,
        _fut: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
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
            poll_cb: PyWeakrefReference::new(self_.bind(py).as_any())?.unbind(),
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

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.py_future)?;
        visit.call(&self.event_loop)
    }

    fn __clear__(&mut self) {
        // Drop the pending Rust future: that releases the op's receiver and
        // its stored waker (the bridge's only non-traversable references).
        // The Py fields drop at dealloc, and the asyncio future's own
        // tp_clear severs the done-callback edge of the cycle.
        if let Ok(mut slot) = self.future.lock() {
            slot.take();
        }
    }
}

/// Bridge a future that already speaks Python (`PyResult<Py<PyAny>>`) — the
/// `run_job` shape, where the payload IS a Python value and a failure IS a
/// Python exception to rethrow as-is (no NativeError mapping).
#[cfg(feature = "anki-core")]
pub(crate) fn pyresult_future_into_py<'py, F>(
    py: Python<'py>,
    fut: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = PyResult<Py<PyAny>>> + Send + 'static,
{
    let asyncio = py.import("asyncio")?;
    let event_loop = asyncio.call_method0("get_running_loop")?;
    let py_future = event_loop.call_method0("create_future")?;

    let erased: ErasedFuture = Box::pin(async move {
        let out = fut.await;
        Box::new(move |_py: Python<'_>| out) as Conversion
    });

    schedule_first_poll(py, erased, &py_future, &event_loop)?;
    Ok(py_future)
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

    schedule_first_poll(py, erased, &py_future, &event_loop)?;
    Ok(py_future)
}

/// Wrap the erased future in a poll callback and kick off its first poll as
/// an ordinary loop callback (shared by both bridge entry points).
fn schedule_first_poll(
    py: Python<'_>,
    erased: ErasedFuture,
    py_future: &Bound<'_, PyAny>,
    event_loop: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let poll_cb = Py::new(
        py,
        PollCallback::new(
            erased,
            py_future.clone().unbind(),
            event_loop.clone().unbind(),
        ),
    )?;
    // The done-callback registration is the strong reference that keeps the
    // bridge state alive while the op is observable — the waker holds the
    // callback weakly, so without it the callback would die after the
    // first Pending poll. It also fires on cancellation, dropping the Rust
    // future promptly via the cancelled branch of `__call__`.
    py_future.call_method1("add_done_callback", (poll_cb.clone_ref(py),))?;
    event_loop.call_method1("call_soon", (poll_cb,))?;
    Ok(())
}

/// Test seam: a bridged future that never resolves but *retains its
/// waker* — the exact in-flight-op shape (a oneshot receiver parks its waker
/// the same way) whose stored waker once formed the leaking cycle.
struct ParkedForever(Option<Waker>);

impl Future for ParkedForever {
    type Output = NativeResult<i64>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0 = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// Test seam: the live [`PollCallback`] count.
#[pyfunction]
pub(crate) fn bridge_live_poll_callbacks() -> usize {
    live_poll_callbacks()
}

/// Test seam: bridge a waker-retaining, never-resolving future.
#[pyfunction]
pub(crate) fn bridge_parked_forever(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    future_into_py(py, ParkedForever(None))
}
