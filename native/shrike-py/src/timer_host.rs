//! The asyncio-backed `TimerHost` (#332, S3c-1) — the harness's timers,
//! injected into the kernel like its executor and its polling loop.
//!
//! `schedule` may be called from any thread (a kernel future completing on
//! the worker), but `loop.call_later` is loop-thread-only — so arming hops
//! through `call_soon_threadsafe`, and the cancel handle covers both phases:
//! cancelled-before-armed suppresses the arm; cancelled-after-armed cancels
//! the asyncio `TimerHandle`. The fired job runs as a plain loop callback
//! (brief; the kernel keeps heavy work on its executor, not in timers).

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use shrike_kernel::{TimerCancel, TimerHost};

type Job = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct TimerState {
    cancelled: bool,
    /// The asyncio TimerHandle once armed (None before the arm callback ran).
    handle: Option<Py<PyAny>>,
}

/// The Python callable that runs on the loop: phase 1 (`__call__` with no
/// args… distinguished by `armed`) arms `call_later`; the fired callback is a
/// second instance in `fire` mode.
#[pyclass]
struct TimerStep {
    event_loop: Py<PyAny>,
    delay_secs: f64,
    job: Arc<Mutex<Option<Job>>>,
    state: Arc<Mutex<TimerState>>,
    /// false → this step arms the timer; true → this step fires the job.
    fire: bool,
}

#[pymethods]
impl TimerStep {
    fn __call__(&self, py: Python<'_>) -> PyResult<()> {
        if self.fire {
            let job = self.job.lock().expect("timer poisoned").take();
            if let Some(job) = job {
                // Fired on the loop thread; release the GIL for the job (it
                // typically submits to the executor / flips kernel state).
                py.detach(job);
            }
            return Ok(());
        }
        let mut state = self.state.lock().expect("timer poisoned");
        if state.cancelled {
            return Ok(()); // cancelled before the arm callback ran
        }
        let fire_step = TimerStep {
            event_loop: self.event_loop.clone_ref(py),
            delay_secs: self.delay_secs,
            job: Arc::clone(&self.job),
            state: Arc::clone(&self.state),
            fire: true,
        };
        let handle = self
            .event_loop
            .bind(py)
            .call_method1("call_later", (self.delay_secs, Py::new(py, fire_step)?))?;
        state.handle = Some(handle.unbind());
        Ok(())
    }
}

struct LoopTimerCancel {
    state: Arc<Mutex<TimerState>>,
    job: Arc<Mutex<Option<Job>>>,
}

impl TimerCancel for LoopTimerCancel {
    fn cancel(&self) {
        let handle = {
            let mut state = self.state.lock().expect("timer poisoned");
            state.cancelled = true;
            state.handle.take()
        };
        // Drop the job either way (a cancel-before-arm must not fire it).
        self.job.lock().expect("timer poisoned").take();
        if let Some(handle) = handle {
            Python::attach(|py| {
                let _ = handle.call_method0(py, "cancel");
            });
        }
    }
}

/// The harness-provided timer host: a `TimerHost` over one asyncio loop.
#[pyclass]
pub(crate) struct LoopTimerHost {
    event_loop: Py<PyAny>,
}

#[pymethods]
impl LoopTimerHost {
    /// Capture the RUNNING loop (call from a coroutine context at assembly).
    #[staticmethod]
    fn capture(py: Python<'_>) -> PyResult<Self> {
        let asyncio = py.import("asyncio")?;
        let event_loop = asyncio.call_method0("get_running_loop")?;
        Ok(Self {
            event_loop: event_loop.unbind(),
        })
    }
}

impl TimerHost for LoopTimerHost {
    fn schedule(&self, delay_secs: f64, job: Job) -> Box<dyn TimerCancel> {
        let state = Arc::new(Mutex::new(TimerState::default()));
        let job = Arc::new(Mutex::new(Some(job)));
        Python::attach(|py| {
            let arm_step = TimerStep {
                event_loop: self.event_loop.clone_ref(py),
                delay_secs,
                job: Arc::clone(&job),
                state: Arc::clone(&state),
                fire: false,
            };
            if let Ok(step) = Py::new(py, arm_step) {
                let _ = self
                    .event_loop
                    .call_method1(py, "call_soon_threadsafe", (step,));
            }
        });
        Box::new(LoopTimerCancel { state, job })
    }
}

/// Test-only seam: arm a timer through the host and resolve a bridged future
/// when it fires — proves schedule → call_later → fire across the FFI, and
/// the cancel path, without waiting for the kernel's saver (S3c-2) to consume
/// the trait.
#[pyfunction]
#[pyo3(signature = (host, delay_secs, cancel_after=None))]
pub(crate) fn timer_probe<'py>(
    py: Python<'py>,
    host: PyRef<'py, LoopTimerHost>,
    delay_secs: f64,
    cancel_after: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (tx, rx) = futures::channel::oneshot::channel::<()>();
    let tx = Arc::new(Mutex::new(Some(tx)));
    let cancel = host.schedule(
        delay_secs,
        Box::new({
            let tx = Arc::clone(&tx);
            move || {
                if let Some(tx) = tx.lock().expect("probe poisoned").take() {
                    let _ = tx.send(());
                }
            }
        }),
    );
    if let Some(after) = cancel_after {
        // Cancel via a second timer — exercised wholly on the loop.
        host.schedule(
            after,
            Box::new(move || {
                cancel.cancel();
            }),
        );
    } else {
        // Leak-free: the handle drops without cancelling (timers stay armed).
        drop(cancel);
    }
    crate::asyncio_bridge::future_into_py(py, async move {
        Ok(rx.await.is_ok()) // false = the sender dropped (cancelled)
    })
}
