//! `NativeEmbedder` (#342 P2): the native engines attached to the kernel
//! slot DIRECTLY — no Python on the embed hot path. The Python facade keeps
//! everything construction-shaped (file/provider resolution, the batch-safety
//! probe, fingerprint assembly, `health()`), then hands the loaded engine
//! pyclass here; this module composes it per the engine contract:
//!
//! ```text
//! OnnxTextEmbedder/ClipEmbedder (loaded engine, pure compute)
//!   └─ WithPolicy        — host-assembled fingerprint/dim + probed safe_batch
//!       └─ OnExecutor    — the route-1 adapter (owns the chunk loop)
//!           └─ AsyncioComputeLane — this host's execution: asyncio loop →
//!              default thread pool, the PyEmbedder dispatch machinery
//!              generalized into a `ComputeExecutor`
//! ```
//!
//! A kernel embed then runs: kernel future → lane submit (`call_soon_threadsafe`)
//! → loop callback → `run_in_executor` → pool thread runs the *Rust* chunk loop
//! with the GIL released → oneshot → kernel future resolves. Python schedules;
//! it never embeds. `PyEmbedder.capture` stays the test seam + custom-backend
//! escape hatch.

use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::BoxFuture;
use pyo3::prelude::*;

use shrike_engine_api::{ComputeExecutor, Embedder, ImageEmbedder, OnExecutor, WithPolicy};
use shrike_ffi::{NativeError, NativeResult};

type Job = Box<dyn FnOnce() + Send>;
type DoneTx = Arc<Mutex<Option<oneshot::Sender<NativeResult<()>>>>>;

/// Loop callback: hand the Rust job to asyncio's default thread pool. (Runs
/// on the loop thread, GIL held — scheduling only, no compute.)
#[pyclass]
struct LaneDispatch {
    event_loop: Py<PyAny>,
    job: Mutex<Option<Job>>,
    done: DoneTx,
}

#[pymethods]
impl LaneDispatch {
    fn __call__(self_: Py<Self>, py: Python<'_>) -> PyResult<()> {
        let me = self_.bind(py).borrow();
        let Some(job) = me.job.lock().expect("lane dispatch poisoned").take() else {
            return Ok(()); // double-fired callback: nothing left to run
        };
        let runner = LaneJob {
            job: Mutex::new(Some(job)),
            done: Arc::clone(&me.done),
        };
        let event_loop = me.event_loop.bind(py);
        match event_loop.call_method1("run_in_executor", (py.None(), Py::new(py, runner)?)) {
            Ok(_) => Ok(()),
            Err(e) => {
                // The pool refused (loop closing): complete the submission
                // with the failure instead of leaving the kernel hanging.
                if let Some(tx) = me.done.lock().expect("lane dispatch poisoned").take() {
                    let _ = tx.send(Err(NativeError::unavailable(format!(
                        "compute lane could not reach the thread pool: {e}"
                    ))));
                }
                Ok(())
            }
        }
    }
}

/// Pool-thread runner: the actual engine compute, GIL released for its whole
/// duration (the job is pure Rust — ort/tokenizers never touch Python).
#[pyclass]
struct LaneJob {
    job: Mutex<Option<Job>>,
    done: DoneTx,
}

#[pymethods]
impl LaneJob {
    fn __call__(&self, py: Python<'_>) {
        if let Some(job) = self.job.lock().expect("lane job poisoned").take() {
            py.detach(job);
        }
        if let Some(tx) = self.done.lock().expect("lane job poisoned").take() {
            let _ = tx.send(Ok(()));
        }
    }
}

/// The asyncio-backed [`ComputeExecutor`]: engine compute runs on asyncio's
/// default thread-pool executor — embedding is blocking and slow, so it gets
/// its own lane, never the collection executor (the re-entrancy rule).
/// Concurrent submissions genuinely overlap (one pool thread each, GIL
/// released), which is what the kernel's `try_join`ed batch futures and the
/// query-embed∥lexical search overlap cash in on.
pub(crate) struct AsyncioComputeLane {
    event_loop: Py<PyAny>,
}

impl AsyncioComputeLane {
    /// Capture the RUNNING loop — call from a coroutine context at assembly,
    /// exactly like `PyEmbedder::capture`.
    pub(crate) fn capture(py: Python<'_>) -> PyResult<Self> {
        let asyncio = py.import("asyncio")?;
        let event_loop = asyncio.call_method0("get_running_loop")?;
        Ok(Self {
            event_loop: event_loop.unbind(),
        })
    }
}

impl ComputeExecutor for AsyncioComputeLane {
    fn submit(&self, job: Job) -> BoxFuture<'static, NativeResult<()>> {
        let (tx, rx) = oneshot::channel::<NativeResult<()>>();
        let scheduled = Python::attach(|py| -> PyResult<()> {
            let dispatch = LaneDispatch {
                event_loop: self.event_loop.clone_ref(py),
                job: Mutex::new(Some(job)),
                done: Arc::new(Mutex::new(Some(tx))),
            };
            self.event_loop
                .bind(py)
                .call_method1("call_soon_threadsafe", (Py::new(py, dispatch)?,))?;
            Ok(())
        });
        Box::pin(async move {
            if let Err(e) = scheduled {
                return Err(NativeError::unavailable(format!(
                    "could not schedule onto the compute lane: {e}"
                )));
            }
            rx.await
                .map_err(|_| NativeError::unavailable("compute lane dropped the job"))?
        })
    }
}

/// The assembled native embedder the kernel slot takes: the text half always,
/// the image half when the engine embeds images (CLIP). Both halves are views
/// of ONE adapted engine sharing one lane.
#[pyclass(frozen)]
pub(crate) struct NativeEmbedder {
    pub(crate) text: Arc<dyn Embedder>,
    pub(crate) images: Option<Arc<dyn ImageEmbedder>>,
}

#[pymethods]
impl NativeEmbedder {
    /// Compose the ONNX text engine for the kernel slot. `fingerprint` is the
    /// facade-assembled identity (it folds `pool=`/`textprep=` policy the
    /// engine can't know), `dim` what the facade probed, `safe_batch` the
    /// batch-safety probe's verdict capped by the operator's batch-size
    /// setting. Call from a coroutine context (captures the running loop).
    #[staticmethod]
    #[pyo3(signature = (engine, *, fingerprint, dim, safe_batch))]
    fn from_onnx(
        py: Python<'_>,
        engine: PyRef<'_, crate::OnnxTextEmbedder>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        safe_batch: usize,
    ) -> PyResult<Self> {
        let lane: Arc<dyn ComputeExecutor> = Arc::new(AsyncioComputeLane::capture(py)?);
        let tuned = Arc::new(WithPolicy::new(
            engine.engine_arc(),
            fingerprint,
            dim,
            safe_batch,
        ));
        Ok(Self {
            text: Arc::new(OnExecutor::new(tuned, lane)),
            images: None,
        })
    }

    /// Compose the CLIP dual encoder: one engine, both modalities — the same
    /// adapted instance serves the text and image halves over one lane.
    #[staticmethod]
    #[pyo3(signature = (engine, *, fingerprint, dim, safe_batch))]
    fn from_clip(
        py: Python<'_>,
        engine: PyRef<'_, crate::ClipEmbedder>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        safe_batch: usize,
    ) -> PyResult<Self> {
        let lane: Arc<dyn ComputeExecutor> = Arc::new(AsyncioComputeLane::capture(py)?);
        let tuned = Arc::new(WithPolicy::new(
            engine.engine_arc(),
            fingerprint,
            dim,
            safe_batch,
        ));
        let adapted = Arc::new(OnExecutor::new(tuned, lane));
        Ok(Self {
            text: Arc::clone(&adapted) as Arc<dyn Embedder>,
            images: Some(adapted as Arc<dyn ImageEmbedder>),
        })
    }
}

/// Test seam: one embed through the full native composition (kernel trait →
/// lane → pool thread → engine chunk loop → oneshot → bridged await), proving
/// the assembly before the kernel's embed-coupled ops ride it.
#[pyfunction]
pub(crate) fn native_embedder_probe<'py>(
    py: Python<'py>,
    embedder: PyRef<'py, NativeEmbedder>,
    texts: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let text = Arc::clone(&embedder.text);
    crate::asyncio_bridge::future_into_py(py, async move { text.embed(texts).await })
}
