//! `PyRecognizer` (#228): the kernel's `Recognizer` seam implemented over the
//! harness's backend — the same inversion as `PyEmbedder` (the kernel drives
//! ANY Python-held recognizer: Apple Vision via pyobjc, Tesseract, a remote
//! engine — without knowing Python exists), and the same threading shape:
//! `recognize()` returns a oneshot-backed future; the call is scheduled onto
//! the asyncio loop, which dispatches the blocking `backend.recognize` to
//! asyncio's default thread-pool executor and resolves the oneshot from the
//! done-callback. Never the collection executor, never a poll callback.
//!
//! Wire contract with the Python backend: `recognize(items: list[bytes]) ->
//! list[tuple[str, float, str]]` — `(text, confidence, segments_json)`, where
//! `segments_json` is a JSON array of `{text, confidence, bbox?}` (or `""`
//! for none). Tuples keep the boundary explicit and dependency-free; the
//! Python protocol layer wraps them in dataclasses.

use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::BoxFuture;
use pyo3::prelude::*;

use shrike_ffi::{NativeError, NativeResult};
use shrike_kernel::{MediaItem, Recognition, Recognizer, Segment};

type RecResult = NativeResult<Vec<Recognition>>;

/// Loop callback: dispatch the blocking recognize to the pool and chain the
/// done-callback. (Runs on the loop thread, GIL held.)
#[pyclass]
struct RecognizeDispatch {
    backend: Py<PyAny>,
    event_loop: Py<PyAny>,
    items: Mutex<Option<Vec<Vec<u8>>>>,
    tx: Arc<Mutex<Option<oneshot::Sender<RecResult>>>>,
}

#[pymethods]
impl RecognizeDispatch {
    fn __call__(&self, py: Python<'_>) -> PyResult<()> {
        let Some(items) = self.items.lock().expect("dispatch poisoned").take() else {
            return Ok(());
        };
        let recognize = self.backend.bind(py).getattr("recognize")?;
        let arg: Py<PyAny> = items.into_pyobject(py)?.into_any().unbind();
        let pool_future = self
            .event_loop
            .bind(py)
            .call_method1("run_in_executor", (py.None(), recognize, arg))?;
        let done = RecognizeDone {
            tx: Arc::clone(&self.tx),
        };
        pool_future.call_method1("add_done_callback", (Py::new(py, done)?,))?;
        Ok(())
    }
}

/// Pool-future done-callback: parse the tuples (or map the failure) and
/// resolve the kernel's oneshot. (Runs on the loop thread, GIL held.)
#[pyclass]
struct RecognizeDone {
    tx: Arc<Mutex<Option<oneshot::Sender<RecResult>>>>,
}

fn parse_results(raw: Vec<(String, f64, String)>) -> RecResult {
    raw.into_iter()
        .map(|(text, confidence, segments_json)| {
            let segments: Vec<Segment> = if segments_json.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str(&segments_json).map_err(|e| {
                    NativeError::internal(format!("recognizer returned bad segments JSON: {e}"))
                })?
            };
            Ok(Recognition {
                text,
                confidence,
                segments,
            })
        })
        .collect()
}

#[pymethods]
impl RecognizeDone {
    fn __call__(&self, py: Python<'_>, future: Bound<'_, PyAny>) -> PyResult<()> {
        let outcome: RecResult = match future.call_method0("result") {
            Ok(value) => value
                .extract::<Vec<(String, f64, String)>>()
                .map_err(|e| {
                    NativeError::internal(format!(
                        "recognizer returned a non-(text, confidence, segments_json) payload: {e}"
                    ))
                })
                .and_then(parse_results),
            Err(e) => Err(NativeError::unavailable(format!(
                "harness recognizer failed: {}",
                e.value(py)
            ))),
        };
        if let Some(tx) = self.tx.lock().expect("recognizer poisoned").take() {
            let _ = tx.send(outcome);
        }
        Ok(())
    }
}

/// The kernel-facing handle: a Python backend + the loop hosting its calls.
/// The fingerprint is captured once at assembly so the kernel's invalidation
/// rules never call into Python mid-op.
pub(crate) struct PyRecognizerHandle {
    backend: Py<PyAny>,
    event_loop: Py<PyAny>,
    fingerprint: Option<String>,
}

impl Recognizer for PyRecognizerHandle {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, RecResult> {
        // The Python wire stays `list[bytes]` (the RecognizerBackend
        // contract); native engines are where the mime hint pays off.
        let items: Vec<Vec<u8>> = items.into_iter().map(|m| m.bytes).collect();
        let (tx, rx) = oneshot::channel::<RecResult>();
        let tx = Arc::new(Mutex::new(Some(tx)));
        let scheduled = Python::attach(|py| -> PyResult<()> {
            let dispatch = RecognizeDispatch {
                backend: self.backend.clone_ref(py),
                event_loop: self.event_loop.clone_ref(py),
                items: Mutex::new(Some(items)),
                tx,
            };
            self.event_loop
                .bind(py)
                .call_method1("call_soon_threadsafe", (Py::new(py, dispatch)?,))?;
            Ok(())
        });
        Box::pin(async move {
            if let Err(e) = scheduled {
                return Err(NativeError::unavailable(format!(
                    "recognize dispatch failed: {e}"
                )));
            }
            rx.await
                .map_err(|_| NativeError::unavailable("recognizer dropped without a result"))?
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }
}

/// The Python-visible wrapper the harness constructs at assembly:
/// `Recognizer(backend, loop, fingerprint)`.
#[pyclass(name = "Recognizer")]
pub struct PyRecognizer {
    pub(crate) handle: Arc<PyRecognizerHandle>,
}

#[pymethods]
impl PyRecognizer {
    /// Capture the harness's recognizer backend + the RUNNING asyncio loop
    /// (call from a coroutine, like `PyEmbedder.capture`). The fingerprint
    /// (`model_fingerprint()`, optional) is read once here so the kernel's
    /// invalidation rules never call into Python mid-op.
    #[staticmethod]
    fn capture(py: Python<'_>, backend: Py<PyAny>) -> PyResult<Self> {
        let asyncio = py.import("asyncio")?;
        let event_loop = asyncio.call_method0("get_running_loop")?;
        let fingerprint = backend
            .bind(py)
            .call_method0("model_fingerprint")
            .ok()
            .and_then(|v| v.extract::<String>().ok());
        Ok(Self {
            handle: Arc::new(PyRecognizerHandle {
                backend,
                event_loop: event_loop.unbind(),
                fingerprint,
            }),
        })
    }
}
