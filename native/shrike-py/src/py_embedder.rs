//! `PyEmbedder` (#332, S3c-2b): the kernel's `Embedder` seam implemented over
//! the *harness's* backend — the inversion that lets the kernel drive ANY
//! Python-held embedder (the ONNX facades, llama-server, a future platform
//! API) without the kernel knowing Python exists.
//!
//! Threading shape (all machinery harness-owned, per the runtime model):
//! `embed()` returns a oneshot-backed future; the call is scheduled onto the
//! asyncio loop (`call_soon_threadsafe`), where a loop callback dispatches the
//! actual `backend.embed_texts` to **asyncio's default thread-pool executor**
//! (`loop.run_in_executor(None, ...)`) — embedding is blocking and slow, so it
//! gets its own lane: never the collection executor (the re-entrancy rule),
//! never a poll callback. A done-callback resolves the oneshot with the
//! vectors (or the error, mapped to `Unavailable` — an embedding failure is a
//! runtime-dependency failure, not a bug).

use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::BoxFuture;
use pyo3::prelude::*;

use shrike_ffi::{NativeError, NativeResult};
use shrike_kernel::Embedder;
use shrike_kernel::{ImageEmbedder, ImageResolver, MediaItem};

type VecResult = NativeResult<Vec<Vec<f32>>>;

/// What one dispatch embeds: a text batch (`embed_texts`) or an image-bytes
/// batch (`embed_images`) — same loop → pool → oneshot shape either way.
enum EmbedPayload {
    Texts(Vec<String>),
    Images(Vec<Vec<u8>>),
}

impl EmbedPayload {
    fn method(&self) -> &'static str {
        match self {
            Self::Texts(_) => "embed_texts",
            Self::Images(_) => "embed_images",
        }
    }
}

/// Loop callback: dispatch the blocking embed to the pool and chain the
/// done-callback. (Runs on the loop thread, GIL held.)
#[pyclass]
struct EmbedDispatch {
    backend: Py<PyAny>,
    event_loop: Py<PyAny>,
    payload: Mutex<Option<EmbedPayload>>,
    tx: Arc<Mutex<Option<oneshot::Sender<VecResult>>>>,
}

#[pymethods]
impl EmbedDispatch {
    fn __call__(&self, py: Python<'_>) -> PyResult<()> {
        let Some(payload) = self.payload.lock().expect("dispatch poisoned").take() else {
            return Ok(()); // double-fired callback: nothing left to send
        };
        let embed = self.backend.bind(py).getattr(payload.method())?;
        let arg: Py<PyAny> = match payload {
            EmbedPayload::Texts(texts) => texts.into_pyobject(py)?.into_any().unbind(),
            EmbedPayload::Images(images) => images.into_pyobject(py)?.into_any().unbind(),
        };
        let pool_future = self
            .event_loop
            .bind(py)
            .call_method1("run_in_executor", (py.None(), embed, arg))?;
        let done = EmbedDone {
            tx: Arc::clone(&self.tx),
        };
        pool_future.call_method1("add_done_callback", (Py::new(py, done)?,))?;
        Ok(())
    }
}

/// Pool-future done-callback: extract the vectors (or map the failure) and
/// resolve the kernel's oneshot. (Runs on the loop thread, GIL held.)
#[pyclass]
struct EmbedDone {
    tx: Arc<Mutex<Option<oneshot::Sender<VecResult>>>>,
}

#[pymethods]
impl EmbedDone {
    fn __call__(&self, py: Python<'_>, future: Bound<'_, PyAny>) -> PyResult<()> {
        let outcome: VecResult = match future.call_method0("result") {
            Ok(value) => value.extract::<Vec<Vec<f32>>>().map_err(|e| {
                NativeError::internal(format!("embedder returned a non-vector payload: {e}"))
            }),
            Err(e) => Err(NativeError::unavailable(format!(
                "harness embedder failed: {}",
                e.value(py)
            ))),
        };
        if let Some(tx) = self.tx.lock().expect("embedder poisoned").take() {
            let _ = tx.send(outcome);
        }
        Ok(())
    }
}

/// The kernel-facing handle: a Python backend + the loop that hosts its calls.
/// Fingerprint/dim are captured once at assembly (cheap Python attribute
/// reads) so the kernel's drift rules never call into Python mid-op.
pub(crate) struct PyEmbedderHandle {
    backend: Py<PyAny>,
    event_loop: Py<PyAny>,
    fingerprint: Option<String>,
    dim: Option<usize>,
    embeds_images: bool,
}

impl PyEmbedderHandle {
    /// Whether the wrapped backend advertises the image modality.
    pub(crate) fn embeds_images(&self) -> bool {
        self.embeds_images
    }

    /// Schedule one embed dispatch onto the loop and await its oneshot — the
    /// shared spine of the text and image halves.
    fn dispatch(&self, payload: EmbedPayload) -> BoxFuture<'_, VecResult> {
        let (tx, rx) = oneshot::channel::<VecResult>();
        let tx = Arc::new(Mutex::new(Some(tx)));
        let scheduled = Python::attach(|py| -> PyResult<()> {
            let dispatch = EmbedDispatch {
                backend: self.backend.clone_ref(py),
                event_loop: self.event_loop.clone_ref(py),
                payload: Mutex::new(Some(payload)),
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
                    "could not schedule the embed onto the loop: {e}"
                )));
            }
            rx.await.map_err(|_| {
                NativeError::unavailable("the embed dispatch was dropped (loop gone?)")
            })?
        })
    }
}

impl Embedder for PyEmbedderHandle {
    fn fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }

    fn dim(&self) -> Option<usize> {
        self.dim
    }

    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, VecResult> {
        self.dispatch(EmbedPayload::Texts(texts))
    }
}

impl ImageEmbedder for PyEmbedderHandle {
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, VecResult> {
        // The Python wire stays `list[bytes]` — the mime hint is for native
        // engines; Python backends (and PIL) sniff.
        self.dispatch(EmbedPayload::Images(
            images.into_iter().map(|m| m.bytes).collect(),
        ))
    }
}

/// The media resolver over harness callables: `read(name) -> bytes | None`
/// and `exists(name) -> bool` (the server closes both over the collection's
/// media dir, exactly like the Python facade's resolver pair). Called
/// synchronously inside orchestrator ops — the reads are local files; the
/// loop briefly hosts them during a poll.
pub(crate) struct PyMediaResolver {
    read: Py<PyAny>,
    exists: Py<PyAny>,
}

impl PyMediaResolver {
    pub(crate) fn new(read: Py<PyAny>, exists: Py<PyAny>) -> Self {
        Self { read, exists }
    }
}

impl ImageResolver for PyMediaResolver {
    fn read(&self, name: &str) -> Option<Vec<u8>> {
        Python::attach(|py| {
            self.read
                .call1(py, (name,))
                .ok()
                .and_then(|v| v.extract::<Option<Vec<u8>>>(py).ok())
                .flatten()
        })
    }

    fn exists(&self, name: &str) -> bool {
        Python::attach(|py| {
            self.exists
                .call1(py, (name,))
                .ok()
                .and_then(|v| v.extract::<bool>(py).ok())
                .unwrap_or(false)
        })
    }
}

/// The Python-facing assembly surface: wraps a harness backend + the running
/// loop into the kernel's embedder seam.
#[pyclass]
pub(crate) struct PyEmbedder {
    pub(crate) handle: Arc<PyEmbedderHandle>,
}

#[pymethods]
impl PyEmbedder {
    /// Capture `backend` (anything with a blocking `embed_texts(list[str])`)
    /// against the RUNNING loop. Call from a coroutine context at assembly.
    #[staticmethod]
    fn capture(py: Python<'_>, backend: Py<PyAny>) -> PyResult<Self> {
        let asyncio = py.import("asyncio")?;
        let event_loop = asyncio.call_method0("get_running_loop")?;
        // Optional metadata for the kernel's drift rules, read once here (the
        // EmbedderBackend protocol's model_fingerprint()/embedding_dim).
        let bound = backend.bind(py);
        let fingerprint = bound
            .call_method0("model_fingerprint")
            .ok()
            .and_then(|v| v.extract::<String>().ok());
        let dim = bound
            .call_method0("embedding_dim")
            .ok()
            .and_then(|v| v.extract::<usize>().ok());
        let embeds_images = bound
            .getattr("modalities")
            .and_then(|m| m.contains("image"))
            .unwrap_or(false);
        Ok(Self {
            handle: Arc::new(PyEmbedderHandle {
                backend,
                event_loop: event_loop.unbind(),
                fingerprint,
                dim,
                embeds_images,
            }),
        })
    }
}

/// Test seam: drive one embed through the kernel's `Embedder` trait and the
/// bridge — proves the full inversion (kernel future → loop dispatch → pool →
/// done-callback → oneshot → bridged await) before the orchestrator's
/// embed-coupled ops consume it.
#[pyfunction]
pub(crate) fn embedder_probe<'py>(
    py: Python<'py>,
    embedder: PyRef<'py, PyEmbedder>,
    texts: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let handle = Arc::clone(&embedder.handle);
    crate::asyncio_bridge::future_into_py(py, async move { handle.embed(texts).await })
}
