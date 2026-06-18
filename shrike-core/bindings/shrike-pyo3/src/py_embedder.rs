//! `PyEmbedder`: the kernel's
//! `Embedder` seam implemented over the *harness's* backend — the inversion
//! that lets the kernel drive ANY Python-held embedder without the kernel
//! knowing Python exists. Every production backend (ONNX, CLIP, llama)
//! bypasses this via `NativeEmbedder`; what rides it is custom/test
//! backends — permanently the escape hatch.
//!
//! Execution shape since the owned runtime: each call moves the blocking
//! Python backend call onto the runtime's blocking pool
//! (`spawn_blocking` + `Python::attach` — a blocking-pool thread may hold
//! the GIL; runtime workers never do). The old loop→`run_in_executor`→
//! oneshot dance is gone.

use std::sync::Arc;

use futures::future::BoxFuture;
use pyo3::prelude::*;

use shrike_engine_api::{Embedder, ImageEmbedder, ImageResolver, MediaItem};
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};

type VecResult = NativeResult<Vec<Vec<f32>>>;

/// The refusal every gated site returns once the interpreter is exiting
/// — the `Unavailable` tier, like any other unreachable backend.
pub(crate) fn shutting_down() -> NativeError {
    NativeError::unavailable("interpreter is exiting; Python backend unreachable")
}

/// What one dispatch embeds: a text batch (`embed_texts`) or an image-bytes
/// batch (`embed_images`) — same blocking-pool shape either way.
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

/// The kernel-facing handle: a Python backend whose blocking calls run on
/// the blocking pool. Fingerprint/dim are captured once at assembly (cheap
/// Python attribute reads) so the kernel's drift rules never call into
/// Python mid-op.
pub(crate) struct PyEmbedderHandle {
    backend: Py<PyAny>,
    fingerprint: Option<String>,
    dim: Option<usize>,
    embeds_images: bool,
}

impl PyEmbedderHandle {
    /// Whether the wrapped backend advertises the image modality.
    pub(crate) fn embeds_images(&self) -> bool {
        self.embeds_images
    }

    /// One blocking backend call on the pool — eager (scheduled before the
    /// returned future is polled), GIL acquired only on the pool thread.
    /// Both attach windows ride the finalization gate: an op still
    /// running while the interpreter exits must not touch Python.
    fn dispatch(&self, payload: EmbedPayload) -> BoxFuture<'static, VecResult> {
        let backend = {
            let Some(_permit) = crate::finalize_gate::permit() else {
                return Box::pin(std::future::ready(Err(shutting_down())));
            };
            Python::attach(|py| self.backend.clone_ref(py))
        };
        let handle = tokio::task::spawn_blocking(move || -> VecResult {
            let Some(_permit) = crate::finalize_gate::permit() else {
                return Err(shutting_down());
            };
            Python::attach(|py| {
                let method = backend.bind(py).getattr(payload.method()).map_err(|e| {
                    NativeError::unavailable(format!("harness embedder missing method: {e}"))
                })?;
                let arg: Py<PyAny> = match payload {
                    EmbedPayload::Texts(texts) => texts
                        .into_pyobject(py)
                        .context(ErrorKind::Internal, "payload convert")?
                        .into_any()
                        .unbind(),
                    EmbedPayload::Images(images) => images
                        .into_pyobject(py)
                        .context(ErrorKind::Internal, "payload convert")?
                        .into_any()
                        .unbind(),
                };
                let value = method.call1((arg,)).map_err(|e| {
                    NativeError::unavailable(format!("harness embedder failed: {}", e.value(py)))
                })?;
                value.extract::<Vec<Vec<f32>>>().map_err(|e| {
                    NativeError::internal(format!("embedder returned a non-vector payload: {e}"))
                })
            })
        });
        Box::pin(async move {
            handle
                .await
                .context(ErrorKind::Internal, "embed task failed")?
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
/// synchronously inside orchestrator ops; the reads are local files.
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
        // Gate-refused ⇒ "unreadable", silently: a log line would just
        // be dropped by the gated pyo3-log wrapper anyway.
        let _permit = crate::finalize_gate::permit()?;
        Python::attach(|py| {
            match self
                .read
                .call1(py, (name,))
                .and_then(|v| v.extract::<Option<Vec<u8>>>(py))
            {
                Ok(bytes) => bytes,
                // A raising/misbehaving resolver degrades to "unreadable"
                // (the kernel skips the item), but never silently.
                Err(e) => {
                    tracing::warn!(image = %name, error = %e, "media resolver read raised");
                    None
                }
            }
        })
    }

    fn exists(&self, name: &str) -> bool {
        // Same refusal shape as `read`: absent.
        let Some(_permit) = crate::finalize_gate::permit() else {
            return false;
        };
        Python::attach(|py| {
            match self
                .exists
                .call1(py, (name,))
                .and_then(|v| v.extract::<bool>(py))
            {
                Ok(present) => present,
                // Same degradation as read: absent, with a signal.
                Err(e) => {
                    tracing::warn!(image = %name, error = %e, "media resolver exists raised");
                    false
                }
            }
        })
    }
}

/// The Python-facing assembly surface: wraps a harness backend into the
/// kernel's embedder seam.
#[pyclass]
pub(crate) struct PyEmbedder {
    pub(crate) handle: Arc<PyEmbedderHandle>,
}

#[pymethods]
impl PyEmbedder {
    /// Capture `backend` (anything with a blocking `embed_texts(list[str])`).
    /// Identity metadata is read once here (the EmbedderBackend protocol's
    /// model_fingerprint()/embedding_dim) so the kernel never calls into
    /// Python mid-op. No loop is captured — execution is the kernel's.
    #[staticmethod]
    fn capture(py: Python<'_>, backend: Py<PyAny>) -> PyResult<Self> {
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
                fingerprint,
                dim,
                embeds_images,
            }),
        })
    }
}

/// Test seam: drive one embed through the kernel's `Embedder` trait — proves
/// the capture → blocking-pool → GIL-attach → completion chain before the
/// orchestrator's embed-coupled ops consume it. Kernel-runtime-bound
/// (`spawn_op`), so anki-core builds only.
#[cfg(feature = "anki-core")]
#[pyfunction]
pub(crate) fn embedder_probe<'py>(
    py: Python<'py>,
    embedder: PyRef<'py, PyEmbedder>,
    texts: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let handle = Arc::clone(&embedder.handle);
    crate::asyncio_bridge::future_into_py(
        py,
        shrike_kernel::spawn_op(async move { handle.embed(texts).await }),
    )
}
