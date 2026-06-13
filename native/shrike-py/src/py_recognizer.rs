//! `PyRecognizer` (#228): the kernel's `Recognizer` seam implemented over the
//! harness's backend — the same inversion as `PyEmbedder` (the kernel drives
//! ANY Python-held recognizer without knowing Python exists). Since #342 P3
//! Apple Vision is native (`AppleVisionRecognizer` below, attached direct);
//! what rides this capture seam is custom/test backends. Since #374 C the
//! blocking backend call rides the kernel runtime's blocking pool
//! (`spawn_blocking` + `Python::attach`) — no loop machinery.
//!
//! Wire contract with the Python backend: `recognize(items: list[bytes]) ->
//! list[tuple[str, float, str]]` — `(text, confidence, segments_json)`, where
//! `segments_json` is a JSON array of `{text, confidence, bbox?}` (or `""`
//! for none). Tuples keep the boundary explicit and dependency-free; the
//! Python protocol layer wraps them in dataclasses.

use std::sync::Arc;

use futures::future::BoxFuture;
use pyo3::prelude::*;

use shrike_engine_api::{MediaItem, Recognition, Recognizer, Segment};
use shrike_ffi::{NativeError, NativeResult};

type RecResult = NativeResult<Vec<Recognition>>;

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

/// The kernel-facing handle: a Python backend whose blocking calls ride the
/// kernel runtime's blocking pool (#374 — no captured loop). The fingerprint
/// is captured once at assembly so the kernel's invalidation rules never
/// call into Python mid-op.
pub(crate) struct PyRecognizerHandle {
    backend: Py<PyAny>,
    fingerprint: Option<String>,
}

impl Recognizer for PyRecognizerHandle {
    fn recognize(&self, items: Vec<MediaItem>) -> BoxFuture<'_, RecResult> {
        // The Python wire stays `list[bytes]` (the RecognizerBackend
        // contract); native engines are where the mime hint pays off. The
        // blocking backend call rides the runtime's blocking pool (#374 C).
        let items: Vec<Vec<u8>> = items.into_iter().map(|m| m.bytes).collect();
        // Both attach windows ride the finalization gate (#435), exactly like
        // PyEmbedderHandle::dispatch.
        let backend = {
            let Some(_permit) = crate::finalize_gate::permit() else {
                return Box::pin(std::future::ready(Err(crate::py_embedder::shutting_down())));
            };
            Python::attach(|py| self.backend.clone_ref(py))
        };
        let handle = tokio::task::spawn_blocking(move || -> RecResult {
            let Some(_permit) = crate::finalize_gate::permit() else {
                return Err(crate::py_embedder::shutting_down());
            };
            Python::attach(|py| {
                let raw = backend
                    .bind(py)
                    .call_method1("recognize", (items,))
                    .map_err(|e| {
                        NativeError::unavailable(format!(
                            "harness recognizer failed: {}",
                            e.value(py)
                        ))
                    })?
                    .extract::<Vec<(String, f64, String)>>()
                    .map_err(|e| {
                        NativeError::internal(format!(
                            "recognizer returned a non-(text, confidence, segments_json) payload: {e}"
                        ))
                    })?;
                parse_results(raw)
            })
        });
        Box::pin(async move {
            handle
                .await
                .map_err(|e| NativeError::internal(format!("recognize task failed: {e}")))?
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }
}

/// The native Apple Vision engine (#342 P3) as the Python-visible backend
/// object: construction fails `unavailable` off macOS (the same
/// degrade-don't-crash a missing pyobjc gave the retired Python backend),
/// `recognize` is the blocking wire shape for direct callers/tests, and the
/// kernel attach path takes `engine_arc()` — recognition then runs native
/// end-to-end (engine → Blocking → the runtime's blocking pool), no Python
/// on the sweep.
#[cfg(feature = "engine-apple")]
#[pyclass(frozen)]
pub(crate) struct AppleVisionRecognizer {
    engine: Arc<shrike_recognize_apple::AppleVisionRecognizer>,
}

#[cfg(feature = "engine-apple")]
impl AppleVisionRecognizer {
    pub(crate) fn engine_arc(&self) -> Arc<shrike_recognize_apple::AppleVisionRecognizer> {
        Arc::clone(&self.engine)
    }
}

#[cfg(feature = "engine-apple")]
#[pymethods]
impl AppleVisionRecognizer {
    #[new]
    fn new() -> PyResult<Self> {
        let engine =
            shrike_recognize_apple::AppleVisionRecognizer::new().map_err(crate::to_py_err)?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    /// The platform identity: `apple-vision-swift:{revision}:macos{X.Y.Z}`
    /// — the hard-cut Swift-glue lineage (#398); a changed fingerprint
    /// re-derives OCR rows exactly like a model change rebuilds vectors.
    fn model_fingerprint(&self) -> Option<String> {
        Some(self.engine.fingerprint_str().to_string())
    }

    /// Blocking direct OCR in the RecognizerBackend wire shape
    /// (`(text, confidence, segments_json)`, `""` for no segments) — for
    /// tests and direct callers; the kernel path never comes through here.
    fn recognize(&self, py: Python<'_>, items: Vec<Vec<u8>>) -> Vec<(String, f64, String)> {
        py.detach(|| {
            items
                .iter()
                .map(|bytes| {
                    let r = self.engine.recognize_one(bytes);
                    let segments_json = if r.segments.is_empty() {
                        String::new()
                    } else {
                        serde_json::to_string(&r.segments).unwrap_or_default()
                    };
                    (r.text, r.confidence, segments_json)
                })
                .collect()
        })
    }
}

/// The remote VLM describe engine (#433/#485) as the Python-visible backend
/// object: the recognizer-attach sibling of `RemoteEmbedder`. Construction
/// validates the API key; the kernel attach path takes `engine_arc()` and
/// adapts it onto the blocking pool via `Blocking` exactly like Apple Vision,
/// so describe runs native end-to-end (engine → Blocking → the runtime's
/// blocking pool), no Python on the sweep. `recognize` is the blocking wire
/// shape for direct callers/tests; `health_ok`/`model_info` serve the host
/// the raw ingredients it folds into the describe fingerprint
/// (`compose_fingerprint` is host policy, mirrored in the crate).
#[cfg(feature = "engine-remote")]
#[pyclass(frozen)]
pub(crate) struct RemoteDescriber {
    engine: Arc<shrike_describe_remote::RemoteDescriber>,
}

#[cfg(feature = "engine-remote")]
impl RemoteDescriber {
    pub(crate) fn engine_arc(&self) -> Arc<shrike_describe_remote::RemoteDescriber> {
        Arc::clone(&self.engine)
    }
}

#[cfg(feature = "engine-remote")]
#[pymethods]
impl RemoteDescriber {
    #[new]
    #[pyo3(signature = (base_url, *, api_key=None, model=None))]
    fn new(base_url: String, api_key: Option<String>, model: Option<String>) -> PyResult<Self> {
        // Construction validates the API key (header-injection guard) — the
        // same discipline as RemoteEmbedder.
        let engine = shrike_describe_remote::RemoteDescriber::new(
            shrike_describe_remote::RemoteDescriberConfig {
                base_url,
                api_key,
                model,
                ..Default::default()
            },
        )
        .map_err(crate::to_py_err)?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    /// Blocking direct describe in the RecognizerBackend wire shape
    /// (`(text, confidence, segments_json)`; describe locates nothing so the
    /// segments JSON is always `""`) — for tests and direct callers; the
    /// kernel path never comes through here. A chunk-level failure (a down
    /// endpoint) raises; a per-item failure degrades to an empty recognition
    /// (the crate's settled error split).
    fn recognize(
        &self,
        py: Python<'_>,
        items: Vec<Vec<u8>>,
    ) -> PyResult<Vec<(String, f64, String)>> {
        use shrike_engine_api::RecognizeMedia as _;
        py.detach(|| {
            let media: Vec<MediaItem> = items.into_iter().map(MediaItem::untyped).collect();
            let recognitions = self.engine.recognize_chunk(&media)?;
            Ok(recognitions
                .into_iter()
                .map(|r| {
                    let segments_json = if r.segments.is_empty() {
                        String::new()
                    } else {
                        serde_json::to_string(&r.segments).unwrap_or_default()
                    };
                    (r.text, r.confidence, segments_json)
                })
                .collect())
        })
        .map_err(crate::to_py_err)
    }

    /// `GET /health` returns 200 — the host's connectivity probe at attach.
    fn health_ok(&self, py: Python<'_>) -> bool {
        py.detach(|| self.engine.health_ok())
    }

    /// `(model_id, meta_json)` from `/v1/models` — `(None, "{}")` when the
    /// endpoint doesn't serve it; fingerprint assembly stays host policy
    /// (`compose_fingerprint`).
    fn model_info(&self, py: Python<'_>) -> (Option<String>, String) {
        py.detach(|| {
            let info = self.engine.model_info();
            let meta = serde_json::Value::Object(info.meta).to_string();
            (info.id, meta)
        })
    }
}

/// The Python-visible wrapper the harness constructs at assembly:
/// `Recognizer.capture(backend)`.
#[pyclass(name = "Recognizer")]
pub struct PyRecognizer {
    pub(crate) handle: Arc<PyRecognizerHandle>,
}

#[pymethods]
impl PyRecognizer {
    /// Capture the harness's recognizer backend. The fingerprint
    /// (`model_fingerprint()`, optional) is read once here so the kernel's
    /// invalidation rules never call into Python mid-op. No loop is
    /// captured — execution is the kernel's (#374).
    #[staticmethod]
    fn capture(py: Python<'_>, backend: Py<PyAny>) -> PyResult<Self> {
        let fingerprint = backend
            .bind(py)
            .call_method0("model_fingerprint")
            .ok()
            .and_then(|v| v.extract::<String>().ok());
        Ok(Self {
            handle: Arc::new(PyRecognizerHandle {
                backend,
                fingerprint,
            }),
        })
    }
}
