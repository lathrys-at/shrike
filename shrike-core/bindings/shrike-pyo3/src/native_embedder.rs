//! `NativeEmbedder`: the native engines
//! attached to the kernel slot DIRECTLY — no Python on the embed hot path,
//! and since the owned runtime, no execution machinery here at all:
//!
//! ```text
//! OnnxTextEmbedder/ClipEmbedder/RemoteEmbedder (loaded engine, pure compute)
//!   └─ WithPolicy   — host-assembled fingerprint/dim + probed safe_batch
//!       └─ Blocking — the one adapter: eager spawn_blocking on the kernel
//!          runtime's blocking pool (chunk loop inside)
//! ```
//!
//! A kernel embed runs: kernel op (on the kernel runtime) → `Blocking`
//! schedules the Rust chunk loop on the blocking pool → completion. Python
//! schedules nothing and computes nothing. `PyEmbedder.capture` stays as the
//! custom/test-backend escape hatch.

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use pyo3::prelude::*;

use shrike_engine_api::{Embedder, ImageEmbedder};
use shrike_error::NativeResult;
// Used only inside the feature-gated engine constructors — a no-engine build
// (anki-core alone) would otherwise warn on unused imports.
// onnx/CLIP are route-1 (sync compute behind `Blocking` + `WithPolicy`, the
// adapter wired to the kernel's compute pool via `compute_dispatch`); the
// remote engines are route-2 async-direct (`AsyncWithPolicy`, no `Blocking`).
#[cfg(feature = "engine-remote")]
use shrike_engine_api::AsyncWithPolicy;
#[cfg(any(feature = "engine-ort", feature = "engine-synthetic"))]
use shrike_engine_api::WithPolicy;

/// The assembled native embedder the kernel slot takes: the text half always,
/// the image half when the engine embeds images (CLIP). Both halves are views
/// of ONE adapted engine.
#[pyclass(frozen)]
pub(crate) struct NativeEmbedder {
    pub(crate) text: Arc<dyn Embedder>,
    pub(crate) images: Option<Arc<dyn ImageEmbedder>>,
}

struct ObservedText {
    inner: Arc<dyn Embedder>,
}

impl Embedder for ObservedText {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let items = texts.len();
        let started = Instant::now();
        let future = self.inner.embed(texts);
        Box::pin(async move {
            let result = future.await;
            shrike_kernel::record_embedding("text", items, started.elapsed(), result.is_ok());
            result
        })
    }

    fn fingerprint(&self) -> Option<String> {
        self.inner.fingerprint()
    }

    fn dim(&self) -> Option<usize> {
        self.inner.dim()
    }
}

struct ObservedImages {
    inner: Arc<dyn ImageEmbedder>,
}

impl ImageEmbedder for ObservedImages {
    fn embed_images(
        &self,
        images: Vec<shrike_engine_api::MediaItem>,
    ) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        let items = images.len();
        let started = Instant::now();
        let future = self.inner.embed_images(images);
        Box::pin(async move {
            let result = future.await;
            shrike_kernel::record_embedding("image", items, started.elapsed(), result.is_ok());
            result
        })
    }
}

impl NativeEmbedder {
    fn observed(text: Arc<dyn Embedder>, images: Option<Arc<dyn ImageEmbedder>>) -> Self {
        Self {
            text: Arc::new(ObservedText { inner: text }),
            images: images
                .map(|inner| Arc::new(ObservedImages { inner }) as Arc<dyn ImageEmbedder>),
        }
    }
}

#[pymethods]
impl NativeEmbedder {
    /// Compose the ONNX text engine for the kernel slot. `fingerprint` is the
    /// facade-assembled identity (it folds `pool=`/`textprep=` policy the
    /// engine can't know), `dim` what the facade probed, `safe_batch` the
    /// batch-safety probe's verdict capped by the operator's batch-size
    /// setting.
    #[cfg(feature = "engine-ort")]
    #[staticmethod]
    #[pyo3(signature = (engine, *, fingerprint, dim, safe_batch))]
    fn from_onnx(
        engine: PyRef<'_, crate::OnnxTextEmbedder>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        safe_batch: usize,
    ) -> Self {
        let tuned = Arc::new(WithPolicy::new(
            engine.engine_arc(),
            fingerprint,
            dim,
            safe_batch,
        ));
        Self::observed(Arc::new(crate::compute_dispatch::blocking(tuned)), None)
    }

    /// Compose the remote-embeddings engine — llama-server today, any
    /// OpenAI-compatible endpoint tomorrow. A route-2 async-direct engine:
    /// it implements the async `Embedder`/`ImageEmbedder` traits
    /// directly over the async reqwest client, so the kernel awaits it on its
    /// runtime — NO `Blocking` adapter, no parked blocking-pool thread. The
    /// host policy (fingerprint, dim, the proven-safe text `safe_batch` that
    /// chunks the text path) rides `AsyncWithPolicy` — the async sibling of
    /// `WithPolicy` + `Blocking`'s chunk loop.
    ///
    /// `images` composes the image half too: the one remote engine
    /// impls both `Embedder` and `ImageEmbedder`, so a single wrapped instance
    /// serves both modalities — the same shape `from_clip` makes for the dual
    /// ONNX encoder. Set for a `modalities: [text, image]` remote entry against
    /// a llama.cpp multimodal endpoint; left off (the default) for a text-only
    /// endpoint or a cloud API.
    #[cfg(feature = "engine-remote")]
    #[staticmethod]
    #[pyo3(signature = (engine, *, fingerprint, dim, safe_batch, images=false))]
    fn from_remote(
        engine: PyRef<'_, crate::RemoteEmbedder>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        safe_batch: usize,
        images: bool,
    ) -> Self {
        let tuned = Arc::new(AsyncWithPolicy::new(
            engine.engine_arc(),
            fingerprint,
            dim,
            safe_batch,
        ));
        Self::observed(
            Arc::clone(&tuned) as Arc<dyn Embedder>,
            images.then_some(tuned as Arc<dyn ImageEmbedder>),
        )
    }

    /// Compose the CLIP dual encoder: one engine, both modalities — the same
    /// adapted instance serves the text and image halves.
    #[cfg(feature = "engine-ort")]
    #[staticmethod]
    #[pyo3(signature = (engine, *, fingerprint, dim, safe_batch))]
    fn from_clip(
        engine: PyRef<'_, crate::ClipEmbedder>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        safe_batch: usize,
    ) -> Self {
        let tuned = Arc::new(WithPolicy::new(
            engine.engine_arc(),
            fingerprint,
            dim,
            safe_batch,
        ));
        let adapted = Arc::new(crate::compute_dispatch::blocking(tuned));
        Self::observed(
            Arc::clone(&adapted) as Arc<dyn Embedder>,
            Some(adapted as Arc<dyn ImageEmbedder>),
        )
    }

    /// Compose the deterministic synthetic engine (#865) — Route-1 sync, the
    /// same shape as [`NativeEmbedder::from_clip`]: one engine serves both the
    /// text and image halves behind `WithPolicy` + `Blocking`. Gated on
    /// `engine-synthetic`, so a release build can't compose it.
    #[cfg(feature = "engine-synthetic")]
    #[staticmethod]
    #[pyo3(signature = (engine, *, fingerprint, dim, safe_batch))]
    fn from_synthetic(
        engine: PyRef<'_, crate::SyntheticEmbedder>,
        fingerprint: Option<String>,
        dim: Option<usize>,
        safe_batch: usize,
    ) -> Self {
        let tuned = Arc::new(WithPolicy::new(
            engine.engine_arc(),
            fingerprint,
            dim,
            safe_batch,
        ));
        let adapted = Arc::new(crate::compute_dispatch::blocking(tuned));
        Self::observed(
            Arc::clone(&adapted) as Arc<dyn Embedder>,
            Some(adapted as Arc<dyn ImageEmbedder>),
        )
    }
}

/// Test seam: one embed through the full native composition (edge spawn →
/// blocking pool → engine chunk loop → completion), proving the assembly
/// before the kernel's embed-coupled ops ride it. Kernel-runtime-bound
/// (`spawn_op`), so anki-core builds only.
#[cfg(feature = "anki-core")]
#[pyfunction]
pub(crate) fn native_embedder_probe<'py>(
    py: Python<'py>,
    embedder: PyRef<'py, NativeEmbedder>,
    texts: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let text = Arc::clone(&embedder.text);
    crate::asyncio_bridge::future_into_py(
        py,
        shrike_kernel::spawn_op(async move { text.embed(texts).await }),
    )
}
