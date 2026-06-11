//! `KernelIndex` (#332, S3c-4): the kernel's `IndexOrchestrator` bound for the
//! Python harness — the orchestration half of `VectorIndex` (drift rules,
//! per-note fingerprints, sidecar persistence, the build state machine,
//! reconcile/rebuild) re-homed in Rust, while the harness keeps its
//! `NativeIndexEngine` handle for the search surface. The two share ONE engine
//! (`Arc<MultiModalIndex>`): the orchestrator maintains the vectors the
//! harness searches.
//!
//! The embed-coupled ops (`add`/`rebuild`/`reconcile`) are kernel async fns
//! driven by the asyncio bridge: the loop polls them; each embed hop goes
//! kernel future → `PyEmbedder` dispatch → thread-pool `embed_texts` →
//! oneshot — so a rebuild runs as a plain asyncio task with no thread owned
//! by anyone. Image inputs ride the `EmbedInput` shape already; the image
//! embed path (resolver + image embedder injection) lands with the facade
//! rewire (S3c-4b).
//!
//! `KernelIndexSaver` binds the kernel's `DebouncedSaver` over the harness's
//! `LoopTimerHost` — the `IndexSaver` replacement (no timer thread; the loop
//! is the timer).

use std::sync::Arc;

use pyo3::prelude::*;

use shrike_kernel::index_orchestrator::{
    DebouncedSaver, EmbedInput, IndexOrchestrator, OrchestratorState,
};
use shrike_kernel::TimerHost;

use crate::asyncio_bridge::future_into_py;
use crate::py_embedder::PyEmbedder;
use crate::timer_host::LoopTimerHost;
use crate::to_py_err;

/// `(note_id, text, image_names)` — the wire shape of one embed input.
type EmbedInputTuple = (i64, String, Vec<String>);

fn to_embed_inputs(inputs: Vec<EmbedInputTuple>) -> Vec<EmbedInput> {
    inputs
        .into_iter()
        .map(|(note_id, text, image_names)| EmbedInput {
            note_id,
            text,
            image_names,
        })
        .collect()
}

/// The kernel's index orchestrator, sharing its engine with the harness's
/// `NativeIndexEngine` search handle.
#[pyclass(frozen)]
pub(crate) struct KernelIndex {
    pub(crate) inner: Arc<IndexOrchestrator>,
}

#[pymethods]
impl KernelIndex {
    /// Open over `dir`, loading any on-disk index + sidecars through the
    /// SAME engine the given search handle wraps.
    #[new]
    fn new(py: Python<'_>, dir: String, engine: PyRef<'_, crate::NativeIndexEngine>) -> Self {
        let shared = Arc::clone(&engine.inner);
        drop(engine);
        let inner = py.detach(move || Arc::new(IndexOrchestrator::open(dir, shared)));
        Self { inner }
    }

    fn state(&self) -> &'static str {
        match self.inner.state() {
            OrchestratorState::Ready => "ready",
            OrchestratorState::Building => "building",
            OrchestratorState::Unavailable => "unavailable",
            OrchestratorState::Error => "error",
        }
    }

    fn build_progress(&self) -> (u64, u64) {
        self.inner.build_progress()
    }

    fn col_mod(&self) -> Option<i64> {
        self.inner.col_mod()
    }

    fn set_col_mod(&self, value: i64) {
        self.inner.set_col_mod(value)
    }

    fn model_id(&self) -> Option<String> {
        self.inner.model_id()
    }

    fn has_note_hashes(&self) -> bool {
        self.inner.has_note_hashes()
    }

    #[pyo3(signature = (current_col_mod, current_model_id=None, embeds_images=false))]
    fn check_drift(
        &self,
        current_col_mod: i64,
        current_model_id: Option<String>,
        embeds_images: bool,
    ) -> bool {
        self.inner
            .check_drift(current_col_mod, current_model_id.as_deref(), embeds_images)
    }

    fn remove(&self, py: Python<'_>, note_ids: Vec<i64>) -> PyResult<usize> {
        py.detach(|| self.inner.remove(&note_ids))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (ndim, col_mod, model_id=None))]
    fn materialize_empty(
        &self,
        py: Python<'_>,
        ndim: usize,
        col_mod: i64,
        model_id: Option<String>,
    ) {
        py.detach(|| {
            self.inner
                .materialize_empty(ndim, col_mod, model_id.as_deref())
        })
    }

    fn save(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.save()).map_err(to_py_err)
    }

    fn calibrate_activation(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.calibrate_activation())
            .map_err(to_py_err)
    }

    /// The status block as JSON (state, size, ndim, stamps, progress,
    /// activation) — the harness's `/status` source.
    fn status_json(&self) -> String {
        self.inner.status().to_string()
    }

    /// Embed + (replace-)add notes — an awaitable on the running loop.
    fn add<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<EmbedInputTuple>,
        embedder: PyRef<'py, PyEmbedder>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let orch = Arc::clone(&self.inner);
        let handle = Arc::clone(&embedder.handle);
        let inputs = to_embed_inputs(inputs);
        future_into_py(py, async move { orch.add(&inputs, &*handle, None).await })
    }

    /// Full rebuild as an awaitable: the loop drives it; `state()`/
    /// `build_progress()` report while it runs.
    #[pyo3(signature = (inputs, col_mod, model_id, embedder))]
    fn rebuild<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<EmbedInputTuple>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: PyRef<'py, PyEmbedder>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let orch = Arc::clone(&self.inner);
        let handle = Arc::clone(&embedder.handle);
        let inputs = to_embed_inputs(inputs);
        future_into_py(py, async move {
            orch.rebuild(inputs, col_mod, model_id, &*handle, None)
                .await
        })
    }

    /// Incremental reconcile (falls back to a full rebuild without prior
    /// per-note state) as an awaitable.
    #[pyo3(signature = (inputs, col_mod, model_id, embedder))]
    fn reconcile<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<EmbedInputTuple>,
        col_mod: i64,
        model_id: Option<String>,
        embedder: PyRef<'py, PyEmbedder>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let orch = Arc::clone(&self.inner);
        let handle = Arc::clone(&embedder.handle);
        let inputs = to_embed_inputs(inputs);
        future_into_py(py, async move {
            orch.reconcile(inputs, col_mod, model_id, &*handle, None)
                .await
        })
    }
}

/// The kernel's debounced index saver over the harness's loop timers — the
/// `IndexSaver` replacement (idle debounce + burst cap, no timer thread).
#[pyclass(frozen)]
pub(crate) struct KernelIndexSaver {
    inner: Arc<DebouncedSaver>,
}

#[pymethods]
impl KernelIndexSaver {
    #[new]
    fn new(
        py: Python<'_>,
        index: PyRef<'_, KernelIndex>,
        timers: PyRef<'_, LoopTimerHost>,
        delay: f64,
        threshold: u64,
    ) -> Self {
        let host: Arc<dyn TimerHost> = Arc::new(timers.sibling(py));
        Self {
            inner: DebouncedSaver::new(Arc::clone(&index.inner), host, delay, threshold),
        }
    }

    /// Record one change: re-arm the idle timer, or flush at the burst cap.
    fn request_save(&self) {
        self.inner.request_save()
    }

    fn pending_changes(&self) -> u64 {
        self.inner.pending_changes()
    }

    fn flush(&self, py: Python<'_>) {
        py.detach(|| self.inner.flush())
    }
}
