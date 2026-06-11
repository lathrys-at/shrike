//! Async kernel bindings (#332, S3a): the kernel's runtime-agnostic futures
//! awaited natively on the harness's asyncio loop via the runtime-less bridge.
//!
//! This first slice binds the kernel's `SerializedCollection` — open / a
//! collection op / close, each an `asyncio.Future` — proving the full chain:
//! kernel future → injected executor → loop-driven polls → Python `await`.
//! Later S3 slices widen this to the kernel's orchestration (index drift,
//! saver debounce, runtime lifecycle) and swap the inline `MutexExecutor`
//! for a harness-injected worker executor.

use std::sync::Arc;

use pyo3::prelude::*;

use shrike_collection::{CreateOutcome, DuplicatePolicy};
use shrike_kernel::{Kernel, MutexExecutor, NoteSpec, SerialExecutor, SerializedCollection};

use crate::asyncio_bridge::future_into_py;
use crate::py_embedder::{PyEmbedder, PyEmbedderHandle, PyMediaResolver};
use crate::timer_host::LoopTimerHost;
use crate::worker_executor::WorkerExecutor;

/// An open collection whose every op is an awaitable serialized through the
/// kernel's injected executor.
#[pyclass]
pub(crate) struct AsyncCollection {
    inner: Arc<SerializedCollection>,
}

/// Open a collection asynchronously; resolves to an [`AsyncCollection`].
///
/// `executor` is the harness-injected scheduler (#332 S3b): pass a
/// [`WorkerExecutor`] whose `worker_loop` runs on a harness-owned thread and
/// collection jobs leave the asyncio loop entirely. Without one, the kernel's
/// inline `MutexExecutor` runs jobs inside the poll callback (conforming —
/// fine for tests and tiny embedded uses).
#[pyfunction]
#[pyo3(signature = (collection_path, executor=None))]
pub(crate) fn async_collection_open<'py>(
    py: Python<'py>,
    collection_path: String,
    executor: Option<PyRef<'py, WorkerExecutor>>,
) -> PyResult<Bound<'py, PyAny>> {
    let executor: Arc<dyn SerialExecutor> = match executor {
        Some(ex) => Arc::new(ex.handle()),
        None => Arc::new(MutexExecutor::default()),
    };
    future_into_py(py, async move {
        let collection = SerializedCollection::open(collection_path, executor).await?;
        Ok(AsyncCollection {
            inner: Arc::new(collection),
        })
    })
}

#[pymethods]
impl AsyncCollection {
    /// The collection's modification stamp (an awaitable).
    fn col_mod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move { inner.run(|core| core.col_mod()).await? })
    }

    /// Note ids matching a raw Anki search (an awaitable).
    fn find_notes<'py>(&self, py: Python<'py>, query: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move {
            inner.run(move |core| core.find_notes(&query)).await?
        })
    }

    /// Close the collection (an awaitable).
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move { inner.close().await })
    }
}

/// One per-item upsert outcome on the wire: `("created", id)`,
/// `("skipped", None)`, or `("error", message-as-id-None…)` — encoded as
/// `(status, id, error)` so Python pattern-matches without a union type.
type UpsertWireResult = (String, Option<i64>, Option<String>);

fn outcome_to_wire(outcome: shrike_ffi::NativeResult<CreateOutcome>) -> UpsertWireResult {
    match outcome {
        Ok(CreateOutcome::Created(id)) => ("created".to_string(), Some(id), None),
        Ok(CreateOutcome::SkippedDuplicate) => ("skipped".to_string(), None, None),
        Err(e) => ("error".to_string(), None, Some(e.to_string())),
    }
}

/// The full kernel bound for the harness (#332, S3d-1b): one open collection
/// + the kernel-internal index orchestration + the derived store, every op an
/// awaitable on the running loop. The harness assembles it from its own parts
/// — a [`WorkerExecutor`] (or none), a [`PyEmbedder`] over its backend, and
/// the loop's timers for the debounced index flush — then shares the kernel's
/// engine/core handles for its read/search surfaces.
#[pyclass(frozen)]
pub(crate) struct AsyncKernel {
    inner: Arc<Kernel<Arc<PyEmbedderHandle>>>,
}

/// Open a kernel asynchronously; resolves to an [`AsyncKernel`]. Call from a
/// coroutine context (the loop hosts the timers and the embed dispatches).
#[pyfunction]
#[pyo3(signature = (collection_path, cache_dir, embedder, executor=None, media_read=None, media_exists=None))]
pub(crate) fn async_kernel_open<'py>(
    py: Python<'py>,
    collection_path: String,
    cache_dir: String,
    embedder: PyRef<'py, PyEmbedder>,
    executor: Option<PyRef<'py, WorkerExecutor>>,
    media_read: Option<Py<PyAny>>,
    media_exists: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let executor: Arc<dyn SerialExecutor> = match executor {
        Some(ex) => Arc::new(ex.handle()),
        None => Arc::new(MutexExecutor::default()),
    };
    let timers: Arc<dyn shrike_kernel::TimerHost> = Arc::new(LoopTimerHost::capture_host(py)?);
    let handle = Arc::clone(&embedder.handle);
    // The image pair exists only when the backend embeds images AND the
    // harness supplied BOTH resolver callables (read + the cheap stat).
    let images: Option<shrike_kernel::KernelImages> =
        match (handle.embeds_images(), media_read, media_exists) {
            (true, Some(read), Some(exists)) => Some((
                Box::new(Arc::clone(&handle)),
                Box::new(PyMediaResolver::new(read, exists)),
            )),
            _ => None,
        };
    future_into_py(py, async move {
        let kernel = Kernel::open(
            &collection_path,
            &cache_dir,
            handle,
            executor,
            Some(timers),
            images,
        )
        .await?;
        Ok(AsyncKernel {
            inner: Arc::new(kernel),
        })
    })
}

#[pymethods]
impl AsyncKernel {
    /// Create a batch of notes (the #77 duplicate policy per item) and index
    /// them — ONE collection job, ONE read job, batched embeds (an awaitable;
    /// per-item results, one bad note never sinks the batch).
    fn upsert_notes<'py>(
        &self,
        py: Python<'py>,
        notes: Vec<(i64, i64, Vec<String>, Vec<String>)>,
        on_duplicate: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let policy = DuplicatePolicy::parse(on_duplicate).map_err(crate::to_py_err)?;
        let specs: Vec<NoteSpec> = notes
            .into_iter()
            .map(|(notetype_id, deck_id, fields, tags)| NoteSpec {
                notetype_id,
                deck_id,
                fields,
                tags,
            })
            .collect();
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move {
            let outcomes = kernel.upsert_notes(specs, policy).await?;
            Ok(outcomes
                .into_iter()
                .map(outcome_to_wire)
                .collect::<Vec<_>>())
        })
    }

    /// Delete notes; vectors, fingerprints, and derived rows go with them.
    fn delete_notes<'py>(
        &self,
        py: Python<'py>,
        note_ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move { kernel.delete_notes(note_ids).await })
    }

    /// Fused search: `(note_id, score, [(signal, rank)])` rows.
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move {
            let hits = kernel.search(&query, top_k).await?;
            Ok(hits
                .into_iter()
                .map(|h| (h.note_id, h.score, h.signals))
                .collect::<Vec<_>>())
        })
    }

    /// The boot/reload drift path (awaitable; drive as a background task).
    fn reindex_if_needed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move { kernel.reindex_if_needed().await })
    }

    fn col_mod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move { kernel.col_mod().await })
    }

    /// The index status block as JSON (state/size/progress/stamps).
    fn index_status_json(&self) -> String {
        self.inner.index().status().to_string()
    }

    /// Flush the index + sidecars now (shutdown path).
    fn save_index(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.index().save())
            .map_err(crate::to_py_err)
    }

    /// A search handle over the kernel's OWN engine (`Arc`-shared): what the
    /// harness's search/action paths read — always the vectors the kernel
    /// maintains.
    fn engine_handle(&self) -> crate::NativeIndexEngine {
        crate::NativeIndexEngine {
            inner: self.inner.index().engine_arc(),
        }
    }

    /// A handle over the kernel's OWN collection core (`Arc`-shared), for the
    /// harness's direct ops — which must honor the same executor discipline
    /// the kernel's jobs run under.
    fn core_handle(&self) -> crate::anki_core::CollectionCore {
        crate::anki_core::CollectionCore::from_arc(self.inner.collection().core_arc())
    }

    /// Run a harness callable as ONE serialized job on the kernel's executor
    /// — the escape hatch carrying the long tail of direct collection ops
    /// (media, prune, note-type edits) without binding each verb: the
    /// callable closes over `core_handle()` and runs where every other
    /// collection job runs (GIL attached for its duration). A Python
    /// exception rethrows as-is through the awaitable. Re-entrancy rule: the
    /// job must never await another kernel op (a deadlock by contract).
    fn run_job<'py>(&self, py: Python<'py>, job: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        crate::asyncio_bridge::pyresult_future_into_py(py, async move {
            kernel
                .collection()
                .run(move |_core| Python::attach(|py| job.call0(py)))
                .await
                .map_err(crate::to_py_err)?
        })
    }

    /// Cooperative idle-release (#64) — awaitable.
    fn release<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move { kernel.release().await })
    }

    /// Re-acquire after a release — awaitable (busy surfaces as the typed
    /// BUSY error tier).
    fn reopen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move { kernel.reopen().await })
    }

    /// Flush the index, then close the collection — awaitable.
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        future_into_py(py, async move {
            let _ = kernel.index().save();
            kernel.collection().close().await
        })
    }
}
