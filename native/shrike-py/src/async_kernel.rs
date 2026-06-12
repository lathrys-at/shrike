//! Async kernel bindings (#332, S3a; reshaped by #374): every op is spawned
//! onto the kernel's owned runtime at this edge (`spawn_op`) and surfaces as
//! an `asyncio.Future` through the one-wake completion bridge.

use std::future::Future;
use std::sync::Arc;

use pyo3::prelude::*;

use shrike_collection::{CreateOutcome, DuplicatePolicy};
use shrike_ffi::NativeResult;
use shrike_kernel::{Kernel, NoteSpec, SerializedCollection};

use crate::asyncio_bridge::future_into_py;
use crate::native_embedder::NativeEmbedder;
use crate::py_embedder::{PyEmbedder, PyEmbedderHandle, PyMediaResolver};

/// THE op edge (#397): spawn a kernel future onto the owned runtime
/// (`spawn_op` — dropping the result detaches observation, never aborts) and
/// bridge its completion to an `asyncio.Future`. Every awaitable below routes
/// through here, so the spawn+bridge composition is audited in exactly one
/// place. (A `macro_rules!` forwarder generator was considered and rejected:
/// `#[pymethods]` doesn't expand macro items inside its block, and a second
/// block needs pyo3's `multiple-pymethods` feature — the helper gets the
/// single-definition property without either.)
fn kernel_op<'py, T, F>(py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = NativeResult<T>> + Send + 'static,
    T: for<'p> IntoPyObject<'p> + Send + 'static,
{
    future_into_py(py, shrike_kernel::spawn_op(fut))
}

/// An open collection whose every op is an awaitable serialized through the
/// kernel's collection actor.
#[pyclass]
pub(crate) struct AsyncCollection {
    inner: Arc<SerializedCollection>,
}

/// Open a collection asynchronously; resolves to an [`AsyncCollection`].
/// Scheduling is the kernel's own (#374): the collection actor spawns onto
/// the owned runtime; this host just awaits completions.
#[pyfunction]
pub(crate) fn async_collection_open<'py>(
    py: Python<'py>,
    collection_path: String,
) -> PyResult<Bound<'py, PyAny>> {
    kernel_op(py, async move {
        let collection = SerializedCollection::open(collection_path).await?;
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
        kernel_op(py, async move { inner.run(|core| core.col_mod()).await? })
    }

    /// Note ids matching a raw Anki search (an awaitable).
    fn find_notes<'py>(&self, py: Python<'py>, query: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        kernel_op(py, async move {
            inner.run(move |core| core.find_notes(&query)).await?
        })
    }

    /// Close the collection (an awaitable).
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        kernel_op(py, async move { inner.close().await })
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

/// The full kernel bound for the harness (#332, S3d-1b; #374): one open
/// collection + the kernel-internal index orchestration + the derived store,
/// every op spawned onto the kernel's own runtime at this edge and awaited
/// as an asyncio future. The harness attaches services (engines via
/// [`NativeEmbedder`]; custom backends via [`PyEmbedder`]) and shares the
/// kernel's engine/core handles for its read/search surfaces.
#[pyclass(frozen)]
pub(crate) struct AsyncKernel {
    inner: Arc<Kernel>,
}

impl AsyncKernel {
    /// The wrapped kernel, for sibling bindings composing over the live
    /// handle (the search action reads its tag state).
    pub(crate) fn kernel_arc(&self) -> Arc<Kernel> {
        Arc::clone(&self.inner)
    }
}

/// Either embedder shape at the attach seam (#342 P2): the native composition
/// (engines direct to the kernel slot, no Python on the embed path) or the
/// captured-Python-backend handle (llama until P4; the test seam + custom
/// backends forever).
#[derive(FromPyObject)]
enum AnyEmbedder<'py> {
    Native(PyRef<'py, NativeEmbedder>),
    Captured(PyRef<'py, PyEmbedder>),
}

/// Either recognizer shape (#342 P3) — the same split as [`AnyEmbedder`]:
/// the native Vision engine (adapted onto the blocking pool at attach) or a
/// captured Python backend (custom/test recognizers).
#[derive(FromPyObject)]
enum AnyRecognizer<'py> {
    Native(PyRef<'py, crate::py_recognizer::AppleVisionRecognizer>),
    Captured(PyRef<'py, crate::py_recognizer::PyRecognizer>),
}

/// Build the kernel's image pair from a captured embedder + the resolver
/// callables: present only when the backend embeds images AND the harness
/// supplied BOTH callables (read + the cheap stat).
fn image_pair(
    handle: &Arc<PyEmbedderHandle>,
    media_read: Option<Py<PyAny>>,
    media_exists: Option<Py<PyAny>>,
) -> Option<shrike_kernel::KernelImages> {
    match (handle.embeds_images(), media_read, media_exists) {
        (true, Some(read), Some(exists)) => Some((
            Box::new(Arc::clone(handle)),
            Box::new(PyMediaResolver::new(read, exists)),
        )),
        _ => None,
    }
}

/// Open a kernel asynchronously; resolves to an [`AsyncKernel`]. Call from a
/// coroutine context (the completion bridge resolves on the running loop).
/// The embedding service attaches separately (`attach_embedder`) — the
/// embedder slot is runtime-swappable (#342), so a kernel opens (and serves
/// lexical search + every collection op) with none.
#[pyfunction]
pub(crate) fn async_kernel_open<'py>(
    py: Python<'py>,
    collection_path: String,
    cache_dir: String,
) -> PyResult<Bound<'py, PyAny>> {
    kernel_op(py, async move {
        let kernel = Kernel::open(&collection_path, &cache_dir).await?;
        Ok(AsyncKernel {
            inner: Arc::new(kernel),
        })
    })
}

#[pymethods]
impl AsyncKernel {
    /// Attach (or swap) the embedding service — embedding start / model
    /// change. Takes either embedder shape ([`AnyEmbedder`]): the native
    /// composition embeds without re-entering Python; the captured handle
    /// dispatches to the Python backend. Follow up with `reindex_if_needed`
    /// (a model change is drift).
    #[pyo3(signature = (embedder, media_read=None, media_exists=None))]
    fn attach_embedder(
        &self,
        embedder: AnyEmbedder<'_>,
        media_read: Option<Py<PyAny>>,
        media_exists: Option<Py<PyAny>>,
    ) {
        match embedder {
            AnyEmbedder::Native(native) => {
                let images = match (&native.images, media_read, media_exists) {
                    (Some(img), Some(read), Some(exists)) => Some((
                        Box::new(Arc::clone(img)) as Box<dyn shrike_kernel::ImageEmbedder>,
                        Box::new(PyMediaResolver::new(read, exists))
                            as Box<dyn shrike_kernel::ImageResolver>,
                    )),
                    _ => None,
                };
                self.inner.attach_embedder(Arc::clone(&native.text), images);
            }
            AnyEmbedder::Captured(captured) => {
                let handle = Arc::clone(&captured.handle);
                let images = image_pair(&handle, media_read, media_exists);
                self.inner.attach_embedder(handle, images);
            }
        }
    }

    /// Detach the embedding service (embedding stop): the index flushes and
    /// reports unavailable; the collection and lexical search stay live.
    fn detach_embedder(&self, py: Python<'_>) {
        py.detach(|| self.inner.detach_embedder())
    }

    /// Attach the recognition service (#228, the second #342 slot): an OCR/ASR
    /// engine plus the media-resolver callables it reads bytes through
    /// (independent of the embed slot — OCR works with a text-only embedder).
    /// Takes either recognizer shape ([`AnyRecognizer`]); the native engine
    /// is adapted onto the asyncio compute lane here (call from a coroutine).
    fn attach_recognizer(
        &self,
        recognizer: AnyRecognizer<'_>,
        media_read: Py<PyAny>,
        media_exists: Py<PyAny>,
    ) {
        let resolver: Arc<dyn shrike_kernel::ImageResolver> =
            Arc::new(PyMediaResolver::new(media_read, media_exists));
        match recognizer {
            AnyRecognizer::Native(native) => {
                let adapted: Arc<dyn shrike_kernel::Recognizer> =
                    Arc::new(shrike_engine_api::Blocking(native.engine_arc()));
                self.inner.attach_recognizer(adapted, resolver);
            }
            AnyRecognizer::Captured(captured) => {
                let handle: Arc<dyn shrike_kernel::Recognizer> = Arc::clone(&captured.handle) as _;
                self.inner.attach_recognizer(handle, resolver);
            }
        }
    }

    /// Detach the recognition service: derived text stays (still valid output
    /// of the engine that produced it); only new recognition stops.
    fn detach_recognizer(&self, py: Python<'_>) {
        py.detach(|| self.inner.detach_recognizer())
    }

    /// One bounded recognition sweep (#228): recognize up to `max_items`
    /// pending images, persist gated text + segments, re-embed the affected
    /// notes. Returns a JSON report ({status, recognized, stored, remaining});
    /// the harness loops in the background while `remaining > 0`.
    fn recognize_pending<'py>(
        &self,
        py: Python<'py>,
        max_items: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let report = inner.recognize_pending(max_items).await?;
            Ok(report.to_string())
        })
    }

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
        kernel_op(py, async move {
            let outcomes = kernel.upsert_notes(specs, policy).await?;
            Ok(outcomes
                .into_iter()
                .map(outcome_to_wire)
                .collect::<Vec<_>>())
        })
    }

    /// The wire-shaped bulk upsert (named fields, create AND update,
    /// dry_run): per-item results JSON in the action's existing vocabulary,
    /// with kernel-internal index/derived maintenance over everything
    /// written — the op the MCP upsert_notes action rides.
    fn upsert_notes_json<'py>(
        &self,
        py: Python<'py>,
        notes_json: String,
        on_duplicate: String,
        dry_run: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            kernel
                .upsert_notes_json(notes_json, on_duplicate, dry_run)
                .await
        })
    }

    /// Drop already-deleted notes from the index + derived store (the prune
    /// path) — awaitable.
    fn forget_notes<'py>(
        &self,
        py: Python<'py>,
        note_ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.forget_notes(note_ids).await })
    }

    /// Advance the watermarks after a metadata-only change (tags/decks/
    /// templates) — no re-embed, no drift on next boot. Awaitable.
    fn metadata_changed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.metadata_changed().await })
    }

    /// Delete notes; vectors, fingerprints, and derived rows go with them.
    fn delete_notes<'py>(
        &self,
        py: Python<'py>,
        note_ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.delete_notes(note_ids).await })
    }

    /// Fused search: `(note_id, score, [(signal, rank)])` rows.
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let hits = kernel.search(&query, top_k).await?;
            Ok(hits
                .into_iter()
                .map(|h| (h.note_id, h.score, h.signals))
                .collect::<Vec<_>>())
        })
    }

    /// Explicit FULL rebuild (the `/index/rebuild` semantics) — awaitable;
    /// resolves to the note count. Progress reads via `index_status_json`.
    fn rebuild_index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.rebuild_index().await })
    }

    /// Re-embed + re-ingest specific notes after a text edit outside the
    /// upsert ops (find/replace, migration) — awaitable.
    fn reindex_notes<'py>(
        &self,
        py: Python<'py>,
        note_ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.reindex_notes(&note_ids).await })
    }

    /// The boot/reload drift path (awaitable; drive as a background task).
    fn reindex_if_needed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.reindex_if_needed().await })
    }

    fn col_mod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.col_mod().await })
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
        // Deliberately UN-spawned (the one PyResult-shaped path): the future
        // is just channel sends/awaits, pollable on the loop as before.
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
        kernel_op(py, async move { kernel.release().await })
    }

    /// Re-acquire after a release — awaitable (busy surfaces as the typed
    /// BUSY error tier).
    fn reopen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.reopen().await })
    }

    /// Flush the index, then close the collection AND drain the actor
    /// (`Kernel::close` — the #374 interpreter-teardown guard: nothing is
    /// mid-job when this resolves). Awaitable.
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let _ = kernel.index().save();
            kernel.close().await
        })
    }
}
