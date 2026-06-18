//! Async kernel bindings (#332, S3a; reshaped by #374): every op is spawned
//! onto the kernel's owned runtime at this edge (`spawn_op`) and surfaces as
//! an `asyncio.Future` through the one-wake completion bridge.

use std::future::Future;
use std::sync::Arc;

use pyo3::prelude::*;

use shrike_collection::{CreateOutcome, DuplicatePolicy, ImportOptions, ImportUpdateCondition};
use shrike_error::NativeResult;
use shrike_kernel::{Embedder, Kernel, NoteSpec, SerializedCollection};

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

fn outcome_to_wire(outcome: shrike_error::NativeResult<CreateOutcome>) -> UpsertWireResult {
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

    /// Route a resolved embedder + image pair into the kernel's embed SET
    /// keyed by space (#233). The explicit `space_key` wins; otherwise the key
    /// is the embedder's own CONTENT fingerprint (so an unkeyed single-embedder
    /// attach lands exactly one space, byte-identical to the pre-#233 slot).
    fn attach_space(
        &self,
        space_key: Option<String>,
        embedder: Arc<dyn shrike_kernel::Embedder>,
        images: Option<shrike_kernel::KernelImages>,
    ) {
        let key = space_key.or_else(|| embedder.fingerprint());
        self.inner.attach_embedder_space(key, embedder, images);
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
/// captured Python backend (custom/test recognizers). The native variant
/// exists only in `engine-apple` builds (#499) — without it the captured
/// handle is the sole shape.
#[derive(FromPyObject)]
enum AnyRecognizer<'py> {
    #[cfg(feature = "engine-apple")]
    Native(PyRef<'py, crate::py_recognizer::AppleVisionRecognizer>),
    #[cfg(feature = "engine-remote")]
    Describe(PyRef<'py, crate::py_recognizer::RemoteDescriber>),
    Captured(PyRef<'py, crate::py_recognizer::PyRecognizer>),
}

/// Map the harness's purpose string onto the kernel's routing enum (#485).
/// The string IS the derived `source` it lands under (`"ocr"`/`"vlm"`/`"asr"`)
/// — the same names the kernel's `RecognitionPurpose::source()` returns.
fn purpose_from_str(purpose: &str) -> PyResult<shrike_kernel::RecognitionPurpose> {
    match purpose {
        "ocr" => Ok(shrike_kernel::RecognitionPurpose::Ocr),
        "vlm" | "describe" => Ok(shrike_kernel::RecognitionPurpose::Describe),
        "asr" => Ok(shrike_kernel::RecognitionPurpose::Asr),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown recognition purpose {other:?} (choices: ocr, describe, asr)"
        ))),
    }
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
#[pyo3(signature = (collection_path, cache_dir, save_delay=None, save_threshold=None))]
pub(crate) fn async_kernel_open<'py>(
    py: Python<'py>,
    collection_path: String,
    cache_dir: String,
    save_delay: Option<f64>,
    save_threshold: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    kernel_op(py, async move {
        let kernel =
            Kernel::open_with(&collection_path, &cache_dir, save_delay, save_threshold).await?;
        Ok(AsyncKernel {
            inner: Arc::new(kernel),
        })
    })
}

#[pymethods]
impl AsyncKernel {
    /// Attach (or swap) an embedding space (#233) — embedding start / model
    /// change / one call per space in the multi-space fan-out. Takes either
    /// embedder shape ([`AnyEmbedder`]): the native composition embeds without
    /// re-entering Python; the captured handle dispatches to the Python
    /// backend. `space_key` pins the space's identity (the CONTENT fingerprint,
    /// reorder-stable, #233) — when `None` the kernel keys off the embedder's
    /// own fingerprint, so an existing single-embedder host attaches exactly as
    /// before (one space, byte-identical). Follow up with `reindex_if_needed`
    /// (a model change is drift).
    #[pyo3(signature = (embedder, media_read=None, media_exists=None, space_key=None))]
    fn attach_embedder(
        &self,
        embedder: AnyEmbedder<'_>,
        media_read: Option<Py<PyAny>>,
        media_exists: Option<Py<PyAny>>,
        space_key: Option<String>,
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
                let handle = Arc::clone(&native.text);
                self.attach_space(space_key, handle, images);
            }
            AnyEmbedder::Captured(captured) => {
                let handle = Arc::clone(&captured.handle);
                let images = image_pair(&handle, media_read, media_exists);
                self.attach_space(space_key, handle, images);
            }
        }
    }

    /// Detach the embedding spaces (embedding stop): with no `space_key`, the
    /// whole set clears (the N=1 stop) — the index flushes and reports
    /// unavailable; with a key, only that one space detaches (the index keeps
    /// serving the primary while another space remains). The collection and
    /// lexical search stay live in both cases.
    #[pyo3(signature = (space_key=None))]
    fn detach_embedder(&self, py: Python<'_>, space_key: Option<String>) {
        py.detach(|| match space_key {
            Some(key) => {
                self.inner.detach_embedder_space(&key);
            }
            None => self.inner.detach_embedder(),
        })
    }

    /// The number of attached embedding spaces (#233) — the multi-space status
    /// surface. The index/search path still consumes only the primary this PR.
    fn embed_space_count(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.embed_space_count())
    }

    /// Attach the OCR recognition service (#228, the second #342 slot) — the
    /// OCR-defaulting convenience over [`attach_recognizer_with`] (#485):
    /// existing hosts/tests keep the single-arg shape and target the OCR
    /// purpose. An OCR/ASR/describe engine plus the media-resolver callables it
    /// reads bytes through (independent of the embed slot — recognition works
    /// with a text-only embedder).
    fn attach_recognizer(
        &self,
        recognizer: AnyRecognizer<'_>,
        media_read: Py<PyAny>,
        media_exists: Py<PyAny>,
    ) -> PyResult<()> {
        self.attach_recognizer_with("ocr", recognizer, media_read, media_exists)
    }

    /// Attach (or swap) the recognition service for a specific purpose (#485)
    /// — OCR, describe, or ASR, each routed to its own pending set / source /
    /// fingerprint / destination by the kernel sweep. Takes either recognizer
    /// shape ([`AnyRecognizer`]); the native engines are adapted onto the
    /// blocking pool here via `Blocking`, exactly like OCR.
    fn attach_recognizer_with(
        &self,
        purpose: &str,
        recognizer: AnyRecognizer<'_>,
        media_read: Py<PyAny>,
        media_exists: Py<PyAny>,
    ) -> PyResult<()> {
        let purpose = purpose_from_str(purpose)?;
        let resolver: Arc<dyn shrike_kernel::ImageResolver> =
            Arc::new(PyMediaResolver::new(media_read, media_exists));
        match recognizer {
            #[cfg(feature = "engine-apple")]
            AnyRecognizer::Native(native) => {
                let adapted: Arc<dyn shrike_kernel::Recognizer> =
                    Arc::new(shrike_engine_api::Blocking(native.engine_arc()));
                self.inner
                    .attach_recognizer_with(purpose, adapted, resolver);
            }
            #[cfg(feature = "engine-remote")]
            AnyRecognizer::Describe(describe) => {
                let adapted: Arc<dyn shrike_kernel::Recognizer> =
                    Arc::new(shrike_engine_api::Blocking(describe.engine_arc()));
                self.inner
                    .attach_recognizer_with(purpose, adapted, resolver);
            }
            AnyRecognizer::Captured(captured) => {
                let handle: Arc<dyn shrike_kernel::Recognizer> = Arc::clone(&captured.handle) as _;
                self.inner.attach_recognizer_with(purpose, handle, resolver);
            }
        }
        Ok(())
    }

    /// Detach the OCR recognition service (the OCR-defaulting convenience):
    /// derived text stays (still valid output of the engine that produced it);
    /// only new OCR recognition stops.
    fn detach_recognizer(&self, py: Python<'_>) {
        py.detach(|| self.inner.detach_recognizer())
    }

    /// Detach the recognition service for a specific purpose (#485).
    fn detach_recognizer_for(&self, py: Python<'_>, purpose: &str) -> PyResult<()> {
        let purpose = purpose_from_str(purpose)?;
        py.detach(|| self.inner.detach_recognizer_for(purpose));
        Ok(())
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
            crate::kernel_actions::wire(&report)
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
        // The FFI still speaks JSON (the marshaling convention), parsed ONCE
        // here into the typed seam (#391) and serialized once on the way out.
        let notes: Vec<shrike_schemas::NoteInput> =
            serde_json::from_str(&notes_json).map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "notes must be a JSON list: {e}"
                )))
            })?;
        let policy = DuplicatePolicy::parse(&on_duplicate).map_err(crate::to_py_err)?;
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let results = kernel.upsert_notes_wire(notes, policy, dry_run).await?;
            serde_json::to_string(&results)
                .map_err(|e| shrike_error::NativeError::internal(e.to_string()))
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

    /// Import an .apkg/.colpkg (#72) — awaitable. MUTATES the collection and
    /// reconciles the index (the drift tail is kernel-side). The conflict
    /// conditions arrive as strings (`if_newer`/`always`/`never`); returns
    /// `(summary_json, reindexed)` — the per-bucket counts JSON and whether the
    /// index reconciled. The derived-store rebuild is the harness's follow-up.
    #[pyo3(signature = (
        package_path, update_notes, update_notetypes, with_scheduling, merge_notetypes
    ))]
    fn import_package<'py>(
        &self,
        py: Python<'py>,
        package_path: String,
        update_notes: String,
        update_notetypes: String,
        with_scheduling: bool,
        merge_notetypes: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = ImportOptions {
            update_notes: ImportUpdateCondition::parse(&update_notes).map_err(crate::to_py_err)?,
            update_notetypes: ImportUpdateCondition::parse(&update_notetypes)
                .map_err(crate::to_py_err)?,
            with_scheduling,
            merge_notetypes,
        };
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            kernel.import_package(package_path, options).await
        })
    }

    /// Advance the watermarks after a metadata-only change (tags/decks/
    /// templates) — no re-embed, no drift on next boot. Awaitable.
    /// `membership_may_have_changed` is the tag-centroid relevance probe (#600):
    /// pass `True` only for a tag-membership change (a centroid input moved);
    /// `False` for deck/template/field-metadata edits, which would otherwise
    /// trigger a full O(collection) recompute behind no relevance signal.
    fn metadata_changed<'py>(
        &self,
        py: Python<'py>,
        membership_may_have_changed: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            kernel.metadata_changed(membership_may_have_changed).await
        })
    }

    /// Delete notes in ONE maintained op (#604): the existence partition, the
    /// anki delete, and the sidecar drop (vectors/fingerprints/derived rows)
    /// run as a single kernel op. Returns `{"deleted": [...], "not_found": [...]}`
    /// JSON (the marshaling convention — parsed once on the Python side), so the
    /// action no longer needs a separate `wrapper.delete_notes` existence
    /// pre-check + `forget_notes` round trip.
    fn delete_notes<'py>(
        &self,
        py: Python<'py>,
        note_ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel.delete_notes(note_ids).await?;
            serde_json::to_string(&response).map_err(|e| {
                shrike_error::NativeError::internal(format!("serializing delete response: {e}"))
            })
        })
    }

    // ── media + maintenance ops (#391 re-home) ──────────────────────────────

    /// The full store_media batch (#70): byte sources prepare concurrently
    /// on the kernel's blocking pool, the batch writes as one collection
    /// job. Per-item results JSON; the host fills nothing.
    #[pyo3(signature = (items_json, allow_private_fetch=false, path_roots=None))]
    fn store_media<'py>(
        &self,
        py: Python<'py>,
        items_json: String,
        allow_private_fetch: bool,
        path_roots: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<shrike_schemas::StoreMediaItem> = serde_json::from_str(&items_json)
            .map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "items must be a JSON list: {e}"
                )))
            })?;
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let results = kernel
                .store_media(items, allow_private_fetch, path_roots.unwrap_or_default())
                .await?;
            crate::kernel_actions::wire(&results)
        })
    }

    /// Locate media files (never bytes): per-item found/missing JSON (the
    /// host fills each found file's serving `url`).
    fn fetch_media<'py>(
        &self,
        py: Python<'py>,
        filenames: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let results = kernel.fetch_media(filenames).await?;
            crate::kernel_actions::wire(&results)
        })
    }

    /// List media filenames (sorted; optional glob pattern + limit), JSON.
    #[pyo3(signature = (pattern=None, limit=None))]
    fn list_media<'py>(
        &self,
        py: Python<'py>,
        pattern: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel.list_media(pattern, limit).await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Move media files to Anki's recoverable trash, JSON result.
    fn delete_media<'py>(
        &self,
        py: Python<'py>,
        filenames: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel.delete_media(filenames).await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Read-only media diagnostics, JSON.
    fn media_check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel.media_check().await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// The #89 prune with its kernel-side maintenance tail; response JSON
    /// (removed note ids stay kernel-internal).
    #[pyo3(signature = (unused_tags=true, empty_notes=true, empty_cards=true, unused_media=true, dry_run=true))]
    fn collection_prune<'py>(
        &self,
        py: Python<'py>,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel
                .collection_prune(unused_tags, empty_notes, empty_cards, unused_media, dry_run)
                .await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Export the collection (or a scope of it) to an Anki package (#71).
    /// `format` is "apkg" | "colpkg"; `scope_kind` is "whole" | "deck" |
    /// "notes" (with `deck`/`note_ids` supplying the scope payload). The host
    /// has already gated `out_path` (the path-safety check). Returns the
    /// `ExportPackageResult` JSON (note_count + the on-disk path).
    #[pyo3(signature = (out_path, format, scope_kind, deck=None, note_ids=None, with_scheduling=false, with_media=true, legacy=false))]
    #[allow(clippy::too_many_arguments)]
    fn export_package<'py>(
        &self,
        py: Python<'py>,
        out_path: String,
        format: String,
        scope_kind: String,
        deck: Option<String>,
        note_ids: Option<Vec<i64>>,
        with_scheduling: bool,
        with_media: bool,
        legacy: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        use shrike_kernel::{ExportScope, PackageFormat};
        let format = match format.as_str() {
            "apkg" => PackageFormat::Apkg,
            "colpkg" => PackageFormat::Colpkg,
            other => {
                return Err(crate::to_py_err(shrike_error::NativeError::invalid_input(
                    format!("format must be apkg/colpkg (got {other:?})"),
                )))
            }
        };
        let scope = match scope_kind.as_str() {
            "whole" => ExportScope::Whole,
            "deck" => ExportScope::Deck(deck.ok_or_else(|| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(
                    "deck scope needs a deck reference",
                ))
            })?),
            "notes" => ExportScope::Notes(note_ids.unwrap_or_default()),
            other => {
                return Err(crate::to_py_err(shrike_error::NativeError::invalid_input(
                    format!("scope_kind must be whole/deck/notes (got {other:?})"),
                )))
            }
        };
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let result = kernel
                .export_package(out_path, format, scope, with_scheduling, with_media, legacy)
                .await?;
            crate::kernel_actions::wire(&result)
        })
    }

    // ── tag + deck ops (#391 re-home, long-tail group 2) ────────────────────

    /// Edit tags on a note set (`set_tags` full-replace XOR add/remove);
    /// response JSON, watermark tail kernel-side.
    #[pyo3(signature = (note_ids, set_tags=None, add=None, remove=None))]
    fn update_note_tags<'py>(
        &self,
        py: Python<'py>,
        note_ids: Vec<i64>,
        set_tags: Option<Vec<String>>,
        add: Option<Vec<String>>,
        remove: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel
                .update_note_tags(
                    note_ids,
                    set_tags,
                    add.unwrap_or_default(),
                    remove.unwrap_or_default(),
                )
                .await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Rename a tag collection-wide (empty `note_ids`) or exactly on a set;
    /// response JSON.
    fn rename_tag<'py>(
        &self,
        py: Python<'py>,
        old: String,
        new: String,
        note_ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel.rename_tag(old, new, note_ids).await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Create or rename decks in bulk; per-item results JSON.
    fn upsert_decks<'py>(
        &self,
        py: Python<'py>,
        decks_json: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let decks: Vec<shrike_schemas::DeckInput> =
            serde_json::from_str(&decks_json).map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "decks must be a JSON list: {e}"
                )))
            })?;
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let results = kernel.upsert_decks(decks).await?;
            crate::kernel_actions::wire(&results)
        })
    }

    /// Delete decks by reference, empty-only; response JSON echoes the refs.
    fn delete_decks<'py>(&self, py: Python<'py>, refs: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel.delete_decks(refs).await?;
            crate::kernel_actions::wire(&response)
        })
    }

    // ── note-type ops (#391 re-home, long-tail group 3) ─────────────────────

    /// Create/update note-type definitions in bulk (#76 positional replace);
    /// per-item results JSON.
    fn upsert_note_types<'py>(
        &self,
        py: Python<'py>,
        note_types_json: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let note_types: Vec<shrike_schemas::NoteTypeInput> = serde_json::from_str(&note_types_json)
            .map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "note_types must be a JSON list: {e}"
                )))
            })?;
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let results = kernel.upsert_note_types(note_types).await?;
            crate::kernel_actions::wire(&results)
        })
    }

    /// Identity-based field ops (add/remove/rename/reposition), atomic; JSON.
    fn update_note_type_fields<'py>(
        &self,
        py: Python<'py>,
        note_type_name: String,
        operations_json: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let operations: Vec<shrike_schemas::FieldOp> = serde_json::from_str(&operations_json)
            .map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "operations must be a JSON list: {e}"
                )))
            })?;
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel
                .update_note_type_fields(note_type_name, operations)
                .await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Identity-based template ops, atomic; JSON.
    fn update_note_type_templates<'py>(
        &self,
        py: Python<'py>,
        note_type_name: String,
        operations_json: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let operations: Vec<shrike_schemas::TemplateOp> = serde_json::from_str(&operations_json)
            .map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "operations must be a JSON list: {e}"
                )))
            })?;
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel
                .update_note_type_templates(note_type_name, operations)
                .await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Literal-or-regex rewrite over one model's template HTML + CSS, with
    /// the kernel-side watermark tail on a real replace; JSON.
    #[pyo3(signature = (note_type_name, search, replacement, regex=false, match_case=true, front=true, back=true, css=true))]
    #[allow(clippy::too_many_arguments)]
    fn find_replace_note_types<'py>(
        &self,
        py: Python<'py>,
        note_type_name: String,
        search: String,
        replacement: String,
        regex: bool,
        match_case: bool,
        front: bool,
        back: bool,
        css: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel
                .find_replace_note_types(
                    note_type_name,
                    search,
                    replacement,
                    regex,
                    match_case,
                    front,
                    back,
                    css,
                )
                .await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Per-field editor metadata (font/size/description), with the kernel's
    /// unconditional watermark tail; JSON.
    fn update_note_type_field_metadata<'py>(
        &self,
        py: Python<'py>,
        note_type_name: String,
        updates_json: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let updates: Vec<shrike_schemas::FieldMetadataInput> = serde_json::from_str(&updates_json)
            .map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "updates must be a JSON list: {e}"
                )))
            })?;
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel
                .update_note_type_field_metadata(note_type_name, updates)
                .await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Change notes' note type via name maps (#75); on apply the kernel
    /// re-embeds the changed notes. An empty `template_map_json` = map
    /// templates by ordinal (the established edge contract); JSON.
    fn migrate_note_type<'py>(
        &self,
        py: Python<'py>,
        note_ids: Vec<i64>,
        new_note_type: String,
        field_map_json: String,
        template_map_json: String,
        dry_run: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let field_map: std::collections::BTreeMap<String, String> =
            serde_json::from_str(&field_map_json).map_err(|e| {
                crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                    "field_map must be a JSON object: {e}"
                )))
            })?;
        let template_map: std::collections::BTreeMap<String, String> =
            if template_map_json.is_empty() {
                Default::default()
            } else {
                serde_json::from_str(&template_map_json).map_err(|e| {
                    crate::to_py_err(shrike_error::NativeError::invalid_input(format!(
                        "template_map must be a JSON object: {e}"
                    )))
                })?
            };
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let response = kernel
                .migrate_note_type(note_ids, new_note_type, field_map, template_map, dry_run)
                .await?;
            crate::kernel_actions::wire(&response)
        })
    }

    /// Delete note types by id, only-if-unused; wraps in
    /// `DeleteNoteTypesResponse` so the `{"results": ...}` wire matches the
    /// sync edge. JSON.
    fn delete_note_types<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let results = kernel.delete_note_types(ids).await?;
            crate::kernel_actions::wire(&shrike_schemas::DeleteNoteTypesResponse { results })
        })
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
    fn rebuild_derived<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.rebuild_derived().await })
    }

    fn reindex_if_needed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.reindex_if_needed().await })
    }

    /// Cross-space search inputs as JSON (#234) — awaitable. Embeds the query
    /// texts into every SECONDARY text-capable space (on the kernel runtime,
    /// where embed is legal — `action_search_notes` runs on the collection-actor
    /// thread and can't await embed, #503) and searches each secondary engine,
    /// returning the per-space `SpaceSemantic` rows the host threads into
    /// `action_search_notes` as `cross_space=`. EMPTY (`"[]"`) when there are no
    /// secondary spaces — the N=1 case, where the host call stays byte-identical.
    /// `fetch_k` is the per-space rank cap.
    fn build_cross_space_json<'py>(
        &self,
        py: Python<'py>,
        source_texts: Vec<String>,
        fetch_k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            let spaces = kernel.build_cross_space(&source_texts, fetch_k).await?;
            serde_json::to_string(&spaces)
                .map_err(|e| shrike_error::NativeError::internal(e.to_string()))
        })
    }

    fn col_mod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move { kernel.col_mod().await })
    }

    /// Recalibrate every secondary cross-space image floor (#576) — awaitable.
    /// The harness drives this after a (re)build / model change. `margin` is the
    /// harness-resolved `search.cross_space_fusion.margin` (#580 — the precision/
    /// recall dial folded into `mean + margin·std`; 1.0 is the default). Returns
    /// the per-space derived floor as `[(space_key, floor_or_None), …]` so the
    /// harness can log/surface the values. No-op (empty) in the N=1 case.
    #[pyo3(signature = (margin))]
    fn calibrate_secondary_floors<'py>(
        &self,
        py: Python<'py>,
        margin: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kernel = Arc::clone(&self.inner);
        kernel_op(py, async move {
            kernel.calibrate_secondary_floors(margin).await
        })
    }

    /// The index status block as JSON (state/size/progress/stamps).
    fn index_status_json(&self) -> PyResult<String> {
        crate::kernel_actions::wire(&self.inner.index().status()).map_err(crate::to_py_err)
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
                .run(move |_core| {
                    // The job's attach window rides the finalization gate
                    // (#435); the refusal is lazy (no Python touched here).
                    let Some(_permit) = crate::finalize_gate::permit() else {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "interpreter is exiting; harness job not run",
                        ));
                    };
                    Python::attach(|py| job.call0(py))
                })
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
