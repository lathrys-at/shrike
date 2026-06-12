//! The anki collection-core binding (#278 series, step 1) — `anki-core`
//! feature builds only.
//!
//! Binds `shrike_collection::CollectionCore` (anki consumed exclusively via
//! its protobuf service layer) for the **parity harness** in `tests/native`:
//! ported wrapper-fixture cases run against this class on its own temp
//! collection. The hard safety rule stands — one collection is only ever
//! touched through ONE core, so the harness never opens a collection through
//! both this binding and the pip `anki` package; cross-core parity cases run
//! the pip side in a subprocess on a *separate* collection file.
//!
//! Marshaling follows the shrike-ffi conventions: strings, i64 keys, small
//! tuples; every collection op runs under `py.detach` (GIL released).

use pyo3::prelude::*;
use shrike_collection::{CollectionCore as Core, CreateOutcome, DuplicatePolicy};

use crate::to_py_err;

/// One note's full raw field map: `(note_id, names, values)`.
type FieldMapRow = (i64, Vec<String>, Vec<String>);

/// One open anki collection, instance-per-collection (mirrors the Rust core's
/// lifecycle; `close()` is explicit, like the facade it will eventually back).
#[pyclass(frozen)]
pub(crate) struct CollectionCore {
    /// `Arc`-shared since #332 (S3d): the kernel and the harness's direct ops
    /// hold ONE open collection (serialization is the executor's discipline).
    inner: std::sync::Arc<Core>,
}

impl CollectionCore {
    /// The wrapped core, for the per-action bindings (#331) that compose
    /// kernel action bodies over the live handle.
    pub(crate) fn core_ref(&self) -> &Core {
        &self.inner
    }

    /// A harness handle over an existing (kernel-owned) core.
    pub(crate) fn from_arc(inner: std::sync::Arc<Core>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CollectionCore {
    /// Open (creating if needed) the collection at `collection_path`.
    #[new]
    fn new(py: Python<'_>, collection_path: String) -> PyResult<Self> {
        let inner = py
            .detach(move || Core::open(&collection_path))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close()).map_err(to_py_err)
    }

    /// Cooperative idle-release (#64): close, keeping the instance reusable.
    fn release(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.release()).map_err(to_py_err)
    }

    /// Re-acquire after a release; lock contention surfaces as
    /// NativeBusyError (retryable), mirroring the Python wrapper.
    fn reopen(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.reopen()).map_err(to_py_err)
    }

    /// The collection-modified watermark (drift detection's anchor).
    fn col_mod(&self, py: Python<'_>) -> PyResult<i64> {
        py.detach(|| self.inner.col_mod()).map_err(to_py_err)
    }

    /// The full Anki search grammar → note ids (read-only).
    fn find_notes(&self, py: Python<'_>, search: String) -> PyResult<Vec<i64>> {
        py.detach(|| self.inner.find_notes(&search))
            .map_err(to_py_err)
    }

    /// Resolve a notetype by name (case-sensitive).
    fn notetype_id(&self, py: Python<'_>, name: String) -> PyResult<i64> {
        py.detach(|| self.inner.notetype_id(&name))
            .map_err(to_py_err)
    }

    /// Read one note: `(id, notetype_id, fields, tags)`.
    fn get_note(
        &self,
        py: Python<'_>,
        note_id: i64,
    ) -> PyResult<(i64, i64, Vec<String>, Vec<String>)> {
        py.detach(|| self.inner.get_note(note_id))
            .map(|n| (n.id, n.notetype_id, n.fields, n.tags))
            .map_err(to_py_err)
    }

    /// Create a note under the #77 duplicate policy. Returns the new note id,
    /// or None when a first-field duplicate was skipped (`on_duplicate="skip"`);
    /// structural problems and policy `"error"` duplicates raise
    /// NativeInputError, exactly like the Rust core.
    #[pyo3(signature = (notetype_id, deck_id, fields, tags, on_duplicate="error"))]
    fn create_note(
        &self,
        py: Python<'_>,
        notetype_id: i64,
        deck_id: i64,
        fields: Vec<String>,
        tags: Vec<String>,
        on_duplicate: &str,
    ) -> PyResult<Option<i64>> {
        let policy = DuplicatePolicy::parse(on_duplicate).map_err(to_py_err)?;
        let outcome = py
            .detach(|| {
                self.inner
                    .create_note(notetype_id, deck_id, &fields, &tags, policy)
            })
            .map_err(to_py_err)?;
        Ok(match outcome {
            CreateOutcome::Created(id) => Some(id),
            CreateOutcome::SkippedDuplicate => None,
        })
    }

    /// Replace a note's fields (and tags, when given — None leaves them).
    #[pyo3(signature = (note_id, fields, tags=None))]
    fn update_note(
        &self,
        py: Python<'_>,
        note_id: i64,
        fields: Vec<String>,
        tags: Option<Vec<String>>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.update_note(note_id, &fields, tags.as_deref()))
            .map_err(to_py_err)
    }

    /// Delete notes by id; returns the removed count.
    fn delete_notes(&self, py: Python<'_>, note_ids: Vec<i64>) -> PyResult<usize> {
        py.detach(|| self.inner.delete_notes(&note_ids))
            .map_err(to_py_err)
    }

    /// The raw Anki search escape hatch — list_notes-shaped JSON (the core
    /// returns the typed response since #391 phase 2; this binding is its
    /// serialization edge — plain serde, explicit nulls).
    #[pyo3(signature = (search, with_fields=true, limit=50))]
    fn query(
        &self,
        py: Python<'_>,
        search: String,
        with_fields: bool,
        limit: usize,
    ) -> PyResult<String> {
        py.detach(|| {
            let resp = self.inner.query(&search, with_fields, limit)?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }

    /// Full raw field map per note id: `(note_id, names, values)` rows
    /// (empty fields included, unlike derived_field_rows).
    fn note_field_map(&self, py: Python<'_>, note_ids: Vec<i64>) -> PyResult<Vec<FieldMapRow>> {
        py.detach(|| self.inner.note_field_map(&note_ids))
            .map_err(to_py_err)
    }

    /// Deck reference (name / id / #id) → canonical deck name, or None for an
    /// explicit id matching no deck.
    fn resolve_deck_ref(&self, py: Python<'_>, reference: String) -> PyResult<Option<String>> {
        py.detach(|| self.inner.resolve_deck_ref(&reference))
            .map_err(to_py_err)
    }

    // ── write surface (#278 step 3) ──────────────────────────────────────────

    /// The bulk named-fields upsert: wrapper-shaped note dicts as JSON in,
    /// per-item results JSON out (`created`/`updated`/`ok`/`skipped`/`error`
    /// with the `_upsert_notes` reason vocabulary).
    #[pyo3(signature = (notes_json, on_duplicate="error", dry_run=false))]
    fn upsert_notes(
        &self,
        py: Python<'_>,
        notes_json: String,
        on_duplicate: &str,
        dry_run: bool,
    ) -> PyResult<String> {
        py.detach(|| -> shrike_ffi::NativeResult<String> {
            let notes: Vec<shrike_schemas::NoteInput> =
                serde_json::from_str(&notes_json).map_err(|e| {
                    shrike_ffi::NativeError::invalid_input(format!(
                        "notes must be a JSON list: {e}"
                    ))
                })?;
            let policy = shrike_collection::DuplicatePolicy::parse(on_duplicate)?;
            let results = self.inner.upsert_notes(&notes, policy, dry_run)?;
            serde_json::to_string(&results)
                .map_err(|e| shrike_ffi::NativeError::internal(e.to_string()))
        })
        .map_err(to_py_err)
    }

    /// Tags on a note set: `set_tags` replaces (exclusive with add/remove,
    /// validated by the caller). Returns `(notes_modified, not_found)`.
    #[pyo3(signature = (note_ids, set_tags=None, add=None, remove=None))]
    fn update_note_tags(
        &self,
        py: Python<'_>,
        note_ids: Vec<i64>,
        set_tags: Option<Vec<String>>,
        add: Option<Vec<String>>,
        remove: Option<Vec<String>>,
    ) -> PyResult<(usize, Vec<i64>)> {
        py.detach(|| {
            self.inner.update_note_tags(
                &note_ids,
                set_tags.as_deref(),
                add.as_deref().unwrap_or(&[]),
                remove.as_deref().unwrap_or(&[]),
            )
        })
        .map_err(to_py_err)
    }

    /// Rename a tag collection-wide (empty `note_ids`) or exactly on a set.
    fn rename_tag(
        &self,
        py: Python<'_>,
        old: String,
        new: String,
        note_ids: Vec<i64>,
    ) -> PyResult<usize> {
        py.detach(|| self.inner.rename_tag(&old, &new, &note_ids))
            .map_err(to_py_err)
    }

    /// Create or rename decks in bulk (JSON in/out; id present = rename).
    fn upsert_decks(&self, py: Python<'_>, decks_json: String) -> PyResult<String> {
        py.detach(|| self.inner.upsert_decks(&decks_json))
            .map_err(to_py_err)
    }

    /// Delete decks by reference, empty-only (JSON result echoes the refs).
    fn delete_decks(&self, py: Python<'_>, refs: Vec<String>) -> PyResult<String> {
        py.detach(|| self.inner.delete_decks(&refs))
            .map_err(to_py_err)
    }

    /// Anki's find_and_replace over a note set + changed-id diff (JSON out).
    #[pyo3(signature = (note_ids, search, replacement, regex=false, match_case=true, field=None))]
    #[allow(clippy::too_many_arguments)]
    fn find_replace_notes(
        &self,
        py: Python<'_>,
        note_ids: Vec<i64>,
        search: String,
        replacement: String,
        regex: bool,
        match_case: bool,
        field: Option<String>,
    ) -> PyResult<String> {
        py.detach(|| {
            self.inner.find_replace_notes(
                &note_ids,
                &search,
                &replacement,
                regex,
                match_case,
                field.as_deref(),
            )
        })
        .map_err(to_py_err)
    }

    /// Delete note types by id, only-if-unused (typed core results, JSON
    /// serialized once here at the edge).
    fn delete_note_types(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<String> {
        py.detach(|| {
            let results = self.inner.delete_note_types(&ids)?;
            crate::kernel_actions::wire(&shrike_schemas::DeleteNoteTypesResponse { results })
        })
        .map_err(to_py_err)
    }

    // ── media + maintenance (#278 step 5a) ───────────────────────────────────

    /// Store one media item from prepared bytes (full result shape: mime,
    /// deduped, extension-from-content-type); use the RETURNED filename.
    #[pyo3(signature = (data, filename=None, content_type=None))]
    fn store_media_bytes(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        filename: Option<String>,
        content_type: Option<String>,
    ) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .store_media_bytes(filename.as_deref(), &data, content_type.as_deref())
        })
        .map_err(to_py_err)
    }

    /// Locate media files (never bytes): per-item found/missing JSON.
    fn fetch_media(&self, py: Python<'_>, filenames: Vec<String>) -> PyResult<String> {
        py.detach(|| self.inner.fetch_media(&filenames))
            .map_err(to_py_err)
    }

    /// List media filenames (sorted; optional glob pattern + limit), JSON.
    #[pyo3(signature = (pattern=None, limit=None))]
    fn list_media(
        &self,
        py: Python<'_>,
        pattern: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<String> {
        py.detach(|| self.inner.list_media(pattern.as_deref(), limit))
            .map_err(to_py_err)
    }

    /// Move media files to Anki's recoverable trash (JSON result echoes refs).
    fn delete_media(&self, py: Python<'_>, filenames: Vec<String>) -> PyResult<String> {
        py.detach(|| self.inner.delete_media(&filenames))
            .map_err(to_py_err)
    }

    /// Read-only media diagnostics (unused/missing/missing-notes/trash), JSON.
    fn media_check(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.media_check()).map_err(to_py_err)
    }

    /// The #89 prune: four cleanups, dry-run previews; `removed_note_ids`
    /// rides in the JSON for the host's index maintenance.
    #[pyo3(signature = (unused_tags=true, empty_notes=true, empty_cards=true, unused_media=true, dry_run=true))]
    fn prune(
        &self,
        py: Python<'_>,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .prune(unused_tags, empty_notes, empty_cards, unused_media, dry_run)
        })
        .map_err(to_py_err)
    }

    // ── media URL/path sources (#278 step 5b — security-review gated) ────────

    /// The full store_media batch: items of `data` (base64) / `url`
    /// (SSRF-guarded download with IP pinning) / server-local `path` (honored
    /// only inside `path_roots`). Per-item results JSON.
    #[pyo3(signature = (items_json, allow_private_fetch=false, path_roots=None))]
    fn store_media_items(
        &self,
        py: Python<'_>,
        items_json: String,
        allow_private_fetch: bool,
        path_roots: Option<Vec<String>>,
    ) -> PyResult<String> {
        py.detach(|| {
            self.inner.store_media_items(
                &items_json,
                allow_private_fetch,
                path_roots.as_deref().unwrap_or(&[]),
            )
        })
        .map_err(to_py_err)
    }

    /// The SSRF allowlist classifier (one IP literal) — the parity surface
    /// the harness compares against Python's `ipaddress.is_global` corpus.
    fn media_ip_allowed(&self, ip: &str) -> PyResult<bool> {
        let addr: std::net::IpAddr = ip
            .parse()
            .map_err(|e| crate::NativeInputError::new_err(format!("bad ip: {e}")))?;
        Ok(shrike_collection::media_fetch::ip_is_allowed(addr))
    }

    // ── note types (#278 step 4) ─────────────────────────────────────────────

    /// Create/update note-type definitions in bulk (the position-keyed
    /// replace with the #76 unsound-move rejection). JSON at this edge only:
    /// typed inputs in, typed per-item results serialized once on the way out.
    fn upsert_note_types(&self, py: Python<'_>, note_types_json: String) -> PyResult<String> {
        py.detach(|| -> shrike_ffi::NativeResult<String> {
            let inputs: Vec<shrike_schemas::NoteTypeInput> = serde_json::from_str(&note_types_json)
                .map_err(|e| {
                    shrike_ffi::NativeError::invalid_input(format!(
                        "note_types must be a JSON list: {e}"
                    ))
                })?;
            let results = self.inner.upsert_note_types(&inputs)?;
            crate::kernel_actions::wire(&results)
        })
        .map_err(to_py_err)
    }

    /// Identity-based field ops (add/remove/rename/reposition), atomic.
    fn update_note_type_fields(
        &self,
        py: Python<'_>,
        note_type_name: String,
        operations_json: String,
    ) -> PyResult<String> {
        py.detach(|| -> shrike_ffi::NativeResult<String> {
            let operations: Vec<shrike_schemas::FieldOp> = serde_json::from_str(&operations_json)
                .map_err(|e| {
                shrike_ffi::NativeError::invalid_input(format!(
                    "operations must be a JSON list: {e}"
                ))
            })?;
            let resp = self
                .inner
                .update_note_type_fields(&note_type_name, &operations)?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }

    /// Identity-based template ops, atomic.
    fn update_note_type_templates(
        &self,
        py: Python<'_>,
        note_type_name: String,
        operations_json: String,
    ) -> PyResult<String> {
        py.detach(|| -> shrike_ffi::NativeResult<String> {
            let operations: Vec<shrike_schemas::TemplateOp> =
                serde_json::from_str(&operations_json).map_err(|e| {
                    shrike_ffi::NativeError::invalid_input(format!(
                        "operations must be a JSON list: {e}"
                    ))
                })?;
            let resp = self
                .inner
                .update_note_type_templates(&note_type_name, &operations)?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }

    /// Literal-or-regex rewrite over one model's template HTML + CSS.
    #[pyo3(signature = (note_type_name, search, replacement, regex=false, match_case=true, front=true, back=true, css=true))]
    #[allow(clippy::too_many_arguments)]
    fn find_replace_note_types(
        &self,
        py: Python<'_>,
        note_type_name: String,
        search: String,
        replacement: String,
        regex: bool,
        match_case: bool,
        front: bool,
        back: bool,
        css: bool,
    ) -> PyResult<String> {
        py.detach(|| {
            let resp = self.inner.find_and_replace_note_types(
                &note_type_name,
                &search,
                &replacement,
                regex,
                match_case,
                front,
                back,
                css,
            )?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }

    /// Per-field editor metadata (font/size/description), atomic.
    fn update_note_type_field_metadata(
        &self,
        py: Python<'_>,
        note_type_name: String,
        updates_json: String,
    ) -> PyResult<String> {
        py.detach(|| -> shrike_ffi::NativeResult<String> {
            let updates: Vec<shrike_schemas::FieldMetadataInput> =
                serde_json::from_str(&updates_json).map_err(|e| {
                    shrike_ffi::NativeError::invalid_input(format!(
                        "updates must be a JSON list: {e}"
                    ))
                })?;
            let resp = self
                .inner
                .update_note_type_field_metadata(&note_type_name, &updates)?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }

    /// Change notes' note type via name maps (Anki's history-safe migration);
    /// `template_map_json` may be empty (= map templates by ordinal).
    #[pyo3(signature = (note_ids, new_note_type, field_map_json, template_map_json="", dry_run=false))]
    fn migrate_note_type(
        &self,
        py: Python<'_>,
        note_ids: Vec<i64>,
        new_note_type: String,
        field_map_json: String,
        template_map_json: &str,
        dry_run: bool,
    ) -> PyResult<String> {
        py.detach(|| -> shrike_ffi::NativeResult<String> {
            let field_map: std::collections::BTreeMap<String, String> =
                serde_json::from_str(&field_map_json).map_err(|e| {
                    shrike_ffi::NativeError::invalid_input(format!(
                        "field_map must be a JSON object: {e}"
                    ))
                })?;
            let template_map: std::collections::BTreeMap<String, String> =
                if template_map_json.is_empty() {
                    Default::default()
                } else {
                    serde_json::from_str(template_map_json).map_err(|e| {
                        shrike_ffi::NativeError::invalid_input(format!(
                            "template_map must be a JSON object: {e}"
                        ))
                    })?
                };
            let resp = self.inner.migrate_note_type(
                &note_ids,
                &new_note_type,
                &field_map,
                &template_map,
                dry_run,
            )?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }

    /// Card ids generated by one note.
    fn cards_of_note(&self, py: Python<'_>, note_id: i64) -> PyResult<Vec<i64>> {
        py.detach(|| self.inner.cards_of_note(note_id))
            .map_err(to_py_err)
    }

    /// `(card_id, template_ordinal)` per card of one note.
    fn card_ords_of_note(&self, py: Python<'_>, note_id: i64) -> PyResult<Vec<(i64, i64)>> {
        py.detach(|| self.inner.card_ords_of_note(note_id))
            .map_err(to_py_err)
    }

    // ── read surface (#278 step 2) ───────────────────────────────────────────

    /// Normalized embedding text per note id ("" for a missing id).
    fn note_texts(&self, py: Python<'_>, note_ids: Vec<i64>) -> PyResult<Vec<String>> {
        py.detach(|| self.inner.note_texts(&note_ids))
            .map_err(to_py_err)
    }

    /// `(note_id, text, image_names)` per note id — the multimodal input.
    fn note_embed_inputs(
        &self,
        py: Python<'_>,
        note_ids: Vec<i64>,
    ) -> PyResult<Vec<(i64, String, Vec<String>)>> {
        py.detach(|| self.inner.note_embed_inputs(&note_ids))
            .map_err(to_py_err)
    }

    /// `(note_id, source, field_name, raw_value)` rows for the derived store.
    fn derived_field_rows(
        &self,
        py: Python<'_>,
        note_ids: Vec<i64>,
    ) -> PyResult<Vec<(i64, String, String, String)>> {
        py.detach(|| self.inner.derived_field_rows(&note_ids))
            .map_err(to_py_err)
    }

    /// One field value through the embedding normalization (the byte-identity
    /// parity surface against `shrike.embed_text.normalize_for_embedding`).
    fn normalize_text(&self, py: Python<'_>, value: String) -> PyResult<String> {
        py.detach(|| self.inner.normalize_text(&value))
            .map_err(to_py_err)
    }

    /// Structured filters → notes; returns the wrapper-shaped JSON
    /// (`{"notes": [...], "total": N, "limit": L}`). `modified_since` is an
    /// epoch-seconds cutoff (the host parses ISO timestamps).
    #[pyo3(signature = (ids=None, deck=None, tags=None, note_type=None, modified_since=None, with_fields=true, limit=50))]
    #[allow(clippy::too_many_arguments)]
    fn list_notes(
        &self,
        py: Python<'_>,
        ids: Option<Vec<i64>>,
        deck: Option<String>,
        tags: Option<Vec<String>>,
        note_type: Option<String>,
        modified_since: Option<i64>,
        with_fields: bool,
        limit: usize,
    ) -> PyResult<String> {
        py.detach(|| {
            let resp = self.inner.list_notes(
                ids.as_deref(),
                deck.as_deref(),
                tags.as_deref(),
                note_type.as_deref(),
                modified_since,
                with_fields,
                limit,
            )?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }

    /// The sectioned collection info dict as JSON (`sections` mirrors
    /// `include`; `"all"` expands; empty = summary).
    #[pyo3(signature = (sections=None, detail_names=None))]
    fn collection_info(
        &self,
        py: Python<'_>,
        sections: Option<Vec<String>>,
        detail_names: Option<Vec<String>>,
    ) -> PyResult<String> {
        py.detach(|| {
            let resp = self.inner.collection_info(
                sections.as_deref().unwrap_or(&[]),
                detail_names.as_deref().unwrap_or(&[]),
            )?;
            crate::kernel_actions::wire(&resp)
        })
        .map_err(to_py_err)
    }
}
