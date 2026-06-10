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

/// One open anki collection, instance-per-collection (mirrors the Rust core's
/// lifecycle; `close()` is explicit, like the facade it will eventually back).
#[pyclass(frozen)]
pub(crate) struct CollectionCore {
    inner: Core,
}

#[pymethods]
impl CollectionCore {
    /// Open (creating if needed) the collection at `collection_path`.
    #[new]
    fn new(py: Python<'_>, collection_path: String) -> PyResult<Self> {
        let inner = py
            .detach(move || Core::open(&collection_path))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close()).map_err(to_py_err)
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
}
