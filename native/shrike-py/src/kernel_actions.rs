//! Per-action bindings for the kernel's action core (#331, S2).
//!
//! Each function is one re-homed action: it takes the live `CollectionCore`
//! handle (the Python harness invokes it *on the collection worker thread*,
//! inside `wrapper.run` — the same serialization every collection op rides),
//! runs the whole action body in `shrike_kernel::actions`, and returns the
//! canonical response as JSON for the Pydantic binding to validate. The GIL is
//! released for the duration (`py.detach`).

use pyo3::prelude::*;

use crate::anki_core::CollectionCore;
use crate::to_py_err;

/// The kernel-side registry — the Python binding asserts its forwarding list
/// against this so the two sides can't drift silently.
#[pyfunction]
pub(crate) fn rehomed_actions() -> Vec<&'static str> {
    shrike_kernel::actions::REHOMED_ACTIONS.to_vec()
}

#[pyfunction]
#[pyo3(signature = (core, include, note_type_details))]
pub(crate) fn action_collection_info(
    py: Python<'_>,
    core: PyRef<'_, CollectionCore>,
    include: Vec<String>,
    note_type_details: Vec<String>,
) -> PyResult<String> {
    let inner = core.core_ref();
    py.detach(|| {
        let info = shrike_kernel::actions::collection_info(inner, &include, &note_type_details)?;
        serde_json::to_string(&info).map_err(|e| shrike_ffi::NativeError::internal(e.to_string()))
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (core, ids=None, deck=None, tags=None, note_type=None, modified_since_epoch=None, with_fields=true, limit=50))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn action_list_notes(
    py: Python<'_>,
    core: PyRef<'_, CollectionCore>,
    ids: Option<Vec<i64>>,
    deck: Option<String>,
    tags: Option<Vec<String>>,
    note_type: Option<String>,
    modified_since_epoch: Option<i64>,
    with_fields: bool,
    limit: usize,
) -> PyResult<String> {
    let inner = core.core_ref();
    let params = shrike_kernel::actions::ListNotesParams {
        ids,
        deck,
        tags,
        note_type,
        modified_since_epoch,
        with_fields,
        limit,
    };
    py.detach(|| {
        let resp = shrike_kernel::actions::list_notes(inner, &params)?;
        serde_json::to_string(&resp).map_err(|e| shrike_ffi::NativeError::internal(e.to_string()))
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (core, query, with_fields=true, limit=50))]
pub(crate) fn action_collection_query(
    py: Python<'_>,
    core: PyRef<'_, CollectionCore>,
    query: String,
    with_fields: bool,
    limit: usize,
) -> PyResult<String> {
    let inner = core.core_ref();
    py.detach(|| {
        let resp = shrike_kernel::actions::collection_query(inner, &query, with_fields, limit)?;
        serde_json::to_string(&resp).map_err(|e| shrike_ffi::NativeError::internal(e.to_string()))
    })
    .map_err(to_py_err)
}
