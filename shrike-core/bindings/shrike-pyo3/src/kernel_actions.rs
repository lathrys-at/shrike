//! Per-action bindings for the kernel's action core.
//!
//! Each function is one re-homed action: it takes the live `CollectionCore`
//! handle (the Python harness invokes it *on the collection worker thread*,
//! inside `wrapper.run` — the same serialization every collection op rides),
//! runs the whole action body in `shrike_kernel::actions`, and returns the
//! canonical response as JSON for the Pydantic binding to validate. The GIL is
//! released for the duration (`py.detach`).
//!
//! THIS is the host edge where a typed response becomes JSON —
//! plain serde of the schema type. One wire convention: an unset `Option` is
//! an explicit `null` (the Pydantic shape the schema contract test pins);
//! every consumer revalidates through the `schemas.py` models, so the wire is
//! shape-compat, not byte-pinned.

use pyo3::prelude::*;

use crate::anki_core::CollectionCore;
use crate::to_py_err;

/// Serialize a typed response onto the host wire; a failure is a native bug.
/// Shared with the direct `CollectionCore` read bindings in `anki_core`.
pub(crate) fn wire<T: serde::Serialize>(value: &T) -> Result<String, shrike_error::NativeError> {
    serde_json::to_string(value)
        .map_err(|e| shrike_error::NativeError::internal(format!("response wire shape: {e}")))
}

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
        let resp = shrike_kernel::actions::collection_info(inner, &include, &note_type_details)?;
        wire(&resp)
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
        wire(&resp)
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
        wire(&resp)
    })
    .map_err(to_py_err)
}

/// `search_notes`: the whole fused-search assembly in the kernel. The
/// harness passes the live engine handles, one query vector per source when
/// semantic ranking is on, and the orchestrator state (image floor, index
/// size).
#[pyfunction]
#[pyo3(signature = (core, index_engine, derived_engine, sources, vectors, top_k, threshold, deck=None, tags=None, exclude=None, image_floor=None, weights=None, semantic=false, index_size=0, kernel=None, cross_space=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn action_search_notes(
    py: Python<'_>,
    core: PyRef<'_, CollectionCore>,
    index_engine: Option<PyRef<'_, crate::NativeIndexEngine>>,
    derived_engine: Option<PyRef<'_, crate::DerivedTextEngine>>,
    sources: Vec<(String, String, bool)>,
    vectors: Vec<Vec<f32>>,
    top_k: usize,
    threshold: f64,
    deck: Option<String>,
    tags: Option<Vec<String>>,
    exclude: Option<Vec<i64>>,
    image_floor: Option<f64>,
    weights: Option<std::collections::BTreeMap<String, f64>>,
    semantic: bool,
    index_size: usize,
    kernel: Option<PyRef<'_, crate::async_kernel::AsyncKernel>>,
    cross_space: Option<String>,
) -> PyResult<String> {
    // The tag-centroid state rides the kernel handle; cloned out so
    // the GIL-bound PyRef never crosses the detach.
    let tag_kernel = kernel.as_ref().map(|k| k.kernel_arc());
    let inner = core.core_ref();
    let index = index_engine
        .as_ref()
        .map(|e| &*e.inner as &dyn shrike_store::VectorIndex);
    let derived = derived_engine
        .as_ref()
        .map(|e| &e.inner as &dyn shrike_store::DerivedStore);
    let sources: Vec<shrike_kernel::actions::SearchSource> = sources
        .into_iter()
        .map(
            |(label, text, is_query)| shrike_kernel::actions::SearchSource {
                label,
                text,
                is_query,
            },
        )
        .collect();
    // Cross-space inputs: the host pre-built these via
    // `build_cross_space_json` (embed on the kernel runtime) and threads the
    // JSON in here. `None`/empty (the N=1 case) → no secondary spaces.
    let cross_space: Vec<shrike_kernel::actions::SpaceSemantic> = match cross_space {
        Some(s) if !s.is_empty() => serde_json::from_str(&s)
            .map_err(|e| shrike_error::NativeError::invalid_input(format!("cross_space: {e}")))
            .map_err(to_py_err)?,
        _ => Vec::new(),
    };
    // Experiment knobs (TEST-ONLY, like `disable_cross_space_gate`):
    // the eval harness selects a cross-space fusion variant + τ + the
    // vision-weight budget via env vars so the MCP tool schema is unchanged and
    // production stays on the `FloorAdmit` default. Unset → production
    // floor-admission exactly.
    let (cross_space_fusion_mode, cross_space_tau, cross_space_budget) =
        cross_space_fusion_from_env();
    let args = shrike_kernel::actions::SearchArgs {
        top_k,
        threshold,
        deck,
        tags: tags.unwrap_or_default(),
        exclude: exclude.unwrap_or_default(),
        image_floor,
        weights: weights.unwrap_or_default(),
        semantic,
        index_size,
        hidden_lexical_sources: shrike_kernel::Kernel::hidden_lexical_sources()
            .into_iter()
            .map(str::to_string)
            .collect(),
        cross_space,
        disable_cross_space_gate: false,
        cross_space_fusion_mode,
        cross_space_tau,
        cross_space_budget,
    };
    py.detach(|| {
        use shrike_kernel::actions::{is_stale_read, FreshnessStamp};
        let tag_keys = tag_kernel.as_ref().map(|k| k.tag_keys());
        // Freshness bracket: this whole closure runs inside the one
        // collection-actor job (the host dispatches via wrapper.run →
        // run_job), so capturing the stamps here and there spans the ENTIRE
        // read — the collection scan AND the Arc-shared index/derived reads the
        // ingest actor writes concurrently. `settled` reads the ingest
        // `outstanding == 0` gauge off the kernel handle; `true` when no kernel
        // rode in (a facade), leaving only the col_mod stability check.
        let settled = || tag_kernel.as_ref().is_none_or(|k| k.is_settled());
        let start = FreshnessStamp {
            col_mod: inner.col_mod()?,
            settled: settled(),
        };
        let groups = shrike_kernel::actions::search_notes(
            inner, index, derived, tag_keys, &sources, &vectors, &args,
        )?;
        let end = FreshnessStamp {
            col_mod: inner.col_mod()?,
            settled: settled(),
        };
        serde_json::to_string(&SearchNotesWire {
            groups,
            stale: is_stale_read(start, end),
        })
        .map_err(|e| shrike_error::NativeError::internal(e.to_string()))
    })
    .map_err(to_py_err)
}

/// The `action_search_notes` wire payload: the fused groups plus the read-time
/// freshness verdict the host threads onto `SearchResponse.stale`.
#[derive(serde::Serialize)]
struct SearchNotesWire {
    groups: Vec<shrike_schemas::SearchResultGroup>,
    stale: bool,
}

/// Resolve the cross-space fusion variant + τ + budget from the environment
/// (`SHRIKE_CROSS_SPACE_FUSION_MODE` ∈ {floor_admit, relative, relative_floor,
/// soft_relative, soft_calibrated, floor_admit_budget, soft_floor_admit,
/// soft_floor_admit_budget}; `SHRIKE_CROSS_SPACE_TAU` a float;
/// `SHRIKE_CROSS_SPACE_BUDGET` a float). The PRODUCTION default (unset) is
/// `FloorAdmit`; the env seam is the EVAL escape hatch the search-quality
/// sweep uses to reproduce the historical decision tables (e.g. `relative_floor`
/// for the earlier production gate, `relative` for the earlier leak). An
/// unrecognized value falls back to the production default (floor-admit) so a
/// typo can never silently weaken the path. The MCP tool schema is unchanged.
fn cross_space_fusion_from_env() -> (shrike_kernel::actions::CrossSpaceFusionMode, f64, f64) {
    use shrike_kernel::actions::CrossSpaceFusionMode as M;
    let mode = match std::env::var("SHRIKE_CROSS_SPACE_FUSION_MODE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        // The historical eval modes — reproduce the decision tables.
        Some("relative") => M::Relative,
        Some("relative_floor") => M::RelativeFloor,
        Some("soft_relative") => M::SoftRelative,
        Some("soft_calibrated") => M::SoftCalibrated,
        // Floor-admission family (production + the dominated/measurement
        // siblings).
        Some("floor_admit") => M::FloorAdmit,
        Some("floor_admit_budget") => M::FloorAdmitBudget,
        Some("soft_floor_admit") => M::SoftFloorAdmit,
        Some("soft_floor_admit_budget") => M::SoftFloorAdmitBudget,
        // "", unset, or anything unrecognized → the shipped production default
        // (floor-admit).
        _ => M::default(),
    };
    let tau = std::env::var("SHRIKE_CROSS_SPACE_TAU")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.05);
    // The vision-weight budget B; `<= 0` / unset reads as the canonical 1.0
    // (`SearchArgs::effective_budget`), so a single fired space keeps full weight.
    let budget = std::env::var("SHRIKE_CROSS_SPACE_BUDGET")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(1.0);
    (mode, tau, budget)
}
