//! `shrike_native._native` — the Shrike PyO3 binding module (#269).
//!
//! The ONE crate that depends on `pyo3` (epic #265 convention 5, enforced by
//! `//native:layering_check`). Every native compute crate (pure Rust) is bound
//! to Python here, following the `shrike-ffi` conventions:
//!
//! - coarse, batched calls; only strings, bytes, f32 vectors, i64 key arrays,
//!   and small JSON-able dicts cross the boundary
//! - all compute under `py.detach` (GIL released; pyo3 ≥0.26 name for allow_threads)
//! - `shrike_ffi::NativeError` kinds map to the exception classes below, which
//!   the Python facades translate into Shrike's existing error surface
//!
//! The module is internal: production code reaches it only through the
//! `shrike_native` package facade, and **no test file imports it** — tests go
//! through the Python facades (`OnnxBackend`, `VectorIndex`, ...), which stay
//! plain (patchable) Python classes.
//!
//! The `parallel_sum`/`checked_div` functions are the conventions' permanent,
//! executable exemplar (and the stubtest fodder): one GIL-released batched
//! compute call, one error-taxonomy round-trip.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use shrike_ffi::{ErrorKind, NativeError};

pyo3::create_exception!(
    _native,
    NativeInputError,
    pyo3::exceptions::PyValueError,
    "Expected bad input crossed the FFI (shrike_ffi ErrorKind::InvalidInput). \
     Facades translate this into the Python-side input-error surface; logged \
     without a traceback."
);
pyo3::create_exception!(
    _native,
    NativeUnavailableError,
    PyRuntimeError,
    "A native runtime dependency isn't up (ErrorKind::Unavailable): model not \
     loaded, backend stopped, file missing."
);
pyo3::create_exception!(
    _native,
    NativeInternalError,
    PyRuntimeError,
    "A native-side bug (ErrorKind::Internal). Logged with a traceback."
);

/// Map the shared native error taxonomy onto the module's exception classes.
fn to_py_err(e: NativeError) -> PyErr {
    match e.kind {
        ErrorKind::InvalidInput => NativeInputError::new_err(e.message),
        ErrorKind::Unavailable => NativeUnavailableError::new_err(e.message),
        ErrorKind::Internal => NativeInternalError::new_err(e.message),
    }
}

/// The native package version (the Cargo workspace version).
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Name the build target, proving this is genuinely native code.
#[pyfunction]
fn build_info() -> String {
    format!(
        "shrike-py (pyo3 abi3) on {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

/// Conventions exemplar: a coarse, batched compute call with the GIL released.
#[pyfunction]
fn parallel_sum(py: Python<'_>, values: Vec<f64>) -> f64 {
    py.detach(move || values.iter().sum())
}

/// Conventions exemplar: the error-taxonomy round-trip (InvalidInput on b == 0).
#[pyfunction]
fn checked_div(py: Python<'_>, a: f64, b: f64) -> PyResult<f64> {
    py.detach(move || {
        if b == 0.0 {
            Err(NativeError::invalid_input("division by zero"))
        } else {
            Ok(a / b)
        }
    })
    .map_err(to_py_err)
}

/// The module init. Its name MUST match the imported module / the `.so`
/// filename (`_native`), since PyO3 exports `PyInit__native` from it.
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_sum, m)?)?;
    m.add_function(wrap_pyfunction!(checked_div, m)?)?;
    m.add("NativeInputError", py.get_type::<NativeInputError>())?;
    m.add("NativeUnavailableError", py.get_type::<NativeUnavailableError>())?;
    m.add("NativeInternalError", py.get_type::<NativeInternalError>())?;
    Ok(())
}
