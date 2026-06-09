//! Trivial PyO3 extension — the Bazel polyglot proof (#247).
//!
//! Exists only to prove the cross-language seam end to end: a Rust crate built
//! by rules_rust, linked against PyO3 (abi3, extension-module), packaged as a
//! `_demo.so` importable from a Bazel `py_test`. This is the exact mechanism the
//! native-backend epics (#219 compiled extension, #224 PyO3 host) build on. It
//! has no Shrike logic and ships in no wheel.

use pyo3::prelude::*;

/// Add two integers — the simplest possible round-trip through the FFI boundary.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Return a string that names the build, proving non-trivial marshaling and that
/// this is genuinely native code (reports the compiled-in target arch).
#[pyfunction]
fn backend_info() -> String {
    format!(
        "shrike-native-demo (pyo3) on {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

/// The module init. Its name MUST match the imported module / the `.so` filename
/// (`_demo`), since PyO3 exports `PyInit__demo` from it.
#[pymodule]
fn _demo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    Ok(())
}
