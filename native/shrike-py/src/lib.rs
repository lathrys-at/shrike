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

// ── Embedding (#270) ─────────────────────────────────────────────────────────

/// Point the ort runtime at a specific onnxruntime shared library (process-wide,
/// idempotent). The facade passes the dylib from the installed onnxruntime
/// Python wheel, so native and Python backends run the same runtime build.
#[pyfunction]
fn init_onnx_runtime(py: Python<'_>, dylib_path: String) -> PyResult<()> {
    py.detach(move || shrike_embed::init_runtime(&dylib_path))
        .map_err(to_py_err)
}

/// The native ONNX text-embedding engine under the `OnnxBackend` facade.
///
/// Coarse, batched calls (one `embed_chunk` per chunk), GIL released for the
/// whole tokenize→run→pool pipeline. Construction loads the session+tokenizer;
/// dropping the object frees them (the facade's `stop()` just drops its
/// reference).
#[pyclass(frozen)]
struct OnnxTextEmbedder {
    inner: shrike_embed::TextEmbedder,
}

#[pymethods]
impl OnnxTextEmbedder {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path, *, providers, pooling, normalize, max_length))]
    fn new(
        py: Python<'_>,
        model_path: String,
        tokenizer_path: String,
        providers: Vec<String>,
        pooling: &str,
        normalize: bool,
        max_length: usize,
    ) -> PyResult<Self> {
        let pooling = shrike_embed::Pooling::parse(pooling).map_err(to_py_err)?;
        let cfg = shrike_embed::TextEmbedderConfig {
            model_path,
            tokenizer_path,
            providers,
            pooling,
            normalize,
            max_length,
        };
        let inner = py
            .detach(move || shrike_embed::TextEmbedder::load(cfg))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Embed one chunk of texts as a single batch (one vector per input).
    fn embed_chunk(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.inner.embed_chunk(&texts)).map_err(to_py_err)
    }

    /// The embedding width, once known (set by the first embed).
    fn dim(&self) -> Option<usize> {
        self.inner.dim()
    }

    /// Execution providers actually registered on the session, in order.
    fn active_providers(&self) -> Vec<String> {
        self.inner.active_providers().to_vec()
    }

    /// Graph inputs outside the supplied sentence-transformers set (diagnostic).
    fn unsupported_inputs(&self) -> Vec<String> {
        self.inner.unsupported_inputs().to_vec()
    }
}

/// The native CLIP dual-encoder engine under the `ClipBackend` facade (#271).
///
/// Image bytes (PNG/JPEG/...) in, vectors out — preprocessing (decode, resize,
/// center-crop, normalize) runs crate-side via the `image` crate, with the
/// whole pipeline GIL-released.
#[pyclass(frozen)]
struct ClipEmbedder {
    inner: shrike_embed::ClipEmbedder,
}

#[pymethods]
impl ClipEmbedder {
    #[new]
    #[pyo3(signature = (text_model_path, vision_model_path, tokenizer_path, *, providers, image_mean, image_std, resize, crop, context))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        text_model_path: String,
        vision_model_path: String,
        tokenizer_path: String,
        providers: Vec<String>,
        image_mean: Vec<f32>,
        image_std: Vec<f32>,
        resize: u32,
        crop: u32,
        context: usize,
    ) -> PyResult<Self> {
        let cfg = shrike_embed::ClipEmbedderConfig {
            text_model_path,
            vision_model_path,
            tokenizer_path,
            providers,
            image_mean,
            image_std,
            resize,
            crop,
            context,
        };
        let inner = py
            .detach(move || shrike_embed::ClipEmbedder::load(cfg))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn embed_text_chunk(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.inner.embed_text_chunk(&texts)).map_err(to_py_err)
    }

    /// Embed one chunk of images, each given as encoded bytes.
    fn embed_image_chunk(
        &self,
        py: Python<'_>,
        images: Vec<Vec<u8>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.inner.embed_image_chunk(&images)).map_err(to_py_err)
    }

    fn dim(&self) -> Option<usize> {
        self.inner.dim()
    }

    fn active_providers(&self) -> Vec<String> {
        self.inner.active_providers().to_vec()
    }
}

/// The module init. Its name MUST match the imported module / the `.so`
/// filename (`_native`), since PyO3 exports `PyInit__native` from it.
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_sum, m)?)?;
    m.add_function(wrap_pyfunction!(checked_div, m)?)?;
    m.add_function(wrap_pyfunction!(init_onnx_runtime, m)?)?;
    m.add_class::<OnnxTextEmbedder>()?;
    m.add_class::<ClipEmbedder>()?;
    // The native image-prep pipeline version — folded into the clip-rs
    // fingerprint by the facade (a pixel-math change must invalidate vectors).
    m.add("IMAGE_PREP_VERSION_RS", shrike_embed::IMAGE_PREP_VERSION_RS)?;
    m.add("NativeInputError", py.get_type::<NativeInputError>())?;
    m.add("NativeUnavailableError", py.get_type::<NativeUnavailableError>())?;
    m.add("NativeInternalError", py.get_type::<NativeInternalError>())?;
    Ok(())
}
