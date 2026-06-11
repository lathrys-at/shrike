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

#[cfg(feature = "anki-core")]
mod anki_core;

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
pyo3::create_exception!(
    _native,
    NativeBusyError,
    PyRuntimeError,
    "The collection is held by another process (ErrorKind::Busy — lock \
     contention, usually Anki desktop). Expected and retryable; the facades \
     map it onto the existing CollectionBusyError surface."
);

/// Map the shared native error taxonomy onto the module's exception classes.
/// The captured Rust span trace rides along as a PEP 678 note (#308), so the
/// native context shows up in the Python traceback the Pythonic way.
pub(crate) fn to_py_err(e: NativeError) -> PyErr {
    let trace = e.trace();
    let err = match e.kind {
        ErrorKind::InvalidInput => NativeInputError::new_err(e.message),
        ErrorKind::Unavailable => NativeUnavailableError::new_err(e.message),
        ErrorKind::Busy => NativeBusyError::new_err(e.message),
        ErrorKind::Internal => NativeInternalError::new_err(e.message),
    };
    if let Some(trace) = trace {
        Python::attach(|py| {
            let value = err.value(py);
            let _ = value.call_method1("add_note", (format!("native trace:\n{trace}"),));
        });
    }
    err
}

/// Rig native observability into the Python host (#308/#310) with the
/// first-class bridges, not a hand-rolled forwarder:
///
/// - `pyo3_log::try_init()` installs the `log` crate's global logger,
///   forwarding every record to the stdlib `logging` module — logger name =
///   the Rust target (`shrike_derived`, ...), levels mapped, Python-side
///   filtering respected (with pyo3-log's level caching, so call this *after*
///   the host configures `logging`).
/// - the native crates instrument with `tracing` only; the `log-always`
///   compat feature (enabled in this crate's Cargo.toml, unified across the
///   build graph) makes every tracing event also emit a `log` record even
///   with a tracing subscriber installed, so events reach Python `logging`
///   through the two established bridges chained together.
/// - a `tracing-error` `ErrorLayer` registry stays installed as the global
///   tracing subscriber, so `NativeError` span traces (the PEP 678 notes on
///   the exceptions above) keep working.
///
/// Idempotent: both globals are first-install-wins; later calls are no-ops.
/// A non-Python host (mobile) skips this and installs its own `tracing`
/// subscriber instead — the kernel/compute crates never log directly.
#[pyfunction]
fn init_logging() -> PyResult<()> {
    use tracing_subscriber::layer::SubscriberExt;

    let _ = pyo3_log::try_init();
    let subscriber = tracing_subscriber::registry().with(tracing_error::ErrorLayer::default());
    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}

/// The Rust-canonical wire contracts (#330): `{python_model_name: json_schema}`
/// for every type in shrike-schemas -- the contract test's Rust side.
#[pyfunction]
fn schema_catalog() -> std::collections::HashMap<&'static str, String> {
    shrike_schemas::schema_catalog().into_iter().collect()
}

/// Deserialize `json` as the named wire type and re-serialize it through the
/// Rust types -- the instance-level wire-parity probe. Raises NativeInputError
/// for an unknown name or a payload the type rejects.
#[pyfunction]
fn schema_roundtrip(name: &str, json: &str) -> PyResult<String> {
    shrike_schemas::roundtrip(name, json).map_err(NativeInputError::new_err)
}

/// The SSRF-guarded media URL fetch (#278 step 5b) standalone: the facades'
/// CONCURRENT prepare path downloads off the collection worker thread, then
/// writes bytes through it — same architecture as the Python original.
#[cfg(feature = "anki-core")]
#[pyfunction]
#[pyo3(signature = (url, allow_private=false))]
fn fetch_media_url(
    py: Python<'_>,
    url: String,
    allow_private: bool,
) -> PyResult<(Vec<u8>, Option<String>)> {
    py.detach(|| shrike_collection::media_fetch::fetch_media_url(&url, allow_private))
        .map_err(to_py_err)
}

/// Base64 media decode with the size cap applied to the ENCODED length.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn decode_media_b64(py: Python<'_>, data: String) -> PyResult<Vec<u8>> {
    py.detach(|| shrike_collection::media_fetch::decode_media_b64(&data))
        .map_err(to_py_err)
}

/// One derived-store MATCH row: (note_id, source, ref, txt, snippet).
type MatchRow = (i64, String, String, Option<String>, Option<String>);
/// Per-query, per-modality parallel rankings: {modality: (note_ids, distances)}.
type ModalityRankings = Vec<std::collections::BTreeMap<String, (Vec<i64>, Vec<f32>)>>;
/// One fused hit: (note_id, score, [(signal, 1-based rank)...]).
type FusedHit = (i64, f64, Vec<(String, i64)>);

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
        py.detach(|| self.inner.embed_chunk(&texts))
            .map_err(to_py_err)
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
        py.detach(|| self.inner.embed_text_chunk(&texts))
            .map_err(to_py_err)
    }

    /// Embed one chunk of images, each given as encoded bytes.
    fn embed_image_chunk(&self, py: Python<'_>, images: Vec<Vec<u8>>) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.inner.embed_image_chunk(&images))
            .map_err(to_py_err)
    }

    fn dim(&self) -> Option<usize> {
        self.inner.dim()
    }

    fn active_providers(&self) -> Vec<String> {
        self.inner.active_providers().to_vec()
    }
}

// ── Derived-text engine (#281) ──────────────────────────────────────────────

/// Whether the linked SQLite has FTS5 with the trigram tokenizer (#300).
/// Constant-true under the bundled default; genuinely probed under platform
/// linkage (a build without the bundled-sqlite feature).
#[pyfunction]
fn derived_fts5_probe() -> bool {
    shrike_derived::fts5_trigram_available()
}

/// Whether this build statically links the bundled SQLite (#300) — for
/// status/diagnostics.
#[pyfunction]
fn derived_sqlite_bundled() -> bool {
    shrike_derived::sqlite_bundled()
}

/// The native FTS5-trigram derived-text engine under the `DerivedTextStore`
/// facade — rusqlite with a bundled SQLite, so FTS5 + trigram are always
/// available. Storage + MATCH queries only; expression building, filtering,
/// and the state machine stay in the facade.
#[pyclass(frozen)]
struct DerivedTextEngine {
    inner: shrike_derived::DerivedEngine,
}

#[pymethods]
impl DerivedTextEngine {
    #[new]
    fn new(py: Python<'_>, path: String, schema_version: i64) -> PyResult<Self> {
        let inner = py
            .detach(move || shrike_derived::DerivedEngine::open(&path, schema_version))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn close(&self) {
        // Dropping happens when the Python object is collected; the facade's
        // close() just stops using it. Nothing to do eagerly — rusqlite closes
        // on drop — but the method keeps the engine surfaces aligned.
    }

    fn get_col_mod(&self) -> Option<i64> {
        self.inner.get_col_mod()
    }

    fn set_col_mod(&self, value: i64) -> PyResult<()> {
        self.inner.set_col_mod(value).map_err(to_py_err)
    }

    fn count(&self) -> i64 {
        self.inner.count()
    }

    fn ingest(
        &self,
        py: Python<'_>,
        note_id: i64,
        source: &str,
        refs_text: Vec<(String, String)>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.ingest(note_id, source, &refs_text))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (note_ids, source=None))]
    fn remove(&self, py: Python<'_>, note_ids: Vec<i64>, source: Option<String>) -> PyResult<()> {
        py.detach(|| self.inner.remove(&note_ids, source.as_deref()))
            .map_err(to_py_err)
    }

    fn build(
        &self,
        py: Python<'_>,
        rows: Vec<(i64, String, String, String)>,
        col_mod: i64,
    ) -> PyResult<()> {
        py.detach(|| self.inner.build(&rows, col_mod))
            .map_err(to_py_err)
    }

    fn match_rows(
        &self,
        py: Python<'_>,
        expr: String,
        limit: i64,
        with_text: bool,
    ) -> PyResult<Vec<MatchRow>> {
        py.detach(|| self.inner.match_rows(&expr, limit, with_text))
            .map_err(to_py_err)
    }
}

// ── Index engine (#273) ─────────────────────────────────────────────────────

/// The native per-modality vector index engine under the `VectorIndex`
/// orchestrator (the frozen #267 `IndexEngine` surface). Coarse, batched calls
/// trafficking in i64 key arrays and f32 vector batches, all GIL-released.
#[pyclass(frozen)]
struct NativeIndexEngine {
    inner: shrike_index::MultiModalIndex,
}

#[pymethods]
impl NativeIndexEngine {
    #[new]
    fn new(modalities: Vec<String>) -> PyResult<Self> {
        let inner = shrike_index::MultiModalIndex::new(modalities).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn ndim(&self) -> Option<usize> {
        self.inner.ndim()
    }

    fn modality_sizes(&self) -> Vec<(String, usize)> {
        self.inner.modality_sizes()
    }

    fn modality_names(&self) -> Vec<String> {
        self.inner.modality_names()
    }

    fn ensure(&self, modality: &str, ndim: usize) -> PyResult<()> {
        self.inner.ensure(modality, ndim).map_err(to_py_err)
    }

    fn clear(&self) {
        self.inner.clear()
    }

    fn drop_modality(&self, modality: &str) {
        self.inner.drop_modality(modality)
    }

    #[pyo3(signature = (path, candidate_keys=None))]
    fn restore(&self, py: Python<'_>, path: String, candidate_keys: Option<Vec<i64>>) -> bool {
        py.detach(move || self.inner.restore(&path, candidate_keys.as_deref()))
    }

    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(move || self.inner.save(&path)).map_err(to_py_err)
    }

    fn add(
        &self,
        py: Python<'_>,
        modality: &str,
        keys: Vec<i64>,
        vectors: Vec<Vec<f32>>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.add(modality, &keys, &vectors))
            .map_err(to_py_err)
    }

    fn remove(&self, py: Python<'_>, keys: Vec<i64>) -> PyResult<usize> {
        py.detach(|| self.inner.remove(&keys)).map_err(to_py_err)
    }

    /// Per-query `{modality: (note_ids, distances)}` rankings (parallel arrays;
    /// the Python adapter zips them into the protocol's dict shape).
    #[pyo3(signature = (queries, k, modalities=None))]
    fn search_by_modality(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<f32>>,
        k: usize,
        modalities: Option<Vec<String>>,
    ) -> PyResult<ModalityRankings> {
        py.detach(|| {
            self.inner
                .search_by_modality(&queries, k, modalities.as_deref())
        })
        .map_err(to_py_err)
    }

    fn contains(&self, key: i64) -> bool {
        self.inner.contains(key)
    }

    fn keys(&self) -> Vec<i64> {
        self.inner.keys()
    }

    fn get(&self, key: i64) -> Option<Vec<Vec<f32>>> {
        self.inner.get(key)
    }

    fn modality_contains(&self, modality: &str, key: i64) -> bool {
        self.inner.modality_contains(modality, key)
    }

    fn modality_keys(&self, modality: &str) -> Vec<i64> {
        self.inner.modality_keys(modality)
    }

    fn modality_get(&self, modality: &str, key: i64) -> Option<Vec<Vec<f32>>> {
        self.inner.modality_get(modality, key)
    }

    fn calibrate_activation(
        &self,
        py: Python<'_>,
        sample_size: usize,
        k: usize,
        min_count: usize,
    ) -> PyResult<Vec<(String, f64, f64, f64)>> {
        py.detach(|| self.inner.calibrate_activation(sample_size, k, min_count))
            .map_err(to_py_err)
    }
}

// ── Fused compute (#274) ────────────────────────────────────────────────────

/// Reciprocal Rank Fusion — the native implementation of the frozen
/// `search_fusion.py` spec. Same canonical accumulation order, same dedup, same
/// `(tier, -score, note_id)` ordering; the Python parity property suite pins
/// the two byte-for-byte.
#[pyfunction]
#[pyo3(signature = (rankings, weights, k=60, priority_signals=vec![]))]
fn rrf_fuse(
    py: Python<'_>,
    rankings: Vec<(String, Vec<i64>)>,
    weights: std::collections::BTreeMap<String, f64>,
    k: i64,
    priority_signals: Vec<String>,
) -> Vec<FusedHit> {
    py.detach(move || {
        let priority: std::collections::HashSet<String> = priority_signals.into_iter().collect();
        shrike_compute::rrf_fuse(&rankings, &weights, k, &priority)
    })
}

/// Embed query texts and rank per modality — one GIL-released composition over
/// the native text embedder + index engine; the vectors never cross the FFI.
#[pyfunction]
#[pyo3(signature = (embedder, engine, texts, k, modalities=None))]
fn fused_search_text(
    py: Python<'_>,
    embedder: Py<OnnxTextEmbedder>,
    engine: Py<NativeIndexEngine>,
    texts: Vec<String>,
    k: usize,
    modalities: Option<Vec<String>>,
) -> PyResult<ModalityRankings> {
    let e = embedder.get();
    let ix = engine.get();
    py.detach(|| {
        shrike_compute::fused_search(&e.inner, &ix.inner, &texts, k, modalities.as_deref())
    })
    .map_err(to_py_err)
}

/// Embed note texts and replace-add them under their ids — one GIL-released
/// composition; the vectors never cross the FFI. Returns the count added.
#[pyfunction]
fn fused_add_text(
    py: Python<'_>,
    embedder: Py<OnnxTextEmbedder>,
    engine: Py<NativeIndexEngine>,
    modality: String,
    keys: Vec<i64>,
    texts: Vec<String>,
    chunk: usize,
) -> PyResult<usize> {
    let e = embedder.get();
    let ix = engine.get();
    py.detach(|| shrike_compute::fused_add(&e.inner, &ix.inner, &modality, &keys, &texts, chunk))
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
    m.add_function(wrap_pyfunction!(init_onnx_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    #[cfg(feature = "anki-core")]
    {
        m.add_function(wrap_pyfunction!(fetch_media_url, m)?)?;
        m.add_function(wrap_pyfunction!(decode_media_b64, m)?)?;
    }
    m.add_class::<OnnxTextEmbedder>()?;
    m.add_class::<ClipEmbedder>()?;
    m.add_class::<NativeIndexEngine>()?;
    m.add_class::<DerivedTextEngine>()?;
    // Feature-gated (#278): present only in `anki-core` builds; the stubtest
    // allowlist covers its absence from default builds.
    #[cfg(feature = "anki-core")]
    m.add_class::<anki_core::CollectionCore>()?;
    m.add_function(wrap_pyfunction!(derived_fts5_probe, m)?)?;
    m.add_function(wrap_pyfunction!(derived_sqlite_bundled, m)?)?;
    m.add_function(wrap_pyfunction!(rrf_fuse, m)?)?;
    m.add_function(wrap_pyfunction!(schema_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(schema_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(fused_search_text, m)?)?;
    m.add_function(wrap_pyfunction!(fused_add_text, m)?)?;
    // The native image-prep pipeline version — folded into the clip-rs
    // fingerprint by the facade (a pixel-math change must invalidate vectors).
    m.add("IMAGE_PREP_VERSION_RS", shrike_embed::IMAGE_PREP_VERSION_RS)?;
    m.add("NativeInputError", py.get_type::<NativeInputError>())?;
    m.add(
        "NativeUnavailableError",
        py.get_type::<NativeUnavailableError>(),
    )?;
    m.add("NativeInternalError", py.get_type::<NativeInternalError>())?;
    m.add("NativeBusyError", py.get_type::<NativeBusyError>())?;
    Ok(())
}
