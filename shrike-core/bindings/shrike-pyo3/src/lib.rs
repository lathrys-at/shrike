//! `shrike_native._native` — the Shrike PyO3 binding module.
//!
//! The ONE crate that depends on `pyo3` (convention 5, enforced by
//! `//shrike-core:layering_check`). Every native compute crate (pure Rust) is bound
//! to Python here.
//!
//! # Marshaling rules
//!
//! Only these types cross the Python↔Rust boundary:
//!
//! - strings (`String`/`&str`) and byte buffers (`Vec<u8>`/`&[u8]`)
//! - f32 vectors / vector batches (zero-copy numpy interchange where arrays
//!   must cross, via the `numpy` crate)
//! - i64 key arrays
//! - small JSON-able maps (stats, health blocks)
//!
//! Never a live Python object, callback, or handle — calls are coarse and
//! batched so the boundary is crossed per *batch*, not per item.
//!
//! # Threading rules
//!
//! - All compute runs under `py.detach` (GIL released; pyo3 ≥0.26's name for
//!   `allow_threads`).
//! - No Python handle may cross into a worker thread; compute crates receive
//!   owned data only.
//!
//! # Error taxonomy
//!
//! `shrike_error::NativeError`'s [`ErrorKind`] maps to the exception classes
//! below ([`to_py_err`]), which the Python facades translate into Shrike's
//! existing error surface. The wire shape is `(kind, message, trace)`: the kind
//! selects the class, the message is the text, and the captured native span
//! trace rides along as a PEP 678 note. The error's Rust-side `#[source]` chain
//! does NOT cross — it's for native logging/`{:?}` only.
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
use shrike_error::{ErrorKind, NativeError};

#[cfg(feature = "anki-core")]
mod anki_core;
#[cfg(feature = "anki-core")]
mod async_kernel;
mod asyncio_bridge;
// Injects the kernel's compute pool into the engine `Blocking` adapter; only
// the route-1 sync engines (onnx/CLIP, Apple Vision, synthetic) use it.
#[cfg(any(
    feature = "engine-ort",
    feature = "engine-apple",
    feature = "engine-synthetic"
))]
mod compute_dispatch;
mod finalize_gate;
mod gated_log;
#[cfg(feature = "anki-core")]
mod kernel_actions;
// The engine containers/capture handles exist to be ATTACHED to a kernel;
// a compute-only build keeps them constructible (the pyclasses are
// part of the module surface) but never reads the attach-path members.
#[cfg_attr(not(feature = "anki-core"), allow(dead_code))]
mod native_embedder;
#[cfg_attr(not(feature = "anki-core"), allow(dead_code))]
mod py_embedder;
#[cfg_attr(not(feature = "anki-core"), allow(dead_code))]
mod py_recognizer;

pyo3::create_exception!(
    _native,
    NativeInputError,
    pyo3::exceptions::PyValueError,
    "Expected bad input crossed the FFI (shrike_error ErrorKind::InvalidInput). \
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
/// The captured Rust span trace rides along as a PEP 678 note, so the
/// native context shows up in the Python traceback the Pythonic way.
pub(crate) fn to_py_err(e: NativeError) -> PyErr {
    let trace = e.trace();
    // ErrorKind is a closed set, so this match is exhaustive: adding a kind is
    // a deliberate compile error here, forcing a new exception mapping.
    let err = match e.kind() {
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

/// Rig native observability into the Python host with the
/// first-class bridges, not a hand-rolled forwarder:
///
/// - a `pyo3_log::Logger` is installed as the `log` crate's global logger,
///   forwarding every record to the stdlib `logging` module — logger name =
///   the Rust target (`shrike_derived`, ...), levels mapped, Python-side
///   filtering respected (with pyo3-log's level caching, so call this *after*
///   the host configures `logging`). It is wrapped in [`gated_log::GatedLog`]:
///   a record emitted from a kernel-runtime thread attaches the GIL
///   inside pyo3-log, so each delivery claims a finalization-gate permit and
///   a record racing interpreter exit is dropped instead of straddling
///   `Py_Finalize` (the abort class). Construction mirrors
///   `pyo3_log::try_init()` — `Caching::LoggersAndLevels`, `Debug` default
///   filter, matching `log::set_max_level` — just with the gate in front.
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
fn init_logging(py: Python<'_>) -> PyResult<()> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Layer;

    let logger = pyo3_log::Logger::new(py, pyo3_log::Caching::LoggersAndLevels)?;
    let gated = gated_log::GatedLog::new(finalize_gate::process_gate(), logger);
    if log::set_boxed_logger(Box::new(gated)).is_ok() {
        // try_init() parity: the max level is the logger's default Debug
        // filter (no per-target filters are configured here).
        log::set_max_level(log::LevelFilter::Debug);
    }
    // Dev fmt layer: opt-in via SHRIKE_TRACE_FMT for local async-debug/benchmark
    // (span timings to stderr). Off by default; the pyo3-log bridge above is the
    // production path. Filter from SHRIKE_TRACE (RUST_LOG-style), default info.
    let fmt_layer = std::env::var_os("SHRIKE_TRACE_FMT").map(|_| {
        let filter = tracing_subscriber::EnvFilter::try_from_env("SHRIKE_TRACE")
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_filter(filter)
            .boxed()
    });
    let subscriber = tracing_subscriber::registry()
        .with(tracing_error::ErrorLayer::default())
        .with(fmt_layer)
        .with(otlp_trace_layer());
    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}

/// The OTLP trace-export seam — **off by default**. Installs a
/// `tracing-opentelemetry` layer only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
/// batch-exporting op-boundary spans over OTLP/gRPC to that collector. Any
/// init failure degrades to no export (logged) rather than failing startup. The
/// cross-machine-spans path (#796); cross-FFI context propagation is #967.
fn otlp_trace_layer<S>() -> Option<Box<dyn tracing_subscriber::Layer<S> + Send + Sync>>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
{
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig as _;
    use tracing_subscriber::Layer as _;

    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .ok()
        .filter(|e| !e.is_empty())?;
    let exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
    {
        Ok(exporter) => exporter,
        Err(err) => {
            tracing::warn!(
                ?err,
                "OTLP span exporter init failed; trace export disabled"
            );
            return None;
        }
    };
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();
    let tracer = provider.tracer("shrike");
    // Keep the provider alive for the process and expose it to the OTel global,
    // so a flush on shutdown can reach it.
    opentelemetry::global::set_tracer_provider(provider);
    Some(tracing_opentelemetry::layer().with_tracer(tracer).boxed())
}

/// The Rust-canonical wire contracts: `{python_model_name: json_schema}`
/// for every type in shrike-schemas -- the contract test's Rust side.
#[pyfunction]
fn schema_catalog() -> std::collections::HashMap<&'static str, String> {
    shrike_schemas::schema_catalog().into_iter().collect()
}

/// The action exchange's protocol version -- the contract test pins
/// the Python mirror equal; a future remote handshake checks it.
#[pyfunction]
fn wire_protocol_version() -> u32 {
    shrike_schemas::WIRE_PROTOCOL_VERSION
}

/// Deserialize `json` as the named wire type and re-serialize it through the
/// Rust types -- the instance-level wire-parity probe. Raises NativeInputError
/// for an unknown name or a payload the type rejects.
#[pyfunction]
fn schema_roundtrip(name: &str, json: &str) -> PyResult<String> {
    shrike_schemas::roundtrip(name, json).map_err(NativeInputError::new_err)
}

/// The SSRF-guarded media URL fetch standalone: the facades'
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
    // The async fetch runs off the runtime (under `py.detach`), driven to
    // completion via the kernel runtime's `block_on` (a legal block_on site —
    // not a runtime worker).
    py.detach(|| shrike_kernel::block_on(shrike_media::fetch_media_url(&url, allow_private)))
        .map_err(to_py_err)
}

/// Base64 media decode with the size cap applied to the ENCODED length.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn decode_media_b64(py: Python<'_>, data: String) -> PyResult<Vec<u8>> {
    py.detach(|| shrike_media::decode_media_b64(&data))
        .map_err(to_py_err)
}

// ── Driven runtime: the harness commits N+2 threads, shrike-core spawns none ──
// The server installs a current_thread runtime in DRIVEN mode at boot, then
// donates GIL-releasing threads to drive it: one IO thread, one collection/sync
// thread, and N compute threads. Each entry below releases the GIL (`py.detach`)
// and parks in the matching pure-Rust loop for the server's life; the asyncio
// bridge still submits ops, invisibly driven by the IO thread.

/// Install a `current_thread` runtime in the driven model — call ONCE, before
/// any kernel op (the kernel has no lazy fallback, so an op before this panics).
/// Returns whether the runtime is now installed: `True` once installed (a fresh
/// install, or a benign re-call where it was already installed — the seam is
/// set-once), `False` only if the runtime fails to build.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn init_driven_runtime(py: Python<'_>, compute_workers: usize) -> PyResult<bool> {
    py.detach(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| NativeError::internal(format!("building the driven runtime: {e}")))?;
        // An already-installed runtime (a second call in one process) returns
        // Err carrying the runtime back; the seam is set-once, so the first
        // install stands. `is_driven` reports whether that install was driven.
        // `compute_workers` is the harness's committed compute-pool width: the
        // pool provisions that many worker deques and the kernel fans parallel
        // work (the chunked lexical reads) across exactly that many workers.
        let _ = shrike_kernel::init_driven_runtime(runtime, compute_workers);
        Ok(shrike_kernel::is_driven())
    })
    .map_err(to_py_err)
}

/// Park this thread as the driven runtime's IO/timer driver until
/// [`drive_pools_shutdown`] is called. GIL released for the thread's life; the
/// asyncio loop submits ops that this thread polls.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn drive_io(py: Python<'_>) -> PyResult<()> {
    py.detach(shrike_kernel::drive_io_until_shutdown)
        .map_err(to_py_err)
}

/// Park this thread as the serialized collection execution thread until the
/// pools are shut down — every anki-collection op runs here. GIL released; a
/// non-runtime context, so anki's own `block_on` is legal here.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn drive_collection(py: Python<'_>) -> PyResult<()> {
    py.detach(shrike_kernel::drive_collection)
        .map_err(to_py_err)
}

/// Park this thread as one of the N CPU-compute workers until the pools are shut
/// down. GIL released, so native engine compute runs with real parallelism (the
/// capture seam re-acquires the GIL only for custom/test backends).
#[cfg(feature = "anki-core")]
#[pyfunction]
fn drive_compute(py: Python<'_>) -> PyResult<()> {
    py.detach(shrike_kernel::drive_compute).map_err(to_py_err)
}

/// Signal every committed driven thread to return so the harness can join them
/// — call after kernel work has quiesced (the collection actor drained). Closes
/// the pool queues and trips the IO thread's shutdown signal. GIL released so
/// the threads it wakes can finish without contending for it.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn drive_pools_shutdown(py: Python<'_>) {
    py.detach(shrike_kernel::shutdown_driven_pools)
}

/// Block until the driven runtime is being driven, by scheduling a trivial
/// executor-only op and blocking this thread until it completes. The harness
/// calls it after spawning the IO thread and BEFORE the sync/compute threads:
/// tokio's `current_thread` runtime hands the IO/timer drivers to the FIRST
/// `block_on` caller, which MUST be the IO thread, so the op completing — while
/// the IO thread is the only driver present — proves it owns the drivers.
///
/// The op touches nothing but the executor: the collection and compute pools are
/// not yet driven when the barrier runs. Implemented with `spawn_op` + a
/// `std::sync::mpsc` channel filled from the op body (rather than awaiting
/// `spawn_op`'s future, which has no driver to block on from this thread). GIL
/// released so the IO thread can run the op without contending for it.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn runtime_probe(py: Python<'_>) {
    py.detach(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
        drop(shrike_kernel::spawn_op(async move {
            let _ = tx.send(());
            Ok(())
        }));
        let _ = rx.recv();
    })
}

/// Render the kernel's Prometheus registry (runtime pools, embedding, index
/// saver) for the control-plane /metrics body — appended after the Python
/// registry. Disjoint `shrike_runtime_*`/`shrike_embedding_*`/`shrike_index_saver_*`
/// prefixes keep the concatenated exposition valid.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn render_prometheus() -> String {
    shrike_kernel::render_prometheus()
}

/// One derived-store MATCH row: (note_id, source, ref, txt, snippet).
type MatchRow = (i64, String, String, Option<String>, Option<String>);
/// One fuzzy lexical row marshalled for Python: `(note_id, source, ref, match)`
/// where `match` is `(segment text, (first byte, last byte))` or `None`. The
/// internal [`shrike_derived::LexicalSpan`] is flattened to a plain tuple here
/// (shrike-store carries no pyo3 dependency, so it has no `IntoPyObject` of its
/// own).
type PyLexicalRow = (i64, String, String, Option<(String, (usize, usize))>);

/// Flatten an internal [`shrike_derived::LexicalRow`] to its [`PyLexicalRow`]
/// Python tuple.
fn lexical_row_to_py(row: shrike_derived::LexicalRow) -> PyLexicalRow {
    let (note_id, source, r#ref, span) = row;
    (note_id, source, r#ref, span.map(|s| (s.text, s.span)))
}
/// Per-query, per-modality parallel rankings: {modality: (note_ids, distances)}.
type ModalityRankings = Vec<std::collections::BTreeMap<String, (Vec<i64>, Vec<f32>)>>;
/// One fused hit: (note_id, score, [(signal, 1-based rank)...]).
#[cfg(feature = "anki-core")]
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
        "shrike-pyo3 (pyo3 abi3) on {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

/// The build-matrix features compiled into this extension. The config
/// layer's capability source: a config entry declaring a `runtime`
/// whose feature is absent here is a config error naming the build profile —
/// never a silent no-op.
#[pyfunction]
fn build_features() -> Vec<&'static str> {
    [
        (cfg!(feature = "anki-core"), "anki-core"),
        (cfg!(feature = "engine-ort"), "engine-ort"),
        (cfg!(feature = "engine-remote"), "engine-remote"),
        (cfg!(feature = "engine-apple"), "engine-apple"),
        (cfg!(feature = "engine-synthetic"), "engine-synthetic"),
        (cfg!(feature = "manage-llama"), "manage-llama"),
        // Not a feature: a build-profile marker. Present on an unoptimized
        // (fastbuild/debug) build, absent under `-c opt`. The perf harness records
        // it so a debug-build latency is never mistaken for a release one.
        (cfg!(debug_assertions), "debug-assertions"),
    ]
    .into_iter()
    .filter_map(|(compiled, name)| compiled.then_some(name))
    .collect()
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

// ── Embedding ─────────────────────────────────────────────────────────
// The ort-backed engines (text + CLIP) are `engine-ort` builds only;
// the remote engine is `engine-remote`; the llama-server manager is
// `manage-llama`. The default feature set carries them all (the server
// profile), so existing builds are unchanged.

/// Point the ort runtime at a specific onnxruntime shared library (process-wide,
/// idempotent). The facade passes the dylib from the installed onnxruntime
/// Python wheel, so native and Python backends run the same runtime build.
#[cfg(feature = "engine-ort")]
#[pyfunction]
fn init_onnx_runtime(py: Python<'_>, dylib_path: String) -> PyResult<()> {
    py.detach(move || shrike_engine::onnx::init_runtime(&dylib_path))
        .map_err(to_py_err)
}

/// The native ONNX text-embedding engine under the `OnnxBackend` facade.
///
/// Coarse, batched calls (one `embed_chunk` per chunk), GIL released for the
/// whole tokenize→run→pool pipeline. Construction loads the session+tokenizer;
/// dropping the object frees them (the facade's `stop()` just drops its
/// reference).
#[cfg(feature = "engine-ort")]
#[pyclass(frozen)]
pub(crate) struct OnnxTextEmbedder {
    inner: std::sync::Arc<shrike_engine::onnx::TextEmbedder>,
}

#[cfg(feature = "engine-ort")]
impl OnnxTextEmbedder {
    /// The loaded engine, shared — `NativeEmbedder` composes the kernel-slot
    /// handle from the same instance the facade probed.
    pub(crate) fn engine_arc(&self) -> std::sync::Arc<shrike_engine::onnx::TextEmbedder> {
        std::sync::Arc::clone(&self.inner)
    }
}

#[cfg(feature = "engine-ort")]
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
        let pooling = shrike_engine::onnx::Pooling::parse(pooling).map_err(to_py_err)?;
        let cfg = shrike_engine::onnx::TextEmbedderConfig {
            model_path,
            tokenizer_path,
            providers,
            pooling,
            normalize,
            max_length,
        };
        let inner = py
            .detach(move || shrike_engine::onnx::TextEmbedder::load(cfg))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
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

/// The native CLIP dual-encoder engine under the `ClipBackend` facade.
///
/// Image bytes (PNG/JPEG/...) in, vectors out — preprocessing (decode, resize,
/// center-crop, normalize) runs crate-side via the `image` crate, with the
/// whole pipeline GIL-released.
#[cfg(feature = "engine-ort")]
#[pyclass(frozen)]
pub(crate) struct ClipEmbedder {
    inner: std::sync::Arc<shrike_engine::onnx::ClipEmbedder>,
}

#[cfg(feature = "engine-ort")]
impl ClipEmbedder {
    /// The loaded engine, shared (see [`OnnxTextEmbedder::engine_arc`]).
    pub(crate) fn engine_arc(&self) -> std::sync::Arc<shrike_engine::onnx::ClipEmbedder> {
        std::sync::Arc::clone(&self.inner)
    }
}

#[cfg(feature = "engine-ort")]
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
        let cfg = shrike_engine::onnx::ClipEmbedderConfig {
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
            .detach(move || shrike_engine::onnx::ClipEmbedder::load(cfg))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    fn embed_text_chunk(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.inner.embed_text_chunk(&texts))
            .map_err(to_py_err)
    }

    /// Embed one chunk of images, each given as encoded bytes.
    fn embed_image_chunk(&self, py: Python<'_>, images: Vec<Vec<u8>>) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.inner.embed_image_bytes_chunk(images))
            .map_err(to_py_err)
    }

    fn dim(&self) -> Option<usize> {
        self.inner.dim()
    }

    fn active_providers(&self) -> Vec<String> {
        self.inner.active_providers().to_vec()
    }
}

/// The generic remote-embeddings engine under the llama facade —
/// and the future `backend: remote` kind (URL + key, no subprocess). One
/// `/v1/embeddings` request per chunk, GIL released; identity ingredients
/// (`/v1/models` id + meta) served raw for the facade's fingerprint policy.
#[cfg(feature = "engine-remote")]
#[pyclass(frozen)]
pub(crate) struct RemoteEmbedder {
    inner: std::sync::Arc<shrike_engine::remote::RemoteEmbedder>,
}

#[cfg(feature = "engine-remote")]
impl RemoteEmbedder {
    /// The engine, shared (see the ort engines' `engine_arc`).
    pub(crate) fn engine_arc(&self) -> std::sync::Arc<shrike_engine::remote::RemoteEmbedder> {
        std::sync::Arc::clone(&self.inner)
    }
}

#[cfg(feature = "engine-remote")]
#[pymethods]
impl RemoteEmbedder {
    #[new]
    #[pyo3(signature = (base_url, *, api_key=None, model=None))]
    fn new(base_url: String, api_key: Option<String>, model: Option<String>) -> PyResult<Self> {
        // Construction validates the API key (header-injection guard).
        let engine = shrike_engine::remote::RemoteEmbedder::new(
            shrike_engine::remote::RemoteEmbedderConfig {
                base_url,
                api_key,
                model,
            },
        )
        .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Arc::new(engine),
        })
    }

    /// Embed one chunk of texts as a single request (one vector per input).
    /// The route-2 async engine is driven off the runtime (under `py.detach`)
    /// via `shrike_kernel::block_on` (a legal block_on site — not a runtime
    /// worker).
    fn embed_chunk(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        use shrike_engine_api::Embedder as _;
        py.detach(|| shrike_kernel::block_on(self.inner.embed(texts)))
            .map_err(to_py_err)
    }

    /// Embed one chunk of images via llama.cpp's native multimodal dialect —
    /// the direct path for the facade's `embed_images` / tests; the
    /// kernel slot rides the `NativeEmbedder` composition instead. Bytes in,
    /// one vector per image; vision-gated inside the engine.
    fn embed_image_chunk(&self, py: Python<'_>, images: Vec<Vec<u8>>) -> PyResult<Vec<Vec<f32>>> {
        use shrike_engine_api::ImageEmbedder as _;
        py.detach(|| {
            let items: Vec<shrike_engine_api::MediaItem> = images
                .into_iter()
                .map(shrike_engine_api::MediaItem::untyped)
                .collect();
            shrike_kernel::block_on(self.inner.embed_images(items))
        })
        .map_err(to_py_err)
    }

    /// Whether the endpoint's loaded model serves image embeddings (its
    /// vision mmproj is loaded), via `GET /props`. The facade checks
    /// this at start so a `modalities: [text, image]` entry against a
    /// text-only endpoint fails fast at boot, not at first image embed.
    fn vision_capable(&self, py: Python<'_>) -> bool {
        py.detach(|| shrike_kernel::block_on(self.inner.props()).is_some_and(|p| p.vision))
    }

    /// `GET /health` returns 200.
    fn health_ok(&self, py: Python<'_>) -> bool {
        py.detach(|| shrike_kernel::block_on(self.inner.health_ok()))
    }

    /// `(model_id, meta_json)` from `/v1/models` — `(None, "{}")` when the
    /// endpoint doesn't serve it; fingerprint assembly stays facade policy.
    fn model_info(&self, py: Python<'_>) -> (Option<String>, String) {
        py.detach(|| {
            let info = shrike_kernel::block_on(self.inner.model_info());
            let meta = serde_json::Value::Object(info.meta).to_string();
            (info.id, meta)
        })
    }
}

/// The deterministic synthetic engine (#865) under the `SyntheticBackend`
/// facade: a `dim`-dimensional unit vector computed purely from the input
/// bytes — no model, negligible cost. Behind `engine-synthetic`, absent from a
/// release build, so a config naming `runtime: synthetic` is refused by
/// capability resolution. For benchmarking and fast deterministic tests; the
/// vectors carry no semantics.
#[cfg(feature = "engine-synthetic")]
#[pyclass(frozen)]
pub(crate) struct SyntheticEmbedder {
    inner: std::sync::Arc<shrike_engine::synthetic::SyntheticEmbedder>,
}

#[cfg(feature = "engine-synthetic")]
impl SyntheticEmbedder {
    /// The engine, shared (see the ort engines' `engine_arc`).
    pub(crate) fn engine_arc(&self) -> std::sync::Arc<shrike_engine::synthetic::SyntheticEmbedder> {
        std::sync::Arc::clone(&self.inner)
    }
}

#[cfg(feature = "engine-synthetic")]
#[pymethods]
impl SyntheticEmbedder {
    #[new]
    #[pyo3(signature = (*, dim))]
    fn new(dim: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(shrike_engine::synthetic::SyntheticEmbedder::new(dim)),
        }
    }

    /// Embed one chunk of texts (one unit vector per input). Pure sync compute,
    /// GIL released — the direct path for the facade's probe/`embed_texts`; the
    /// kernel slot rides the `NativeEmbedder` composition instead.
    fn embed_chunk(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        use shrike_engine_api::EmbedText as _;
        py.detach(|| self.inner.embed_chunk(&texts))
            .map_err(to_py_err)
    }

    /// Embed one chunk of images (encoded bytes in, one vector per image).
    fn embed_image_chunk(&self, py: Python<'_>, images: Vec<Vec<u8>>) -> PyResult<Vec<Vec<f32>>> {
        use shrike_engine_api::EmbedImages as _;
        py.detach(|| {
            let items: Vec<shrike_engine_api::MediaItem> = images
                .into_iter()
                .map(shrike_engine_api::MediaItem::untyped)
                .collect();
            self.inner.embed_image_chunk(&items)
        })
        .map_err(to_py_err)
    }

    /// The fixed embedding dimension (known up front — no probe needed).
    fn dim(&self) -> Option<usize> {
        use shrike_engine_api::EmbedText as _;
        self.inner.dim()
    }
}

/// The llama-server lifecycle manager under the llama facade:
/// spawn + health-wait + orphan reaping + escalating stop, all native. NOT
/// an embedder — the facade composes manager → endpoint → RemoteEmbedder.
#[cfg(feature = "manage-llama")]
#[pyclass]
pub(crate) struct LlamaServerManager {
    inner: std::sync::Mutex<shrike_llama_server::LlamaServerManager>,
    /// Non-blocking observer state (the review's loop-stall finding): the
    /// lifecycle Mutex is held for up to the 30s health-wait, and the
    /// harness reads `running` on the LOOP thread — observers must never
    /// contend with it. The cell is shared with the manager (set at spawn,
    /// cleared on stop/observed exit); passthrough is pure config,
    /// precomputed at construction.
    pid_cell: std::sync::Arc<std::sync::Mutex<Option<u32>>>,
    passthrough: Vec<String>,
}

#[cfg(feature = "manage-llama")]
#[pymethods]
impl LlamaServerManager {
    #[new]
    #[pyo3(signature = (model, *, host, port, binary=None, log_dir=None, context_size=None, threads=None, gpu_layers=None, pooling=None, extra_args=vec![], pid_file=None, mmprojs=vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: String,
        host: String,
        port: u16,
        binary: Option<String>,
        log_dir: Option<String>,
        context_size: Option<u32>,
        threads: Option<u32>,
        gpu_layers: Option<i32>,
        pooling: Option<String>,
        extra_args: Vec<String>,
        pid_file: Option<String>,
        mmprojs: Vec<String>,
    ) -> Self {
        let cfg = shrike_llama_server::LlamaServerConfig::new(
            shrike_llama_server::ModelSpec::Single(model),
            host,
            port,
        )
        .with_binary(binary)
        .with_log_dir(log_dir.map(std::path::PathBuf::from))
        .with_context_size(context_size)
        .with_threads(threads)
        .with_gpu_layers(gpu_layers)
        .with_pooling(pooling)
        .with_extra_args(extra_args)
        .with_pid_file(pid_file.map(std::path::PathBuf::from));
        // Per-modality projectors for a multimodal embeddings server:
        // an embeddings server (not chat_mode) that loads vision/audio
        // mmprojs to embed media. Empty = a text-only embeddings server.
        let manager = shrike_llama_server::LlamaServerManager::new(cfg).with_mmprojs(mmprojs);
        Self {
            pid_cell: manager.pid_cell(),
            passthrough: manager.passthrough_tokens(false),
            inner: std::sync::Mutex::new(manager),
        }
    }

    /// Router mode: ONE llama-server serving a directory of models, with
    /// the request's `model` field selecting among them (`--models-dir` +
    /// optional `--models-max`, no `--model`). A separate constructor so the
    /// single-model `new` stays byte-for-byte unchanged. mmprojs are
    /// model-specific and have no router meaning, so this shape omits them.
    #[staticmethod]
    #[pyo3(signature = (models_dir, *, host, port, models_max=None, binary=None, log_dir=None, context_size=None, threads=None, gpu_layers=None, pooling=None, extra_args=vec![], pid_file=None))]
    #[allow(clippy::too_many_arguments)]
    fn router(
        models_dir: String,
        host: String,
        port: u16,
        models_max: Option<u32>,
        binary: Option<String>,
        log_dir: Option<String>,
        context_size: Option<u32>,
        threads: Option<u32>,
        gpu_layers: Option<i32>,
        pooling: Option<String>,
        extra_args: Vec<String>,
        pid_file: Option<String>,
    ) -> Self {
        let cfg = shrike_llama_server::LlamaServerConfig::new(
            shrike_llama_server::ModelSpec::Router {
                dir: models_dir,
                max: models_max,
            },
            host,
            port,
        )
        .with_binary(binary)
        .with_log_dir(log_dir.map(std::path::PathBuf::from))
        .with_context_size(context_size)
        .with_threads(threads)
        .with_gpu_layers(gpu_layers)
        .with_pooling(pooling)
        .with_extra_args(extra_args)
        .with_pid_file(pid_file.map(std::path::PathBuf::from));
        let manager = shrike_llama_server::LlamaServerManager::new(cfg);
        Self {
            pid_cell: manager.pid_cell(),
            passthrough: manager.passthrough_tokens(false),
            inner: std::sync::Mutex::new(manager),
        }
    }

    /// Spawn + health-wait (blocking — the facade runs it off the loop).
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.lock().expect("manager poisoned").start())
            .map_err(to_py_err)
    }

    /// SIGTERM → SIGKILL stop (blocking, up to the shutdown tiers).
    fn stop(&self, py: Python<'_>) {
        py.detach(|| self.inner.lock().expect("manager poisoned").stop())
    }

    /// Non-blocking: a real child poll when the lifecycle lock is free; the
    /// observed-PID cell when a start/stop holds it (mid-start a spawned
    /// child reads as running — the Python facade's `poll()` semantics).
    fn running(&self) -> bool {
        match self.inner.try_lock() {
            Ok(mut manager) => manager.running(),
            Err(_) => self.pid_cell.lock().expect("pid cell poisoned").is_some(),
        }
    }

    /// Non-blocking, same split as [`Self::running`].
    fn pid(&self) -> Option<u32> {
        match self.inner.try_lock() {
            Ok(manager) => manager.pid(),
            Err(_) => *self.pid_cell.lock().expect("pid cell poisoned"),
        }
    }

    /// The effective passthrough (reserved flags stripped, silent) — pure
    /// config, precomputed; the facade folds it into the fingerprint's
    /// `args=` suffix.
    fn passthrough_tokens(&self) -> Vec<String> {
        self.passthrough.clone()
    }
}

// ── Derived-text engine ──────────────────────────────────────────────

/// Whether the linked SQLite has FTS5 with the trigram tokenizer.
/// Constant-true under the bundled default; genuinely probed under platform
/// linkage (a build without the bundled-sqlite feature).
#[pyfunction]
fn derived_fts5_probe() -> bool {
    shrike_derived::fts5_trigram_available()
}

/// Whether this build statically links the bundled SQLite — for
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
    pub(crate) inner: shrike_derived::DerivedEngine,
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

    fn get_col_mod(&self, py: Python<'_>) -> Option<i64> {
        py.detach(|| self.inner.get_col_mod())
    }

    fn set_col_mod(&self, py: Python<'_>, value: i64) -> PyResult<()> {
        // A write-transaction commit (a journal sync) — off the GIL like the
        // sibling methods.
        py.detach(|| self.inner.set_col_mod(value))
            .map_err(to_py_err)
    }

    fn count(&self, py: Python<'_>) -> PyResult<i64> {
        py.detach(|| self.inner.count()).map_err(to_py_err)
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

    #[pyo3(signature = (rows, col_mod, live_notes=None))]
    fn build(
        &self,
        py: Python<'_>,
        rows: Vec<(i64, String, String, String)>,
        col_mod: i64,
        live_notes: Option<Vec<i64>>,
    ) -> PyResult<()> {
        // `live_notes` is the authoritative collection note set governing the
        // recognition-row prune. The standalone host rebuilds in one synchronous
        // pass with no concurrent recognition ingest, so when it is omitted the
        // snapshot's own note ids are the live set.
        let live = live_notes.unwrap_or_else(|| rows.iter().map(|r| r.0).collect());
        py.detach(|| self.inner.build(&rows, &live, col_mod))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (expr, limit, with_text, scope=None))]
    fn match_rows(
        &self,
        py: Python<'_>,
        expr: String,
        limit: i64,
        with_text: bool,
        scope: Option<Vec<i64>>,
    ) -> PyResult<Vec<MatchRow>> {
        py.detach(|| {
            self.inner
                .match_rows(&expr, limit, with_text, scope.as_deref(), &[])
        })
        .map_err(to_py_err)
    }

    /// Fuzzy (trigram/typo) rows, best-first, deduped per note — the sole lexical
    /// read (the kernel layers the `exact` tier on top). Each row is
    /// `(note_id, source, ref, match)` where `match` is `(text, (first, last))`
    /// (the matched derived segment text + the overlap's UTF-8 byte span) or
    /// `None`.
    #[pyo3(signature = (query, top_k, scope=None))]
    fn search_fuzzy(
        &self,
        py: Python<'_>,
        query: String,
        top_k: i64,
        scope: Option<Vec<i64>>,
    ) -> PyResult<Vec<PyLexicalRow>> {
        let rows = py
            .detach(|| {
                self.inner
                    .search_fuzzy(&query, top_k, scope.as_deref(), &[])
            })
            .map_err(to_py_err)?;
        Ok(rows.into_iter().map(lexical_row_to_py).collect())
    }

    /// Fuzzy rows for a BATCH of queries — one result list per query in order,
    /// sharing one connection and one DF lookup. The fuzzy-recall eval calls this
    /// to measure the fuzzy signal in isolation across thousands of typo queries.
    #[pyo3(signature = (queries, top_k, scope=None))]
    fn search_fuzzy_batch(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        top_k: i64,
        scope: Option<Vec<i64>>,
    ) -> PyResult<Vec<Vec<PyLexicalRow>>> {
        let batched = py
            .detach(|| {
                let refs: Vec<&str> = queries.iter().map(String::as_str).collect();
                self.inner
                    .search_fuzzy_batch(&refs, top_k, scope.as_deref(), &[])
            })
            .map_err(to_py_err)?;
        Ok(batched
            .into_iter()
            .map(|rows| rows.into_iter().map(lexical_row_to_py).collect())
            .collect())
    }

    /// Set the cost-budget pruner policy: `typo_floor` (`F`, min rarest trigrams kept),
    /// `cost_budget` (`B`, scan-cost admitted past the floor), `max_terms` (`k_max`),
    /// and the cost coefficients `cost_per_term` (`α`, per kept list) / `cost_per_df`
    /// (`β`, per rowid). The recall eval sweeps `F`/`B` (and profiles `α`/`β`);
    /// production never calls it.
    #[allow(clippy::too_many_arguments)]
    fn set_prune_policy(
        &self,
        py: Python<'_>,
        typo_floor: usize,
        cost_budget: f64,
        max_terms: usize,
        cost_per_term: f64,
        cost_per_df: f64,
    ) {
        py.detach(|| {
            self.inner.set_prune_policy(shrike_derived::PrunePolicy {
                typo_floor,
                cost_budget,
                max_terms,
                cost_per_term,
                cost_per_df,
            });
        });
    }

    /// The current `(typo_floor, cost_budget, max_terms, cost_per_term, cost_per_df)`
    /// prune policy.
    fn prune_policy(&self, py: Python<'_>) -> (usize, f64, usize, f64, f64) {
        py.detach(|| {
            let p = self.inner.prune_policy();
            (
                p.typo_floor,
                p.cost_budget,
                p.max_terms,
                p.cost_per_term,
                p.cost_per_df,
            )
        })
    }

    /// Set the materialization DF ceiling `C`: a trigram in fewer than `C` idx rows
    /// is materialized as an incrementally-maintained base bitmap, commoner trigrams
    /// fall to the live posting read. A pure performance dial (results are
    /// path-invariant). The perf harness sweeps it to find the per-write-cost /
    /// query-coverage knee; production opens with the default and never calls it.
    fn set_materialize_ceiling(&self, py: Python<'_>, ceiling: usize) {
        py.detach(|| self.inner.set_materialize_ceiling(ceiling));
    }

    /// The current materialization DF ceiling `C`.
    fn materialize_ceiling(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.materialize_ceiling())
    }
}

// ── Index engine ─────────────────────────────────────────────────────

/// The native per-modality vector index engine under the `VectorIndex`
/// orchestrator (the frozen `IndexEngine` surface). Coarse, batched calls
/// trafficking in i64 key arrays and f32 vector batches, all GIL-released.
#[pyclass(frozen)]
struct NativeIndexEngine {
    /// `Arc`-shared with the kernel's `IndexOrchestrator` (`KernelIndex`):
    /// one engine, two roles — the harness searches it, the kernel maintains it.
    pub(crate) inner: std::sync::Arc<dyn shrike_store::VectorIndex>,
}

#[pymethods]
impl NativeIndexEngine {
    #[new]
    fn new(modalities: Vec<String>) -> PyResult<Self> {
        let inner =
            std::sync::Arc::new(shrike_index::MultiModalIndex::new(modalities).map_err(to_py_err)?);
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

    fn modality_stats(&self) -> Vec<(String, usize, Option<usize>)> {
        self.inner.modality_stats()
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
                .search_by_modality(&queries, k, modalities.as_deref(), None)
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

// ── Fused compute ────────────────────────────────────────────────────

/// Reciprocal Rank Fusion — the native implementation of the frozen
/// `search_fusion.py` spec. Same canonical accumulation order, same dedup, same
/// `(tier, -score, note_id)` ordering; the Python parity property suite pins
/// the two byte-for-byte. The one implementation lives in the kernel
/// (`shrike_kernel::fusion`), so anki-core builds only.
#[cfg(feature = "anki-core")]
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
        shrike_kernel::fusion::rrf_fuse(&rankings, &weights, k, &priority)
    })
}

/// The path-derived per-collection index identity. The single
/// implementation lives in `shrike-cache` (cracked out of the kernel);
/// this binds it so the host can resolve the same `<cache_dir>/index/<namespace>/`
/// the kernel writes (the routing capstone, status reporting, tests). A
/// Python parity test pins the two byte-for-byte.
#[cfg(feature = "anki-core")]
#[pyfunction]
fn index_namespace(py: Python<'_>, collection_path: String) -> String {
    py.detach(move || shrike_cache::index_namespace(&collection_path))
}

/// The path-derived per-collection derived-store path:
/// `<cache_dir>/derived/<namespace>/shrike.db`. Binds `shrike-cache`'s single
/// implementation (cracked out of the kernel) so the host
/// `DerivedTextStore` opens exactly the file the kernel's `DerivedEngine`
/// writes (they share one db). A Python parity test pins the two byte-for-byte.
/// Returns a lossy-UTF-8 string (paths the host hands in are already UTF-8).
#[cfg(feature = "anki-core")]
#[pyfunction]
fn derived_db_path(py: Python<'_>, cache_dir: String, collection_path: String) -> String {
    py.detach(move || {
        shrike_cache::derived_db_path(&cache_dir, &collection_path)
            .to_string_lossy()
            .into_owned()
    })
}

/// The module init. Its name MUST match the imported module / the `.so`
/// filename (`_native`), since PyO3 exports `PyInit__native` from it.
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Process-global SQLite tuning, at import — before any connection opens, so it
    // actually takes (sqlite3_config is a no-op once SQLite has initialized).
    shrike_derived::configure_sqlite_perf();
    // The interpreter-finalization gate: exported (the teardown tests
    // close it deliberately) and armed via atexit in every importing process.
    m.add_function(wrap_pyfunction!(finalize_gate::finalize_gate_close, m)?)?;
    finalize_gate::register_exit_hook(m)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(build_features, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_sum, m)?)?;
    m.add_function(wrap_pyfunction!(checked_div, m)?)?;
    #[cfg(feature = "engine-ort")]
    m.add_function(wrap_pyfunction!(init_onnx_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    #[cfg(feature = "anki-core")]
    {
        m.add_function(wrap_pyfunction!(fetch_media_url, m)?)?;
        m.add_function(wrap_pyfunction!(decode_media_b64, m)?)?;
        m.add_function(wrap_pyfunction!(init_driven_runtime, m)?)?;
        m.add_function(wrap_pyfunction!(drive_io, m)?)?;
        m.add_function(wrap_pyfunction!(drive_collection, m)?)?;
        m.add_function(wrap_pyfunction!(drive_compute, m)?)?;
        m.add_function(wrap_pyfunction!(drive_pools_shutdown, m)?)?;
        m.add_function(wrap_pyfunction!(runtime_probe, m)?)?;
        m.add_function(wrap_pyfunction!(render_prometheus, m)?)?;
    }
    // The engine/manager matrix: a class is present exactly when its
    // feature is compiled — a lean build simply lacks the name (the Python
    // facade only rides full server builds, so nothing suppresses these).
    #[cfg(feature = "engine-ort")]
    {
        m.add_class::<OnnxTextEmbedder>()?;
        m.add_class::<ClipEmbedder>()?;
    }
    #[cfg(feature = "engine-remote")]
    m.add_class::<RemoteEmbedder>()?;
    #[cfg(feature = "engine-synthetic")]
    m.add_class::<SyntheticEmbedder>()?;
    #[cfg(feature = "manage-llama")]
    m.add_class::<LlamaServerManager>()?;
    m.add_class::<NativeIndexEngine>()?;
    m.add_class::<DerivedTextEngine>()?;
    m.add_class::<py_recognizer::PyRecognizer>()?;
    #[cfg(feature = "engine-apple")]
    m.add_class::<py_recognizer::AppleVisionRecognizer>()?;
    #[cfg(feature = "engine-remote")]
    m.add_class::<py_recognizer::RemoteDescriber>()?;
    // Feature-gated: present only in `anki-core` builds; the stubtest
    // allowlist covers its absence from default builds.
    #[cfg(feature = "anki-core")]
    m.add_class::<anki_core::CollectionCore>()?;
    #[cfg(feature = "anki-core")]
    {
        m.add_function(wrap_pyfunction!(kernel_actions::rehomed_actions, m)?)?;
        m.add_function(wrap_pyfunction!(kernel_actions::action_collection_info, m)?)?;
        m.add_function(wrap_pyfunction!(kernel_actions::action_list_notes, m)?)?;
        m.add_function(wrap_pyfunction!(
            kernel_actions::action_collection_query,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(kernel_actions::action_search_notes, m)?)?;
        m.add_class::<async_kernel::AsyncCollection>()?;
        m.add_function(wrap_pyfunction!(async_kernel::async_collection_open, m)?)?;
        m.add_class::<async_kernel::AsyncKernel>()?;
        m.add_function(wrap_pyfunction!(async_kernel::async_kernel_open, m)?)?;
        m.add_function(wrap_pyfunction!(index_namespace, m)?)?;
        m.add_function(wrap_pyfunction!(derived_db_path, m)?)?;
    }
    m.add_class::<py_embedder::PyEmbedder>()?;
    m.add_class::<native_embedder::NativeEmbedder>()?;
    #[cfg(feature = "anki-core")]
    {
        m.add_function(wrap_pyfunction!(py_embedder::embedder_probe, m)?)?;
        m.add_function(wrap_pyfunction!(native_embedder::native_embedder_probe, m)?)?;
    }
    m.add_function(wrap_pyfunction!(derived_fts5_probe, m)?)?;
    m.add_function(wrap_pyfunction!(derived_sqlite_bundled, m)?)?;
    // Bridge lifecycle test seams: the leak tripwire counter + a
    // waker-retaining pending future to park on it.
    m.add_function(wrap_pyfunction!(
        asyncio_bridge::bridge_live_poll_callbacks,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(asyncio_bridge::bridge_parked_forever, m)?)?;
    #[cfg(feature = "anki-core")]
    m.add_function(wrap_pyfunction!(rrf_fuse, m)?)?;
    m.add_function(wrap_pyfunction!(schema_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(wire_protocol_version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_roundtrip, m)?)?;
    // The native image-prep pipeline version — folded into the clip-rs
    // fingerprint by the facade (a pixel-math change must invalidate vectors).
    #[cfg(feature = "engine-ort")]
    m.add(
        "IMAGE_PREP_VERSION_RS",
        shrike_engine::onnx::IMAGE_PREP_VERSION_RS,
    )?;
    // The kernel saver's built-in flush tuning — the host's
    // --index-save-* help text names the defaults it would override.
    #[cfg(feature = "anki-core")]
    {
        m.add(
            "INDEX_SAVE_DELAY_DEFAULT",
            shrike_kernel::index_orchestrator::DEFAULT_SAVE_DELAY,
        )?;
        m.add(
            "INDEX_SAVE_THRESHOLD_DEFAULT",
            shrike_kernel::index_orchestrator::DEFAULT_SAVE_THRESHOLD,
        )?;
    }
    // The batch-safety probe surface: the spiked set + tolerance,
    // for tests that pin sensitivity/ceiling against the same texts the
    // native probe embeds.
    m.add(
        "BATCH_PROBE_TEXTS",
        shrike_engine_api::probe::BATCH_PROBE_TEXTS.to_vec(),
    )?;
    // The vision probe set: the same canonical synthetic images the
    // native vision probe embeds, so the Python CLIP host probes the vision
    // path against the identical set (mirrors BATCH_PROBE_TEXTS).
    m.add(
        "BATCH_PROBE_IMAGES",
        shrike_engine_api::probe::batch_probe_images(),
    )?;
    m.add("BATCH_DRIFT_TOL", shrike_engine_api::probe::BATCH_DRIFT_TOL)?;
    m.add("NativeInputError", py.get_type::<NativeInputError>())?;
    m.add(
        "NativeUnavailableError",
        py.get_type::<NativeUnavailableError>(),
    )?;
    m.add("NativeInternalError", py.get_type::<NativeInternalError>())?;
    m.add("NativeBusyError", py.get_type::<NativeBusyError>())?;
    Ok(())
}
