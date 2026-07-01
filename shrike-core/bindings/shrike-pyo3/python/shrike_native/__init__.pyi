# Re-exports of the compiled module's surface; __all__ marks them exported.
# CollectionCore (#278, anki-core builds) and AppleVisionRecognizer (#499,
# engine-apple/mobile builds — never the server build) are feature-gated:
# re-exported via the `as` form (outside __all__), allowlisted for stubtest
# on builds that lack them.
from shrike_native._native import (
    BATCH_DRIFT_TOL,
    BATCH_PROBE_IMAGES,
    BATCH_PROBE_TEXTS,
    IMAGE_PREP_VERSION_RS,
    ClipEmbedder,
    DerivedTextEngine,
    LlamaServerManager,
    NativeBusyError,
    NativeEmbedder,
    NativeIndexEngine,
    NativeInputError,
    NativeInternalError,
    NativeUnavailableError,
    OnnxTextEmbedder,
    PyEmbedder,
    Recognizer,
    RemoteDescriber,
    RemoteEmbedder,
    bridge_live_poll_callbacks,
    bridge_parked_forever,
    build_features,
    build_info,
    checked_div,
    derived_fts5_probe,
    derived_sqlite_bundled,
    finalize_gate_close,
    init_logging,
    init_onnx_runtime,
    parallel_sum,
    schema_catalog,
    schema_roundtrip,
    version,
    wire_protocol_version,
)
from shrike_native._native import INDEX_SAVE_DELAY_DEFAULT as INDEX_SAVE_DELAY_DEFAULT
from shrike_native._native import INDEX_SAVE_THRESHOLD_DEFAULT as INDEX_SAVE_THRESHOLD_DEFAULT
from shrike_native._native import AppleVisionRecognizer as AppleVisionRecognizer
from shrike_native._native import AsyncCollection as AsyncCollection
from shrike_native._native import AsyncKernel as AsyncKernel
from shrike_native._native import CollectionCore as CollectionCore
from shrike_native._native import SyntheticEmbedder as SyntheticEmbedder
from shrike_native._native import action_collection_info as action_collection_info
from shrike_native._native import action_collection_query as action_collection_query
from shrike_native._native import action_list_notes as action_list_notes
from shrike_native._native import action_search_notes as action_search_notes
from shrike_native._native import async_collection_open as async_collection_open
from shrike_native._native import async_kernel_open as async_kernel_open
from shrike_native._native import decode_media_b64 as decode_media_b64
from shrike_native._native import derived_db_path as derived_db_path
from shrike_native._native import drive_collection as drive_collection
from shrike_native._native import drive_compute as drive_compute
from shrike_native._native import drive_io as drive_io
from shrike_native._native import drive_pools_shutdown as drive_pools_shutdown
from shrike_native._native import embedder_probe as embedder_probe
from shrike_native._native import fetch_media_url as fetch_media_url
from shrike_native._native import index_namespace as index_namespace
from shrike_native._native import init_driven_runtime as init_driven_runtime
from shrike_native._native import native_embedder_probe as native_embedder_probe
from shrike_native._native import rehomed_actions as rehomed_actions
from shrike_native._native import render_prometheus as render_prometheus
from shrike_native._native import rrf_fuse as rrf_fuse
from shrike_native._native import runtime_probe as runtime_probe

__all__ = [
    "BATCH_DRIFT_TOL",
    "BATCH_PROBE_IMAGES",
    "BATCH_PROBE_TEXTS",
    "ClipEmbedder",
    "DerivedTextEngine",
    "IMAGE_PREP_VERSION_RS",
    "NativeIndexEngine",
    "NativeBusyError",
    "NativeInputError",
    "NativeInternalError",
    "NativeUnavailableError",
    "OnnxTextEmbedder",
    "bridge_live_poll_callbacks",
    "bridge_parked_forever",
    "build_features",
    "build_info",
    "checked_div",
    "derived_fts5_probe",
    "derived_sqlite_bundled",
    "finalize_gate_close",
    "init_logging",
    "LlamaServerManager",
    "NativeEmbedder",
    "PyEmbedder",
    "RemoteDescriber",
    "RemoteEmbedder",
    "Recognizer",
    "init_onnx_runtime",
    "schema_catalog",
    "schema_roundtrip",
    "parallel_sum",
    "version",
    "wire_protocol_version",
]
