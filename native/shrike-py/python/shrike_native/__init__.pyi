# Re-exports of the compiled module's surface; __all__ marks them exported.
# CollectionCore is feature-gated (#278, anki-core builds only): re-exported
# via the `as` form (outside __all__), allowlisted for default-build stubtest.
from shrike_native._native import CollectionCore as CollectionCore
from shrike_native._native import action_collection_info as action_collection_info
from shrike_native._native import action_collection_query as action_collection_query
from shrike_native._native import action_list_notes as action_list_notes
from shrike_native._native import AsyncCollection as AsyncCollection
from shrike_native._native import AsyncKernel as AsyncKernel
from shrike_native._native import action_search_notes as action_search_notes
from shrike_native._native import async_collection_open as async_collection_open
from shrike_native._native import async_kernel_open as async_kernel_open
from shrike_native._native import decode_media_b64 as decode_media_b64
from shrike_native._native import rehomed_actions as rehomed_actions
from shrike_native._native import INDEX_SAVE_DELAY_DEFAULT as INDEX_SAVE_DELAY_DEFAULT
from shrike_native._native import INDEX_SAVE_THRESHOLD_DEFAULT as INDEX_SAVE_THRESHOLD_DEFAULT
from shrike_native._native import embedder_probe as embedder_probe
from shrike_native._native import native_embedder_probe as native_embedder_probe
from shrike_native._native import rrf_fuse as rrf_fuse
from shrike_native._native import fetch_media_url as fetch_media_url
from shrike_native._native import (
    AppleVisionRecognizer,
    BATCH_DRIFT_TOL,
    BATCH_PROBE_TEXTS,
    IMAGE_PREP_VERSION_RS,
    ClipEmbedder,
    DerivedTextEngine,
    NativeIndexEngine,
    NativeBusyError,
    NativeInputError,
    NativeInternalError,
    NativeUnavailableError,
    OnnxTextEmbedder,
    bridge_live_poll_callbacks,
    bridge_parked_forever,
    build_info,
    checked_div,
    derived_fts5_probe,
    derived_sqlite_bundled,
    fused_add_text,
    fused_search_text,
    init_logging,
    LlamaServerManager,
    NativeEmbedder,
    PyEmbedder,
    RemoteEmbedder,
    Recognizer,
    init_onnx_runtime,
    parallel_sum,
    schema_catalog,
    schema_roundtrip,
    version,
)

__all__ = [
    "AppleVisionRecognizer",
    "BATCH_DRIFT_TOL",
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
    "build_info",
    "checked_div",
    "derived_fts5_probe",
    "derived_sqlite_bundled",
    "fused_add_text",
    "fused_search_text",
    "init_logging",
    "LlamaServerManager",
    "NativeEmbedder",
    "PyEmbedder",
    "RemoteEmbedder",
    "Recognizer",
    "init_onnx_runtime",
    "schema_catalog",
    "schema_roundtrip",
    "parallel_sum",
    "version",
]
