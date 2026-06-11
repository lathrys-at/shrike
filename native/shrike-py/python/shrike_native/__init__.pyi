# Re-exports of the compiled module's surface; __all__ marks them exported.
# CollectionCore is feature-gated (#278, anki-core builds only): re-exported
# via the `as` form (outside __all__), allowlisted for default-build stubtest.
from shrike_native._native import CollectionCore as CollectionCore
from shrike_native._native import decode_media_b64 as decode_media_b64
from shrike_native._native import fetch_media_url as fetch_media_url
from shrike_native._native import (
    IMAGE_PREP_VERSION_RS,
    ClipEmbedder,
    DerivedTextEngine,
    NativeIndexEngine,
    NativeBusyError,
    NativeInputError,
    NativeInternalError,
    NativeUnavailableError,
    OnnxTextEmbedder,
    build_info,
    checked_div,
    derived_fts5_probe,
    derived_sqlite_bundled,
    fused_add_text,
    fused_search_text,
    init_logging,
    init_onnx_runtime,
    parallel_sum,
    rrf_fuse,
    version,
)

__all__ = [
    "ClipEmbedder",
    "DerivedTextEngine",
    "IMAGE_PREP_VERSION_RS",
    "NativeIndexEngine",
    "NativeBusyError",
    "NativeInputError",
    "NativeInternalError",
    "NativeUnavailableError",
    "OnnxTextEmbedder",
    "build_info",
    "checked_div",
    "derived_fts5_probe",
    "derived_sqlite_bundled",
    "fused_add_text",
    "fused_search_text",
    "init_logging",
    "init_onnx_runtime",
    "rrf_fuse",
    "parallel_sum",
    "version",
]
