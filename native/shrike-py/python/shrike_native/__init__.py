"""Python face of the Shrike native extension (#269).

Re-exports the compiled ``shrike_native._native`` module's surface. Production
code in ``shrike`` imports *this package* (lazily, inside the facades, so a
missing native install degrades to a clean ``ImportError``) — never ``_native``
directly. The package ships ``.pyi`` stubs + ``py.typed``; ``mypy.stubtest``
in the native CI lane keeps them honest.
"""

from shrike_native._native import (
    IMAGE_PREP_VERSION_RS,
    ClipEmbedder,
    DerivedTextEngine,
    NativeIndexEngine,
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
    init_onnx_runtime,
    init_tracing,
    parallel_sum,
    rrf_fuse,
    version,
)

__all__ = [
    "ClipEmbedder",
    "DerivedTextEngine",
    "IMAGE_PREP_VERSION_RS",
    "NativeIndexEngine",
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
    "init_onnx_runtime",
    "init_tracing",
    "rrf_fuse",
    "parallel_sum",
    "version",
]
