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
    init_onnx_runtime,
    parallel_sum,
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
    "init_onnx_runtime",
    "parallel_sum",
    "version",
]
