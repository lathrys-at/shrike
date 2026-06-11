"""Python face of the Shrike native extension (#269).

Re-exports the compiled ``shrike_native._native`` module's surface. Production
code in ``shrike`` imports *this package* (lazily, inside the facades, so a
missing native install degrades to a clean ``ImportError``) — never ``_native``
directly. The package ships ``.pyi`` stubs + ``py.typed``; ``mypy.stubtest``
in the native CI lane keeps them honest.
"""

import contextlib

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
    WorkerExecutor,
    parallel_sum,
    rrf_fuse,
    schema_catalog,
    schema_roundtrip,
    version,
)

# Feature-gated (#278 series, step 1): present only in `anki-core` builds
# (scripts/build-native.sh --anki-core). Deliberately outside __all__ — the
# parity harness (tests/native) imports it explicitly; on a default build the
# name simply doesn't exist (the harness skips).
with contextlib.suppress(ImportError):
    from shrike_native._native import CollectionCore  # noqa: F401
    from shrike_native._native import action_collection_info  # noqa: F401
    from shrike_native._native import action_collection_query  # noqa: F401
    from shrike_native._native import action_list_notes  # noqa: F401
    from shrike_native._native import action_search_notes  # noqa: F401
    from shrike_native._native import AsyncCollection  # noqa: F401
    from shrike_native._native import async_collection_open  # noqa: F401
    from shrike_native._native import decode_media_b64  # noqa: F401
    from shrike_native._native import fetch_media_url  # noqa: F401
    from shrike_native._native import rehomed_actions  # noqa: F401

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
    "WorkerExecutor",
    "rrf_fuse",
    "parallel_sum",
    "schema_catalog",
    "schema_roundtrip",
    "version",
]
