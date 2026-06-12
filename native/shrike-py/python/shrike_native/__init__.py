"""Python face of the Shrike native extension (#269).

Re-exports the compiled ``shrike_native._native`` module's surface. Production
code in ``shrike`` imports *this package* (lazily, inside the facades, so a
missing native install degrades to a clean ``ImportError``) — never ``_native``
directly. The package ships ``.pyi`` stubs + ``py.typed``; ``mypy.stubtest``
in the native CI lane keeps them honest.
"""

import contextlib

from shrike_native._native import (
    BATCH_DRIFT_TOL,
    BATCH_PROBE_TEXTS,
    IMAGE_PREP_VERSION_RS,
    AppleVisionRecognizer,
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
    RemoteEmbedder,
    bridge_live_poll_callbacks,
    bridge_parked_forever,
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
)

# Feature-gated (#278 series, step 1): present only in `anki-core` builds
# (scripts/build-native.sh --anki-core). Deliberately outside __all__ — the
# parity harness (tests/native) imports it explicitly; on a default build the
# name simply doesn't exist (the harness skips).
with contextlib.suppress(ImportError):
    from shrike_native._native import (
        INDEX_SAVE_DELAY_DEFAULT,  # noqa: F401
        INDEX_SAVE_THRESHOLD_DEFAULT,  # noqa: F401
        AsyncCollection,  # noqa: F401
        AsyncKernel,  # noqa: F401
        CollectionCore,  # noqa: F401
        action_collection_info,  # noqa: F401
        action_collection_query,  # noqa: F401
        action_list_notes,  # noqa: F401
        action_search_notes,  # noqa: F401
        async_collection_open,  # noqa: F401
        async_kernel_open,  # noqa: F401
        decode_media_b64,  # noqa: F401
        embedder_probe,  # noqa: F401
        fetch_media_url,  # noqa: F401
        native_embedder_probe,  # noqa: F401
        rehomed_actions,  # noqa: F401
        rrf_fuse,  # noqa: F401
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
    "finalize_gate_close",
    "init_logging",
    "init_onnx_runtime",
    "LlamaServerManager",
    "NativeEmbedder",
    "PyEmbedder",
    "RemoteEmbedder",
    "Recognizer",
    "parallel_sum",
    "schema_catalog",
    "schema_roundtrip",
    "version",
]
