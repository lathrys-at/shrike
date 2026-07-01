"""Python face of the Shrike native extension.

Re-exports the compiled ``shrike_native._native`` module's surface. Production
code in ``shrike`` imports *this package* (lazily, inside the facades, so a
missing native install degrades to a clean ``ImportError``) — never ``_native``
directly. The package ships ``.pyi`` stubs + ``py.typed``; ``mypy.stubtest``
in the native CI lane keeps them honest.
"""

import contextlib

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

# Feature-gated: present only in `engine-apple` builds — the mobile set, NEVER
# the server build, on any OS (docs/distribution.md's boundary). Outside
# __all__; consumers (recognition.py, harness.py) look it up with getattr and
# degrade when it's absent.
with contextlib.suppress(ImportError):
    from shrike_native._native import AppleVisionRecognizer  # noqa: F401

# Feature-gated: present only in `engine-synthetic` builds (#865) — the perf
# lane and fast deterministic tests, never the release wheel. Outside __all__;
# the SyntheticBackend facade imports it directly, and a lean build refuses
# `runtime: synthetic` at config resolution, so this is never reached there.
with contextlib.suppress(ImportError):
    from shrike_native._native import SyntheticEmbedder  # noqa: F401

# Feature-gated: present only in `anki-core` builds (the
# scripts/build-native.sh default). Deliberately outside __all__ — the parity
# harness (tests/native) imports it explicitly; on a build without the feature
# the name simply doesn't exist (the harness skips).
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
        derived_db_path,  # noqa: F401
        drive_collection,  # noqa: F401
        drive_compute,  # noqa: F401
        drive_io,  # noqa: F401
        drive_pools_shutdown,  # noqa: F401
        embedder_probe,  # noqa: F401
        fetch_media_url,  # noqa: F401
        index_namespace,  # noqa: F401
        init_driven_runtime,  # noqa: F401
        native_embedder_probe,  # noqa: F401
        rehomed_actions,  # noqa: F401
        render_prometheus,  # noqa: F401
        rrf_fuse,  # noqa: F401
        runtime_probe,  # noqa: F401
    )

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
    "init_onnx_runtime",
    "LlamaServerManager",
    "NativeEmbedder",
    "PyEmbedder",
    "RemoteDescriber",
    "RemoteEmbedder",
    "Recognizer",
    "parallel_sum",
    "schema_catalog",
    "schema_roundtrip",
    "version",
    "wire_protocol_version",
]
