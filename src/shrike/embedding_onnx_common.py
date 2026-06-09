"""Shared in-process onnxruntime helpers for the text (`OnnxBackend`) and CLIP (`ClipBackend`)
backends — execution-provider resolution and the onnx-int-input dtype map.

Kept here (not in `embedding_base.py`) so the backend *protocol* stays free of any onnxruntime
coupling; only the two onnxruntime backends import this.
"""

from __future__ import annotations

import numpy as np

# An onnx integer input's declared type → the numpy dtype to feed it. onnxruntime does not
# auto-cast int64<->int32, so each declared integer input must be fed as its declared type
# (some quantized exports declare int32 ids). Unknown types fall back to int64 at the call site.
ORT_INT_DTYPES = {"tensor(int64)": np.int64, "tensor(int32)": np.int32}


def resolve_execution_providers(
    available: list[str], requested: list[str]
) -> tuple[list[str], list[str]]:
    """Resolve requested onnxruntime execution providers against what's available.

    Keeps the requested providers that onnxruntime actually has (in request order), always
    appends ``CPUExecutionProvider`` as the final fallback, and dedups. Returns
    ``(resolved, dropped)`` — *dropped* being the requested providers that aren't available, so
    the caller can warn rather than rely on onnxruntime's silent CPU fallback.
    """
    resolved: list[str] = []
    for p in [*requested, "CPUExecutionProvider"]:
        if p in available and p not in resolved:
            resolved.append(p)
    dropped = [p for p in requested if p not in available]
    return resolved, dropped
