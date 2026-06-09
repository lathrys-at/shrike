"""Resilient, cache-friendly fetch of the test embedding model (#83).

The embedding CI lane was flaking on HuggingFace ``429 Too Many Requests``: the
GGUF model was re-downloaded on every run with no retry and no persistent cache,
so HF rate-limited the runner IP and the whole lane errored at setup. This module
adds backoff on transient HTTP failures and lets the download be reused from a
stable directory (populated by ``actions/cache`` in CI).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

# Pinned test embedding model. Bump manually (URL + name together) to change it;
# the CI cache-warmer and the test fixture both read these so they stay in sync.
EMBEDDING_MODEL_URL = (
    "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF"
    "/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf"
)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2-Q4_K_M.gguf"

# Pinned test ONNX embedding model (text-only), used to exercise the ONNX backend
# alongside the GGUF/llama-server one. Same all-MiniLM-L6-v2 architecture (384-dim,
# so the two backends share semantic assertions), but the *portable int8-quantized*
# export from Xenova (transformers.js packaging): ~22 MB vs the 86 MB fp32 export,
# comparable to the 20 MB GGUF, and — unlike sentence-transformers' ISA-tagged
# `model_qint8_avx512`/`_arm64` files — a dynamic quantization with no instruction-set
# assumption, so it runs on the linux-x64 lane, the macOS-arm64 rc lane, and local
# dev alike. Saved as `model.onnx` so OnnxBackend's dir resolver finds it. Bump the
# URLs + dir name together (and the CI cache key) to change it.
ONNX_MODEL_DIR_NAME = "all-MiniLM-L6-v2-onnx-int8"
ONNX_MODEL_FILES = {
    "model.onnx": (
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
    ),
    "tokenizer.json": (
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
    ),
}

# A second, architecturally-different ONNX model for the embedding lane (#172
# review): all-distilroberta-v1 is DistilRoBERTa — **768-dim** (not MiniLM's 384,
# so its own vector space, no cross-model comparison) and a **BPE tokenizer with no
# `[PAD]`** (`token_to_id("[PAD]")` is None, so OnnxBackend's `<pad>` resolution
# fires for real). int8 Xenova export, Apache-2.0. Pins the RoBERTa-only deltas the
# MiniLM can't reach; ~82 MB.
DISTILROBERTA_MODEL_DIR_NAME = "all-distilroberta-v1-onnx-int8"
DISTILROBERTA_MODEL_FILES = {
    "model.onnx": (
        "https://huggingface.co/Xenova/all-distilroberta-v1/resolve/main/onnx/model_quantized.onnx"
    ),
    "tokenizer.json": (
        "https://huggingface.co/Xenova/all-distilroberta-v1/resolve/main/tokenizer.json"
    ),
}

# The fp32 (non-quantized) export of the same MiniLM (#174). Unlike the int8 model it
# batches **bit-exact** (no dynamic activation quantization), so it's the fixture that
# proves the batch-safety probe enables batching where it's safe. ~86 MB.
ONNX_FP32_MODEL_DIR_NAME = "all-MiniLM-L6-v2-onnx-fp32"
ONNX_FP32_MODEL_FILES = {
    "model.onnx": "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
    "tokenizer.json": "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
}

# The small CLIP (image<->text) for the clip backend (#162 Phase 3b): an int8 dual-encoder
# (separate text + vision graphs) + the CLIP tokenizer + image-preprocessing config. Graphs are
# stored flat (the backend's _resolve_files finds them at the dir root via the `variant` suffix).
# ~147 MB. jina-clip-v2 is the production-quality option; this is the CI/test fixture.
CLIP_MODEL_DIR_NAME = "clip-vit-base-patch32-onnx"
_CLIP_BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
CLIP_MODEL_FILES = {
    "text_model_quantized.onnx": f"{_CLIP_BASE}/onnx/text_model_quantized.onnx",
    "vision_model_quantized.onnx": f"{_CLIP_BASE}/onnx/vision_model_quantized.onnx",
    "tokenizer.json": f"{_CLIP_BASE}/tokenizer.json",
    "preprocessor_config.json": f"{_CLIP_BASE}/preprocessor_config.json",
}

# Statuses worth retrying: HF rate-limit plus transient gateway/server errors.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _backoff_delay(attempt: int, response: httpx.Response | None) -> float:
    """Seconds to wait before the next attempt.

    Honors a numeric ``Retry-After`` header when present (capped at 60s);
    otherwise exponential backoff capped at 30s.
    """
    if response is not None:
        retry_after = response.headers.get("retry-after", "").strip()
        if retry_after.isdigit():
            return min(float(retry_after), 60.0)
    return min(2.0**attempt, 30.0)


def download_with_retry(
    url: str,
    dest: Path,
    *,
    attempts: int = 5,
    timeout: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download *url* to *dest*, retrying transient HTTP failures with backoff.

    Retries on the statuses in ``_RETRY_STATUSES`` and on transport errors
    (connect/read timeouts, resets); any other HTTP status raises immediately.
    Raises the last error if every attempt fails. ``sleep`` is injectable so
    tests don't actually wait.
    """
    last_exc: Exception
    for attempt in range(attempts):
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code not in _RETRY_STATUSES or attempt == attempts - 1:
                raise
            sleep(_backoff_delay(attempt, exc.response))
            continue
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            sleep(_backoff_delay(attempt, None))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest
    raise last_exc  # pragma: no cover - loop either returns or raises above


def cached_model_path(model_name: str, fallback_dir: Path) -> Path:
    """Resolve where the model file should live.

    Uses ``$SHRIKE_TEST_MODEL_DIR`` when set — a stable, cache-restored directory
    in CI so the model is fetched at most once — otherwise *fallback_dir* (a
    per-session temp dir locally). The directory is created if missing.
    """
    root = os.environ.get("SHRIKE_TEST_MODEL_DIR")
    base = Path(root) if root else fallback_dir
    base.mkdir(parents=True, exist_ok=True)
    return base / model_name


def _cached_model_dir(fallback_dir: Path, dir_name: str, files: dict[str, str]) -> Path:
    """Resolve (and populate) an ONNX model directory.

    Like :func:`cached_model_path`, uses ``$SHRIKE_TEST_MODEL_DIR`` when set (a
    stable, CI-cached dir) else *fallback_dir*. Downloads each of *files* into
    ``<base>/<dir_name>/`` with retry/backoff if missing, and returns that directory
    (pass it to ``--embedding-model`` with the onnx backend).
    """
    root = os.environ.get("SHRIKE_TEST_MODEL_DIR")
    base = Path(root) if root else fallback_dir
    model_dir = base / dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in files.items():
        dest = model_dir / name
        if not (dest.exists() and dest.stat().st_size > 0):
            download_with_retry(url, dest)
    return model_dir


def cached_onnx_model_dir(fallback_dir: Path) -> Path:
    """The pinned MiniLM int8 ONNX model dir (384-dim, BERT/WordPiece)."""
    return _cached_model_dir(fallback_dir, ONNX_MODEL_DIR_NAME, ONNX_MODEL_FILES)


def cached_distilroberta_model_dir(fallback_dir: Path) -> Path:
    """The pinned DistilRoBERTa int8 ONNX model dir (768-dim, BPE, no ``[PAD]``)."""
    return _cached_model_dir(fallback_dir, DISTILROBERTA_MODEL_DIR_NAME, DISTILROBERTA_MODEL_FILES)


def cached_onnx_fp32_model_dir(fallback_dir: Path) -> Path:
    """The pinned fp32 MiniLM ONNX model dir (384-dim, batches bit-exact)."""
    return _cached_model_dir(fallback_dir, ONNX_FP32_MODEL_DIR_NAME, ONNX_FP32_MODEL_FILES)


def cached_clip_model_dir(fallback_dir: Path) -> Path:
    """The pinned small CLIP dir (int8 text+vision graphs, 512-dim shared space)."""
    return _cached_model_dir(fallback_dir, CLIP_MODEL_DIR_NAME, CLIP_MODEL_FILES)
