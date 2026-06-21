"""Resilient, cache-friendly fetch of the test embedding model.

Re-downloading the GGUF model on every run with no retry and no persistent cache
draws HuggingFace ``429 Too Many Requests`` rate-limits, erroring the whole lane
at setup. This module adds backoff on transient HTTP failures and lets the
download be reused from a stable directory (populated by ``actions/cache`` in CI).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx


def default_model_cache_base() -> Path:
    """The shared dev-cache models dir (the ``$SHRIKE_TEST_MODEL_DIR``-unset default):
    ``~/.cache/shrike-dev/models`` (``$XDG_CACHE_HOME`` honored). Fixture models are
    expensive and checkout-independent, so they live in the shared dev cache — NOT
    per-checkout, ``~/.cache`` top-level, or a per-session ``/tmp`` dir (CLAUDE.md
    "Cacheable dev artifacts"). Each model is keyed by its pinned dir-name/filename,
    so two checkouts pinning different models never collide. Pass this where a
    *fallback_dir* is taken; the ``cached_*`` resolvers give ``$SHRIKE_TEST_MODEL_DIR``
    precedence over it (CI / cross-checkout overrides set it)."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "shrike-dev" / "models"


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

# A second, architecturally-different ONNX model for the embedding lane:
# all-distilroberta-v1 is DistilRoBERTa — **768-dim** (not MiniLM's 384,
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

# The fp32 (non-quantized) export of the same MiniLM. Unlike the int8 model it
# batches **bit-exact** (no dynamic activation quantization), so it's the fixture that
# proves the batch-safety probe enables batching where it's safe. ~86 MB.
ONNX_FP32_MODEL_DIR_NAME = "all-MiniLM-L6-v2-onnx-fp32"
ONNX_FP32_MODEL_FILES = {
    "model.onnx": "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
    "tokenizer.json": "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
}

# The small CLIP (image<->text) for the clip backend: clip-vit-base-patch32 at q4
# (4-bit) weight quantization — a dual-encoder (separate text + vision graphs) + the
# CLIP tokenizer + image-preprocessing config, stored flat (the backend's
# _resolve_files finds them at the dir root via the `variant` suffix). Same pure-ViT
# model as the int8 export but ~half the bytes; the tokenizer + preprocessor are
# byte-identical to the int8 export. jina-clip-v2 is the production-quality option;
# this is the CI/test fixture.
CLIP_MODEL_DIR_NAME = "clip-vit-base-patch32-onnx-q4"
_CLIP_BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
CLIP_MODEL_FILES = {
    "text_model_q4.onnx": f"{_CLIP_BASE}/onnx/text_model_q4.onnx",
    "vision_model_q4.onnx": f"{_CLIP_BASE}/onnx/vision_model_q4.onnx",
    "tokenizer.json": f"{_CLIP_BASE}/tokenizer.json",
    "preprocessor_config.json": f"{_CLIP_BASE}/preprocessor_config.json",
}

# -- Wave-2 offline-integration profile models -------------------------------
#
# These back the pure-ONNX multi-space profile (scripts/profiles/onnx-multispace.yml:
# embeddinggemma + MobileCLIP2), plus jina-clip-v2 pre-staged for native
# fused-graph ClipBackend support (not consumed by any current profile). They are
# NOT used by any CI test (the per-PR lane never downloads multi-GB models); they
# exist so //scripts:serve_<profile> can
# materialize them by dir-name (Bazel runfiles under `bazel run`, this fetch source
# off Bazel). Each is sha-pinned (the sha256s are the HuggingFace LFS oids, or the
# byte sha256 for a git-stored file) and warm-cache restorable like the other
# model_* externals. Bump the URLs + dir name + sha together to change one.

# EmbeddingGemma-300m ONNX (text-only, 768-dim, mean pooling, 2048 ctx) — the text
# leg of the onnx-multispace profile. The export ships ONLY quantized graphs (no
# plain model.onnx), each as a tiny graph stub PLUS an external weight-data file:
# OnnxBackend's variant-suffix resolver finds model_quantized.onnx, and onnxruntime
# loads the co-located model_quantized.onnx_data transparently (the external-data
# landmine — BOTH files must materialize). int8 dynamic-quant, so the batch-safety
# probe embeds it serially. ~309 MB weights + 20 MB tokenizer. Apache-2.0 base
# (Gemma terms); ONNX export by onnx-community. The .onnx_data filename MUST stay
# `model_quantized.onnx_data` (the graph references it by that exact relative name).
EMBEDDINGGEMMA_MODEL_DIR_NAME = "embeddinggemma-300m-onnx-int8"
_EMBEDDINGGEMMA_BASE = "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main"
EMBEDDINGGEMMA_MODEL_FILES = {
    "model_quantized.onnx": f"{_EMBEDDINGGEMMA_BASE}/onnx/model_quantized.onnx",
    "model_quantized.onnx_data": f"{_EMBEDDINGGEMMA_BASE}/onnx/model_quantized.onnx_data",
    "tokenizer.json": f"{_EMBEDDINGGEMMA_BASE}/tokenizer.json",
}

# MobileCLIP2-S2 ONNX (text+image, 512-dim shared space) — the image leg of the
# onnx-multispace profile. plhery/mobileclip2-onnx rev ba95759a loads through
# ClipBackend AS-IS (flat text_model.onnx + vision_model.onnx +
# preprocessor_config.json per size subdir, a repo-root tokenizer.json) for BOTH S0
# and S2. S2 is the better default for dogfooding: same 254 MB text encoder, a
# larger/better 143 MB vision encoder (vs S0's 45 MB) — +98 MB buys cleaner
# cross-modal separation (cat↔cat +0.2932 S2 vs +0.2657 S0). fp32 graphs
# (batch-cleared). Apple Sample Code License (apple-amlr) on the base weights.
# Pinned to an exact revision so the bytes never drift. The preprocessor +
# repo-root tokenizer are shared across sizes (byte-identical to S0).
MOBILECLIP2_MODEL_DIR_NAME = "mobileclip2-s2-onnx"
_MOBILECLIP2_REV = "ba95759a5bdbaca53e9111e2550a76ec09c8fd9e"
_MOBILECLIP2_BASE = f"https://huggingface.co/plhery/mobileclip2-onnx/resolve/{_MOBILECLIP2_REV}"
MOBILECLIP2_MODEL_FILES = {
    "text_model.onnx": f"{_MOBILECLIP2_BASE}/onnx/s2/text_model.onnx",
    "vision_model.onnx": f"{_MOBILECLIP2_BASE}/onnx/s2/vision_model.onnx",
    "preprocessor_config.json": f"{_MOBILECLIP2_BASE}/onnx/s2/preprocessor_config.json",
    "tokenizer.json": f"{_MOBILECLIP2_BASE}/tokenizer.json",
}

# jina-clip-v2 ONNX (text+image) — pre-staged for native fused-graph ClipBackend
# support. NOTE: the canonical jinaai/jina-clip-v2 ONNX export is a SINGLE FUSED
# model.onnx that takes text AND image inputs on every call — it does NOT ship the
# separate text_model.onnx + vision_model.onnx graphs ClipBackend requires, so it
# is NOT a drop-in for the dual-encoder contract as-published. The deferred native
# work to drive that fused graph will consume exactly this export. It is NOT
# consumed by any current profile — jina-text-clip uses MobileCLIP2 for its image
# leg. The dir/files below pin the combined int8 export + the root
# tokenizer/preprocessor. ~874 MB int8. Apache-2.0/CC.
JINA_CLIP_V2_MODEL_DIR_NAME = "jina-clip-v2-onnx-int8"
_JINA_CLIP_V2_BASE = "https://huggingface.co/jinaai/jina-clip-v2/resolve/main"
JINA_CLIP_V2_MODEL_FILES = {
    "model_quantized.onnx": f"{_JINA_CLIP_V2_BASE}/onnx/model_quantized.onnx",
    "tokenizer.json": f"{_JINA_CLIP_V2_BASE}/tokenizer.json",
    "preprocessor_config.json": f"{_JINA_CLIP_V2_BASE}/preprocessor_config.json",
}

# The small multimodal embedding model for the manual image-embed harness
# (test_multimodal.py): jina-embeddings-v5-omni nano — a text GGUF + a vision
# mmproj that embed text AND images into one 768-dim space. ~625 MB (431 MB
# text + 194 MB vision projector). NOT used by any CI test, for two reasons:
#   1. The model needs llama.cpp patches not upstream as of b9616 — the
#      official/pinned llama-server segfaults on image-embedding extraction.
#      The harness needs a binary built from jina-ai/llama.cpp `feat-v5-omni`.
#   2. The text GGUF must be the **F16** (unquantized) variant, NOT a K-quant.
#      The fork's encoder-combined-decode path reads the token-embedding tensor
#      element-by-element with `ggml_get_f32_1d`, which aborts on a
#      block-quantized type — so a K-quant (whose `token_embd.weight` is
#      quantized) crashes the server on the first image embed. Verified on
#      Q4_K_M and Q5_K_M (the quant jina's card recommends), both with and
#      without `--pooling last`; F16 keeps the table readable. The harness
#      also passes `--pooling last` (jina-v5-omni is a last-token model).
# This entry exists so the model is fetched into the SAME cache as every other
# fixture (pre-seed it with cached_multimodal_model_dir), not so CI downloads it.
MULTIMODAL_MODEL_DIR_NAME = "jina-embeddings-v5-omni-nano"
MULTIMODAL_TEXT_NAME = "jina-embeddings-v5-omni-nano-classification-F16.gguf"
MULTIMODAL_VISION_MMPROJ_NAME = "jina-embeddings-v5-omni-nano-classification-vision-mmproj-F16.gguf"
_MULTIMODAL_BASE = (
    "https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-classification-GGUF/resolve/main"
)
MULTIMODAL_MODEL_FILES = {
    MULTIMODAL_TEXT_NAME: f"{_MULTIMODAL_BASE}/{MULTIMODAL_TEXT_NAME}",
    MULTIMODAL_VISION_MMPROJ_NAME: f"{_MULTIMODAL_BASE}/{MULTIMODAL_VISION_MMPROJ_NAME}",
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
    in CI so the model is fetched at most once — otherwise *fallback_dir* (callers
    pass :func:`default_model_cache_base`). The directory is created if missing.
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
    """The pinned small CLIP dir (q4 text+vision graphs, 512-dim shared space)."""
    return _cached_model_dir(fallback_dir, CLIP_MODEL_DIR_NAME, CLIP_MODEL_FILES)


def cached_embeddinggemma_model_dir(fallback_dir: Path) -> Path:
    """The embeddinggemma-300m int8 ONNX dir (768-dim text, external weight data).

    Fetches BOTH model_quantized.onnx and its model_quantized.onnx_data companion
    (the external-data landmine) plus tokenizer.json. ~329 MB; only fetched when
    //scripts:serve_onnx_multispace fetches the onnx-multispace profile off Bazel, or when
    called directly to pre-seed the shared cache."""
    return _cached_model_dir(
        fallback_dir, EMBEDDINGGEMMA_MODEL_DIR_NAME, EMBEDDINGGEMMA_MODEL_FILES
    )


def cached_mobileclip2_model_dir(fallback_dir: Path) -> Path:
    """The MobileCLIP2-S2 ONNX dir (text+image, 512-dim shared space; ClipBackend).

    Flat text_model.onnx + vision_model.onnx + preprocessor_config.json + tokenizer.json
    (the layout ClipBackend loads as-is). ~399 MB."""
    return _cached_model_dir(fallback_dir, MOBILECLIP2_MODEL_DIR_NAME, MOBILECLIP2_MODEL_FILES)


def cached_jina_clip_v2_model_dir(fallback_dir: Path) -> Path:
    """The jina-clip-v2 int8 ONNX dir (combined model.onnx + tokenizer + preprocessor).

    Pre-staged for native fused-graph ClipBackend support; not consumed by any
    current profile. NOTE the combined-graph caveat in the module-level
    declaration: the published export is NOT a ClipBackend drop-in (single fused
    graph, not split text/vision). ~874 MB."""
    return _cached_model_dir(fallback_dir, JINA_CLIP_V2_MODEL_DIR_NAME, JINA_CLIP_V2_MODEL_FILES)


def cached_multimodal_model_dir(fallback_dir: Path) -> Path:
    """The jina-v5-omni nano dir (text GGUF + vision mmproj) for the manual
    image-embed harness. ~625 MB; only fetched when the harness actually runs
    (a patched llama-server is present) or when called directly to pre-seed the
    shared cache. Returns the dir; the two files are named by MULTIMODAL_TEXT_NAME /
    MULTIMODAL_VISION_MMPROJ_NAME."""
    return _cached_model_dir(fallback_dir, MULTIMODAL_MODEL_DIR_NAME, MULTIMODAL_MODEL_FILES)
