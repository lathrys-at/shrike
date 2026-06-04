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
