"""S11a-1 repro (preserved by lead; rev-S11a worktree reaped).
BACKEND_ALIASES applied at construction (embedding.py:632) but NOT on the
start(backend=...) override path (embedding.py:738-739) → a documented alias
('onnx-rs'/'clip-rs') 400s on /embedding/start AND permanently poisons
_backend_kind (mutate-before-validate). RED at fa54f8c (2 failed).
Also includes the S11b-1 Python-sibling check (PASSES → Python lane safe).
Run: SHRIKE_SKIP_NATIVE_STALE_CHECK=1 .venv/bin/python -m pytest <this> -q -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from shrike.embed_batching import _probe_items
from shrike.embedding import EmbeddingRuntime


def test_alias_normalized_at_construction_but_not_on_start_override():
    rt = EmbeddingRuntime(backend="onnx-rs", model="/nonexistent")
    assert rt.backend_kind == "onnx"  # construction normalizes
    rt2 = EmbeddingRuntime(backend="llama", model="/nonexistent")
    with pytest.raises(Exception) as ei:
        rt2.start(backend="onnx-rs")
    msg = str(ei.value)
    assert "Unknown embedding backend" not in msg, (
        f"alias 'onnx-rs' must normalize on the start() override path too; got: {msg}"
    )


def test_bad_alias_override_poisons_backend_kind_for_subsequent_starts():
    rt = EmbeddingRuntime(backend="onnx", model="/nonexistent")
    assert rt.backend_kind == "onnx"
    with pytest.raises(ValueError):
        rt.start(backend="onnx-rs")
    assert rt.backend_kind in ("onnx", "clip", "llama", "remote"), (
        f"_backend_kind corrupted to {rt.backend_kind!r} after a bad-alias override"
    )


# S11b-1 Python-sibling check — PASSES (Python lane SAFE, no shared bug):
def test_python_probe_treats_nan_drift_as_unsafe():
    dim = 4
    def nan_under_batching(items):
        if len(items) == 1:
            return [[1.0] * dim]
        out = [[1.0] * dim for _ in items]
        out[1][0] = float("nan")
        return out
    items = [f"t{i}" for i in range(4)]
    safe = _probe_items(items, nan_under_batching, tol=1e-3, attempts=3)
    assert safe == 1, "a NaN-under-batching model must be classified serial (batch-variant)"


def test_numpy_max_propagates_nan_sanity():
    ref = np.array([[1.0, 2.0]], dtype=np.float64)
    batched = np.array([[1.0, float("nan")]], dtype=np.float64)
    drift = float(np.max(np.abs(ref - batched)))
    assert drift != drift  # NaN
    assert not (drift <= 1e-3)  # → variant → safe
