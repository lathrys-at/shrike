"""Tests for the batch-safety probe (shrike.embed_batching)."""

from __future__ import annotations

from shrike.embed_batching import BATCH_PROBE_TEXTS, probe_max_safe_batch


def _vec(text: str) -> list[float]:
    """A deterministic, batch-independent vector for a text."""
    return [float(len(text)), float(sum(map(ord, text)) % 97)]


def _deterministic(texts: list[str]) -> list[list[float]]:
    """Batch-safe: each vector is a pure function of its own text."""
    return [_vec(t) for t in texts]


def _variant(texts: list[str]) -> list[list[float]]:
    """Batch-variant: every vector is shifted by the batch size (like int8 dynamic quant)."""
    bias = float(len(texts))
    return [[_vec(t)[0] + bias, _vec(t)[1]] for t in texts]


def _safe_only_to_2(texts: list[str]) -> list[list[float]]:
    """Safe at batch sizes 1-2, variant at 3+ (exercises the partial/stop path)."""
    bias = 1.0 if len(texts) > 2 else 0.0
    return [[_vec(t)[0] + bias, _vec(t)[1]] for t in texts]


def test_deterministic_backend_is_safe_to_max() -> None:
    assert probe_max_safe_batch(_deterministic) == 16


def test_variant_backend_falls_back_to_serial() -> None:
    assert probe_max_safe_batch(_variant) == 1


def test_partial_safety_returns_last_safe_size() -> None:
    # Diverges at size 4, so the largest proven-safe size is 2.
    assert probe_max_safe_batch(_safe_only_to_2) == 2


def test_tolerance_absorbs_float_noise() -> None:
    # A tiny per-call jitter (well under tol) must still read as safe.
    counter = {"n": 0}

    def jittery(texts: list[str]) -> list[list[float]]:
        counter["n"] += 1
        eps = 1e-6 * counter["n"]
        return [[_vec(t)[0] + eps, _vec(t)[1]] for t in texts]

    assert probe_max_safe_batch(jittery) == 16


def test_probe_texts_are_varied() -> None:
    # Mixed lengths so batching actually pads (the condition that triggers variance).
    lengths = {len(t) for t in BATCH_PROBE_TEXTS}
    assert len(BATCH_PROBE_TEXTS) >= 16
    assert len(lengths) >= 8
