"""Tests for the batch-safety probe (shrike.embed_batching)."""

from __future__ import annotations

import numpy as np
import pytest

from shrike.harness.engines.embedding.batching import (
    BATCH_DRIFT_TOL,
    BATCH_PROBE_IMAGES,
    BATCH_PROBE_TEXTS,
    ProbeError,
    _probe_items,
    max_probe_drift,
    probe_image_max_safe_batch,
    probe_max_safe_batch,
)

_N = len(BATCH_PROBE_TEXTS)


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


def _variant_only_when_large(texts: list[str]) -> list[list[float]]:
    """Deterministic alone and in small batches, but variant once the batch is large — the
    case an escalating 2/4/8 sweep could pass while the real (full-batch) runtime diverges."""
    bias = 1.0 if len(texts) >= 4 else 0.0
    return [[_vec(t)[0] + bias, _vec(t)[1]] for t in texts]


def test_deterministic_backend_is_safe_to_set_size() -> None:
    assert probe_max_safe_batch(_deterministic) == _N


def test_variant_backend_falls_back_to_serial() -> None:
    assert probe_max_safe_batch(_variant) == 1


def test_variant_only_at_large_batch_is_caught() -> None:
    # The probe compares against one *full* batch, so a model that only diverges when batched
    # large is classified serial — the soundness gap a small escalating sweep would miss.
    assert probe_max_safe_batch(_variant_only_when_large) == 1


def test_tolerance_absorbs_float_noise() -> None:
    counter = {"n": 0}

    def jittery(texts: list[str]) -> list[list[float]]:
        counter["n"] += 1
        eps = 1e-6 * counter["n"]
        return [[_vec(t)[0] + eps, _vec(t)[1]] for t in texts]

    assert probe_max_safe_batch(jittery) == _N


def test_retry_survives_a_transient_failure() -> None:
    raised = {"once": False}

    def flaky(texts: list[str]) -> list[list[float]]:
        if not raised["once"]:
            raised["once"] = True
            raise RuntimeError("transient blip")
        return _deterministic(texts)

    # First attempt aborts on the blip; the retry completes cleanly.
    assert probe_max_safe_batch(flaky) == _N


def test_raises_after_persistent_failure() -> None:
    def broken(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder down")

    with pytest.raises(ProbeError):
        probe_max_safe_batch(broken)


def test_serial_reference_failure_raises_not_serial() -> None:
    # If even a single-text embed fails, the model can't be driven at all → ProbeError
    # (so a backend can fail loud), never a silent serial classification.
    def serial_broken(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("missing required input")

    with pytest.raises(ProbeError):
        probe_max_safe_batch(serial_broken)


def test_batch_only_failure_falls_back_to_serial() -> None:
    # Serial reference embeds fine; only the batched call fails (e.g. a fixed batch-1 graph).
    # That degrades to serial (1), not an error.
    def batch_only_broken(texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            raise RuntimeError("model only supports batch size 1")
        return _deterministic(texts)

    assert probe_max_safe_batch(batch_only_broken) == 1


def test_max_probe_drift() -> None:
    assert max_probe_drift(_deterministic) == 0.0
    assert max_probe_drift(_variant) > BATCH_DRIFT_TOL


def test_probe_texts_are_spiked() -> None:
    # Spiked for activation magnitude (not just length): a long text, numeric/symbolic, and
    # non-ASCII content — the inputs that actually drive int8 batch-variance.
    assert _N >= 12
    assert any(len(t) > 200 for t in BATCH_PROBE_TEXTS)  # long, wide activation profile
    assert any(any(c.isdigit() for c in t) for t in BATCH_PROBE_TEXTS)  # numeric/hex
    assert any(not t.isascii() for t in BATCH_PROBE_TEXTS)  # mixed script / emoji


def test_probe_ceiling_covers_index_chunk() -> None:
    # The probe-set size is the batch ceiling, and the kernel hands embed calls chunks of up
    # to its BATCH_SIZE (shrike_kernel::index_orchestrator::BATCH_SIZE, mirrored here — the
    # facade-era shrike.index.BATCH_SIZE retired with #355). If the set were smaller, a
    # probe-safe model would be capped below the chunk — never incorrect, but a silent
    # throughput regression. Pin the "probe at the size we use".
    kernel_embed_chunk = 64

    assert len(BATCH_PROBE_TEXTS) >= kernel_embed_chunk


# -- vision probe (#211) -----------------------------------------------------

_NI = len(BATCH_PROBE_IMAGES)


def _img_vec(image: bytes) -> list[float]:
    """A deterministic, batch-independent vector for an image (keyed on content)."""
    return [float(len(image)), float(sum(image[:16]) % 97)]


def _img_deterministic(images: list[bytes]) -> list[list[float]]:
    """Batch-safe vision graph: each vector is a pure function of its own image."""
    return [_img_vec(im) for im in images]


def _img_variant(images: list[bytes]) -> list[list[float]]:
    """Batch-variant vision graph: every vector shifted by the batch size (int8-style)."""
    bias = float(len(images))
    return [[_img_vec(im)[0] + bias, _img_vec(im)[1]] for im in images]


def test_image_probe_set_is_real_images() -> None:
    # The set is non-trivial and every entry is real image bytes (a BMP magic 'BM').
    assert _NI >= 12
    assert all(im[:2] == b"BM" for im in BATCH_PROBE_IMAGES)
    # Heterogeneous content → distinct bytes, so the probe maximizes batch variance.
    assert len(set(BATCH_PROBE_IMAGES)) == _NI


def test_image_probe_set_matches_text_ceiling() -> None:
    # Sized to the text set so a uniform-safe CLIP pair's min(text, vision) never lowers
    # the batch (both probe to the same ceiling); only a variant path collapses it to 1.
    assert len(BATCH_PROBE_TEXTS) == _NI


def test_deterministic_vision_is_safe_to_set_size() -> None:
    assert probe_image_max_safe_batch(_img_deterministic) == _NI


def test_variant_vision_falls_back_to_serial() -> None:
    assert probe_image_max_safe_batch(_img_variant) == 1


def test_vision_serial_reference_failure_raises() -> None:
    def serial_broken(images: list[bytes]) -> list[list[float]]:
        raise RuntimeError("vision graph needs an input we don't supply")

    with pytest.raises(ProbeError):
        probe_image_max_safe_batch(serial_broken)


def test_vision_batch_only_failure_falls_back_to_serial() -> None:
    def batch_only_broken(images: list[bytes]) -> list[list[float]]:
        if len(images) > 1:
            raise RuntimeError("vision graph only supports batch size 1")
        return _img_deterministic(images)

    assert probe_image_max_safe_batch(batch_only_broken) == 1


# --- #602 S11b sibling check: the Python probe lane is NaN-safe (PASSES today) ---
# These confirm there is no Python-side twin of the Rust-only NaN-weight drift
# concern: a model that emits NaN only under batching is classified serial, and
# numpy's max propagates NaN so the `drift <= tol` comparison is False (→ variant
# → serial), never silently "safe". They are passing controls, not regressions.


def test_python_probe_treats_nan_drift_as_unsafe() -> None:
    dim = 4

    def nan_under_batching(items: list[str]) -> list[list[float]]:
        if len(items) == 1:
            return [[1.0] * dim]
        out = [[1.0] * dim for _ in items]
        out[1][0] = float("nan")
        return out

    items = [f"t{i}" for i in range(4)]
    safe = _probe_items(items, nan_under_batching, tol=1e-3, attempts=3)
    assert safe == 1, "a NaN-under-batching model must be classified serial (batch-variant)"


def test_numpy_max_propagates_nan_sanity() -> None:
    ref = np.array([[1.0, 2.0]], dtype=np.float64)
    batched = np.array([[1.0, float("nan")]], dtype=np.float64)
    drift = float(np.max(np.abs(ref - batched)))
    assert drift != drift  # NaN
    assert not (drift <= 1e-3)  # → variant → safe
