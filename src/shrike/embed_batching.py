"""Empirical batch-safety self-check for embedding backends.

Some backends produce a *different* vector for a note depending on what else is in its
batch. The clearest case is dynamically-quantized int8 ONNX models: onnxruntime computes
a per-tensor activation scale over the *whole batch tensor*, so a batch-mate's content
shifts every element (~0.06). That breaks the index's invariant that a note's vector is a
pure function of its text — which is what makes a ``reconcile``'s end state identical to a
full rebuild. Non-quantized models (fp32/fp16) and llama-server batch deterministically.

Rather than guess from a model's quantization scheme, every backend probes this at startup:
embed a fixed set of varied texts **serially** (the reference) and **all in one batch** —
the largest, most heterogeneous batch, which maximizes any batch-variance — and compare. If
they match within a tolerance, the model is safe to batch *up to the probe-set size* (which
the caller then honours as the cap); if not, it embeds serially.

Two deliberate choices make the check trustworthy rather than wishful:

- **The probe set is "spiked" for activation magnitude, not just length.** int8 drift on a
  text T is maximized when T is calm (no outlier activations) and a batch-mate is spiky
  (drives a large activation range, moving the per-tensor min/max). So the set mixes calm
  anchors with deliberately spiky inputs (long, numeric/hex/code, symbol soup, a degenerate
  repeated token, mixed-script/emoji, ALL CAPS). An fp model has no activation quant, so its
  batched-vs-serial drift is exactly 0 regardless of content — spiking only raises
  sensitivity to variant models, never false-positives a safe one. The set's sensitivity is
  pinned by a test against the real int8 fixtures (`test_onnx_models.py`).
- **We probe the batch size we actually use.** The probe compares against *one* batch of all
  probe texts, and the caller never batches larger than the probe-set size — so "proven safe"
  and "what we do" are the same size (no extrapolating from a small sweep to a larger runtime
  batch). This is empirical, not a proof for *every* possible model; see `docs/decisions.md`
  for the heuristic caveat and the ONNX-specific deterministic fallback (scan for
  `DynamicQuantizeLinear`/`MatMulInteger`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import shrike_native

# The spiked probe set and tolerance are SOURCED FROM THE NATIVE CONTRACT
# (shrike-engine-api's probe module, #342 P4) — one set, two hosts: native
# hosts run the Rust probe; this module is the Python host's adapter of the
# same policy over any EmbedChunk callable (facades, fakes, HTTP clients).
BATCH_DRIFT_TOL: float = shrike_native.BATCH_DRIFT_TOL

BATCH_PROBE_TEXTS: list[str] = list(shrike_native.BATCH_PROBE_TEXTS)

# Embeds a list of texts as a single batch, returning one vector per input.
EmbedChunk = Callable[[list[str]], list[list[float]]]

# How many times to (re)run the probe before giving up. The probe issues many embed calls
# (a serial reference + one batch); a single transient failure shouldn't condemn a session
# to serial, so we retry the whole probe before raising. 3 is a cheap trade for better
# tolerance of a flaky embedder during startup.
PROBE_ATTEMPTS = 3


class ProbeError(RuntimeError):
    """The batch-safety probe could not complete (every attempt's embed calls failed)."""


def probe_max_safe_batch(
    embed_chunk: EmbedChunk,
    *,
    tol: float = BATCH_DRIFT_TOL,
    probe_texts: Sequence[str] | None = None,
    attempts: int = PROBE_ATTEMPTS,
) -> int:
    """Return the batch size proven safe (the probe-set size) or 1 (embed serially).

    Embeds each probe text **alone** (the serial reference), then all of them in **one batch**
    — the largest, most heterogeneous batch — and compares (max-abs per element). Match within
    *tol* → the model batches deterministically and is safe up to the probe-set size (the
    caller caps there); mismatch → 1.

    The two failure modes are kept distinct. The **serial reference** is what the model must be
    able to do at all; it's retried up to *attempts* times and a persistent failure raises
    :class:`ProbeError` (so the caller can fail loud — e.g. a model that needs an input we don't
    supply). A **batch-only** failure (the serial reference succeeded but the batched call
    didn't — e.g. a graph fixed to batch size 1) is *not* an error: it returns 1, embedding
    serially.
    """
    texts = list(probe_texts if probe_texts is not None else BATCH_PROBE_TEXTS)
    if len(texts) < 2:
        return 1
    reference: np.ndarray | None = None
    last_exc: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            reference = np.asarray([embed_chunk([t])[0] for t in texts], dtype=np.float64)
            break
        except Exception as e:  # noqa: BLE001 — retry the serial reference, then surface
            last_exc = e
    if reference is None:
        raise ProbeError(f"serial embedding failed after {attempts} attempt(s): {last_exc}")
    # The model can embed serially. Does it also batch deterministically? A batch-only
    # failure (e.g. a fixed batch-1 graph) degrades to serial rather than erroring.
    try:
        batched = np.asarray(embed_chunk(texts), dtype=np.float64)
    except Exception:  # noqa: BLE001 — can embed serially but not batched → serial
        return 1
    drift = float(np.max(np.abs(reference - batched)))
    return len(texts) if drift <= tol else 1


def max_probe_drift(
    embed_chunk: EmbedChunk,
    *,
    probe_texts: Sequence[str] | None = None,
) -> float:
    """Max-abs serial-vs-batched drift over the probe set — for sensitivity tests."""
    texts = list(probe_texts if probe_texts is not None else BATCH_PROBE_TEXTS)
    reference = np.asarray([embed_chunk([t])[0] for t in texts], dtype=np.float64)
    batched = np.asarray(embed_chunk(texts), dtype=np.float64)
    return float(np.max(np.abs(reference - batched)))
