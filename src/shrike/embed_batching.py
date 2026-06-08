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

# Tolerance for "batched == serial". Sits well above float-reduction noise (llama-server
# ~4e-5, fp32 ONNX exactly 0) and far below dynamic-int8 batch drift (~0.06), so it cleanly
# separates a batch-safe backend from a batch-variant one.
BATCH_DRIFT_TOL = 1e-3

# Probe texts, spiked for activation magnitude (see module docstring). The content is
# irrelevant to the result — only whether a text's vector changes with its batch-mates —
# but the spread of magnitudes is what makes a variant model actually diverge here. The
# set's *length* is also the ceiling: the probe verifies a batch of exactly this many texts
# and the backend never batches larger (it caps `--embedding-batch-size` here), so "proven
# safe" and "what we batch" stay the same size. 32 trades a fully-amortized batch for a
# still-trivial in-process startup cost; grow it if a GPU/multimodal path wants larger.
BATCH_PROBE_TEXTS: list[str] = [
    # Calm anchors (real-note-shaped, low activation range).
    "mitochondria are the powerhouse of the cell",
    "Spaced repetition strengthens long-term memory through timed review.",
    "Newton's second law relates force, mass, and acceleration.",
    "An integral accumulates a quantity over an interval.",
    "DNA encodes genetic information in sequences of nucleotides.",
    "Supply and demand determine prices in a competitive market.",
    "Mitochondrial DNA is inherited maternally in most animals.",
    "The boiling point of water at sea level is 100 degrees Celsius.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog repeatedly.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    # Long (a wide activation profile over many tokens).
    "the derivative measures the instantaneous rate of change of a function with respect to "
    "its variable, while the definite integral accumulates the signed area under a curve over "
    "an interval, and together by the fundamental theorem of calculus they form inverse "
    "operations that underpin much of classical analysis and its applications in physics, "
    "engineering, economics, and statistics across countless practical and theoretical settings",
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt "
    "ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation",
    # Numbers / hex / code (outlier-prone token embeddings).
    "0xDEADBEEF 1234567890 SELECT * FROM t WHERE id=42 AND ratio=3.14159265;",
    "SELECT id, name FROM users WHERE created_at > '2020-01-01' ORDER BY name DESC;",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
    "kHz MHz GHz THz 3.0e8 m/s 6.022e23 1.602e-19 9.81 273.15 -40",
    # Markup / structured.
    "<html><body><h1>Title</h1><p>paragraph &amp; entity</p></body></html>",
    "user@example.com +1-555-0123 https://test.org/page#anchor?q=1&r=2",
    "https://example.com/path?q=1&r=2#frag 5f4dcc3b5aa765d61d8327deb882cf99",
    # Symbol / math soup.
    "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./",
    "Σ Δ Ω α β γ ∫ ∂ ∇ √ ∞ ≈ ≠ ≤ ≥ ± × ÷ → ⇒ ∈ ∀ ∃",
    # Degenerate / repeated tokens (a spike with a tiny activation range of its own).
    "the the the the the the the the the the the the",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    # Mixed script / emoji (rare tokens, large embedding norms).
    "café 日本語 Привет мир 🧠🔬🧬 naïve résumé Größe",
    "🎉🎊🥳 congratulations on completing the course 🏆✨🚀",
    "ПриветПривет こんにちは こんにちは 안녕하세요 你好世界",
    # ALL CAPS.
    "URGENT WARNING SYSTEM FAILURE IMMINENT EVACUATE THE BUILDING NOW",
    "ALL CAPS SHORT",
    # Short.
    "x",
]

# Embeds a list of texts as a single batch, returning one vector per input.
EmbedChunk = Callable[[list[str]], list[list[float]]]

# How many times to (re)run the probe before giving up. The probe issues many embed calls
# (a serial reference + one batch); a single transient failure shouldn't condemn a session
# to serial, so we retry the whole probe before raising.
PROBE_ATTEMPTS = 2


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

    Embeds each probe text **alone** (the reference), then all of them in **one batch** —
    the largest, most heterogeneous batch — and compares (max-abs per element). Match within
    *tol* → the model batches deterministically and is safe up to the probe-set size (the
    caller caps there); mismatch → 1. Retries the whole probe up to *attempts* times on a
    transient embed failure, raising :class:`ProbeError` only if every attempt fails.
    """
    texts = list(probe_texts if probe_texts is not None else BATCH_PROBE_TEXTS)
    if len(texts) < 2:
        return 1
    last_exc: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            reference = np.asarray([embed_chunk([t])[0] for t in texts], dtype=np.float64)
            batched = np.asarray(embed_chunk(texts), dtype=np.float64)
        except Exception as e:  # noqa: BLE001 — any embed failure is retried, then surfaced
            last_exc = e
            continue
        drift = float(np.max(np.abs(reference - batched)))
        return len(texts) if drift <= tol else 1
    raise ProbeError(f"batch-safety probe failed after {attempts} attempt(s): {last_exc}")


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
