"""Empirical batch-safety self-check for embedding backends.

Some backends produce a *different* vector for a note depending on what else is in its
batch. The clearest case is dynamically-quantized int8 ONNX models: onnxruntime computes
activation scales over the whole batch tensor, so a batch-mate's content shifts every
element (~0.06). That breaks the index's invariant that a note's vector is a pure function
of its text — which is what makes a ``reconcile``'s end state identical to a full rebuild.
Non-quantized models (fp32/fp16) and llama-server batch deterministically.

Rather than guess from a model's quantization scheme, every backend probes this at startup:
embed a fixed set of varied texts serially (the reference) and at a few batch sizes, and
compare. The largest batch size whose results match the reference within a tolerance is the
backend's safe batch size; 1 means "batch-variant — embed serially."
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Tolerance for "batched == serial". Sits well above float-reduction noise (llama-server
# ~4e-5, fp32 ONNX exactly 0) and far below dynamic-int8 batch drift (~0.06), so it cleanly
# separates a batch-safe backend from a batch-variant one.
BATCH_DRIFT_TOL = 1e-3

# Varied probe texts (mixed lengths, so batching actually pads). The content is irrelevant —
# only whether a text's vector changes with its batch-mates.
BATCH_PROBE_TEXTS: list[str] = [
    "a",
    "the cell",
    "mitochondria produce ATP",
    "What is the capital of France?",
    "Spaced repetition strengthens long-term memory through timed review.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "x",
    "two words",
    "Newton's second law relates force, mass, and acceleration.",
    "An integral accumulates a quantity over an interval.",
    "DNA encodes genetic information in sequences of nucleotides.",
    "short",
    "a slightly longer sentence than the previous one here",
    "Quantum entanglement links the states of two particles.",
    "Supply and demand determine prices in a competitive market.",
    "The mitochondrion is often called the powerhouse of the cell.",
]

# Embeds a list of texts as a single batch, returning one vector per input.
EmbedChunk = Callable[[list[str]], list[list[float]]]


def _chunked(embed_chunk: EmbedChunk, texts: list[str], size: int) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), size):
        out.extend(embed_chunk(texts[i : i + size]))
    return out


def probe_max_safe_batch(
    embed_chunk: EmbedChunk,
    *,
    sizes: Sequence[int] = (2, 4, 8, 16),
    tol: float = BATCH_DRIFT_TOL,
    probe_texts: list[str] | None = None,
) -> int:
    """Return the largest batch size at which *embed_chunk* matches serial embedding.

    Embeds the probe texts serially (the reference), then chunked at each candidate *size*,
    and returns the largest size whose results match the reference within *tol* (max-abs per
    element). Returns 1 ("embed serially") if even size 2 diverges. By the mechanism this is
    effectively binary (safe-at-2 implies safe), but the sweep guards against a size-dependent
    surprise — the first size that diverges stops the search, since larger sizes can't be safer.
    """
    texts = probe_texts if probe_texts is not None else BATCH_PROBE_TEXTS
    reference = np.asarray([embed_chunk([t])[0] for t in texts], dtype=np.float64)
    safe = 1
    for size in sizes:
        if size > len(texts):
            break
        batched = np.asarray(_chunked(embed_chunk, texts, size), dtype=np.float64)
        if float(np.max(np.abs(reference - batched))) <= tol:
            safe = size
        else:
            break
    return safe
