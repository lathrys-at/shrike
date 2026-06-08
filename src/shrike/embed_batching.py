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
# safe" and "what we batch" stay the same size. Sized to 64 — the index's `BATCH_SIZE` chunk
# — so a probe-safe (fp / non-dynamic-quant) model batches at the full chunk a GPU favours;
# the cost is a one-time serial reference at startup (trivial in-process; ~0.7 s for a remote
# llama-server). Batching past 64 would also need `index.BATCH_SIZE` raised — a later slice.
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
    # --- second half (to reach the 64-text / batch-64 ceiling) ---
    # More calm anchors.
    "The speed of sound in dry air is about 343 meters per second.",
    "Shakespeare wrote thirty-seven plays and over one hundred fifty sonnets.",
    "Water is composed of two hydrogen atoms and one oxygen atom.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "A leap year occurs every four years to keep the calendar aligned.",
    "Insulin regulates the level of glucose in the bloodstream.",
    "The Great Wall of China stretches thousands of kilometers.",
    "Gravity causes objects to accelerate toward the center of the Earth.",
    "The human heart pumps roughly five liters of blood per minute.",
    "Tectonic plates drift slowly across the planet's molten mantle.",
    "Vaccines train the immune system to recognize specific pathogens.",
    "Electrons occupy discrete energy levels around an atomic nucleus.",
    "The periodic table organizes elements by their atomic number.",
    # More numbers / code.
    "def f(x): return x**2 + 3*x - 7  # a simple quadratic",
    "git commit -m 'fix: off-by-one in loop bound' && git push origin main",
    "3.141592653589793 2.718281828459045 1.618033988749895 0.5772156649",
    "IPv4 192.168.0.1 IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 :8080",
    # More symbol / math / structured.
    "∮ E·dl = -dΦ/dt    ∇×B = μ₀J + μ₀ε₀ ∂E/∂t",
    '{ "key": [1, 2, 3], "nested": { "a": true, "b": null } }',
    # More mixed script / emoji.
    "中文 العربية हिन्दी ไทย Ελληνικά עברית 한국어 русский язык",
    "😀😃😄😁😆😅😂🤣☺️😊😇🙂🙃😉😌😍🥰😘🤗",
    "Zürich Köln München São Paulo Bogotá Reykjavík İstanbul",
    # More long.
    "in computer science a hash table implements an associative array abstract data type, a "
    "structure that maps keys to values using a hash function to compute an index into an array "
    "of buckets from which the desired value can be found, offering average constant-time "
    "complexity for insertion, deletion, and lookup under a good hash distribution and load",
    "a regular expression is a sequence of characters that specifies a search pattern in text, "
    "used by string-searching algorithms for find and replace operations or input validation, "
    "and supported with varying syntax across editors, command-line tools, and programming "
    "languages from grep and sed to Perl, Python, and the Rust regex crate among many others",
    # More ALL CAPS.
    "BREAKING NEWS MARKETS RALLY AS INFLATION COOLS FASTER THAN EXPECTED",
    "TODO FIXME XXX HACK NOTE WARNING DEPRECATED REVIEW",
    # More degenerate / repeated.
    "na na na na na na na na na na na batman",
    "0000000000000000000000000000000000000000",
    # More short.
    "ok",
    "42",
    # Misc.
    "e = mc²   F = ma   PV = nRT   a² + b² = c²",
    "colorless green ideas sleep furiously",
]

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
