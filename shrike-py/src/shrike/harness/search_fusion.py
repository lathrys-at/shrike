"""Reciprocal Rank Fusion (RRF) — the signal-agnostic combiner behind multi-signal search.

`search_notes` blends several retrieval signals (semantic cosine, exact substring, n-gram fuzzy,
tag-centroid, per-modality semantic). Those live on incommensurable scales: cosine clusters in a
narrow ~0.3–0.7 band, exact match is near-binary, a per-modality cosine sits a constant offset
below within-modal. Normalize-and-sum inherits every pathology — a card's order wobbles with *what
else* was retrieved.

RRF sidesteps all of it by fusing on **rank position**, not raw score::

    score(note) = Σ_signals  w_s · 1 / (k + rank_s(note))        # rank_s is 1-based; k ≈ 60

Why it fits: it never reconciles a cosine-0.7 against a binary hit (magnitude is discarded); a note
absent from a signal contributes nothing (rank = ∞), which *is* the graceful degradation we want
for untagged / no-match cards; orderings are stable across queries; and a per-modality constant
offset is invisible to a rank-based combiner, so the multimodal rankers plug in with no
normalization. The one thing RRF gives up is magnitude — an exact literal hit should outrank a
merely-similar one regardless of its rank gap — so the combiner supports a *priority tier*
(`priority_signals`) that floats notes carrying a chosen signal above the rest, RRF-ordered within.

This module is pure: rankings of ints in, fused order out. No embedding / index / Anki deps.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# RRF dampening constant. ~60 is the standard from Cormack et al. — large enough that the gap
# between rank 1 and 2 isn't dramatic (a single signal can't unilaterally dominate the fusion),
# small enough that early ranks still matter. Becomes a `--search-*` knob with the tuning harness.
RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    """One note's fused result: its score and which signals contributed at what rank.

    ``signals`` maps each contributing signal name to the note's 1-based rank in that signal — the
    seam per-result provenance reads to report *why* a result surfaced and to debug/tune.
    """

    note_id: int
    score: float
    # Excluded from eq/hash: a dict field would make this frozen dataclass unhashable, and a
    # provenance consumer may reasonably expect "frozen ⇒ hashable". Identity is (id, score).
    signals: dict[str, int] = field(default_factory=dict, compare=False)


def rrf_fuse(
    rankings: Mapping[str, Sequence[int]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
    priority_signals: frozenset[str] = frozenset(),
) -> list[FusedHit]:
    """Fuse per-signal rankings of note ids into one ordered list by Reciprocal Rank Fusion.

    ``rankings`` maps a signal name to its candidate note ids **best-first**. ``weights`` scales
    each signal's contribution (default 1.0; a signal absent from ``weights`` uses 1.0). A note
    listed twice by one signal counts once, at its best (first) rank. ``priority_signals`` floats
    any note carrying one of those signals above all others (the exact-match override) — within a
    tier, by fused score. Ordering is deterministic: ``(tier, -score, note_id)``, so the result is
    independent of the input dict's iteration order.
    """
    weights = weights or {}
    scores: dict[int, float] = defaultdict(float)
    contributions: dict[int, dict[str, int]] = {}

    # Accumulate in a canonical (sorted) signal order, not the input dict's order: float addition
    # isn't associative, so a note in 3+ signals would otherwise get a score differing by ~1 ULP
    # with the dict order — enough to flip a near-tie and silently weaken "stable across queries".
    for signal in sorted(rankings):
        ids = rankings[signal]
        # Sanitize a non-finite weight to the default (1.0) before it reaches the score
        # accumulation and the sort. A NaN weight poisons a note's score to NaN, and
        # the native port orders NaN scores differently (Rust `total_cmp` total-orders NaN;
        # this Python `-h.score` sort key leaves NaN comparisons false → input-order-dependent),
        # which broke the frozen-reference parity contract. A non-finite weight is meaningless
        # as a scale, so coerce it to 1.0; finite-weight RRF (incl. 0.0 and negatives) is
        # unchanged. `shrike_kernel::fusion::rrf_fuse` applies the identical coercion.
        w = weights.get(signal, 1.0)
        if not math.isfinite(w):
            w = 1.0
        seen: set[int] = set()
        for pos, note_id in enumerate(ids):
            nid = int(note_id)
            if nid in seen:
                continue  # one signal, one rank per note (its best)
            seen.add(nid)
            rank = pos + 1  # 1-based
            scores[nid] += w / (k + rank)
            contributions.setdefault(nid, {})[signal] = rank

    hits = [FusedHit(nid, scores[nid], contributions[nid]) for nid in scores]
    hits.sort(key=lambda h: (0 if priority_signals & h.signals.keys() else 1, -h.score, h.note_id))
    return hits


# ── The SearchPipeline seam ──────────────────────────────────────────────────


@runtime_checkable
class SearchPipeline(Protocol):
    """The fusion seam `search_notes` composes against.

    The Python composition (`rrf_fuse` above) is the **reference
    implementation** — the readable spec, what test doubles fake, and what the
    parity property suite compares the native implementation against. The
    native pipeline (`shrike_kernel::fusion`) is a second implementation of
    the same contract, selected when the native extension is in play.
    """

    def fuse(
        self,
        rankings: Mapping[str, Sequence[int]],
        *,
        weights: Mapping[str, float] | None = None,
        k: int = RRF_K,
        priority_signals: frozenset[str] = frozenset(),
    ) -> list[FusedHit]:
        """Fuse per-signal best-first rankings into one ordered list (see rrf_fuse)."""
        ...


class ReferenceSearchPipeline:
    """The frozen Python reference — delegates to :func:`rrf_fuse` verbatim."""

    def fuse(
        self,
        rankings: Mapping[str, Sequence[int]],
        *,
        weights: Mapping[str, float] | None = None,
        k: int = RRF_K,
        priority_signals: frozenset[str] = frozenset(),
    ) -> list[FusedHit]:
        return rrf_fuse(rankings, weights=weights, k=k, priority_signals=priority_signals)


class NativeSearchPipeline:
    """The native fusion: `shrike_kernel::fusion::rrf_fuse` via shrike_native.

    Same semantics by construction (the parity property suite is the drift
    alarm); GIL released for the fusion itself. Ordinary patchable Python.
    """

    def __init__(self) -> None:
        import shrike_native

        self._rrf = shrike_native.rrf_fuse

    def fuse(
        self,
        rankings: Mapping[str, Sequence[int]],
        *,
        weights: Mapping[str, float] | None = None,
        k: int = RRF_K,
        priority_signals: frozenset[str] = frozenset(),
    ) -> list[FusedHit]:
        raw = self._rrf(
            [(signal, [int(i) for i in ids]) for signal, ids in rankings.items()],
            dict(weights or {}),
            k,
            sorted(priority_signals),
        )
        return [FusedHit(nid, score, dict(signals)) for nid, score, signals in raw]


def make_search_pipeline() -> SearchPipeline:
    """The native fused pipeline. The pure-Python rrf_fuse stays in this module
    as the documented reference the parity tests pin against the native
    fusion."""
    return NativeSearchPipeline()
