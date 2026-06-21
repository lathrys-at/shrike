"""Latency-distribution statistics for the perf harness.

Pure math, no I/O: turn a list of per-iteration timings into a distribution
(p50/p90/p99/max plus mean/min/n), discarding warmup iterations. A perf result
is a *distribution*, never a single number — the harness reports the shape, and
the conditions it was taken under (see :mod:`result`), so a regression is read
off the tail (p99), not just the median.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Distribution:
    """A latency distribution over the kept (post-warmup) samples, in
    milliseconds. ``n`` is the kept-sample count, not the total run count.
    ``samples_ms`` keeps the post-warmup timings in ITERATION order (the percentiles
    are over the same set, sorted) so a *trend* across the run is recoverable — see
    :attr:`drift_ms`."""

    n: int
    min_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    samples_ms: tuple[float, ...]

    @property
    def drift_ms(self) -> float:
        """The within-run drift: the median of the run's SECOND half minus that of
        its FIRST half, in iteration order. ~0 for a stable op; positive means the
        op got slower AS THE RUN PROGRESSED — the index-fragmentation signal #938's
        churn workload reads (flat with compaction, climbing without). 0 for < 2
        kept samples."""
        if len(self.samples_ms) < 2:
            return 0.0
        mid = len(self.samples_ms) // 2
        first = sorted(self.samples_ms[:mid])
        second = sorted(self.samples_ms[mid:])
        return _percentile(second, 0.50) - _percentile(first, 0.50)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Distribution:
        fields = {k: raw[k] for k in cls.__dataclass_fields__}
        fields["samples_ms"] = tuple(fields["samples_ms"])
        return cls(**fields)


def _percentile(ordered: list[float], q: float) -> float:
    """The ``q`` quantile (0..1) of an already-sorted list, by linear
    interpolation between the two nearest ranks (numpy's default method).

    # Errors

    Raises :class:`ValueError` on an empty input.
    """
    if not ordered:
        raise ValueError("percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def summarize(samples_ms: list[float], *, warmup: int = 0) -> Distribution:
    """Summarize per-iteration timings (ms) into a :class:`Distribution`,
    discarding the first ``warmup`` samples (a cold start, JIT/cache warm-up).

    # Errors

    Raises :class:`ValueError` if no samples remain after the warmup discard.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0 (got {warmup})")
    kept = samples_ms[warmup:]
    if not kept:
        raise ValueError(f"no samples left after discarding {warmup} warmup of {len(samples_ms)}")
    ordered = sorted(kept)
    return Distribution(
        n=len(ordered),
        min_ms=ordered[0],
        p50_ms=_percentile(ordered, 0.50),
        p90_ms=_percentile(ordered, 0.90),
        p99_ms=_percentile(ordered, 0.99),
        max_ms=ordered[-1],
        mean_ms=sum(ordered) / len(ordered),
        # Iteration order preserved (NOT sorted) — drift_ms reads the trend.
        samples_ms=tuple(kept),
    )
