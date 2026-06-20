"""Baseline-vs-run comparison for the perf harness.

Diffs two :class:`~tests.manual.perf.result.RunResult`s workload-by-workload —
the p50/p90/p99 deltas, absolute and relative. It REFUSES to compare runs whose
invariant conditions differ (a different machine, build, or corpus): a
cross-context latency delta is noise dressed as signal. A delta is read by hand —
there is no automated gate; this is the diff mechanism, the policy is human
judgement.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.manual.perf.result import RunResult


class IncomparableRuns(ValueError):
    """The two runs were taken under different invariant conditions (machine,
    build, profile, or corpus), so a numeric diff would mislead. The message
    names every field that differs."""


@dataclass(frozen=True)
class WorkloadDelta:
    """One workload's change from baseline to run, per percentile. ``pct`` is the
    relative change (``+0.10`` = 10% slower); positive is a regression."""

    workload: str
    baseline_ms: float
    current_ms: float
    delta_ms: float
    pct: float


def _delta(baseline: float, current: float) -> tuple[float, float]:
    delta = current - baseline
    pct = (delta / baseline) if baseline > 0 else 0.0
    return delta, pct


@dataclass(frozen=True)
class Comparison:
    """The per-workload p50 deltas from baseline to current. Workloads present in
    only one run are reported separately so a renamed/added workload is visible
    rather than silently dropped."""

    deltas: list[WorkloadDelta]
    only_in_baseline: list[str]
    only_in_current: list[str]

    def regressions(self, threshold_pct: float) -> list[WorkloadDelta]:
        """Workloads whose p50 grew by more than ``threshold_pct`` (e.g. ``0.10``
        for 10%) — surfaced for a human reading the diff, not an automated gate."""
        return [d for d in self.deltas if d.pct > threshold_pct]


def compare(baseline: RunResult, current: RunResult) -> Comparison:
    """Diff ``current`` against ``baseline`` on p50.

    # Errors

    Raises :class:`IncomparableRuns` if the runs' invariant conditions differ.
    """
    mismatched = current.conditions.differs_from(baseline.conditions)
    if mismatched:
        details = ", ".join(
            f"{f}: {getattr(baseline.conditions, f)!r} -> {getattr(current.conditions, f)!r}"
            for f in mismatched
        )
        raise IncomparableRuns(f"runs differ on invariant condition(s): {details}")

    base_by_name = {r.workload: r for r in baseline.results}
    cur_by_name = {r.workload: r for r in current.results}

    deltas: list[WorkloadDelta] = []
    for name in sorted(base_by_name.keys() & cur_by_name.keys()):
        b = base_by_name[name].distribution.p50_ms
        c = cur_by_name[name].distribution.p50_ms
        delta_ms, pct = _delta(b, c)
        deltas.append(
            WorkloadDelta(workload=name, baseline_ms=b, current_ms=c, delta_ms=delta_ms, pct=pct)
        )
    return Comparison(
        deltas=deltas,
        only_in_baseline=sorted(base_by_name.keys() - cur_by_name.keys()),
        only_in_current=sorted(cur_by_name.keys() - base_by_name.keys()),
    )


def render_comparison(cmp: Comparison) -> str:
    """A human-readable diff table — one row per shared workload, p50 delta with
    its sign, plus any workloads present in only one run."""
    lines = [
        f"{'workload':<24}{'base ms':>12}{'cur ms':>12}{'Δ ms':>12}{'Δ %':>10}",
    ]
    for d in cmp.deltas:
        lines.append(
            f"{d.workload:<24}{d.baseline_ms:>12.3f}{d.current_ms:>12.3f}"
            f"{d.delta_ms:>+12.3f}{d.pct * 100:>+9.1f}%"
        )
    if cmp.only_in_baseline:
        lines.append(f"# dropped (baseline only): {', '.join(cmp.only_in_baseline)}")
    if cmp.only_in_current:
        lines.append(f"# new (current only): {', '.join(cmp.only_in_current)}")
    return "\n".join(lines)
