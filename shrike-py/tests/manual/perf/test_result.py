"""The perf result artifact + baseline comparison — pure serialization and
diff logic, no native, so it runs on every lane. (``Conditions.capture`` touches
the native extension + git; it's exercised by the manual runner, not here.)"""

from __future__ import annotations

import pytest

from tests.manual.perf.compare import IncomparableRuns, compare
from tests.manual.perf.result import Conditions, RunResult, WorkloadResult
from tests.manual.perf.stats import summarize


def _conditions(**overrides) -> Conditions:
    base = {
        "commit": "abc123",
        "dirty": False,
        "machine": "arm64",
        "system": "Darwin",
        "python": "3.12.0",
        "native_version": "0.0.0",
        "optimized": True,
        "profile": "perf-stub",
        "corpus_size": 500,
        "corpus_variant": "text",
        "repeats": 5,
        "warmup": 1,
    }
    base.update(overrides)
    return Conditions(**base)


def _run(conditions: Conditions, p50: float) -> RunResult:
    dist = summarize([p50] * 5)
    return RunResult(
        conditions=conditions,
        results=[WorkloadResult("ingest", dist, items=500)],
        timestamp="2026-01-01T00:00:00Z",
    )


def test_runresult_json_round_trips():
    run = _run(_conditions(), 10.0)
    assert RunResult.from_json(run.to_json()) == run


def test_compatible_when_invariants_match_despite_advisory_differences():
    # commit/repeats are advisory — differing on them stays comparable.
    assert _conditions().differs_from(_conditions(commit="other", repeats=99)) == []


def test_incompatible_lists_the_differing_invariants():
    diffs = _conditions().differs_from(_conditions(machine="x86_64", corpus_size=5000))
    assert set(diffs) == {"machine", "corpus_size"}


def test_optimized_is_an_invariant_so_debug_and_release_never_compare():
    assert "optimized" in _conditions().differs_from(_conditions(optimized=False))
    a = _run(_conditions(optimized=True), 10.0)
    b = _run(_conditions(optimized=False), 5.0)  # a debug run is slower/noisier
    with pytest.raises(IncomparableRuns, match="optimized"):
        compare(a, b)


def test_compare_refuses_runs_on_different_machines():
    a = _run(_conditions(), 10.0)
    b = _run(_conditions(machine="x86_64"), 12.0)
    with pytest.raises(IncomparableRuns, match="machine"):
        compare(a, b)


def test_compare_computes_p50_delta_and_flags_regressions():
    baseline = _run(_conditions(), 10.0)
    current = _run(_conditions(commit="newer"), 12.0)  # advisory diff only
    cmp = compare(baseline, current)
    (delta,) = cmp.deltas
    assert delta.workload == "ingest"
    assert delta.delta_ms == pytest.approx(2.0)
    assert delta.pct == pytest.approx(0.2)
    assert cmp.regressions(threshold_pct=0.10) == [delta]
    assert cmp.regressions(threshold_pct=0.30) == []
