"""The perf result artifact + baseline comparison — pure serialization and
diff logic, no native, so it runs on every lane. (``Conditions.capture`` touches
the native extension + git; it's exercised by the manual runner, not here.)"""

from __future__ import annotations

import pytest

from tests.manual.perf.compare import IncomparableRuns, compare, render_markdown_comparison
from tests.manual.perf.result import (
    Conditions,
    RunResult,
    WorkloadResult,
    render_markdown_table,
    render_table,
)
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
        "ops": 100,
    }
    base.update(overrides)
    return Conditions(**base)


def _run(conditions: Conditions, p50: float) -> RunResult:
    dist = summarize([p50] * 5)
    return RunResult(
        conditions=conditions,
        results=[WorkloadResult("upsert", {"response": dist}, items=500)],
        timestamp="2026-01-01T00:00:00Z",
    )


def _settling_run(conditions: Conditions, *, response: float, settle: float) -> RunResult:
    # A write workload: response + settle + total phases (total = response+settle).
    return RunResult(
        conditions=conditions,
        results=[
            WorkloadResult(
                "upsert-batch",
                {
                    "response": summarize([response] * 5),
                    "settle": summarize([settle] * 5),
                    "total": summarize([response + settle] * 5),
                },
                items=100,
            )
        ],
        timestamp="2026-01-01T00:00:00Z",
    )


def test_runresult_json_round_trips():
    run = _run(_conditions(), 10.0)
    assert RunResult.from_json(run.to_json()) == run


def test_multiphase_result_round_trips_and_renders_one_row_per_phase():
    run = _settling_run(_conditions(), response=4.0, settle=20.0)
    assert RunResult.from_json(run.to_json()) == run
    (wr,) = run.results
    assert [p for p, _ in wr.ordered_phases()] == ["response", "settle", "total"]
    assert wr.distribution.p50_ms == pytest.approx(4.0)  # the response phase
    table = render_table(run)
    # One row per phase; the workload name + item count print once, on the first.
    assert table.count("upsert-batch") == 1
    for phase in ("response", "settle", "total"):
        assert phase in table
    # The amortized (per-op) column: each phase p50 over items (100). response
    # 4.0 -> 0.040, settle 20.0 -> 0.200, total 24.0 -> 0.240.
    assert "p50 (amortized) ms" in table
    assert "0.040" in table
    assert "0.200" in table


def test_render_table_amortizes_p50_over_items():
    run = _run(_conditions(), 10.0)  # one workload, items=500, response p50=10.0
    table = render_table(run)
    assert "p50 (amortized) ms" in table
    assert "0.020" in table  # 10.0 / 500


def test_render_table_widens_name_column_for_long_workload_names():
    # A long workload name (a scoped variant, 19 chars) must not overrun the phase
    # column, and every row's phase column stays aligned to the same offset.
    run = RunResult(
        conditions=_conditions(),
        results=[
            WorkloadResult("search-batch", {"response": summarize([1.0] * 5)}, items=20),
            WorkloadResult("search-scoped-batch", {"response": summarize([2.0] * 5)}, items=20),
        ],
        timestamp="2026-01-01T00:00:00Z",
    )
    table = render_table(run)
    assert "search-scoped-batchresponse" not in table  # name never glued to phase
    phase_rows = [ln for ln in table.splitlines() if "response" in ln]
    assert len(phase_rows) == 2
    assert len({ln.index("response") for ln in phase_rows}) == 1  # columns aligned


def test_render_markdown_table_is_paste_ready():
    run = _settling_run(_conditions(), response=4.0, settle=20.0)
    md = render_markdown_table(run)
    lines = md.splitlines()
    # A bold title, never an H1 — a `#` line would blow up to a heading in a comment.
    assert lines[0].startswith("**perf run")
    assert not md.lstrip().startswith("#")
    # A GitHub table: a header row followed by an alignment-separator row.
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| workload |"))
    assert "p50 (amortized) ms" in lines[header_idx]
    sep = lines[header_idx + 1]
    assert set(sep.replace(" ", "")) <= set("|-:")
    assert sep.count("|") == lines[header_idx].count("|")  # separator spans every column
    # One data row per phase; amortized per-op = phase p50 / items (100).
    assert "| upsert-batch | response |" in md
    assert "| 0.040 |" in md  # 4.0 / 100
    assert "| 0.200 |" in md  # 20.0 / 100


def test_render_markdown_comparison_is_paste_ready():
    baseline = _run(_conditions(), 10.0)
    current = _run(_conditions(commit="newer"), 12.0)  # advisory diff only
    md = render_markdown_comparison(compare(baseline, current))
    lines = md.splitlines()
    assert lines[0].startswith("| workload/phase |")
    assert set(lines[1].replace(" ", "")) <= set("|-:")  # the alignment row
    assert lines[1].count("|") == lines[0].count("|")  # separator spans every column
    assert "| upsert/response |" in md
    assert "+2.000" in md  # the signed p50 delta


def test_compatible_when_invariants_match_despite_advisory_differences():
    # commit/repeats are advisory — differing on them stays comparable.
    assert _conditions().differs_from(_conditions(commit="other", repeats=99)) == []


def test_incompatible_lists_the_differing_invariants():
    diffs = _conditions().differs_from(_conditions(machine="x86_64", corpus_size=5000))
    assert set(diffs) == {"machine", "corpus_size"}


def test_ops_is_an_invariant_so_different_n_never_compares():
    # A different N changes the per-iteration work, so the latencies aren't
    # comparable — the diff must refuse rather than read more-work as a regression.
    assert "ops" in _conditions().differs_from(_conditions(ops=50))
    a = _run(_conditions(ops=100), 10.0)
    b = _run(_conditions(ops=50), 6.0)  # fewer ops -> less work per iteration
    with pytest.raises(IncomparableRuns, match="ops"):
        compare(a, b)


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
    assert delta.workload == "upsert"
    assert delta.phase == "response"
    assert delta.delta_ms == pytest.approx(2.0)
    assert delta.pct == pytest.approx(0.2)
    assert cmp.regressions(threshold_pct=0.10) == [delta]
    assert cmp.regressions(threshold_pct=0.30) == []


def test_compare_diffs_each_phase_independently():
    baseline = _settling_run(_conditions(), response=4.0, settle=20.0)
    current = _settling_run(_conditions(commit="newer"), response=4.0, settle=30.0)
    cmp = compare(baseline, current)
    by_phase = {d.phase: d for d in cmp.deltas}
    assert set(by_phase) == {"response", "settle", "total"}
    assert by_phase["response"].delta_ms == pytest.approx(0.0)
    assert by_phase["settle"].delta_ms == pytest.approx(10.0)
    assert by_phase["total"].delta_ms == pytest.approx(10.0)
    # Only the settle/total phases regressed; the labels are workload/phase.
    assert {d.label for d in cmp.regressions(threshold_pct=0.10)} == {
        "upsert-batch/settle",
        "upsert-batch/total",
    }


def test_compare_reports_phases_present_in_only_one_run():
    baseline = _run(_conditions(), 10.0)  # response only
    current = _settling_run(_conditions(commit="newer"), response=4.0, settle=20.0)
    cmp = compare(baseline, current)
    # Different workload names entirely, so every phase is one-sided.
    assert cmp.only_in_baseline == ["upsert/response"]
    assert cmp.only_in_current == [
        "upsert-batch/response",
        "upsert-batch/settle",
        "upsert-batch/total",
    ]
    assert cmp.deltas == []
