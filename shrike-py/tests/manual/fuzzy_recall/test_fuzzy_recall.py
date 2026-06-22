"""Fuzzy-recall eval entrypoint + generator unit tests.

The full eval (build the corpus, run every arm) is gated behind
``SHRIKE_FUZZY_RECALL=1`` and is a manual lane — it is heavy and its absolute
numbers are a tuning signal, not a pass/fail. The generator unit tests run
unconditionally (they are pure and fast, no corpus, offline) so the typo model and
gold-by-construction logic stay pinned.
"""

from __future__ import annotations

import os

import pytest

from tests.manual.fuzzy_recall import misspellings as misspellings_mod
from tests.manual.fuzzy_recall.typo_queries import (
    EditKind,
    generate_queries,
    length_bucket,
)

_GATE = "SHRIKE_FUZZY_RECALL"


def _gated() -> bool:
    return os.environ.get(_GATE) == "1"


# ── generator unit tests (pure, fast, always run) ───────────────────────────────


def test_length_buckets_straddle_the_floor_and_ceiling() -> None:
    # The buckets must split at the cap floor (6) and ceiling (12) so a global mean
    # never hides the curve's effect on long queries.
    assert length_bucket(2) == "n<=6"
    assert length_bucket(6) == "n<=6"
    assert length_bucket(7) == "n7-11"
    assert length_bucket(11) == "n7-11"
    assert length_bucket(12) == "n12-17"
    assert length_bucket(17) == "n12-17"
    assert length_bucket(18) == "n18+"


def test_generation_is_deterministic_and_gold_is_constructed() -> None:
    # The same (texts, seed) yields a byte-identical query set, and every query's
    # gold contains the source note (gold = all notes containing the clean phrase).
    texts = {
        100: "mitochondria is the powerhouse of the cell membrane",
        101: "cellular respiration releases energy from glucose molecules",
        102: "the membrane potential drives the action signal forward",
    }

    # A gold resolver over the same texts: every note whose text contains the phrase.
    def resolver(phrase: str) -> list[int]:
        p = phrase.lower()
        return [nid for nid, t in texts.items() if p in t.lower()]

    a = generate_queries(texts, resolver, seed=7, sample_size=3, misspellings={})
    b = generate_queries(texts, resolver, seed=7, sample_size=3, misspellings={})
    assert a == b, "generation must be byte-reproducible for a fixed seed"
    assert a, "the sample should produce queries"
    for q in a:
        assert q.source_note_id in q.gold_ids, "the source note is always in gold"
        assert q.query != q.clean_phrase, "every query carries an effective typo"
        assert q.edits, "every query records its applied edits"
        assert q.typo_count == len(q.edits)


def test_typo_count_strata_are_exercised() -> None:
    # Across a reasonable sample the generator must produce queries at >1 typo count
    # (the strata cycle 1/2/3), so the eval can show degradation behaviour.
    texts = {
        i: f"membrane potential gradient signal {i} releases stored energy molecules cleanly"
        for i in range(200, 230)
    }

    def resolver(phrase: str) -> list[int]:
        p = phrase.lower()
        return [nid for nid, t in texts.items() if p in t.lower()]

    qs = generate_queries(texts, resolver, seed=3, sample_size=30, misspellings={})
    counts = {q.typo_count for q in qs}
    assert counts & {2, 3}, f"expected multi-typo queries, got counts {counts}"


def test_real_misspelling_is_injected_when_the_word_is_present() -> None:
    # A phrase containing a known correct word ("receive") must be perturbable into
    # its real misspelling ("recieve"), tagged REAL_MISSPELLING.
    texts = {500: "students receive the separate environment government calendar handout"}
    misspell = misspellings_mod.load_misspellings()  # embedded fallback offline

    def resolver(phrase: str) -> list[int]:
        return [500] if phrase.lower() in texts[500].lower() else []

    # Force single-word phrases keyed on a known correction; iterate seeds until one
    # lands on a correctable word, then assert the misspelling fired.
    found = False
    for seed in range(50):
        qs = generate_queries(texts, resolver, seed=seed, sample_size=1, misspellings=misspell)
        for q in qs:
            if EditKind.REAL_MISSPELLING in q.edits:
                # The perturbed query holds a known misspelling of a corpus word.
                found = True
                break
        if found:
            break
    assert found, "a phrase with a known word should yield a real-misspelling query"


def test_misspellings_fallback_parses() -> None:
    # The embedded fallback is non-empty and well-formed (offline must still inject
    # some real misspellings).
    pairs = misspellings_mod.load_misspellings()
    assert pairs
    assert pairs.get("recieve") == "receive"
    assert all(w.isalpha() and r.isalpha() for w, r in pairs.items())


# ── cap grid + frontier unit tests (pure, fast, always run) ─────────────────────


def test_cap_grid_is_control_then_full_k_by_ceiling() -> None:
    # The grid is the control followed by the full k×ceiling cross product (floor
    # pinned), so the frontier sees every cell of the 2D sweep.
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        CONTROL,
        GRID_CEILINGS,
        GRID_FLOOR,
        GRID_KS,
        cap_grid,
    )

    grid = cap_grid()
    assert grid[0] is CONTROL, "the control anchors the deltas first"
    body = grid[1:]
    assert len(body) == len(GRID_KS) * len(GRID_CEILINGS)
    assert {(p.k, p.ceiling) for p in body} == {(k, c) for k in GRID_KS for c in GRID_CEILINGS}, (
        "every (k, ceiling) cell is present exactly once"
    )
    assert all(p.floor == GRID_FLOOR for p in body), "floor is pinned across the grid"


def _lat(p50: float, p90: float = 0.0, p95: float = 0.0, p99: float = 0.0, mx: float = 0.0):  # type: ignore[no-untyped-def]
    """A SingleQueryLatency where unset percentiles default to p50 (so a test that only
    cares about one percentile stays terse)."""
    from tests.manual.fuzzy_recall.fuzzy_recall import SingleQueryLatency

    return SingleQueryLatency(
        p50_ms=p50,
        p90_ms=p90 or p50,
        p95_ms=p95 or p90 or p50,
        p99_ms=p99 or p95 or p90 or p50,
        max_ms=mx or p99 or p95 or p90 or p50,
    )


def test_frontier_picks_max_recall_within_p95_budget() -> None:
    # The cap pick is the highest-recall arm whose single-query p95 clears the budget —
    # an over-budget arm with higher recall must NOT win.
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        BUDGET_MS,
        ArmResult,
        CapPolicy,
        EvalRun,
        render_results,
    )

    def arm(label: str, recall: float, p95: float) -> ArmResult:
        return ArmResult(
            policy=CapPolicy(label, floor=6, k=2.7, ceiling=12),
            recall_at_k=recall,
            mrr=recall,
            n_queries=10,
            by_length={},
            by_typo_count={},
            by_edit={},
            latency=_lat(p95, p95=p95),
        )

    # control in-budget; a richer in-budget arm; an over-budget arm with the best recall.
    control = arm("fixed-6 (control)", 0.50, BUDGET_MS - 5.0)
    rich_ok = arm("k=3 ceil=16", 0.62, BUDGET_MS - 1.0)
    over = arm("k=8 ceil=24", 0.70, BUDGET_MS + 5.0)
    run = EvalRun(notes=50000, seed=0, sample_size=10, n_queries=10, arms=[control, rich_ok, over])

    report = render_results(run)
    # The latency section is present but loudly labelled untrusted (it's a harness, not a
    # result, until the maintainer's clean-env run).
    assert "single-query latency harness" in report and "UNTRUSTED" in report
    # The tail columns are surfaced (p95 is the budget, p90/p99 bracket it).
    assert "p95 ms" in report and "p95≤" in report and "p90≤" in report and "p99≤" in report
    # The pick is computed mechanically (max-recall within p95 budget) — valid only on a
    # clean run; the richer in-budget arm wins it, the over-budget arm does not.
    picks = report.split("**Cap pick")[1]
    p95_line = next(line for line in picks.splitlines() if "p95 budget:" in line)
    assert "k=3 ceil=16" in p95_line
    assert "k=8 ceil=24" not in p95_line


def test_frontier_reports_when_no_arm_clears_budget() -> None:
    # If every arm is over budget, the pick announces the budget cut rather than
    # silently picking an over-budget arm.
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        BUDGET_MS,
        ArmResult,
        CapPolicy,
        EvalRun,
        render_results,
    )

    over = ArmResult(
        policy=CapPolicy("fixed-6 (control)", floor=6, k=2.7, ceiling=6),
        recall_at_k=0.5,
        mrr=0.5,
        n_queries=10,
        by_length={},
        by_typo_count={},
        by_edit={},
        latency=_lat(
            BUDGET_MS + 1.0, p90=BUDGET_MS + 2.0, p95=BUDGET_MS + 3.0, p99=BUDGET_MS + 4.0
        ),
    )
    report = render_results(EvalRun(notes=50000, seed=0, sample_size=10, n_queries=10, arms=[over]))
    # Both budget percentiles announce the cut rather than silently picking over-budget.
    assert "no arm clears" in report
    picks = report.split("**Cap pick")[1]
    assert picks.count("no arm clears") == 2


def test_frontier_tail_can_disqualify_a_p50_in_budget_arm() -> None:
    # An arm comfortable at p50 but blowing the budget at p95 is the as-you-type failure
    # the tail budget exists to surface: it wins the p50 contrast but NOT the p95 (budget)
    # pick. This pins that the budget reads the tail, not the median.
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        BUDGET_MS,
        ArmResult,
        CapPolicy,
        EvalRun,
        render_results,
    )

    def arm(label: str, recall: float, p50: float, p95: float) -> ArmResult:
        return ArmResult(
            policy=CapPolicy(label, floor=6, k=2.7, ceiling=12),
            recall_at_k=recall,
            mrr=recall,
            n_queries=10,
            by_length={},
            by_typo_count={},
            by_edit={},
            latency=_lat(p50, p90=p95, p95=p95),
        )

    # tail_heavy: best recall, p50 in budget but p95 OVER. tame: lower recall, both in budget.
    tail_heavy = arm("k=8 ceil=24", 0.70, BUDGET_MS - 2.0, BUDGET_MS + 5.0)
    tame = arm("k=3 ceil=12", 0.60, BUDGET_MS - 4.0, BUDGET_MS - 1.0)
    report = render_results(
        EvalRun(notes=50000, seed=0, sample_size=10, n_queries=10, arms=[tail_heavy, tame])
    )
    picks = report.split("**Cap pick")[1]
    p95_line = next(line for line in picks.splitlines() if "p95 budget:" in line)
    p50_line = next(line for line in picks.splitlines() if "p50 budget:" in line)
    # p95 (budget) pick rejects the tail-heavy arm for the tame one; the p50 contrast takes it.
    assert "k=3 ceil=12" in p95_line and "k=8 ceil=24" not in p95_line
    assert "k=8 ceil=24" in p50_line


def test_tail_diagnostic_renders_when_samples_captured() -> None:
    # When the control arm carries per-query samples, the frontier renders the tail
    # diagnostic: the slow-vs-fast feature contrast + the slowest-query table.
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        ArmResult,
        CapPolicy,
        EvalRun,
        QuerySample,
        render_results,
    )

    # 20 samples: latency rises with n_trigrams + candidates, so the slow decile must
    # show a higher mean for both features than the fast decile.
    samples = [
        QuerySample(
            query=f"q{i}", latency_ms=float(i), n_trigrams=3 + i, typo_count=1, candidates=2 * i
        )
        for i in range(20)
    ]
    control = ArmResult(
        policy=CapPolicy("fixed-6 (control)", floor=6, k=2.7, ceiling=6),
        recall_at_k=0.5,
        mrr=0.5,
        n_queries=20,
        by_length={},
        by_typo_count={},
        by_edit={},
        latency=_lat(1.0, p95=2.0),
        samples=samples,
    )
    report = render_results(
        EvalRun(notes=50000, seed=0, sample_size=20, n_queries=20, arms=[control])
    )
    assert "tail diagnostic" in report
    assert "slowest 10%" in report and "fastest 10%" in report
    assert "n_trigrams" in report and "candidates" in report
    # The slowest-query table lists the slowest query (q19, latency 19).
    assert "`q19`" in report


def test_tail_diagnostic_absent_without_samples() -> None:
    # No captured samples (latency-off, or a non-control arm) → no diagnostic section.
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        ArmResult,
        CapPolicy,
        EvalRun,
        render_results,
    )

    arm = ArmResult(
        policy=CapPolicy("fixed-6 (control)", floor=6, k=2.7, ceiling=6),
        recall_at_k=0.5,
        mrr=0.5,
        n_queries=10,
        by_length={},
        by_typo_count={},
        by_edit={},
        latency=_lat(1.0),
        samples=None,
    )
    report = render_results(EvalRun(notes=50000, seed=0, sample_size=10, n_queries=10, arms=[arm]))
    assert "tail diagnostic" not in report


# ── the full eval (gated, manual) ───────────────────────────────────────────────


@pytest.mark.skipif(not _gated(), reason=f"set {_GATE}=1 to run the fuzzy-recall eval")
def test_fuzzy_recall_eval_runs_and_writes_results(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Run the eval at a small size (so the manual lane proves the wiring) and render
    the results artifact. Asserts only that the run produces queries and a result per
    arm — the recall NUMBERS are a tuning signal, not a pass/fail gate (the cap
    decision is a separate, eval-gated follow-up). The corpus cache and the artifact
    go under ``tmp_path`` so the lane runs hermetically (e.g. in a bazel sandbox)."""
    from tests.manual.fuzzy_recall.fuzzy_recall import (
        DEFAULT_ARMS,
        render_results,
        run_eval,
    )

    # Small + offline by default so the gated test is fast and hermetic; a real run
    # uses the CLI with --notes 5000 and the downloaded assets.
    notes = int(os.environ.get("SHRIKE_FUZZY_RECALL_NOTES", "500"))
    sample = int(os.environ.get("SHRIKE_FUZZY_RECALL_SAMPLE", "150"))
    offline = os.environ.get("SHRIKE_FUZZY_RECALL_OFFLINE", "1") == "1"

    # measure_latency=True so the single-query latency pass + the frontier render are
    # exercised end-to-end (the NUMBERS are noisy here and not asserted — only that
    # the wiring produces a well-formed latency record + a frontier section).
    run = run_eval(
        notes=notes,
        seed=0,
        sample_size=sample,
        offline=offline,
        measure_latency=True,
        cache_root=tmp_path / "corpora",
    )
    assert run.n_queries > 0
    assert len(run.arms) == len(DEFAULT_ARMS)
    for arm in run.arms:
        assert 0.0 <= arm.recall_at_k <= 1.0
        assert 0.0 <= arm.mrr <= 1.0
        assert arm.latency is not None, "the latency pass ran"
        lat = arm.latency
        # The percentiles are monotone non-decreasing (p50 ≤ p90 ≤ p95 ≤ p99 ≤ max).
        assert 0.0 <= lat.p50_ms <= lat.p90_ms <= lat.p95_ms <= lat.p99_ms <= lat.max_ms
    # The tail diagnostic is captured once, on the control (first) arm.
    assert run.arms[0].samples, "control arm carries per-query samples"
    assert all(a.samples is None for a in run.arms[1:]), "only the control captures samples"

    report = render_results(run)
    assert "single-query latency harness" in report, "the latency harness section renders"
    assert "UNTRUSTED" in report, "the latency section is labelled untrusted"
    assert "tail diagnostic" in report, "the tail diagnostic is rendered"
    out = tmp_path / "RESULTS.md"
    out.write_text(report + "\n")
    assert out.is_file()
