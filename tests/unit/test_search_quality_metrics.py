"""Unit tests for the search-quality metric engine + manifest loader (#559).

Pure, fast, no server: the metric engine is a function of ``(returned, gold)``,
so these pin its recall/precision/nDCG math and the failure-kind tagging
directly — the same engine the in-process CI classes and the PR2 manual suite
consume. Because it's parameterizable (``k``/``weights``/``RRF_K``), the same
tests double as the #234 sweep's correctness guard.
"""

from __future__ import annotations

from pathlib import Path

from tests.search_quality.manifest import load_manifest
from tests.search_quality.metrics import (
    RRF_WEIGHTS,
    GradedGold,
    RankedCard,
    ReturnedCard,
    SuiteReport,
    evaluate_query,
    ndcg_at,
    rrf_order_from_ranks,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "search_quality" / "fixtures" / "example_manifest.json"
)


class TestRRFRecompute:
    def test_priority_tier_floats_an_exact_hit(self) -> None:
        # text ranks [1,2,3]; the exact signal lands card 3 → it floats to rank
        # 1 (the kernel's fusion.rs golden case), then 1, 2 by RRF score.
        cards = [
            RankedCard(1, {"text": 1}),
            RankedCard(2, {"text": 2}),
            RankedCard(3, {"text": 3, "exact": 1}),
        ]
        assert rrf_order_from_ranks(cards) == [3, 1, 2]

    def test_fuzzy_weight_lowers_a_near_miss(self) -> None:
        # A text-rank-3 card boosted by a fuzzy rank-1 hit overtakes the
        # text-rank-1/2 cards (0.5/61 boost > the rank gaps).
        cards = [
            RankedCard(1, {"text": 1}),
            RankedCard(2, {"text": 2}),
            RankedCard(3, {"text": 3, "fuzzy": 1}),
        ]
        assert rrf_order_from_ranks(cards)[0] == 3

    def test_weights_are_a_sweep_seam(self) -> None:
        # The same ranks, different weights → different order: the engine is
        # parameterizable for the #234 threshold/weight sweep.
        cards = [RankedCard(1, {"text": 2}), RankedCard(2, {"fuzzy": 1})]
        assert rrf_order_from_ranks(cards, weights=RRF_WEIGHTS)[0] == 1
        assert rrf_order_from_ranks(cards, weights={**RRF_WEIGHTS, "fuzzy": 1.0})[0] == 2


class TestRecallPrecision:
    def _returned(self, order, *, exact=()):
        return [
            ReturnedCard(
                note_id=nid,
                rank=i + 1,
                signals=frozenset({"exact"}) if nid in exact else frozenset({"text"}),
                score=None if nid in exact else 0.7,
                has_substring=nid in exact,
            )
            for i, nid in enumerate(order)
        ]

    def test_recall_family(self) -> None:
        gold = GradedGold(grades={10: 3, 20: 2, 30: 0}, expected_signal="text", top_k=10)
        r = evaluate_query("q", "c", self._returned([10, 30, 20]), gold, expects_degradation=False)
        assert r.recall_at_1 == 0.5  # 1 of 2 relevant in top-1
        assert r.recall_at_5 == 1.0
        assert r.mrr == 1.0  # first relevant at rank 1
        # the grade-0 card at rank 2 demotes the grade-2 below it in nDCG
        assert 0.0 < r.ndcg_at_10 < 1.0

    def test_precision_false_positive(self) -> None:
        gold = GradedGold(grades={10: 3, 30: 0}, expected_signal="text", top_k=10)
        r = evaluate_query("q", "c", self._returned([10, 30]), gold)
        assert r.false_positive_rate == 0.5  # 1 of 2 returned is grade-0
        kinds = {f.kind.value for f in r.failures}
        assert "precision_fp" in kinds

    def test_over_return_on_null_gold(self) -> None:
        gold = GradedGold(grades={}, expected_signal=None, top_k=10)
        r = evaluate_query("off", "over_return", self._returned([99]), gold)
        assert r.over_returned is True
        assert any(f.kind.value == "over_return" for f in r.failures)

    def test_exact_tier_purity(self) -> None:
        # A grade-0 card floated by the exact override → impurity flagged.
        gold = GradedGold(
            grades={10: 3, 30: 0}, expected_signal="text", closed_world=False, top_k=10
        )
        r = evaluate_query("q", "c", self._returned([30, 10], exact=(30,)), gold)
        assert r.exact_tier_pure is False

    def test_degrade_silent_tag(self) -> None:
        gold = GradedGold(grades={10: 3}, expected_signal="text", top_k=10)
        returned = self._returned([10])
        silent = evaluate_query(
            "q", "c", returned, gold, expects_degradation=True, response_announced_degradation=False
        )
        announced = evaluate_query(
            "q", "c", returned, gold, expects_degradation=True, response_announced_degradation=True
        )
        assert any(f.kind.value == "degrade_silent" for f in silent.failures)
        assert not any(f.kind.value == "degrade_silent" for f in announced.failures)


class TestNDCG:
    def test_perfect_ranking_is_one(self) -> None:
        gold = GradedGold(grades={1: 3, 2: 2, 3: 1}, top_k=10)
        cards = [ReturnedCard(1, 1), ReturnedCard(2, 2), ReturnedCard(3, 3)]
        assert ndcg_at(cards, gold, k=10) == 1.0

    def test_demotion_lowers_ndcg(self) -> None:
        gold = GradedGold(grades={1: 3, 2: 2}, top_k=10)
        good = ndcg_at([ReturnedCard(1, 1), ReturnedCard(2, 2)], gold, k=10)
        bad = ndcg_at([ReturnedCard(2, 1), ReturnedCard(1, 2)], gold, k=10)
        assert good == 1.0
        assert bad < good


class TestSuiteAggregate:
    def test_means_and_failure_rollup(self) -> None:
        gold = GradedGold(grades={10: 3}, expected_signal="text", top_k=10)
        ok = evaluate_query("a", "cls", [ReturnedCard(10, 1, frozenset({"text"}), 0.8)], gold)
        miss = evaluate_query("b", "cls", [ReturnedCard(99, 1, frozenset({"text"}), 0.8)], gold)
        suite = SuiteReport(queries=(ok, miss))
        assert suite.passed is False
        assert suite.mean_recall_at_k() == 0.5  # one hit, one miss
        from tests.search_quality.metrics import FailureKind

        rolled = suite.failures_by_kind()
        # the missed query records a RECALL_MISS for gold note 10
        miss_ids = {f.note_id for f in rolled[FailureKind.RECALL_MISS]}
        assert miss_ids == {10}
        assert "cls" in suite.classes()


class TestManifestLoader:
    def test_loads_reconciled_schema(self) -> None:
        man = load_manifest(FIXTURE)
        assert man.closed_world is True
        assert len(man.cards) == 3
        # the image card's logical media handle is preserved for $IMG substitution
        img = man.card_by_id[2]
        assert img.kind == "image_only"
        assert img.media[0].handle == "heart"
        assert img.media[0].source == "generated"
        assert "$IMG:heart" in img.fields["Back"]

    def test_gold_and_hard_negatives_merge_with_grades(self) -> None:
        man = load_manifest(FIXTURE)
        q = next(q for q in man.queries if q.adversarial_class == "semantic_text")
        # gold defaults grade 3, hard-negatives default grade 0
        assert q.gold.grades == {1: 3, 301: 0}
        assert q.gold.relevant_ids == frozenset({1})
        assert q.expected_signal == "text"

    def test_null_gold_query_is_a_precision_query(self) -> None:
        man = load_manifest(FIXTURE)
        q = next(q for q in man.queries if q.adversarial_class == "over_return")
        assert q.expected_signal is None
        assert not q.gold.has_relevant
        assert q.gold.grades == {}

    def test_image_gold_defaults_grade_three(self) -> None:
        man = load_manifest(FIXTURE)
        q = next(q for q in man.queries if q.adversarial_class == "modality_gap")
        assert q.gold.grades == {2: 3}  # gold row with no explicit grade → 3
        assert q.modality == "image"
