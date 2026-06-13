"""Deterministic, CI-stable search-quality classes (#559, PR1).

These run IN CI on every PR: a stub embedder (controlled vectors, no model, no
network) drives the REAL ``search_notes`` MCP action in-process, so the RRF
fusion arithmetic, the exact-override tier, the activation gate, and the
graceful-degradation paths are pinned without flake. They are the permanent
regression guard for the fusion logic; the heavy real-model recall/precision
numbers live in the manual suite (PR2).

Everything is asserted through the response provenance
(``provenance[].signal`` / ``substring`` / ``fuzzy`` / ``score`` /
``message`` / ``completeness``) — proving WHY a card ranked, not just that it
did — and exact golden orders are derived from the canonical RRF constants
(``RRF_K=60``, the per-signal weights, the exact priority tier).
"""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from tests.search_quality.inprocess import (  # noqa: E402
    StubEmbedder,
    build_harness,
    onehot,
    to_ranked_cards,
    to_returned_cards,
)
from tests.search_quality.metrics import (  # noqa: E402
    GradedGold,
    evaluate_query,
    rrf_order_from_ranks,
)

DIM = 16


def _ids(matches: list[dict]) -> list[int]:
    return [m["id"] for m in matches]


def _signals(match: dict) -> set[str]:
    return {p["signal"] for p in match.get("provenance", [])}


def _card(marker: str, body: str, *, back: str = "review chapter") -> dict:
    """A text card whose field carries the planting marker + distinctive text."""
    return {
        "note_type": "Basic",
        "deck": "AdversarialEval::Deterministic",
        "fields": {"Front": f"@@P:{marker}@@ {body}", "Back": back},
    }


class TestSignalDisagreement:
    """Golden top-k when signals DISAGREE: the fused order must be exactly what
    the RRF constants produce. Asserted two ways — the returned order equals a
    recompute from each card's OWN provenance ranks (pins the fusion
    *arithmetic* against the constants) AND a concrete reorder consequence."""

    def test_text_and_fuzzy_disagree_fuses_by_the_constants(self, tmp_path) -> None:
        async def flow() -> None:
            backend = StubEmbedder(dim=DIM, fingerprint="stub:disagree:v1")
            # Three cards on axis 0; the query sits on the pure axis, so the
            # text (semantic) rank is a1 > a2 > a3 by cosine.
            backend.plant_text("a1", onehot(DIM, 0, 0.20))
            backend.plant_text("a2", onehot(DIM, 0, 0.55))
            backend.plant_text("a3", onehot(DIM, 0, 1.10))
            # The query embeds to the pure axis (planted) AND carries a typo of a
            # distinctive token living ONLY on a3 → a3 alone fuzzy-hits.
            query = "azzledorf mitochodria"
            backend.plant_query(query, onehot(DIM, 0))

            ip = await build_harness(tmp_path, backend, attach_media=False)
            try:
                notes = await ip.harness.wrapper.upsert_notes(
                    [
                        _card("a1", "alpha topic one"),
                        _card("a2", "beta topic two"),
                        _card("a3", "gamma mitochondria topic three"),
                    ]
                )
                a1, a2, a3 = (n["id"] for n in notes)
                await ip.finalize()

                # threshold 0.0: all three cards participate in the text signal at
                # their cosine rank; a3 also picks up the fuzzy signal.
                matches = await ip.matches(query, top_k=10, threshold=0.0)
                got = _ids(matches)

                a3_match = next((m for m in matches if m["id"] == a3), None)
                assert a3_match is not None, f"a3 must be returned, got {got}"
                assert "fuzzy" in _signals(a3_match), "a3 carries the fuzzy signal"
                assert "text" in _signals(a3_match), "a3 also semantically ranks"

                # (1) The fused order recomputed PURELY from the reported
                # per-signal ranks equals the returned order — pins the RRF
                # arithmetic (score = Σ w_s/(k+rank_s), k=60, fuzzy weight 0.5).
                golden = rrf_order_from_ranks(to_ranked_cards(matches))
                assert got == golden, f"returned {got} != RRF-recomputed {golden}"

                # (2) The concrete disagreement consequence: text alone ranks
                # a3 LAST (a1<a2<a3), but the fuzzy 0.5-weight boost lifts a3
                # above a2 — a reorder a single signal could never produce.
                assert got.index(a3) < got.index(a2), (
                    "fuzzy lifts a3 above a2 despite a worse text rank"
                )
            finally:
                await ip.harness.close()

        asyncio.run(flow())

    def test_fuzzy_weight_below_text_is_the_mitigation(self) -> None:
        # The control (methodology #559): fuzzy is the ONLY signal weighted below
        # the rest (0.5 vs 1.0) — a near-miss is weaker evidence than a literal
        # or semantic hit. Two pure-arithmetic facts over the canonical
        # constants, no server:
        from tests.search_quality.metrics import RRF_WEIGHTS, RankedCard

        # (1) The weight is genuinely lower than every other signal's.
        others = {s: w for s, w in RRF_WEIGHTS.items() if s != "fuzzy"}
        assert RRF_WEIGHTS["fuzzy"] == 0.5
        assert all(w == 1.0 for w in others.values()), "every non-fuzzy signal weighs 1.0"

        # (2) The weight is load-bearing: a card surfaced by text+fuzzy outranks
        # a fuzzy-ONLY near-miss at the same fuzzy rank — and flipping fuzzy to
        # 1.0 would NOT change that here, but DROPPING the gap (fuzzy at 1.0)
        # lets a fuzzy-only card overtake a same-rank text-only card it must not.
        text_only = RankedCard(1, {"text": 3})  # a real semantic hit, rank 3
        fuzzy_only = RankedCard(2, {"fuzzy": 1})  # a near-miss at fuzzy rank 1
        at_half = rrf_order_from_ranks([text_only, fuzzy_only], weights=RRF_WEIGHTS)
        at_one = rrf_order_from_ranks(
            [text_only, fuzzy_only], weights={**RRF_WEIGHTS, "fuzzy": 1.0}
        )
        assert at_half[0] == 1, "at 0.5 the semantic hit outranks the fuzzy-only near-miss"
        assert at_one[0] == 2, "at 1.0 the fuzzy-only near-miss would wrongly overtake it"


class TestExactOverride:
    """The exact-substring priority tier: a literal hit floats above the rest,
    RRF-ordered within. Two faces — it is CORRECT (a genuine literal answer
    wins) AND it has a pathological edge (a grade-0 literal out-ranking a
    grade-3 semantic hit), surfaced as the design-tension characterization."""

    def test_literal_hit_floats_above_a_better_semantic_card(self, tmp_path) -> None:
        async def flow() -> None:
            backend = StubEmbedder(dim=DIM, fingerprint="stub:exact:v1")
            # Semantic rank b1 > b2 > b3 on axis 2; query is the pure axis.
            backend.plant_text("b1", onehot(DIM, 2, 0.20))
            backend.plant_text("b2", onehot(DIM, 2, 0.55))
            backend.plant_text("b3", onehot(DIM, 2, 1.10))
            query = "quetzalcoatlus"  # literal token present ONLY in b3's field
            backend.plant_query(query, onehot(DIM, 2))

            ip = await build_harness(tmp_path, backend, attach_media=False)
            try:
                notes = await ip.harness.wrapper.upsert_notes(
                    [
                        _card("b1", "delta thermo one"),
                        _card("b2", "epsilon thermo two"),
                        _card("b3", "zeta quetzalcoatlus thermo three"),
                    ]
                )
                b1, b2, b3 = (n["id"] for n in notes)
                await ip.finalize()

                matches = await ip.matches(query, top_k=10, threshold=0.0)
                got = _ids(matches)

                # b3 is the literal hit (semantically the WORST of the three) and
                # takes rank 1 via the exact priority tier — the override.
                assert got[0] == b3, f"the literal hit floats to rank 1, got {got}"
                b3_match = next(m for m in matches if m["id"] == b3)
                assert "exact" in _signals(b3_match), "rank 1 carries the exact signal"
                assert b3_match.get("substring") is not None, "rank 1 carries the substring annotation"

                # The recomputed RRF order (priority tier included) matches.
                golden = rrf_order_from_ranks(to_ranked_cards(matches))
                assert got == golden, f"returned {got} != RRF-recomputed {golden}"
            finally:
                await ip.harness.close()

        asyncio.run(flow())

    def test_pathological_domination_grade0_literal_above_grade3_semantic(self, tmp_path) -> None:
        async def flow() -> None:
            backend = StubEmbedder(dim=DIM, fingerprint="stub:patho:v1")
            # p_answer is the canonical (grade-3) semantic answer on axis 4;
            # p_trap is a planted distractor (grade-0) that happens to contain
            # the query as a literal substring.
            backend.plant_text("p_answer", onehot(DIM, 4))
            backend.plant_text("p_trap", onehot(DIM, 4, 1.10))  # far worse cosine
            query = "obscureliteraltoken"
            backend.plant_query(query, onehot(DIM, 4))

            ip = await build_harness(tmp_path, backend, attach_media=False)
            try:
                notes = await ip.harness.wrapper.upsert_notes(
                    [
                        _card("p_answer", "the genuine canonical answer card"),
                        _card("p_trap", "junk card mentioning obscureliteraltoken once"),
                    ]
                )
                answer, trap = (n["id"] for n in notes)
                await ip.finalize()

                matches = await ip.matches(query, top_k=10, threshold=0.0)
                got = _ids(matches)

                # CHARACTERIZATION (#559 design-tension finding): the grade-0
                # literal out-ranks the grade-3 semantic answer because the exact
                # tier is blind to relevance. We PIN this (flag-on-change), not
                # bless it — RRF's exact tier is "wrong" exactly here.
                assert got[0] == trap, (
                    f"the grade-0 literal dominates the grade-3 answer (got {got})"
                )

                gold = GradedGold(
                    grades={answer: 3, trap: 0},
                    expected_signal="text",
                    closed_world=False,
                    top_k=10,
                )
                report = evaluate_query(
                    query, "exact_override_pathological", to_returned_cards(matches), gold
                )
                assert report.exact_tier_pure is False, (
                    "exact-tier purity is violated: a grade-0 literal was floated"
                )
                fp_ids = {f.note_id for f in report.failures if f.kind.value == "precision_fp"}
                assert trap in fp_ids, "the metric engine tags the grade-0 literal as a false positive"
            finally:
                await ip.harness.close()

        asyncio.run(flow())
