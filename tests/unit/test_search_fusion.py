"""Tests for the RRF combiner (shrike.search_fusion)."""

from __future__ import annotations

from shrike.search_fusion import RRF_K, FusedHit, rrf_fuse


def _order(hits: list[FusedHit]) -> list[int]:
    return [h.note_id for h in hits]


class TestRrfFuse:
    def test_single_signal_preserves_rank_order(self) -> None:
        # RRF over one signal is just that signal's order.
        assert _order(rrf_fuse({"semantic": [3, 1, 2]})) == [3, 1, 2]

    def test_empty(self) -> None:
        assert rrf_fuse({}) == []
        assert rrf_fuse({"semantic": []}) == []

    def test_missing_signal_contributes_nothing(self) -> None:
        # note 9 is only in signal b; its score is exactly b's term, no phantom contribution from a.
        hits = {h.note_id: h for h in rrf_fuse({"a": [1, 2], "b": [9]})}
        assert hits[9].signals == {"b": 1}
        assert hits[9].score == 1.0 / (RRF_K + 1)

    def test_multi_signal_accrues_multiple_terms(self) -> None:
        # A note matching two signals at rank 1 outranks a note matching one signal at rank 1.
        hits = rrf_fuse({"a": [5, 6], "b": [5, 7]})
        assert _order(hits)[0] == 5  # in both at rank 1 → 2/(k+1)
        assert {h.note_id: h.signals for h in hits}[5] == {"a": 1, "b": 1}

    def test_weights_shift_order(self) -> None:
        # Equal ranks but a heavier signal wins. note 1 leads a (rank1), note 2 leads b (rank1).
        base = _order(rrf_fuse({"a": [1, 9], "b": [2, 8]}))
        weighted = _order(rrf_fuse({"a": [1, 9], "b": [2, 8]}, weights={"b": 10.0}))
        assert base[0] == 1  # tie on score → lower note_id first
        assert weighted[0] == 2  # b's weight floats its rank-1 note above a's

    def test_input_order_independent(self) -> None:
        # The fused order must not depend on the rankings dict's iteration order (stability).
        a = rrf_fuse({"sem": [1, 2, 3], "exact": [2]})
        b = rrf_fuse({"exact": [2], "sem": [1, 2, 3]})
        assert _order(a) == _order(b)
        assert [h.score for h in a] == [h.score for h in b]

    def test_ties_broken_by_note_id(self) -> None:
        # Two notes at the same rank in the same single signal can't tie, but across symmetric
        # signals they can — resolve deterministically by ascending note_id.
        assert _order(rrf_fuse({"a": [7], "b": [3]})) == [3, 7]

    def test_duplicate_in_signal_counts_once_at_best_rank(self) -> None:
        hits = {h.note_id: h for h in rrf_fuse({"a": [5, 5, 6]})}
        assert hits[5].signals == {"a": 1}  # first (best) occurrence, not rank 2
        assert hits[5].score == 1.0 / (RRF_K + 1)

    def test_priority_signal_overrides_score(self) -> None:
        # The exact-match override: a literal hit that's weak/absent in the other signals (note 9
        # is exact-only) still tiers to the top, where bare RRF would not float it there.
        rankings = {"semantic": [1, 2, 3, 4], "exact": [9]}
        overridden = rrf_fuse(rankings, priority_signals=frozenset({"exact"}))
        assert _order(overridden)[0] == 9  # exact hit floats above strong-semantic notes
        assert _order(overridden)[1:] == [1, 2, 3, 4]  # non-literal tier, RRF-ordered
        # Without the override, note 9 (one term, 1/(k+1)) only ties semantic rank 1 → not on top.
        assert _order(rrf_fuse(rankings))[0] != 9

    def test_priority_tier_internally_rrf_ordered(self) -> None:
        # Multiple exact hits within the priority tier order by fused score, not arbitrarily.
        hits = rrf_fuse(
            {"semantic": [8, 9], "exact": [9, 8]},
            priority_signals=frozenset({"exact"}),
        )
        # Both are exact (priority tier); 9 is exact-rank1 + semantic-rank2, 8 is exact-rank2 +
        # semantic-rank1 → symmetric scores → tie → note_id asc.
        assert set(_order(hits)) == {8, 9}
        assert _order(hits) == [8, 9]

    def test_contributions_carry_ranks(self) -> None:
        hits = {h.note_id: h for h in rrf_fuse({"a": [10, 20], "b": [20]})}
        assert hits[10].signals == {"a": 1}
        assert hits[20].signals == {"a": 2, "b": 1}

    def test_three_signal_score_input_order_independent(self) -> None:
        # Float addition isn't associative, so a note in 3+ signals would get a score differing by
        # ~1 ULP with the input dict's order — enough to flip a near-tie. Canonical (sorted)
        # accumulation makes it bit-identical (#236 review F2). Note 7 sits at ranks 1, 2, 8.
        ranked = {"a": [7], "b": [99, 7], "c": [91, 92, 93, 94, 95, 96, 97, 7]}
        reordered = {"c": ranked["c"], "a": ranked["a"], "b": ranked["b"]}
        assert {h.note_id: h.score for h in rrf_fuse(ranked)}[7] == (
            {h.note_id: h.score for h in rrf_fuse(reordered)}[7]
        )  # bit-identical, not just close

    def test_fused_hit_is_hashable(self) -> None:
        # FusedHit is advertised as the provenance object (#182); "frozen ⇒ hashable" must hold
        # despite the dict field (#236 review F3).
        hit = rrf_fuse({"a": [1, 2]})[0]
        assert hash(hit) == hash(hit)  # does not raise
        assert hit in {hit}  # usable in a set
