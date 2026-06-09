"""Tests for search_notes, upsert neighbor attachment, and delete index updates in tools.py.

These test the tool registration layer with mocked index and real collection.
Uses FastMCP.call_tool() (the public API) so Pydantic parsing is exercised.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.index import IndexState, VectorIndex
from shrike.tools import register_tools

BASIC_NOTE = {
    "deck": "Test",
    "note_type": "Basic",
    "fields": {"Front": "Q", "Back": "A"},
}


def _seed(wrapper, notes):
    """Seed notes synchronously via the wrapper's worker thread.

    Defaults to ``on_duplicate="allow"``: these neighbor tests deliberately
    create notes identical to seeded ones to exercise similarity lookup, which
    the default error-on-duplicate policy would otherwise reject.
    """
    return wrapper.run_sync(lambda _c: wrapper._upsert_notes(notes, on_duplicate="allow"))


def _call(mcp: FastMCP, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    _, structured = asyncio.run(mcp.call_tool(name, args or {}))
    return structured


def _upsert(mcp: FastMCP, notes: list[dict], **extra: Any) -> dict[str, Any]:
    # See _seed: duplicates are intentional here, so allow them unless a test
    # opts into a different policy.
    extra.setdefault("on_duplicate", "allow")
    return _call(mcp, "upsert_notes", {"notes": notes, **extra})


def _text_hits(per_query: list[list[dict]]) -> list[dict[str, list[dict]]]:
    """Wrap legacy ``[[hit, ...], ...]`` semantic returns as per-modality ``{"text": [...]}`` maps —
    the shape ``search_by_modality`` returns (one per query). Tests that want an image-modality
    ranking build the dict directly."""
    return [{"text": hits} for hits in per_query]


@pytest.fixture()
def mock_index():
    idx = MagicMock(spec=VectorIndex)
    idx.state = IndexState.READY
    idx.available = True
    idx.build_progress = (0, 0)
    # search_notes ranks per modality; the upsert neighbour path still uses plain search().
    idx.search_by_modality = MagicMock(return_value=[])
    idx.search = MagicMock(return_value=[])
    # Uncalibrated by default → no activation floor → the #201b image gate is off (= #201a
    # behaviour); the gate-specific tests set this explicitly.
    idx.activation_stats = {}
    idx.col_mod = 0
    idx.size = 100
    return idx


@pytest.fixture()
def mcp_app(wrapper, mock_index):
    mcp = FastMCP("test")
    register_tools(mcp, wrapper, index=mock_index)
    return mcp


@pytest.fixture()
def mcp_no_index(wrapper):
    mcp = FastMCP("test")
    register_tools(mcp, wrapper, index=None)
    return mcp


class TestSearchNotesStates:
    def test_unavailable_still_runs_exact(self, mcp_app, mock_index):
        # Semantic down, but substring matching needs no index: the call still
        # runs (no literal "test" in a fresh collection → empty group) and notes
        # that semantic ranking was skipped.
        mock_index.state = IndexState.UNAVAILABLE
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        assert "exact text matches" in result["message"]
        assert all(not g["matches"] for g in result["results"])

    def test_building_returns_progress(self, mcp_app, mock_index):
        mock_index.state = IndexState.BUILDING
        mock_index.build_progress = (50, 100)
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        assert "50/100" in result["message"]

    def test_error_returns_message(self, mcp_app, mock_index):
        mock_index.state = IndexState.ERROR
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        assert "error" in result["message"]

    def test_no_index_still_runs_exact(self, mcp_no_index):
        result = _call(mcp_no_index, "search_notes", {"queries": ["test"]})
        assert "exact text matches" in result["message"]
        assert all(not g["matches"] for g in result["results"])

    def test_ids_only_with_no_index_returns_message(self, mcp_no_index, wrapper):
        # An id anchor has no literal text to match, so with no index there's
        # nothing to do — message, no results.
        seeded = _seed(wrapper, [BASIC_NOTE])
        result = _call(mcp_no_index, "search_notes", {"ids": [seeded[0]["id"]]})
        assert result["results"] == []
        assert "unavailable" in result["message"]

    def test_requires_queries_or_ids(self, mcp_app):
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="queries or ids"):
            _call(mcp_app, "search_notes", {})


class TestSearchNotesResults:
    def test_text_query(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": basic_note, "distance": 0.1}]]
        )
        result = _call(mcp_app, "search_notes", {"queries": ["math question"]})
        assert len(result["results"]) == 1
        assert result["results"][0]["source"] == "math question"
        matches = result["results"][0]["matches"]
        assert len(matches) == 1
        assert matches[0]["id"] == basic_note
        assert matches[0]["score"] == 0.9

    def test_id_query(self, wrapper, mock_index, mcp_app, basic_note):
        other = _seed(
            wrapper, [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]
        )[0]["id"]
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": other, "distance": 0.2}]]
        )
        result = _call(mcp_app, "search_notes", {"ids": [basic_note]})
        assert len(result["results"]) == 1
        assert result["results"][0]["source"] == f"note #{basic_note}"

    def test_exclude_ids(self, wrapper, mock_index, mcp_app, basic_note):
        other = _seed(
            wrapper, [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]
        )[0]["id"]
        mock_index.search_by_modality.return_value = _text_hits(
            [
                [
                    {"note_id": basic_note, "distance": 0.05},
                    {"note_id": other, "distance": 0.2},
                ]
            ]
        )
        result = _call(mcp_app, "search_notes", {"queries": ["test"], "exclude_ids": [basic_note]})
        matches = result["results"][0]["matches"]
        assert all(m["id"] != basic_note for m in matches)

    def test_deck_filter(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": basic_note, "distance": 0.1}]]
        )
        result = _call(mcp_app, "search_notes", {"queries": ["test"], "deck": "Nonexistent"})
        assert result["results"][0]["matches"] == []


class TestUnifiedSearch:
    """Each query is matched by semantics AND exact substring, folded together."""

    def _seed_front(self, wrapper, front: str) -> int:
        note = {"deck": "Test", "note_type": "Basic", "fields": {"Front": front, "Back": "x"}}
        return _seed(wrapper, [note])[0]["id"]

    def test_exact_match_without_semantic(self, wrapper, mock_index, mcp_app):
        mock_index.search_by_modality.return_value = _text_hits([[]])  # no semantic hits
        self._seed_front(wrapper, "Electron transport chain")
        m = _call(mcp_app, "search_notes", {"queries": ["transport"]})["results"][0]["matches"]
        assert len(m) == 1
        assert m[0]["score"] is None
        assert m[0]["substring"]["matched_fields"] == ["Front"]

    def test_both_score_and_substring(self, wrapper, mock_index, mcp_app):
        nid = self._seed_front(wrapper, "Electron transport chain")
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": nid, "distance": 0.1}]]
        )  # score 0.9
        m = _call(mcp_app, "search_notes", {"queries": ["transport"]})["results"][0]["matches"][0]
        assert m["score"] == 0.9
        assert m["substring"] is not None

    def test_threshold_does_not_drop_exact(self, wrapper, mock_index, mcp_app):
        nid = self._seed_front(wrapper, "unique phrase here")
        # Semantic score 0.01 is below threshold → not attached; exact still includes it.
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": nid, "distance": 0.99}]]
        )
        m = _call(mcp_app, "search_notes", {"queries": ["unique phrase"], "threshold": 0.5})[
            "results"
        ][0]["matches"]
        assert len(m) == 1
        assert m[0]["score"] is None
        assert m[0]["substring"] is not None

    def test_exact_first_ordering(self, wrapper, mock_index, mcp_app):
        exact_nid = self._seed_front(wrapper, "alpha beta gamma")
        sem_only = self._seed_front(wrapper, "unrelated content")
        # semantic returns the unrelated note with a high score; exact match has none
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": sem_only, "distance": 0.05}]]
        )  # 0.95
        m = _call(mcp_app, "search_notes", {"queries": ["beta gamma"]})["results"][0]["matches"]
        ids = [x["id"] for x in m]
        assert ids[0] == exact_nid  # literal hit ranks first despite no score

    def test_literal_hit_missed_by_prefilter_still_floats(self, wrapper, mock_index, mcp_app):
        # #236 review F1: the exact tier follows the `substring` annotation, not search_substring
        # membership — so a literal hit Anki's normalized *term* pre-filter misses (markup text) or
        # that fell beyond its limit still floats. Simulate the miss by emptying the pre-filter; the
        # note's content still literally contains the query, so the semantic recompute floats it.
        from unittest.mock import AsyncMock

        literal = self._seed_front(wrapper, "alpha beta gamma")
        sem_only = self._seed_front(wrapper, "unrelated content")
        wrapper.search_substring = AsyncMock(return_value=[])  # pre-filter "misses" everything
        mock_index.search_by_modality.return_value = _text_hits(
            [
                [
                    {"note_id": sem_only, "distance": 0.05},  # 0.95 — strong semantic, no literal
                    {"note_id": literal, "distance": 0.20},  # 0.80 — weaker, literal "beta gamma"
                ]
            ]
        )
        m = _call(mcp_app, "search_notes", {"queries": ["beta gamma"], "threshold": 0.5})[
            "results"
        ][0]["matches"]
        assert [x["id"] for x in m][0] == literal  # floats above the stronger-semantic non-literal
        assert m[0]["substring"] is not None

    def test_tags_filter(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": basic_note, "distance": 0.1}]]
        )
        result = _call(mcp_app, "search_notes", {"queries": ["test"], "tags": ["nonexistent-tag"]})
        assert result["results"][0]["matches"] == []

    def test_result_includes_content(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": basic_note, "distance": 0.1}]]
        )
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        match = result["results"][0]["matches"][0]
        assert "content" in match
        assert match["content"]["Front"] == "What is 2+2?"

    def test_top_k_out_of_range_rejected(self, mcp_app, mock_index):
        """top_k is schema-constrained (ge=1, le=50); out-of-range is rejected."""
        from mcp.server.fastmcp.exceptions import ToolError

        mock_index.search_by_modality.return_value = _text_hits([[]])
        with pytest.raises(ToolError):
            _call(mcp_app, "search_notes", {"queries": ["test"], "top_k": 0})

    def test_too_many_queries_rejected(self, mcp_app, mock_index):
        """queries is capped at 50 (schema max_length) to bound embedding load."""
        from mcp.server.fastmcp.exceptions import ToolError

        mock_index.search_by_modality.return_value = _text_hits([[]])
        with pytest.raises(ToolError):
            _call(mcp_app, "search_notes", {"queries": [f"q{i}" for i in range(51)]})

    def test_too_many_ids_rejected(self, mcp_app, mock_index):
        """ids (search anchors) is likewise capped at 50."""
        from mcp.server.fastmcp.exceptions import ToolError

        mock_index.search_by_modality.return_value = _text_hits([[]])
        with pytest.raises(ToolError):
            _call(mcp_app, "search_notes", {"ids": list(range(51))})

    def test_deck_filter_overfetches(self, mcp_app, mock_index):
        """With a deck filter, search over-fetches a wider window (2.3)."""
        mock_index.search_by_modality.return_value = _text_hits([[]])
        _call(mcp_app, "search_notes", {"queries": ["test"], "deck": "D", "top_k": 5})
        assert mock_index.search_by_modality.call_args[1]["top_k"] >= 50  # >= top_k * 10

    def test_no_overfetch_without_filter(self, mcp_app, mock_index):
        """Without deck/tag filters, the window is just top_k (+ excludes)."""
        mock_index.search_by_modality.return_value = _text_hits([[]])
        _call(mcp_app, "search_notes", {"queries": ["test"], "top_k": 5})
        assert mock_index.search_by_modality.call_args[1]["top_k"] == 5

    def test_deck_filter_returns_deep_in_scope_match(
        self, wrapper, mock_index, mcp_app, basic_note
    ):
        """An in-deck note ranked behind out-of-deck neighbors is still returned
        — the widened window must not silently under-return (audit 2.3).

        ``basic_note`` is in deck "Test"; the nearest neighbor here is in another
        deck and ranks ahead of it. A deck-scoped search must skip past the
        out-of-deck hit and still surface the in-deck one.
        """
        other = _seed(
            wrapper,
            [{"deck": "Other", "note_type": "Basic", "fields": {"Front": "O", "Back": "A"}}],
        )[0]["id"]
        mock_index.search_by_modality.return_value = _text_hits(
            [
                [
                    {"note_id": other, "distance": 0.05},
                    {"note_id": basic_note, "distance": 0.20},
                ]
            ]
        )
        result = _call(mcp_app, "search_notes", {"queries": ["q"], "deck": "Test"})
        matches = result["results"][0]["matches"]
        assert [m["id"] for m in matches] == [basic_note]

    def test_score_rounded_to_3_decimals(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": basic_note, "distance": 0.12345}]]
        )
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        score = result["results"][0]["matches"][0]["score"]
        assert score == round(1.0 - 0.12345, 3)

    def test_image_modality_hit_surfaces_unthresholded(self, wrapper, mock_index, mcp_app):
        # An image-modality match with no text match still surfaces the note: the image ranking is
        # its own RRF signal and is NOT thresholded (the text-calibrated threshold is meaningless
        # across the CLIP gap; flooring image hits is the #201b activation gate's job). The surfaced
        # score is the (gap-depressed but real) image cosine.
        nid = self._seed_front(wrapper, "diagram of the krebs cycle")
        mock_index.search_by_modality.return_value = [
            {"image": [{"note_id": nid, "distance": 0.7}]}  # 0.30 sim — below threshold, still kept
        ]
        m = _call(mcp_app, "search_notes", {"queries": ["mitochondria"], "threshold": 0.5})[
            "results"
        ][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["score"] == 0.3

    def test_text_modality_stays_thresholded(self, wrapper, mock_index, mcp_app):
        # The text ranking keeps its threshold: a weak text-only hit with no literal match drops.
        nid = self._seed_front(wrapper, "unrelated content here")
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": nid, "distance": 0.9}]]  # 0.10 sim — below threshold
        )
        m = _call(mcp_app, "search_notes", {"queries": ["xyz"], "threshold": 0.5})["results"][0][
            "matches"
        ]
        assert m == []

    def test_score_is_max_over_matched_modalities(self, wrapper, mock_index, mcp_app):
        # A note matching in both text and image gets the *max* similarity as its surfaced score.
        nid = self._seed_front(wrapper, "alpha")
        mock_index.search_by_modality.return_value = [
            {
                "text": [{"note_id": nid, "distance": 0.1}],  # 0.90 text
                "image": [{"note_id": nid, "distance": 0.7}],  # 0.30 image
            }
        ]
        m = _call(mcp_app, "search_notes", {"queries": ["alpha query"]})["results"][0]["matches"][0]
        assert m["score"] == 0.9  # max(0.90 text, 0.30 image)

    def test_image_gate_passes_strong_match(self, wrapper, mock_index, mcp_app):
        # #201b: calibrated floor = mean + ACTIVATION_MARGIN·std = 0.20 + 1.0·0.05 = 0.25. A best
        # image sim of 0.30 clears it → the (image-only) note surfaces, scored by the image sim.
        nid = self._seed_front(wrapper, "krebs cycle diagram")
        mock_index.activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        mock_index.search_by_modality.return_value = [
            {"image": [{"note_id": nid, "distance": 0.70}]}  # sim 0.30 > 0.25
        ]
        m = _call(mcp_app, "search_notes", {"queries": ["mitochondria"]})["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["score"] == 0.3

    def test_image_gate_drops_weak_match(self, wrapper, mock_index, mcp_app):
        # Best image sim 0.20 is below the 0.25 floor → the image modality is gated out, so an
        # image-only match does not surface (no spurious image card for an off-topic query).
        nid = self._seed_front(wrapper, "krebs cycle diagram")
        mock_index.activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        mock_index.search_by_modality.return_value = [
            {"image": [{"note_id": nid, "distance": 0.80}]}  # sim 0.20 <= 0.25
        ]
        m = _call(mcp_app, "search_notes", {"queries": ["mitochondria"]})["results"][0]["matches"]
        assert m == []

    def test_image_gate_keeps_text_matched_note(self, wrapper, mock_index, mcp_app):
        # Gating the image modality must not drop a note that *also* matches text above threshold;
        # it surfaces with the text score, and the gated image sim is not folded into `score`.
        nid = self._seed_front(wrapper, "alpha")
        mock_index.activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        mock_index.search_by_modality.return_value = [
            {
                "text": [{"note_id": nid, "distance": 0.20}],  # sim 0.80 (above threshold)
                "image": [{"note_id": nid, "distance": 0.80}],  # sim 0.20 (gated out)
            }
        ]
        m = _call(mcp_app, "search_notes", {"queries": ["alpha query"]})["results"][0]["matches"][0]
        assert m["id"] == nid
        assert m["score"] == 0.8  # text sim only; the gated image sim is not the max

    def test_image_gate_judges_surviving_hit(self, wrapper, mock_index, mcp_app):
        # #201b review F1: the gate must judge the best image hit that *survives* exclusion/scope,
        # not the raw rank-1. Here the strong rank-1 image hit is the excluded anchor; the only
        # surviving image hit is weak (below the 0.25 floor) → the modality must be gated out.
        anchor = self._seed_front(wrapper, "anchor card")
        weak = self._seed_front(wrapper, "weakly related card")
        mock_index.activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        mock_index.search_by_modality.return_value = [
            {
                "image": [
                    {"note_id": anchor, "distance": 0.65},  # sim 0.35 > floor, but excluded
                    {"note_id": weak, "distance": 0.80},  # sim 0.20 <= floor → the surviving best
                ]
            }
        ]
        m = _call(mcp_app, "search_notes", {"queries": ["q"], "exclude_ids": [anchor]})["results"][
            0
        ]["matches"]
        assert m == []  # gated on the surviving (weak) hit, so nothing surfaces

    def test_image_gate_passes_strong_surviving_hit(self, wrapper, mock_index, mcp_app):
        # The mirror: with the strong anchor excluded, a surviving hit that itself clears the floor
        # still surfaces — the gate isn't fooled in either direction.
        anchor = self._seed_front(wrapper, "anchor card")
        strong = self._seed_front(wrapper, "strongly matching card")
        mock_index.activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        mock_index.search_by_modality.return_value = [
            {
                "image": [
                    {"note_id": anchor, "distance": 0.55},  # sim 0.45, excluded
                    {"note_id": strong, "distance": 0.66},  # sim 0.34 > floor → surfaces
                ]
            }
        ]
        m = _call(mcp_app, "search_notes", {"queries": ["q"], "exclude_ids": [anchor]})["results"][
            0
        ]["matches"]
        assert [x["id"] for x in m] == [strong]


class TestProvenance:
    """Per-result provenance (#182): which signals surfaced each match, at what rank."""

    def _seed_front(self, wrapper, front: str) -> int:
        note = {"deck": "Test", "note_type": "Basic", "fields": {"Front": front, "Back": "x"}}
        return _seed(wrapper, [note])[0]["id"]

    @staticmethod
    def _matches(mcp_app, query: str) -> list[dict]:
        return _call(mcp_app, "search_notes", {"queries": [query]})["results"][0]["matches"]

    def test_text_only(self, wrapper, mock_index, mcp_app):
        nid = self._seed_front(wrapper, "mitochondria powerhouse")
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": nid, "distance": 0.2}]]
        )
        m = self._matches(mcp_app, "cellular energy")[0]
        assert [(p["signal"], p["rank"]) for p in m["provenance"]] == [("text", 1)]
        assert m["score"] == 0.8  # back-compat field stays, consistent with the text signal

    def test_image_modality_facet(self, wrapper, mock_index, mcp_app):
        # The semantic signal name *is* the matched-modality facet — `image` ⇒ "matched on image".
        nid = self._seed_front(wrapper, "krebs cycle diagram card")
        mock_index.search_by_modality.return_value = [
            {"image": [{"note_id": nid, "distance": 0.7}]}  # uncalibrated → gate off → surfaces
        ]
        m = self._matches(mcp_app, "mitochondria")[0]
        assert [p["signal"] for p in m["provenance"]] == ["image"]
        assert m["score"] == 0.3

    def test_exact_only(self, wrapper, mock_index, mcp_app):
        self._seed_front(wrapper, "unique exact phrase")
        mock_index.search_by_modality.return_value = _text_hits([[]])  # no semantic hit
        m = self._matches(mcp_app, "exact phrase")[0]
        assert [p["signal"] for p in m["provenance"]] == ["exact"]
        assert m["score"] is None  # back-compat: exact-only carries no score
        assert m["substring"] is not None  # ...but the substring detail stays

    def test_text_and_exact(self, wrapper, mock_index, mcp_app):
        nid = self._seed_front(wrapper, "Electron transport chain")
        mock_index.search_by_modality.return_value = _text_hits(
            [[{"note_id": nid, "distance": 0.1}]]
        )
        m = self._matches(mcp_app, "transport")[0]
        # Both fire at rank 1 → ordered by signal name (exact < text); back-compat fields agree.
        assert {p["signal"]: p["rank"] for p in m["provenance"]} == {"text": 1, "exact": 1}
        assert m["score"] == 0.9
        assert m["substring"] is not None

    def test_ordered_by_rank_then_signal(self, wrapper, mock_index, mcp_app):
        a = self._seed_front(wrapper, "alpha card")
        b = self._seed_front(wrapper, "beta card")
        nid = self._seed_front(wrapper, "gamma card")
        # nid trails a, b in text (rank 3) but leads the image ranking (rank 1).
        mock_index.search_by_modality.return_value = [
            {
                "text": [
                    {"note_id": a, "distance": 0.10},
                    {"note_id": b, "distance": 0.15},
                    {"note_id": nid, "distance": 0.20},
                ],
                "image": [{"note_id": nid, "distance": 0.65}],
            }
        ]
        matches = self._matches(mcp_app, "q")
        assert all(m["provenance"] for m in matches)  # every returned match carries provenance
        prov = {m["id"]: [(p["signal"], p["rank"]) for p in m["provenance"]] for m in matches}
        assert prov[nid] == [("image", 1), ("text", 3)]  # strongest (lowest-rank) signal first
        assert prov[a] == [("text", 1)]


class TestUpsertNeighbors:
    def test_neighbors_attached_on_create(self, wrapper, mock_index, mcp_app):
        existing = _seed(wrapper, [BASIC_NOTE])[0]["id"]
        mock_index.search.return_value = [[{"note_id": existing, "distance": 0.2}]]
        result = _upsert(mcp_app, [BASIC_NOTE])
        r = result["results"][0]
        assert r["status"] == "created"
        assert "neighbors" in r
        assert len(r["neighbors"]) == 1
        assert r["neighbors"][0]["id"] == existing
        assert r["neighbors"][0]["score"] == 0.8

    def test_neighbors_have_tags(self, wrapper, mock_index, mcp_app):
        existing = _seed(wrapper, [{**BASIC_NOTE, "tags": ["science", "physics"]}])[0]["id"]
        mock_index.search.return_value = [[{"note_id": existing, "distance": 0.1}]]
        result = _upsert(mcp_app, [BASIC_NOTE])
        neighbors = result["results"][0]["neighbors"]
        assert set(neighbors[0]["tags"]) == {"science", "physics"}

    def test_threshold_filters_low_scores(self, wrapper, mock_index, mcp_app):
        existing = _seed(wrapper, [BASIC_NOTE])[0]["id"]
        mock_index.search.return_value = [[{"note_id": existing, "distance": 0.8}]]
        result = _upsert(mcp_app, [BASIC_NOTE], neighbor_threshold=0.5)
        neighbors = result["results"][0].get("neighbors", [])
        assert len(neighbors) == 0

    def test_default_threshold_filters_irrelevant(self, wrapper, mock_index, mcp_app):
        existing = _seed(wrapper, [BASIC_NOTE])[0]["id"]
        mock_index.search.return_value = [[{"note_id": existing, "distance": 0.7}]]
        result = _upsert(mcp_app, [BASIC_NOTE])
        neighbors = result["results"][0].get("neighbors", [])
        assert len(neighbors) == 0

    def test_custom_threshold(self, wrapper, mock_index, mcp_app):
        existing = _seed(wrapper, [BASIC_NOTE])[0]["id"]
        mock_index.search.return_value = [[{"note_id": existing, "distance": 0.15}]]
        result = _upsert(mcp_app, [BASIC_NOTE], neighbor_threshold=0.9)
        neighbors = result["results"][0].get("neighbors", [])
        assert len(neighbors) == 0

    def test_top_k_limits_neighbors(self, wrapper, mock_index, mcp_app):
        ids = []
        for i in range(5):
            nid = _seed(wrapper, [{**BASIC_NOTE, "fields": {"Front": f"E{i}", "Back": "A"}}])[0][
                "id"
            ]
            ids.append(nid)
        mock_index.search.return_value = [[{"note_id": nid, "distance": 0.1} for nid in ids]]
        result = _upsert(mcp_app, [BASIC_NOTE], top_k_neighbors=2)
        neighbors = result["results"][0]["neighbors"]
        assert len(neighbors) == 2

    def test_excludes_batch_notes_from_neighbors(self, wrapper, mock_index, mcp_app):
        mock_index.search.return_value = [[], []]
        _upsert(mcp_app, [BASIC_NOTE, BASIC_NOTE])
        call_args = mock_index.search.call_args
        top_k_used = call_args[1]["top_k"]
        assert top_k_used >= 7

    def test_no_neighbors_without_index(self, mcp_no_index):
        result = _upsert(mcp_no_index, [BASIC_NOTE])
        # No embedding service: the success variant carries an empty neighbor list.
        assert result["results"][0]["neighbors"] == []

    def test_neighbor_failure_doesnt_fail_upsert(self, wrapper, mock_index, mcp_app):
        mock_index.search.side_effect = RuntimeError("embedding service down")
        result = _upsert(mcp_app, [BASIC_NOTE])
        assert result["results"][0]["status"] == "created"

    def test_neighbor_failure_flags_retry(self, wrapper, mock_index, mcp_app):
        """A neighbor-search hiccup flags the result and hints the retry path."""
        mock_index.search.side_effect = RuntimeError("embedding service down")
        result = _upsert(mcp_app, [BASIC_NOTE])
        r = result["results"][0]
        nid = r["id"]
        assert r["status"] == "created"
        assert r["neighbors"] == []
        assert r["neighbors_unavailable"] is True
        assert f"search_notes(ids=[{nid}])" in result["message"]

    def test_neighbors_on_update(self, wrapper, mock_index, mcp_app, basic_note):
        other = _seed(wrapper, [BASIC_NOTE])[0]["id"]
        mock_index.search.return_value = [[{"note_id": other, "distance": 0.3}]]
        result = _upsert(mcp_app, [{"id": basic_note, "fields": {"Front": "Updated"}}])
        r = result["results"][0]
        assert r["status"] == "updated"
        assert "neighbors" in r

    def test_error_results_have_no_neighbors(self, wrapper, mock_index, mcp_app):
        mock_index.search.return_value = [[]]
        result = _upsert(
            mcp_app,
            [BASIC_NOTE, {"note_type": "Nonexistent", "fields": {"X": "Y"}}],
        )
        ok = [r for r in result["results"] if r.get("status") == "created"]
        err = [r for r in result["results"] if r.get("status") == "error"]
        assert len(ok) == 1
        assert len(err) == 1
        # The error variant has no neighbors field at all (discriminated union).
        assert "neighbors" not in err[0]


class TestDeleteIndexUpdate:
    def test_removes_from_index(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.remove.return_value = 1
        result = _call(mcp_app, "delete_notes", {"ids": [basic_note]})
        assert basic_note in result["deleted"]
        mock_index.remove.assert_called_once_with([basic_note])

    def test_index_failure_doesnt_fail_delete(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.remove.side_effect = RuntimeError("index broken")
        result = _call(mcp_app, "delete_notes", {"ids": [basic_note]})
        assert basic_note in result["deleted"]

    def test_no_index_call_on_not_found(self, wrapper, mock_index, mcp_app):
        _call(mcp_app, "delete_notes", {"ids": [9999999999999]})
        mock_index.remove.assert_not_called()

    def test_updates_col_mod(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.remove.return_value = 1
        _call(mcp_app, "delete_notes", {"ids": [basic_note]})
        assert mock_index.col_mod == wrapper.col.mod


class TestUpsertIndexUpdate:
    def test_adds_to_index(self, wrapper, mock_index, mcp_app):
        mock_index.search.return_value = [[]]
        result = _upsert(mcp_app, [BASIC_NOTE])
        nid = result["results"][0]["id"]
        mock_index.add.assert_called_once()
        call_inputs = mock_index.add.call_args[0][0]
        assert nid in [i.note_id for i in call_inputs]

    def test_updates_col_mod_after_upsert(self, wrapper, mock_index, mcp_app):
        mock_index.search.return_value = [[]]
        _upsert(mcp_app, [BASIC_NOTE])
        assert mock_index.col_mod == wrapper.col.mod

    def test_index_add_failure_doesnt_fail_upsert(self, wrapper, mock_index, mcp_app):
        mock_index.add.side_effect = RuntimeError("embed failed")
        result = _upsert(mcp_app, [BASIC_NOTE])
        assert result["results"][0]["status"] == "created"

    def test_index_add_failure_flags_retry(self, wrapper, mock_index, mcp_app):
        """An index.add hiccup also flags neighbors_unavailable and hints retry."""
        mock_index.add.side_effect = RuntimeError("embed failed")
        result = _upsert(mcp_app, [BASIC_NOTE])
        r = result["results"][0]
        assert r["neighbors_unavailable"] is True
        assert f"search_notes(ids=[{r['id']}])" in result["message"]

    def test_note_texts_failure_doesnt_fail_upsert(self, wrapper, mock_index, mcp_app):
        """If note_texts_for_embedding raises, the already-committed notes must
        still report created — not a NameError-driven false failure (audit 3.3).

        Before the fix, an exception here left `texts` unbound and the later
        neighbor-attach raised NameError, surfacing as a whole-call error even
        though the upsert had succeeded.
        """

        async def boom(_ids):
            raise RuntimeError("embedding text build failed")

        wrapper.note_embed_inputs = boom
        result = _upsert(mcp_app, [BASIC_NOTE])
        r = result["results"][0]
        assert r["status"] == "created"
        assert r["neighbors"] == []
        assert r["neighbors_unavailable"] is True
        assert f"search_notes(ids=[{r['id']}])" in result["message"]

    def test_no_retry_hint_on_success(self, wrapper, mock_index, mcp_app):
        """Successful neighbor computation carries no retry flag or message."""
        mock_index.search.return_value = [[]]
        result = _upsert(mcp_app, [BASIC_NOTE])
        r = result["results"][0]
        assert r["neighbors"] == []
        assert r["neighbors_unavailable"] is False
        assert result["message"] is None


class TestUpsertPolicyTool:
    """The upsert_notes *tool* defaults (error-on-duplicate); dry_run echoes."""

    def test_tool_default_errors_on_duplicate(self, wrapper, mock_index, mcp_app):
        # Call the tool directly (not the _upsert helper, which forces allow) so
        # the registered default on_duplicate="error" is what's exercised.
        first = _call(mcp_app, "upsert_notes", {"notes": [BASIC_NOTE]})
        assert first["results"][0]["status"] == "created"

        second = _call(mcp_app, "upsert_notes", {"notes": [BASIC_NOTE]})
        assert second["results"][0]["status"] == "error"
        assert second["results"][0]["reason"] == "duplicate"

    def test_dry_run_echoed_and_skips_index(self, wrapper, mock_index, mcp_app):
        result = _call(mcp_app, "upsert_notes", {"notes": [BASIC_NOTE], "dry_run": True})
        assert result["dry_run"] is True
        assert result["results"][0] == {"status": "ok", "index": 0, "action": "create"}
        # No write, so the index is never touched on a dry run.
        mock_index.add.assert_not_called()
