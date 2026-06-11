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
    import json

    return wrapper.run_sync(lambda c: json.loads(c.upsert_notes(json.dumps(notes), "allow", False)))


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


def _unit_vec(sim: float) -> list[float]:
    """A 2-dim unit vector at cosine ``sim`` against the [1, 0] query."""
    import math

    s = max(min(sim, 1.0), -1.0)
    return [s, math.sqrt(max(0.0, 1.0 - s * s))]


def _plant(index: VectorIndex, items: list[tuple[int, float]], modality: str = "text") -> None:
    """Plant vectors so the REAL engine ranks ``items`` at exactly those distances.

    The fake backend embeds every query as [1, 0]; a note planted at cosine
    ``1 - distance`` therefore comes back from the native engine at
    ``distance`` — the search clusters keep their scripted numbers while the
    re-homed (#331) Rust assembly runs against the real index.
    """
    import numpy as np

    keys = [nid for nid, _ in items]
    vecs = np.asarray([_unit_vec(1.0 - d) for _, d in items], dtype=np.float32)
    index._engine.add(modality, np.asarray(keys, dtype=np.int64), vecs)


@pytest.fixture()
def sem_index(tmp_path):
    """A REAL VectorIndex (native engine) with a fake [1,0]-embedding backend.

    The acknowledged #279/#331 churn: the search assembly runs in Rust, so the
    old scripted ``search_by_modality`` mocks can't inject — these tests plant
    vectors instead and exercise the genuine path end to end.
    """
    backend = MagicMock()
    backend.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    backend.modalities = frozenset({"text"})
    idx = VectorIndex(tmp_path / "sem-index", backend=backend)
    idx.materialize_empty(2, col_mod=1, model_id="m")
    return idx


@pytest.fixture()
def mcp_sem(wrapper, sem_index):
    mcp = FastMCP("test")
    register_tools(mcp, wrapper, index=sem_index)
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
    def test_text_query(self, wrapper, sem_index, mcp_sem, basic_note):
        _plant(sem_index, [(basic_note, 0.1)])
        result = _call(mcp_sem, "search_notes", {"queries": ["math question"]})
        assert len(result["results"]) == 1
        assert result["results"][0]["source"] == "math question"
        matches = result["results"][0]["matches"]
        assert len(matches) == 1
        assert matches[0]["id"] == basic_note
        assert matches[0]["score"] == 0.9

    def test_id_query(self, wrapper, sem_index, mcp_sem, basic_note):
        other = _seed(
            wrapper, [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]
        )[0]["id"]
        _plant(sem_index, [(other, 0.2)])
        result = _call(mcp_sem, "search_notes", {"ids": [basic_note]})
        assert len(result["results"]) == 1
        assert result["results"][0]["source"] == f"note #{basic_note}"

    def test_exclude_ids(self, wrapper, sem_index, mcp_sem, basic_note):
        other = _seed(
            wrapper, [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]
        )[0]["id"]
        _plant(sem_index, [(basic_note, 0.05), (other, 0.2)])
        result = _call(mcp_sem, "search_notes", {"queries": ["test"], "exclude_ids": [basic_note]})
        matches = result["results"][0]["matches"]
        assert all(m["id"] != basic_note for m in matches)

    def test_deck_filter(self, wrapper, sem_index, mcp_sem, basic_note):
        _plant(sem_index, [(basic_note, 0.1)])
        result = _call(mcp_sem, "search_notes", {"queries": ["test"], "deck": "Nonexistent"})
        assert result["results"][0]["matches"] == []


class TestUnifiedSearch:
    """Each query is matched by semantics AND exact substring, folded together."""

    def _seed_front(self, wrapper, front: str) -> int:
        note = {"deck": "Test", "note_type": "Basic", "fields": {"Front": front, "Back": "x"}}
        return _seed(wrapper, [note])[0]["id"]

    def test_exact_match_without_semantic(self, wrapper, sem_index, mcp_sem):
        # Nothing planted → no semantic hits; the literal path alone surfaces it.
        self._seed_front(wrapper, "Electron transport chain")
        m = _call(mcp_sem, "search_notes", {"queries": ["transport"]})["results"][0]["matches"]
        assert len(m) == 1
        assert m[0]["score"] is None
        assert m[0]["substring"]["matched_fields"] == ["Front"]

    def test_both_score_and_substring(self, wrapper, sem_index, mcp_sem):
        nid = self._seed_front(wrapper, "Electron transport chain")
        _plant(sem_index, [(nid, 0.1)])  # score 0.9
        m = _call(mcp_sem, "search_notes", {"queries": ["transport"]})["results"][0]["matches"][0]
        assert m["score"] == 0.9
        assert m["substring"] is not None

    def test_threshold_does_not_drop_exact(self, wrapper, sem_index, mcp_sem):
        nid = self._seed_front(wrapper, "unique phrase here")
        # Semantic score 0.01 is below threshold → not attached; exact still includes it.
        _plant(sem_index, [(nid, 0.99)])
        m = _call(mcp_sem, "search_notes", {"queries": ["unique phrase"], "threshold": 0.5})[
            "results"
        ][0]["matches"]
        assert len(m) == 1
        assert m[0]["score"] is None
        assert m[0]["substring"] is not None

    def test_exact_first_ordering(self, wrapper, sem_index, mcp_sem):
        exact_nid = self._seed_front(wrapper, "alpha beta gamma")
        sem_only = self._seed_front(wrapper, "unrelated content")
        # semantic ranks the unrelated note with a high score; exact match has none
        _plant(sem_index, [(sem_only, 0.05)])  # 0.95
        m = _call(mcp_sem, "search_notes", {"queries": ["beta gamma"]})["results"][0]["matches"]
        ids = [x["id"] for x in m]
        assert ids[0] == exact_nid  # literal hit ranks first despite no score

    def test_literal_hit_missed_by_prefilter_still_floats(self, wrapper, sem_index, mcp_sem):
        # #236 review F1: the exact tier follows the `substring` annotation, not pre-filter
        # membership — a literal hit that reaches note_data only through the SEMANTIC ranking
        # (no derived store here, and a deck scope below would be the other route) still gets
        # the annotation recompute and floats. The query contains a '*' so Anki's wildcard
        # pre-filter can't literally match it, but the field text does contain it.
        literal = self._seed_front(wrapper, "alpha *beta* gamma")
        sem_only = self._seed_front(wrapper, "unrelated content")
        _plant(sem_index, [(sem_only, 0.05), (literal, 0.20)])
        m = _call(mcp_sem, "search_notes", {"queries": ["*beta* gamma"], "threshold": 0.5})[
            "results"
        ][0]["matches"]
        assert [x["id"] for x in m][0] == literal  # floats above the stronger-semantic non-literal
        assert m[0]["substring"] is not None

    def test_tags_filter(self, wrapper, sem_index, mcp_sem, basic_note):
        _plant(sem_index, [(basic_note, 0.1)])
        result = _call(mcp_sem, "search_notes", {"queries": ["test"], "tags": ["nonexistent-tag"]})
        assert result["results"][0]["matches"] == []

    def test_result_includes_content(self, wrapper, sem_index, mcp_sem, basic_note):
        _plant(sem_index, [(basic_note, 0.1)])
        result = _call(mcp_sem, "search_notes", {"queries": ["test"]})
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

    # (The over-fetch window internals moved into the kernel with the #331
    # re-home; the outcome they exist for is pinned by
    # test_deck_filter_returns_deep_in_scope_match below.)

    def test_deck_filter_returns_deep_in_scope_match(self, wrapper, sem_index, mcp_sem, basic_note):
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
        _plant(sem_index, [(other, 0.05), (basic_note, 0.20)])
        result = _call(mcp_sem, "search_notes", {"queries": ["qry"], "deck": "Test"})
        matches = result["results"][0]["matches"]
        assert [m["id"] for m in matches] == [basic_note]

    def test_score_rounded_to_3_decimals(self, wrapper, sem_index, mcp_sem, basic_note):
        _plant(sem_index, [(basic_note, 0.12345)])
        result = _call(mcp_sem, "search_notes", {"queries": ["test"]})
        score = result["results"][0]["matches"][0]["score"]
        assert score == round(1.0 - 0.12345, 3)

    def test_image_modality_hit_surfaces_unthresholded(self, wrapper, sem_index, mcp_sem):
        # An image-modality match with no text match still surfaces the note: the image ranking is
        # its own RRF signal and is NOT thresholded (the text-calibrated threshold is meaningless
        # across the CLIP gap; flooring image hits is the #201b activation gate's job). The surfaced
        # score is the (gap-depressed but real) image cosine.
        nid = self._seed_front(wrapper, "diagram of the krebs cycle")
        _plant(sem_index, [(nid, 0.7)], modality="image")  # 0.30 sim — below threshold, kept
        m = _call(mcp_sem, "search_notes", {"queries": ["mitochondria"], "threshold": 0.5})[
            "results"
        ][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["score"] == 0.3

    def test_text_modality_stays_thresholded(self, wrapper, sem_index, mcp_sem):
        # The text ranking keeps its threshold: a weak text-only hit with no literal match drops.
        nid = self._seed_front(wrapper, "unrelated content here")
        _plant(sem_index, [(nid, 0.9)])  # 0.10 sim — below threshold
        m = _call(mcp_sem, "search_notes", {"queries": ["xyz"], "threshold": 0.5})["results"][0][
            "matches"
        ]
        assert m == []

    def test_score_is_max_over_matched_modalities(self, wrapper, sem_index, mcp_sem):
        # A note matching in both text and image gets the *max* similarity as its surfaced score.
        nid = self._seed_front(wrapper, "alpha")
        _plant(sem_index, [(nid, 0.1)])  # 0.90 text
        _plant(sem_index, [(nid, 0.7)], modality="image")  # 0.30 image
        m = _call(mcp_sem, "search_notes", {"queries": ["alpha query"]})["results"][0]["matches"][0]
        assert m["score"] == 0.9  # max(0.90 text, 0.30 image)

    def test_image_gate_passes_strong_match(self, wrapper, sem_index, mcp_sem):
        # #201b: calibrated floor = mean + ACTIVATION_MARGIN·std = 0.20 + 1.0·0.05 = 0.25. A best
        # image sim of 0.30 clears it → the (image-only) note surfaces, scored by the image sim.
        nid = self._seed_front(wrapper, "krebs cycle diagram")
        sem_index._activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(sem_index, [(nid, 0.70)], modality="image")  # sim 0.30 > 0.25
        m = _call(mcp_sem, "search_notes", {"queries": ["mitochondria"]})["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["score"] == 0.3

    def test_image_gate_drops_weak_match(self, wrapper, sem_index, mcp_sem):
        # Best image sim 0.20 is below the 0.25 floor → the image modality is gated out, so an
        # image-only match does not surface (no spurious image card for an off-topic query).
        nid = self._seed_front(wrapper, "krebs cycle diagram")
        sem_index._activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(sem_index, [(nid, 0.80)], modality="image")  # sim 0.20 <= 0.25
        m = _call(mcp_sem, "search_notes", {"queries": ["mitochondria"]})["results"][0]["matches"]
        assert m == []

    def test_image_gate_keeps_text_matched_note(self, wrapper, sem_index, mcp_sem):
        # Gating the image modality must not drop a note that *also* matches text above threshold;
        # it surfaces with the text score, and the gated image sim is not folded into `score`.
        nid = self._seed_front(wrapper, "alpha")
        sem_index._activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(sem_index, [(nid, 0.20)])  # sim 0.80 (above threshold)
        _plant(sem_index, [(nid, 0.80)], modality="image")  # sim 0.20 (gated out)
        m = _call(mcp_sem, "search_notes", {"queries": ["alpha query"]})["results"][0]["matches"][0]
        assert m["id"] == nid
        assert m["score"] == 0.8  # text sim only; the gated image sim is not the max

    def test_image_gate_judges_surviving_hit(self, wrapper, sem_index, mcp_sem):
        # #201b review F1: the gate must judge the best image hit that *survives* exclusion/scope,
        # not the raw rank-1. Here the strong rank-1 image hit is the excluded anchor; the only
        # surviving image hit is weak (below the 0.25 floor) → the modality must be gated out.
        anchor = self._seed_front(wrapper, "anchor card")
        weak = self._seed_front(wrapper, "weakly related card")
        sem_index._activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(sem_index, [(anchor, 0.65), (weak, 0.80)], modality="image")
        m = _call(mcp_sem, "search_notes", {"queries": ["qry"], "exclude_ids": [anchor]})[
            "results"
        ][0]["matches"]
        assert m == []  # gated on the surviving (weak) hit, so nothing surfaces

    def test_image_gate_passes_strong_surviving_hit(self, wrapper, sem_index, mcp_sem):
        # The mirror: with the strong anchor excluded, a surviving hit that itself clears the floor
        # still surfaces — the gate isn't fooled in either direction.
        anchor = self._seed_front(wrapper, "anchor card")
        strong = self._seed_front(wrapper, "strongly matching card")
        sem_index._activation_stats = {"image": {"n": 40, "mean": 0.20, "std": 0.05}}
        _plant(sem_index, [(anchor, 0.55), (strong, 0.66)], modality="image")
        m = _call(mcp_sem, "search_notes", {"queries": ["qry"], "exclude_ids": [anchor]})[
            "results"
        ][0]["matches"]
        assert [x["id"] for x in m] == [strong]


class TestProvenance:
    """Per-result provenance (#182): which signals surfaced each match, at what rank."""

    def _seed_front(self, wrapper, front: str) -> int:
        note = {"deck": "Test", "note_type": "Basic", "fields": {"Front": front, "Back": "x"}}
        return _seed(wrapper, [note])[0]["id"]

    @staticmethod
    def _matches(mcp_app, query: str) -> list[dict]:
        return _call(mcp_app, "search_notes", {"queries": [query]})["results"][0]["matches"]

    def test_text_only(self, wrapper, sem_index, mcp_sem):
        nid = self._seed_front(wrapper, "mitochondria powerhouse")
        _plant(sem_index, [(nid, 0.2)])
        m = self._matches(mcp_sem, "cellular energy")[0]
        assert [(p["signal"], p["rank"]) for p in m["provenance"]] == [("text", 1)]
        assert m["score"] == 0.8  # back-compat field stays, consistent with the text signal

    def test_image_modality_facet(self, wrapper, sem_index, mcp_sem):
        # The semantic signal name *is* the matched-modality facet — `image` ⇒ "matched on image".
        nid = self._seed_front(wrapper, "krebs cycle diagram card")
        _plant(sem_index, [(nid, 0.7)], modality="image")  # uncalibrated → gate off → surfaces
        m = self._matches(mcp_sem, "mitochondria")[0]
        assert [p["signal"] for p in m["provenance"]] == ["image"]
        assert m["score"] == 0.3

    def test_exact_only(self, wrapper, sem_index, mcp_sem):
        self._seed_front(wrapper, "unique exact phrase")
        m = self._matches(mcp_sem, "exact phrase")[0]  # nothing planted → no semantic hit
        assert [p["signal"] for p in m["provenance"]] == ["exact"]
        assert m["score"] is None  # back-compat: exact-only carries no score
        assert m["substring"] is not None  # ...but the substring detail stays

    def test_text_and_exact(self, wrapper, sem_index, mcp_sem):
        nid = self._seed_front(wrapper, "Electron transport chain")
        _plant(sem_index, [(nid, 0.1)])
        m = self._matches(mcp_sem, "transport")[0]
        # Both fire at rank 1 → ordered by signal name (exact < text); back-compat fields agree.
        assert {p["signal"]: p["rank"] for p in m["provenance"]} == {"text": 1, "exact": 1}
        assert m["score"] == 0.9
        assert m["substring"] is not None

    def test_ordered_by_rank_then_signal(self, wrapper, sem_index, mcp_sem):
        a = self._seed_front(wrapper, "alpha card")
        b = self._seed_front(wrapper, "beta card")
        nid = self._seed_front(wrapper, "gamma card")
        # nid trails a, b in text (rank 3) but leads the image ranking (rank 1).
        _plant(sem_index, [(a, 0.10), (b, 0.15), (nid, 0.20)])
        _plant(sem_index, [(nid, 0.65)], modality="image")
        matches = self._matches(mcp_sem, "qry")
        assert all(m["provenance"] for m in matches)  # every returned match carries provenance
        prov = {m["id"]: [(p["signal"], p["rank"]) for p in m["provenance"]] for m in matches}
        assert prov[nid] == [("image", 1), ("text", 3)]  # strongest (lowest-rank) signal first
        assert prov[a] == [("text", 1)]


def _build_derived(wrapper, derived) -> None:
    """Build the derived store from the collection's current notes (what the boot path does)."""

    rows, mod = wrapper.run_sync(
        lambda c: (
            c.derived_field_rows(c.find_notes("deck:*")),
            c.col_mod(),
        )
    )
    derived.build(rows, mod)


class TestDerivedSearch:
    """search_notes wired to the derived store: substring-via-store + the fuzzy signal (#98)."""

    @pytest.fixture()
    def derived(self, tmp_path):
        from shrike.derived import DerivedTextStore

        s = DerivedTextStore(path=tmp_path / "shrike.db")
        yield s
        s.close()

    @pytest.fixture()
    def mcp_derived(self, wrapper, sem_index, derived):
        mcp = FastMCP("test")
        register_tools(mcp, wrapper, index=sem_index, derived=derived)
        return mcp

    def _seed_front(self, wrapper, front: str) -> int:
        note = {"deck": "Test", "note_type": "Basic", "fields": {"Front": front, "Back": "x"}}
        return _seed(wrapper, [note])[0]["id"]

    def test_substring_via_store_matches_find_notes(self, wrapper, mcp_derived, derived):
        # An exact substring hit comes through the store (candidate) + substring_info (authority),
        # identical to the find_notes path: matched field + the `exact` provenance.
        nid = self._seed_front(wrapper, "Electron transport chain")
        _build_derived(wrapper, derived)
        res = _call(mcp_derived, "search_notes", {"queries": ["transport"]})
        m = res["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        assert m[0]["substring"]["matched_fields"] == ["Front"]
        assert m[0]["substring"]["source"] == "field"
        # A literal hit shares every trigram so it's *trivially* also a fuzzy match, but `fuzzy` is
        # suppressed on exact hits (review F4) — `exact` is the distinguishing lexical signal.
        assert [p["signal"] for p in m[0]["provenance"]] == ["exact"]
        assert m[0].get("fuzzy") is None

    def test_fuzzy_only_hit_surfaces_with_provenance(self, wrapper, mcp_derived, derived):
        # A typo query the note doesn't literally contain surfaces via the `fuzzy` signal alone:
        # no score, no substring, provenance == [fuzzy], carrying the source/ref/snippet window.
        nid = self._seed_front(wrapper, "Mitochondria are the powerhouse")
        _build_derived(wrapper, derived)
        res = _call(mcp_derived, "search_notes", {"queries": ["mitochndria"]})
        m = res["results"][0]["matches"]
        assert [x["id"] for x in m] == [nid]
        hit = m[0]
        assert hit["score"] is None
        assert hit["substring"] is None
        assert [p["signal"] for p in hit["provenance"]] == ["fuzzy"]
        assert hit["fuzzy"]["source"] == "field"
        assert hit["fuzzy"]["ref"] == "Front"
        assert "Mitochondria" in hit["fuzzy"]["snippet"]

    def test_literal_tiers_above_fuzzy(self, wrapper, mcp_derived, derived):
        # The exact-match override still wins: a literal hit floats above a fuzzy-only near-miss.
        literal = self._seed_front(wrapper, "Mitochondria diagram")
        fuzzy_only = self._seed_front(wrapper, "mitochndrial membrane")  # typo → no literal hit
        _build_derived(wrapper, derived)
        m = _call(mcp_derived, "search_notes", {"queries": ["mitochondria"]})["results"][0][
            "matches"
        ]
        assert [x["id"] for x in m][0] == literal  # literal floats to the top
        prov = {x["id"]: [p["signal"] for p in x["provenance"]] for x in m}
        assert "exact" in prov[literal]
        assert prov[fuzzy_only] == ["fuzzy"]  # the near-miss is fuzzy-only

    def test_no_fuzzy_signal_when_store_unavailable(self, wrapper, sem_index, mcp_sem):
        # Fallback safety: with no derived store (mcp_app), a typo query emits no fuzzy match —
        # substring still works via find_notes, exactly as before #98.
        self._seed_front(wrapper, "Mitochondria are the powerhouse")
        m = _call(mcp_sem, "search_notes", {"queries": ["mitochndria"]})["results"][0]["matches"]
        assert m == []

    def test_exact_hit_carries_no_fuzzy(self, wrapper, mcp_derived, derived):
        # Review F4: a clean exact (literal) match must not also be badged `fuzzy`, even though it
        # shares every trigram — `fuzzy` is reserved for the distinguishing near-miss signal.
        nid = self._seed_front(wrapper, "powerhouse of the cell")
        _build_derived(wrapper, derived)
        m = _call(mcp_derived, "search_notes", {"queries": ["powerhouse"]})["results"][0]["matches"]
        hit = next(x for x in m if x["id"] == nid)
        assert "fuzzy" not in [p["signal"] for p in hit["provenance"]]
        assert hit.get("fuzzy") is None

    def test_result_capped_at_top_k(self, wrapper, mcp_derived, derived):
        # Review F5: the fused union (text/image/exact/fuzzy, each up to top_k) is capped to top_k,
        # so a broad fuzzy signal can't inflate a query's result count past the documented cap.
        for i in range(8):
            self._seed_front(wrapper, f"mitochondrion variant {i}")  # all fuzzy-match the typo
        _build_derived(wrapper, derived)
        m = _call(mcp_derived, "search_notes", {"queries": ["mitochndrion"], "top_k": 3})[
            "results"
        ][0]["matches"]
        assert len(m) == 3


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
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())


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
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())

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


class TestTwoTierSearch:
    """The live-search tier contract (#181): tier='live' runs only the
    no-embedding signals and reports partial; the min-query gate keeps typing
    fragments from burning embedding calls; `version` echoes verbatim."""

    def test_live_tier_skips_semantic_and_reports_partial(self, wrapper, sem_index, mcp_sem):
        planted = _seed(
            wrapper,
            [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "sem only", "Back": "A"}}],
        )[0]["id"]
        _plant(sem_index, [(planted, 0.05)])
        result = _call(mcp_sem, "search_notes", {"queries": ["qry"], "tier": "live", "version": 7})
        assert result["completeness"] == "partial"
        assert result["version"] == 7
        # The semantically-planted note does not surface on the live tier.
        ids = [m["id"] for m in result["results"][0]["matches"]]
        assert planted not in ids

    def test_full_tier_reports_full_and_finds_semantic(self, wrapper, sem_index, mcp_sem):
        planted = _seed(
            wrapper,
            [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "sem hit", "Back": "A"}}],
        )[0]["id"]
        _plant(sem_index, [(planted, 0.05)])
        result = _call(mcp_sem, "search_notes", {"queries": ["qry"]})
        assert result["completeness"] == "full"
        assert result["version"] is None
        assert planted in [m["id"] for m in result["results"][0]["matches"]]

    def test_min_query_gate_skips_semantic_but_is_final(self, wrapper, sem_index, mcp_sem):
        planted = _seed(
            wrapper,
            [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "ab gate", "Back": "A"}}],
        )[0]["id"]
        _plant(sem_index, [(planted, 0.05)])
        result = _call(mcp_sem, "search_notes", {"queries": ["ab"]})
        # Final for this query (a client must not poll for more) + advisory.
        assert result["completeness"] == "full"
        assert "skipped" in result["message"]
        # The literal substring still matches (the cheap signals ran).
        assert planted in [m["id"] for m in result["results"][0]["matches"]]

    def test_id_anchors_are_never_gated(self, wrapper, sem_index, mcp_sem):
        a, b = (
            r["id"]
            for r in _seed(
                wrapper,
                [
                    {
                        "deck": "Test",
                        "note_type": "Basic",
                        "fields": {"Front": "anchor", "Back": "A"},
                    },
                    {
                        "deck": "Test",
                        "note_type": "Basic",
                        "fields": {"Front": "neighbor", "Back": "A"},
                    },
                ],
            )
        )
        _plant(sem_index, [(a, 0.30), (b, 0.10)])
        result = _call(mcp_sem, "search_notes", {"ids": [a]})
        assert result["completeness"] == "full"
        assert b in [m["id"] for m in result["results"][0]["matches"]]


class TestDedupNeighbors:
    """The generation-dedup hardening (#204): the lexical-overlap signal
    (#206), the precision-oriented threshold (#207), and per-signal
    provenance (#208) on upsert neighbors."""

    @pytest.fixture()
    def derived(self, tmp_path):
        from shrike.derived import DerivedTextStore

        s = DerivedTextStore(path=tmp_path / "shrike.db")
        yield s
        s.close()

    @pytest.fixture()
    def mcp_dedup(self, wrapper, sem_index, derived):
        mcp = FastMCP("test")
        register_tools(mcp, wrapper, index=sem_index, derived=derived)
        return mcp

    def _neighbors(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        return result["results"][0].get("neighbors", [])

    def test_near_verbatim_dupe_surfaces_lexically(self, wrapper, sem_index, mcp_dedup, derived):
        # An existing near-verbatim card, planted semantically FAR (cosine 0 —
        # under any threshold) so only the trigram overlap can catch it.
        existing = _seed(
            wrapper,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {
                        "Front": "the krebs cycle produces atp in mitochondria",
                        "Back": "x",
                    },
                }
            ],
        )[0]["id"]
        _build_derived(wrapper, derived)
        _plant(sem_index, [(existing, 1.0)])  # distance 1.0 → cosine 0

        result = _upsert(
            mcp_dedup,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {
                        "Front": "the krebs cycle produces atp in mitochondria!",
                        "Back": "y",
                    },
                }
            ],
        )
        neighbors = self._neighbors(result)
        hit = next(n for n in neighbors if n["id"] == existing)
        assert hit["score"] is None, "lexical-only — no cosine to report"
        assert [p["signal"] for p in hit["provenance"]] == ["fuzzy"]

    def test_semantic_neighbor_carries_text_provenance(
        self, wrapper, sem_index, mcp_dedup, derived
    ):
        existing = _seed(
            wrapper,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "photosynthesis overview", "Back": "x"},
                }
            ],
        )[0]["id"]
        _build_derived(wrapper, derived)
        _plant(sem_index, [(existing, 0.1)])  # cosine 0.9 ≥ the 0.6 default

        result = _upsert(
            mcp_dedup,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "zz completely different words zz", "Back": "y"},
                }
            ],
        )
        neighbors = self._neighbors(result)
        hit = next(n for n in neighbors if n["id"] == existing)
        assert hit["score"] == 0.9
        assert {p["signal"] for p in hit["provenance"]} == {"text"}

    def test_both_signals_merge_on_one_candidate(self, wrapper, sem_index, mcp_dedup, derived):
        existing = _seed(
            wrapper,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "glycolysis happens in the cytoplasm", "Back": "x"},
                }
            ],
        )[0]["id"]
        _build_derived(wrapper, derived)
        _plant(sem_index, [(existing, 0.1)])

        result = _upsert(
            mcp_dedup,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "glycolysis happens in the cytoplasm too", "Back": "y"},
                }
            ],
        )
        hit = next(n for n in self._neighbors(result) if n["id"] == existing)
        assert hit["score"] == 0.9
        assert {p["signal"] for p in hit["provenance"]} == {"text", "fuzzy"}

    def test_default_threshold_is_precision_oriented(self, wrapper, sem_index, mcp_dedup, derived):
        # Cosine 0.55: above the old shared 0.5 search default, below the
        # deliberate 0.6 dedup default (#207) — and lexically unrelated, so
        # nothing backstops it. It must NOT surface.
        existing = _seed(
            wrapper,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "qq unrelated wording qq", "Back": "x"},
                }
            ],
        )[0]["id"]
        _build_derived(wrapper, derived)
        _plant(sem_index, [(existing, 0.45)])

        result = _upsert(
            mcp_dedup,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "zz different thing zz", "Back": "y"},
                }
            ],
        )
        # Not a SEMANTIC neighbor at the default gate (an incidental trigram
        # overlap may still surface it lexically — that's the backstop, and
        # it reports score None, never a cosine).
        semantic_ids = [n["id"] for n in self._neighbors(result) if n["score"] is not None]
        assert existing not in semantic_ids
        # An explicit lower threshold opts back in (the knob stayed).
        result = _upsert(
            mcp_dedup,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "zz third thing zz", "Back": "y"},
                }
            ],
            neighbor_threshold=0.5,
        )
        hit = next(n for n in self._neighbors(result) if n["id"] == existing)
        assert hit["score"] == 0.55
