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
    """Seed notes synchronously via the wrapper's worker thread."""
    return wrapper.run_sync(lambda _c: wrapper._upsert_notes(notes))


def _call(mcp: FastMCP, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    _, structured = asyncio.run(mcp.call_tool(name, args or {}))
    return structured


def _upsert(mcp: FastMCP, notes: list[dict], **extra: Any) -> dict[str, Any]:
    return _call(mcp, "upsert_notes", {"notes": notes, **extra})


@pytest.fixture()
def mock_index():
    idx = MagicMock(spec=VectorIndex)
    idx.state = IndexState.READY
    idx.available = True
    idx.build_progress = (0, 0)
    idx.search = MagicMock(return_value=[])
    idx.col_mod = 0
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
    def test_unavailable_returns_message(self, mcp_app, mock_index):
        mock_index.state = IndexState.UNAVAILABLE
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        assert result["results"] == []
        assert "not available" in result["_message"]

    def test_building_returns_progress(self, mcp_app, mock_index):
        mock_index.state = IndexState.BUILDING
        mock_index.build_progress = (50, 100)
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        assert "50/100" in result["_message"]

    def test_error_returns_message(self, mcp_app, mock_index):
        mock_index.state = IndexState.ERROR
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        assert "error" in result["_message"]

    def test_no_index_returns_message(self, mcp_no_index):
        result = _call(mcp_no_index, "search_notes", {"queries": ["test"]})
        assert result["results"] == []
        assert "not available" in result["_message"]

    def test_requires_queries_or_ids(self, mcp_app):
        result = _call(mcp_app, "search_notes", {})
        assert "error" in result


class TestSearchNotesResults:
    def test_text_query(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search.return_value = [[{"note_id": basic_note, "distance": 0.1}]]
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
        mock_index.search.return_value = [[{"note_id": other, "distance": 0.2}]]
        result = _call(mcp_app, "search_notes", {"ids": [basic_note]})
        assert len(result["results"]) == 1
        assert result["results"][0]["source"] == f"note #{basic_note}"

    def test_exclude_ids(self, wrapper, mock_index, mcp_app, basic_note):
        other = _seed(
            wrapper, [{"deck": "Test", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]
        )[0]["id"]
        mock_index.search.return_value = [
            [
                {"note_id": basic_note, "distance": 0.05},
                {"note_id": other, "distance": 0.2},
            ]
        ]
        result = _call(mcp_app, "search_notes", {"queries": ["test"], "exclude_ids": [basic_note]})
        matches = result["results"][0]["matches"]
        assert all(m["id"] != basic_note for m in matches)

    def test_deck_filter(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search.return_value = [[{"note_id": basic_note, "distance": 0.1}]]
        result = _call(mcp_app, "search_notes", {"queries": ["test"], "deck": "Nonexistent"})
        assert result["results"][0]["matches"] == []

    def test_tags_filter(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search.return_value = [[{"note_id": basic_note, "distance": 0.1}]]
        result = _call(mcp_app, "search_notes", {"queries": ["test"], "tags": ["nonexistent-tag"]})
        assert result["results"][0]["matches"] == []

    def test_result_includes_content(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search.return_value = [[{"note_id": basic_note, "distance": 0.1}]]
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        match = result["results"][0]["matches"][0]
        assert "content" in match
        assert match["content"]["Front"] == "What is 2+2?"

    def test_top_k_clamped(self, mcp_app, mock_index):
        mock_index.search.return_value = [[]]
        _call(mcp_app, "search_notes", {"queries": ["test"], "top_k": 0})
        args = mock_index.search.call_args
        assert args[1]["top_k"] >= 1

    def test_score_rounded_to_3_decimals(self, wrapper, mock_index, mcp_app, basic_note):
        mock_index.search.return_value = [[{"note_id": basic_note, "distance": 0.12345}]]
        result = _call(mcp_app, "search_notes", {"queries": ["test"]})
        score = result["results"][0]["matches"][0]["score"]
        assert score == round(1.0 - 0.12345, 3)


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
        assert "neighbors" not in result["results"][0]

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
        assert "neighbors" not in r
        assert r["neighbors_unavailable"] is True
        assert "_message" in result
        assert f"search_notes(ids=[{nid}])" in result["_message"]

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
        call_ids = mock_index.add.call_args[0][0]
        assert nid in call_ids

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
        assert f"search_notes(ids=[{r['id']}])" in result["_message"]

    def test_no_retry_hint_on_success(self, wrapper, mock_index, mcp_app):
        """Successful neighbor computation carries no retry flag or message."""
        mock_index.search.return_value = [[]]
        result = _upsert(mcp_app, [BASIC_NOTE])
        r = result["results"][0]
        assert "neighbors" in r
        assert "neighbors_unavailable" not in r
        assert "_message" not in result
