"""Tool-layer tests for collection_prune (#89): index maintenance + defaults.

Empty notes/cards delete notes, so their vectors must leave the index (like
delete_notes); clearing unused tags is a col_mod-only metadata change. Dry-run
(the default) must touch nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.index import IndexSaver, IndexState, VectorIndex
from shrike.tools import register_tools


def _call(mcp: FastMCP, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    _, structured = asyncio.run(mcp.call_tool(name, args or {}))
    return structured


@pytest.fixture()
def mock_index():
    idx = MagicMock(spec=VectorIndex)
    idx.state = IndexState.READY
    idx.available = True
    idx.col_mod = 0
    return idx


@pytest.fixture()
def mock_saver():
    return MagicMock(spec=IndexSaver)


@pytest.fixture()
def mcp_app(wrapper, mock_index, mock_saver):
    mcp = FastMCP("test")
    register_tools(mcp, wrapper, index=mock_index, saver=mock_saver)
    return mcp


def _blank_note(wrapper):
    nid = wrapper.run_sync(lambda c: _add(c))
    wrapper.run_sync(lambda c: _clear(c, nid))
    return nid


def _add(c):
    n = c.new_note(c.models.by_name("Basic"))
    n["Front"], n["Back"] = "tmp", "x"
    c.add_note(n, c.decks.id("D"))
    return n.id


def _clear(c, nid):
    n = c.get_note(nid)
    for f in list(n.keys()):
        n[f] = ""
    c.update_note(n)


def _orphan_tag(wrapper):
    nid = wrapper.run_sync(lambda c: _add_tagged(c))
    wrapper.run_sync(lambda c: c.tags.bulk_remove([nid], "orphan"))


def _add_tagged(c):
    n = c.new_note(c.models.by_name("Basic"))
    n["Front"], n["Back"] = "Q", "A"
    n.tags = ["orphan"]
    c.add_note(n, c.decks.id("D"))
    return n.id


class TestCollectionPruneTool:
    def test_no_flags_runs_all_in_dry_run(self, wrapper, mock_index, mock_saver, mcp_app):
        result = _call(mcp_app, "collection_prune", {})
        assert result["dry_run"] is True
        # All three sections present (all cleanups ran).
        assert result["unused_tags"] is not None
        assert result["empty_notes"] is not None
        assert result["empty_cards"] is not None
        # Dry-run touches nothing.
        mock_index.remove.assert_not_called()
        assert mock_index.col_mod == 0
        mock_saver.request_save.assert_not_called()

    def test_apply_empty_notes_removes_from_index(self, wrapper, mock_index, mock_saver, mcp_app):
        blank = _blank_note(wrapper)
        result = _call(mcp_app, "collection_prune", {"empty_notes": True, "dry_run": False})
        assert result["empty_notes"]["removed"] == [blank]
        mock_index.remove.assert_called_once_with([blank])
        assert mock_index.col_mod == wrapper.col.mod
        mock_saver.request_save.assert_called_once()

    def test_apply_unused_tags_bumps_without_index_remove(
        self, wrapper, mock_index, mock_saver, mcp_app
    ):
        _orphan_tag(wrapper)
        result = _call(mcp_app, "collection_prune", {"unused_tags": True, "dry_run": False})
        assert result["unused_tags"]["removed"] >= 1
        mock_index.remove.assert_not_called()  # no notes deleted
        assert mock_index.col_mod == wrapper.col.mod  # but col_mod advanced
        mock_saver.request_save.assert_called_once()

    def test_dry_run_default_does_not_mutate(self, wrapper, mock_index, mock_saver, mcp_app):
        blank = _blank_note(wrapper)
        result = _call(mcp_app, "collection_prune", {"empty_notes": True})
        assert result["dry_run"] is True
        assert result["empty_notes"]["removed"] == [blank]
        # Still there; index untouched.
        assert wrapper.run_sync(lambda c: c.find_notes(f"nid:{blank}"))
        mock_index.remove.assert_not_called()
        mock_saver.request_save.assert_not_called()

    def test_unrequested_cleanup_absent_from_response(self, wrapper, mcp_app):
        result = _call(mcp_app, "collection_prune", {"empty_notes": True})
        assert result["empty_notes"] is not None
        assert result["unused_tags"] is None
        assert result["empty_cards"] is None
