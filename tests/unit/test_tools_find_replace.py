"""Tool-layer find_replace_notes (#85): validation + re-embed of changed notes."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.index import IndexSaver, IndexState, VectorIndex
from shrike.tools import register_tools
from tests.unit.conftest import make_notes


def _call(mcp: FastMCP, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    _, structured = asyncio.run(mcp.call_tool(name, args or {}))
    return structured


def _seed(wrapper, deck: str, front: str) -> int:
    note = {"deck": deck, "note_type": "Basic", "fields": {"Front": front, "Back": "x"}}
    return make_notes(wrapper, [note])[0]["id"]


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


class TestValidation:
    def test_requires_scope(self, mcp_app):
        with pytest.raises(ToolError, match="scope"):
            _call(mcp_app, "find_replace_notes", {"search": "a", "replace": "b"})

    def test_empty_search_rejected(self, mcp_app):
        with pytest.raises(ToolError):
            _call(mcp_app, "find_replace_notes", {"search": "", "replace": "b", "deck": "Bio"})


class TestReembed:
    def test_apply_reembeds_changed_notes(self, wrapper, mock_index, mock_saver, mcp_app):
        nid = _seed(wrapper, "Bio", "teh cell")
        _seed(wrapper, "Bio", "no match here")
        args = {"search": "teh", "replace": "the", "deck": "Bio"}
        result = _call(mcp_app, "find_replace_notes", args)
        assert result["notes_changed"] == 1
        mock_index.add.assert_called_once()
        assert [i.note_id for i in mock_index.add.call_args[0][0]] == [nid]  # only the changed note
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()

    def test_dry_run_does_not_touch_index(self, wrapper, mock_index, mock_saver, mcp_app):
        _seed(wrapper, "Bio", "teh cell")
        result = _call(
            mcp_app,
            "find_replace_notes",
            {"search": "teh", "replace": "the", "deck": "Bio", "dry_run": True},
        )
        assert result["dry_run"] is True
        assert result["notes_changed"] == 1
        mock_index.add.assert_not_called()
        mock_saver.request_save.assert_not_called()

    def test_no_match_no_reembed(self, wrapper, mock_index, mock_saver, mcp_app):
        _seed(wrapper, "Bio", "nothing")
        args = {"search": "zzz", "replace": "x", "deck": "Bio"}
        result = _call(mcp_app, "find_replace_notes", args)
        assert result["notes_changed"] == 0
        mock_index.add.assert_not_called()
