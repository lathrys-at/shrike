"""Tool-layer tests for deck ops (#74): col_mod bump without vector changes.

Deck create/rename/delete-empty never change a note's embedding text, so they
must advance the stored index col_mod (avoiding a spurious rebuild) WITHOUT
touching vectors.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

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


class TestUpsertDecksBump:
    def test_create_bumps_col_mod_without_vectors(self, wrapper, mock_index, mock_saver, mcp_app):
        result = _call(mcp_app, "upsert_decks", {"decks": [{"name": "New"}]})
        assert result["results"][0]["status"] == "created"
        mock_index.add.assert_not_called()
        mock_index.remove.assert_not_called()
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()

    def test_all_error_does_not_bump(self, wrapper, mock_index, mock_saver, mcp_app):
        result = _call(mcp_app, "upsert_decks", {"decks": [{"id": 9999999999999, "name": "X"}]})
        assert result["results"][0]["status"] == "error"
        assert mock_index.col_mod == 0
        mock_saver.request_save.assert_not_called()

    def test_empty_list_rejected(self, mcp_app):
        with pytest.raises(ToolError):
            _call(mcp_app, "upsert_decks", {"decks": []})


class TestDeleteDecksTool:
    def test_delete_empty_bumps(self, wrapper, mock_index, mock_saver, mcp_app):
        _call(mcp_app, "upsert_decks", {"decks": [{"name": "Temp"}]})
        mock_saver.reset_mock()
        mock_index.col_mod = 0

        result = _call(mcp_app, "delete_decks", {"decks": ["Temp"]})
        assert result["deleted"] == ["Temp"]
        mock_index.remove.assert_not_called()
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()

    def test_non_empty_reported_and_no_bump(self, wrapper, mock_index, mock_saver, mcp_app):
        _call(
            mcp_app,
            "upsert_notes",
            {
                "notes": [
                    {"deck": "Full", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}
                ]
            },
        )
        mock_saver.reset_mock()
        mock_index.col_mod = 0

        result = _call(mcp_app, "delete_decks", {"decks": ["Full"]})
        assert result["not_empty"] == ["Full"]
        assert result["deleted"] == []
        assert mock_index.col_mod == 0
        mock_saver.request_save.assert_not_called()
