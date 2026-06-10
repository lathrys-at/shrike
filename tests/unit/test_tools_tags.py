"""Tool-layer tests for tag ops (#73): input validation + index col_mod bump.

A tag change never alters a note's embedding vector (tags aren't part of the
embedding text), but it does bump col.mod. The tools must advance the stored
index col_mod (and request a save) WITHOUT touching vectors, so the next startup
doesn't trigger a needless full rebuild.
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


class TestUpdateNoteTagsValidation:
    def test_set_and_add_rejected(self, mcp_app, basic_note):
        with pytest.raises(ToolError, match="not both"):
            _call(
                mcp_app, "update_note_tags", {"note_ids": [basic_note], "set": ["a"], "add": ["b"]}
            )

    def test_no_mode_rejected(self, mcp_app, basic_note):
        with pytest.raises(ToolError, match="Specify"):
            _call(mcp_app, "update_note_tags", {"note_ids": [basic_note]})


class TestUpdateNoteTagsBump:
    def test_set_modifies_and_bumps_col_mod(
        self, wrapper, mock_index, mock_saver, mcp_app, basic_note
    ):
        result = _call(mcp_app, "update_note_tags", {"note_ids": [basic_note], "set": ["x"]})
        assert result["notes_modified"] == 1
        # Vectors untouched; only col_mod advanced + save requested.
        mock_index.add.assert_not_called()
        mock_index.remove.assert_not_called()
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()

    def test_add_remove_bumps_col_mod(self, wrapper, mock_index, mock_saver, mcp_app, basic_note):
        _call(mcp_app, "update_note_tags", {"note_ids": [basic_note], "add": ["new"]})
        mock_index.add.assert_not_called()
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()

    def test_no_match_does_not_bump(self, wrapper, mock_index, mock_saver, mcp_app):
        result = _call(mcp_app, "update_note_tags", {"note_ids": [9999999999999], "add": ["x"]})
        assert result["notes_modified"] == 0
        assert mock_index.col_mod == 0
        mock_saver.request_save.assert_not_called()


class TestRenameTagTool:
    def test_identical_rejected(self, mcp_app):
        with pytest.raises(ToolError, match="identical"):
            _call(mcp_app, "rename_tag", {"old": "a", "new": "a"})

    def test_rename_bumps_col_mod(self, wrapper, mock_index, mock_saver, mcp_app, basic_note):
        # basic_note has tag "math"; collection-wide rename.
        result = _call(mcp_app, "rename_tag", {"old": "math", "new": "arithmetic"})
        assert result["notes_modified"] == 1
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()
