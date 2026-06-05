"""Tool-layer tests for find_replace_note_types (#76, findAndReplaceInModels).

Editing a note type's template HTML / CSS never changes any note's embedding
text, so a successful replace must advance the stored index col_mod (avoiding a
spurious rebuild) WITHOUT touching vectors — the same metadata-bump contract as
the tag/deck ops. A no-op replace (no matches) bumps nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.index import IndexSaver, IndexState, VectorIndex
from shrike.note_types import upsert_note_types
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


@pytest.fixture()
def model(wrapper):
    return wrapper.run_sync(
        lambda c: upsert_note_types(
            c,
            [
                {
                    "name": "FR",
                    "fields": ["F"],
                    "templates": [{"name": "C", "front": "old {{F}}", "back": "{{F}}"}],
                    "css": ".card { color: red; }",
                }
            ],
        )[0]["id"]
    )


class TestFindReplaceNoteTypesTool:
    def test_replace_bumps_col_mod_without_vectors(
        self, wrapper, model, mock_index, mock_saver, mcp_app
    ):
        result = _call(
            mcp_app,
            "find_replace_note_types",
            {"note_type": "FR", "search": "red", "replace": "blue"},
        )
        assert result["replacements"] == 1
        assert result["css_changed"] is True
        # Templates/CSS are not embedding text — no vectors touched.
        mock_index.add.assert_not_called()
        mock_index.remove.assert_not_called()
        assert mock_index.col_mod == wrapper.col.mod
        mock_saver.request_save.assert_called_once()

    def test_no_match_does_not_bump(self, wrapper, model, mock_index, mock_saver, mcp_app):
        result = _call(
            mcp_app,
            "find_replace_note_types",
            {"note_type": "FR", "search": "absent", "replace": "x"},
        )
        assert result["replacements"] == 0
        assert mock_index.col_mod == 0
        mock_saver.request_save.assert_not_called()

    def test_unknown_note_type_is_tool_error(self, model, mcp_app):
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "find_replace_note_types",
                {"note_type": "Nope", "search": "a", "replace": "b"},
            )

    def test_no_location_selected_is_tool_error(self, model, mcp_app):
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "find_replace_note_types",
                {
                    "note_type": "FR",
                    "search": "old",
                    "replace": "x",
                    "front": False,
                    "back": False,
                    "css": False,
                },
            )

    def test_empty_search_rejected(self, model, mcp_app):
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "find_replace_note_types",
                {"note_type": "FR", "search": "", "replace": "x"},
            )

    def test_invalid_regex_is_tool_error(self, model, mcp_app):
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "find_replace_note_types",
                {"note_type": "FR", "search": "(unclosed", "replace": "x", "regex": True},
            )
