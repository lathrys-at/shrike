"""Tool-layer tests for migrate_note_type (#75): index re-embed + validation.

Remapped fields change embedding text, so an applied migration must re-embed the
changed notes (their ids are unchanged) — like find_replace_notes. Dry-run and
validation failures must not touch the index.
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


def _add_basic(wrapper, front, back="x"):
    def build(c):
        n = c.new_note(c.models.by_name("Basic"))
        n["Front"], n["Back"] = front, back
        c.add_note(n, c.decks.id("D"))
        return n.id

    return wrapper.run_sync(build)


class TestMigrateNoteTypeTool:
    def test_apply_reembeds_changed_notes(self, wrapper, mock_index, mock_saver, mcp_app):
        nid = _add_basic(wrapper, "hello")
        result = _call(
            mcp_app,
            "migrate_note_type",
            {
                "note_ids": [nid],
                "new_note_type": "Cloze",
                "field_map": {"Front": "Text", "Back": "Back Extra"},
                "dry_run": False,
            },
        )
        assert result["changed"] == [nid]
        mock_index.add.assert_called_once()
        assert [i.note_id for i in mock_index.add.call_args.args[0]] == [nid]
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()

    def test_dry_run_does_not_touch_index(self, wrapper, mock_index, mock_saver, mcp_app):
        nid = _add_basic(wrapper, "hi")
        result = _call(
            mcp_app,
            "migrate_note_type",
            {
                "note_ids": [nid],
                "new_note_type": "Cloze",
                "field_map": {"Front": "Text"},
                "dry_run": True,
            },
        )
        assert result["dry_run"] is True
        mock_index.add.assert_not_called()
        mock_saver.request_save.assert_not_called()

    def test_bad_field_map_is_tool_error(self, wrapper, mcp_app):
        nid = _add_basic(wrapper, "x")
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "migrate_note_type",
                {"note_ids": [nid], "new_note_type": "Cloze", "field_map": {"Nope": "Text"}},
            )

    def test_empty_field_map_rejected(self, wrapper, mcp_app):
        nid = _add_basic(wrapper, "x")
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "migrate_note_type",
                {"note_ids": [nid], "new_note_type": "Cloze", "field_map": {}},
            )

    def test_empty_note_ids_rejected(self, mcp_app):
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "migrate_note_type",
                {"note_ids": [], "new_note_type": "Cloze", "field_map": {"Front": "Text"}},
            )
