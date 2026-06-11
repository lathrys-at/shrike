"""Tool-layer tests for update_note_type_field_metadata (#119): col_mod bump, no re-embed."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.index import IndexSaver, IndexState, VectorIndex
from shrike.tools import register_tools
from tests.unit._native_shims import upsert_note_types


def _call(mcp: FastMCP, name: str, args: dict[str, Any]) -> dict[str, Any]:
    _, structured = asyncio.run(mcp.call_tool(name, args))
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
    wrapper.run_sync(
        lambda c: upsert_note_types(
            c,
            [
                {
                    "name": "M",
                    "fields": ["F"],
                    "templates": [{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
                    "css": "",
                }
            ],
        )
    )


class TestSetFieldMetadataTool:
    def test_bumps_col_mod_without_reembed(self, wrapper, model, mock_index, mock_saver, mcp_app):
        result = _call(
            mcp_app,
            "update_note_type_field_metadata",
            {"note_type": "M", "fields": [{"name": "F", "size": 28, "description": "prompt"}]},
        )
        assert result["fields_updated"] == ["F"]
        # Editor metadata isn't embedding text: no re-embed, but col_mod advances.
        mock_index.add.assert_not_called()
        assert mock_index.col_mod == wrapper.run_sync(lambda c: c.col_mod())
        mock_saver.request_save.assert_called_once()

    def test_unknown_field_is_tool_error(self, wrapper, model, mcp_app):
        with pytest.raises(ToolError):
            _call(
                mcp_app,
                "update_note_type_field_metadata",
                {"note_type": "M", "fields": [{"name": "Nope", "size": 1}]},
            )

    def test_empty_fields_rejected(self, wrapper, model, mcp_app):
        with pytest.raises(ToolError):
            _call(mcp_app, "update_note_type_field_metadata", {"note_type": "M", "fields": []})
