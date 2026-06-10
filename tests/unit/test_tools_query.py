"""Tool-layer tests for collection_query (#97)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.tools import register_tools
from tests.unit.conftest import make_notes


def _call(mcp: FastMCP, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    _, structured = asyncio.run(mcp.call_tool(name, args or {}))
    return structured


@pytest.fixture()
def mcp_app(wrapper):
    mcp = FastMCP("test")
    register_tools(mcp, wrapper)
    return mcp


def _add(wrapper, front, *, tags=None):
    results = make_notes(
        wrapper,
        [{"note_type": "Basic", "deck": "D", "fields": {"Front": front, "Back": "x"},
          "tags": list(tags or [])}],
    )
    return results[0]["id"]


class TestCollectionQueryTool:
    def test_returns_matches(self, wrapper, mcp_app):
        _add(wrapper, "hello", tags=["q"])
        result = _call(mcp_app, "collection_query", {"query": "tag:q"})
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "hello"

    def test_meta_drops_content(self, wrapper, mcp_app):
        _add(wrapper, "hi", tags=["q"])
        result = _call(mcp_app, "collection_query", {"query": "tag:q", "fields": "meta"})
        assert result["notes"][0]["content"] is None

    def test_bad_query_is_tool_error_without_isolation_marks(self, wrapper, mcp_app):
        with pytest.raises(ToolError) as exc:
            _call(mcp_app, "collection_query", {"query": "(unbalanced"})
        assert "⁨" not in str(exc.value)
        assert "⁩" not in str(exc.value)

    def test_limit_out_of_range_rejected(self, mcp_app):
        with pytest.raises(ToolError):
            _call(mcp_app, "collection_query", {"query": "deck:*", "limit": 999})

    def test_empty_query_rejected(self, mcp_app):
        with pytest.raises(ToolError):
            _call(mcp_app, "collection_query", {"query": ""})
