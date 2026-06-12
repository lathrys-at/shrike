"""Tool-layer tests for collection_query (#97). Kernel-harness port (#355)."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.tools import register_tools


@pytest.fixture()
def mcp_app(kharness):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kharness.kernel)
    return mcp


def _add(kharness, front, *, tags=None):
    return kharness.seed_note(front, deck="D", tags=tags)


class TestCollectionQueryTool:
    def test_returns_matches(self, kharness, mcp_app):
        _add(kharness, "hello", tags=["q"])
        result = kharness.call_tool(mcp_app, "collection_query", {"query": "tag:q"})
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "hello"

    def test_meta_drops_content(self, kharness, mcp_app):
        _add(kharness, "hi", tags=["q"])
        result = kharness.call_tool(
            mcp_app, "collection_query", {"query": "tag:q", "fields": "meta"}
        )
        assert result["notes"][0]["content"] is None

    def test_bad_query_is_tool_error_without_isolation_marks(self, kharness, mcp_app):
        with pytest.raises(ToolError) as exc:
            kharness.call_tool(mcp_app, "collection_query", {"query": "(unbalanced"})
        assert "⁨" not in str(exc.value)
        assert "⁩" not in str(exc.value)

    def test_limit_out_of_range_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(mcp_app, "collection_query", {"query": "deck:*", "limit": 999})

    def test_empty_query_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(mcp_app, "collection_query", {"query": ""})
