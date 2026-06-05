"""Tool-layer surfacing of CollectionBusyError (#65)."""

from __future__ import annotations

import asyncio

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.collection import CollectionBusyError
from shrike.schemas import COLLECTION_BUSY_CODE
from shrike.tools import register_tools


def test_busy_surfaces_as_tool_error_with_code(wrapper, monkeypatch):
    mcp = FastMCP("test")
    register_tools(mcp, wrapper)

    async def boom(_fn):
        raise CollectionBusyError()

    # Every wrapper op funnels through .run; make it report busy.
    monkeypatch.setattr(wrapper, "run", boom)

    with pytest.raises(ToolError) as exc:
        asyncio.run(mcp.call_tool("collection_info", {}))
    # The coded message reaches the client unchanged.
    assert COLLECTION_BUSY_CODE in str(exc.value)
