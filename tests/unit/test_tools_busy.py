"""Tool-layer surfacing of CollectionBusyError (#65). Kernel-harness port (#355)."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.harness.collection import CollectionBusyError
from shrike.schemas import COLLECTION_BUSY_CODE
from shrike.api.tools import register_tools


def test_busy_surfaces_as_tool_error_with_code(kharness, monkeypatch):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kharness.kernel)

    async def boom(_fn):
        raise CollectionBusyError()

    # Every wrapper op funnels through .run; make it report busy.
    monkeypatch.setattr(kharness.wrapper, "run", boom)

    with pytest.raises(ToolError) as exc:
        kharness.call_tool(mcp, "collection_info", {})
    # The coded message reaches the client unchanged.
    assert COLLECTION_BUSY_CODE in str(exc.value)
