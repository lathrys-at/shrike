"""Tool-layer tests for rename_tag: input validation + index col_mod bump.

A tag change never alters a note's embedding vector (tags aren't part of the
embedding text), but it does bump col.mod. The kernel op itself advances the
stored watermarks WITHOUT re-embedding, so the next startup doesn't trigger a
needless full rebuild. "Vectors untouched" is asserted as "no new embed call",
and "watermark advanced" as "index col_mod matches the collection + no drift" —
observable state, not host-side spies.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.api.tools import register_tools
from tests.unit.conftest import EmbedRecorder


@pytest.fixture()
def backend():
    return EmbedRecorder()


@pytest.fixture()
def mcp_app(kharness, backend):
    kharness.attach_embedder(backend)
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kharness.kernel)
    return mcp


class TestRenameTagTool:
    def test_identical_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError, match="identical"):
            kharness.call_tool(mcp_app, "rename_tag", {"old": "a", "new": "a"})

    def test_rename_bumps_col_mod(self, kharness, backend, mcp_app, kbasic_note):
        # kbasic_note has tag "math"; collection-wide rename.
        embeds_before = len(backend.calls)
        result = kharness.call_tool(mcp_app, "rename_tag", {"old": "math", "new": "arithmetic"})
        assert result["notes_modified"] == 1
        assert len(backend.calls) == embeds_before
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
