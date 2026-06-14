"""Tool-layer list_notes input validation (#599).

A malformed ``modified_since`` is caller-supplied bad input: it must surface as
a clean ToolInputError (WARNING, no traceback), not the catch-all "Unhandled
error" + traceback (which also leaks fromisoformat's parse detail).
"""

from __future__ import annotations

import logging

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.tools import register_tools


@pytest.fixture()
def mcp_app(kharness):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kharness.kernel)
    return mcp


class TestModifiedSinceValidation:
    def test_malformed_modified_since_is_clean_input_error(self, kharness, mcp_app, caplog):
        kharness.seed_note("hello", deck="D")
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
            with pytest.raises(ToolError):
                kharness.call_tool(
                    mcp_app, "list_notes", {"deck": "D", "modified_since": "not-a-date"}
                )
        unhandled = [r for r in caplog.records if "Unhandled error" in r.getMessage()]
        assert not unhandled, (
            "malformed modified_since logged as an unhandled server bug (with traceback): "
            f"{[r.getMessage() for r in unhandled]}"
        )

    def test_valid_modified_since_is_accepted(self, kharness, mcp_app):
        # The guard rejects only bad input — a well-formed ISO 8601 value works.
        kharness.seed_note("hello", deck="D")
        result = kharness.call_tool(
            mcp_app, "list_notes", {"deck": "D", "modified_since": "2000-01-01T00:00:00"}
        )
        assert result["total"] >= 1
