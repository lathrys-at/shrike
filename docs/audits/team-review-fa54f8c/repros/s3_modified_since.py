"""S3-1 repro (preserved by lead; rev-S3 worktree reaped).
Malformed `modified_since` surfaces as a server bug (ERROR + traceback) instead
of a clean ToolInputError. Asserts CORRECT behavior → RED at fa54f8c.
Place in tests/unit/ (needs kharness fixture from tests/unit/conftest.py).
Run: SHRIKE_SKIP_NATIVE_STALE_CHECK=1 .venv/bin/python -m pytest tests/unit/test_scratch_s3a.py -q -p no:cacheprovider
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


def test_malformed_modified_since_is_clean_input_error(kharness, mcp_app, caplog):
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
