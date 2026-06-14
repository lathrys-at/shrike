"""S3-2 repro (preserved by lead; rev-S3 worktree reaped).
Invalid regex/backref in find_replace_notes surfaces as a server bug (ERROR +
traceback); the dry-run preview loop runs on EVERY call incl dry_run=False.
Asserts CORRECT behavior → RED at fa54f8c.
Place in tests/unit/ (needs kharness fixture).
Run: SHRIKE_SKIP_NATIVE_STALE_CHECK=1 .venv/bin/python -m pytest tests/unit/test_scratch_s3b.py -q -p no:cacheprovider
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


def test_invalid_regex_is_clean_input_error(kharness, mcp_app, caplog):
    kharness.seed_note("hello", deck="Bio")
    with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "find_replace_notes",
                {"search": "(unbalanced", "replace": "x", "deck": "Bio", "regex": True},
            )
    unhandled = [r for r in caplog.records if "Unhandled error" in r.getMessage()]
    assert not unhandled, (
        "invalid regex logged as an unhandled server bug (with traceback): "
        f"{[r.getMessage() for r in unhandled]}"
    )
