"""Tool-layer find_replace_notes (#85): validation + re-embed of changed notes.

Kernel-harness port (#355): an applied replace routes the changed set through
kernel.reindex_notes (re-embed + re-ingest); the re-embed is observable as a
fresh embed call carrying the edited text.
"""

from __future__ import annotations

import logging

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.api.tools import register_tools
from tests.unit.conftest import EmbedRecorder


@pytest.fixture()
def backend():
    return EmbedRecorder()


@pytest.fixture()
def kproxy(kharness, backend):
    kharness.attach_embedder(backend)
    proxy = kharness.proxy()
    proxy.spy("reindex_notes")
    return proxy


@pytest.fixture()
def mcp_app(kharness, kproxy):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kproxy)
    return mcp


class TestValidation:
    def test_requires_scope(self, kharness, mcp_app):
        with pytest.raises(ToolError, match="scope"):
            kharness.call_tool(mcp_app, "find_replace_notes", {"search": "a", "replace": "b"})

    def test_empty_search_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app, "find_replace_notes", {"search": "", "replace": "b", "deck": "Bio"}
            )

    def test_invalid_regex_is_clean_input_error(self, kharness, mcp_app, caplog):
        # #599: an invalid regex/backref is caller-supplied bad input — a clean
        # ToolInputError (WARNING, no traceback), not the catch-all "Unhandled
        # error" + traceback. The preview loop compiles the pattern on EVERY
        # call, so this bites a real apply, not only dry_run.
        kharness.seed_note("hello", deck="Bio")
        with (
            caplog.at_level(logging.DEBUG, logger="shrike.tools"),
            pytest.raises(ToolError),
        ):
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


class TestReembed:
    def test_apply_reembeds_changed_notes(self, kharness, backend, kproxy, mcp_app):
        kharness.seed_note("teh cell", deck="Bio")
        kharness.seed_note("no match here", deck="Bio")
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app, "find_replace_notes", {"search": "teh", "replace": "the", "deck": "Bio"}
        )
        assert result["notes_changed"] == 1
        assert kproxy.calls["reindex_notes"] == 1
        # Exactly the changed note re-embedded, with its edited text.
        new_calls = backend.calls[embeds_before:]
        assert len(new_calls) == 1
        assert len(new_calls[0]) == 1
        assert "the cell" in new_calls[0][0]
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_dry_run_does_not_touch_index(self, kharness, backend, kproxy, mcp_app):
        kharness.seed_note("teh cell", deck="Bio")
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app,
            "find_replace_notes",
            {"search": "teh", "replace": "the", "deck": "Bio", "dry_run": True},
        )
        assert result["dry_run"] is True
        assert result["notes_changed"] == 1
        assert kproxy.calls["reindex_notes"] == 0
        assert len(backend.calls) == embeds_before

    def test_no_match_no_reembed(self, kharness, backend, kproxy, mcp_app):
        kharness.seed_note("nothing", deck="Bio")
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app, "find_replace_notes", {"search": "zzz", "replace": "x", "deck": "Bio"}
        )
        assert result["notes_changed"] == 0
        assert kproxy.calls["reindex_notes"] == 0
        assert len(backend.calls) == embeds_before
