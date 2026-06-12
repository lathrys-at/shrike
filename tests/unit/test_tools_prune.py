"""Tool-layer tests for collection_prune (#89): index maintenance + defaults.

Empty notes/cards delete notes, so their vectors must leave the index (the
kernel's forget_notes — like delete_notes); clearing unused tags is a
col_mod-only metadata change (kernel.metadata_changed). Dry-run (the default)
must touch nothing. Kernel-harness port (#355): assertions read the shared
engine + the spied kernel ops instead of facade mocks.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.tools import register_tools
from tests.unit.conftest import EmbedRecorder


@pytest.fixture()
def backend():
    return EmbedRecorder()


@pytest.fixture()
def kproxy(kharness, backend):
    kharness.attach_embedder(backend)
    proxy = kharness.proxy()
    proxy.spy("metadata_changed")
    proxy.spy("forget_notes")
    return proxy


@pytest.fixture()
def mcp_app(kharness, kproxy):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kproxy)
    return mcp


def _blank_note(kharness) -> int:
    """A note whose every field is blank (created with text, then cleared via
    a raw core edit — an external-style change the kernel didn't index)."""
    nid = kharness.seed_note("tmp", deck="D")

    def _clear(c):
        _, _, fields, _ = c.get_note(nid)
        c.update_note(nid, ["" for _ in fields])

    kharness.run(kharness.wrapper.run(_clear))
    return nid


def _orphan_tag(kharness) -> None:
    nid = kharness.seed_note("Q", deck="D", back="A", tags=["orphan"])
    kharness.run(kharness.wrapper.run(lambda c: c.update_note_tags([nid], remove=["orphan"])))


class TestCollectionPruneTool:
    def test_no_flags_runs_all_in_dry_run(self, kharness, kproxy, mcp_app):
        result = kharness.call_tool(mcp_app, "collection_prune", {})
        assert result["dry_run"] is True
        # All three sections present (all cleanups ran).
        assert result["unused_tags"] is not None
        assert result["empty_notes"] is not None
        assert result["empty_cards"] is not None
        # Dry-run touches nothing.
        assert kproxy.calls["forget_notes"] == 0
        assert kproxy.calls["metadata_changed"] == 0

    def test_apply_empty_notes_removes_from_index(self, kharness, kproxy, mcp_app):
        blank = _blank_note(kharness)
        assert kharness.engine.contains(blank)  # indexed when it still had text
        result = kharness.call_tool(
            mcp_app, "collection_prune", {"empty_notes": True, "dry_run": False}
        )
        assert result["empty_notes"]["removed"] == [blank]
        assert kproxy.calls["forget_notes"] == 1
        assert not kharness.engine.contains(blank)  # vectors left with the note
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_apply_unused_tags_bumps_without_index_remove(self, kharness, kproxy, mcp_app):
        _orphan_tag(kharness)
        result = kharness.call_tool(
            mcp_app, "collection_prune", {"unused_tags": True, "dry_run": False}
        )
        assert result["unused_tags"]["removed"] >= 1
        assert kproxy.calls["forget_notes"] == 0  # no notes deleted
        assert kproxy.calls["metadata_changed"] == 1  # but the watermark advanced
        assert kharness.index_status()["col_mod"] == kharness.col_mod()

    def test_dry_run_default_does_not_mutate(self, kharness, kproxy, mcp_app):
        blank = _blank_note(kharness)
        result = kharness.call_tool(mcp_app, "collection_prune", {"empty_notes": True})
        assert result["dry_run"] is True
        assert result["empty_notes"]["removed"] == [blank]
        # Still there; index untouched.
        found = kharness.run(kharness.wrapper.run(lambda c: c.find_notes(f"nid:{blank}")))
        assert found
        assert kharness.engine.contains(blank)
        assert kproxy.calls["forget_notes"] == 0

    def test_unrequested_cleanup_absent_from_response(self, kharness, mcp_app):
        result = kharness.call_tool(mcp_app, "collection_prune", {"empty_notes": True})
        assert result["empty_notes"] is not None
        assert result["unused_tags"] is None
        assert result["empty_cards"] is None
