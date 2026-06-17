"""Tool-layer tests for collection_prune (#89): index maintenance + defaults.

Empty notes/cards delete notes, so their vectors must leave the index;
clearing unused tags is a col_mod-only metadata change. Since the #391
re-home the whole op — cleanups AND that maintenance tail — runs inside the
kernel's collection_prune, so the assertions read observable state (the
shared engine, the index watermark) rather than spying host-side kernel
calls that no longer happen.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP

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
    def test_no_flags_runs_all(self, kharness, mcp_app):
        # No cleanup flags -> all cleanups run. dry_run:true previews them all.
        result = kharness.call_tool(mcp_app, "collection_prune", {"dry_run": True})
        assert result["dry_run"] is True
        # All three sections present (all cleanups ran).
        assert result["unused_tags"] is not None
        assert result["empty_notes"] is not None
        assert result["empty_cards"] is not None

    def test_apply_empty_notes_removes_from_index(self, kharness, mcp_app):
        blank = _blank_note(kharness)
        assert kharness.engine.contains(blank)  # indexed when it still had text
        result = kharness.call_tool(
            mcp_app, "collection_prune", {"empty_notes": True, "dry_run": False}
        )
        assert result["empty_notes"]["removed"] == [blank]
        assert not kharness.engine.contains(blank)  # vectors left with the note
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_apply_unused_tags_bumps_without_index_remove(self, kharness, mcp_app):
        _orphan_tag(kharness)
        result = kharness.call_tool(
            mcp_app, "collection_prune", {"unused_tags": True, "dry_run": False}
        )
        assert result["unused_tags"]["removed"] >= 1
        # No notes deleted, but the watermark advanced (the kernel tail's
        # metadata_changed) — the col_mod bump doesn't read as drift.
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_dry_run_does_not_mutate(self, kharness, mcp_app):
        blank = _blank_note(kharness)
        result = kharness.call_tool(
            mcp_app, "collection_prune", {"empty_notes": True, "dry_run": True}
        )
        assert result["dry_run"] is True
        assert result["empty_notes"]["removed"] == [blank]
        # Still there; index untouched.
        found = kharness.run(kharness.wrapper.run(lambda c: c.find_notes(f"nid:{blank}")))
        assert found
        assert kharness.engine.contains(blank)

    def test_applies_by_default(self, kharness, mcp_app):
        # dry_run defaults to false now — omitting it applies the cleanup.
        blank = _blank_note(kharness)
        result = kharness.call_tool(mcp_app, "collection_prune", {"empty_notes": True})
        assert result["dry_run"] is False
        assert result["empty_notes"]["removed"] == [blank]
        # Gone; its vectors left the index too.
        found = kharness.run(kharness.wrapper.run(lambda c: c.find_notes(f"nid:{blank}")))
        assert not found
        assert not kharness.engine.contains(blank)

    def test_unrequested_cleanup_absent_from_response(self, kharness, mcp_app):
        result = kharness.call_tool(
            mcp_app, "collection_prune", {"empty_notes": True, "dry_run": True}
        )
        assert result["empty_notes"] is not None
        assert result["unused_tags"] is None
        assert result["empty_cards"] is None
