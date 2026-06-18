"""Tool-layer tests for tag ops (#73): input validation + index col_mod bump.

A tag change never alters a note's embedding vector (tags aren't part of the
embedding text), but it does bump col.mod. Since the #391 re-home the kernel
op itself advances the stored watermarks WITHOUT re-embedding, so the next
startup doesn't trigger a needless full rebuild. "Vectors untouched" is
asserted as "no new embed call", and "watermark advanced" as "index col_mod
matches the collection + no drift" — observable state, not host-side spies.
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


class TestUpdateNoteTagsValidation:
    def test_set_and_add_rejected(self, kharness, mcp_app, kbasic_note):
        with pytest.raises(ToolError, match="not both"):
            kharness.call_tool(
                mcp_app,
                "update_note_tags",
                {"note_ids": [kbasic_note], "set": ["a"], "add": ["b"]},
            )

    def test_no_mode_rejected(self, kharness, mcp_app, kbasic_note):
        with pytest.raises(ToolError, match="Specify"):
            kharness.call_tool(mcp_app, "update_note_tags", {"note_ids": [kbasic_note]})


class TestUpdateNoteTagsBump:
    def test_set_modifies_and_bumps_col_mod(self, kharness, backend, mcp_app, kbasic_note):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app, "update_note_tags", {"note_ids": [kbasic_note], "set": ["x"]}
        )
        assert result["notes_modified"] == 1
        # Vectors untouched; only the watermark advanced (the kernel tail).
        assert len(backend.calls) == embeds_before
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_add_remove_bumps_col_mod(self, kharness, backend, mcp_app, kbasic_note):
        embeds_before = len(backend.calls)
        kharness.call_tool(mcp_app, "update_note_tags", {"note_ids": [kbasic_note], "add": ["new"]})
        assert len(backend.calls) == embeds_before
        assert kharness.index_status()["col_mod"] == kharness.col_mod()

    def test_no_match_does_not_bump(self, kharness, mcp_app):
        result = kharness.call_tool(
            mcp_app, "update_note_tags", {"note_ids": [9999999999999], "add": ["x"]}
        )
        assert result["notes_modified"] == 0
        # Nothing modified, nothing written: collection and index stay
        # consistent with no drift (the kernel tail no-ops on changed=0).
        assert kharness.reindex_if_needed() is False


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
