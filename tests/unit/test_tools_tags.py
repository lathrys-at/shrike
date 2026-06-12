"""Tool-layer tests for tag ops (#73): input validation + index col_mod bump.

A tag change never alters a note's embedding vector (tags aren't part of the
embedding text), but it does bump col.mod. The kernel must advance the stored
watermarks (kernel.metadata_changed) WITHOUT re-embedding, so the next startup
doesn't trigger a needless full rebuild. Runs on the #355 kernel harness:
"vectors untouched" is asserted as "no new embed call", and "watermark
advanced" as "index col_mod matches the collection + no drift".
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

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
    return proxy


@pytest.fixture()
def mcp_app(kharness, kproxy):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kproxy)
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
    def test_set_modifies_and_bumps_col_mod(self, kharness, backend, kproxy, mcp_app, kbasic_note):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app, "update_note_tags", {"note_ids": [kbasic_note], "set": ["x"]}
        )
        assert result["notes_modified"] == 1
        # Vectors untouched; only the watermark advanced.
        assert len(backend.calls) == embeds_before
        assert kproxy.calls["metadata_changed"] == 1
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_add_remove_bumps_col_mod(self, kharness, backend, kproxy, mcp_app, kbasic_note):
        embeds_before = len(backend.calls)
        kharness.call_tool(mcp_app, "update_note_tags", {"note_ids": [kbasic_note], "add": ["new"]})
        assert len(backend.calls) == embeds_before
        assert kproxy.calls["metadata_changed"] == 1
        assert kharness.index_status()["col_mod"] == kharness.col_mod()

    def test_no_match_does_not_bump(self, kharness, kproxy, mcp_app):
        result = kharness.call_tool(
            mcp_app, "update_note_tags", {"note_ids": [9999999999999], "add": ["x"]}
        )
        assert result["notes_modified"] == 0
        assert kproxy.calls["metadata_changed"] == 0


class TestRenameTagTool:
    def test_identical_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError, match="identical"):
            kharness.call_tool(mcp_app, "rename_tag", {"old": "a", "new": "a"})

    def test_rename_bumps_col_mod(self, kharness, backend, kproxy, mcp_app, kbasic_note):
        # kbasic_note has tag "math"; collection-wide rename.
        embeds_before = len(backend.calls)
        result = kharness.call_tool(mcp_app, "rename_tag", {"old": "math", "new": "arithmetic"})
        assert result["notes_modified"] == 1
        assert len(backend.calls) == embeds_before
        assert kproxy.calls["metadata_changed"] == 1
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
