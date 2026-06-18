"""Tool-layer tests for migrate_note_type (#75): index re-embed + validation.

Remapped fields change embedding text, so an applied migration must re-embed
the changed notes (their ids are unchanged). Since the #391 re-home that
tail runs inside the kernel's op, so the assertions read observable state
(the embed-call log, the index watermark) rather than spying host-side
kernel calls that no longer happen. Dry-run and validation failures must
not touch the index.
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


class TestMigrateNoteTypeTool:
    def test_apply_reembeds_changed_notes(self, kharness, backend, mcp_app):
        nid = kharness.seed_note("hello", deck="D")
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app,
            "migrate_note_type",
            {
                "note_ids": [nid],
                "new_note_type": "Cloze",
                "field_map": {"Front": "Text", "Back": "Back Extra"},
                "dry_run": False,
            },
        )
        assert result["changed"] == [nid]
        # Exactly the migrated note re-embedded (the kernel tail).
        new_calls = backend.calls[embeds_before:]
        assert len(new_calls) == 1
        assert len(new_calls[0]) == 1
        assert "hello" in new_calls[0][0]
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_dry_run_does_not_touch_index(self, kharness, backend, mcp_app):
        nid = kharness.seed_note("hi", deck="D")
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app,
            "migrate_note_type",
            {
                "note_ids": [nid],
                "new_note_type": "Cloze",
                "field_map": {"Front": "Text"},
                "dry_run": True,
            },
        )
        assert result["dry_run"] is True
        assert len(backend.calls) == embeds_before

    def test_bad_field_map_is_tool_error(self, kharness, mcp_app):
        nid = kharness.seed_note("x", deck="D")
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "migrate_note_type",
                {"note_ids": [nid], "new_note_type": "Cloze", "field_map": {"Nope": "Text"}},
            )

    def test_empty_field_map_rejected(self, kharness, mcp_app):
        nid = kharness.seed_note("x", deck="D")
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "migrate_note_type",
                {"note_ids": [nid], "new_note_type": "Cloze", "field_map": {}},
            )

    def test_empty_note_ids_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "migrate_note_type",
                {"note_ids": [], "new_note_type": "Cloze", "field_map": {"Front": "Text"}},
            )
