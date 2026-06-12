"""Tool-layer tests for migrate_note_type (#75): index re-embed + validation.

Remapped fields change embedding text, so an applied migration must re-embed
the changed notes (their ids are unchanged) via kernel.reindex_notes — like
find_replace_notes. Dry-run and validation failures must not touch the index.
Kernel-harness port (#355).
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
    proxy.spy("reindex_notes")
    return proxy


@pytest.fixture()
def mcp_app(kharness, kproxy):
    mcp = FastMCP("test")
    register_tools(mcp, kharness.wrapper, kernel=kproxy)
    return mcp


class TestMigrateNoteTypeTool:
    def test_apply_reembeds_changed_notes(self, kharness, backend, kproxy, mcp_app):
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
        assert kproxy.calls["reindex_notes"] == 1
        # Exactly the migrated note re-embedded.
        new_calls = backend.calls[embeds_before:]
        assert len(new_calls) == 1
        assert len(new_calls[0]) == 1
        assert "hello" in new_calls[0][0]
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_dry_run_does_not_touch_index(self, kharness, backend, kproxy, mcp_app):
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
        assert kproxy.calls["reindex_notes"] == 0
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
