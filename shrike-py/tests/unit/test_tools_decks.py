"""Tool-layer tests for deck ops: col_mod bump without vector changes.

Deck create/rename/delete-empty never change a note's embedding text, so they
must advance the kernel's stored watermark (avoiding a spurious rebuild)
WITHOUT touching vectors. The kernel op itself carries that tail: "no vectors
touched" is "no new embed call"; "bumped" is "the index col_mod matches the
collection + no drift" — observable state, not host-side spies.
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


class TestUpsertDecksBump:
    def test_create_bumps_col_mod_without_vectors(self, kharness, backend, mcp_app):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(mcp_app, "upsert_decks", {"decks": [{"name": "New"}]})
        assert result["results"][0]["status"] == "created"
        assert len(backend.calls) == embeds_before  # no re-embed
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_all_error_does_not_bump(self, kharness, mcp_app):
        result = kharness.call_tool(
            mcp_app, "upsert_decks", {"decks": [{"id": 9999999999999, "name": "X"}]}
        )
        assert result["results"][0]["status"] == "error"
        # Nothing changed, nothing written: no drift (the kernel tail
        # no-ops on an all-error batch).
        assert kharness.reindex_if_needed() is False

    def test_empty_list_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(mcp_app, "upsert_decks", {"decks": []})


class TestDeleteDecksTool:
    def test_delete_empty_bumps(self, kharness, backend, mcp_app):
        kharness.call_tool(mcp_app, "upsert_decks", {"decks": [{"name": "Temp"}]})
        embeds_before = len(backend.calls)

        result = kharness.call_tool(mcp_app, "delete_decks", {"decks": ["Temp"]})
        assert result["deleted"] == ["Temp"]
        assert len(backend.calls) == embeds_before
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_non_empty_reported_and_no_bump(self, kharness, mcp_app):
        kharness.seed_note("Q", deck="Full", back="A")

        result = kharness.call_tool(mcp_app, "delete_decks", {"decks": ["Full"]})
        assert result["not_empty"] == ["Full"]
        assert result["deleted"] == []
        # Nothing deleted, nothing written: no drift either way.
        assert kharness.reindex_if_needed() is False
