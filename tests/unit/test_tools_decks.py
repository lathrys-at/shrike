"""Tool-layer tests for deck ops (#74): col_mod bump without vector changes.

Deck create/rename/delete-empty never change a note's embedding text, so they
must advance the kernel's stored watermark (avoiding a spurious rebuild)
WITHOUT touching vectors. Kernel-harness port (#355): "no vectors touched" is
"no new embed call"; "bumped" is "kernel.metadata_changed ran and the index
col_mod matches the collection".
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


class TestUpsertDecksBump:
    def test_create_bumps_col_mod_without_vectors(self, kharness, backend, kproxy, mcp_app):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(mcp_app, "upsert_decks", {"decks": [{"name": "New"}]})
        assert result["results"][0]["status"] == "created"
        assert len(backend.calls) == embeds_before  # no re-embed
        assert kproxy.calls["metadata_changed"] == 1
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_all_error_does_not_bump(self, kharness, kproxy, mcp_app):
        result = kharness.call_tool(
            mcp_app, "upsert_decks", {"decks": [{"id": 9999999999999, "name": "X"}]}
        )
        assert result["results"][0]["status"] == "error"
        assert kproxy.calls["metadata_changed"] == 0

    def test_empty_list_rejected(self, kharness, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(mcp_app, "upsert_decks", {"decks": []})


class TestDeleteDecksTool:
    def test_delete_empty_bumps(self, kharness, backend, kproxy, mcp_app):
        kharness.call_tool(mcp_app, "upsert_decks", {"decks": [{"name": "Temp"}]})
        kproxy.calls["metadata_changed"] = 0
        embeds_before = len(backend.calls)

        result = kharness.call_tool(mcp_app, "delete_decks", {"decks": ["Temp"]})
        assert result["deleted"] == ["Temp"]
        assert len(backend.calls) == embeds_before
        assert kproxy.calls["metadata_changed"] == 1
        assert kharness.index_status()["col_mod"] == kharness.col_mod()

    def test_non_empty_reported_and_no_bump(self, kharness, kproxy, mcp_app):
        kharness.seed_note("Q", deck="Full", back="A")
        kproxy.calls["metadata_changed"] = 0

        result = kharness.call_tool(mcp_app, "delete_decks", {"decks": ["Full"]})
        assert result["not_empty"] == ["Full"]
        assert result["deleted"] == []
        assert kproxy.calls["metadata_changed"] == 0
