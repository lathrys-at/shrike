"""Tool-layer tests for update_note_type_field_metadata (#119): col_mod bump,
no re-embed. Kernel-harness port (#355)."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.tools import register_tools
from tests.unit._native_shims import upsert_note_types
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


@pytest.fixture()
def model(kharness):
    kharness.run(
        kharness.wrapper.run(
            lambda c: upsert_note_types(
                c,
                [
                    {
                        "name": "M",
                        "fields": ["F"],
                        "templates": [{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
                        "css": "",
                    }
                ],
            )
        )
    )


class TestSetFieldMetadataTool:
    def test_bumps_col_mod_without_reembed(self, kharness, model, backend, kproxy, mcp_app):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app,
            "update_note_type_field_metadata",
            {"note_type": "M", "fields": [{"name": "F", "size": 28, "description": "prompt"}]},
        )
        assert result["fields_updated"] == ["F"]
        # Editor metadata isn't embedding text: no re-embed, but col_mod advances.
        assert len(backend.calls) == embeds_before
        assert kproxy.calls["metadata_changed"] == 1
        assert kharness.index_status()["col_mod"] == kharness.col_mod()

    def test_unknown_field_is_tool_error(self, kharness, model, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "update_note_type_field_metadata",
                {"note_type": "M", "fields": [{"name": "Nope", "size": 1}]},
            )

    def test_empty_fields_rejected(self, kharness, model, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app, "update_note_type_field_metadata", {"note_type": "M", "fields": []}
            )
