"""Tool-layer tests for update_note_type_field_metadata: col_mod bump,
no re-embed. The watermark tail runs inside the kernel's op, so the assertions
read observable state (the index watermark, the embed-call log) rather than
spying host-side kernel calls that no longer happen."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from shrike.api.tools import register_tools
from tests.unit._native_shims import upsert_note_types
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
    def test_bumps_col_mod_without_reembed(self, kharness, model, backend, mcp_app):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app,
            "update_note_type_field_metadata",
            {"note_type": "M", "fields": [{"name": "F", "size": 28, "description": "prompt"}]},
        )
        assert result["fields_updated"] == ["F"]
        # Editor metadata isn't embedding text: no re-embed, but the kernel
        # tail advances the watermark — the col_mod bump isn't drift.
        assert len(backend.calls) == embeds_before
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

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
