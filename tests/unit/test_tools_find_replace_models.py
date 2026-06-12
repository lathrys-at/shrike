"""Tool-layer tests for find_replace_note_types (#76, findAndReplaceInModels).

Editing a note type's template HTML / CSS never changes any note's embedding
text, so a successful replace must advance the kernel's stored watermark
(avoiding a spurious rebuild) WITHOUT touching vectors — the same
metadata-bump contract as the tag/deck ops. A no-op replace (no matches)
bumps nothing. Kernel-harness port (#355).
"""

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
    return kharness.run(
        kharness.wrapper.run(
            lambda c: upsert_note_types(
                c,
                [
                    {
                        "name": "FR",
                        "fields": ["F"],
                        "templates": [{"name": "C", "front": "old {{F}}", "back": "{{F}}"}],
                        "css": ".card { color: red; }",
                    }
                ],
            )[0]["id"]
        )
    )


class TestFindReplaceNoteTypesTool:
    def test_replace_bumps_col_mod_without_vectors(self, kharness, model, backend, kproxy, mcp_app):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app,
            "find_replace_note_types",
            {"note_type": "FR", "search": "red", "replace": "blue"},
        )
        assert result["replacements"] == 1
        assert result["css_changed"] is True
        # Templates/CSS are not embedding text — no vectors touched.
        assert len(backend.calls) == embeds_before
        assert kproxy.calls["metadata_changed"] == 1
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_no_match_does_not_bump(self, kharness, model, kproxy, mcp_app):
        result = kharness.call_tool(
            mcp_app,
            "find_replace_note_types",
            {"note_type": "FR", "search": "absent", "replace": "x"},
        )
        assert result["replacements"] == 0
        assert kproxy.calls["metadata_changed"] == 0

    def test_unknown_note_type_is_tool_error(self, kharness, model, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "find_replace_note_types",
                {"note_type": "Nope", "search": "a", "replace": "b"},
            )

    def test_no_location_selected_is_tool_error(self, kharness, model, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "find_replace_note_types",
                {
                    "note_type": "FR",
                    "search": "old",
                    "replace": "x",
                    "front": False,
                    "back": False,
                    "css": False,
                },
            )

    def test_empty_search_rejected(self, kharness, model, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "find_replace_note_types",
                {"note_type": "FR", "search": "", "replace": "x"},
            )

    def test_invalid_regex_is_tool_error(self, kharness, model, mcp_app):
        with pytest.raises(ToolError):
            kharness.call_tool(
                mcp_app,
                "find_replace_note_types",
                {"note_type": "FR", "search": "(unclosed", "replace": "x", "regex": True},
            )
