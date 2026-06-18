"""Tool-layer tests for find_replace_note_types (findAndReplaceInModels).

Editing a note type's template HTML / CSS never changes any note's embedding
text, so a successful replace must advance the kernel's stored watermark
(avoiding a spurious rebuild) WITHOUT touching vectors — the same
metadata-bump contract as the tag/deck ops. A no-op replace (no matches)
saves nothing and bumps nothing. That tail runs inside the kernel's op, so the
assertions read observable state (the index watermark, col.mod itself) rather
than spying host-side kernel calls that no longer happen.
"""

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
    def test_replace_bumps_col_mod_without_vectors(self, kharness, model, backend, mcp_app):
        embeds_before = len(backend.calls)
        result = kharness.call_tool(
            mcp_app,
            "find_replace_note_types",
            {"note_type": "FR", "search": "red", "replace": "blue"},
        )
        assert result["replacements"] == 1
        assert result["css_changed"] is True
        # Templates/CSS are not embedding text — no vectors touched, but the
        # kernel tail advanced the watermark so the bump isn't drift.
        assert len(backend.calls) == embeds_before
        assert kharness.index_status()["col_mod"] == kharness.col_mod()
        assert kharness.reindex_if_needed() is False

    def test_no_match_does_not_bump(self, kharness, model, mcp_app):
        before = kharness.col_mod()
        result = kharness.call_tool(
            mcp_app,
            "find_replace_note_types",
            {"note_type": "FR", "search": "absent", "replace": "x"},
        )
        assert result["replacements"] == 0
        # No match → the model is never saved: col.mod unmoved, no drift.
        assert kharness.col_mod() == before
        assert kharness.reindex_if_needed() is False

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
