"""Integration tests for the find_replace_notes MCP tool over transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestFindReplace:
    """find_replace_notes over transport."""

    def _make(self, mcp, deck, front, back="x"):
        note = {"deck": deck, "note_type": "Basic", "fields": {"Front": front, "Back": back}}
        mcp("upsert_notes", {"notes": [note]})

    def test_dry_run_then_apply(self, mcp):
        self._make(mcp, "FR", "teh cell", "teh power")
        args = {"search": "teh", "replace": "the", "deck": "FR", "dry_run": True}
        dry = mcp("find_replace_notes", args)
        assert dry["dry_run"] is True
        assert dry["notes_changed"] == 1
        assert dry["samples"]
        # dry-run changed nothing
        assert mcp("list_notes", {"deck": "FR"})["notes"][0]["content"]["Front"] == "teh cell"

        applied = mcp("find_replace_notes", {"search": "teh", "replace": "the", "deck": "FR"})
        assert applied["notes_changed"] == 1
        note = mcp("list_notes", {"deck": "FR"})["notes"][0]
        assert note["content"]["Front"] == "the cell"
        assert note["content"]["Back"] == "the power"

    def test_regex(self, mcp):
        self._make(mcp, "FRrx", "colour and flavour")
        applied = mcp(
            "find_replace_notes",
            {"search": "(colou?r|flavou?r)", "replace": "X", "regex": True, "deck": "FRrx"},
        )
        assert applied["notes_changed"] == 1
        assert mcp("list_notes", {"deck": "FRrx"})["notes"][0]["content"]["Front"] == "X and X"

    def test_requires_scope(self, mcp):
        with pytest.raises(RuntimeError, match="scope"):
            mcp("find_replace_notes", {"search": "a", "replace": "b"})
