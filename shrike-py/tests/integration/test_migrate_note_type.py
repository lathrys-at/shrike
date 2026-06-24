"""Integration tests for the migrate_note_type MCP tool over transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestMigrateNoteType:
    """migrate_note_type (change note type) over transport. Self-seeding (xdist)."""

    def _basic(self, mcp, front, back="x"):
        r = mcp(
            "upsert_notes",
            {
                "notes": [
                    {"deck": "M", "note_type": "Basic", "fields": {"Front": front, "Back": back}}
                ]
            },
        )
        return r["results"][0]["id"]

    def test_basic_to_cloze(self, mcp):
        nid = self._basic(mcp, "migrate-front", "migrate-back")
        result = mcp(
            "migrate_note_type",
            {
                "note_ids": [nid],
                "new_note_type": "Cloze",
                "field_map": {"Front": "Text", "Back": "Back Extra"},
                "dry_run": False,
            },
        )
        assert result["changed"] == [nid]
        assert result["to_note_type"] == "Cloze"
        note = mcp("list_notes", {"ids": [nid]})["notes"][0]
        assert note["note_type"] == "Cloze"
        assert note["content"]["Text"] == "migrate-front"

    def test_dry_run_reports_drops_without_changing(self, mcp):
        nid = self._basic(mcp, "keep-front", "drop-back")
        result = mcp(
            "migrate_note_type",
            {
                "note_ids": [nid],
                "new_note_type": "Cloze",
                "field_map": {"Front": "Text"},
                "dry_run": True,
            },
        )
        assert result["dry_run"] is True
        assert result["dropped_fields"] == ["Back"]
        assert mcp("list_notes", {"ids": [nid]})["notes"][0]["note_type"] == "Basic"

    def test_bad_map_errors(self, mcp):
        nid = self._basic(mcp, "bad-map")
        with pytest.raises(RuntimeError):
            mcp(
                "migrate_note_type",
                {"note_ids": [nid], "new_note_type": "Cloze", "field_map": {"Nope": "Text"}},
            )
