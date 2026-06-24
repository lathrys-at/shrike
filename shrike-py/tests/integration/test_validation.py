"""Integration tests for input validation and batch-size limits over transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestValidation:
    """Input validation errors reported over transport."""

    def test_list_notes_requires_filter(self, mcp):
        with pytest.raises(RuntimeError, match="filter"):
            mcp("list_notes", {})

    def test_upsert_notes_missing_deck(self, mcp):
        result = mcp(
            "upsert_notes",
            {"notes": [{"note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}]},
        )
        assert result["results"][0]["status"] == "error"

    def test_upsert_notes_invalid_note_type(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Test",
                        "note_type": "Nonexistent",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "error"
        assert "not found" in result["results"][0]["error"].lower()

    def test_upsert_notes_nonexistent_field(self, mcp):
        result = mcp(
            "upsert_notes",
            {"notes": [{"deck": "Test", "note_type": "Basic", "fields": {"Nonexistent": "Q"}}]},
        )
        assert result["results"][0]["status"] == "error"

    def test_update_nonexistent_note(self, mcp):
        result = mcp("upsert_notes", {"notes": [{"id": 9999999999999, "fields": {"Front": "Q"}}]})
        assert result["results"][0]["status"] == "error"

    def test_list_nonexistent_id(self, mcp):
        result = mcp("list_notes", {"ids": [9999999999999]})
        assert result["total"] == 0
        assert result["notes"] == []


class TestBatchLimits:
    """Verify the server enforces maximum batch sizes."""

    def test_upsert_notes_over_100_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="at most 100"):
            mcp(
                "upsert_notes",
                {
                    "notes": [
                        {
                            "deck": "Batch",
                            "note_type": "Basic",
                            "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                        }
                        for i in range(101)
                    ]
                },
            )

    def test_upsert_note_types_over_10_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="at most 10"):
            mcp(
                "upsert_note_types",
                {
                    "note_types": [
                        {
                            "name": f"Type{i}",
                            "fields": ["F"],
                            "templates": [{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
                            "css": "",
                        }
                        for i in range(11)
                    ]
                },
            )

    def test_delete_notes_over_100_rejected(self, mcp):
        with pytest.raises(RuntimeError, match="at most 100"):
            mcp("delete_notes", {"ids": list(range(101))})
