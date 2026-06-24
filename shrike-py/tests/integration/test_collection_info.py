"""Integration tests for the collection_info MCP tool over HTTP transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionInfo:
    """collection_info on an empty collection."""

    def test_returns_summary_by_default(self, mcp):
        result = mcp("collection_info")
        assert "summary" in result
        summary = result["summary"]
        assert "path" in summary
        assert "notes" in summary
        assert "decks" in summary

    def test_include_filters(self, mcp):
        result = mcp("collection_info", {"include": ["stats"]})
        assert result["stats"] is not None
        assert result["note_types"] is None
        assert result["decks"] is None

    def test_include_multiple(self, mcp):
        result = mcp("collection_info", {"include": ["decks", "tags"]})
        assert result["decks"] is not None
        assert result["tags"] is not None
        assert result["note_types"] is None

    def test_note_type_details(self, mcp):
        result = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Basic"]},
        )
        basic = next(nt for nt in result["note_types"] if nt["name"] == "Basic")
        assert basic["detail"]["templates"]
        assert "css" in basic["detail"]

    def test_default_note_types(self, mcp):
        result = mcp("collection_info", {"include": ["note_types"]})
        names = {nt["name"] for nt in result["note_types"]}
        assert "Basic" in names
        assert "Cloze" in names

    def test_default_deck(self, mcp):
        result = mcp("collection_info", {"include": ["decks"]})
        assert any(d["name"] == "Default" for d in result["decks"])

    def test_stats_empty_collection(self, mcp):
        result = mcp("collection_info", {"include": ["stats"]})
        assert result["stats"]["total_notes"] == 0
        assert result["stats"]["total_cards"] == 0

    def test_stats_after_adding_notes(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Stats",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(3)
                ]
            },
        )
        result = mcp("collection_info", {"include": ["stats", "decks"]})
        assert result["stats"]["total_notes"] == 3
        stats_deck = next(d for d in result["decks"] if d["name"] == "Stats")
        assert stats_deck["note_count"] == 3

    def test_tags_appear(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                        "tags": ["alpha", "beta"],
                    }
                ]
            },
        )
        result = mcp("collection_info", {"include": ["tags"]})
        assert "alpha" in result["tags"]
        assert "beta" in result["tags"]
