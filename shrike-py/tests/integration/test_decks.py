"""Integration tests for the deck upsert/delete MCP tools over transport."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestDeckOps:
    """upsert_decks + delete_decks over transport."""

    def _deck_names(self, mcp):
        return {d["name"] for d in mcp("collection_info", {"include": ["decks"]})["decks"]}

    def test_create(self, mcp):
        result = mcp("upsert_decks", {"decks": [{"name": "Japanese::Vocab"}]})
        assert result["results"][0]["status"] == "created"
        assert "Japanese::Vocab" in self._deck_names(mcp)

    def test_create_existing_is_updated(self, mcp):
        mcp("upsert_decks", {"decks": [{"name": "Dup"}]})
        again = mcp("upsert_decks", {"decks": [{"name": "Dup"}]})
        assert again["results"][0]["status"] == "updated"

    def test_rename(self, mcp):
        created = mcp("upsert_decks", {"decks": [{"name": "Before"}]})
        did = created["results"][0]["id"]
        renamed = mcp("upsert_decks", {"decks": [{"id": did, "name": "After"}]})
        assert renamed["results"][0]["status"] == "updated"
        names = self._deck_names(mcp)
        assert "After" in names and "Before" not in names

    def test_rename_onto_existing_errors(self, mcp):
        a = mcp("upsert_decks", {"decks": [{"name": "Aa"}]})
        mcp("upsert_decks", {"decks": [{"name": "Bb"}]})
        result = mcp("upsert_decks", {"decks": [{"id": a["results"][0]["id"], "name": "Bb"}]})
        assert result["results"][0]["status"] == "error"

    def test_delete_empty(self, mcp):
        mcp("upsert_decks", {"decks": [{"name": "Temp"}]})
        result = mcp("delete_decks", {"decks": ["Temp"]})
        assert result["deleted"] == ["Temp"]
        assert "Temp" not in self._deck_names(mcp)

    def test_delete_non_empty_refused(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {"deck": "Full", "note_type": "Basic", "fields": {"Front": "Q", "Back": "A"}}
                ]
            },
        )
        result = mcp("delete_decks", {"decks": ["Full"]})
        assert result["not_empty"] == ["Full"]
        assert "Full" in self._deck_names(mcp)

    def test_delete_not_found(self, mcp):
        result = mcp("delete_decks", {"decks": ["Ghost"]})
        assert result["not_found"] == ["Ghost"]

    def test_deck_by_id_across_tools(self, mcp):
        # create deck, then reference it by #id (create note) and numeric id (list)
        did = mcp("upsert_decks", {"decks": [{"name": "ByID"}]})["results"][0]["id"]
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {"deck": f"#{did}", "note_type": "Basic", "fields": {"Front": "q", "Back": "a"}}
                ]
            },
        )
        assert mcp("list_notes", {"deck": str(did)})["total"] == 1
        # non-empty deck deleted by #id is refused (echoing the ref)
        assert mcp("delete_decks", {"decks": [f"#{did}"]})["not_empty"] == [f"#{did}"]
