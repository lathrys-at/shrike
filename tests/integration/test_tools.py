"""Integration tests that exercise every MCP tool over HTTP transport.

Each test class gets its own isolated server with a fresh collection.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionInfo:
    """collection_info on an empty collection."""

    def test_returns_all_sections(self, mcp):
        result = mcp("collection_info")
        assert "note_types" in result
        assert "decks" in result
        assert "tags" in result
        assert "stats" in result

    def test_include_filters(self, mcp):
        result = mcp("collection_info", {"include": ["stats"]})
        assert "stats" in result
        assert "note_types" not in result
        assert "decks" not in result

    def test_include_multiple(self, mcp):
        result = mcp("collection_info", {"include": ["decks", "tags"]})
        assert "decks" in result
        assert "tags" in result
        assert "note_types" not in result

    def test_note_type_details(self, mcp):
        result = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Basic"]},
        )
        basic = next(nt for nt in result["note_types"] if nt["name"] == "Basic")
        assert "templates" in basic
        assert "css" in basic

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


class TestNoteLifecycle:
    """Full create -> list -> update -> delete cycle over HTTP."""

    def test_create_and_retrieve(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Lifecycle",
                        "note_type": "Basic",
                        "fields": {"Front": "Capital of France?", "Back": "Paris"},
                        "tags": ["geography"],
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"
        note_id = result["results"][0]["id"]

        listed = mcp("list_notes", {"ids": [note_id]})
        assert listed["total"] == 1
        note = listed["notes"][0]
        assert note["content"]["Front"] == "Capital of France?"
        assert note["deck"] == "Lifecycle"
        assert "geography" in note["tags"]

    def test_list_by_deck(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "DeckFilter",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(3)
                ]
            },
        )
        result = mcp("list_notes", {"deck": "DeckFilter"})
        assert result["total"] == 3

    def test_list_by_tags(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Tags",
                        "note_type": "Basic",
                        "fields": {"Front": "Q1", "Back": "A1"},
                        "tags": ["findme"],
                    },
                    {
                        "deck": "Tags",
                        "note_type": "Basic",
                        "fields": {"Front": "Q2", "Back": "A2"},
                        "tags": ["other"],
                    },
                ]
            },
        )
        result = mcp("list_notes", {"tags": ["findme"]})
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "Q1"

    def test_list_by_note_type(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        result = mcp("list_notes", {"note_type": "Basic"})
        assert result["total"] >= 1
        assert all(n["note_type"] == "Basic" for n in result["notes"])

    def test_list_meta_only(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Meta",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = result["results"][0]["id"]
        listed = mcp("list_notes", {"ids": [note_id], "fields": "meta"})
        note = listed["notes"][0]
        assert "content" not in note
        assert note["note_type"] == "Basic"
        assert note["deck"] == "Meta"

    def test_list_with_limit(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Limit",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(5)
                ]
            },
        )
        result = mcp("list_notes", {"deck": "Limit", "limit": 2})
        assert len(result["notes"]) == 2
        assert result["total"] == 5

    def test_update_fields(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Update",
                        "note_type": "Basic",
                        "fields": {"Front": "Original Q", "Back": "Original A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        updated = mcp(
            "upsert_notes",
            {"notes": [{"id": note_id, "fields": {"Back": "Updated A"}}]},
        )
        assert updated["results"][0]["status"] == "updated"

        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert note["content"]["Back"] == "Updated A"
        assert note["content"]["Front"] == "Original Q"

    def test_update_tags(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                        "tags": ["old"],
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        mcp("upsert_notes", {"notes": [{"id": note_id, "tags": ["new", "updated"]}]})
        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert set(note["tags"]) == {"new", "updated"}

    def test_move_deck(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Source",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        mcp("upsert_notes", {"notes": [{"id": note_id, "deck": "Destination"}]})
        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert note["deck"] == "Destination"

    def test_delete(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Delete",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        result = mcp("delete_notes", {"ids": [note_id]})
        assert note_id in result["deleted"]

        listed = mcp("list_notes", {"ids": [note_id]})
        assert listed["total"] == 0

    def test_note_has_modified_timestamp(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]
        note = mcp("list_notes", {"ids": [note_id]})["notes"][0]
        assert "modified" in note


class TestListNotesAdvanced:
    """Tests for modified_since, query, and limit clamping."""

    def test_modified_since(self, mcp):
        import time
        from datetime import UTC, datetime

        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "TimeDeck",
                        "note_type": "Basic",
                        "fields": {"Front": "Old", "Back": "Note"},
                    }
                ]
            },
        )
        time.sleep(1)
        cutoff = datetime.now(UTC).isoformat()
        time.sleep(1)
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "TimeDeck",
                        "note_type": "Basic",
                        "fields": {"Front": "New", "Back": "Note"},
                    }
                ]
            },
        )

        result = mcp("list_notes", {"deck": "TimeDeck", "modified_since": cutoff})
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "New"

    def test_query_raw_search(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Search",
                        "note_type": "Basic",
                        "fields": {"Front": "mitochondria", "Back": "powerhouse"},
                    },
                    {
                        "deck": "Search",
                        "note_type": "Basic",
                        "fields": {"Front": "ribosome", "Back": "protein"},
                    },
                ]
            },
        )
        result = mcp("list_notes", {"query": "mitochondria"})
        assert result["total"] == 1
        assert "mitochondria" in result["notes"][0]["content"]["Front"]

    def test_limit_clamped_to_max(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Clamp",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(3)
                ]
            },
        )
        result = mcp("list_notes", {"deck": "Clamp", "limit": 999})
        assert result["limit"] == 200
        assert result["total"] == 3

    def test_limit_clamped_to_min(self, mcp):
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "ClampMin",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        result = mcp("list_notes", {"deck": "ClampMin", "limit": -5})
        assert result["limit"] == 1


class TestBulkOperations:
    """Batch creates, partial failures, mixed deletes."""

    def test_create_multiple(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Bulk",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(5)
                ]
            },
        )
        assert len(result["results"]) == 5
        assert all(r["status"] == "created" for r in result["results"])

        listed = mcp("list_notes", {"deck": "Bulk"})
        assert listed["total"] == 5

    def test_partial_failure(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Bulk",
                        "note_type": "Basic",
                        "fields": {"Front": "Good", "Back": "Note"},
                    },
                    {
                        "deck": "Bulk",
                        "note_type": "DoesNotExist",
                        "fields": {"Front": "Bad", "Back": "Note"},
                    },
                ]
            },
        )
        assert result["results"][0]["status"] == "created"
        assert result["results"][1]["status"] == "error"

    def test_delete_nonexistent(self, mcp):
        result = mcp("delete_notes", {"ids": [9999999999999]})
        assert result["deleted"] == []
        assert 9999999999999 in result["not_found"]

    def test_delete_mixed(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": "Q", "Back": "A"},
                    }
                ]
            },
        )
        real_id = created["results"][0]["id"]
        result = mcp("delete_notes", {"ids": [real_id, 9999999999999]})
        assert real_id in result["deleted"]
        assert 9999999999999 in result["not_found"]

    def test_delete_multiple(self, mcp):
        created = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Basic",
                        "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
                    }
                    for i in range(3)
                ]
            },
        )
        ids = [r["id"] for r in created["results"]]
        result = mcp("delete_notes", {"ids": ids})
        assert set(result["deleted"]) == set(ids)


class TestNoteTypeLifecycle:
    """Create, inspect, use, and update custom note types."""

    def test_create_and_inspect(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Custom",
                        "fields": ["Term", "Definition", "Example"],
                        "templates": [
                            {
                                "name": "Forward",
                                "front": "<div>{{Term}}</div>",
                                "back": "{{FrontSide}}<hr>{{Definition}}<br>{{Example}}",
                            }
                        ],
                        "css": ".card { font-family: sans-serif; }",
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

        info = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Custom"]},
        )
        nt = next(nt for nt in info["note_types"] if nt["name"] == "Custom")
        assert nt["fields"] == ["Term", "Definition", "Example"]
        assert len(nt["templates"]) == 1
        assert "font-family" in nt["css"]

    def test_create_note_with_custom_type(self, mcp):
        mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Vocab",
                        "fields": ["Word", "Meaning"],
                        "templates": [
                            {"name": "Card 1", "front": "{{Word}}", "back": "{{Meaning}}"}
                        ],
                        "css": ".card { font-size: 16px; }",
                    }
                ]
            },
        )
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Default",
                        "note_type": "Vocab",
                        "fields": {"Word": "shrike", "Meaning": "A predatory songbird"},
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

    def test_update_css(self, mcp):
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Styled",
                        "fields": ["Q", "A"],
                        "templates": [{"name": "Card 1", "front": "{{Q}}", "back": "{{A}}"}],
                        "css": ".card { color: black; }",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]

        mcp(
            "upsert_note_types",
            {"note_types": [{"id": nt_id, "css": ".card { color: red; }"}]},
        )
        info = mcp(
            "collection_info",
            {"include": ["note_types"], "note_type_details": ["Styled"]},
        )
        nt = next(nt for nt in info["note_types"] if nt["name"] == "Styled")
        assert "color: red" in nt["css"]

    def test_update_name(self, mcp):
        created = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "OldName",
                        "fields": ["F"],
                        "templates": [{"name": "Card 1", "front": "{{F}}", "back": "{{F}}"}],
                        "css": "",
                    }
                ]
            },
        )
        nt_id = created["results"][0]["id"]

        result = mcp("upsert_note_types", {"note_types": [{"id": nt_id, "name": "NewName"}]})
        assert result["results"][0]["status"] == "updated"

        info = mcp("collection_info", {"include": ["note_types"]})
        names = {nt["name"] for nt in info["note_types"]}
        assert "NewName" in names
        assert "OldName" not in names

    def test_duplicate_name_rejected(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Basic",
                        "fields": ["A"],
                        "templates": [{"name": "C", "front": "{{A}}", "back": "{{A}}"}],
                        "css": "",
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "error"


class TestSearchNotesStub:
    def test_returns_stub_message(self, mcp):
        result = mcp("search_notes", {"queries": ["anything"]})
        assert result["results"] == []
        assert "_message" in result

    def test_requires_queries_or_ids(self, mcp):
        result = mcp("search_notes", {})
        assert "error" in result


class TestValidation:
    """Input validation errors reported over transport."""

    def test_list_notes_requires_filter(self, mcp):
        result = mcp("list_notes", {})
        assert "error" in result

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
        result = mcp(
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
        assert "error" in result
        assert "100" in result["error"]

    def test_upsert_note_types_over_10_rejected(self, mcp):
        result = mcp(
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
        assert "error" in result
        assert "10" in result["error"]

    def test_delete_notes_over_100_rejected(self, mcp):
        result = mcp("delete_notes", {"ids": list(range(101))})
        assert "error" in result
        assert "100" in result["error"]
