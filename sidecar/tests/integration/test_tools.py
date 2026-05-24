"""Integration tests that exercise every MCP tool over HTTP transport.

These tests run against a live Shrike server started by the session-scoped
`server` fixture. They share a single collection and run in order within
each class, so later tests may depend on state created by earlier ones.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestCollectionInfo:
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

    def test_note_type_details(self, mcp):
        result = mcp(
            "collection_info",
            {
                "include": ["note_types"],
                "note_type_details": ["Basic"],
            },
        )
        basic = next(nt for nt in result["note_types"] if nt["name"] == "Basic")
        assert "templates" in basic
        assert "css" in basic


class TestNoteLifecycle:
    """Create → read → update → verify → delete, all over HTTP."""

    _note_id: int | None = None

    def test_01_create_note(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Integration",
                        "note_type": "Basic",
                        "fields": {"Front": "Capital of France?", "Back": "Paris"},
                        "tags": ["geography"],
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"
        TestNoteLifecycle._note_id = result["results"][0]["id"]

    def test_02_list_by_deck(self, mcp):
        result = mcp("list_notes", {"deck": "Integration"})
        assert result["total"] >= 1
        note = next(n for n in result["notes"] if n["id"] == self._note_id)
        assert note["content"]["Front"] == "Capital of France?"
        assert note["deck"] == "Integration"

    def test_03_list_by_id(self, mcp):
        result = mcp("list_notes", {"ids": [self._note_id]})
        assert result["total"] == 1
        assert result["notes"][0]["id"] == self._note_id

    def test_04_list_meta_only(self, mcp):
        result = mcp("list_notes", {"ids": [self._note_id], "fields": "meta"})
        note = result["notes"][0]
        assert "content" not in note
        assert note["note_type"] == "Basic"

    def test_05_update_fields(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "id": self._note_id,
                        "fields": {"Back": "Paris (Île-de-France)"},
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "updated"

    def test_06_verify_update(self, mcp):
        result = mcp("list_notes", {"ids": [self._note_id]})
        note = result["notes"][0]
        assert note["content"]["Back"] == "Paris (Île-de-France)"
        assert note["content"]["Front"] == "Capital of France?"  # unchanged

    def test_07_update_tags(self, mcp):
        mcp("upsert_notes", {"notes": [{"id": self._note_id, "tags": ["geography", "europe"]}]})
        result = mcp("list_notes", {"ids": [self._note_id]})
        assert set(result["notes"][0]["tags"]) == {"geography", "europe"}

    def test_08_move_deck(self, mcp):
        mcp("upsert_notes", {"notes": [{"id": self._note_id, "deck": "Integration::Moved"}]})
        result = mcp("list_notes", {"ids": [self._note_id]})
        assert result["notes"][0]["deck"] == "Integration::Moved"

    def test_09_delete(self, mcp):
        result = mcp("delete_notes", {"ids": [self._note_id]})
        assert self._note_id in result["deleted"]

    def test_10_verify_deleted(self, mcp):
        result = mcp("list_notes", {"ids": [self._note_id]})
        assert result["total"] == 0


class TestBulkOperations:
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

    def test_delete_mixed_ids(self, mcp):
        result = mcp("delete_notes", {"ids": [9999999999999]})
        assert result["deleted"] == []
        assert 9999999999999 in result["not_found"]


class TestNoteTypeLifecycle:
    _note_type_id: int | None = None

    def test_01_create_note_type(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "name": "Integration Test Type",
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
        TestNoteTypeLifecycle._note_type_id = result["results"][0]["id"]

    def test_02_visible_in_collection_info(self, mcp):
        result = mcp(
            "collection_info",
            {
                "include": ["note_types"],
                "note_type_details": ["Integration Test Type"],
            },
        )
        nt = next(nt for nt in result["note_types"] if nt["name"] == "Integration Test Type")
        assert nt["fields"] == ["Term", "Definition", "Example"]
        assert len(nt["templates"]) == 1
        assert "font-family" in nt["css"]

    def test_03_create_note_with_custom_type(self, mcp):
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Integration",
                        "note_type": "Integration Test Type",
                        "fields": {
                            "Term": "Photosynthesis",
                            "Definition": "Conversion of light to chemical energy",
                            "Example": "Plants use chlorophyll",
                        },
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

    def test_04_update_note_type_css(self, mcp):
        result = mcp(
            "upsert_note_types",
            {
                "note_types": [
                    {
                        "id": self._note_type_id,
                        "css": ".card { font-family: serif; color: #333; }",
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "updated"

        info = mcp(
            "collection_info",
            {
                "include": ["note_types"],
                "note_type_details": ["Integration Test Type"],
            },
        )
        nt = next(nt for nt in info["note_types"] if nt["name"] == "Integration Test Type")
        assert "serif" in nt["css"]


class TestSearchNotesStub:
    def test_returns_stub_message(self, mcp):
        result = mcp("search_notes", {"queries": ["anything"]})
        assert result["results"] == []
        assert "_message" in result

    def test_requires_queries_or_ids(self, mcp):
        result = mcp("search_notes", {})
        assert "error" in result


class TestValidationOverTransport:
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

    def test_update_nonexistent_note(self, mcp):
        result = mcp("upsert_notes", {"notes": [{"id": 9999999999999, "fields": {"Front": "Q"}}]})
        assert result["results"][0]["status"] == "error"

    def test_create_duplicate_note_type(self, mcp):
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
