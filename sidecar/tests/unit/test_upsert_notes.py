from __future__ import annotations


class TestCreateNotes:
    def test_create_basic(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert len(results) == 1
        assert results[0]["status"] == "created"
        assert isinstance(results[0]["id"], int)

    def test_create_with_tags(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                    "tags": ["tag1", "tag2"],
                }
            ]
        )
        nid = results[0]["id"]
        note = wrapper.list_notes(ids=[nid])["notes"][0]
        assert set(note["tags"]) == {"tag1", "tag2"}

    def test_create_deck_auto_created(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "New::Nested::Deck",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert results[0]["status"] == "created"
        note = wrapper.list_notes(ids=[results[0]["id"]])["notes"][0]
        assert note["deck"] == "New::Nested::Deck"

    def test_create_missing_note_type(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Nonexistent",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"].lower()

    def test_create_missing_deck(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Back": "A"},
                }
            ]
        )
        assert results[0]["status"] == "error"

    def test_create_missing_fields(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                }
            ]
        )
        assert results[0]["status"] == "error"

    def test_create_invalid_field_name(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q", "Wrong": "A"},
                }
            ]
        )
        assert results[0]["status"] == "error"
        assert "Wrong" in results[0]["error"]

    def test_create_bulk(self, wrapper):
        notes = [
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
            }
            for i in range(10)
        ]
        results = wrapper.upsert_notes(notes)
        assert len(results) == 10
        assert all(r["status"] == "created" for r in results)

    def test_partial_failure(self, wrapper):
        """One bad note in a batch should not block the others."""
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                },
                {
                    "deck": "Test",
                    "note_type": "Nonexistent",
                    "fields": {"Front": "Q2", "Back": "A2"},
                },
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q3", "Back": "A3"},
                },
            ]
        )
        assert results[0]["status"] == "created"
        assert results[1]["status"] == "error"
        assert results[2]["status"] == "created"


class TestUpdateNotes:
    def test_update_fields(self, wrapper, basic_note):
        results = wrapper.upsert_notes([{"id": basic_note, "fields": {"Back": "Four"}}])
        assert results[0]["status"] == "updated"

        note = wrapper.list_notes(ids=[basic_note])["notes"][0]
        assert note["content"]["Back"] == "Four"
        assert note["content"]["Front"] == "What is 2+2?"  # unchanged

    def test_update_tags(self, wrapper, basic_note):
        results = wrapper.upsert_notes([{"id": basic_note, "tags": ["new-tag"]}])
        assert results[0]["status"] == "updated"

        note = wrapper.list_notes(ids=[basic_note])["notes"][0]
        assert note["tags"] == ["new-tag"]

    def test_update_move_deck(self, wrapper, basic_note):
        results = wrapper.upsert_notes([{"id": basic_note, "deck": "Other"}])
        assert results[0]["status"] == "updated"

        note = wrapper.list_notes(ids=[basic_note])["notes"][0]
        assert note["deck"] == "Other"

    def test_update_nonexistent_note(self, wrapper):
        results = wrapper.upsert_notes([{"id": 9999999999999, "fields": {"Front": "Q"}}])
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"].lower()

    def test_update_cannot_change_note_type(self, wrapper, basic_note):
        results = wrapper.upsert_notes([{"id": basic_note, "note_type": "Cloze"}])
        assert results[0]["status"] == "error"
        assert "cannot change" in results[0]["error"].lower()

    def test_update_invalid_field_name(self, wrapper, basic_note):
        results = wrapper.upsert_notes([{"id": basic_note, "fields": {"Nonexistent": "value"}}])
        assert results[0]["status"] == "error"
