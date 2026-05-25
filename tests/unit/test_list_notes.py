from __future__ import annotations


class TestListNotes:
    def test_by_id(self, wrapper, basic_note):
        result = wrapper.list_notes(ids=[basic_note])
        assert result["total"] == 1
        note = result["notes"][0]
        assert note["id"] == basic_note
        assert note["content"]["Front"] == "What is 2+2?"
        assert note["content"]["Back"] == "4"

    def test_by_deck(self, wrapper, basic_note):
        result = wrapper.list_notes(deck="Test")
        assert result["total"] == 1
        assert result["notes"][0]["deck"] == "Test"

    def test_by_tags(self, wrapper, basic_note):
        result = wrapper.list_notes(tags=["math"])
        assert result["total"] == 1

    def test_by_tags_exclude(self, wrapper, basic_note):
        result = wrapper.list_notes(tags=["-math"], deck="Test")
        assert result["total"] == 0

    def test_by_note_type(self, wrapper, basic_note):
        result = wrapper.list_notes(note_type="Basic")
        assert result["total"] == 1

    def test_no_match(self, wrapper, basic_note):
        result = wrapper.list_notes(deck="Nonexistent")
        assert result["total"] == 0
        assert result["notes"] == []

    def test_meta_fields_mode(self, wrapper, basic_note):
        result = wrapper.list_notes(ids=[basic_note], fields_mode="meta")
        note = result["notes"][0]
        assert "content" not in note
        assert "id" in note
        assert "note_type" in note
        assert "deck" in note
        assert "tags" in note
        assert "modified" in note

    def test_limit(self, wrapper):
        # Create 5 notes
        notes = [
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
            }
            for i in range(5)
        ]
        wrapper.upsert_notes(notes)

        result = wrapper.list_notes(deck="Test", limit=3)
        assert result["total"] == 5
        assert len(result["notes"]) == 3
        assert result["limit"] == 3

    def test_nonexistent_id_skipped(self, wrapper):
        result = wrapper.list_notes(ids=[9999999999999])
        assert result["total"] == 0
        assert result["notes"] == []

    def test_requires_at_least_one_filter(self, wrapper):
        result = wrapper.list_notes()
        assert "error" in result

    def test_combined_filters(self, wrapper):
        wrapper.upsert_notes(
            [
                {
                    "deck": "A",
                    "note_type": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                    "tags": ["x"],
                },
                {
                    "deck": "B",
                    "note_type": "Basic",
                    "fields": {"Front": "Q2", "Back": "A2"},
                    "tags": ["x"],
                },
            ]
        )
        result = wrapper.list_notes(deck="A", tags=["x"])
        assert result["total"] == 1
        assert result["notes"][0]["deck"] == "A"

    def test_note_has_modified_timestamp(self, wrapper, basic_note):
        result = wrapper.list_notes(ids=[basic_note])
        note = result["notes"][0]
        assert "modified" in note
        assert "T" in note["modified"]  # ISO 8601 format
