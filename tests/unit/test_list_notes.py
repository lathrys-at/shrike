from __future__ import annotations

import time
from datetime import UTC, datetime


class TestListNotes:
    async def test_by_id(self, wrapper, basic_note):
        result = await wrapper.list_notes(ids=[basic_note])
        assert result["total"] == 1
        note = result["notes"][0]
        assert note["id"] == basic_note
        assert note["content"]["Front"] == "What is 2+2?"
        assert note["content"]["Back"] == "4"

    async def test_by_deck(self, wrapper, basic_note):
        result = await wrapper.list_notes(deck="Test")
        assert result["total"] == 1
        assert result["notes"][0]["deck"] == "Test"

    async def test_by_tags(self, wrapper, basic_note):
        result = await wrapper.list_notes(tags=["math"])
        assert result["total"] == 1

    async def test_by_tags_exclude(self, wrapper, basic_note):
        result = await wrapper.list_notes(tags=["-math"], deck="Test")
        assert result["total"] == 0

    async def test_by_note_type(self, wrapper, basic_note):
        result = await wrapper.list_notes(note_type="Basic")
        assert result["total"] == 1

    async def test_no_match(self, wrapper, basic_note):
        result = await wrapper.list_notes(deck="Nonexistent")
        assert result["total"] == 0
        assert result["notes"] == []

    async def test_meta_fields_mode(self, wrapper, basic_note):
        result = await wrapper.list_notes(ids=[basic_note], fields_mode="meta")
        note = result["notes"][0]
        assert "content" not in note
        assert "id" in note
        assert "note_type" in note
        assert "deck" in note
        assert "tags" in note
        assert "modified" in note

    async def test_limit(self, wrapper):
        # Create 5 notes
        notes = [
            {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": f"Q{i}", "Back": f"A{i}"},
            }
            for i in range(5)
        ]
        await wrapper.upsert_notes(notes)

        result = await wrapper.list_notes(deck="Test", limit=3)
        assert result["total"] == 5
        assert len(result["notes"]) == 3
        assert result["limit"] == 3

    async def test_nonexistent_id_skipped(self, wrapper):
        result = await wrapper.list_notes(ids=[9999999999999])
        assert result["total"] == 0
        assert result["notes"] == []

    async def test_requires_at_least_one_filter(self, wrapper):
        result = await wrapper.list_notes()
        assert "error" in result

    async def test_combined_filters(self, wrapper):
        await wrapper.upsert_notes(
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
        result = await wrapper.list_notes(deck="A", tags=["x"])
        assert result["total"] == 1
        assert result["notes"][0]["deck"] == "A"

    async def test_note_has_modified_timestamp(self, wrapper, basic_note):
        result = await wrapper.list_notes(ids=[basic_note])
        note = result["notes"][0]
        assert "modified" in note
        assert "T" in note["modified"]  # ISO 8601 format

    async def test_modified_since_filters_old_notes(self, wrapper):
        await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Old", "Back": "Note"},
                }
            ]
        )
        time.sleep(1)
        cutoff = datetime.now(UTC).isoformat()
        time.sleep(1)
        await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "New", "Back": "Note"},
                }
            ]
        )

        result = await wrapper.list_notes(modified_since=cutoff)
        assert result["total"] == 1
        assert result["notes"][0]["content"]["Front"] == "New"

    async def test_modified_since_no_matches(self, wrapper, basic_note):
        time.sleep(1)
        future = datetime.now(UTC).isoformat()
        result = await wrapper.list_notes(modified_since=future)
        assert result["total"] == 0

    async def test_modified_since_with_deck_filter(self, wrapper):
        await wrapper.upsert_notes(
            [
                {
                    "deck": "A",
                    "note_type": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                },
                {
                    "deck": "B",
                    "note_type": "Basic",
                    "fields": {"Front": "Q2", "Back": "A2"},
                },
            ]
        )
        past = "2000-01-01T00:00:00+00:00"
        result = await wrapper.list_notes(deck="A", modified_since=past)
        assert result["total"] == 1
        assert result["notes"][0]["deck"] == "A"

    async def test_modified_since_naive_datetime(self, wrapper, basic_note):
        past = "2000-01-01T00:00:00"
        result = await wrapper.list_notes(modified_since=past)
        assert result["total"] >= 1

    async def test_ids_combined_with_deck_filter(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "A",
                    "note_type": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                },
                {
                    "deck": "B",
                    "note_type": "Basic",
                    "fields": {"Front": "Q2", "Back": "A2"},
                },
            ]
        )
        id_in_a = results[0]["id"]
        id_in_b = results[1]["id"]

        # Both IDs, but restricted to deck A — only one should match
        result = await wrapper.list_notes(ids=[id_in_a, id_in_b], deck="A")
        assert result["total"] == 1
        assert result["notes"][0]["id"] == id_in_a

    async def test_ids_combined_with_tags_filter(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                    "tags": ["target"],
                },
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q2", "Back": "A2"},
                    "tags": ["other"],
                },
            ]
        )
        id1 = results[0]["id"]
        id2 = results[1]["id"]

        result = await wrapper.list_notes(ids=[id1, id2], tags=["target"])
        assert result["total"] == 1
        assert result["notes"][0]["id"] == id1
