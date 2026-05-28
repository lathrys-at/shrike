"""Tests for CollectionWrapper.note_texts_for_embedding."""

from __future__ import annotations


class TestNoteTextsForEmbedding:
    def test_single_note(self, wrapper, basic_note):
        texts = wrapper.note_texts_for_embedding([basic_note])
        assert len(texts) == 1
        assert "Front: What is 2+2?" in texts[0]
        assert "Back: 4" in texts[0]

    def test_multiple_notes(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                },
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Q2", "Back": "A2"},
                },
            ]
        )
        ids = [r["id"] for r in results]
        texts = wrapper.note_texts_for_embedding(ids)
        assert len(texts) == 2
        assert "Q1" in texts[0]
        assert "Q2" in texts[1]

    def test_nonexistent_note_returns_empty_string(self, wrapper):
        texts = wrapper.note_texts_for_embedding([9999999999999])
        assert texts == [""]

    def test_mixed_existing_and_nonexistent(self, wrapper, basic_note):
        texts = wrapper.note_texts_for_embedding([basic_note, 9999999999999])
        assert len(texts) == 2
        assert texts[0] != ""
        assert texts[1] == ""

    def test_empty_ids(self, wrapper):
        texts = wrapper.note_texts_for_embedding([])
        assert texts == []

    def test_skips_empty_fields(self, wrapper):
        results = wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Only front", "Back": ""},
                },
            ]
        )
        nid = results[0]["id"]
        texts = wrapper.note_texts_for_embedding([nid])
        assert "Front: Only front" in texts[0]
        assert "Back:" not in texts[0]
