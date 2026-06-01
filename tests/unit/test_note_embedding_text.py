"""Tests for CollectionWrapper.note_texts_for_embedding."""

from __future__ import annotations


class TestNoteTextsForEmbedding:
    async def test_single_note(self, wrapper, basic_note):
        texts = await wrapper.note_texts_for_embedding([basic_note])
        assert len(texts) == 1
        assert "Front: What is 2+2?" in texts[0]
        assert "Back: 4" in texts[0]

    async def test_multiple_notes(self, wrapper):
        results = await wrapper.upsert_notes(
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
        texts = await wrapper.note_texts_for_embedding(ids)
        assert len(texts) == 2
        assert "Q1" in texts[0]
        assert "Q2" in texts[1]

    async def test_nonexistent_note_returns_empty_string(self, wrapper):
        texts = await wrapper.note_texts_for_embedding([9999999999999])
        assert texts == [""]

    async def test_mixed_existing_and_nonexistent(self, wrapper, basic_note):
        texts = await wrapper.note_texts_for_embedding([basic_note, 9999999999999])
        assert len(texts) == 2
        assert texts[0] != ""
        assert texts[1] == ""

    async def test_empty_ids(self, wrapper):
        texts = await wrapper.note_texts_for_embedding([])
        assert texts == []

    async def test_skips_empty_fields(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "Only front", "Back": ""},
                },
            ]
        )
        nid = results[0]["id"]
        texts = await wrapper.note_texts_for_embedding([nid])
        assert "Front: Only front" in texts[0]
        assert "Back:" not in texts[0]


class TestNormalizationThroughWrapper:
    """The rendered/cleaned text reaches the embedder, not raw markup."""

    async def test_cloze_field_is_filled(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Cloze",
                    "fields": {"Text": "The capital of {{c1::France}} is Paris."},
                }
            ]
        )
        (text,) = await wrapper.note_texts_for_embedding([results[0]["id"]])
        assert "Text: The capital of France is Paris." in text
        assert "{{c1::" not in text  # template syntax must not leak through

    async def test_html_and_media_stripped(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {
                        "Front": "<b>Hello</b>&nbsp;world",
                        "Back": 'see<br>this <img src="x.png">[sound:y.mp3]',
                    },
                }
            ]
        )
        (text,) = await wrapper.note_texts_for_embedding([results[0]["id"]])
        assert "Front: Hello world" in text
        assert "Back: see this" in text
        assert "<" not in text and "[sound:" not in text and "&nbsp;" not in text
