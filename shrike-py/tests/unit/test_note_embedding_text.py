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


class TestExtractImageRefs:
    """embed_text.extract_image_refs — pull <img src> filenames for multimodal embedding (#162)."""

    def test_basic_quoted(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img src="cat.png">') == ["cat.png"]

    def test_unquoted_and_single_quoted(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs("<img src=a.png> x <img src='b.gif'>") == ["a.png", "b.gif"]

    def test_dedup_in_order(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img src="x.png"> <img src="y.png"> <img src="x.png">') == [
            "x.png",
            "y.png",
        ]

    def test_basename_only(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img src="sub/dir/d.jpg">') == ["d.jpg"]

    def test_remote_url_skipped(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img src="https://e.com/r.png">') == []

    def test_other_attributes(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img alt="a cat" src="c.png" width="10">') == ["c.png"]

    def test_html_entity_in_name(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img src="a&amp;b.png">') == ["a&b.png"]

    def test_no_image(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs("plain <b>text</b> [sound:x.mp3]") == []

    def test_data_src_decoy_ignored(self):
        # A real parser must not grab the earlier data-src= as the tag's src (regex did, #213).
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img data-src="decoy.png" src="real.png">') == ["real.png"]

    def test_src_inside_other_attribute_value_ignored(self):
        from tests.oracles.embed_text_oracle import extract_image_refs

        assert extract_image_refs('<img alt="x src=y" src="real.png">') == ["real.png"]


class TestNoteEmbedInputs:
    """CollectionWrapper.note_embed_inputs — text + image names per note (#162)."""

    async def test_text_only_note(self, wrapper, basic_note):
        inputs = await wrapper.note_embed_inputs([basic_note])
        assert len(inputs) == 1
        assert inputs[0].note_id == basic_note
        assert "What is 2+2?" in inputs[0].text
        assert inputs[0].image_names == []

    async def test_note_with_image(self, wrapper):
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": 'see <img src="diagram.png"> here', "Back": "answer"},
                }
            ]
        )
        nid = results[0]["id"]
        inputs = await wrapper.note_embed_inputs([nid])
        assert inputs[0].image_names == ["diagram.png"]
        # The <img> is stripped from the embedding text but the name is captured.
        assert "see here" in inputs[0].text
        assert "diagram.png" not in inputs[0].text

    async def test_nonexistent_note(self, wrapper):
        inputs = await wrapper.note_embed_inputs([9999999999999])
        assert len(inputs) == 1
        assert inputs[0].text == "" and inputs[0].image_names == []
