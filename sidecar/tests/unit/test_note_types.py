from __future__ import annotations

from shrike.note_types import upsert_note_types


class TestCreateNoteType:
    def test_create_standard(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {
                "name": "Custom",
                "fields": ["Term", "Definition"],
                "templates": [
                    {
                        "name": "Card 1",
                        "front": "{{Term}}",
                        "back": "{{FrontSide}}<hr>{{Definition}}",
                    }
                ],
                "css": ".card { font-size: 20px; }",
            }
        ])
        assert len(results) == 1
        assert results[0]["status"] == "created"
        assert results[0]["name"] == "Custom"
        assert isinstance(results[0]["id"], int)

        # Verify it appears in collection_info
        info = wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Custom"]
        )
        custom = next(nt for nt in info["note_types"] if nt["name"] == "Custom")
        assert custom["fields"] == ["Term", "Definition"]
        assert custom["type"] == "standard"
        assert len(custom["templates"]) == 1
        assert custom["css"] == ".card { font-size: 20px; }"

    def test_create_cloze(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {
                "name": "My Cloze",
                "fields": ["Text", "Extra"],
                "is_cloze": True,
                "templates": [
                    {
                        "name": "Cloze",
                        "front": "{{cloze:Text}}",
                        "back": "{{cloze:Text}}<br>{{Extra}}",
                    }
                ],
                "css": "",
            }
        ])
        assert results[0]["status"] == "created"

        info = wrapper.get_collection_info(include=["note_types"])
        my_cloze = next(nt for nt in info["note_types"] if nt["name"] == "My Cloze")
        assert my_cloze["type"] == "cloze"

    def test_create_multiple_templates(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {
                "name": "Vocab",
                "fields": ["Word", "Meaning"],
                "templates": [
                    {
                        "name": "Recognition",
                        "front": "{{Word}}",
                        "back": "{{FrontSide}}<hr>{{Meaning}}",
                    },
                    {
                        "name": "Recall",
                        "front": "{{Meaning}}",
                        "back": "{{FrontSide}}<hr>{{Word}}",
                    },
                ],
                "css": "",
            }
        ])
        assert results[0]["status"] == "created"

        info = wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Vocab"]
        )
        vocab = next(nt for nt in info["note_types"] if nt["name"] == "Vocab")
        assert len(vocab["templates"]) == 2

    def test_create_duplicate_name_fails(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {
                "name": "Basic",
                "fields": ["A", "B"],
                "templates": [{"name": "C1", "front": "{{A}}", "back": "{{B}}"}],
                "css": "",
            }
        ])
        assert results[0]["status"] == "error"
        assert "already exists" in results[0]["error"].lower()

    def test_create_missing_required_fields(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {"name": "Incomplete"}
        ])
        assert results[0]["status"] == "error"

    def test_can_create_notes_with_new_type(self, wrapper):
        upsert_note_types(wrapper.col, [
            {
                "name": "Custom",
                "fields": ["Q", "A"],
                "templates": [
                    {"name": "C1", "front": "{{Q}}", "back": "{{A}}"}
                ],
                "css": "",
            }
        ])
        results = wrapper.upsert_notes([
            {
                "deck": "Test",
                "note_type": "Custom",
                "fields": {"Q": "question", "A": "answer"},
            }
        ])
        assert results[0]["status"] == "created"


class TestUpdateNoteType:
    def _create_custom_type(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {
                "name": "Editable",
                "fields": ["Front", "Back"],
                "templates": [
                    {"name": "Card 1", "front": "{{Front}}", "back": "{{Back}}"}
                ],
                "css": ".card {}",
            }
        ])
        return results[0]["id"]

    def test_update_name(self, wrapper):
        nt_id = self._create_custom_type(wrapper)
        results = upsert_note_types(wrapper.col, [
            {"id": nt_id, "name": "Renamed"}
        ])
        assert results[0]["status"] == "updated"
        assert results[0]["name"] == "Renamed"

    def test_update_css(self, wrapper):
        nt_id = self._create_custom_type(wrapper)
        upsert_note_types(wrapper.col, [
            {"id": nt_id, "css": ".card { color: red; }"}
        ])
        info = wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Editable"]
        )
        editable = next(nt for nt in info["note_types"] if nt["id"] == nt_id)
        assert "color: red" in editable["css"]

    def test_update_nonexistent(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {"id": 9999999999, "name": "Nope"}
        ])
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"].lower()

    def test_cannot_change_cloze_type(self, wrapper):
        results = upsert_note_types(wrapper.col, [
            {
                "name": "StdType",
                "fields": ["F"],
                "templates": [{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
                "css": "",
            }
        ])
        nt_id = results[0]["id"]
        results = upsert_note_types(wrapper.col, [
            {"id": nt_id, "is_cloze": True}
        ])
        assert results[0]["status"] == "error"
        assert "cannot change" in results[0]["error"].lower()
