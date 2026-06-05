from __future__ import annotations

from shrike.note_types import upsert_note_types


class TestCreateNoteType:
    async def test_create_standard(self, wrapper):
        results = upsert_note_types(
            wrapper.col,
            [
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
            ],
        )
        assert len(results) == 1
        assert results[0]["status"] == "created"
        assert results[0]["name"] == "Custom"
        assert isinstance(results[0]["id"], int)

        # Verify it appears in collection_info
        info = await wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Custom"]
        )
        custom = next(nt for nt in info["note_types"] if nt["name"] == "Custom")
        assert custom["fields"] == ["Term", "Definition"]
        assert custom["type"] == "standard"
        assert len(custom["detail"]["templates"]) == 1
        assert custom["detail"]["css"] == ".card { font-size: 20px; }"

    async def test_create_cloze(self, wrapper):
        results = upsert_note_types(
            wrapper.col,
            [
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
            ],
        )
        assert results[0]["status"] == "created"

        info = await wrapper.get_collection_info(include=["note_types"])
        my_cloze = next(nt for nt in info["note_types"] if nt["name"] == "My Cloze")
        assert my_cloze["type"] == "cloze"

    async def test_create_multiple_templates(self, wrapper):
        results = upsert_note_types(
            wrapper.col,
            [
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
            ],
        )
        assert results[0]["status"] == "created"

        info = await wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Vocab"]
        )
        vocab = next(nt for nt in info["note_types"] if nt["name"] == "Vocab")
        assert len(vocab["detail"]["templates"]) == 2

    async def test_create_duplicate_name_fails(self, wrapper):
        results = upsert_note_types(
            wrapper.col,
            [
                {
                    "name": "Basic",
                    "fields": ["A", "B"],
                    "templates": [{"name": "C1", "front": "{{A}}", "back": "{{B}}"}],
                    "css": "",
                }
            ],
        )
        assert results[0]["status"] == "error"
        assert "already exists" in results[0]["error"].lower()

    async def test_create_missing_required_fields(self, wrapper):
        results = upsert_note_types(wrapper.col, [{"name": "Incomplete"}])
        assert results[0]["status"] == "error"

    async def test_can_create_notes_with_new_type(self, wrapper):
        upsert_note_types(
            wrapper.col,
            [
                {
                    "name": "Custom",
                    "fields": ["Q", "A"],
                    "templates": [{"name": "C1", "front": "{{Q}}", "back": "{{A}}"}],
                    "css": "",
                }
            ],
        )
        results = await wrapper.upsert_notes(
            [
                {
                    "deck": "Test",
                    "note_type": "Custom",
                    "fields": {"Q": "question", "A": "answer"},
                }
            ]
        )
        assert results[0]["status"] == "created"


class TestUpdateNoteType:
    def _create_custom_type(self, wrapper):
        results = upsert_note_types(
            wrapper.col,
            [
                {
                    "name": "Editable",
                    "fields": ["Front", "Back"],
                    "templates": [{"name": "Card 1", "front": "{{Front}}", "back": "{{Back}}"}],
                    "css": ".card {}",
                }
            ],
        )
        return results[0]["id"]

    async def test_update_name(self, wrapper):
        nt_id = self._create_custom_type(wrapper)
        results = upsert_note_types(wrapper.col, [{"id": nt_id, "name": "Renamed"}])
        assert results[0]["status"] == "updated"
        assert results[0]["name"] == "Renamed"

    async def test_update_css(self, wrapper):
        nt_id = self._create_custom_type(wrapper)
        upsert_note_types(wrapper.col, [{"id": nt_id, "css": ".card { color: red; }"}])
        info = await wrapper.get_collection_info(
            include=["note_types"], note_type_details=["Editable"]
        )
        editable = next(nt for nt in info["note_types"] if nt["id"] == nt_id)
        assert "color: red" in editable["detail"]["css"]

    async def test_update_nonexistent(self, wrapper):
        results = upsert_note_types(wrapper.col, [{"id": 9999999999, "name": "Nope"}])
        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"].lower()

    async def test_cannot_change_cloze_type(self, wrapper):
        results = upsert_note_types(
            wrapper.col,
            [
                {
                    "name": "StdType",
                    "fields": ["F"],
                    "templates": [{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
                    "css": "",
                }
            ],
        )
        nt_id = results[0]["id"]
        results = upsert_note_types(wrapper.col, [{"id": nt_id, "is_cloze": True}])
        assert results[0]["status"] == "error"
        assert "cannot change" in results[0]["error"].lower()


class TestDeleteNoteType:
    def _create_unused_type(self, wrapper):
        results = upsert_note_types(
            wrapper.col,
            [
                {
                    "name": "Deletable",
                    "fields": ["F"],
                    "templates": [{"name": "C1", "front": "{{F}}", "back": "{{F}}"}],
                    "css": "",
                }
            ],
        )
        return results[0]["id"]

    async def test_delete_unused_type(self, wrapper):
        nt_id = self._create_unused_type(wrapper)
        result = await wrapper.delete_note_types([nt_id])
        assert result["results"][0]["status"] == "deleted"
        assert result["results"][0]["name"] == "Deletable"

        info = await wrapper.get_collection_info(include=["note_types"])
        assert not any(nt["id"] == nt_id for nt in info["note_types"])

    async def test_delete_type_with_notes_fails(self, wrapper, basic_note):
        info = await wrapper.get_collection_info(include=["note_types"])
        basic = next(nt for nt in info["note_types"] if nt["name"] == "Basic")

        result = await wrapper.delete_note_types([basic["id"]])
        assert result["results"][0]["status"] == "error"
        assert "note(s) use this type" in result["results"][0]["error"]

    async def test_delete_nonexistent(self, wrapper):
        result = await wrapper.delete_note_types([9999999999])
        assert result["results"][0]["status"] == "not_found"

    async def test_delete_multiple_mixed(self, wrapper, basic_note):
        nt_id = self._create_unused_type(wrapper)
        info = await wrapper.get_collection_info(include=["note_types"])
        basic_id = next(nt["id"] for nt in info["note_types"] if nt["name"] == "Basic")

        result = await wrapper.delete_note_types([nt_id, basic_id, 9999999999])
        statuses = {r["id"]: r["status"] for r in result["results"]}
        assert statuses[nt_id] == "deleted"
        assert statuses[basic_id] == "error"
        assert statuses[9999999999] == "not_found"


def _type_with_note(wrapper, fields, values, templates=None):
    """Create a note type plus one note carrying `values`; return (mid, nid)."""
    tmpls = templates or [
        {"name": "C", "front": "{{" + fields[0] + "}}", "back": "{{" + fields[-1] + "}}"}
    ]

    def build(c):
        mid = upsert_note_types(
            c, [{"name": "DataSafe", "fields": fields, "templates": tmpls, "css": ""}]
        )[0]["id"]
        note = c.new_note(c.models.get(mid))
        for k, v in values.items():
            note[k] = v
        c.add_note(note, c.decks.id("DataSafe"))
        return mid, note.id

    return wrapper.run_sync(build)


def _content(wrapper, nid):
    return wrapper.run_sync(lambda c: dict(c.get_note(nid).items()))


def _update_type(wrapper, payload):
    return wrapper.run_sync(lambda c: upsert_note_types(c, [payload]))


class TestUpdateNoteTypeFieldsPreserveData:
    """Regression: a whole-list field/template replace must not destroy note data.

    Previously `_update_note_type` rebuilt `flds`/`tmpls` from fresh objects, so
    any update carrying a `fields` key blanked every note's content and any
    `templates` key deleted every card (#76).
    """

    async def test_identical_fields_preserve_data(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        _update_type(wrapper, {"id": mid, "fields": ["Front", "Back"]})
        assert _content(wrapper, nid) == {"Front": "Q", "Back": "A"}

    async def test_rename_field_carries_data(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        _update_type(wrapper, {"id": mid, "fields": ["Frente", "Back"]})
        assert _content(wrapper, nid) == {"Frente": "Q", "Back": "A"}

    async def test_add_field_keeps_existing_and_adds_empty(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        _update_type(wrapper, {"id": mid, "fields": ["Front", "Back", "Extra"]})
        assert _content(wrapper, nid) == {"Front": "Q", "Back": "A", "Extra": ""}

    async def test_remove_trailing_field_keeps_rest(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["A", "B", "C"], {"A": "va", "B": "vb", "C": "vc"})
        _update_type(wrapper, {"id": mid, "fields": ["A", "B"]})
        assert _content(wrapper, nid) == {"A": "va", "B": "vb"}

    async def test_identical_templates_keep_cards(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        before = wrapper.run_sync(lambda c: c.find_cards(f"nid:{nid}"))
        _update_type(
            wrapper,
            {"id": mid, "templates": [{"name": "C", "front": "{{Front}}", "back": "{{Back}}"}]},
        )
        after = wrapper.run_sync(lambda c: c.find_cards(f"nid:{nid}"))
        assert after == before  # same cards, scheduling history intact

    async def test_edit_template_body_keeps_cards(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        before = wrapper.run_sync(lambda c: c.find_cards(f"nid:{nid}"))
        _update_type(
            wrapper,
            {"id": mid, "templates": [{"name": "C", "front": "{{Front}}!", "back": "{{Back}}"}]},
        )
        after = wrapper.run_sync(lambda c: c.find_cards(f"nid:{nid}"))
        assert after == before
        # the body change actually took effect
        info = await wrapper.get_collection_info(
            include=["note_types"], note_type_details=["DataSafe"]
        )
        ds = next(nt for nt in info["note_types"] if nt["id"] == mid)
        assert ds["detail"]["templates"][0]["front"] == "{{Front}}!"

    async def test_fields_and_data_round_trip_in_info(self, wrapper):
        mid, _ = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        _update_type(wrapper, {"id": mid, "fields": ["Frente", "Back"]})
        info = await wrapper.get_collection_info(include=["note_types"])
        ds = next(nt for nt in info["note_types"] if nt["id"] == mid)
        assert ds["fields"] == ["Frente", "Back"]
