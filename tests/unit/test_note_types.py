from __future__ import annotations

import pytest

from shrike.note_types import (
    NoteTypeOpError,
    find_and_replace_in_note_type,
    update_note_type_fields,
    update_note_type_templates,
    upsert_note_types,
)


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


class TestUpsertFieldReplaceRejectsUnsound:
    """upsert_note_types' positional field replace refuses moves/inserts/
    non-trailing removes — they would silently mislabel note data — and points
    the caller at update_note_type_fields. Rename/append/trailing-remove stay
    allowed (covered by TestUpdateNoteTypeFieldsPreserveData)."""

    async def test_reorder_rejected_and_data_untouched(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["A", "B", "C"], {"A": "va", "B": "vb", "C": "vc"})
        res = _update_type(wrapper, {"id": mid, "fields": ["C", "A", "B"]})
        assert res[0]["status"] == "error"
        assert "update_note_type_fields" in res[0]["error"]
        assert _content(wrapper, nid) == {"A": "va", "B": "vb", "C": "vc"}

    async def test_swap_rejected(self, wrapper):
        mid, _ = _type_with_note(wrapper, ["A", "B"], {"A": "va", "B": "vb"})
        res = _update_type(wrapper, {"id": mid, "fields": ["B", "A"]})
        assert res[0]["status"] == "error"
        assert "update_note_type_fields" in res[0]["error"]

    async def test_remove_non_trailing_rejected(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["A", "B", "C"], {"A": "va", "B": "vb", "C": "vc"})
        res = _update_type(wrapper, {"id": mid, "fields": ["A", "C"]})
        assert res[0]["status"] == "error"
        assert "update_note_type_fields" in res[0]["error"]
        assert _content(wrapper, nid) == {"A": "va", "B": "vb", "C": "vc"}

    async def test_insert_before_rejected(self, wrapper):
        mid, _ = _type_with_note(wrapper, ["A", "B"], {"A": "va", "B": "vb"})
        res = _update_type(wrapper, {"id": mid, "fields": ["A", "X", "B"]})
        assert res[0]["status"] == "error"
        assert "update_note_type_fields" in res[0]["error"]

    async def test_rename_plus_append_still_allowed(self, wrapper):
        # A rename in place combined with an append shifts no existing field, so
        # it stays a sound positional replace.
        mid, nid = _type_with_note(wrapper, ["A", "B"], {"A": "va", "B": "vb"})
        res = _update_type(wrapper, {"id": mid, "fields": ["Aa", "B", "C"]})
        assert res[0]["status"] == "updated"
        assert _content(wrapper, nid) == {"Aa": "va", "B": "vb", "C": ""}


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


def _apply_ops(wrapper, name, ops):
    return wrapper.run_sync(lambda c: update_note_type_fields(c, name, ops))


class TestUpdateNoteTypeFields:
    """Explicit, identity-based field operations (#76) preserve note data."""

    async def test_rename_carries_data(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        res = _apply_ops(wrapper, "DataSafe", [{"op": "rename", "name": "Front", "new_name": "Q"}])
        assert res == {"id": mid, "name": "DataSafe", "fields": ["Q", "Back"]}
        assert _content(wrapper, nid) == {"Q": "Q", "Back": "A"}

    async def test_reposition_is_a_true_move(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["A", "B", "C"], {"A": "va", "B": "vb", "C": "vc"})
        res = _apply_ops(wrapper, "DataSafe", [{"op": "reposition", "name": "C", "position": 0}])
        assert res["fields"] == ["C", "A", "B"]
        # data moves with the field, not with the position
        assert _content(wrapper, nid) == {"C": "vc", "A": "va", "B": "vb"}

    async def test_remove_non_trailing_field(self, wrapper):
        _mid, nid = _type_with_note(wrapper, ["A", "B", "C"], {"A": "va", "B": "vb", "C": "vc"})
        res = _apply_ops(wrapper, "DataSafe", [{"op": "remove", "name": "A"}])
        assert res["fields"] == ["B", "C"]
        assert _content(wrapper, nid) == {"B": "vb", "C": "vc"}

    async def test_add_at_position(self, wrapper):
        _mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        res = _apply_ops(wrapper, "DataSafe", [{"op": "add", "name": "Hint", "position": 1}])
        assert res["fields"] == ["Front", "Hint", "Back"]
        assert _content(wrapper, nid) == {"Front": "Q", "Hint": "", "Back": "A"}

    async def test_add_appends_when_no_position(self, wrapper):
        mid, _nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        res = _apply_ops(wrapper, "DataSafe", [{"op": "add", "name": "Extra"}])
        assert res["fields"] == ["Front", "Back", "Extra"]

    async def test_sequence_applies_in_order(self, wrapper):
        _mid, nid = _type_with_note(wrapper, ["A", "B"], {"A": "va", "B": "vb"})
        # rename A->Aa, then reposition the now-named Aa to the end, then add C
        res = _apply_ops(
            wrapper,
            "DataSafe",
            [
                {"op": "rename", "name": "A", "new_name": "Aa"},
                {"op": "reposition", "name": "Aa", "position": 1},
                {"op": "add", "name": "C"},
            ],
        )
        assert res["fields"] == ["B", "Aa", "C"]
        assert _content(wrapper, nid) == {"B": "vb", "Aa": "va", "C": ""}

    async def test_invalid_op_is_atomic(self, wrapper):
        _mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        # second op is invalid (renaming a nonexistent field); nothing should apply
        with pytest.raises(NoteTypeOpError, match="not found"):
            _apply_ops(
                wrapper,
                "DataSafe",
                [
                    {"op": "rename", "name": "Front", "new_name": "Frente"},
                    {"op": "rename", "name": "Nope", "new_name": "X"},
                ],
            )
        # untouched: original field names and data intact
        info = await wrapper.get_collection_info(include=["note_types"])
        ds = next(nt for nt in info["note_types"] if nt["name"] == "DataSafe")
        assert ds["fields"] == ["Front", "Back"]
        assert _content(wrapper, nid) == {"Front": "Q", "Back": "A"}

    async def test_rename_to_existing_name_rejected(self, wrapper):
        _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        with pytest.raises(NoteTypeOpError, match="already exists"):
            _apply_ops(wrapper, "DataSafe", [{"op": "rename", "name": "Front", "new_name": "Back"}])

    async def test_add_duplicate_name_rejected(self, wrapper):
        _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        with pytest.raises(NoteTypeOpError, match="already exists"):
            _apply_ops(wrapper, "DataSafe", [{"op": "add", "name": "Front"}])

    async def test_remove_last_field_rejected(self, wrapper):
        _type_with_note(wrapper, ["Only"], {"Only": "v"})
        with pytest.raises(NoteTypeOpError, match="at least one"):
            _apply_ops(wrapper, "DataSafe", [{"op": "remove", "name": "Only"}])

    async def test_reposition_out_of_range_rejected(self, wrapper):
        _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        with pytest.raises(NoteTypeOpError, match="out of range"):
            _apply_ops(wrapper, "DataSafe", [{"op": "reposition", "name": "Front", "position": 9}])

    async def test_unknown_note_type_rejected(self, wrapper):
        with pytest.raises(NoteTypeOpError, match="not found"):
            _apply_ops(wrapper, "Nonexistent", [{"op": "add", "name": "X"}])


def _apply_template_ops(wrapper, name, ops):
    return wrapper.run_sync(lambda c: update_note_type_templates(c, name, ops))


def _type_with_cards(wrapper, template_names, field="F"):
    """Create a note type with N always-rendering templates and one note (N cards)."""
    tmpls = [
        {"name": t, "front": t + " {{" + field + "}}", "back": "{{" + field + "}}"}
        for t in template_names
    ]

    def build(c):
        mid = upsert_note_types(
            c, [{"name": "TmplSafe", "fields": [field], "templates": tmpls, "css": ""}]
        )[0]["id"]
        note = c.new_note(c.models.get(mid))
        note[field] = "x"
        c.add_note(note, c.decks.id("TmplSafe"))
        return mid, note.id

    return wrapper.run_sync(build)


def _cards_by_template(wrapper, nid, mid):
    """Map each of the note's cards to its template name -> card id."""

    def q(c):
        out = {}
        for cid in c.find_cards(f"nid:{nid}"):
            ord_ = c.get_card(cid).ord
            out[c.models.get(mid)["tmpls"][ord_]["name"]] = cid
        return out

    return wrapper.run_sync(q)


class TestUpdateNoteTypeTemplates:
    """Identity-based template operations (#76) preserve cards by template."""

    async def test_rename_keeps_cards(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta", "Tb"])
        before = _cards_by_template(wrapper, nid, mid)
        res = _apply_template_ops(
            wrapper, "TmplSafe", [{"op": "rename", "name": "Ta", "new_name": "Recall"}]
        )
        assert res == {"id": mid, "name": "TmplSafe", "templates": ["Recall", "Tb"]}
        after = _cards_by_template(wrapper, nid, mid)
        # same card ids; only the template label changed (Ta -> Recall)
        assert after == {"Recall": before["Ta"], "Tb": before["Tb"]}

    async def test_reposition_is_a_true_move(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta", "Tb", "Tc"])
        before = _cards_by_template(wrapper, nid, mid)
        res = _apply_template_ops(
            wrapper, "TmplSafe", [{"op": "reposition", "name": "Tc", "position": 0}]
        )
        assert res["templates"] == ["Tc", "Ta", "Tb"]
        # each card stays with its template (same ids), just reordered
        assert _cards_by_template(wrapper, nid, mid) == before

    async def test_remove_non_trailing_keeps_other_cards(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta", "Tb", "Tc"])
        before = _cards_by_template(wrapper, nid, mid)
        res = _apply_template_ops(wrapper, "TmplSafe", [{"op": "remove", "name": "Tb"}])
        assert res["templates"] == ["Ta", "Tc"]
        after = _cards_by_template(wrapper, nid, mid)
        assert after == {"Ta": before["Ta"], "Tc": before["Tc"]}  # only Tb's card gone

    async def test_add_generates_a_card(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta"])
        res = _apply_template_ops(
            wrapper,
            "TmplSafe",
            [{"op": "add", "name": "Tb", "front": "Tb {{F}}", "back": "{{F}}"}],
        )
        assert res["templates"] == ["Ta", "Tb"]
        assert set(_cards_by_template(wrapper, nid, mid)) == {"Ta", "Tb"}

    async def test_add_at_position(self, wrapper):
        _mid, _nid = _type_with_cards(wrapper, ["Ta", "Tb"])
        res = _apply_template_ops(
            wrapper,
            "TmplSafe",
            [{"op": "add", "name": "Mid", "front": "Mid {{F}}", "back": "{{F}}", "position": 1}],
        )
        assert res["templates"] == ["Ta", "Mid", "Tb"]

    async def test_sequence_applies_in_order(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta", "Tb"])
        before = _cards_by_template(wrapper, nid, mid)
        res = _apply_template_ops(
            wrapper,
            "TmplSafe",
            [
                {"op": "rename", "name": "Ta", "new_name": "Aa"},
                {"op": "reposition", "name": "Aa", "position": 1},
            ],
        )
        assert res["templates"] == ["Tb", "Aa"]
        assert _cards_by_template(wrapper, nid, mid) == {"Tb": before["Tb"], "Aa": before["Ta"]}

    async def test_invalid_op_is_atomic(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta", "Tb"])
        before = _cards_by_template(wrapper, nid, mid)
        with pytest.raises(NoteTypeOpError, match="not found"):
            _apply_template_ops(
                wrapper,
                "TmplSafe",
                [
                    {"op": "rename", "name": "Ta", "new_name": "Aa"},
                    {"op": "remove", "name": "Nope"},
                ],
            )
        info = await wrapper.get_collection_info(
            include=["note_types"], note_type_details=["TmplSafe"]
        )
        ts = next(nt for nt in info["note_types"] if nt["name"] == "TmplSafe")
        assert [t["name"] for t in ts["detail"]["templates"]] == ["Ta", "Tb"]
        assert _cards_by_template(wrapper, nid, mid) == before

    async def test_remove_last_template_rejected(self, wrapper):
        _type_with_cards(wrapper, ["Only"])
        with pytest.raises(NoteTypeOpError, match="at least one"):
            _apply_template_ops(wrapper, "TmplSafe", [{"op": "remove", "name": "Only"}])

    async def test_rename_to_existing_rejected(self, wrapper):
        _type_with_cards(wrapper, ["Ta", "Tb"])
        with pytest.raises(NoteTypeOpError, match="already exists"):
            _apply_template_ops(
                wrapper, "TmplSafe", [{"op": "rename", "name": "Ta", "new_name": "Tb"}]
            )

    async def test_unknown_note_type_rejected(self, wrapper):
        with pytest.raises(NoteTypeOpError, match="not found"):
            _apply_template_ops(
                wrapper, "Nonexistent", [{"op": "rename", "name": "x", "new_name": "y"}]
            )


class TestUpsertTemplateReplaceRejectsUnsound:
    """upsert_note_types' positional template replace refuses reorders/inserts/
    non-trailing removes (they'd re-label cards) and redirects to
    update_note_type_templates. In-place edits stay allowed."""

    async def test_reorder_rejected(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta", "Tb"])
        before = _cards_by_template(wrapper, nid, mid)
        res = _update_type(
            wrapper,
            {
                "id": mid,
                "templates": [
                    {"name": "Tb", "front": "Tb {{F}}", "back": "{{F}}"},
                    {"name": "Ta", "front": "Ta {{F}}", "back": "{{F}}"},
                ],
            },
        )
        assert res[0]["status"] == "error"
        assert "update_note_type_templates" in res[0]["error"]
        assert _cards_by_template(wrapper, nid, mid) == before  # untouched

    async def test_in_place_html_edit_allowed(self, wrapper):
        mid, nid = _type_with_cards(wrapper, ["Ta", "Tb"])
        before = _cards_by_template(wrapper, nid, mid)
        res = _update_type(
            wrapper,
            {
                "id": mid,
                "templates": [
                    {"name": "Ta", "front": "Ta! {{F}}", "back": "{{F}}"},
                    {"name": "Tb", "front": "Tb {{F}}", "back": "{{F}}"},
                ],
            },
        )
        assert res[0]["status"] == "updated"
        assert _cards_by_template(wrapper, nid, mid) == before  # cards preserved
        info = await wrapper.get_collection_info(
            include=["note_types"], note_type_details=["TmplSafe"]
        )
        ts = next(nt for nt in info["note_types"] if nt["name"] == "TmplSafe")
        assert ts["detail"]["templates"][0]["front"] == "Ta! {{F}}"


def _make_model(wrapper, *, templates, css, fields=("F",)):
    def build(c):
        res = upsert_note_types(
            c, [{"name": "FR", "fields": list(fields), "templates": templates, "css": css}]
        )
        assert res[0]["status"] != "error", res[0].get("error")
        return res[0]["id"]

    return wrapper.run_sync(build)


async def _detail(wrapper):
    info = await wrapper.get_collection_info(include=["note_types"], note_type_details=["FR"])
    nt = next(n for n in info["note_types"] if n["name"] == "FR")
    return nt["detail"]


def _replace(wrapper, **kw):
    return wrapper.run_sync(lambda c: find_and_replace_in_note_type(c, "FR", **kw))


class TestFindAndReplaceInNoteType:
    """find_and_replace_in_note_type rewrites template HTML and CSS in place,
    scoped by the front/back/css selectors, returning a replacement count. No
    note field values are involved."""

    async def test_replace_in_front_and_back(self, wrapper):
        # Old and New are both real fields (Anki rejects a template that
        # references a missing field, at create *and* on save).
        _make_model(
            wrapper,
            fields=("Old", "New"),
            templates=[{"name": "C", "front": "{{Old}}", "back": "see {{Old}} again"}],
            css="",
        )
        res = _replace(wrapper, search="{{Old}}", replacement="{{New}}")
        assert res["replacements"] == 2
        assert res["templates_changed"] == ["C"]
        assert res["css_changed"] is False
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "{{New}}"
        assert detail["templates"][0]["back"] == "see {{New}} again"

    async def test_replace_in_css(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
            css=".card { color: red; }",
        )
        res = _replace(wrapper, search="red", replacement="blue", front=False, back=False)
        assert res["replacements"] == 1
        assert res["templates_changed"] == []
        assert res["css_changed"] is True
        detail = await _detail(wrapper)
        assert "color: blue" in detail["css"]

    async def test_selectors_scope_the_search(self, wrapper):
        # "z" appears on front, back, and css; only the front is selected.
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "z {{F}}", "back": "z"}],
            css=".card { /* z */ }",
        )
        res = _replace(wrapper, search="z", replacement="y", back=False, css=False)
        assert res["replacements"] == 1
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "y {{F}}"
        assert detail["templates"][0]["back"] == "z"
        assert "/* z */" in detail["css"]

    async def test_match_case_default_is_sensitive(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "Red red RED {{F}}", "back": "{{F}}"}],
            css="",
        )
        res = _replace(wrapper, search="red", replacement="blue")
        assert res["replacements"] == 1  # only the exact-case "red"
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "Red blue RED {{F}}"

    async def test_match_case_false(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "Red red RED {{F}}", "back": "{{F}}"}],
            css="",
        )
        res = _replace(wrapper, search="red", replacement="x", match_case=False)
        assert res["replacements"] == 3

    async def test_literal_replacement_is_not_group_ref(self, wrapper):
        # A literal replace containing "\1" must be inserted verbatim, not as a
        # regex backreference.
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "A {{F}}", "back": "{{F}}"}],
            css="",
        )
        _replace(wrapper, search="A", replacement=r"\1B")
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == r"\1B {{F}}"

    async def test_regex_with_capture_group(self, wrapper):
        _make_model(
            wrapper,
            fields=("Old", "F"),
            templates=[{"name": "C", "front": "{{Old}} {{F}}", "back": "{{Old}}"}],
            css="",
        )
        # Match the Old field ref only, leaving {{F}} on the front intact (so the
        # front keeps a field reference and stays valid).
        res = _replace(wrapper, search=r"\{\{(Old)\}\}", replacement=r"[\1]", regex=True)
        assert res["replacements"] == 2
        detail = await _detail(wrapper)
        assert detail["templates"][0]["front"] == "[Old] {{F}}"
        assert detail["templates"][0]["back"] == "[Old]"

    async def test_multiple_templates(self, wrapper):
        _make_model(
            wrapper,
            templates=[
                {"name": "C1", "front": "z {{F}}", "back": "{{F}}"},
                {"name": "C2", "front": "{{F}} two", "back": "z {{F}}"},
                {"name": "C3", "front": "{{F}} three", "back": "{{F}}"},
            ],
            css="",
        )
        res = _replace(wrapper, search="z", replacement="q")
        assert res["replacements"] == 2
        assert res["templates_changed"] == ["C1", "C2"]

    async def test_no_match_makes_no_change(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
            css=".card {}",
        )
        res = _replace(wrapper, search="absent", replacement="x")
        assert res["replacements"] == 0
        assert res["templates_changed"] == []
        assert res["css_changed"] is False

    async def test_unknown_note_type_raises(self, wrapper):
        with pytest.raises(NoteTypeOpError, match="not found"):
            wrapper.run_sync(
                lambda c: find_and_replace_in_note_type(c, "Nope", search="a", replacement="b")
            )

    async def test_invalid_regex_raises(self, wrapper):
        _make_model(
            wrapper,
            templates=[{"name": "C", "front": "{{F}}", "back": "{{F}}"}],
            css="",
        )
        with pytest.raises(NoteTypeOpError, match="invalid regex"):
            _replace(wrapper, search="(unclosed", replacement="x", regex=True)
