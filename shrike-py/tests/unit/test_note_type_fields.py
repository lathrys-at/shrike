from __future__ import annotations

import json

import pytest

from tests.unit._native_shims import (
    NoteTypeOpError,
    update_note_type_fields,
    upsert_note_types,
)


def _type_with_note(wrapper, fields, values, templates=None):
    """Create a note type plus one note carrying `values`; return (mid, nid)."""
    tmpls = templates or [
        {"name": "C", "front": "{{" + fields[0] + "}}", "back": "{{" + fields[-1] + "}}"}
    ]

    def build(c):
        mid = upsert_note_types(
            c, [{"name": "DataSafe", "fields": fields, "templates": tmpls, "css": ""}]
        )[0]["id"]
        created = json.loads(
            c.upsert_notes(
                json.dumps([{"note_type": "DataSafe", "deck": "DataSafe", "fields": dict(values)}]),
                "allow",
                False,
            )
        )[0]
        assert created["status"] == "created", created
        return mid, created["id"]

    return wrapper.run_sync(build)


def _content(wrapper, nid):
    def read(c):
        rows = c.note_field_map([nid])
        assert rows, f"note {nid} missing"
        _, names, values = rows[0]
        return dict(zip(names, values, strict=False))

    return wrapper.run_sync(read)


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
    """A whole-list field/template replace must not destroy note data.

    A `_update_note_type` that rebuilt `flds`/`tmpls` from fresh objects would
    blank every note's content on any update carrying a `fields` key and delete
    every card on a `templates` key — this guards against that.
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
        before = wrapper.run_sync(lambda c: c.cards_of_note(nid))
        _update_type(
            wrapper,
            {"id": mid, "templates": [{"name": "C", "front": "{{Front}}", "back": "{{Back}}"}]},
        )
        after = wrapper.run_sync(lambda c: c.cards_of_note(nid))
        assert after == before  # same cards, scheduling history intact

    async def test_edit_template_body_keeps_cards(self, wrapper):
        mid, nid = _type_with_note(wrapper, ["Front", "Back"], {"Front": "Q", "Back": "A"})
        before = wrapper.run_sync(lambda c: c.cards_of_note(nid))
        _update_type(
            wrapper,
            {"id": mid, "templates": [{"name": "C", "front": "{{Front}}!", "back": "{{Back}}"}]},
        )
        after = wrapper.run_sync(lambda c: c.cards_of_note(nid))
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
    """Explicit, identity-based field operations preserve note data."""

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
