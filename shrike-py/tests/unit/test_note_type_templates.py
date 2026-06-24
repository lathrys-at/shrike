from __future__ import annotations

import json

import pytest

from tests.unit._native_shims import (
    NoteTypeOpError,
    update_note_type_templates,
    upsert_note_types,
)


def _update_type(wrapper, payload):
    return wrapper.run_sync(lambda c: upsert_note_types(c, [payload]))


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
        created = json.loads(
            c.upsert_notes(
                json.dumps([{"note_type": "TmplSafe", "deck": "TmplSafe", "fields": {field: "x"}}]),
                "allow",
                False,
            )
        )[0]
        assert created["status"] == "created", created
        return mid, created["id"]

    return wrapper.run_sync(build)


def _cards_by_template(wrapper, nid, mid):
    """Map each of the note's cards to its template name -> card id."""

    def q(c):
        detail = json.loads(c.collection_info(["note_types"], [_name_of(c, mid)]))
        names = [
            t["name"]
            for nt in detail["note_types"]
            if nt["id"] == mid
            for t in nt["detail"]["templates"]
        ]
        return {names[ord_]: cid for cid, ord_ in c.card_ords_of_note(nid)}

    return wrapper.run_sync(q)


def _name_of(c, mid):
    info = json.loads(c.collection_info(["note_types"], []))
    return next(nt["name"] for nt in info["note_types"] if nt["id"] == mid)


class TestUpdateNoteTypeTemplates:
    """Identity-based template operations preserve cards by template."""

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
