"""Collection-layer tests for migrate_note_type (#75): Anki's models.change.

Exercises CollectionWrapper._migrate_note_type directly (no server): field/
template remap, reported drops, validation, and id/card preservation.
"""

from __future__ import annotations

import pytest

from shrike.note_types import upsert_note_types


def _add(wrapper, model, fields, *, deck="D"):
    def build(c):
        n = c.new_note(c.models.by_name(model))
        for k, v in fields.items():
            n[k] = v
        c.add_note(n, c.decks.id(deck))
        return n.id

    return wrapper.run_sync(build)


def _make_type(wrapper, name, fields):
    tmpls = [{"name": "C", "front": "{{" + fields[0] + "}}", "back": "{{" + fields[-1] + "}}"}]
    wrapper.run_sync(
        lambda c: upsert_note_types(
            c, [{"name": name, "fields": fields, "templates": tmpls, "css": ""}]
        )
    )


def _migrate(wrapper, ids, to, fmap, **kw):
    return wrapper.run_sync(lambda c: wrapper._migrate_note_type(ids, to, fmap, **kw))


def _note(wrapper, nid):
    return wrapper.run_sync(lambda c: (c.get_note(nid).mid, dict(c.get_note(nid).items())))


class TestMigrateNoteType:
    def test_basic_to_cloze_moves_content_preserves_id(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "FRONTVAL", "Back": "BACKVAL"})
        cloze_mid = wrapper.run_sync(lambda c: c.models.by_name("Cloze")["id"])

        result = _migrate(wrapper, [nid], "Cloze", {"Front": "Text", "Back": "Back Extra"})
        assert result["changed"] == [nid]
        assert result["from_note_type"] == "Basic"
        assert result["to_note_type"] == "Cloze"
        assert result["dropped_fields"] == []

        mid, content = _note(wrapper, nid)
        assert mid == cloze_mid  # same note id, new type
        assert content["Text"] == "FRONTVAL"
        assert content["Back Extra"] == "BACKVAL"

    def test_unmapped_source_field_is_dropped_and_reported(self, wrapper):
        _make_type(wrapper, "OneField", ["Only"])
        nid = _add(wrapper, "Basic", {"Front": "keep", "Back": "lose"})
        result = _migrate(wrapper, [nid], "OneField", {"Front": "Only"})
        assert result["dropped_fields"] == ["Back"]
        assert result["new_empty_fields"] == []
        _, content = _note(wrapper, nid)
        assert content == {"Only": "keep"}

    def test_new_empty_fields_reported(self, wrapper):
        _make_type(wrapper, "ThreeField", ["A", "B", "C"])
        nid = _add(wrapper, "Basic", {"Front": "x", "Back": "y"})
        result = _migrate(wrapper, [nid], "ThreeField", {"Front": "A"})
        assert result["dropped_fields"] == ["Back"]
        assert result["new_empty_fields"] == ["B", "C"]

    def test_dry_run_reports_but_does_not_change(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "a", "Back": "b"})
        basic_mid = wrapper.run_sync(lambda c: c.get_note(nid).mid)
        result = _migrate(wrapper, [nid], "Cloze", {"Front": "Text"}, dry_run=True)
        assert result["dry_run"] is True
        assert result["dropped_fields"] == ["Back"]
        mid, content = _note(wrapper, nid)
        assert mid == basic_mid  # unchanged
        assert content["Front"] == "a"

    def test_card_count_preserved_single_template(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "q", "Back": "a"})
        _migrate(wrapper, [nid], "Cloze", {"Front": "Text", "Back": "Back Extra"})
        assert len(wrapper.run_sync(lambda c: c.card_ids_of_note(nid))) == 1

    def test_template_map(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "q", "Back": "a"})
        result = _migrate(
            wrapper,
            [nid],
            "Cloze",
            {"Front": "Text", "Back": "Back Extra"},
            template_map={"Card 1": "Cloze"},
        )
        assert result["to_note_type"] == "Cloze"

    # -- validation ----------------------------------------------------------

    def test_unknown_source_field_errors(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "a", "Back": "b"})
        with pytest.raises(ValueError, match="Source field 'Nope'"):
            _migrate(wrapper, [nid], "Cloze", {"Nope": "Text"})

    def test_unknown_target_field_errors(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "a", "Back": "b"})
        with pytest.raises(ValueError, match="Target field 'Nope'"):
            _migrate(wrapper, [nid], "Cloze", {"Front": "Nope"})

    def test_ambiguous_map_errors(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "a", "Back": "b"})
        with pytest.raises(ValueError, match="same target field"):
            _migrate(wrapper, [nid], "Cloze", {"Front": "Text", "Back": "Text"})

    def test_same_type_errors(self, wrapper):
        nid = _add(wrapper, "Basic", {"Front": "a", "Back": "b"})
        with pytest.raises(ValueError, match="already use"):
            _migrate(wrapper, [nid], "Basic", {"Front": "Front"})

    def test_mixed_source_types_error(self, wrapper):
        b = _add(wrapper, "Basic", {"Front": "a", "Back": "b"})
        cz = _add(wrapper, "Cloze", {"Text": "{{c1::x}}"})
        with pytest.raises(ValueError, match="share one note type"):
            _migrate(wrapper, [b, cz], "Cloze", {"Front": "Text"})

    def test_missing_note_errors(self, wrapper):
        with pytest.raises(ValueError, match="not found"):
            _migrate(wrapper, [9999999999], "Cloze", {"Front": "Text"})
