"""Field editor-metadata get/set (#119): note_types.update_note_type_field_metadata + the
per-field metadata folded into collection_info note-type details."""

from __future__ import annotations

import pytest

from tests.unit._native_shims import (
    NoteTypeOpError,
    update_note_type_field_metadata,
    upsert_note_types,
)


def _make(wrapper, name="Meta", fields=("Front", "Back")):
    tmpls = [{"name": "C", "front": "{{" + fields[0] + "}}", "back": "{{" + fields[-1] + "}}"}]
    wrapper.run_sync(
        lambda c: upsert_note_types(
            c, [{"name": name, "fields": list(fields), "templates": tmpls, "css": ""}]
        )
    )


def _set(wrapper, name, updates):
    return wrapper.run_sync(lambda c: update_note_type_field_metadata(c, name, updates))


async def _detail_fields(wrapper, name):
    info = await wrapper.get_collection_info(include=["note_types"], note_type_details=[name])
    nt = next(n for n in info["note_types"] if n["name"] == name)
    return {f["name"]: f for f in nt["detail"]["fields"]}


class TestSetFieldMetadata:
    async def test_set_all_three(self, wrapper):
        _make(wrapper)
        result = _set(
            wrapper, "Meta", [{"name": "Front", "font": "Georgia", "size": 28, "description": "Q"}]
        )
        assert result["fields_updated"] == ["Front"]
        fields = await _detail_fields(wrapper, "Meta")
        assert fields["Front"]["font"] == "Georgia"
        assert fields["Front"]["size"] == 28
        assert fields["Front"]["description"] == "Q"

    async def test_only_provided_attrs_change(self, wrapper):
        _make(wrapper)
        before = await _detail_fields(wrapper, "Meta")
        _set(wrapper, "Meta", [{"name": "Front", "size": 30}])
        after = await _detail_fields(wrapper, "Meta")
        assert after["Front"]["size"] == 30
        assert after["Front"]["font"] == before["Front"]["font"]  # unchanged
        assert after["Front"]["description"] == before["Front"]["description"]

    async def test_multiple_fields_in_one_call(self, wrapper):
        _make(wrapper)
        result = _set(
            wrapper,
            "Meta",
            [{"name": "Front", "description": "front"}, {"name": "Back", "description": "back"}],
        )
        assert result["fields_updated"] == ["Front", "Back"]

    def test_unknown_field_errors(self, wrapper):
        _make(wrapper)
        with pytest.raises(NoteTypeOpError, match="field 'Nope'"):
            _set(wrapper, "Meta", [{"name": "Nope", "size": 20}])

    def test_unknown_note_type_errors(self, wrapper):
        with pytest.raises(NoteTypeOpError, match="not found"):
            _set(wrapper, "Nope", [{"name": "Front", "size": 20}])

    def test_empty_update_errors(self, wrapper):
        _make(wrapper)
        with pytest.raises(NoteTypeOpError, match="at least one"):
            _set(wrapper, "Meta", [{"name": "Front"}])

    def test_atomic_bad_name_changes_nothing(self, wrapper):
        _make(wrapper)
        with pytest.raises(NoteTypeOpError):
            _set(wrapper, "Meta", [{"name": "Front", "size": 99}, {"name": "Nope", "size": 99}])
        # The valid update did not apply (validation precedes any write).
        fields = wrapper.run_sync(
            lambda c: {
                f["name"]: f["size"]
                for nt in __import__("json").loads(c.collection_info(["note_types"], ["Meta"]))[
                    "note_types"
                ]
                if nt["name"] == "Meta"
                for f in nt["detail"]["fields"]
            }
        )
        assert fields["Front"] != 99


class TestCollectionInfoFieldMetadata:
    async def test_detail_includes_field_metadata(self, wrapper):
        _make(wrapper)
        fields = await _detail_fields(wrapper, "Meta")
        # Anki defaults present even before any set.
        assert set(fields["Front"]) == {"name", "font", "size", "description"}
        assert isinstance(fields["Front"]["size"], int)
