"""Note-type-surface behaviors.

The native note-type ops against the Python `shrike.note_types` results. The
data-safety properties (note data surviving renames/moves) are pinned natively,
where the note contents can be read back.
"""

from __future__ import annotations

import json

import pytest

shrike_native = pytest.importorskip("shrike_native")

from .conftest import requires_anki_core  # noqa: E402

pytestmark = requires_anki_core

CREATE_BATCH = [
    {
        "name": "Custom",
        "fields": ["A", "B", "C"],
        "templates": [{"name": "Card 1", "front": "{{A}}", "back": "{{B}}"}],
        "css": ".card { color: red; }",
    },
    {"name": "Custom"},  # missing fields/templates/css → per-item error
]

FIELD_OPS = [
    {"op": "rename", "name": "A", "new_name": "A2"},
    {"op": "add", "name": "D", "position": 1},
    {"op": "reposition", "name": "C", "position": 0},
]


def _no_id(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "id"}


def _no_ids(items: list) -> list:
    return [_no_id(i) for i in items]


def test_note_type_surface_end_to_end(native_core):
    # The per-case behavior suite lives in tests/unit; this keeps the
    # end-to-end flow + result shapes pinned at the binding level.
    native_create = json.loads(native_core.upsert_note_types(json.dumps(CREATE_BATCH)))
    assert native_create[0]["status"] == "created"
    assert native_create[1]["status"] == "error"

    native_ops = json.loads(native_core.update_note_type_fields("Custom", json.dumps(FIELD_OPS)))
    assert native_ops["fields"] == ["C", "A2", "D", "B"]

    native_fr = json.loads(
        native_core.find_replace_note_types("Custom", "color: red", "color: blue")
    )
    assert native_fr["replacements"] == 1 and native_fr["css_changed"] is True
    native_fr_regex = json.loads(
        native_core.find_replace_note_types(
            "Custom", r"\{\{(A2)\}\}", r"<b>{{\1}}</b>", True, True, True, True, False
        )
    )
    assert native_fr_regex["replacements"] == 1

    native_meta = json.loads(
        native_core.update_note_type_field_metadata(
            "Custom", json.dumps([{"name": "A2", "font": "Courier", "size": 14}])
        )
    )
    assert native_meta["fields_updated"] == ["A2"]

    # migrate parity (counts + drop reporting; ids are per-collection).
    created = json.loads(
        native_core.upsert_notes(
            json.dumps(
                [
                    {
                        "note_type": "Custom",
                        "deck": "Default",
                        "fields": {"C": "c-data", "A2": "a-data"},
                    }
                ]
            )
        )
    )
    nid = created[0]["id"]
    native_migrate = json.loads(
        native_core.migrate_note_type([nid], "Basic", json.dumps({"C": "Front", "A2": "Back"}))
    )
    assert native_migrate["to_note_type"] == "Basic"
    assert native_migrate["dropped_fields"] == ["D", "B"] or set(
        native_migrate["dropped_fields"]
    ) == {"D", "B"}
    # data followed the map
    assert native_core.get_note(nid)[2] == ["c-data", "a-data"]


def test_positional_replace_data_safety(native_core):
    """The positional-replace regression class, end to end through the binding:
    a positional rename/append keeps note data; a move is refused with the
    identity-tool pointer; identity ops migrate data by name."""
    native_core.upsert_note_types(
        json.dumps(
            [
                {
                    "name": "DS",
                    "fields": ["X", "Y"],
                    "templates": [{"name": "Card 1", "front": "{{X}}", "back": "{{Y}}"}],
                    "css": "",
                }
            ]
        )
    )
    created = json.loads(
        native_core.upsert_notes(
            json.dumps([{"note_type": "DS", "deck": "Default", "fields": {"X": "x1", "Y": "y1"}}])
        )
    )
    nid = created[0]["id"]
    ds_id = native_core.notetype_id("DS")

    # rename-in-place + append: data survives.
    results = json.loads(
        native_core.upsert_note_types(json.dumps([{"id": ds_id, "fields": ["X2", "Y", "Z"]}]))
    )
    assert results[0]["status"] == "updated"
    assert native_core.get_note(nid)[2] == ["x1", "y1", ""]

    # a move is refused, nothing changed.
    rejected = json.loads(
        native_core.upsert_note_types(json.dumps([{"id": ds_id, "fields": ["Y", "X2", "Z"]}]))
    )
    assert rejected[0]["status"] == "error"
    assert "update_note_type_fields" in rejected[0]["error"]
    assert native_core.get_note(nid)[2] == ["x1", "y1", ""]

    # identity reposition: data follows the field.
    json.loads(
        native_core.update_note_type_fields(
            "DS", json.dumps([{"op": "reposition", "name": "X2", "position": 2}])
        )
    )
    assert native_core.get_note(nid)[2] == ["y1", "", "x1"]
