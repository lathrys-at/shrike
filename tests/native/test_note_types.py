"""Note-type-surface parity (#278 series, step 4).

The native note-type ops against the Python `shrike.note_types` results: the
cross-core case runs the same sequence through the pip core in a subprocess on
a separate collection file and compares the result dicts (ids stripped — they
are creation timestamps). The data-safety properties (note data surviving
renames/moves) are pinned natively, where the note contents can be read back.
"""

from __future__ import annotations

import json
import subprocess
import sys

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

_PIP_SIDE = r"""
import asyncio, json, sys
from shrike.collection import CollectionWrapper
from shrike.note_types import (
    find_and_replace_note_types,
    update_note_type_field_metadata,
    update_note_type_fields,
    upsert_note_types,
)

CREATE = json.loads(sys.argv[2])
OPS = json.loads(sys.argv[3])

async def main():
    w = CollectionWrapper(sys.argv[1])
    out = {}

    def on_worker(col):
        out["create"] = upsert_note_types(col, CREATE)
        out["field_ops"] = update_note_type_fields(col, "Custom", OPS)
        out["fr"] = find_and_replace_note_types(
            col, "Custom", search="color: red", replacement="color: blue",
        )
        out["fr_regex"] = find_and_replace_note_types(
            col, "Custom", search=r"\{\{(A2)\}\}", replacement=r"<b>{{\1}}</b>",
            regex=True, css=False,
        )
        out["meta"] = update_note_type_field_metadata(
            col, "Custom", [{"name": "A2", "font": "Courier", "size": 14}]
        )
        return None

    await w.run(on_worker)
    # migrate: create one note then move it to Basic, dropping fields.
    created = await w.upsert_notes([{
        "note_type": "Custom", "deck": "Default",
        "fields": {"C": "c-data", "A2": "a-data"},
    }])
    nid = created[0]["id"]
    out["migrate"] = await w.migrate_note_type(
        [nid], "Basic", {"C": "Front", "A2": "Back"}, dry_run=False
    )
    out["migrate"]["changed"] = len(out["migrate"]["changed"])
    w.close()
    print(json.dumps(out))

asyncio.run(main())
"""


def _no_id(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "id"}


def _no_ids(items: list) -> list:
    return [_no_id(i) for i in items]


def test_cross_core_note_type_parity(tmp_path, native_core):
    pip_col = tmp_path / "pip" / "collection.anki2"
    pip_col.parent.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _PIP_SIDE,
            str(pip_col),
            json.dumps(CREATE_BATCH),
            json.dumps(FIELD_OPS),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pip = json.loads(proc.stdout)

    native_create = json.loads(native_core.upsert_note_types(json.dumps(CREATE_BATCH)))
    assert _no_ids(native_create) == _no_ids(pip["create"])

    native_ops = json.loads(native_core.update_note_type_fields("Custom", json.dumps(FIELD_OPS)))
    assert _no_id(native_ops) == _no_id(pip["field_ops"])

    native_fr = json.loads(
        native_core.find_replace_note_types("Custom", "color: red", "color: blue")
    )
    assert _no_id(native_fr) == _no_id(pip["fr"])
    native_fr_regex = json.loads(
        native_core.find_replace_note_types(
            "Custom", r"\{\{(A2)\}\}", r"<b>{{\1}}</b>", True, True, True, True, False
        )
    )
    assert _no_id(native_fr_regex) == _no_id(pip["fr_regex"])

    native_meta = json.loads(
        native_core.update_note_type_field_metadata(
            "Custom", json.dumps([{"name": "A2", "font": "Courier", "size": 14}])
        )
    )
    assert _no_id(native_meta) == _no_id(pip["meta"])

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
    native_migrate["changed"] = len(native_migrate["changed"])
    assert native_migrate == pip["migrate"]
    # data followed the map
    assert native_core.get_note(nid)[2] == ["c-data", "a-data"]


def test_positional_replace_data_safety(native_core):
    """The #99 regression class, end to end through the binding: a positional
    rename/append keeps note data; a move is refused with the identity-tool
    pointer; identity ops migrate data by name."""
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
