"""Write-surface parity (#278 series, step 3).

The native upsert/tags/decks/find-replace/delete-note-types ops against their
CollectionWrapper result shapes: the cross-core case runs the same batch
through the pip core in a subprocess on a separate collection file and
compares the per-item result dicts (the `_upsert_notes` status/reason
vocabulary is the contract), plus native-only coverage for the paths whose
effects the Rust round-trip already pins.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

shrike_native = pytest.importorskip("shrike_native")

from .conftest import requires_anki_core  # noqa: E402

pytestmark = requires_anki_core

# One batch exercising the whole per-item result union: create (auto-deck),
# duplicate (skip policy), unknown note type, unknown field, structural empty.
UPSERT_BATCH = [
    {
        "note_type": "Basic",
        "deck": "Science::Physics",
        "fields": {"Front": "alpha", "Back": "beta"},
        "tags": ["t1"],
    },
    {"note_type": "Basic", "deck": "Science::Physics", "fields": {"Front": "alpha", "Back": "d"}},
    {"note_type": "Nope", "deck": "D", "fields": {"Front": "x"}},
    {"note_type": "Basic", "deck": "D", "fields": {"Bogus": "x"}},
    {"note_type": "Basic", "deck": "D", "fields": {"Front": "", "Back": "y"}},
]

_PIP_SIDE = r"""
import asyncio, json, sys
from shrike.collection import CollectionWrapper

BATCH = json.loads(sys.argv[2])

async def main():
    w = CollectionWrapper(sys.argv[1])
    out = {}
    out["upsert"] = await w.upsert_notes(BATCH, on_duplicate="skip")
    nid = out["upsert"][0]["id"]
    out["dry"] = await w.upsert_notes(
        [{"note_type": "Basic", "deck": "DryDeck", "fields": {"Front": "q", "Back": "a"}}],
        dry_run=True,
    )
    out["update"] = await w.upsert_notes(
        [{"id": nid, "fields": {"Back": "new back"}, "tags": ["t2"], "deck": "Default"}]
    )
    out["tags"] = await w.update_note_tags([nid, 999], set_tags=None, add=["x1"], remove=["t2"])
    out["rename"] = await w.rename_tag("x1", "renamed", [nid])
    out["decks"] = await w.upsert_decks([{"name": "Empty::Leaf"}])
    out["delete_decks"] = await w.delete_decks(["Empty::Leaf", "Default", "Ghost"])
    out["replace"] = await w.find_replace(
        "alpha", "omega", regex=False, match_case=True, ids=[nid], sample_limit=0,
    )
    w.close()
    print(json.dumps(out))

asyncio.run(main())
"""


def _strip_ids(results: list) -> list:
    return [{k: v for k, v in r.items() if k != "id"} for r in results]


def test_cross_core_write_parity(tmp_path, native_core):
    pip_col = tmp_path / "pip" / "collection.anki2"
    pip_col.parent.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _PIP_SIDE, str(pip_col), json.dumps(UPSERT_BATCH)],
        capture_output=True,
        text=True,
        check=True,
    )
    pip = json.loads(proc.stdout)

    # Upsert: identical per-item result dicts modulo the created note id.
    native_results = json.loads(native_core.upsert_notes(json.dumps(UPSERT_BATCH), "skip"))
    assert _strip_ids(native_results) == _strip_ids(pip["upsert"])
    nid = native_results[0]["id"]

    # dry_run result shape.
    native_dry = json.loads(
        native_core.upsert_notes(
            json.dumps(
                [{"note_type": "Basic", "deck": "DryDeck", "fields": {"Front": "q", "Back": "a"}}]
            ),
            "error",
            True,
        )
    )
    assert native_dry == pip["dry"]

    # Update result shape + observable effect.
    native_update = json.loads(
        native_core.upsert_notes(
            json.dumps(
                [{"id": nid, "fields": {"Back": "new back"}, "tags": ["t2"], "deck": "Default"}]
            )
        )
    )
    assert _strip_ids(native_update) == _strip_ids(pip["update"])
    assert native_core.get_note(nid)[2][1] == "new back"

    # Tags / rename: same counts + not_found echo (typed wire since #391 —
    # both routes ride the same binding, so the dicts match whole).
    native_tags = json.loads(native_core.update_note_tags([nid, 999], add=["x1"], remove=["t2"]))
    assert native_tags == pip["tags"]
    native_rename = json.loads(native_core.rename_tag("x1", "renamed", [nid]))
    assert native_rename == pip["rename"]

    # Decks: upsert + empty-only delete result dicts (ids differ per file).
    native_decks = json.loads(native_core.upsert_decks(json.dumps([{"name": "Empty::Leaf"}])))
    assert _strip_ids(native_decks) == _strip_ids(pip["decks"])
    native_del = json.loads(native_core.delete_decks(["Empty::Leaf", "Default", "Ghost"]))
    assert native_del == pip["delete_decks"]

    # find_replace: same changed count; changed_ids echo the native note
    # (a typed tuple since #513 — the id set never rides a wire).
    notes_changed, changed_ids = native_core.find_replace_notes(
        [nid], "alpha", "omega", False, True, None
    )
    assert notes_changed == pip["replace"]["notes_changed"] == 1
    assert changed_ids == [nid]
    assert native_core.get_note(nid)[2][0] == "omega"


def test_set_note_tags_bulk_replaces_tags_and_preserves_fields(native_core):
    """The set_tags (replace) path rides set_note_tags_bulk, which (since #716)
    rebuilds each UpdateNotes row from one batched DB read instead of a GetNote
    per note. Pin the behavior that read must preserve: tags are replaced
    exactly across the whole set, and every note's fields/notetype survive."""
    basic = native_core.notetype_id("Basic")
    a = native_core.create_note(basic, 1, ["a-front", "a-back"], ["old1", "old2"])
    b = native_core.create_note(basic, 1, ["b-front", "b-back"], ["keep"])

    out = json.loads(native_core.update_note_tags([a, b], set_tags=["fresh", "new"]))
    assert out["notes_modified"] == 2
    assert out["not_found"] == []

    # Tags replaced exactly (order-independent), fields + notetype untouched.
    for nid, fields in ((a, ["a-front", "a-back"]), (b, ["b-front", "b-back"])):
        _id, notetype_id, got_fields, got_tags = native_core.get_note(nid)
        assert notetype_id == basic
        assert got_fields == fields
        assert sorted(got_tags) == ["fresh", "new"]

    # Empty set clears every tag (still one read + one write).
    cleared = json.loads(native_core.update_note_tags([a], set_tags=[]))
    assert cleared["notes_modified"] == 1
    assert native_core.get_note(a)[3] == []


def test_delete_note_types(native_core):
    basic = native_core.notetype_id("Basic")
    native_core.create_note(basic, 1, ["a", "b"], [])
    out = json.loads(native_core.delete_note_types([basic, 424242]))
    assert out["results"][0]["status"] == "error"
    assert "use this type" in out["results"][0]["error"]
    assert out["results"][1] == {"id": 424242, "status": "not_found"}
    cloze = native_core.notetype_id("Cloze")
    out2 = json.loads(native_core.delete_note_types([cloze]))
    assert out2["results"][0]["status"] == "deleted"
    with pytest.raises(shrike_native.NativeInputError):
        native_core.notetype_id("Cloze")
