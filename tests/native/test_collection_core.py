"""Parity harness for the native CollectionCore binding (#278 series, step 1).

Ported wrapper-fixture cases (the duplicate-policy matrix, structural
validation, the search grammar, the col_mod watermark, the CRUD round trip)
run against the native core on its own temp collection. The cross-core parity
case runs the SAME sequence through the pip `anki` package **in a subprocess
on a separate collection file** — the harness process touches a collection
through one core only, ever — and compares the observable outcomes.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

shrike_native = pytest.importorskip("shrike_native")

from .conftest import requires_anki_core  # noqa: E402

pytestmark = requires_anki_core

DEFAULT_DECK = 1


class TestRoundTrip:
    def test_open_create_read_update_delete(self, native_core):
        basic = native_core.notetype_id("Basic")
        nid = native_core.create_note(basic, DEFAULT_DECK, ["front text", "back text"], ["tag-a"])
        assert isinstance(nid, int)

        assert native_core.find_notes("deck:*") == [nid]
        note_id, notetype_id, fields, tags = native_core.get_note(nid)
        assert (note_id, notetype_id) == (nid, basic)
        assert fields == ["front text", "back text"]
        assert tags == ["tag-a"]

        native_core.update_note(nid, ["front text", "new back"])
        assert native_core.get_note(nid)[2][1] == "new back"
        # tags untouched by a fields-only update
        assert native_core.get_note(nid)[3] == ["tag-a"]

        native_core.update_note(nid, ["front text", "new back"], tags=["tag-b"])
        assert native_core.get_note(nid)[3] == ["tag-b"]

        assert native_core.delete_notes([nid]) == 1
        assert native_core.find_notes("deck:*") == []

    def test_unknown_notetype_is_input_error(self, native_core):
        with pytest.raises(shrike_native.NativeInputError):
            native_core.notetype_id("No Such Type")


class TestDuplicatePolicy:
    """The #77 policy matrix, byte-for-byte the Rust tripwire's semantics."""

    def test_matrix(self, native_core):
        basic = native_core.notetype_id("Basic")
        fields = ["same front", "back"]
        first = native_core.create_note(basic, DEFAULT_DECK, fields, [])
        assert isinstance(first, int)

        # error (default): reported, not written
        with pytest.raises(shrike_native.NativeInputError, match="duplicate"):
            native_core.create_note(basic, DEFAULT_DECK, fields, [])
        # skip: not written, signalled as None
        assert native_core.create_note(basic, DEFAULT_DECK, fields, [], on_duplicate="skip") is None
        # allow: written anyway
        allowed = native_core.create_note(basic, DEFAULT_DECK, fields, [], on_duplicate="allow")
        assert isinstance(allowed, int)
        assert len(native_core.find_notes("deck:*")) == 2

    def test_empty_first_field_always_errors(self, native_core):
        basic = native_core.notetype_id("Basic")
        # policy never overrides structural errors
        with pytest.raises(shrike_native.NativeInputError):
            native_core.create_note(basic, DEFAULT_DECK, ["", "back"], [], on_duplicate="allow")

    def test_bad_policy_string_is_input_error(self, native_core):
        basic = native_core.notetype_id("Basic")
        with pytest.raises(shrike_native.NativeInputError, match="on_duplicate"):
            native_core.create_note(basic, DEFAULT_DECK, ["a", "b"], [], on_duplicate="maybe")


class TestSearchGrammar:
    def test_grammar_and_malformed_expression(self, native_core):
        basic = native_core.notetype_id("Basic")
        native_core.create_note(basic, DEFAULT_DECK, ["alpha", "beta"], ["mytag"])
        assert len(native_core.find_notes("tag:mytag")) == 1
        assert len(native_core.find_notes("tag:nope")) == 0
        assert len(native_core.find_notes("alpha")) == 1
        with pytest.raises(shrike_native.NativeInputError):
            native_core.find_notes("added:notanumber")


class TestColMod:
    def test_advances_on_write(self, native_core):
        basic = native_core.notetype_id("Basic")
        before = native_core.col_mod()
        native_core.create_note(basic, DEFAULT_DECK, ["a", "b"], [])
        after = native_core.col_mod()
        assert after >= before
        assert after > 0


# The pip-core side of the cross-core parity case. Runs in a SUBPROCESS on its
# own collection file (one collection, one core) and prints JSON outcomes for
# the same op sequence the native side runs.
_PIP_SIDE = r"""
import json, sys
from anki.collection import Collection
from anki.notes import NoteFieldsCheckResult

col = Collection(sys.argv[1])
basic = col.models.by_name("Basic")["id"]

def new_note(fields, tags=()):
    note = col.new_note(col.models.get(basic))
    note.fields = list(fields)
    for t in tags:
        note.tags.append(t)
    return note

out = {}
first = new_note(["same front", "back"], ["mytag"])
col.add_note(first, 1)

dup = new_note(["same front", "x"])
out["dup_state"] = NoteFieldsCheckResult.Name(dup.fields_check())
empty = new_note(["", "x"])
out["empty_state"] = NoteFieldsCheckResult.Name(empty.fields_check())

out["tag_hits"] = len(col.find_notes("tag:mytag"))
out["miss_hits"] = len(col.find_notes("tag:nope"))
out["word_hits"] = len(col.find_notes("same"))
try:
    col.find_notes("added:notanumber")
    out["malformed"] = "ok"
except Exception:
    out["malformed"] = "error"

first.fields[1] = "new back"
col.update_note(first)
out["updated_back"] = col.get_note(first.id).fields[1]
col.remove_notes([first.id])
out["after_delete"] = len(col.find_notes("deck:*"))
col.close()
print(json.dumps(out))
"""


def test_cross_core_parity(tmp_path, native_core):
    """The same observable sequence through both cores, separate collections,
    separate processes: classifications, search counts, update/delete effects
    must agree."""
    pip_col = tmp_path / "pip" / "collection.anki2"
    pip_col.parent.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _PIP_SIDE, str(pip_col)],
        capture_output=True,
        text=True,
        check=True,
    )
    # The JSON is the LAST stdout line — the anki import can emit benign
    # chatter to stdout first (#458) — and a parse failure must show what the
    # subprocess actually printed, not an opaque JSONDecodeError.
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        pip = json.loads(lines[-1] if lines else "")
    except json.JSONDecodeError:
        pytest.fail(
            f"pip-side subprocess produced no JSON.\n"
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
        )

    basic = native_core.notetype_id("Basic")
    nid = native_core.create_note(basic, DEFAULT_DECK, ["same front", "back"], ["mytag"])

    # Duplicate / empty classification parity (#77: anki's own fields_check).
    assert pip["dup_state"] == "DUPLICATE"
    assert native_core.create_note(basic, 1, ["same front", "x"], [], on_duplicate="skip") is None
    assert pip["empty_state"] == "EMPTY"
    with pytest.raises(shrike_native.NativeInputError):
        native_core.create_note(basic, 1, ["", "x"], [])

    # Search grammar parity.
    assert len(native_core.find_notes("tag:mytag")) == pip["tag_hits"] == 1
    assert len(native_core.find_notes("tag:nope")) == pip["miss_hits"] == 0
    assert len(native_core.find_notes("same")) == pip["word_hits"] == 1
    assert pip["malformed"] == "error"
    with pytest.raises(shrike_native.NativeInputError):
        native_core.find_notes("added:notanumber")

    # Update + delete effect parity.
    native_core.update_note(nid, ["same front", "new back"])
    assert native_core.get_note(nid)[2][1] == pip["updated_back"] == "new back"
    native_core.delete_notes([nid])
    assert len(native_core.find_notes("deck:*")) == pip["after_delete"] == 0
