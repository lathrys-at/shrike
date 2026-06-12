"""Read-surface parity (#278 series, step 2).

Two layers:

1. **Byte-identity of the embed-text normalization** — the Rust
   `normalize_text` against the Python `shrike.embed_text.normalize_for_embedding`
   over a corpus of nasty field values. The output is part of the vector space
   (folded into the index fingerprint), so this is an exact-equality contract,
   not a similarity check. Importing `shrike.embed_text` here is safe: it is
   pure text processing — no collection is opened through the pip core in this
   process (the one-core-per-collection rule).

2. **Cross-core read parity** — `list_notes` / `collection_info` /
   `note_texts` against the same sequence through CollectionWrapper in a
   subprocess on a separate collection file, comparing the JSON shapes.
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

# The normalization corpus: cloze (plain/hinted/nested), HTML inline + block,
# entities, NBSP, sound, legacy LaTeX, MathJax, mixed mess, multi-script,
# empties, markup-only.
NORMALIZE_CORPUS = [
    "",
    "plain text",
    "  leading and   trailing   ",
    "{{c1::France}}",
    "{{c1::France::country}}",
    "{{c2::outer {{c1::inner}} rest}}",
    "a<br>b",
    "a<br/>b<div>c</div>",
    "<b>bold</b> and <i>italic</i>",
    "x &amp; y &lt;tag&gt; &nbsp; z",
    "café au lait",
    "[sound:audio.mp3] spoken text",
    "[latex]\\frac{1}{2}[/latex] and [$]x^2[/$] and [$$]y[/$$]",
    "\\(E = mc^2\\) inline and \\[block\\] and $$display$$",
    '<img src="diagram.png"> labelled diagram',
    "<ul><li>one</li><li>two</li></ul>",
    "nested <div><p>para <b>deep</b></p></div> end",
    "&#x1F600; numeric &#65; entities",
    "日本語のテキスト<br>第二行",
    "<p>malformed <b>unclosed",
    "{{c1::answer with <b>html</b>::hint with <i>html</i>}}",
    "edge {{c1::}} empty cloze",
    "tab\tand\nnewline   collapse",
]


def test_normalize_byte_identity(native_core):
    from tests.oracles.embed_text_oracle import normalize_for_embedding

    for value in NORMALIZE_CORPUS:
        assert native_core.normalize_text(value) == normalize_for_embedding(value), (
            f"normalization diverged for {value!r}"
        )


def test_extract_image_refs_parity(native_core):
    """The Rust img-src extraction against the Python HTMLParser one — the
    parser-not-regex property must hold on the lazy-load / quoted-attr cases.
    The native side has no direct binding; covered via note_embed_inputs below
    and the Rust unit tests; here we pin the Python reference on the same
    cases the Rust tests assert, so the two suites can't drift silently."""
    from tests.oracles.embed_text_oracle import extract_image_refs

    assert extract_image_refs('<img data-src="lazy.png" src="real.png">') == ["real.png"]
    assert extract_image_refs("<img alt=\"src=fake.png\" src='dir/pic.jpg'>") == ["pic.jpg"]
    assert extract_image_refs('<IMG SRC="a.png"><img src="a.png"><img src="http://x/b.png">') == [
        "a.png"
    ]
    assert extract_image_refs('<img src="a&amp;b.png">') == ["a&b.png"]


# The pip-core side: same notes through CollectionWrapper, dump list_notes,
# collection_info, and note_texts as JSON. Separate process + collection file.
_PIP_SIDE = r"""
import asyncio, json, sys
from shrike.collection import CollectionWrapper

async def main():
    w = CollectionWrapper(sys.argv[1])
    notes = [
        {"note_type": "Basic", "deck": "Default",
         "fields": {"Front": "the <b>mitochondria</b>&nbsp;powerhouse",
                    "Back": "energy of the cell<br>line2"},
         "tags": ["bio"]},
        {"note_type": "Basic", "deck": "Science::Physics",
         "fields": {"Front": "newton's \\(F=ma\\)", "Back": "[sound:x.mp3] mechanics"},
         "tags": ["physics"]},
    ]
    created = await w.upsert_notes(notes)
    ids = [r["id"] for r in created]
    out = {}
    out["list_by_tag"] = await w.list_notes(tags=["bio"], fields_mode="full", limit=50)
    info = await w.get_collection_info(["all"], ["Basic"])
    # summary path/timestamps vary per file; compare the stable parts
    info["summary"].pop("path"); info["summary"].pop("modified"); info["summary"].pop("created")
    out["info"] = info
    out["note_texts"] = await w.note_texts_for_embedding(ids)
    out["ids"] = ids
    w.close()
    print(json.dumps(out))

asyncio.run(main())
"""


def _strip_volatile(info: dict) -> dict:
    info = json.loads(json.dumps(info))
    info["summary"].pop("path", None)
    info["summary"].pop("modified", None)
    info["summary"].pop("created", None)
    return info


def _strip_note_volatile(listing: dict) -> dict:
    listing = json.loads(json.dumps(listing))
    for note in listing["notes"]:
        note.pop("id", None)
        note.pop("modified", None)
    return listing


def test_cross_core_read_parity(tmp_path, native_core):
    pip_col = tmp_path / "pip" / "collection.anki2"
    pip_col.parent.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _PIP_SIDE, str(pip_col)],
        capture_output=True,
        text=True,
        check=True,
    )
    pip = json.loads(proc.stdout)

    # Same notes through the native core (deck "Science::Physics" needs to
    # exist first on the native side; the pip wrapper auto-creates — the
    # native create-deck op is a later series step, so park both notes in
    # Default and scope the deck comparison to the pip side's own output).
    basic = native_core.notetype_id("Basic")
    n1 = native_core.create_note(
        basic,
        DEFAULT_DECK,
        ["the <b>mitochondria</b>&nbsp;powerhouse", "energy of the cell<br>line2"],
        ["bio"],
    )
    n2 = native_core.create_note(
        basic, DEFAULT_DECK, ["newton's \\(F=ma\\)", "[sound:x.mp3] mechanics"], ["physics"]
    )

    # list_notes by tag: identical note payloads modulo ids/deck/timestamps.
    native_listing = json.loads(native_core.list_notes(tags=["bio"]))
    pip_listing = _strip_note_volatile(pip.pop("list_by_tag"))
    for note in pip_listing["notes"]:
        note.pop("deck", None)
    native_cmp = _strip_note_volatile(native_listing)
    for note in native_cmp["notes"]:
        note.pop("deck", None)
    assert native_cmp == pip_listing

    # note_texts: byte-identical rendered embedding text, both notes.
    assert native_core.note_texts([n1, n2]) == pip["note_texts"]

    # collection_info: stable summary numbers + note_types section identical.
    native_info = _strip_volatile(json.loads(native_core.collection_info(["all"], ["Basic"])))
    pip_info = pip["info"]
    assert native_info["summary"]["notes"] == pip_info["summary"]["notes"] == 2
    assert native_info["summary"]["cards"] == pip_info["summary"]["cards"]
    assert native_info["summary"]["note_types"] == pip_info["summary"]["note_types"]

    # notetype ids are creation timestamps — different per collection file.
    def _no_ids(note_types: list) -> list:
        return [{k: v for k, v in nt.items() if k != "id"} for nt in note_types]

    assert _no_ids(native_info["note_types"]) == _no_ids(pip_info["note_types"])
    assert native_info["tags"] == pip_info["tags"] == ["bio", "physics"]
    # decks/stats differ (the pip side created Science::Physics) — compare the
    # shared Default deck's note accounting only.
    native_default = next(d for d in native_info["decks"] if d["name"] == "Default")
    assert native_default["note_count"] == 2


def test_list_notes_filters_and_errors(native_core):
    basic = native_core.notetype_id("Basic")
    nid = native_core.create_note(basic, DEFAULT_DECK, ["alpha", "beta"], ["t1"])
    native_core.create_note(basic, DEFAULT_DECK, ["gamma", "delta"], ["t2"])

    by_ids = json.loads(native_core.list_notes(ids=[nid]))
    assert by_ids["total"] == 1
    assert by_ids["notes"][0]["content"] == {"Front": "alpha", "Back": "beta"}

    neg = json.loads(native_core.list_notes(tags=["-t1"]))
    assert [n["content"]["Front"] for n in neg["notes"]] == ["gamma"]

    by_type = json.loads(native_core.list_notes(note_type="Basic", with_fields=False))
    assert by_type["total"] == 2
    assert "content" not in by_type["notes"][0]

    recent = json.loads(native_core.list_notes(modified_since=0))
    assert recent["total"] == 2
    future = json.loads(native_core.list_notes(modified_since=2**33))
    assert future["total"] == 0

    with pytest.raises(shrike_native.NativeInputError, match="filter"):
        native_core.list_notes()


def test_read_wire_bytes_are_the_legacy_format(native_core):
    """#391 phase 2 byte pin: the read surface now returns typed structs in
    Rust, serialized once at the binding edge — and the bytes Python receives
    must stay exactly the pre-seam hand-built-``Value`` format: compact,
    keys sorted (serde_json's map is a BTreeMap), ``None`` fields omitted
    (no ``content`` key in meta mode, only requested sections), never an
    explicit ``null``. ``json.dumps(sort_keys=True, separators=(",", ":"))``
    of the parse reproduces that format exactly, so equality here is a
    byte-level pin of the wire."""
    basic = native_core.notetype_id("Basic")
    nid = native_core.create_note(basic, DEFAULT_DECK, ["alpha", "beta"], ["t1"])

    payloads = [
        native_core.list_notes(ids=[nid]),
        native_core.list_notes(tags=["t1"], with_fields=False),
        native_core.query("tag:t1", with_fields=True, limit=10),
        native_core.collection_info(["summary", "decks"], []),
        native_core.collection_info(["all"], ["Basic"]),
    ]
    for raw in payloads:
        canonical = json.dumps(
            json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        assert raw == canonical
        assert "null" not in raw

    meta = json.loads(payloads[1])
    assert "content" not in meta["notes"][0]
    subset = json.loads(payloads[3])
    assert set(subset) == {"summary", "decks"}


def test_note_embed_inputs_and_derived_rows(native_core):
    basic = native_core.notetype_id("Basic")
    nid = native_core.create_note(basic, DEFAULT_DECK, ['<img src="pic.png"> a diagram', ""], [])
    ((note_id, text, images),) = native_core.note_embed_inputs([nid])
    assert note_id == nid
    assert text == "Front: a diagram"
    assert images == ["pic.png"]
    rows = native_core.derived_field_rows([nid])
    assert rows == [(nid, "field", "Front", '<img src="pic.png"> a diagram')]
