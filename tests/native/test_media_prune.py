"""Media + prune: wrapper-vs-binding consistency (#278 step 5a; the
subprocess half now exercises the NATIVE-backed wrapper end to end — the local halves; the SSRF
URL-fetch path is step 5b under the security-review gate).

Cross-core: the same store/fetch/list/delete/check/prune sequence through
CollectionWrapper in a subprocess on a separate collection file, comparing the
result dicts (paths and media_dir stripped — they are per-collection)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

shrike_native = pytest.importorskip("shrike_native")

from .conftest import requires_anki_core  # noqa: E402

pytestmark = requires_anki_core

_PIP_SIDE = r"""
import asyncio, base64, json, sys
from shrike.collection import CollectionWrapper

async def main():
    w = CollectionWrapper(sys.argv[1])
    out = {}
    data = base64.b64encode(b"PNGDATA").decode()
    item = [{"filename": "pic.png", "data": data}]
    out["store"] = await w.store_media(item, allow_private_fetch=False)
    out["store_dup"] = await w.store_media(item, allow_private_fetch=False)
    await w.upsert_notes([{
        "note_type": "Basic", "deck": "Default",
        "fields": {"Front": '<img src="pic.png"> labelled', "Back": "kept"},
        "tags": ["usedtag"],
    }])
    # an unused media file + a note that prune will empty
    out["store_unused"] = await w.store_media(
        [{"filename": "junk.bin", "data": base64.b64encode(b"JUNK").decode()}],
        allow_private_fetch=False,
    )
    created = await w.upsert_notes([{
        "note_type": "Basic", "deck": "Default",
        "fields": {"Front": "temp", "Back": ""}, "tags": ["onlytag"],
    }])
    nid = created[0]["id"]
    await w.upsert_notes([{"id": nid, "fields": {"Front": "<b> </b>&nbsp;"}}])

    out["fetch"] = await w.fetch_media(["pic.png", "ghost.png"])
    for r in out["fetch"]:
        r.pop("path", None)
    listing = await w.list_media(pattern="*.png", limit=None)
    listing.pop("media_dir")
    out["list"] = listing
    check = await w.media_check()
    check.pop("media_dir")
    out["check"] = check
    preview, _ = await w.prune(
        unused_tags=True, empty_notes=True, empty_cards=True,
        unused_media=True, dry_run=True,
    )
    preview["empty_notes"]["removed"] = len(preview["empty_notes"]["removed"])
    out["preview"] = preview
    applied, removed = await w.prune(
        unused_tags=True, empty_notes=True, empty_cards=True,
        unused_media=True, dry_run=False,
    )
    applied["empty_notes"]["removed"] = len(applied["empty_notes"]["removed"])
    out["applied"] = applied
    out["removed_count"] = len(removed)
    out["delete"] = await w.delete_media(["pic.png", "nope.png"])
    w.close()
    print(json.dumps(out))

asyncio.run(main())
"""


def test_cross_core_media_prune_parity(tmp_path, native_core):
    pip_col = tmp_path / "pip" / "collection.anki2"
    pip_col.parent.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _PIP_SIDE, str(pip_col)],
        capture_output=True,
        text=True,
        check=True,
    )
    pip = json.loads(proc.stdout)

    # Same sequence natively.
    stored = json.loads(native_core.store_media_bytes(b"PNGDATA", filename="pic.png"))
    assert stored["filename"] == pip["store"][0]["filename"] == "pic.png"
    dup = json.loads(native_core.store_media_bytes(b"PNGDATA", filename="pic.png"))
    assert dup["filename"] == pip["store_dup"][0]["filename"] == "pic.png"
    native_core.upsert_notes(
        json.dumps(
            [
                {
                    "note_type": "Basic",
                    "deck": "Default",
                    "fields": {"Front": '<img src="pic.png"> labelled', "Back": "kept"},
                    "tags": ["usedtag"],
                }
            ]
        )
    )
    json.loads(native_core.store_media_bytes(b"JUNK", filename="junk.bin"))
    created = json.loads(
        native_core.upsert_notes(
            json.dumps(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "temp", "Back": ""},
                        "tags": ["onlytag"],
                    }
                ]
            )
        )
    )
    nid = created[0]["id"]
    native_core.upsert_notes(json.dumps([{"id": nid, "fields": {"Front": "<b> </b>&nbsp;"}}]))

    native_fetch = json.loads(native_core.fetch_media(["pic.png", "ghost.png"]))
    for r in native_fetch:
        r.pop("path", None)
    assert native_fetch == pip["fetch"]

    native_list = json.loads(native_core.list_media("*.png"))
    native_list.pop("media_dir")
    assert native_list == pip["list"]

    native_check = json.loads(native_core.media_check())
    native_check.pop("media_dir")
    assert native_check == pip["check"]

    native_preview = json.loads(native_core.prune())
    native_preview.pop("removed_note_ids")
    native_preview["empty_notes"]["removed"] = len(native_preview["empty_notes"]["removed"])
    assert native_preview == pip["preview"]

    native_applied = json.loads(native_core.prune(dry_run=False))
    removed = native_applied.pop("removed_note_ids")
    native_applied["empty_notes"]["removed"] = len(native_applied["empty_notes"]["removed"])
    assert native_applied == pip["applied"]
    assert len(removed) == pip["removed_count"] == 1
    assert removed == [nid]

    native_delete = json.loads(native_core.delete_media(["pic.png", "nope.png"]))
    assert native_delete == pip["delete"]
