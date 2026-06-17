#!/usr/bin/env python
"""Build the QA fixture collection from a declarative JSON corpus.

Reads ``collection.json`` (decks, note types, notes + tags) and writes a fresh
``collection.anki2`` by routing every note through the native core's
``upsert_notes`` — the same write path the server uses — so the fixture is built
exactly the way real notes are.

The corpus is the checked-in source of truth; the built collection is a
disposable, gitignored artifact regenerated on every QA launch. There is no
binary fixture in git.

Usage:
    python tests/manual/skill_quality/build_collection.py --out tests/manual/skill_quality/run/working.anki2
    python tests/manual/skill_quality/build_collection.py --out /tmp/c.anki2 --spec other.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without an editable install (e.g. a bare checkout).
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shrike.collection import CollectionWrapper  # noqa: E402


def build(spec_path: Path, out_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    notes = spec.get("notes", [])
    if not notes:
        print(f"!! no notes in {spec_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A fresh build every time: anki opens (and would reuse) an existing file.
    for suffix in ("", "-wal", "-shm"):
        leftover = out_path.with_name(out_path.name + suffix)
        if leftover.exists():
            leftover.unlink()

    wrapper = CollectionWrapper(str(out_path))
    try:
        # The same write path the server uses: the native core's upsert_notes,
        # which resolves decks/note types and returns the per-item result JSON.
        results = wrapper.run_sync(
            lambda core: json.loads(core.upsert_notes(json.dumps(notes), "error", False))
        )
    finally:
        wrapper.close()

    ok = [r for r in results if r.get("status") in ("created", "updated")]
    errors = [r for r in results if r.get("status") == "error"]
    decks = _deck_count(notes)
    print(f"Built {out_path}: {len(ok)} notes across {decks} decks, {len(errors)} errors")
    for err in errors:
        print(f"  ! note[{err.get('index')}]: {err.get('error')}", file=sys.stderr)
    return 1 if errors else 0


def _deck_count(notes: list[dict]) -> int:
    return len({n.get("deck") for n in notes if n.get("deck")})


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=here / "collection.json",
        help="Path to the JSON corpus (default: tests/manual/skill_quality/collection.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the built collection.anki2.",
    )
    args = parser.parse_args()
    return build(args.spec, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
