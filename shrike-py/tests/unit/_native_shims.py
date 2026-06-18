"""The old note-type module functions, re-expressed over the native core.

A test-only shim: the runtime implementations moved to Rust at the #278
cutover; these keep the existing per-case unit tests (which exercise every
edge of the old surface) running against the native core with their bodies
untouched. Signatures mirror the retired shrike.note_types functions.
"""

from __future__ import annotations

import json
from typing import Any

from shrike_native import NativeInputError as NoteTypeOpError  # noqa: F401


def upsert_note_types(c: Any, note_types: list[dict]) -> list[dict]:
    return json.loads(c.upsert_note_types(json.dumps(note_types)))


def update_note_type_fields(c: Any, name: str, operations: list[dict]) -> dict:
    return json.loads(c.update_note_type_fields(name, json.dumps(operations)))


def update_note_type_templates(c: Any, name: str, operations: list[dict]) -> dict:
    return json.loads(c.update_note_type_templates(name, json.dumps(operations)))


def update_note_type_field_metadata(c: Any, name: str, updates: list[dict]) -> dict:
    return json.loads(c.update_note_type_field_metadata(name, json.dumps(updates)))


def find_and_replace_note_types(
    c: Any,
    name: str,
    *,
    search: str,
    replacement: str,
    regex: bool = False,
    match_case: bool = True,
    front: bool = True,
    back: bool = True,
    css: bool = True,
) -> dict:
    return json.loads(
        c.find_replace_note_types(name, search, replacement, regex, match_case, front, back, css)
    )
