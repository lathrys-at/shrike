from __future__ import annotations

import logging
from typing import Any

from anki.collection import Collection
from anki.consts import MODEL_CLOZE

logger = logging.getLogger("shrike.note_types")


def upsert_note_types(col: Collection, note_types: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for i, nt_input in enumerate(note_types):
        try:
            if "id" in nt_input and nt_input["id"] is not None:
                results.append(_update_note_type(col, nt_input))
            else:
                results.append(_create_note_type(col, nt_input))
        except Exception as e:
            results.append(
                {
                    "status": "error",
                    "index": i,
                    "error": str(e),
                }
            )
    return results


def _create_note_type(col: Collection, nt_input: dict[str, Any]) -> dict[str, Any]:
    name = nt_input.get("name")
    fields = nt_input.get("fields")
    templates = nt_input.get("templates")
    css = nt_input.get("css")

    if not name:
        raise ValueError("name is required for new note types")
    if not fields:
        raise ValueError("fields is required for new note types")
    if not templates:
        raise ValueError("templates is required for new note types")
    if css is None:
        raise ValueError("css is required for new note types")

    if col.models.by_name(name) is not None:
        raise ValueError(f"Note type '{name}' already exists")

    notetype = col.models.new(name)

    if nt_input.get("is_cloze"):
        notetype["type"] = MODEL_CLOZE

    notetype["css"] = css

    for field_name in fields:
        field = col.models.new_field(field_name)
        col.models.add_field(notetype, field)

    for tmpl_input in templates:
        tmpl = col.models.new_template(tmpl_input["name"])
        tmpl["qfmt"] = tmpl_input["front"]
        tmpl["afmt"] = tmpl_input["back"]
        col.models.add_template(notetype, tmpl)

    changes = col.models.add(notetype)

    logger.debug(
        "Created note type %r (id=%d, fields=%d, templates=%d)",
        name,
        changes.id,
        len(fields),
        len(templates),
    )
    return {
        "status": "created",
        "id": changes.id,
        "name": name,
    }


def _update_note_type(col: Collection, nt_input: dict[str, Any]) -> dict[str, Any]:
    nt_id = nt_input["id"]
    notetype = col.models.get(nt_id)
    if notetype is None:
        raise ValueError(f"Note type with ID {nt_id} not found")

    if "is_cloze" in nt_input and nt_input["is_cloze"] is not None:
        current_is_cloze = notetype.get("type") == MODEL_CLOZE
        if nt_input["is_cloze"] != current_is_cloze:
            raise ValueError("Cannot change a note type between standard and cloze")

    if "name" in nt_input and nt_input["name"] is not None:
        notetype["name"] = nt_input["name"]

    if "css" in nt_input and nt_input["css"] is not None:
        notetype["css"] = nt_input["css"]

    if "fields" in nt_input and nt_input["fields"] is not None:
        _set_fields(col, notetype, nt_input["fields"])

    if "templates" in nt_input and nt_input["templates"] is not None:
        _set_templates(col, notetype, nt_input["templates"])

    col.models.update_dict(notetype)

    logger.debug("Updated note type %r (id=%d)", notetype["name"], nt_id)
    return {
        "status": "updated",
        "id": nt_id,
        "name": notetype["name"],
    }


def _set_fields(col: Collection, notetype: dict[str, Any], names: list[str]) -> None:
    """Replace a note type's field list, preserving note data by position.

    Anki migrates a note's field *values* by field ordinal: a field keeps its
    data as long as the field at that ordinal survives. Rebuilding the list from
    fresh ``new_field`` objects (ord unset) makes the backend treat every field
    as newly added and *every existing field as removed*, blanking all note data
    for the type. So we reuse the existing field dicts in place — renaming the
    ones whose position survives, appending new fields only for added positions,
    and dropping the tail for removed ones (the only positions whose data Anki
    discards). This makes a whole-list field replace data-safe, matching Anki's
    "fields are keyed by position" rule. (A true by-identity reorder is a
    separate, explicit operation — see #76; here a reordered name list is
    interpreted positionally, i.e. as renames, which is non-destructive.)
    """
    old = notetype["flds"]
    new = []
    for i, name in enumerate(names):
        if i < len(old):
            field = old[i]
            field["name"] = name
        else:
            field = col.models.new_field(name)
        new.append(field)
    notetype["flds"] = new


def _set_templates(
    col: Collection, notetype: dict[str, Any], templates: list[dict[str, Any]]
) -> None:
    """Replace a note type's templates, preserving existing cards by position.

    Same hazard as ``_set_fields``: a card belongs to a template by ordinal, so
    rebuilding ``tmpls`` from fresh ``new_template`` objects makes the backend
    drop every existing template and the cards generated from it — re-sending the
    *same* templates deletes all of a note's cards (and their scheduling history).
    Reuse the existing template dicts in place, updating name/front/back, append
    only for added positions, and drop the tail for removed ones (whose cards are
    intentionally removed).
    """
    old = notetype["tmpls"]
    new = []
    for i, tmpl_input in enumerate(templates):
        tmpl = old[i] if i < len(old) else col.models.new_template(tmpl_input["name"])
        tmpl["name"] = tmpl_input["name"]
        tmpl["qfmt"] = tmpl_input["front"]
        tmpl["afmt"] = tmpl_input["back"]
        new.append(tmpl)
    notetype["tmpls"] = new
