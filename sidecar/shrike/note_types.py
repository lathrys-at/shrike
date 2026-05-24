from __future__ import annotations

import logging
from typing import Any

from anki.collection import Collection
from anki.consts import MODEL_CLOZE

logger = logging.getLogger("shrike")


def upsert_note_types(
    col: Collection, note_types: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results = []
    for i, nt_input in enumerate(note_types):
        try:
            if "id" in nt_input and nt_input["id"] is not None:
                results.append(_update_note_type(col, nt_input))
            else:
                results.append(_create_note_type(col, nt_input))
        except Exception as e:
            results.append({
                "status": "error",
                "index": i,
                "error": str(e),
            })
    return results


def _create_note_type(
    col: Collection, nt_input: dict[str, Any]
) -> dict[str, Any]:
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

    return {
        "status": "created",
        "id": changes.id,
        "name": name,
    }


def _update_note_type(
    col: Collection, nt_input: dict[str, Any]
) -> dict[str, Any]:
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
        notetype["flds"] = []
        for field_name in nt_input["fields"]:
            field = col.models.new_field(field_name)
            col.models.add_field(notetype, field)

    if "templates" in nt_input and nt_input["templates"] is not None:
        notetype["tmpls"] = []
        for tmpl_input in nt_input["templates"]:
            tmpl = col.models.new_template(tmpl_input["name"])
            tmpl["qfmt"] = tmpl_input["front"]
            tmpl["afmt"] = tmpl_input["back"]
            col.models.add_template(notetype, tmpl)

    col.models.update_dict(notetype)

    return {
        "status": "updated",
        "id": nt_id,
        "name": notetype["name"],
    }
