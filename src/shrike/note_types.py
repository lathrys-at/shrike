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


class FieldOpError(ValueError):
    """An explicit field operation was invalid (unknown field, name clash, etc.).

    A plain ``ValueError`` subclass so the tool layer can translate it into a
    ``ToolInputError`` (logged without a traceback — it's caller error).
    """


def update_note_type_fields(
    col: Collection, note_type_name: str, operations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply explicit, identity-based field operations to a note type.

    Unlike ``upsert_note_types``' position-keyed whole-list replace, these
    operations are addressed by field *name*, so they can express a true move
    (``reposition``), a non-trailing ``remove``, or an insert at a position —
    all via Anki's data-safe primitives, which migrate note data by field
    identity. Operations apply in order; ``rename`` then a later op referencing
    the new name is valid.

    The whole call is atomic: the sequence is validated against a simulated
    field list first, so an invalid op fails without mutating anything. Only
    once every op is known-valid are the real primitives applied to one
    in-memory notetype and persisted with a single ``update_dict``.
    """
    notetype = col.models.by_name(note_type_name)
    if notetype is None:
        raise FieldOpError(f"Note type '{note_type_name}' not found")

    # Validate the whole sequence first (atomic: nothing applied if any op is
    # bad). `sim` tracks field names as each op would leave them.
    sim = [f["name"] for f in notetype["flds"]]
    for i, op in enumerate(operations):
        _simulate_field_op(sim, op, i)

    # `by_name` returned the live notetype dict; the Anki primitives mutate it
    # (and its `flds`) in place, so apply them all to it and persist once.
    for op in operations:
        _apply_field_op(col, notetype, op)
    col.models.update_dict(notetype)

    final = [f["name"] for f in notetype["flds"]]
    logger.debug(
        "update_note_type_fields %r: applied %d op(s) -> %s",
        note_type_name,
        len(operations),
        final,
    )
    return {"id": notetype["id"], "name": note_type_name, "fields": final}


def _simulate_field_op(sim: list[str], op: dict[str, Any], i: int) -> None:
    kind = op["op"]
    if kind == "add":
        name = op["name"]
        if name in sim:
            raise FieldOpError(f"op {i} (add): field '{name}' already exists")
        pos = op.get("position")
        if pos is None:
            sim.append(name)
        elif not 0 <= pos <= len(sim):
            raise FieldOpError(f"op {i} (add): position {pos} out of range 0..{len(sim)}")
        else:
            sim.insert(pos, name)
    elif kind == "remove":
        name = op["name"]
        if name not in sim:
            raise FieldOpError(f"op {i} (remove): field '{name}' not found")
        if len(sim) == 1:
            raise FieldOpError(f"op {i} (remove): a note type must keep at least one field")
        sim.remove(name)
    elif kind == "rename":
        name, new = op["name"], op["new_name"]
        if name not in sim:
            raise FieldOpError(f"op {i} (rename): field '{name}' not found")
        if new != name and new in sim:
            raise FieldOpError(f"op {i} (rename): field '{new}' already exists")
        sim[sim.index(name)] = new
    elif kind == "reposition":
        name, pos = op["name"], op["position"]
        if name not in sim:
            raise FieldOpError(f"op {i} (reposition): field '{name}' not found")
        if not 0 <= pos < len(sim):
            raise FieldOpError(
                f"op {i} (reposition): position {pos} out of range 0..{len(sim) - 1}"
            )
        sim.remove(name)
        sim.insert(pos, name)


def _apply_field_op(col: Collection, notetype: dict[str, Any], op: dict[str, Any]) -> None:
    # Look fields up by their current name each call, so an op that follows a
    # rename/reposition sees the up-to-date list.
    by_name = {f["name"]: f for f in notetype["flds"]}
    kind = op["op"]
    if kind == "add":
        field = col.models.new_field(op["name"])
        col.models.add_field(notetype, field)
        if op.get("position") is not None:
            col.models.reposition_field(notetype, field, op["position"])
    elif kind == "remove":
        col.models.remove_field(notetype, by_name[op["name"]])
    elif kind == "rename":
        col.models.rename_field(notetype, by_name[op["name"]], op["new_name"])
    elif kind == "reposition":
        col.models.reposition_field(notetype, by_name[op["name"]], op["position"])
