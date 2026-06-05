from __future__ import annotations

import logging
import re
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
    "fields are keyed by position" rule.

    Because the replace is positional, it can only express a rename-in-place, an
    append, or a trailing remove. Anything that *moves* an existing field —
    a reorder, an insert before another field, or a non-trailing remove — would
    silently re-label note data (the value stays in its slot while the name on
    that slot changes). We refuse those: see ``_reject_unsound_field_replace``.
    Use ``update_note_type_fields`` for identity-based moves/inserts/removes.
    """
    old = [f["name"] for f in notetype["flds"]]
    _reject_unsound_positional_replace(
        old, names, what="field", mislabels="note data", mover_tool="update_note_type_fields"
    )

    old_flds = notetype["flds"]
    new = []
    for i, name in enumerate(names):
        if i < len(old_flds):
            field = old_flds[i]
            field["name"] = name
        else:
            field = col.models.new_field(name)
        new.append(field)
    notetype["flds"] = new


def _reject_unsound_positional_replace(
    old: list[str], new: list[str], *, what: str, mislabels: str, mover_tool: str
) -> None:
    """Reject a positional field/template replace that would mislabel data.

    A positional replace renames the entry *at position i* to ``new[i]``; the
    data (a field's note values, a template's cards) never leaves its slot.
    That's sound only while every existing name stays at its current position.
    If an existing name appears at a *different* position in ``new``, the caller
    is really asking to move it (reorder, insert, or non-trailing remove, which
    shifts the names after it) — which positionally becomes a silent re-label.
    Refuse it and point at the explicit identity-based tool.
    """
    old_index = {name: i for i, name in enumerate(old)}
    for i, name in enumerate(new):
        if name in old_index and old_index[name] != i:
            raise ValueError(
                f"{what.capitalize()} '{name}' would move from position "
                f"{old_index[name]} to {i}. upsert_note_types replaces {what}s by "
                f"position — it can only rename a {what} in place, append new {what}s, "
                f"or drop trailing {what}s; moving, inserting, or removing a "
                f"non-trailing {what} this way would silently mislabel {mislabels}. "
                f"Use {mover_tool} (reposition / add / remove / rename) for that."
            )


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

    Like ``_set_fields``, the positional replace can only rename/edit in place,
    append, or drop trailing templates. A move/insert/non-trailing remove would
    silently re-label cards, so it's refused — use ``update_note_type_templates``.
    """
    old = notetype["tmpls"]
    _reject_unsound_positional_replace(
        [t["name"] for t in old],
        [t["name"] for t in templates],
        what="template",
        mislabels="cards (and their scheduling history)",
        mover_tool="update_note_type_templates",
    )
    new = []
    for i, tmpl_input in enumerate(templates):
        tmpl = old[i] if i < len(old) else col.models.new_template(tmpl_input["name"])
        tmpl["name"] = tmpl_input["name"]
        tmpl["qfmt"] = tmpl_input["front"]
        tmpl["afmt"] = tmpl_input["back"]
        new.append(tmpl)
    notetype["tmpls"] = new


class NoteTypeOpError(ValueError):
    """An explicit field/template operation was invalid (unknown name, clash, etc.).

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
        raise NoteTypeOpError(f"Note type '{note_type_name}' not found")

    # Validate the whole sequence first (atomic: nothing applied if any op is
    # bad). `sim` tracks field names as each op would leave them.
    sim = [f["name"] for f in notetype["flds"]]
    for i, op in enumerate(operations):
        _simulate_struct_op(sim, op, i, what="field")

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


def update_note_type_templates(
    col: Collection, note_type_name: str, operations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply explicit, identity-based card-template operations to a note type.

    The template counterpart of ``update_note_type_fields``. Templates are
    addressed by *name*, so these ops can move a template, insert one, or remove
    a non-trailing one — all via Anki's data-safe primitives, which migrate
    *cards* by template identity (a card belongs to its template by ordinal, so a
    reposition carries the card and its scheduling along; a remove deletes only
    that template's cards). A ``rename`` only changes the template's label — cards
    key by ordinal, not name, so no card is touched. (To edit a template's
    front/back HTML in place, use ``upsert_note_types``, whose positional replace
    is data-safe for in-place edits.)

    Atomic and ordered, exactly like ``update_note_type_fields``: the sequence is
    validated against a simulated template-name list first; only if every op is
    valid are the primitives applied to one in-memory notetype and persisted with
    a single ``update_dict``.
    """
    notetype = col.models.by_name(note_type_name)
    if notetype is None:
        raise NoteTypeOpError(f"Note type '{note_type_name}' not found")

    sim = [t["name"] for t in notetype["tmpls"]]
    for i, op in enumerate(operations):
        _simulate_struct_op(sim, op, i, what="template")

    for op in operations:
        _apply_template_op(col, notetype, op)
    col.models.update_dict(notetype)

    final = [t["name"] for t in notetype["tmpls"]]
    logger.debug(
        "update_note_type_templates %r: applied %d op(s) -> %s",
        note_type_name,
        len(operations),
        final,
    )
    return {"id": notetype["id"], "name": note_type_name, "templates": final}


def _simulate_struct_op(sim: list[str], op: dict[str, Any], i: int, *, what: str) -> None:
    """Validate one field/template op against a simulated name list (in place).

    ``what`` is "field" or "template" — only used for error wording. The ops
    share a shape (add/remove/rename/reposition by name), so one simulator
    serves both.
    """
    kind = op["op"]
    if kind == "add":
        name = op["name"]
        if name in sim:
            raise NoteTypeOpError(f"op {i} (add): {what} '{name}' already exists")
        pos = op.get("position")
        if pos is None:
            sim.append(name)
        elif not 0 <= pos <= len(sim):
            raise NoteTypeOpError(f"op {i} (add): position {pos} out of range 0..{len(sim)}")
        else:
            sim.insert(pos, name)
    elif kind == "remove":
        name = op["name"]
        if name not in sim:
            raise NoteTypeOpError(f"op {i} (remove): {what} '{name}' not found")
        if len(sim) == 1:
            raise NoteTypeOpError(f"op {i} (remove): a note type must keep at least one {what}")
        sim.remove(name)
    elif kind == "rename":
        name, new = op["name"], op["new_name"]
        if name not in sim:
            raise NoteTypeOpError(f"op {i} (rename): {what} '{name}' not found")
        if new != name and new in sim:
            raise NoteTypeOpError(f"op {i} (rename): {what} '{new}' already exists")
        sim[sim.index(name)] = new
    elif kind == "reposition":
        name, pos = op["name"], op["position"]
        if name not in sim:
            raise NoteTypeOpError(f"op {i} (reposition): {what} '{name}' not found")
        if not 0 <= pos < len(sim):
            raise NoteTypeOpError(
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


def _apply_template_op(col: Collection, notetype: dict[str, Any], op: dict[str, Any]) -> None:
    by_name = {t["name"]: t for t in notetype["tmpls"]}
    kind = op["op"]
    if kind == "add":
        tmpl = col.models.new_template(op["name"])
        tmpl["qfmt"] = op["front"]
        tmpl["afmt"] = op["back"]
        col.models.add_template(notetype, tmpl)
        if op.get("position") is not None:
            col.models.reposition_template(notetype, tmpl, op["position"])
    elif kind == "remove":
        col.models.remove_template(notetype, by_name[op["name"]])
    elif kind == "rename":
        # Templates key cards by ordinal, not name, so a rename is a pure label
        # change — no Anki primitive, no card migration.
        by_name[op["name"]]["name"] = op["new_name"]
    elif kind == "reposition":
        col.models.reposition_template(notetype, by_name[op["name"]], op["position"])


def _subn_text(
    value: str, search: str, replacement: str, *, regex: bool, match_case: bool
) -> tuple[str, int]:
    """Substitute in one string; return (new_value, replacements_made).

    Literal mode escapes ``search`` and treats ``replacement`` as plain text (no
    ``\\1`` group-ref interpretation, via a constant lambda). Regex mode honours
    capture refs in ``replacement``. ``match_case=False`` adds ``re.IGNORECASE``.
    """
    flags = 0 if match_case else re.IGNORECASE
    if regex:
        return re.subn(search, replacement, value, flags=flags)
    return re.subn(re.escape(search), lambda _m: replacement, value, flags=flags)


def find_and_replace_note_types(
    col: Collection,
    note_type_name: str,
    *,
    search: str,
    replacement: str,
    regex: bool = False,
    match_case: bool = True,
    front: bool = True,
    back: bool = True,
    css: bool = True,
) -> dict[str, Any]:
    """Find/replace literal-or-regex text inside one note type's templates and CSS.

    Walks the model definition — each card template's front (``qfmt``) and back
    (``afmt``) HTML and the shared ``css`` — substituting ``search`` with
    ``replacement`` in whichever of ``front``/``back``/``css`` are enabled. This
    edits the *note type*, not note field values: no note is touched and every
    embedding vector stays valid (the caller advances ``col.mod`` without
    re-embedding). Typical use is fixing a ``{{OldField}}`` reference across a
    model's templates after a field rename, or a CSS class/colour swap.

    The model is persisted once (a single ``update_dict``) only if at least one
    replacement was made. Returns the total replacement count, the names of the
    templates whose front/back changed, and whether the CSS changed.
    """
    notetype = col.models.by_name(note_type_name)
    if notetype is None:
        raise NoteTypeOpError(f"Note type '{note_type_name}' not found")

    if regex:
        try:
            re.compile(search, 0 if match_case else re.IGNORECASE)
        except re.error as e:
            raise NoteTypeOpError(f"invalid regex: {e}") from e

    def sub(value: str) -> tuple[str, int]:
        return _subn_text(value, search, replacement, regex=regex, match_case=match_case)

    total = 0
    templates_changed: list[str] = []
    for tmpl in notetype["tmpls"]:
        changed = 0
        if front:
            new, n = sub(tmpl["qfmt"])
            if n:
                tmpl["qfmt"], changed = new, changed + n
        if back:
            new, n = sub(tmpl["afmt"])
            if n:
                tmpl["afmt"], changed = new, changed + n
        if changed:
            templates_changed.append(tmpl["name"])
            total += changed

    css_changed = False
    if css:
        new, n = sub(notetype["css"])
        if n:
            notetype["css"], css_changed, total = new, True, total + n

    if total:
        col.models.update_dict(notetype)

    logger.debug(
        "find_and_replace_note_types %r: %d replacement(s), templates=%s css=%s",
        note_type_name,
        total,
        templates_changed,
        css_changed,
    )
    return {
        "id": notetype["id"],
        "name": note_type_name,
        "replacements": total,
        "templates_changed": templates_changed,
        "css_changed": css_changed,
    }
