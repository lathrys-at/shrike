from __future__ import annotations

import functools
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from shrike.collection import CollectionWrapper
from shrike.index import VectorIndex

logger = logging.getLogger("shrike.tools")


class TemplateInput(BaseModel):
    name: str = Field(description="Template name (e.g., 'Recognition', 'Recall').")
    front: str = Field(description="Front side HTML. Use {{FieldName}} to insert field values.")
    back: str = Field(
        description=(
            "Back side HTML. Use {{FieldName}} for fields and "
            "{{FrontSide}} to insert the rendered front side."
        )
    )


class NoteInput(BaseModel):
    id: int | None = Field(
        default=None,
        description="Note ID. Present = update existing note, absent = create new note.",
    )
    deck: str | None = Field(
        default=None,
        description=(
            'Target deck (e.g., "Japanese::Vocabulary"). '
            "Required for new notes. On update, moves the note to this deck."
        ),
    )
    note_type: str | None = Field(
        default=None,
        description=(
            'Note type (e.g., "Basic", "Cloze"). '
            "Required for new notes. Cannot be changed on update."
        ),
    )
    fields: dict[str, str] | None = Field(
        default=None,
        description=(
            "Field key-value pairs matching the note type's field names. "
            "Required for new notes. On update, only specified fields are modified."
        ),
    )
    tags: list[str] | None = Field(
        default=None,
        description=(
            "Tags for the note. On create, these are the note's tags. "
            "On update, replaces all existing tags."
        ),
    )


class NoteTypeInput(BaseModel):
    id: int | None = Field(
        default=None,
        description="Note type ID. Present = update, absent = create.",
    )
    name: str | None = Field(
        default=None,
        description="Name for the note type. Required for new note types.",
    )
    fields: list[str] | None = Field(
        default=None,
        description=(
            "Ordered list of field names. Required for new note types. "
            "On update, replaces the full field list."
        ),
    )
    templates: list[TemplateInput] | None = Field(
        default=None,
        description="Card templates. Required for new note types.",
    )
    css: str | None = Field(
        default=None,
        description="CSS styling shared across all cards. Required for new note types.",
    )
    is_cloze: bool | None = Field(
        default=None,
        description="If true, this is a cloze deletion note type. Cannot be changed on update.",
    )


def _safe_tool(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            logger.exception("Unhandled error in %s", fn.__name__)
            return {"error": f"Internal error: {e}"}

    return wrapper


def register_tools(mcp: FastMCP, wrapper: CollectionWrapper) -> None:
    from shrike.note_types import upsert_note_types as _upsert_note_types

    index = VectorIndex()

    @mcp.tool()
    @_safe_tool
    def collection_info(
        include: list[str] | None = None,
        note_type_details: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get the structure and summary statistics of the Anki collection.

        Returns available note types with their field names, deck names with
        note counts, all tags in use, and scheduling statistics.

        Use this to orient yourself before creating or searching for notes —
        especially to discover which note types, fields, and decks exist.

        With no arguments, returns summaries of everything. Use `include` to
        request only a subset. Note type summaries include field names and
        type (standard/cloze) but not full template HTML or CSS — use
        `note_type_details` to request full definitions for specific note
        types when you need to inspect or author templates."""
        sections = include or ["note_types", "decks", "tags", "stats"]
        logger.info("collection_info sections=%s", ",".join(sections))
        return wrapper.get_collection_info(include, note_type_details)

    @mcp.tool()
    @_safe_tool
    def list_notes(
        ids: list[int] | None = None,
        deck: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
        modified_since: str | None = None,
        query: str | None = None,
        fields: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Retrieve notes matching structured filters.

        Filter by deck, tags, note type, note IDs, or modification date.
        Returns note metadata and field content.

        Use this for precise lookups: fetching specific notes by ID, listing
        a deck's contents, or filtering by exact criteria. For conceptual or
        fuzzy queries, use search_notes instead.

        At least one filter must be provided. Combine filters freely — they
        are ANDed together. Use `fields: "meta"` to return only metadata for
        large result sets. The response includes `total` (full match count);
        if more notes matched than `limit` allows, narrow your filters."""
        if limit < 1:
            limit = 1
        elif limit > 200:
            limit = 200

        if not any([ids, deck, tags, note_type, modified_since, query]):
            return {
                "error": (
                    "At least one filter (ids, deck, tags, note_type,"
                    " modified_since, or query) must be provided."
                ),
            }

        filters = [
            f
            for f in [
                f"deck={deck}" if deck else "",
                f"tags={tags}" if tags else "",
                f"type={note_type}" if note_type else "",
                f"ids={len(ids)}" if ids else "",
                f"since={modified_since}" if modified_since else "",
                f"query={query!r}" if query else "",
            ]
            if f
        ]
        logger.info("list_notes %s limit=%d", " ".join(filters), limit)

        result = wrapper.list_notes(
            ids=ids,
            deck=deck,
            tags=tags,
            note_type=note_type,
            modified_since=modified_since,
            query=query,
            fields_mode=fields or "full",
            limit=limit,
        )
        logger.info(
            "list_notes returned %d/%d notes",
            len(result.get("notes", [])),
            result.get("total", 0),
        )
        return result

    @mcp.tool()
    @_safe_tool
    def search_notes(
        queries: list[str] | None = None,
        ids: list[int] | None = None,
        top_k: int = 10,
        deck: str | None = None,
        tags: list[str] | None = None,
        exclude_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Semantic similarity search over the Anki collection.

        Accepts natural-language query strings, note IDs (to find
        conceptually similar notes), or both. Returns the top matches
        ranked by similarity score.

        Use this for conceptual queries that keyword search cannot handle:
        finding cards about a topic, checking if a concept is already
        covered before creating new cards, or exploring thematic clusters
        in the collection.

        At least one of `queries` or `ids` must be provided."""
        if not queries and not ids:
            return {"error": "At least one of queries or ids must be provided."}

        logger.info(
            "search_notes queries=%d ids=%d top_k=%d",
            len(queries or []),
            len(ids or []),
            top_k,
        )

        if not index.available:
            return {
                "results": [],
                "_message": (
                    "Semantic search is not available. "
                    "The vector index has not been built. "
                    "Use list_notes for structured filtering instead."
                ),
            }

        if top_k < 1:
            top_k = 1
        elif top_k > 50:
            top_k = 50

        return index.search(
            queries=queries,
            ids=ids,
            top_k=top_k,
            deck=deck,
            tags=tags,
            exclude_ids=exclude_ids,
        )

    @mcp.tool()
    @_safe_tool
    def upsert_notes(notes: list[NoteInput]) -> dict[str, Any]:
        """Create or update notes in bulk (1-100 per call).

        If a note object includes an `id`, the existing note is updated;
        if `id` is absent, a new note is created.

        For new notes, `deck`, `note_type`, and `fields` are required. For
        updates, only `id` and the properties being changed are needed —
        omitted properties are left unchanged.

        Duplicate detection is handled by the application and surfaced in
        its own UI, not controlled through this tool."""
        if len(notes) > 100:
            return {"error": "Maximum 100 notes per call."}

        creates = sum(1 for n in notes if n.id is None)
        updates = len(notes) - creates
        logger.info("upsert_notes count=%d (creates=%d, updates=%d)", len(notes), creates, updates)

        note_dicts = [n.model_dump(exclude_none=True) for n in notes]
        results = wrapper.upsert_notes(note_dicts)

        created = sum(1 for r in results if r.get("status") == "created")
        updated = sum(1 for r in results if r.get("status") == "updated")
        errors = sum(1 for r in results if r.get("status") == "error")
        logger.info(
            "upsert_notes completed: %d created, %d updated, %d errors",
            created,
            updated,
            errors,
        )

        changed_ids = [r["id"] for r in results if r.get("status") in ("created", "updated")]
        if changed_ids:
            index.on_notes_changed(changed_ids)

        return {"results": results}

    @mcp.tool()
    @_safe_tool
    def upsert_note_types(note_types: list[NoteTypeInput]) -> dict[str, Any]:
        """Create or update note type definitions (1-10 per call).

        A note type defines the schema for notes: its fields, card templates
        (front/back HTML), and shared CSS styling.

        If a note type object includes an `id`, the existing note type is
        updated; if `id` is absent, a new note type is created. For new note
        types, `name`, `fields`, `templates`, and `css` are required.

        Card templates use Anki's replacement syntax: {{FieldName}} inserts
        a field value, {{FrontSide}} on the back template inserts the
        rendered front side. Cloze note types use {{cloze:FieldName}}.

        Note: removing fields from an existing note type deletes that
        field's data from all notes of that type."""
        if len(note_types) > 10:
            return {"error": "Maximum 10 note types per call."}

        names = [nt.name or f"id={nt.id}" for nt in note_types]
        logger.info("upsert_note_types count=%d names=%s", len(note_types), ", ".join(names))

        nt_dicts = [nt.model_dump(exclude_none=True) for nt in note_types]
        results = _upsert_note_types(wrapper.col, nt_dicts)

        for r in results:
            status = r.get("status", "unknown")
            if status == "error":
                logger.warning(
                    "upsert_note_types failed for %s: %s", r.get("name", "?"), r["error"]
                )

        return {"results": results}

    @mcp.tool()
    @_safe_tool
    def delete_notes(ids: list[int]) -> dict[str, Any]:
        """Permanently delete notes and all their associated cards.

        This cannot be undone. Use list_notes or search_notes first to
        verify which notes will be deleted."""
        if len(ids) > 100:
            return {"error": "Maximum 100 note IDs per call."}

        logger.info("delete_notes requested=%d", len(ids))
        result = wrapper.delete_notes(ids)
        logger.info(
            "delete_notes completed: %d deleted, %d not found",
            len(result["deleted"]),
            len(result["not_found"]),
        )
        if result["deleted"]:
            index.on_notes_deleted(result["deleted"])
        return result
