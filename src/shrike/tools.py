from __future__ import annotations

import functools
import inspect
import logging
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from shrike.collection import CollectionWrapper
from shrike.index import IndexState, VectorIndex
from shrike.schemas import (
    CollectionInfo,
    DeleteNotesResponse,
    DeleteNoteTypesResponse,
    ListNotesResponse,
    NoteInput,
    NoteTypeInput,
    SearchResponse,
    UpsertNotesResponse,
    UpsertNoteTypesResponse,
)

logger = logging.getLogger("shrike.tools")


def _safe_tool(fn: Any) -> Any:
    """Wrap a tool so unhandled exceptions return an error payload.

    The returned ``{"error": ...}`` dict is coerced by FastMCP into the tool's
    declared response model (every response model defaults all non-error fields),
    so the catch-all stays valid against the generated ``outputSchema``.

    The wrapped function's docstring is dedented with ``inspect.cleandoc`` so the
    tool description FastMCP advertises to clients has no source indentation.
    """
    cleaned_doc = inspect.cleandoc(fn.__doc__) if fn.__doc__ else None

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Unhandled error in %s", fn.__name__)
                return {"error": f"Internal error: {e}"}

        async_wrapper.__doc__ = cleaned_doc
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            logger.exception("Unhandled error in %s", fn.__name__)
            return {"error": f"Internal error: {e}"}

    wrapper.__doc__ = cleaned_doc
    return wrapper


def register_tools(
    mcp: FastMCP,
    wrapper: CollectionWrapper,
    index: VectorIndex | None = None,
) -> None:
    from shrike.note_types import upsert_note_types as _upsert_note_types

    @mcp.tool()
    @_safe_tool
    async def collection_info(
        include: Annotated[
            list[Literal["summary", "note_types", "decks", "tags", "stats", "all"]] | None,
            Field(
                description=(
                    'Sections to return. Any combination of "summary" (counts, dates, '
                    'path), "note_types" (note types and their fields), "decks" (deck '
                    'hierarchy with note counts), "tags" (all tags in use), "stats" (card '
                    'counts, due counts, per-deck summaries), or "all" for everything. '
                    'Defaults to ["summary"].'
                )
            ),
        ] = None,
        note_type_details: Annotated[
            list[str] | None,
            Field(
                description=(
                    "List of note type names to return full definitions for, including "
                    "card template HTML and CSS styling. Omit to return only summaries "
                    "(field names and type)."
                )
            ),
        ] = None,
    ) -> CollectionInfo:
        """Get the structure and summary statistics of the Anki collection.

        Returns available note types with their field names, deck names with
        note counts, all tags in use, and scheduling statistics.

        Use this to orient yourself before creating or searching for notes —
        especially to discover which note types, fields, and decks exist.

        With no arguments, returns a compact summary (counts, dates, path).
        Use `include` to request specific sections: "summary", "note_types",
        "decks", "tags", "stats", or "all" for everything. Note type summaries
        include field names and
        type (standard/cloze) but not full template HTML or CSS — use
        `note_type_details` to request full definitions for specific note
        types when you need to inspect or author templates."""
        include_list: list[str] | None = [str(s) for s in include] if include else None
        logger.info("collection_info sections=%s", ",".join(include_list or ["summary"]))
        result = await wrapper.get_collection_info(include_list, note_type_details)
        return CollectionInfo.model_validate(result)

    @mcp.tool()
    @_safe_tool
    async def list_notes(
        ids: Annotated[
            list[int] | None, Field(description="Specific note IDs to retrieve.")
        ] = None,
        deck: Annotated[
            str | None,
            Field(
                description=(
                    'Filter to notes in this deck. Use "::" for nested decks '
                    '(e.g., "Japanese::Vocabulary"). Includes child decks.'
                )
            ),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(
                description=(
                    'Filter to notes having all of these tags. Prefix with "-" to '
                    'exclude (e.g., ["-leech", "verb"] matches notes tagged "verb" '
                    'but not "leech").'
                )
            ),
        ] = None,
        note_type: Annotated[
            str | None,
            Field(description='Filter to notes using this note type (e.g., "Basic", "Cloze").'),
        ] = None,
        modified_since: Annotated[
            str | None,
            Field(
                description=(
                    "ISO 8601 date or datetime. Only return notes modified after this "
                    'time (e.g., "2026-05-01" or "2026-05-01T14:00:00Z").'
                )
            ),
        ] = None,
        query: Annotated[
            str | None,
            Field(
                description=(
                    'Raw Anki search query for advanced filtering (e.g., "is:due", '
                    '"prop:ivl>=30"). Combined with other filters via AND.'
                )
            ),
        ] = None,
        fields: Annotated[
            Literal["full", "meta"] | None,
            Field(
                description=(
                    '"full" (default) returns all field content. "meta" returns only '
                    "note ID, note type, deck, tags, and modification time — useful for "
                    "large result sets."
                )
            ),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, le=200, description="Maximum notes to return. Default 50.")
        ] = 50,
    ) -> ListNotesResponse:
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
        if not any([ids, deck, tags, note_type, modified_since, query]):
            return ListNotesResponse(
                error=(
                    "At least one filter (ids, deck, tags, note_type,"
                    " modified_since, or query) must be provided."
                )
            )

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

        result = await wrapper.list_notes(
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
        return ListNotesResponse.model_validate(result)

    @mcp.tool()
    @_safe_tool
    async def search_notes(
        queries: Annotated[
            list[str] | None,
            Field(description="Natural-language search strings, each matched independently."),
        ] = None,
        ids: Annotated[
            list[int] | None,
            Field(
                description=(
                    "Note IDs to use as search anchors — returns notes semantically "
                    "similar to these existing notes. Source notes are excluded from results."
                )
            ),
        ] = None,
        top_k: Annotated[
            int,
            Field(ge=1, le=50, description="Maximum results per query or source ID. Default 10."),
        ] = 10,
        threshold: Annotated[
            float,
            Field(
                ge=0.0,
                le=1.0,
                description="Minimum similarity score for a match to be included. Default 0.5.",
            ),
        ] = 0.5,
        deck: Annotated[
            str | None,
            Field(description="Restrict search to notes in this deck (includes child decks)."),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(description="Restrict search to notes matching all of these tags."),
        ] = None,
        exclude_ids: Annotated[
            list[int] | None,
            Field(description="Additional note IDs to exclude from results."),
        ] = None,
    ) -> SearchResponse:
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
            return SearchResponse(error="At least one of queries or ids must be provided.")

        logger.info(
            "search_notes queries=%d ids=%d top_k=%d threshold=%.2f",
            len(queries or []),
            len(ids or []),
            top_k,
            threshold,
        )

        if index is None or index.state == IndexState.UNAVAILABLE:
            return SearchResponse(
                message=(
                    "Semantic search is not available — the embedding service is not "
                    "running. Start it with 'shrike embedding start', or use list_notes "
                    "for structured filtering instead."
                )
            )

        if index.state == IndexState.BUILDING:
            indexed, total = index.build_progress
            return SearchResponse(
                message=(
                    f"The vector index is building ({indexed}/{total} notes indexed). "
                    "Try again shortly."
                )
            )

        if index.state == IndexState.ERROR:
            return SearchResponse(
                message=(
                    "The vector index encountered an error during the last build. "
                    "It will retry on next server restart. "
                    "Use list_notes for structured filtering instead."
                )
            )

        if not index.available:
            return SearchResponse(
                message=(
                    "Semantic search is not available. "
                    "The vector index has not been built. "
                    "Use list_notes for structured filtering instead."
                )
            )

        exclude_set = set(exclude_ids or [])

        all_query_texts: list[str] = []
        sources: list[str] = []

        for q in queries or []:
            all_query_texts.append(q)
            sources.append(q)

        if ids:
            note_texts = await wrapper.note_texts_for_embedding(ids)
            for nid, text in zip(ids, note_texts, strict=True):
                if text:
                    all_query_texts.append(text)
                    sources.append(f"note #{nid}")
                    exclude_set.add(nid)

        if not all_query_texts:
            return SearchResponse(message="No valid queries or note IDs to search.")

        # Over-fetch to cover excluded ids. When a deck/tag filter is set, the
        # filtering happens post-hoc, so widen the window aggressively — otherwise
        # a deck-scoped search whose nearest neighbors sit outside the deck can
        # silently under-return. (Heuristic: still possible to under-return if the
        # in-scope notes rank very deep; documented in docs/mcp-tools.md.)
        fetch_k = top_k + len(exclude_set)
        if deck or tags:
            fetch_k = max(fetch_k, top_k * 10)
            if index.size:
                fetch_k = min(fetch_k, index.size)
        raw_results = index.search(all_query_texts, top_k=fetch_k)

        results: list[dict[str, Any]] = []
        for source, matches in zip(sources, raw_results, strict=True):
            enriched: list[dict[str, Any]] = []
            for m in matches:
                nid = m["note_id"]
                if nid in exclude_set:
                    continue

                score = round(1.0 - m["distance"], 3)
                if score < threshold:
                    break

                try:
                    note_data = await wrapper.note_to_dict(nid, "full")
                except Exception:
                    logger.debug("search_notes: skipping unreadable note %s", nid, exc_info=True)
                    continue

                if deck and note_data.get("deck") != deck:
                    continue
                if tags:
                    note_tags = set(note_data.get("tags", []))
                    if not all(t in note_tags for t in tags):
                        continue

                enriched.append(
                    {
                        **note_data,
                        "score": score,
                    }
                )

                if len(enriched) >= top_k:
                    break

            results.append({"source": source, "matches": enriched})

        logger.info(
            "search_notes returned %d groups, %d total matches",
            len(results),
            sum(len(r["matches"]) for r in results),
        )
        return SearchResponse.model_validate({"results": results})

    @mcp.tool()
    @_safe_tool
    async def upsert_notes(
        notes: Annotated[
            list[NoteInput],
            Field(
                min_length=1,
                max_length=100,
                description="Array of note objects to create or update.",
            ),
        ],
        top_k_neighbors: Annotated[
            int,
            Field(
                ge=0,
                le=20,
                description=(
                    "Maximum similar-note neighbors to return per result. Default 5. "
                    "Set to 0 to disable neighbor lookup."
                ),
            ),
        ] = 5,
        neighbor_threshold: Annotated[
            float,
            Field(
                ge=0.0,
                le=1.0,
                description="Minimum cosine similarity for a neighbor to be included. Default 0.5.",
            ),
        ] = 0.5,
    ) -> UpsertNotesResponse:
        """Create or update notes in bulk (1-100 per call).

        If a note object includes an `id`, the existing note is updated;
        if `id` is absent, a new note is created.

        For new notes, `deck`, `note_type`, and `fields` are required. For
        updates, only `id` and the properties being changed are needed —
        omitted properties are left unchanged.

        When a vector index is available, each result includes `neighbors`:
        the most similar existing notes ranked by cosine similarity, filtered
        to those above `neighbor_threshold` (default 0.5) and capped at
        `top_k_neighbors` (default 5). Use these for tag consistency (adopt
        tags from nearby notes), detecting near-duplicates (high scores
        suggest overlap), or understanding where a new note sits in the
        collection. Neighbors include note ID, similarity score, and tags —
        use list_notes or search_notes to inspect content if needed.

        If the index update fails transiently (e.g. the embedding service is
        briefly unavailable), the notes are still saved but `neighbors` is
        omitted. Each affected result is flagged `neighbors_unavailable: true`
        and the response carries a top-level `message`. Recover the exact same
        neighbor data afterward with search_notes(ids=[<note id>]) — it embeds
        the same note text against the same index, so the result is identical
        to what would have been attached here."""
        creates = sum(1 for n in notes if n.id is None)
        updates = len(notes) - creates
        logger.info("upsert_notes count=%d (creates=%d, updates=%d)", len(notes), creates, updates)

        note_dicts = [n.model_dump(exclude_none=True) for n in notes]
        results = await wrapper.upsert_notes(note_dicts)

        created = sum(1 for r in results if r.get("status") == "created")
        updated = sum(1 for r in results if r.get("status") == "updated")
        errors = sum(1 for r in results if r.get("status") == "error")
        logger.info(
            "upsert_notes completed: %d created, %d updated, %d errors",
            created,
            updated,
            errors,
        )

        if index and index.available:
            changed_ids = [r["id"] for r in results if r.get("status") in ("created", "updated")]
            if changed_ids:
                # Index maintenance is best-effort: the notes are already
                # committed to the collection, so a failure here must never
                # turn a successful upsert into an error response. Keep the
                # neighbor lookup inside this try for the same reason —
                # otherwise an embedding failure leaves `texts` unbound.
                neighbors_ok = False
                try:
                    texts = await wrapper.note_texts_for_embedding(changed_ids)
                    index.add(changed_ids, texts)
                    index.col_mod = await wrapper.run(lambda c: c.mod)
                    logger.debug("Index updated: %d vectors added/replaced", len(changed_ids))
                    neighbors_ok = await _attach_neighbors(
                        results, changed_ids, texts, top_k_neighbors, neighbor_threshold
                    )
                except Exception:
                    logger.warning("Failed to update index after upsert", exc_info=True)

                if not neighbors_ok:
                    # Notes are saved, but neighbors could not be computed
                    # (transient index/embedding hiccup). Flag each affected
                    # result and point the caller at the recovery path: the
                    # exact same neighbor data is reproducible via a similarity
                    # search keyed on the note's own ID. We only reach here when
                    # the index was available, so search_notes(ids=...) is a
                    # viable retry.
                    pending = [
                        r["id"]
                        for r in results
                        if r.get("status") in ("created", "updated") and "neighbors" not in r
                    ]
                    for r in results:
                        if r.get("id") in pending:
                            r["neighbors_unavailable"] = True
                    if pending:
                        logger.info(
                            "Neighbors unavailable for %d note(s); caller can retry via "
                            "search_notes(ids=%s)",
                            len(pending),
                            pending,
                        )
                        return UpsertNotesResponse.model_validate(
                            {
                                "results": results,
                                "message": (
                                    "Notes were saved, but the vector index update failed, "
                                    "so neighbors could not be computed. Retry with "
                                    f"search_notes(ids={pending}) to fetch the same "
                                    "neighbor data."
                                ),
                            }
                        )

        return UpsertNotesResponse.model_validate({"results": results})

    async def _attach_neighbors(
        results: list[dict[str, Any]],
        changed_ids: list[int],
        texts: list[str],
        top_k: int,
        threshold: float,
    ) -> bool:
        """Search for similar notes and attach neighbors to each upsert result.

        Returns True if the neighbor search completed (each created/updated
        result now carries a `neighbors` list, possibly empty), False if it
        failed — in which case the caller should signal a retry.
        """
        assert index is not None
        try:
            exclude_set = set(changed_ids)
            raw_results = index.search(texts, top_k=top_k + len(exclude_set))

            id_to_neighbors: dict[int, list[dict[str, Any]]] = {}
            for nid, matches in zip(changed_ids, raw_results, strict=True):
                neighbors: list[dict[str, Any]] = []
                for m in matches:
                    score = round(1.0 - m["distance"], 3)
                    if score < threshold:
                        break
                    neighbor_id = m["note_id"]
                    if neighbor_id in exclude_set:
                        continue
                    try:
                        note_data = await wrapper.note_to_dict(neighbor_id, "meta")
                    except Exception:
                        logger.debug(
                            "neighbor lookup: skipping unreadable note %s",
                            neighbor_id,
                            exc_info=True,
                        )
                        continue
                    neighbors.append(
                        {
                            "id": neighbor_id,
                            "score": score,
                            "tags": note_data.get("tags", []),
                        }
                    )
                    if len(neighbors) >= top_k:
                        break
                id_to_neighbors[nid] = neighbors

            for r in results:
                if r.get("status") in ("created", "updated") and r.get("id") in id_to_neighbors:
                    r["neighbors"] = id_to_neighbors[r["id"]]
            return True
        except Exception:
            logger.warning("Failed to compute neighbors after upsert", exc_info=True)
            return False

    @mcp.tool()
    @_safe_tool
    async def upsert_note_types(
        note_types: Annotated[
            list[NoteTypeInput],
            Field(
                min_length=1,
                max_length=10,
                description="Array of note type definitions to create or update.",
            ),
        ],
    ) -> UpsertNoteTypesResponse:
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
        names = [nt.name or f"id={nt.id}" for nt in note_types]
        logger.info("upsert_note_types count=%d names=%s", len(note_types), ", ".join(names))

        nt_dicts = [nt.model_dump(exclude_none=True) for nt in note_types]
        results = await wrapper.run(lambda c: _upsert_note_types(c, nt_dicts))

        for r in results:
            status = r.get("status", "unknown")
            if status == "error":
                logger.warning(
                    "upsert_note_types failed for %s: %s", r.get("name", "?"), r["error"]
                )

        return UpsertNoteTypesResponse.model_validate({"results": results})

    @mcp.tool()
    @_safe_tool
    async def delete_notes(
        ids: Annotated[
            list[int],
            Field(
                min_length=1,
                max_length=100,
                description="Note IDs to delete. Their cards are deleted too.",
            ),
        ],
    ) -> DeleteNotesResponse:
        """Permanently delete notes and all their associated cards.

        This cannot be undone. Use list_notes or search_notes first to
        verify which notes will be deleted."""
        logger.info("delete_notes requested=%d", len(ids))
        result = await wrapper.delete_notes(ids)
        logger.info(
            "delete_notes completed: %d deleted, %d not found",
            len(result["deleted"]),
            len(result["not_found"]),
        )

        if index and index.available and result["deleted"]:
            try:
                removed = index.remove(result["deleted"])
                index.col_mod = await wrapper.run(lambda c: c.mod)
                logger.debug("Index updated: %d vectors removed", removed)
            except Exception:
                logger.warning("Failed to update index after delete", exc_info=True)

        return DeleteNotesResponse.model_validate(result)

    @mcp.tool()
    @_safe_tool
    async def delete_note_types(
        ids: Annotated[
            list[int],
            Field(
                min_length=1,
                max_length=10,
                description="Note type IDs to delete. Fails for any note type still in use.",
            ),
        ],
    ) -> DeleteNoteTypesResponse:
        """Delete note type definitions by ID.

        A note type can only be deleted if no notes currently use it.
        Check use counts via collection_info first."""
        logger.info("delete_note_types requested=%d", len(ids))
        result = await wrapper.delete_note_types(ids)
        statuses: dict[str, int] = {}
        for r in result["results"]:
            s = r["status"]
            statuses[s] = statuses.get(s, 0) + 1
        logger.info("delete_note_types completed: %s", statuses)
        return DeleteNoteTypesResponse.model_validate(result)
