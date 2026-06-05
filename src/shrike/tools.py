from __future__ import annotations

import functools
import inspect
import logging
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from shrike.collection import CollectionWrapper, substring_info
from shrike.index import IndexSaver, IndexState, VectorIndex
from shrike.schemas import (
    CollectionInfo,
    DeckInput,
    DeleteDecksResponse,
    DeleteNotesResponse,
    DeleteNoteTypesResponse,
    ListNotesResponse,
    NoteInput,
    NoteTypeInput,
    RenameTagResponse,
    SearchResponse,
    UpdateNoteTagsResponse,
    UpsertDecksResponse,
    UpsertNotesResponse,
    UpsertNoteTypesResponse,
)

logger = logging.getLogger("shrike.tools")


class ToolInputError(Exception):
    """A tool was called with invalid arguments.

    Surfaced to the caller as an MCP ``isError`` result (so the client raises
    ``ServerError``), but logged without a traceback — it's the caller's mistake,
    not a server bug.
    """


def _safe_tool(fn: Any) -> Any:
    """Wrap a tool to log unhandled exceptions, then re-raise.

    A re-raised exception becomes an MCP ``isError`` result (FastMCP converts
    it), which the client surfaces as a ``ServerError``. Tools therefore never
    embed an ``error`` field in a success payload — protocol errors live in the
    protocol, and response models stay clean. ``ToolInputError`` (expected bad
    input) re-raises quietly; anything else logs with a traceback.

    The wrapped function's docstring is dedented with ``inspect.cleandoc`` so the
    tool description FastMCP advertises to clients has no source indentation.
    """
    cleaned_doc = inspect.cleandoc(fn.__doc__) if fn.__doc__ else None

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolInputError:
                raise
            except Exception:
                logger.exception("Unhandled error in %s", fn.__name__)
                raise

        async_wrapper.__doc__ = cleaned_doc
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolInputError:
            raise
        except Exception:
            logger.exception("Unhandled error in %s", fn.__name__)
            raise

    wrapper.__doc__ = cleaned_doc
    return wrapper


def register_tools(
    mcp: FastMCP,
    wrapper: CollectionWrapper,
    index: VectorIndex | None = None,
    saver: IndexSaver | None = None,
) -> None:
    from shrike.note_types import upsert_note_types as _upsert_note_types

    @mcp.tool()
    @_safe_tool
    async def collection_info(
        include: Annotated[
            list[Literal["summary", "note_types", "decks", "tags", "stats", "all"]],
            Field(
                default_factory=list,
                description=(
                    'Sections to return. Any combination of "summary" (counts, dates, '
                    'path), "note_types" (note types and their fields), "decks" (deck '
                    'hierarchy with note counts), "tags" (all tags in use), "stats" (card '
                    'counts, due counts, per-deck summaries), or "all" for everything. '
                    'Defaults to ["summary"].'
                ),
            ),
        ],
        note_type_details: Annotated[
            list[str],
            Field(
                default_factory=list,
                description=(
                    "List of note type names to return full definitions for, including "
                    "card template HTML and CSS styling. Omit to return only summaries "
                    "(field names and type)."
                ),
            ),
        ],
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
        *,
        ids: Annotated[
            list[int], Field(default_factory=list, description="Specific note IDs to retrieve.")
        ],
        deck: Annotated[
            str | None,
            Field(
                description=(
                    'Filter to notes in this deck. Use "::" for nested decks '
                    '(e.g., "Japanese::Vocabulary"). Includes child decks. Accepts a '
                    "deck name, numeric deck ID, or #ID."
                )
            ),
        ] = None,
        tags: Annotated[
            list[str],
            Field(
                default_factory=list,
                description=(
                    'Filter to notes having all of these tags. Prefix with "-" to '
                    'exclude (e.g., ["-leech", "verb"] matches notes tagged "verb" '
                    'but not "leech").'
                ),
            ),
        ],
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
        exact-text queries, use search_notes instead.

        At least one filter must be provided. Combine filters freely — they
        are ANDed together. Use `fields: "meta"` to return only metadata for
        large result sets. The response includes `total` (full match count);
        if more notes matched than `limit` allows, narrow your filters."""
        if not any([ids, deck, tags, note_type, modified_since]):
            raise ToolInputError(
                "At least one filter (ids, deck, tags, note_type,"
                " or modified_since) must be provided."
            )

        filters = [
            f
            for f in [
                f"deck={deck}" if deck else "",
                f"tags={tags}" if tags else "",
                f"type={note_type}" if note_type else "",
                f"ids={len(ids)}" if ids else "",
                f"since={modified_since}" if modified_since else "",
            ]
            if f
        ]
        logger.info("list_notes %s limit=%d", " ".join(filters), limit)

        result = await wrapper.list_notes(
            ids=ids or None,
            deck=deck,
            tags=tags or None,
            note_type=note_type,
            modified_since=modified_since,
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
        *,
        queries: Annotated[
            list[str],
            Field(
                default_factory=list,
                max_length=50,
                description="Search strings, each matched independently both by semantic "
                "similarity and as an exact (case-insensitive) substring of note fields "
                "(max 50 per call).",
            ),
        ],
        ids: Annotated[
            list[int],
            Field(
                default_factory=list,
                max_length=50,
                description=(
                    "Note IDs to use as search anchors — returns notes semantically "
                    "similar to these existing notes. Source notes are excluded from results "
                    "(max 50 per call)."
                ),
            ),
        ],
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
            Field(
                description="Restrict search to notes in this deck (includes child "
                "decks). Accepts a deck name, numeric deck ID, or #ID."
            ),
        ] = None,
        tags: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Restrict search to notes matching all of these tags.",
            ),
        ],
        exclude_ids: Annotated[
            list[int],
            Field(
                default_factory=list,
                description="Additional note IDs to exclude from results.",
            ),
        ],
    ) -> SearchResponse:
        """Search the collection by meaning and by exact text in one call.

        Each query string is matched two ways and the results are folded
        together: by semantic similarity (the vector index) and by exact,
        case-insensitive substring over note fields. Every match carries a
        `score` when it was semantically ranked and a `substring` annotation
        (matched fields + a snippet) when the query text occurs literally — both
        when both apply. Note IDs in `ids` are semantic anchors only (no literal
        text to match).

        Exact substring matches are returned even when the embedding index is
        unavailable (the response carries a `message` noting semantic ranking was
        skipped) and are not subject to `threshold`. Within a group, matches that
        contain the text literally are listed first, then by descending score.

        Use this for conceptual queries keyword search can't handle and for
        finding exact wording. At least one of `queries` or `ids` is required."""
        if not queries and not ids:
            raise ToolInputError("At least one of queries or ids must be provided.")

        logger.info(
            "search_notes queries=%d ids=%d top_k=%d threshold=%.2f",
            len(queries or []),
            len(ids or []),
            top_k,
            threshold,
        )
        if queries:
            logger.debug("search_notes query strings: %s", queries)
        if ids:
            logger.debug("search_notes source ids: %s", ids)

        # Substring matching needs no embeddings; semantic ranking does.
        semantic_ok = index is not None and index.available and index.state == IndexState.READY
        message: str | None = None
        if not semantic_ok:
            if index is not None and index.state == IndexState.BUILDING:
                indexed, total = index.build_progress
                message = (
                    f"Semantic ranking unavailable (index building {indexed}/{total}); "
                    "returning exact text matches only."
                )
            elif index is not None and index.state == IndexState.ERROR:
                message = (
                    "Semantic ranking unavailable (index error); returning exact text matches only."
                )
            else:
                message = (
                    "Semantic ranking unavailable (embedding service not running); "
                    "returning exact text matches only. Start it with "
                    "'shrike embedding start'."
                )
            # Pure semantic request with nothing to substring-match → nothing to do.
            if not queries:
                return SearchResponse(message=message)

        if deck:
            # Accept a deck name, #id, or numeric id; an explicit id that matches
            # nothing yields no results.
            deck = await wrapper.resolve_deck_ref(deck)
            if deck is None:
                return SearchResponse(results=[], message="No deck matches that reference.")

        exclude_set = set(exclude_ids or [])

        # Assemble search sources: each query string (semantic + substring) and
        # each id anchor (semantic only). Anchors are excluded from their results.
        text_sources: list[tuple[str, str, str]] = []  # (kind, label, text)
        for q in queries or []:
            text_sources.append(("query", q, q))
        if ids:
            note_texts = await wrapper.note_texts_for_embedding(ids)
            for nid, text in zip(ids, note_texts, strict=True):
                if text:
                    text_sources.append(("id", f"note #{nid}", text))
                    exclude_set.add(nid)

        if not text_sources:
            return SearchResponse(message="No valid queries or note IDs to search.")

        # Semantic pass (batched). Over-fetch to cover excluded ids and post-hoc
        # deck/tag/substring filtering, which can otherwise under-return.
        sem_raw: dict[int, list[dict[str, Any]]] = {}
        if semantic_ok:
            assert index is not None
            fetch_k = top_k + len(exclude_set)
            if deck or tags:
                fetch_k = max(fetch_k, top_k * 10)
                if index.size:
                    fetch_k = min(fetch_k, index.size)
            raw = index.search([t for (_, _, t) in text_sources], top_k=fetch_k)
            sem_raw = dict(enumerate(raw))

        def _in_scope(note_data: dict[str, Any]) -> bool:
            if deck and note_data.get("deck") != deck:
                return False
            return not (tags and not all(t in set(note_data.get("tags", [])) for t in tags))

        results: list[dict[str, Any]] = []
        for i, (kind, label, text) in enumerate(text_sources):
            merged: dict[int, dict[str, Any]] = {}

            # Exact substring (query sources only; deck/tags/exclude applied inside).
            if kind == "query":
                exact = await wrapper.search_substring(
                    text, deck=deck, tags=tags or None, exclude_ids=list(exclude_set), limit=top_k
                )
                for note in exact:
                    merged[note["id"]] = {**note, "score": None}

            # Semantic.
            sem_count = 0
            for m in sem_raw.get(i, []):
                nid = m["note_id"]
                if nid in exclude_set:
                    continue
                score = round(1.0 - m["distance"], 3)
                if score < threshold:
                    break  # raw is distance-ascending → the rest are below threshold
                try:
                    note_data = await wrapper.note_to_dict(nid, "full")
                except Exception:
                    logger.debug("search_notes: skipping unreadable note %s", nid, exc_info=True)
                    continue
                if not _in_scope(note_data):
                    continue
                if nid in merged:
                    merged[nid]["score"] = score
                else:
                    entry = {**note_data, "score": score}
                    if kind == "query":
                        entry["substring"] = substring_info(note_data.get("content"), text)
                    merged[nid] = entry
                sem_count += 1
                if sem_count >= top_k:
                    break

            # Literal hits first, then by descending score.
            ordered = sorted(
                merged.values(),
                key=lambda e: (
                    0 if e.get("substring") is not None else 1,
                    -(e["score"] if e.get("score") is not None else -1.0),
                ),
            )
            results.append({"source": label, "matches": ordered})

        logger.info(
            "search_notes returned %d groups, %d total matches",
            len(results),
            sum(len(r["matches"]) for r in results),
        )
        return SearchResponse.model_validate({"results": results, "message": message})

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
        on_duplicate: Annotated[
            Literal["error", "skip", "allow"],
            Field(
                description=(
                    "Policy for a new note whose first field exactly duplicates an existing "
                    "note of the same type (Anki's own duplicate rule). `error` (default): "
                    "report the item as an error, don't write it. `skip`: leave it unwritten "
                    "and report `status: skipped`. `allow`: create it anyway. Applies only to "
                    "creates; updates are unaffected."
                ),
            ),
        ] = "error",
        dry_run: Annotated[
            bool,
            Field(
                description=(
                    "If true, validate every note and report what would happen, but write "
                    "nothing. Each result is `ok` (with `action: create|update`), `skipped`, "
                    "or `error`. Use it as a pre-flight sanity check before a real upsert."
                ),
            ),
        ] = False,
    ) -> UpsertNotesResponse:
        """Create or update notes in bulk (1-100 per call).

        If a note object includes an `id`, the existing note is updated;
        if `id` is absent, a new note is created.

        For new notes, `deck`, `note_type`, and `fields` are required. For
        updates, only `id` and the properties being changed are needed —
        omitted properties are left unchanged.

        Each new note is validated against Anki's own add-note rule before it
        is written. A first-field duplicate (same first field as an existing
        note of that type) is governed by `on_duplicate`: `error` (default,
        reported and not written), `skip` (`status: skipped`), or `allow`
        (created anyway). Notes that are malformed regardless of policy — an
        empty first field, or broken cloze structure — are always reported as
        errors with a `reason` and never written. The whole batch still
        proceeds: one bad note doesn't block the rest.

        Set `dry_run: true` to validate everything and write nothing — a
        pre-flight sanity check. Each result is `ok` (with `action`),
        `skipped`, or `error`; the response echoes `dry_run: true`.

        When a vector index is available (and not a dry run), each created or
        updated result includes `neighbors`: the most similar existing notes
        ranked by cosine similarity, filtered to those above
        `neighbor_threshold` (default 0.5) and capped at `top_k_neighbors`
        (default 5). Use these for tag consistency (adopt tags from nearby
        notes), spotting near-duplicates by meaning (a high score is a softer
        signal than the exact `on_duplicate` rule), or understanding where a
        new note sits in the collection. Neighbors include note ID, similarity
        score, and tags — use list_notes or search_notes to inspect content.

        If the index update fails transiently (e.g. the embedding service is
        briefly unavailable), the notes are still saved but `neighbors` is
        omitted. Each affected result is flagged `neighbors_unavailable: true`
        and the response carries a top-level `message`. Recover the exact same
        neighbor data afterward with search_notes(ids=[<note id>]) — it embeds
        the same note text against the same index, so the result is identical
        to what would have been attached here."""
        creates = sum(1 for n in notes if n.id is None)
        updates = len(notes) - creates
        logger.info(
            "upsert_notes count=%d (creates=%d, updates=%d) on_duplicate=%s dry_run=%s",
            len(notes),
            creates,
            updates,
            on_duplicate,
            dry_run,
        )

        note_dicts = [n.model_dump(exclude_none=True) for n in notes]
        results = await wrapper.upsert_notes(note_dicts, on_duplicate=on_duplicate, dry_run=dry_run)

        counts: dict[str, int] = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        logger.info(
            "upsert_notes completed: %d created, %d updated, %d ok, %d skipped, %d errors",
            counts.get("created", 0),
            counts.get("updated", 0),
            counts.get("ok", 0),
            counts.get("skipped", 0),
            counts.get("error", 0),
        )

        # A dry run writes nothing, so there is no index maintenance or neighbor
        # lookup to do — the results are pure validation outcomes.
        if not dry_run and index and index.available:
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
                    # Schedule a debounced flush so a hard kill while idle
                    # doesn't force a full re-embed. Non-blocking and last in
                    # the try: it must not mask the neighbors attached above.
                    if saver is not None:
                        saver.request_save()
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
                                "dry_run": False,
                                "message": (
                                    "Notes were saved, but the vector index update failed, "
                                    "so neighbors could not be computed. Retry with "
                                    f"search_notes(ids={pending}) to fetch the same "
                                    "neighbor data."
                                ),
                            }
                        )

        return UpsertNotesResponse.model_validate({"results": results, "dry_run": dry_run})

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

        On update, `fields` and `templates` replace the whole list but are
        applied **by position**, so existing note data and cards are preserved:
        the field/template at each position keeps its data even when renamed or
        retitled. Only shortening the list discards the trailing entries —
        removing a field drops that field's data, removing a template deletes
        its cards. Adding entries appends empty fields / new cards. (To move a
        field or template rather than rename-by-position, use a dedicated
        reorder once available.)"""
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
                if saver is not None:
                    saver.request_save()
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

    async def _bump_col_mod_after_metadata_change() -> None:
        """Keep the index from needlessly rebuilding after a vectors-unchanged edit.

        Tag and deck operations don't touch a note's embedding text, so every
        vector stays valid — but they still bump ``col.mod``. Advance the stored
        ``col_mod`` (and request a debounced save) so the next startup sees no
        drift and skips a full re-embed. Best-effort: index bookkeeping must
        never fail the operation, which is already committed to the collection.
        """
        if index is None or not index.available:
            return
        try:
            index.col_mod = await wrapper.run(lambda c: c.mod)
            if saver is not None:
                saver.request_save()
        except Exception:
            logger.warning("Failed to advance index col_mod after metadata change", exc_info=True)

    @mcp.tool()
    @_safe_tool
    async def update_note_tags(
        note_ids: Annotated[
            list[int],
            Field(
                min_length=1,
                max_length=1000,
                description="Note IDs whose tags to edit.",
            ),
        ],
        *,
        set: Annotated[  # noqa: A002 — `set` is the wire/CLI name for full-replace mode
            list[str] | None,
            Field(
                description=(
                    "Full replace: the notes end up with exactly these tags (an empty "
                    "list clears all tags). Mutually exclusive with add/remove."
                ),
            ),
        ] = None,
        add: Annotated[
            list[str],
            Field(default_factory=list, description="Tags to add, leaving other tags intact."),
        ],
        remove: Annotated[
            list[str],
            Field(default_factory=list, description="Tags to remove, leaving other tags intact."),
        ],
    ) -> UpdateNoteTagsResponse:
        """Edit tags on a set of notes (1-1000 per call).

        Choose exactly one mode — there is no default:
        - `set`: full replace. The notes end up with exactly the tags you pass;
          pass an empty list to clear all tags.
        - `add` and/or `remove`: additive/subtractive. Add tags without
          disturbing existing ones, remove specific tags, or both in one call
          (e.g. add ["jp","verbs"] + remove ["jp-verbs"] swaps one tag for two).

        `set` cannot be combined with `add`/`remove`. To replace a note's tags
        as part of a broader edit (fields, deck), use upsert_notes instead.

        Returns the number of notes the operation applied to and any requested
        IDs that were not found."""
        set_mode = set is not None
        addremove_mode = bool(add or remove)
        if set_mode and addremove_mode:
            raise ToolInputError("Provide either `set` (full replace) or `add`/`remove`, not both.")
        if not set_mode and not addremove_mode:
            raise ToolInputError("Specify `set`, or `add` and/or `remove`.")

        if set_mode:
            logger.info("update_note_tags notes=%d set=%s", len(note_ids), set)
        else:
            logger.info("update_note_tags notes=%d add=%s remove=%s", len(note_ids), add, remove)

        result = await wrapper.update_note_tags(note_ids, set_tags=set, add=add, remove=remove)
        logger.info(
            "update_note_tags modified %d note(s), %d not found",
            result["notes_modified"],
            len(result["not_found"]),
        )

        if result["notes_modified"]:
            await _bump_col_mod_after_metadata_change()

        return UpdateNoteTagsResponse.model_validate(result)

    @mcp.tool()
    @_safe_tool
    async def rename_tag(
        old: Annotated[str, Field(min_length=1, description="The tag to rename.")],
        new: Annotated[str, Field(min_length=1, description="The new tag name.")],
        note_ids: Annotated[
            list[int],
            Field(
                default_factory=list,
                description=(
                    "Restrict the rename to these notes. Omit (empty) to rename the tag "
                    "across the entire collection."
                ),
            ),
        ],
    ) -> RenameTagResponse:
        """Rename a tag, collection-wide or on a set of notes.

        With no `note_ids`, the tag is renamed everywhere it appears (and so are
        its children, e.g. renaming "history" moves "history::ww2"). With
        `note_ids`, only those notes are affected and the tag is matched exactly
        — renaming "jp" never touches "jp-verbs".

        Returns the number of notes whose tags changed."""
        if old == new:
            raise ToolInputError("`old` and `new` tags are identical — nothing to rename.")
        logger.info("rename_tag %r -> %r (scope=%d notes)", old, new, len(note_ids))
        result = await wrapper.rename_tag(old, new, note_ids)
        logger.info("rename_tag modified %d note(s)", result["notes_modified"])
        if result["notes_modified"]:
            await _bump_col_mod_after_metadata_change()
        return RenameTagResponse.model_validate(result)

    @mcp.tool()
    @_safe_tool
    async def upsert_decks(
        decks: Annotated[
            list[DeckInput],
            Field(
                min_length=1,
                max_length=100,
                description="Array of deck objects to create or rename.",
            ),
        ],
    ) -> UpsertDecksResponse:
        """Create or rename decks in bulk (1-100 per call).

        Same shape as upsert_notes: each item's `name` is the desired deck name.
        If an item includes an `id`, that existing deck is renamed to `name`
        (and reparented if the name has a new `::` prefix); renaming onto a name
        already used by a different deck is an error (decks do not merge — move
        the notes instead). If `id` is absent, a deck named `name` is created (or
        left as-is if it already exists). Nested decks use "::"
        (e.g. "Japanese::Vocabulary").

        Each result reports `status` ("created", "updated", or "error") with the
        deck `id` and `name`. Use this to set up deck structure before adding
        notes, or to reorganize the hierarchy. To delete a deck, empty it first
        (move its notes elsewhere) then call delete_decks."""
        creates = sum(1 for d in decks if d.id is None)
        logger.info("upsert_decks count=%d (renames=%d)", len(decks), len(decks) - creates)

        deck_dicts = [d.model_dump(exclude_none=True) for d in decks]
        results = await wrapper.upsert_decks(deck_dicts)

        created = sum(1 for r in results if r.get("status") == "created")
        updated = sum(1 for r in results if r.get("status") == "updated")
        errors = sum(1 for r in results if r.get("status") == "error")
        logger.info(
            "upsert_decks completed: %d created, %d updated, %d errors", created, updated, errors
        )
        for r in results:
            if r.get("status") == "error":
                logger.warning("upsert_decks item %d failed: %s", r.get("index"), r["error"])

        if created or updated:
            await _bump_col_mod_after_metadata_change()
        return UpsertDecksResponse.model_validate({"results": results})

    @mcp.tool()
    @_safe_tool
    async def delete_decks(
        decks: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=100,
                description="Decks to delete (name, numeric ID, or #ID). Each must be "
                "empty (see below).",
            ),
        ],
    ) -> DeleteDecksResponse:
        """Delete decks by name — only if they are already empty.

        A deck is deletable only when neither it nor any of its subdecks contains
        cards. To remove a non-empty deck, first move its notes elsewhere (e.g.
        upsert_notes with a new `deck`, or rename the deck onto another to merge),
        then delete the now-empty deck. This keeps deletion from ever destroying
        notes.

        Returns `deleted`, `not_found`, and `not_empty` name lists; a non-empty
        or missing deck is skipped, not an error."""
        logger.info("delete_decks requested=%d", len(decks))
        result = await wrapper.delete_decks(decks)
        logger.info(
            "delete_decks completed: %d deleted, %d not found, %d not empty",
            len(result["deleted"]),
            len(result["not_found"]),
            len(result["not_empty"]),
        )
        if result["deleted"]:
            await _bump_col_mod_after_metadata_change()
        return DeleteDecksResponse.model_validate(result)
