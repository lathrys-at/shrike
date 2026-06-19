"""The transport-neutral action core.

Every tool the server exposes is defined here as an :class:`ActionDef` —
``(name, JSON-schema'd input, JSON-schema'd output, coarse impl)`` — with no
FastMCP coupling: the input contract is the impl's typed (Annotated) signature,
the output contract its response-model return annotation, both serde-mappable
(``schemas.py`` stays canonical). Implementations take an :class:`ActionContext`
(the kernel's view of the world) instead of closing over loose server objects,
and raise the transport-neutral error contract: :class:`ToolInputError`
(expected bad input, no traceback), ``CollectionBusyError`` (the
``collection_busy`` sentinel), anything else is a bug.

The MCP binding lives in ``mcp_adapter.py`` (registration + the ``_safe_tool``
policy); ``tools.py`` is the composition shim that keeps ``register_tools``'s
signature. A future agent-runtime adapter (on-device function-calling bindings)
iterates the same registry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import shrike_native
from pydantic import Field, TypeAdapter

from shrike.harness.collection import (
    CollectionWrapper,
)
from shrike.harness.derived import DerivedTextStore
from shrike.harness.index import ACTIVATION_MARGIN, IndexState, activation_floor
from shrike.platform.pathsafety import output_path_within_any_root, path_within_any_root
from shrike.schemas import (
    CollectionCheckResponse,
    CollectionInfo,
    CollectionPruneResponse,
    DeckInput,
    DeleteDecksResponse,
    DeleteMediaResponse,
    DeleteNotesResponse,
    DeleteNoteTypesResponse,
    ExportPackagePath,
    ExportPackageResponse,
    ExportPackageUrl,
    FetchMediaResponse,
    FieldMetadataInput,
    FieldOp,
    FindReplaceNoteTypesResponse,
    FindReplaceResponse,
    ImportPackageResponse,
    ListMediaResponse,
    ListNotesResponse,
    ListProfilesResponse,
    MigrateNoteTypeResponse,
    NoteInput,
    NoteTypeInput,
    ProfileEntry,
    RenameTagResponse,
    SearchResponse,
    SearchResultGroup,
    StoreMediaItem,
    StoreMediaResponse,
    TemplateOp,
    UpdateNoteTagsResponse,
    UpdateNoteTypeFieldMetadataResponse,
    UpdateNoteTypeFieldsResponse,
    UpdateNoteTypeTemplatesResponse,
    UpsertDecksResponse,
    UpsertNotesResponse,
    UpsertNoteTypesResponse,
)

logger = logging.getLogger("shrike.tools")

# The per-call outcome fragment for the single completion log line: an action
# records what happened ("3/3 notes", "2 created, 1 error"); the adapter folds
# it — with the call's params and duration — into the ONE INFO line each served
# call emits. Adapter-agnostic (a contextvar, not an MCP coupling).
_call_outcome: ContextVar[str | None] = ContextVar("shrike_call_outcome", default=None)


def note_outcome(message: str) -> None:
    """Record the action's result fragment for the single completion log line."""
    _call_outcome.set(message)


# The per-signal RRF weights live kernel-side: `shrike_kernel::fusion::search_weights`
# is the single source of truth, applied when the host passes none. The action below passes no
# weights; a future config/`--search-*` knob re-enters through the same parameter as an override.

# The live-search min-query gate: query strings shorter than this skip the
# embedding-bearing tier even on tier="full" — single letters and typing
# fragments must not burn an embedding call. Ids-anchored searches are never
# gated (no typing-fragment problem).
MIN_SEMANTIC_QUERY_CHARS = 3

# `limit` == 0 means "return all": no upper cap. Used only on the paths
# whose native cap is a plain `.take()`/`.truncate()`/SQLite `LIMIT` — list_notes,
# collection_query, list_media (None there) — where a large sentinel just reads as
# "all" for any real collection. The SEMANTIC search path does NOT use this: it
# over-fetches `k * SEARCH_OVERFETCH` into USearch's `search(k)`, which *allocates*
# a buffer of that size, so search_notes bounds `limit==0` to `index.size` instead
# (the true result ceiling). 1e9 is orders of magnitude past any Anki collection.
_UNBOUNDED_LIMIT = 1_000_000_000

# The per-call collection selector. Every routable tool carries this optional
# param; it names a registered profile to operate on, defaulting to the active
# profile (and, with none set, the daemon's boot collection). The shared
# description keeps the wire contract uniform across the tools.
COLLECTION_SELECTOR_DESCRIPTION = (
    "Which collection to operate on, by registered profile name (see "
    "list_profiles). Omit to use the active default collection. On a "
    "single-collection server, omit it."
)


class ToolInputError(Exception):
    """A tool was called with invalid arguments.

    Surfaced to the caller as an MCP ``isError`` result (so the client raises
    ``ServerError``), but logged without a traceback — it's the caller's mistake,
    not a server bug.
    """


@dataclass(frozen=True)
class CollectionBundle:
    """The per-collection handles an action operates on.

    A selector resolves to exactly one of these — the right collection's
    wrapper + kernel + search-index view + derived store + dedup recorder. In
    single-collection / standalone / test contexts there is one bundle (the
    boot collection); in multi-collection mode the resolver returns the bundle
    for the routed collection. Frozen + per-call, so concurrent callers to
    different collections never share mutable state (``stateless_http``-safe).
    """

    wrapper: CollectionWrapper
    index: Any | None = None
    derived: DerivedTextStore | None = None
    kernel: Any | None = None
    dedup_stats: Any | None = None

    def unpack(self) -> tuple[Any, Any, Any, Any, Any]:
        """``(wrapper, index, kernel, derived, dedup_stats)`` — the order the
        action bodies bind their per-call locals in.

        Typed ``Any`` (not ``... | None``) so the bound locals match what the
        action bodies expect: ``kernel``/``wrapper`` are always present (the
        kernel is required — ``build_actions`` rejects a None context kernel),
        and ``index``/``derived``/``dedup_stats`` are duck-typed handles the
        bodies guard with explicit ``is None`` checks at runtime — exactly the
        ``Any`` shape the closure locals had."""
        return (self.wrapper, self.index, self.kernel, self.derived, self.dedup_stats)


@dataclass(frozen=True)
class ActionContext:
    """What action implementations see of the server — the kernel's surface.

    One context object instead of loose closures over wrapper/index/kernel:
    the registry can be built against any host that assembles these.

    Routing: an action resolves its per-call :class:`CollectionBundle`
    from the ``collection`` selector via :attr:`resolver` (async — lazy
    assembly may await). When no resolver is set (standalone / tests /
    single-collection), the fixed ``wrapper``/``index``/``kernel``/``derived``/
    ``dedup_stats`` ARE the one bundle, and a selector is rejected (nothing to
    route to). The resolver, when present, is the multi-collection manager.
    """

    wrapper: CollectionWrapper
    # The KernelIndexView (duck-typed search-facing surface over the kernel's
    # engine), or None when no embedding is configured. Annotated Any so any
    # duck-typed view (tests) can stand in.
    index: Any | None = None
    derived: DerivedTextStore | None = None
    # The AsyncKernel — REQUIRED: write actions route through its maintained ops
    # (upsert_notes_json/delete_notes/reindex_notes/forget_notes/
    # metadata_changed), which carry the index + derived + watermark bookkeeping
    # kernel-side. ``build_actions`` rejects a None.
    kernel: Any | None = None
    # The dedup best-match recorder — harness-owned; None in standalone/test
    # contexts that don't care.
    dedup_stats: Any | None = None
    allow_private_fetch: bool = False
    server_path_roots: list[str] | None = None
    # Server-local roots an import `.apkg`/`.colpkg` path must be contained in.
    # DISTINCT from server_path_roots (media read) by design: import is a
    # whole-collection overwrite — a higher blast radius — so it gets its own
    # `--import-path-root`, never inheriting a media-read root. None/empty →
    # import-by-server-path is disabled.
    server_import_path_roots: list[str] | None = None
    media_base_url: str | None = None
    # Export: the operator-allowed server-local OUTPUT roots (the
    # --export-path-root capability, write counterpart of server_path_roots),
    # the download store (server-named temp packages → tokens), and whether the
    # server is purely-local (the second gate on a server-local output_path,
    # exactly like store_media's path source). All None/empty → export still
    # works via the download url; only the opt-in server-local output_path is
    # gated off.
    export_path_roots: list[str] | None = None
    export_store: Any | None = None
    server_purely_local: bool = False
    # The collection/profile registry — a Registry snapshot for the read-only
    # `list_profiles` enumeration. None disables the action's data (an empty
    # registry) without removing the action.
    registry: Any | None = None
    # The per-call collection router: an async callable
    # ``selector -> CollectionBundle``. None → single-collection mode (the
    # fixed handles above are the only bundle; a non-None selector is an error).
    resolver: Any | None = None


@dataclass(frozen=True)
class ActionDef:
    """One registry entry: the action's name, contract, and implementation.

    ``request_schema`` is carried by ``impl``'s typed signature (what FastMCP —
    or any other adapter — generates the input JSON Schema from); the response
    model is the return annotation. ``doc`` is the human contract.
    """

    name: str
    impl: Any
    doc: str | None


def build_actions(ctx: ActionContext) -> list[ActionDef]:
    """Build the full action registry against one context (27 actions)."""
    from urllib.parse import quote

    if ctx.kernel is None:
        # No standalone (facade) mode: every write action routes through a
        # maintained kernel op. Tests drive a real AsyncKernel via the unit
        # harness (tests/unit/conftest.py).
        raise ValueError(
            "actions require kernel mode (#355): pass kernel=<AsyncKernel> "
            "to register_tools/ActionContext"
        )

    # Unpack once: the action bodies below close over these. In
    # single-collection mode these ARE the one bundle; in multi-collection mode
    # each routable action rebinds wrapper/index/kernel/derived/dedup_stats PER
    # CALL from the resolved bundle (`wrapper, index, kernel, derived,
    # dedup_stats = (await _route(collection)).unpack()`).
    wrapper = ctx.wrapper
    index = ctx.index
    derived = ctx.derived
    kernel = ctx.kernel
    dedup_stats = ctx.dedup_stats
    allow_private_fetch = ctx.allow_private_fetch
    server_path_roots = ctx.server_path_roots
    server_import_path_roots = ctx.server_import_path_roots
    media_base_url = ctx.media_base_url
    export_path_roots = ctx.export_path_roots or []
    export_store = ctx.export_store
    server_purely_local = ctx.server_purely_local
    registry = ctx.registry
    resolver = ctx.resolver

    # The single-collection bundle: the fixed handles, used when no resolver is
    # set (standalone / tests) or a routable action gets no selector and the
    # resolver returns the default. Built once; immutable.
    _default_bundle = CollectionBundle(
        wrapper=wrapper, index=index, derived=derived, kernel=kernel, dedup_stats=dedup_stats
    )

    async def _route(selector: str | None) -> CollectionBundle:
        """Resolve the per-call collection bundle.

        No resolver → single-collection mode: a selector is a caller error
        (there is nothing to route to); None yields the one fixed bundle. With
        a resolver, it owns resolution (selector → registry → default → the
        boot collection) and lazy assembly; an unknown selector surfaces as a
        ``ToolInputError`` so the caller sees a clean rejection.
        """
        if resolver is None:
            if selector is not None:
                raise ToolInputError(
                    f"collection routing is not enabled on this server (selector {selector!r})"
                )
            return _default_bundle
        try:
            bundle: CollectionBundle = await resolver(selector)
        except ToolInputError:
            raise
        except Exception as e:  # the manager's RoutingError → a clean input error
            raise ToolInputError(str(e)) from e
        return bundle

    actions: list[ActionDef] = []

    def _action(fn: Any) -> Any:
        actions.append(ActionDef(name=fn.__name__, impl=fn, doc=fn.__doc__))
        return fn

    # The note-type ops run in the native core; its input error is a ValueError
    # and plays the NoteTypeOpError role.
    from shrike_native import NativeInputError as NoteTypeOpError

    def _media_url(filename: str) -> str | None:
        """The GET /media/<name> URL for a media file, or None if the server
        didn't advertise a base URL (e.g. direct library use)."""
        if not media_base_url:
            return None
        return f"{media_base_url}/media/{quote(filename)}"

    @_action
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
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
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
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        include_list: list[str] = [str(s) for s in include] if include else []
        logger.debug("collection_info sections=%s", ",".join(include_list or ["summary"]))
        # The whole body runs in shrike_kernel::actions, on the collection
        # worker (the same serialization every op rides).
        raw = await wrapper.run(
            lambda c: shrike_native.action_collection_info(c, include_list, note_type_details)
        )
        return CollectionInfo.model_validate_json(raw)

    @_action
    async def list_profiles() -> ListProfilesResponse:
        """List the collection profiles this server knows about.

        Returns the registered profiles (each a friendly `name` and its
        collection `path`) and which one is the active default. The registry is
        a superset of Anki's profiles — any collection path can be registered,
        not only ones under Anki's base directory.

        Use this to discover what collections exist by name. This is read-only
        and does not change which collection the server is operating on —
        selecting a collection per call is a separate capability."""
        # Host-side enumeration: no collection/kernel involvement. The registry
        # snapshot was loaded from config at assembly; an absent one (None)
        # reports an empty registry rather than erroring.
        entries = list(registry.profiles) if registry is not None else []
        default = registry.default if registry is not None else None
        note_outcome(f"{len(entries)} profile(s)")
        return ListProfilesResponse(
            profiles=[
                ProfileEntry(name=p.name, path=p.path, is_default=(p.name == default))
                for p in entries
            ],
            default=default,
        )

    @_action
    async def export_package(
        *,
        deck: Annotated[
            str | None,
            Field(
                description=(
                    "Export only this deck (by name, numeric id, or #id). Omit for the whole "
                    "collection. Mutually exclusive with `note_ids`."
                ),
            ),
        ] = None,
        note_ids: Annotated[
            list[int],
            Field(
                default_factory=list,
                description="Export only these notes (by id). Mutually exclusive with `deck`.",
            ),
        ],
        format: Annotated[
            str,
            Field(
                description=(
                    "'apkg' (a shareable note package, scopable) or 'colpkg' (a whole-collection "
                    "backup — cannot be scoped). Default 'apkg'."
                ),
            ),
        ] = "apkg",
        include_scheduling: Annotated[
            bool,
            Field(
                description=(
                    "Include review/scheduling data (and deck options). Default false — a "
                    "shareable package usually omits the exporter's review history."
                )
            ),
        ] = False,
        include_media: Annotated[
            bool,
            Field(description="Bundle referenced media files into the package. Default true."),
        ] = True,
        output_path: Annotated[
            str | None,
            Field(
                description=(
                    "Write the package to this server-local path instead of returning a download "
                    "URL. Honored only on a purely-local server with the path inside an operator-"
                    "allowed --export-path-root; otherwise an error. Omit (the default) to get a "
                    "download `url`."
                ),
            ),
        ] = None,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> ExportPackageResponse:
        """Export the collection (or a deck/note selection) to an Anki package.

        Writes a `.apkg` (a shareable, scopable note package) or a `.colpkg`
        (a whole-collection backup — scoping is rejected). Scope to one `deck`
        or a set of `note_ids` (not both); omit both for the whole collection.
        `include_scheduling` carries review data + deck options; `include_media`
        bundles referenced files.

        By default the server writes the package to a temporary file and returns
        a download `url` — GET it to retrieve the bytes (never base64). On a
        purely-local server you may instead set `output_path` to a server-local
        file inside an operator-allowed export root; the response then carries
        that `path`. Use this for backups, sharing a deck, or moving a
        collection between machines."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        # Scope: deck XOR note_ids XOR whole.
        if deck is not None and note_ids:
            raise ToolInputError("Provide at most one of `deck` or `note_ids`, not both.")
        fmt = format.strip().lower()
        if fmt not in ("apkg", "colpkg"):
            raise ToolInputError("`format` must be 'apkg' or 'colpkg'.")
        if fmt == "colpkg" and (deck is not None or note_ids):
            raise ToolInputError(
                "A .colpkg is a whole-collection backup and cannot be scoped — "
                "use format='apkg' to export a deck or notes."
            )
        if deck is not None:
            scope_kind, scope_deck, scope_notes = "deck", deck, None
        elif note_ids:
            scope_kind, scope_deck, scope_notes = "notes", None, note_ids
        else:
            scope_kind, scope_deck, scope_notes = "whole", None, None
        logger.debug("export_package format=%s scope=%s output=%s", fmt, scope_kind, output_path)

        suffix = f".{fmt}"

        async def _run(target: str) -> Any:
            return json.loads(
                await kernel.export_package(
                    target,
                    fmt,
                    scope_kind,
                    scope_deck,
                    scope_notes,
                    with_scheduling=include_scheduling,
                    with_media=include_media,
                    legacy=False,  # always the modern format
                )
            )

        if output_path is not None:
            # Server-local output (off by default): gated exactly like
            # store_media's `path` — purely-local AND contained in an
            # operator-allowed export root (the WRITE gate: realpaths the parent,
            # catching ../symlinked-parent escapes; the kernel's temp+rename then
            # closes a symlinked-basename redirect).
            if not (
                server_purely_local and output_path_within_any_root(output_path, export_path_roots)
            ):
                raise ToolInputError(
                    "output_path is not permitted: the server must be purely-local and the path "
                    "must be inside an --export-path-root. Omit output_path to receive a "
                    "download url instead."
                )
            result = await _run(output_path)
            note_outcome(f"{result['note_count']} notes -> {output_path}")
            return ExportPackagePath(
                delivery="path",
                note_count=int(result["note_count"]),
                bytes=os.path.getsize(result["out_path"]),
                format=fmt,
                path=result["out_path"],
            )

        # Default: write a server-named temp under the cache dir and hand back a
        # download url (never bytes). Requires the host to have wired an export
        # store + a base url (a running HTTP server).
        if export_store is None or not media_base_url:
            raise ToolInputError(
                "download-url export is unavailable here; pass output_path with a configured "
                "--export-path-root, or run against an HTTP server."
            )
        token, temp_path = export_store.new_temp_path(suffix=suffix)
        result = await _run(temp_path)
        export_store.register(token, result["out_path"], fmt)
        note_outcome(f"{result['note_count']} notes -> url:{token[:8]}…")
        return ExportPackageUrl(
            delivery="url",
            note_count=int(result["note_count"]),
            bytes=os.path.getsize(result["out_path"]),
            format=fmt,
            url=f"{media_base_url}/export/{quote(token)}",
        )

    @_action
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
            int,
            Field(
                ge=0,
                le=200,
                description="Maximum notes to return. Default 20. 0 returns all.",
            ),
        ] = 20,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
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
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
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
        logger.debug("list_notes %s limit=%d", " ".join(filters), limit)

        cutoff: int | None = None
        if modified_since:
            try:
                dt = datetime.fromisoformat(modified_since)
            except ValueError as e:
                # Caller-supplied bad input, not a server bug: a clean rejection
                # (WARNING, no traceback) rather than the catch-all's "Unhandled
                # error" + traceback + leaked isoformat detail.
                raise ToolInputError(
                    f"`modified_since` is not a valid ISO 8601 datetime: {modified_since!r}"
                ) from e
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            cutoff = int(dt.timestamp())
        # limit==0 means "return all": the native cap is `.truncate()`, so
        # a large sentinel reads as "all".
        effective_limit = limit if limit > 0 else _UNBOUNDED_LIMIT
        # The whole body runs in shrike_kernel::actions.
        raw = await wrapper.run(
            lambda c: shrike_native.action_list_notes(
                c,
                ids=ids or None,
                deck=deck,
                tags=tags or None,
                note_type=note_type,
                modified_since_epoch=cutoff,
                with_fields=(fields or "full") == "full",
                limit=effective_limit,
            )
        )
        result = ListNotesResponse.model_validate_json(raw)
        note_outcome(f"{len(result.notes)}/{result.total} notes")
        return result

    @_action
    async def collection_query(
        query: Annotated[
            str,
            Field(
                min_length=1,
                description=(
                    "A raw Anki search expression, e.g. 'is:due prop:ivl>=30', "
                    "'added:7 -tag:done', 'deck:Japanese (tag:verb OR tag:adj)'. "
                    "See https://docs.ankiweb.net/searching.html."
                ),
            ),
        ],
        fields: Annotated[
            Literal["full", "meta"],
            Field(
                description=(
                    '"full" (default) returns all field content. "meta" returns only '
                    "note ID, note type, deck, tags, and modification time."
                )
            ),
        ] = "full",
        limit: Annotated[
            int,
            Field(
                ge=0,
                le=200,
                description="Maximum notes to return. Default 20. 0 returns all.",
            ),
        ] = 20,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> ListNotesResponse:
        """Find notes with a raw Anki search expression.

        This is the power-user escape hatch: the `query` string is passed
        straight to Anki's search engine, so the full expression language is
        available — `is:due`, `prop:ivl>=30`, `added:`, `rated:`, `flag:`,
        `nid:`/`cid:`, and boolean `OR` / `-` / parentheses.

        Use this when you need predicates the structured tools don't expose. For
        conceptual or exact-text search use search_notes; for plain deck/tag/type
        filters use list_notes. Returns the same note shape as list_notes, with
        `total` the full match count before `limit`. An invalid expression is
        reported as an input error."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("collection_query %r fields=%s limit=%d", query, fields, limit)
        try:
            # limit==0 means "return all": a large sentinel reads as "all".
            effective_limit = limit if limit > 0 else _UNBOUNDED_LIMIT
            # The whole body runs in shrike_kernel::actions.
            raw = await wrapper.run(
                lambda c: shrike_native.action_collection_query(
                    c, query, with_fields=fields == "full", limit=effective_limit
                )
            )
        except NoteTypeOpError as e:
            # The native input error (a malformed search expression); the
            # decoder already strips Anki's U+2068/U+2069 isolation marks.
            raise ToolInputError(str(e)) from e
        result = ListNotesResponse.model_validate_json(raw)
        note_outcome(f"{len(result.notes)}/{result.total} notes")
        return result

    @_action
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
        limit: Annotated[
            int,
            Field(
                ge=0,
                le=50,
                description="Maximum results per query or source ID. Default 20. 0 returns all.",
            ),
        ] = 20,
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
        tier: Annotated[
            Literal["full", "live"],
            Field(
                description=(
                    "The live-search tier contract (#181): 'live' runs only the "
                    "no-embedding signals (exact substring + fuzzy) for per-keystroke "
                    "latency and returns completeness='partial'; 'full' (default) adds "
                    "the semantic + tag signals. Same fused result shape either way."
                ),
            ),
        ] = "full",
        version: Annotated[
            int | None,
            Field(
                description=(
                    "Opaque client sequence number, echoed back verbatim — drop any "
                    "response whose echo doesn't match your latest request (the "
                    "stale-live-search guard; the server is stateless per request)."
                ),
            ),
        ] = None,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
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
        skipped) and are not subject to `threshold`. Within a group, results are
        ordered by Reciprocal Rank Fusion of the signals, with every literal
        (`substring`) hit floated above non-literal ones, then by fused rank.

        Use this for conceptual queries keyword search can't handle and for
        finding exact wording. At least one of `queries` or `ids` is required."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        if not queries and not ids:
            raise ToolInputError("At least one of queries or ids must be provided.")

        logger.debug(
            "search_notes queries=%d ids=%d limit=%d threshold=%.2f",
            len(queries or []),
            len(ids or []),
            limit,
            threshold,
        )
        if queries:
            logger.debug("search_notes query strings: %s", queries)
        if ids:
            logger.debug("search_notes source ids: %s", ids)

        # Substring matching needs no embeddings; semantic ranking does.
        semantic_ok = index is not None and index.available and index.state == IndexState.READY
        message: str | None = None
        # The live tier: the caller wants the cheap signals only — "partial"
        # promises a fuller answer on a tier="full" re-request.
        completeness: Literal["partial", "full"] = (
            "partial" if (tier == "live" and semantic_ok) else "full"
        )
        if tier == "live":
            semantic_ok = False
        elif (
            semantic_ok
            and queries
            and not ids
            and all(len(q.strip()) < MIN_SEMANTIC_QUERY_CHARS for q in queries)
        ):
            # The min-query gate: typing fragments never burn an embedding
            # call. This IS the final answer for this query → stays "full".
            semantic_ok = False
            message = (
                f"Semantic ranking skipped (queries shorter than "
                f"{MIN_SEMANTIC_QUERY_CHARS} characters); exact text matches only."
            )
        if tier != "live" and not semantic_ok and message is None:
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
                return SearchResponse(message=message, completeness=completeness, version=version)
        elif not semantic_ok and not queries:
            # The live tier with id anchors only: anchors are semantic-only,
            # so there is nothing the cheap signals can do.
            return SearchResponse(message=message, completeness=completeness, version=version)

        if deck:
            # Accept a deck name, #id, or numeric id; an explicit id that matches
            # nothing yields no results.
            deck = await wrapper.resolve_deck_ref(deck)
            if deck is None:
                return SearchResponse(
                    results=[],
                    message="No deck matches that reference.",
                    completeness=completeness,
                    version=version,
                )

        exclude_set = set(exclude_ids or [])

        # Assemble search sources: each query string (semantic + substring) and
        # each id anchor (semantic only). Anchors are excluded from their results.
        sources: list[tuple[str, str, bool]] = []  # (label, text, is_query)
        for q in queries or []:
            sources.append((q, q, True))
        if ids:
            note_texts = await wrapper.note_texts_for_embedding(ids)
            for nid, text in zip(ids, note_texts, strict=True):
                if text:
                    sources.append((f"note #{nid}", text, False))
                    exclude_set.add(nid)

        if not sources:
            return SearchResponse(
                message="No valid queries or note IDs to search.",
                completeness=completeness,
                version=version,
            )

        # Query vectors (host-side embedding); the assembly itself runs in
        # shrike_kernel::actions.
        vectors: list[list[float]] = []
        if semantic_ok:
            assert index is not None
            # Off the event loop: embed_queries blocks on backend inference /
            # HTTP; inline it froze every concurrent request.
            embedded = await asyncio.to_thread(index.embed_queries, [t for (_, t, _) in sources])
            if embedded is None:
                semantic_ok = False
            else:
                vectors = embedded

        # Cross-space inputs: the PRIMARY space stays host-embedded above (the
        # query LRU). Each SECONDARY text-capable space embeds the query with
        # its own model + searches its own engine on the KERNEL runtime (where
        # embed is legal — action_search_notes runs on the collection-actor
        # thread and can't await embed), returning the per-space SpaceSemantic
        # rows the kernel fuses with the gate. EMPTY ("[]") when there are no
        # secondary spaces — the N=1 case stays byte-identical.
        # limit==0 means "return all". The native search applies the cap
        # lazily (`.take()`), so a large sentinel reads as "all" for the lexical
        # signals. The semantic path is the exception: the per-modality engine
        # over-fetches `k * SEARCH_OVERFETCH` and hands that to USearch's
        # `search(k)`, which *allocates* a result buffer of that size — a raw
        # `_UNBOUNDED_LIMIT` would ask USearch for billions of slots and hang. An
        # index holds at most `index.size` vectors, so that is the true "all"
        # bound for the semantic over-fetch; the lexical-only path (no usable
        # index) keeps the cheap FTS5 `LIMIT` sentinel.
        if limit > 0:
            fetch_k = limit
        elif semantic_ok and index is not None:
            fetch_k = max(index.size, 1)
        else:
            fetch_k = _UNBOUNDED_LIMIT

        cross_space_json: str | None = None
        if semantic_ok and kernel is not None:
            source_texts = [t for (_, t, _) in sources]
            cross_space_json = await kernel.build_cross_space_json(source_texts, fetch_k)

        # Orchestrator state: the image activation floor and the index size for
        # the over-fetch clamp.
        image_floor = (
            activation_floor(index.activation_stats.get("image"), ACTIVATION_MARGIN)
            if index is not None
            else None
        )
        # The kernel's Arc-shared native engine handle (KernelIndexView.engine).
        index_handle = index.engine if (semantic_ok and index is not None) else None
        derived_handle = (
            derived._engine._rust
            if derived is not None and derived.available and derived._engine is not None
            else None
        )

        raw = await wrapper.run(
            lambda c: shrike_native.action_search_notes(
                c,
                index_handle,
                derived_handle,
                sources,
                vectors,
                fetch_k,
                threshold,
                deck=deck,
                tags=tags or None,
                exclude=sorted(exclude_set),
                kernel=kernel,
                image_floor=image_floor,
                semantic=semantic_ok,
                index_size=index.size if index is not None else 0,
                cross_space=cross_space_json,
            )
        )
        groups = TypeAdapter(list[SearchResultGroup]).validate_json(raw)
        # Activation-floor calibration feedstock: one sample per query group —
        # the best SEMANTIC cosine, or a no-match tick. A lexical-only hit has
        # `score=None` and never pollutes the semantic sample.
        if dedup_stats is not None:
            for group in groups:
                sem_scores = [m.score for m in group.matches if m.score is not None]
                dedup_stats.record(max(sem_scores) if sem_scores else None)
        note_outcome(f"{len(groups)} groups, {sum(len(g.matches) for g in groups)} matches")
        return SearchResponse(
            results=groups, message=message, completeness=completeness, version=version
        )

    @_action
    async def upsert_notes(
        notes: Annotated[
            list[NoteInput],
            Field(
                min_length=1,
                max_length=100,
                description="Array of note objects to create or update.",
            ),
        ],
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
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
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

        This is a write-only op: it commits the notes and returns their per-item
        status. The vector index and lexical store update in the background
        (eventual consistency), so a just-written note may not appear in a
        `search_notes` immediately. To check for near-duplicates, `search_notes`
        with the planned content *before* writing; to find notes similar to one
        you just wrote, `search_notes` with its content or `ids=[<note id>]`."""
        _, _, kernel, _, _ = (await _route(collection)).unpack()
        creates = sum(1 for n in notes if n.id is None)
        updates = len(notes) - creates
        logger.debug(
            "upsert_notes count=%d (creates=%d, updates=%d) on_duplicate=%s dry_run=%s",
            len(notes),
            creates,
            updates,
            on_duplicate,
            dry_run,
        )

        note_dicts = [n.model_dump(exclude_none=True) for n in notes]
        # ONE maintained kernel op — write + index + derived + watermarks all
        # happen kernel-side; the response is write-only (per-item status + id).
        results = json.loads(
            await kernel.upsert_notes_json(json.dumps(note_dicts), on_duplicate, dry_run)
        )

        counts: dict[str, int] = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        note_outcome(
            f"{counts.get('created', 0)} created, {counts.get('updated', 0)} updated, "
            f"{counts.get('ok', 0)} ok, {counts.get('skipped', 0)} skipped, "
            f"{counts.get('error', 0)} errors"
        )

        # Write-only: the notes are committed (and the index/derived maintenance
        # is enqueued kernel-side, draining in the background). No read-after-
        # write on the response path; the dedup/activation calibration sampler
        # rides the `search_notes` path instead.
        return UpsertNotesResponse.model_validate({"results": results, "dry_run": dry_run})

    @_action
    async def upsert_note_types(
        note_types: Annotated[
            list[NoteTypeInput],
            Field(
                min_length=1,
                max_length=10,
                description="Array of note type definitions to create or update.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
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
        its cards. Adding entries appends empty fields / new cards.

        Because the replace is positional, a `fields` update may only rename in
        place, append, or drop trailing fields. Anything that would *move* an
        existing field — a reorder, an insert before another field, or a
        non-trailing remove — is rejected (it would silently mislabel note
        data); use update_note_type_fields (reposition / add / remove / rename)
        for those."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        names = [nt.name or f"id={nt.id}" for nt in note_types]
        logger.debug("upsert_note_types count=%d names=%s", len(note_types), ", ".join(names))

        nt_dicts = [nt.model_dump(exclude_none=True) for nt in note_types]
        results = json.loads(await kernel.upsert_note_types(json.dumps(nt_dicts)))

        for r in results:
            status = r.get("status", "unknown")
            if status == "error":
                logger.warning(
                    "upsert_note_types failed for %s: %s", r.get("name", "?"), r["error"]
                )

        return UpsertNoteTypesResponse.model_validate({"results": results})

    @_action
    async def update_note_type_fields(
        note_type: Annotated[
            str, Field(min_length=1, description="Name of the note type to edit.")
        ],
        operations: Annotated[
            list[FieldOp],
            Field(
                min_length=1,
                max_length=50,
                description="Field operations to apply, in order.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> UpdateNoteTypeFieldsResponse:
        """Edit a note type's fields by name, preserving note data.

        Apply a sequence of field operations to an existing note type:
        - `add`: add a new field (optionally at a 0-based `position`; appended
          otherwise).
        - `remove`: remove a field by name — drops that field's data from every
          note of this type.
        - `rename`: rename a field; its data is preserved.
        - `reposition`: move a field to a new 0-based `position`; its data moves
          with it.

        Operations apply in order, so a `rename` followed by an op naming the
        new name is valid. The whole call is atomic: if any operation is invalid
        (unknown field, name clash, out-of-range position, or removing the last
        remaining field), nothing is changed.

        Unlike upsert_note_types — which replaces the whole field list by
        position, so it can only rename in place, append, or drop the tail —
        these operations are addressed by field name and can truly move, insert,
        or remove a non-trailing field. Returns the resulting ordered field
        names."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("update_note_type_fields %r ops=%d", note_type, len(operations))
        op_dicts = [op.model_dump(exclude_none=True) for op in operations]
        try:
            result = json.loads(
                await kernel.update_note_type_fields(note_type, json.dumps(op_dicts))
            )
        except NoteTypeOpError as e:
            raise ToolInputError(str(e)) from e
        note_outcome(f"fields -> {result['fields']}")
        return UpdateNoteTypeFieldsResponse.model_validate(result)

    @_action
    async def update_note_type_templates(
        note_type: Annotated[
            str, Field(min_length=1, description="Name of the note type to edit.")
        ],
        operations: Annotated[
            list[TemplateOp],
            Field(
                min_length=1,
                max_length=50,
                description="Card-template operations to apply, in order.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> UpdateNoteTypeTemplatesResponse:
        """Edit a note type's card templates by name, preserving cards.

        The template counterpart of update_note_type_fields. Apply a sequence of
        operations to an existing note type's card templates:
        - `add`: add a new template (`front`/`back` HTML; optional 0-based
          `position`, appended otherwise) — generates a new card per note.
        - `remove`: remove a template by name — deletes that template's cards
          (and their scheduling) from every note of this type.
        - `rename`: rename a template; a label change only, cards are untouched.
        - `reposition`: move a template to a new 0-based `position`; its cards
          (and scheduling) move with it.

        Operations apply in order; the whole call is atomic (an invalid op —
        unknown template, name clash, out-of-range position, or removing the
        last remaining template — changes nothing).

        Unlike upsert_note_types — which replaces the whole template list by
        position, so it can only rename/edit in place, append, or drop the tail
        — these operations are addressed by template name and can truly move,
        insert, or remove a non-trailing template. To change a template's
        front/back HTML in place, use upsert_note_types. Returns the resulting
        ordered template names."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("update_note_type_templates %r ops=%d", note_type, len(operations))
        op_dicts = [op.model_dump(exclude_none=True) for op in operations]
        try:
            result = json.loads(
                await kernel.update_note_type_templates(note_type, json.dumps(op_dicts))
            )
        except NoteTypeOpError as e:
            raise ToolInputError(str(e)) from e
        note_outcome(f"templates -> {result['templates']}")
        return UpdateNoteTypeTemplatesResponse.model_validate(result)

    @_action
    async def find_replace_note_types(
        note_type: Annotated[
            str, Field(min_length=1, description="Name of the note type to edit.")
        ],
        search: Annotated[
            str,
            Field(min_length=1, description="Text (or regex, if `regex`) to find."),
        ],
        replace: Annotated[
            str,
            Field(
                description=(
                    "Replacement text. Literal by default; with `regex`, $1/\\1 "
                    "refer to capture groups."
                )
            ),
        ],
        front: Annotated[
            bool, Field(description="Search each card template's front (question) HTML.")
        ] = True,
        back: Annotated[
            bool, Field(description="Search each card template's back (answer) HTML.")
        ] = True,
        css: Annotated[bool, Field(description="Search the note type's shared CSS.")] = True,
        regex: Annotated[
            bool, Field(description="Treat `search` as a Python regular expression.")
        ] = False,
        match_case: Annotated[
            bool,
            Field(description="Case-sensitive match. Default true — template/CSS text is code."),
        ] = True,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> FindReplaceNoteTypesResponse:
        """Find and replace text inside one note type's templates and CSS.

        Edits the note type *definition* — each card template's front (`qfmt`)
        and back (`afmt`) HTML and the shared CSS — not note field values. No
        note is touched. Use `front`/`back`/`css` to pick where to search (all
        on by default). Typical uses: fix a `{{OldField}}` reference across a
        model's templates after a field rename, swap a CSS class or colour, or
        correct a typo in template markup for all of a note type's cards at once.

        `search` is literal text unless `regex` is set, in which case it is a
        Python regular expression and `replace` may use `$1`/`\\1` capture
        references. `match_case` defaults to true because template and CSS text
        is code (field names, class names) where case is significant. The model
        is saved only if at least one replacement is made. Returns the total
        replacement count, the templates whose front/back changed, and whether
        the CSS changed.

        For renaming a *field* itself (and migrating note data), use
        update_note_type_fields; this tool only rewrites the template text that
        references fields."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        if not (front or back or css):
            raise ToolInputError("Enable at least one of `front`, `back`, or `css`.")
        logger.debug(
            "find_replace_note_types %r search=%r front=%s back=%s css=%s regex=%s",
            note_type,
            search,
            front,
            back,
            css,
            regex,
        )
        # The kernel op carries the watermark tail (a real replace bumps
        # col.mod without touching vectors; a no-op bumps nothing).
        try:
            result = json.loads(
                await kernel.find_replace_note_types(
                    note_type,
                    search,
                    replace,
                    regex,
                    match_case,
                    front,
                    back,
                    css,
                )
            )
        except NoteTypeOpError as e:
            raise ToolInputError(str(e)) from e
        note_outcome(
            f"{result['replacements']} replacement(s) in "
            f"{len(result['templates_changed'])} template(s), css={result['css_changed']}"
        )
        return FindReplaceNoteTypesResponse.model_validate(result)

    @_action
    async def update_note_type_field_metadata(
        note_type: Annotated[
            str, Field(min_length=1, description="Name of the note type to edit.")
        ],
        fields: Annotated[
            list[FieldMetadataInput],
            Field(
                min_length=1,
                max_length=100,
                description="Per-field metadata updates, addressed by field name.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> UpdateNoteTypeFieldMetadataResponse:
        """Set a note type's per-field editor metadata: font, size, description.

        These are **editor cosmetics** — the font and size used when editing a
        field in Anki, and the description (hint text) shown for it. They have no
        effect on note content, card rendering, or search. Each update is keyed by
        field `name` and sets only the attributes you provide (`font`, `size`,
        `description`); others are left unchanged. At least one attribute per
        update. The call is atomic — an unknown field name changes nothing.

        Read the current values from collection_info's note type details
        (`note_type_details`), which include each field's font/size/description.
        To change which fields exist or their order, use update_note_type_fields."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("update_note_type_field_metadata %r fields=%d", note_type, len(fields))
        updates = [f.model_dump(exclude_none=True) for f in fields]
        # The kernel op carries the watermark tail (editor metadata isn't
        # embedding text — col_mod advances, no re-embed).
        try:
            result = json.loads(
                await kernel.update_note_type_field_metadata(note_type, json.dumps(updates))
            )
        except NoteTypeOpError as e:
            raise ToolInputError(str(e)) from e
        note_outcome(f"updated {result['fields_updated']}")
        return UpdateNoteTypeFieldMetadataResponse.model_validate(result)

    @_action
    async def delete_notes(
        ids: Annotated[
            list[int],
            Field(
                min_length=1,
                max_length=100,
                description="Note IDs to delete. Their cards are deleted too.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> DeleteNotesResponse:
        """Permanently delete notes and all their associated cards.

        This cannot be undone. Use list_notes or search_notes first to
        verify which notes will be deleted."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("delete_notes requested=%d", len(ids))
        # ONE maintained kernel op: the existence partition, the anki delete,
        # and the sidecar drop (vectors + fingerprints + derived rows +
        # watermark advance) run in a single op — the maintained-write-path
        # invariant. Best-effort tail is internal to the op: the notes are gone
        # from the collection either way, and a failed sidecar leaves the
        # watermark behind for next-boot drift.
        result = json.loads(await kernel.delete_notes(ids))
        note_outcome(f"{len(result['deleted'])} deleted, {len(result['not_found'])} not found")
        return DeleteNotesResponse.model_validate(result)

    @_action
    async def find_replace_notes(
        search: Annotated[
            str, Field(min_length=1, description="Text (or regex) to find in note fields.")
        ],
        replace: Annotated[
            str,
            Field(description="Replacement text. In regex mode, capture refs use Anki's `$1`."),
        ],
        *,
        regex: Annotated[
            bool, Field(description="Treat `search` as a regular expression (Anki's engine).")
        ] = False,
        match_case: Annotated[
            bool, Field(description="Case-sensitive match. Default false.")
        ] = False,
        field: Annotated[
            str | None,
            Field(description="Restrict to this single field name; omit to search all fields."),
        ] = None,
        deck: Annotated[
            str | None,
            Field(
                description="Scope to this deck (name, numeric id, or #id; includes child decks)."
            ),
        ] = None,
        tags: Annotated[
            list[str],
            Field(default_factory=list, description="Scope to notes having all of these tags."),
        ],
        note_type: Annotated[
            str | None, Field(description="Scope to notes using this note type.")
        ] = None,
        ids: Annotated[
            list[int], Field(default_factory=list, description="Scope to these note IDs.")
        ],
        dry_run: Annotated[
            bool,
            Field(
                description="Preview only — report what would change without modifying. "
                "Default false (applies the edit)."
            ),
        ] = False,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> FindReplaceResponse:
        """Find and replace text across the fields of a scoped set of notes.

        A scope is **required**: at least one of `deck`, `tags`, `note_type`, or
        `ids` (the same filters as list_notes). `search` is literal by default;
        set `regex` for a regular expression (Anki's engine — capture references
        in `replace` use `$1`). `field` restricts to one field; otherwise all
        fields are searched.

        By default this **applies** the change and returns `notes_changed` with a
        sample of before/after edits; pass `dry_run` to preview without modifying.
        Changed notes are re-embedded so semantic search stays correct, and the
        edit is undoable in Anki. For literal searches the dry-run preview matches
        the apply exactly; for regex the preview is a best-effort sample and the
        apply is authoritative."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        if not any([deck, tags, note_type, ids]):
            raise ToolInputError("A scope is required: deck, tags, note_type, or ids.")

        logger.debug(
            "find_replace_notes search=%r regex=%s field=%s dry_run=%s",
            search,
            regex,
            field,
            dry_run,
        )
        try:
            result = await wrapper.find_replace(
                search,
                replace,
                regex=regex,
                match_case=match_case,
                field=field,
                deck=deck,
                tags=tags or None,
                note_type=note_type,
                ids=ids or None,
                dry_run=dry_run,
            )
        except re.error as e:
            # A malformed pattern/backref is caller-supplied bad input, not a
            # server bug — and the preview loop compiles it on EVERY call,
            # including a real apply. Raise a clean rejection (WARNING, no
            # traceback) instead of the catch-all's "Unhandled error" + traceback.
            raise ToolInputError(f"Invalid regular expression: {e}") from e
        changed_ids = result.pop("changed_ids", [])
        note_outcome(
            f"{'would change' if dry_run else 'changed'} {result['notes_changed']} note(s)"
        )

        if not dry_run and changed_ids:
            # One maintained kernel op re-embeds + re-ingests the set.
            try:
                await kernel.reindex_notes(changed_ids)
            except Exception:
                logger.warning("Failed to update index after find_replace", exc_info=True)

        return FindReplaceResponse.model_validate(result)

    @_action
    async def migrate_note_type(
        note_ids: Annotated[
            list[int],
            Field(
                min_length=1,
                max_length=1000,
                description=(
                    "Notes to migrate. They must all currently share one note type "
                    "(a single field/template map can't apply to mixed types)."
                ),
            ),
        ],
        new_note_type: Annotated[
            str, Field(min_length=1, description="Name of the note type to migrate the notes to.")
        ],
        field_map: Annotated[
            dict[str, str],
            Field(
                min_length=1,
                description=(
                    "Map of source field name to target field name. Source fields "
                    "omitted here are dropped (their content is lost). Two source "
                    "fields may not map to the same target field."
                ),
            ),
        ],
        template_map: Annotated[
            dict[str, str],
            Field(
                default_factory=dict,
                description=(
                    "Optional map of source card-template name to target template "
                    "name. Omit to let Anki map templates by position."
                ),
            ),
        ],
        dry_run: Annotated[
            bool,
            Field(
                description=(
                    "Preview only — report what would change (including dropped "
                    "fields) without modifying anything. Default false (applies)."
                )
            ),
        ] = False,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> MigrateNoteTypeResponse:
        """Change a set of notes from one note type to another.

        Migrates `note_ids` (which must all currently share one note type) to
        `new_note_type`, moving field content per `field_map` (source field name →
        target field name). This is Anki's "Change Note Type": note IDs and — for
        mapped card templates — review scheduling are preserved, so it's the
        history-safe way to convert Basic↔Cloze, consolidate redundant note types,
        or adopt a richer template.

        It is **data-affecting**: a source field not named in `field_map` is
        dropped and its content lost (reported in `dropped_fields`); target fields
        nothing maps into start empty (`new_empty_fields`). The mapping is
        explicit — unknown field names, or two source fields mapping to one
        target, are errors rather than guesses. Use `dry_run` to preview the drops
        first. To create or edit notes without changing type, use upsert_notes."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug(
            "migrate_note_type notes=%d -> %s dry_run=%s",
            len(note_ids),
            new_note_type,
            dry_run,
        )
        # The kernel op migrates and, on apply, re-embeds + re-ingests the
        # changed notes (empty template_map = map by ordinal).
        try:
            result = json.loads(
                await kernel.migrate_note_type(
                    note_ids,
                    new_note_type,
                    json.dumps(field_map),
                    json.dumps(template_map) if template_map else "",
                    dry_run,
                )
            )
        except ValueError as e:
            raise ToolInputError(str(e)) from e

        note_outcome(
            f"{'would migrate' if dry_run else 'migrated'} {len(result['changed'])} note(s) "
            f"{result['from_note_type']} -> {result['to_note_type']}, "
            f"dropped={result['dropped_fields']}"
        )
        return MigrateNoteTypeResponse.model_validate(result)

    @_action
    async def delete_note_types(
        ids: Annotated[
            list[int],
            Field(
                min_length=1,
                max_length=10,
                description="Note type IDs to delete. Fails for any note type still in use.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> DeleteNoteTypesResponse:
        """Delete note type definitions by ID.

        A note type can only be deleted if no notes currently use it.
        Check use counts via collection_info first."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("delete_note_types requested=%d", len(ids))
        result = json.loads(await kernel.delete_note_types(ids))
        statuses: dict[str, int] = {}
        for r in result["results"]:
            s = r["status"]
            statuses[s] = statuses.get(s, 0) + 1
        note_outcome(str(statuses))
        return DeleteNoteTypesResponse.model_validate(result)

    @_action
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
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
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
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        set_mode = set is not None
        addremove_mode = bool(add or remove)
        if set_mode and addremove_mode:
            raise ToolInputError("Provide either `set` (full replace) or `add`/`remove`, not both.")
        if not set_mode and not addremove_mode:
            raise ToolInputError("Specify `set`, or `add` and/or `remove`.")

        if set_mode:
            logger.debug("update_note_tags notes=%d set=%s", len(note_ids), set)
        else:
            logger.debug("update_note_tags notes=%d add=%s remove=%s", len(note_ids), add, remove)

        # The kernel op carries the watermark tail.
        result = UpdateNoteTagsResponse.model_validate_json(
            await kernel.update_note_tags(note_ids, set_tags=set, add=add, remove=remove)
        )
        note_outcome(f"modified {result.notes_modified} note(s), {len(result.not_found)} not found")
        return result

    @_action
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
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> RenameTagResponse:
        """Rename a tag, collection-wide or on a set of notes.

        With no `note_ids`, the tag is renamed everywhere it appears (and so are
        its children, e.g. renaming "history" moves "history::ww2"). With
        `note_ids`, only those notes are affected and the tag is matched exactly
        — renaming "jp" never touches "jp-verbs".

        Returns the number of notes whose tags changed."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        if old == new:
            raise ToolInputError("`old` and `new` tags are identical — nothing to rename.")
        logger.debug("rename_tag %r -> %r (scope=%d notes)", old, new, len(note_ids))
        # The kernel op carries the watermark tail.
        result = RenameTagResponse.model_validate_json(await kernel.rename_tag(old, new, note_ids))
        note_outcome(f"modified {result.notes_modified} note(s)")
        return result

    @_action
    async def collection_prune(
        unused_tags: Annotated[
            bool,
            Field(description="Remove tag-registry names that no note uses any more."),
        ] = False,
        empty_notes: Annotated[
            bool,
            Field(
                description=(
                    "Delete notes whose every field is blank. A field counts as "
                    "blank only if it has no text and no media, so an image- or "
                    "audio-only note is kept."
                )
            ),
        ] = False,
        empty_cards: Annotated[
            bool,
            Field(
                description=(
                    "Remove cards that render empty (e.g. a cloze card with no "
                    "matching deletion). A note that loses its last card is deleted."
                )
            ),
        ] = False,
        unused_media: Annotated[
            bool,
            Field(
                description=(
                    "Move media files that no note references to Anki's trash "
                    "(recoverable). See collection_check to preview them without pruning."
                )
            ),
        ] = False,
        dry_run: Annotated[
            bool,
            Field(
                description=(
                    "Preview only — report what would be removed without changing "
                    "anything. Defaults to false (the cleanups apply); pass true to "
                    "preview first."
                )
            ),
        ] = False,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> CollectionPruneResponse:
        """Clean up the collection: unused tags, empty notes, empty cards, unused media.

        Runs the cleanups you enable. If you set **none** of `unused_tags`,
        `empty_notes`, `empty_cards`, or `unused_media`, all run. Each enabled
        cleanup is reported in its own section of the response; a section is
        absent when its cleanup was not requested.

        This **applies by default** — it removes the unused tags, empty notes,
        empty cards, and unused media it finds. Pass `dry_run: true` to preview
        instead: it reports the unused tag names, the empty note IDs, the empty
        card count (with any notes that would be deleted), and the unused media
        filenames without mutating anything. The op is destructive (media goes
        to Anki's recoverable trash; deleted notes/cards do not), so preview
        with `dry_run: true` first if unsure.

        An empty note is one whose every field is blank, where a field is blank
        only if it has no text *and* no media — so a card that is just an image
        or audio clip is never removed. On apply, empty notes are removed first,
        then empty cards, then unused tags, then unused media (so tags and media
        freed by the deletions are cleared in the same call)."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        # No selection means "prune everything".
        if not (unused_tags or empty_notes or empty_cards or unused_media):
            unused_tags = empty_notes = empty_cards = unused_media = True
        logger.debug(
            "collection_prune unused_tags=%s empty_notes=%s empty_cards=%s "
            "unused_media=%s dry_run=%s",
            unused_tags,
            empty_notes,
            empty_cards,
            unused_media,
            dry_run,
        )
        # The kernel op runs the cleanups AND the index maintenance tail
        # (deletions drop their sidecars; a tags-only prune advances the
        # watermarks). Removed note ids stay kernel-internal.
        result = CollectionPruneResponse.model_validate_json(
            await kernel.collection_prune(
                unused_tags, empty_notes, empty_cards, unused_media, dry_run
            )
        )
        removed = (len(result.empty_notes.removed) if result.empty_notes else 0) + (
            len(result.empty_cards.notes_deleted) if result.empty_cards else 0
        )
        note_outcome(
            f"{'previewed' if dry_run else 'applied'}: {removed} note(s) removed, "
            f"tags={result.unused_tags.removed if result.unused_tags else '-'}"
        )
        return result

    @_action
    async def store_media(
        items: Annotated[
            list[StoreMediaItem],
            Field(
                min_length=1,
                max_length=10,
                description="Media files to store (1-10). Each carries base64 `data`, a "
                "`url`, or a server-local `path`.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> StoreMediaResponse:
        """Store media files in the collection's media folder (1-10 per call).

        Each item provides exactly one source: base64 `data` (requires a
        `filename` with an extension — the bytes alone don't say what the file is),
        a `url` the server fetches (filename derived from the URL or its
        Content-Type if you omit it), or a server-local `path` (see below). This is
        the write path for authoring cards with images or audio: store the asset,
        then reference the returned filename in a note field (`<img src="NAME">` or
        `[sound:NAME]`).

        URL fetches are restricted to http/https and refuse private/loopback
        addresses by default (an SSRF guard). A `path` reads a file on the
        **server's** filesystem and is **off by default**: it is honored only when
        the operator has configured a `--media-path-root` on a purely-local daemon,
        and only for files contained in that root; a `path` item is rejected
        otherwise. To store a local file against any server, the CLI
        `shrike media store PATH` reads it and sends the bytes.

        Anki resolves name collisions: identical content keeps the name (reported
        `deduped`), different content under the same name gets a hashed suffix, so
        the stored `filename` may differ from what you asked for — always reference
        the returned name. Per-item errors (bad base64, unfetchable URL, disabled
        or out-of-root path, oversize) are reported per item and don't sink the batch."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("store_media count=%d", len(items))
        # The kernel prepares byte sources concurrently on its blocking pool
        # and writes the batch as one collection job.
        item_dicts = [i.model_dump(exclude_none=True) for i in items]
        results = json.loads(
            await kernel.store_media(
                json.dumps(item_dicts),
                allow_private_fetch,
                server_path_roots or [],
            )
        )
        stored = sum(1 for r in results if r.get("status") == "stored")
        errors = len(results) - stored
        note_outcome(f"{stored} stored, {errors} errors")
        for r in results:
            if r.get("status") == "error":
                logger.warning("store_media item %d failed: %s", r.get("index"), r["error"])
        return StoreMediaResponse.model_validate({"results": results})

    @_action
    async def fetch_media(
        filenames: Annotated[
            list[str],
            Field(min_length=1, max_length=10, description="Media filenames to look up (1-10)."),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> FetchMediaResponse:
        """Locate media files in the collection's media folder (1-10 per call).

        This resolves names to where their bytes live — it never returns the bytes
        (base64 is useless to a model and wrecks context). Each present file comes
        back as `found` with a `url` (the server's `GET /media/<name>`) and a
        server-side `path`; a non-existent file is `missing`. **To get the actual
        bytes, GET the `url`** with your download/fetch tool, or read `path` if you
        share the server's disk. Every `found` file reports `url`, `path`, `mime`,
        and `size_bytes`."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("fetch_media count=%d", len(filenames))
        results = json.loads(await kernel.fetch_media(filenames))
        for r in results:
            if r["status"] == "found":
                r["url"] = _media_url(r["filename"])
        note_outcome(
            f"{sum(1 for r in results if r['status'] == 'found')} found, "
            f"{sum(1 for r in results if r['status'] == 'missing')} missing"
        )
        return FetchMediaResponse.model_validate({"results": results})

    @_action
    async def list_media(
        pattern: Annotated[
            str | None,
            Field(description="Optional glob (e.g. '*.png', 'cell-*') to filter filenames."),
        ] = None,
        limit: Annotated[
            int,
            Field(
                ge=0,
                le=1000,
                description=(
                    "Maximum filenames to return (the total still counts). "
                    "Default 20. 0 returns all."
                ),
            ),
        ] = 20,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> ListMediaResponse:
        """List filenames in the collection's media folder, and report its path.

        Optionally filter by a glob `pattern`. `count` is the total number of
        matching files; `files` is capped at `limit` (each with its `url`, `mime`,
        and `size_bytes`). `media_dir` is the absolute media-folder path; fetch any
        file's bytes by GETting its `url` (the server's `GET /media/<name>`)."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("list_media pattern=%s limit=%d", pattern, limit)
        # limit==0 means "return all": the native list_media treats a None
        # limit as unbounded.
        result = json.loads(await kernel.list_media(pattern, limit if limit > 0 else None))
        for f in result["files"]:
            f["url"] = _media_url(f["filename"])
        note_outcome(f"{len(result['files'])}/{result['count']} file(s)")
        return ListMediaResponse.model_validate(result)

    @_action
    async def delete_media(
        filenames: Annotated[
            list[str],
            Field(min_length=1, max_length=1000, description="Media filenames to delete."),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> DeleteMediaResponse:
        """Delete media files by name, moving them to Anki's media trash.

        Deletion is recoverable (Anki's trash) and sync-aware; it does not check
        whether any note still references the file, so removing a referenced asset
        will leave a broken `<img>`/`[sound:]` — use collection_check to find
        unreferenced ('unused') media first. Returns `deleted` and `not_found`
        name lists; a missing file is skipped, not an error."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("delete_media requested=%d", len(filenames))
        result = json.loads(await kernel.delete_media(filenames))
        note_outcome(f"{len(result['deleted'])} deleted, {len(result['not_found'])} not found")
        return DeleteMediaResponse.model_validate(result)

    @_action
    async def collection_check(
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> CollectionCheckResponse:
        """Report collection media-integrity issues (read-only, the sibling of collection_prune).

        Runs Anki's media check and returns `unused` (media files on disk that no
        note references — candidates for `collection_prune unused_media`), `missing`
        (filenames referenced by notes but absent from the media folder),
        `missing_media_notes` (the note IDs with such references), and `have_trash`
        (whether Anki's media trash holds anything). Nothing is modified."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("collection_check")
        result = json.loads(await kernel.media_check())
        note_outcome(
            f"{len(result['unused'])} unused, {len(result['missing'])} missing, "
            f"trash={result['have_trash']}"
        )
        return CollectionCheckResponse.model_validate(result)

    @_action
    async def import_package(
        path: Annotated[
            str,
            Field(
                min_length=1,
                description="Server-local path to the .apkg/.colpkg to import. The "
                "server reads this file off its own filesystem, so it is honored only "
                "when the operator configured an --import-path-root containing it (off "
                "by default).",
            ),
        ],
        update_notes: Annotated[
            Literal["if_newer", "always", "never"],
            Field(
                description="How to handle an imported note whose GUID matches an "
                "existing one: 'if_newer' (default) updates only when the imported note "
                "is newer; 'always' overwrites; 'never' keeps the existing note. New "
                "notes always add."
            ),
        ] = "if_newer",
        update_notetypes: Annotated[
            Literal["if_newer", "always", "never"],
            Field(description="Same condition, applied to note types. Default 'if_newer'."),
        ] = "if_newer",
        with_scheduling: Annotated[
            bool,
            Field(
                description="Import the package's review scheduling (due dates, "
                "intervals). Default false — Shrike manages cards, it does not review, "
                "so scheduling is normally left out."
            ),
        ] = False,
        merge_notetypes: Annotated[
            bool,
            Field(
                description="Merge imported note types into existing ones by name "
                "rather than adding new ones. Default false."
            ),
        ] = False,
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> ImportPackageResponse:
        """Import an Anki package (.apkg/.colpkg) into the collection.

        Pulls a shared deck (or the notes from a backup) into the collection, via
        anki's importer. This is a **merge**, NOT a destructive restore: notes are
        added or updated alongside what's already there — your collection is never
        replaced, even for a `.colpkg` (its notes are merged in like an `.apkg`'s).
        Returns per-bucket counts (notes added/updated/duplicate/conflicting/…) —
        see the response fields. The conflict behaviour is governed by
        `update_notes`/`update_notetypes` (default: update a same-GUID note only
        when the imported one is newer); brand-new notes always add. Scheduling is
        not imported by default.

        The `path` is read from the **server's** filesystem and is **off by
        default**: it is honored only when the operator configured an
        `--import-path-root` (on a purely-local daemon) containing the file —
        import writes into the collection, so it gets its own root distinct from
        media's. Importing mutates the collection, so the search index is
        reconciled afterward (`reindexed` reports whether that ran)."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("import_package path=%s", path)

        # Server-local-path safety gate, gated exactly like store_media's
        # `path` and export's `output_path` (the shared mechanism): the server
        # must be purely-local AND the package must resolve inside an
        # operator-allowed --import-path-root. `path_within_any_root` is the READ
        # gate — the package must already exist — with commonpath on realpath'd
        # sides closing `..`/symlink-escape. Fail-closed: empty roots (the
        # default) or a non-purely-local server rejects, opening nothing.
        if not (server_purely_local and path_within_any_root(path, server_import_path_roots or [])):
            raise ToolInputError(
                "import from a server-local path is not permitted: the server must be "
                "purely-local and the path must be inside an --import-path-root (and must "
                "exist). It is off by default — the operator enables it with --import-path-root."
            )

        raw, reindexed = await kernel.import_package(
            path, update_notes, update_notetypes, with_scheduling, merge_notetypes
        )
        summary = json.loads(raw)
        summary["reindexed"] = reindexed
        note_outcome(
            f"{summary['new']} new, {summary['updated']} updated, "
            f"{summary['duplicate']} duplicate (reindexed={reindexed})"
        )
        return ImportPackageResponse.model_validate(summary)

    @_action
    async def upsert_decks(
        decks: Annotated[
            list[DeckInput],
            Field(
                min_length=1,
                max_length=100,
                description="Array of deck objects to create or rename.",
            ),
        ],
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
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
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        creates = sum(1 for d in decks if d.id is None)
        logger.debug("upsert_decks count=%d (renames=%d)", len(decks), len(decks) - creates)

        # The kernel op carries the watermark tail.
        deck_dicts = [d.model_dump(exclude_none=True) for d in decks]
        results = json.loads(await kernel.upsert_decks(json.dumps(deck_dicts)))

        created = sum(1 for r in results if r.get("status") == "created")
        updated = sum(1 for r in results if r.get("status") == "updated")
        errors = sum(1 for r in results if r.get("status") == "error")
        note_outcome(f"{created} created, {updated} updated, {errors} errors")
        for r in results:
            if r.get("status") == "error":
                logger.warning("upsert_decks item %d failed: %s", r.get("index"), r["error"])
        return UpsertDecksResponse.model_validate({"results": results})

    @_action
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
        collection: Annotated[
            str | None, Field(default=None, description=COLLECTION_SELECTOR_DESCRIPTION)
        ] = None,
    ) -> DeleteDecksResponse:
        """Delete decks by name — only if they are already empty.

        A deck is deletable only when neither it nor any of its subdecks contains
        cards. To remove a non-empty deck, first move its notes elsewhere (e.g.
        upsert_notes with a new `deck`, or rename the deck onto another to merge),
        then delete the now-empty deck. This keeps deletion from ever destroying
        notes.

        Returns `deleted`, `not_found`, and `not_empty` name lists; a non-empty
        or missing deck is skipped, not an error."""
        wrapper, index, kernel, derived, dedup_stats = (await _route(collection)).unpack()
        logger.debug("delete_decks requested=%d", len(decks))
        # The kernel op carries the watermark tail.
        result = DeleteDecksResponse.model_validate_json(await kernel.delete_decks(decks))
        note_outcome(
            f"{len(result.deleted)} deleted, {len(result.not_found)} not found, "
            f"{len(result.not_empty)} not empty"
        )
        return result

    return actions
