"""Pydantic models for every Shrike tool request/response and server-status shape.

This module is the single source of truth for the wire contract. The server
tools return the response models here (so FastMCP emits an ``outputSchema`` for
each tool), the standalone client validates responses into them, the CLI renders
them, and ``scripts/gen_schema.py`` generates ``docs/mcp-schema.json`` from them.

Design rules:

- **Make illegal states unrepresentable.** When a field's presence is
  *correlated* with another (a hidden state — "you get progress only while
  building", "an error carries a message, a success carries an id"), model it as
  a discriminated union, not one bag of optionals. The pattern is a type alias::

      class Foo(BaseModel):
          status: Literal["foo"]
          ...
      class Bar(BaseModel):
          status: Literal["bar"]
          ...
      Thing = Annotated[Foo | Bar, Field(discriminator="status")]

  Each variant is a ``BaseModel`` with a ``Literal`` discriminator field, and the
  union is an ``Annotated[... , Field(discriminator=...)]`` alias. Prefer this
  everywhere a set of fields travels together under a tag. Two fields that always
  appear or vanish as a pair are the same smell at smaller scale — group them
  into a nested sub-model (``detail: Detail | None``) rather than two optionals.
  Because the alias is not a ``BaseModel``, validate it with
  ``TypeAdapter(Thing).validate_python(...)`` (a model field typed as ``Thing``
  validates automatically).
- A bare ``X | None`` is reserved for *independent* optionality — a datum that
  may genuinely be absent on its own, uncorrelated with any other field (e.g.
  ``col_mod`` before the index is built, a field omitted from a partial update).
  Annotate why, so the next reader knows it isn't laziness.
- Whole-call failures are not modeled here at all: tools raise, and the failure
  surfaces as an MCP ``isError`` result the client raises on. No response model
  carries an ``error`` field.
- Models tolerate unknown keys (Pydantic's default ``extra="ignore"``) so a
  newer server adding a field doesn't break an older client.
- No imports from the rest of ``shrike`` — keep this leaf-level to avoid cycles.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# Stable wire marker prefixing a tool's MCP isError text when the call failed
# because the collection couldn't be acquired (another process holds it, under
# cooperative locking — #65). The single source of truth shared by the server
# (collection.py builds the message) and the dependency-light client (which maps
# it to CollectionBusyError). Not a response model: busy is a transport-level
# error class, orthogonal to every tool's response (the op never ran).
COLLECTION_BUSY_CODE = "collection_busy"

# ============================================================================
# Request models (tool inputs)
# ============================================================================


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
    # A partial upsert input: ``None`` means "omitted" — absent on create, or
    # "leave unchanged" on update. Create-time requirements (deck/note_type/
    # fields) are enforced by the tool with field-specific messages, which read
    # better than a union-match failure, so this stays one partial model rather
    # than a NoteCreate | NoteUpdate union.
    id: int | None = Field(
        default=None,
        description="Note ID. Present = update existing note, absent = create new note.",
    )
    deck: str | None = Field(
        default=None,
        description=(
            'Target deck (e.g., "Japanese::Vocabulary"). Required for new notes. '
            "On update, moves the note to this deck. Accepts a deck name, numeric "
            "deck ID, or #ID; an unknown ID is an error, a new name is created."
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
    # Partial upsert input, same convention as NoteInput: ``None`` = omitted /
    # leave unchanged; create-time requirements enforced by the tool.
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
            "Ordered list of field names. Required for new note types. On update, "
            "replaces the field list by position — the field at each position keeps "
            "its note data even when renamed; only shortening the list drops the "
            "trailing fields' data. May only rename in place, append, or drop trailing "
            "fields; moving/inserting/removing a non-trailing field is rejected (use "
            "update_note_type_fields)."
        ),
    )
    templates: list[TemplateInput] | None = Field(
        default=None,
        description=(
            "Card templates. Required for new note types. On update, replaced by "
            "position like fields, preserving existing cards and their scheduling. "
            "May only rename/edit in place, append, or drop trailing templates; "
            "moving/inserting/removing a non-trailing template is rejected (use "
            "update_note_type_templates)."
        ),
    )
    css: str | None = Field(
        default=None,
        description="CSS styling shared across all cards. Required for new note types.",
    )
    is_cloze: bool | None = Field(
        default=None,
        description="If true, this is a cloze deletion note type. Cannot be changed on update.",
    )


# ============================================================================
# Shared nested models
# ============================================================================


class Note(BaseModel):
    """A note as returned by list_notes (mirrors CollectionWrapper._note_to_dict)."""

    id: int
    note_type: str
    deck: str
    tags: list[str] = []
    modified: str
    # Independent projection: present in "full" mode, omitted in "meta" mode.
    content: dict[str, str] | None = None


class SubstringInfo(BaseModel):
    """Evidence that the query text occurs literally in a note."""

    matched_fields: list[str] = []
    snippet: str | None = None


class SearchMatch(Note):
    """A search result with per-mechanism match evidence.

    A hit can be semantically ranked, an exact-substring hit, or both. Each
    annotation below is independently optional and absent when its mechanism did
    not contribute — but a returned match always carries at least one. This is the
    extension point for future retrieval backends (n-gram / fuzzy / prefix, #98):
    they add further optional evidence fields here, never a new param or tool.
    """

    # Semantic similarity (0-1); None when the hit matched only by exact substring
    # or the vector index was unavailable. Independent of `substring`.
    score: float | None = None
    # Present when the query text occurs literally in the note.
    substring: SubstringInfo | None = None


class Neighbor(BaseModel):
    """A similar note attached to an upsert result."""

    id: int
    score: float
    tags: list[str] = []


class TemplateInfo(BaseModel):
    name: str
    front: str
    back: str


class FieldDetail(BaseModel):
    """A field's editor metadata (cosmetics — no bearing on note data or cards)."""

    name: str
    font: str  # edit-time font family
    size: int  # edit-time font size (px)
    description: str  # hint text shown in the note editor


class NoteTypeDetail(BaseModel):
    """Full definition (templates/CSS) plus per-field editor metadata."""

    templates: list[TemplateInfo]
    css: str
    fields: list[FieldDetail] = []


class NoteTypeInfo(BaseModel):
    name: str
    id: int
    fields: list[str] = []
    type: str = "standard"
    # Grouped (not two correlated optionals): the full definition is present only
    # for note types named in collection_info's `note_type_details`. A single
    # response can mix summary (detail=None) and detailed entries, so this is
    # genuine independent optionality at the model level.
    detail: NoteTypeDetail | None = None


class DeckInfo(BaseModel):
    name: str
    id: int
    note_count: int = 0


class Summary(BaseModel):
    path: str
    created: str
    modified: str
    notes: int
    cards: int
    decks: int
    note_types: int
    tags: int
    due_today: int


class DeckStat(BaseModel):
    notes: int = 0
    due: int = 0


class Stats(BaseModel):
    total_notes: int = 0
    total_cards: int = 0
    cards_due_today: int = 0
    new_cards: int = 0
    decks_summary: dict[str, DeckStat] = {}


# ============================================================================
# Per-item result variants (discriminated unions on `status`)
#
# Each tool reports per-item outcomes as a precise variant — a success carries
# its real fields, an error carries its own — so the schema (and the LLM) sees
# exactly which fields accompany each status, with no optional soup. Whole-call
# failures are NOT modeled here: they surface as MCP ``isError`` results.
# ============================================================================


# Why a candidate note cannot be added, mirroring Anki's own NoteFieldsCheckResult
# (run via note.fields_check()) plus the two structural problems we catch before
# that check. This is Anki's add-note rule, distinct from the semantic-neighbour
# hint upsert_notes attaches.
NoteValidationReason = Literal[
    "duplicate",
    "empty",
    "missing_cloze",
    "notetype_not_cloze",
    "field_not_cloze",
    "unknown_note_type",
    "unknown_field",
]


class UpsertNoteOk(BaseModel):
    status: Literal["created", "updated"]
    id: int
    neighbors: list[Neighbor] = []
    neighbors_unavailable: bool = False


class UpsertNoteValidated(BaseModel):
    # A dry-run outcome: the note passed validation and *would* be written, but
    # nothing was. `action` is what a real run would have done. Distinct status
    # ("ok") so it can never be mistaken for an actual write.
    status: Literal["ok"]
    index: int
    action: Literal["create", "update"]


class UpsertNoteSkipped(BaseModel):
    # on_duplicate="skip": an exact first-field duplicate, left unwritten.
    status: Literal["skipped"]
    index: int
    reason: Literal["duplicate"]


class UpsertNoteError(BaseModel):
    status: Literal["error"]
    index: int
    error: str
    # Set when the failure is a structured validation result (Anki's
    # fields_check, or an unknown note type / field); None for ad-hoc errors
    # (note not found, deck missing, unexpected exception). Genuine independent
    # optionality — many error paths carry no machine-readable reason.
    reason: NoteValidationReason | None = None


UpsertNoteResult = Annotated[
    UpsertNoteOk | UpsertNoteValidated | UpsertNoteSkipped | UpsertNoteError,
    Field(discriminator="status"),
]


class NoteTypeOk(BaseModel):
    status: Literal["created", "updated"]
    id: int
    name: str


class NoteTypeError(BaseModel):
    status: Literal["error"]
    index: int
    error: str


NoteTypeResult = Annotated[NoteTypeOk | NoteTypeError, Field(discriminator="status")]


# -- explicit note-type field operations (update_note_type_fields) -----------
# Identity-based field edits that the position-keyed `upsert_note_types` replace
# can't express safely: a true move, a non-trailing remove, an insert at a
# position. Each is a discriminated variant on `op`; applied in order.


class FieldAdd(BaseModel):
    op: Literal["add"]
    name: str = Field(description="Name for the new field.")
    position: int | None = Field(
        default=None,
        ge=0,
        description="0-based position to insert at. Appended to the end if omitted.",
    )


class FieldRemove(BaseModel):
    op: Literal["remove"]
    name: str = Field(
        description="Field to remove. Its data is dropped from every note of this type."
    )


class FieldRename(BaseModel):
    op: Literal["rename"]
    name: str = Field(description="Current name of the field to rename.")
    new_name: str = Field(description="New name. Field data is preserved.")


class FieldReposition(BaseModel):
    op: Literal["reposition"]
    name: str = Field(description="Field to move. Its data moves with it.")
    position: int = Field(ge=0, description="New 0-based position for the field.")


FieldOp = Annotated[
    FieldAdd | FieldRemove | FieldRename | FieldReposition,
    Field(discriminator="op"),
]


class FieldMetadataInput(BaseModel):
    """A per-field editor-metadata update (#119). Only the set attrs change."""

    name: str = Field(description="Name of the field to update.")
    font: str | None = Field(default=None, description="Edit-time font family.")
    size: int | None = Field(default=None, ge=1, description="Edit-time font size (px).")
    description: str | None = Field(
        default=None, description="Hint text shown for the field in the note editor."
    )


class UpdateNoteTypeFieldsResponse(BaseModel):
    id: int
    name: str
    fields: list[str]  # the resulting ordered field names


class UpdateNoteTypeFieldMetadataResponse(BaseModel):
    id: int
    name: str
    fields_updated: list[str]  # names of the fields whose metadata changed


# -- explicit note-type template operations (update_note_type_templates) ------
# The template counterpart of the field ops: identity-based card-template edits
# the positional `upsert_note_types` replace can't express safely (move, insert,
# non-trailing remove). Same discriminator pattern.


class TemplateOpAdd(BaseModel):
    op: Literal["add"]
    name: str = Field(description="Name for the new card template.")
    front: str = Field(description="Front-side HTML. Use {{FieldName}} to insert field values.")
    back: str = Field(
        description="Back-side HTML. Use {{FieldName}} and {{FrontSide}} for the rendered front."
    )
    position: int | None = Field(
        default=None,
        ge=0,
        description="0-based position to insert at. Appended to the end if omitted.",
    )


class TemplateOpRemove(BaseModel):
    op: Literal["remove"]
    name: str = Field(
        description="Template to remove. Its cards are deleted from every note of this type."
    )


class TemplateOpRename(BaseModel):
    op: Literal["rename"]
    name: str = Field(description="Current name of the template to rename.")
    new_name: str = Field(description="New name. A label change only — cards are untouched.")


class TemplateOpReposition(BaseModel):
    op: Literal["reposition"]
    name: str = Field(description="Template to move. Its cards move with it.")
    position: int = Field(ge=0, description="New 0-based position for the template.")


TemplateOp = Annotated[
    TemplateOpAdd | TemplateOpRemove | TemplateOpRename | TemplateOpReposition,
    Field(discriminator="op"),
]


class UpdateNoteTypeTemplatesResponse(BaseModel):
    id: int
    name: str
    templates: list[str]  # the resulting ordered template names


class FindReplaceNoteTypesResponse(BaseModel):
    # Find/replace inside one note type's card-template HTML and shared CSS (no
    # note field values are touched). Flat, not a union: every field is always
    # present — `replacements` is 0 and the lists empty on a no-op.
    id: int
    name: str
    replacements: int  # total substitutions made across selected templates + CSS
    templates_changed: list[str]  # names of templates whose front/back changed
    css_changed: bool


class DeckInput(BaseModel):
    # Upsert input mirroring NoteInput: ``name`` is the desired deck name; an
    # optional ``id`` selects an existing deck to rename to that name. Absent id =
    # create (or no-op if the name already exists).
    id: int | None = Field(
        default=None,
        description="Deck ID. Present = rename this deck to `name`, absent = create `name`.",
    )
    name: str = Field(description='Deck name (e.g., "Japanese::Vocabulary"); "::" denotes nesting.')


# -- media (#70) -------------------------------------------------------------
# One store item carries exactly one source: base64 `data` (which needs an
# explicit `filename` with an extension — the file's bytes don't tell us what it
# is) or a `url` the server fetches (name/extension derived from the URL/Content-
# Type). The model_validator makes the illegal "both/neither source" and
# "data without filename" states a request rejection (structural), the same way
# an out-of-range bound is; *execution* failures (bad base64, unfetchable URL)
# are per-item results, so one bad item doesn't sink the batch.


class StoreMediaItem(BaseModel):
    filename: str | None = Field(
        default=None,
        description="Desired filename with extension (e.g. 'cell.png'). Required "
        "with `data`; optional with `url` (derived from the URL if omitted). Anki "
        "resolves collisions, so the stored name may differ.",
    )
    data: str | None = Field(
        default=None, description="Base64-encoded file bytes. Mutually exclusive with `url`."
    )
    url: str | None = Field(
        default=None,
        description="http(s) URL the server fetches and stores. Mutually exclusive with `data`.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> StoreMediaItem:
        sources = [s for s in (self.data, self.url) if s is not None]
        if len(sources) != 1:
            raise ValueError("provide exactly one of `data` or `url`")
        if self.data is not None and not self.filename:
            raise ValueError("`filename` (with an extension) is required when `data` is given")
        return self


class UpsertDeckOk(BaseModel):
    status: Literal["created", "updated"]
    id: int
    name: str


class UpsertDeckError(BaseModel):
    status: Literal["error"]
    index: int
    name: str | None = None
    error: str


UpsertDeckResult = Annotated[UpsertDeckOk | UpsertDeckError, Field(discriminator="status")]


class NoteTypeDeleted(BaseModel):
    status: Literal["deleted"]
    id: int
    name: str


class NoteTypeNotFound(BaseModel):
    status: Literal["not_found"]
    id: int


class NoteTypeDeleteError(BaseModel):
    status: Literal["error"]
    id: int
    name: str
    error: str


DeleteNoteTypeResult = Annotated[
    NoteTypeDeleted | NoteTypeNotFound | NoteTypeDeleteError,
    Field(discriminator="status"),
]


# ============================================================================
# Tool response models
#
# No ``error`` field: a whole-call failure (bad input, unhandled exception) is
# an MCP ``isError`` result, which the client raises on. ``message`` is a
# genuine optional advisory (e.g. index-building notice, neighbor-retry hint).
# ============================================================================


class CollectionInfo(BaseModel):
    # Each section is an independent, caller-selected slice: you get exactly the
    # sections named in `include`. None = "not requested", so the optionality is
    # real and uncorrelated (any subset is valid).
    summary: Summary | None = None
    note_types: list[NoteTypeInfo] | None = None
    decks: list[DeckInfo] | None = None
    tags: list[str] | None = None
    stats: Stats | None = None


class ListNotesResponse(BaseModel):
    notes: list[Note] = []
    total: int = 0
    limit: int = 50


class MigrateNoteTypeResponse(BaseModel):
    # Result of changing a set of notes from one note type to another (#75).
    # The two list fields surface the data-affecting parts of the migration so
    # the caller sees exactly what was lost / left empty.
    changed: list[int] = []  # note ids migrated (or that would be, on dry-run)
    from_note_type: str
    to_note_type: str
    dropped_fields: list[str] = []  # source fields with no mapping — content lost
    new_empty_fields: list[str] = []  # target fields nothing mapped into
    dry_run: bool = False


class SearchResultGroup(BaseModel):
    source: str
    matches: list[SearchMatch] = []


class SearchResponse(BaseModel):
    results: list[SearchResultGroup] = []
    message: str | None = None


class UpsertNotesResponse(BaseModel):
    results: list[UpsertNoteResult] = []
    # Echoes the request: when true, nothing was written and each result is a
    # validation outcome (`ok`/`skipped`/`error`), never `created`/`updated`.
    dry_run: bool = False
    message: str | None = None


class UpsertNoteTypesResponse(BaseModel):
    results: list[NoteTypeResult] = []


class DeleteNotesResponse(BaseModel):
    deleted: list[int] = []
    not_found: list[int] = []


class DeleteNoteTypesResponse(BaseModel):
    results: list[DeleteNoteTypeResult] = []


class UpdateNoteTagsResponse(BaseModel):
    notes_modified: int = 0
    not_found: list[int] = []
    message: str | None = None


class RenameTagResponse(BaseModel):
    notes_modified: int = 0


# -- collection_prune (#89) --------------------------------------------------
# One maintenance tool runs several cleanups. Each cleanup's result is its own
# nested sub-model, present only when that cleanup was requested. The `| None`
# here is genuine independent optionality ("this cleanup wasn't run"), not a
# correlated hidden state, so it's a plain optional rather than a union — counts
# (not unions) describe a single cleanup's outcome.


class PruneUnusedTags(BaseModel):
    removed: int = 0  # unused tag-registry names removed (or that would be, on dry-run)
    tags: list[str] = []  # the names


class PruneEmptyNotes(BaseModel):
    removed: list[int] = []  # note ids removed (or that would be)


class PruneEmptyCards(BaseModel):
    cards_removed: int = 0  # empty cards removed (or that would be)
    notes_deleted: list[int] = []  # notes that lost their last card and were deleted


class PruneUnusedMedia(BaseModel):
    removed: int = 0  # media files trashed (or that would be, on dry-run)
    files: list[str] = []  # the filenames


class CollectionPruneResponse(BaseModel):
    dry_run: bool = True
    unused_tags: PruneUnusedTags | None = None  # None = cleanup not requested
    empty_notes: PruneEmptyNotes | None = None
    empty_cards: PruneEmptyCards | None = None
    unused_media: PruneUnusedMedia | None = None


class UpsertDecksResponse(BaseModel):
    results: list[UpsertDeckResult] = []


class DeleteDecksResponse(BaseModel):
    deleted: list[str] = []
    not_found: list[str] = []
    not_empty: list[str] = []


# -- media responses (#70) ---------------------------------------------------


class StoreMediaOk(BaseModel):
    status: Literal["stored"]
    index: int  # position in the request batch, so callers can correlate
    filename: str  # final stored name (Anki may rename on a content collision)
    mime: str | None = None  # mimetypes.guess_type — None for an unknown extension
    size_bytes: int
    deduped: bool = False  # Anki kept the requested name (identical content already present)


class StoreMediaError(BaseModel):
    status: Literal["error"]
    index: int
    filename: str | None = None
    error: str


StoreMediaResult = Annotated[StoreMediaOk | StoreMediaError, Field(discriminator="status")]


class StoreMediaResponse(BaseModel):
    results: list[StoreMediaResult] = []


# fetch: per item the file is either ``found`` or ``missing``. fetch never returns
# bytes — base64 in a tool response is useless to a model (it can't render it) and
# wrecks context — so a found file carries only where to get the bytes: ``url``
# (the server's GET /media/<name>, the retrieval path for any caller) and ``path``
# (read directly if co-located). ``url`` is None only when the server didn't
# advertise a base URL (direct library use without a running HTTP server).
class MediaFile(BaseModel):
    status: Literal["found"]
    filename: str
    url: str | None = None  # GET this for the bytes (no base64)
    path: str  # absolute server-side path in the media dir
    mime: str | None = None
    size_bytes: int


class MediaMissing(BaseModel):
    status: Literal["missing"]
    filename: str


MediaFetchResult = Annotated[MediaFile | MediaMissing, Field(discriminator="status")]


class FetchMediaResponse(BaseModel):
    results: list[MediaFetchResult] = []


class MediaFileInfo(BaseModel):
    filename: str
    url: str | None = None  # GET /media/<name> on the server (None for direct lib use)
    mime: str | None = None
    size_bytes: int


class ListMediaResponse(BaseModel):
    media_dir: str
    count: int = 0
    files: list[MediaFileInfo] = []


class DeleteMediaResponse(BaseModel):
    deleted: list[str] = []
    not_found: list[str] = []


class CollectionCheckResponse(BaseModel):
    """Read-only collection diagnostics (the sibling of collection_prune).

    Wraps Anki's ``col.media.check()``: ``unused`` are media files on disk no note
    references, ``missing`` are filenames referenced by notes but absent from the
    media dir, ``missing_media_notes`` are the note ids with such references, and
    ``have_trash`` reports whether Anki's media trash holds anything.
    """

    media_dir: str
    unused: list[str] = []
    missing: list[str] = []
    missing_media_notes: list[int] = []
    have_trash: bool = False


class FindReplaceSample(BaseModel):
    id: int
    field: str
    before: str
    after: str


class FindReplaceResponse(BaseModel):
    notes_changed: int = 0
    dry_run: bool = False
    samples: list[FindReplaceSample] = []


# ============================================================================
# Server status / custom-endpoint models (not tool returns; client-side use)
# ============================================================================


class EmbeddingRunning(BaseModel):
    """A live embedding service. ``url``/``model`` are absent only if the
    /health probe failed while the process was up (then ``available`` is False);
    that probe-dependence is genuine independent optionality."""

    state: Literal["running"] = "running"
    available: bool = False
    pid: int | None = None
    url: str | None = None
    model: str | None = None


class EmbeddingDown(BaseModel):
    """No live service — never started, stopped, or failed to start. Carries no
    pid/url/model, so the down states can't masquerade as running."""

    state: Literal["stopped", "failed", "not_configured"]
    available: Literal[False] = False


EmbeddingStatus = Annotated[EmbeddingRunning | EmbeddingDown, Field(discriminator="state")]


class IndexProgress(BaseModel):
    indexed: int = 0
    total: int = 0


class _IndexBase(BaseModel):
    """On-disk index contents, shared across build states.

    ``ndim``/``path``/``col_mod``/``model_id`` are independently optional —
    absent until the index has vectors / has been built — regardless of state.
    """

    available: bool = False
    size: int = 0
    ndim: int | None = None
    path: str | None = None
    col_mod: int | None = None
    model_id: str | None = None


class IndexUnavailable(_IndexBase):
    state: Literal["unavailable"] = "unavailable"


class IndexBuilding(_IndexBase):
    state: Literal["building"]
    progress: IndexProgress


class IndexReady(_IndexBase):
    state: Literal["ready"]


class IndexErrored(_IndexBase):
    state: Literal["error"]
    error: str


# Discriminated on build state: `progress` lives only on building, `error` only
# on the errored variant — no "maybe-progress, maybe-error" bag.
IndexStatus = Annotated[
    IndexUnavailable | IndexBuilding | IndexReady | IndexErrored,
    Field(discriminator="state"),
]


class ServerStatus(BaseModel):
    """A responding server's self-report from ``GET /status``.

    This models exactly one thing — what a live, responsive server reports — so
    its fields are required (a server that answers always knows its pid, url,
    collection, and embedding/index state). The "not running" and "running but
    unresponsive" cases are *connection* states the client/CLI determine from
    the daemon lock, not server payloads, so they're handled there rather than
    smuggled in here as optionals. ``uptime`` is best-effort (omitted if the
    start time can't be parsed); ``log`` is filled in by the CLI.
    """

    running: Literal[True] = True
    pid: int
    url: str
    collection: str
    log_level: str
    log_dir: str
    uptime: str | None = None
    log: str | None = None
    embedding: EmbeddingStatus
    index: IndexStatus
    # Collection-lock state (#64): the locking mode and whether the collection is
    # currently held open. In the default permanent-hold mode it's always held;
    # in cooperative mode it's released when idle. Defaulted so older payloads
    # (and the permanent-mode common case) validate.
    locking: Literal["permanent", "cooperative"] = "permanent"
    collection_held: bool = True


# -- custom-endpoint responses (discriminated on `status`) -------------------
# Each route returns one well-defined shape per status, so the fields that
# accompany a status are required on that variant rather than optional on a
# shared bag. (HTTP error responses surface as ServerHTTPError, not as a variant.)


class IndexRebuildStarted(BaseModel):
    status: Literal["started"]
    total: int


class IndexRebuildComplete(BaseModel):
    status: Literal["complete"]
    size: int


class IndexRebuildAlreadyBuilding(BaseModel):
    status: Literal["already_building"]
    progress: IndexProgress


IndexRebuildResponse = Annotated[
    IndexRebuildStarted | IndexRebuildComplete | IndexRebuildAlreadyBuilding,
    Field(discriminator="status"),
]


class IndexSaved(BaseModel):
    status: Literal["saved"]
    size: int
    pending: int  # incremental changes flushed by this save (0 if already current)


class IndexSaveEmpty(BaseModel):
    status: Literal["empty"]  # no index built yet — nothing to persist


class IndexSaveBuilding(BaseModel):
    status: Literal["building"]  # a rebuild is in progress; refused to avoid a partial save
    progress: IndexProgress


IndexSaveResponse = Annotated[
    IndexSaved | IndexSaveEmpty | IndexSaveBuilding,
    Field(discriminator="status"),
]


class EmbeddingStarted(BaseModel):
    status: Literal["started"]
    embedding: EmbeddingStatus
    index: IndexStatus


class EmbeddingAlreadyRunning(BaseModel):
    status: Literal["already_running"]
    embedding: EmbeddingStatus


EmbeddingStartResponse = Annotated[
    EmbeddingStarted | EmbeddingAlreadyRunning,
    Field(discriminator="status"),
]


class EmbeddingStopped(BaseModel):
    status: Literal["stopped"]
    index: IndexStatus


class EmbeddingNotRunning(BaseModel):
    status: Literal["not_running"]


EmbeddingStopResponse = Annotated[
    EmbeddingStopped | EmbeddingNotRunning,
    Field(discriminator="status"),
]


class ShutdownResponse(BaseModel):
    status: str
    pid: int


class ReloadResponse(BaseModel):
    # POST /reload — closed and re-opened the collection (#79).
    status: Literal["reloaded"] = "reloaded"
    col_mod: int  # collection mod stamp after re-opening
    rebuilding: bool = False  # whether the re-open drift check started an index rebuild


class StopSucceeded(BaseModel):
    """The daemon was stopped. ``pid`` is None only if the pid file was
    unreadable; ``forced`` is True if a graceful stop timed out into a kill."""

    stopped: Literal[True] = True
    pid: int | None = None
    forced: bool = False


class StopFailed(BaseModel):
    """Nothing to stop — the daemon wasn't running."""

    stopped: Literal[False] = False
    reason: str


# Discriminated on the bool ``stopped``: a success carries pid/forced, a no-op
# carries the reason. (daemon.stop_server is the source.)
StopResponse = Annotated[StopSucceeded | StopFailed, Field(discriminator="stopped")]
