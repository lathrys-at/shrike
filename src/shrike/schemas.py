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

from pydantic import BaseModel, Field

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


class SearchMatch(Note):
    """A search result: a note plus its similarity score."""

    score: float


class Neighbor(BaseModel):
    """A similar note attached to an upsert result."""

    id: int
    score: float
    tags: list[str] = []


class TemplateInfo(BaseModel):
    name: str
    front: str
    back: str


class NoteTypeDetail(BaseModel):
    """Full template/CSS definition — the two always travel together."""

    templates: list[TemplateInfo]
    css: str


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


class UpsertNoteOk(BaseModel):
    status: Literal["created", "updated"]
    id: int
    neighbors: list[Neighbor] = []
    neighbors_unavailable: bool = False


class UpsertNoteError(BaseModel):
    status: Literal["error"]
    index: int
    error: str


UpsertNoteResult = Annotated[UpsertNoteOk | UpsertNoteError, Field(discriminator="status")]


class NoteTypeOk(BaseModel):
    status: Literal["created", "updated"]
    id: int
    name: str


class NoteTypeError(BaseModel):
    status: Literal["error"]
    index: int
    error: str


NoteTypeResult = Annotated[NoteTypeOk | NoteTypeError, Field(discriminator="status")]


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


class SearchResultGroup(BaseModel):
    source: str
    matches: list[SearchMatch] = []


class SearchResponse(BaseModel):
    results: list[SearchResultGroup] = []
    message: str | None = None


class UpsertNotesResponse(BaseModel):
    results: list[UpsertNoteResult] = []
    message: str | None = None


class UpsertNoteTypesResponse(BaseModel):
    results: list[NoteTypeResult] = []


class DeleteNotesResponse(BaseModel):
    deleted: list[int] = []
    not_found: list[int] = []


class DeleteNoteTypesResponse(BaseModel):
    results: list[DeleteNoteTypeResult] = []


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
