"""Pydantic models for every Shrike tool request/response and server-status shape.

This module is the single source of truth for the wire contract. The server
tools return the response models here (so FastMCP emits an ``outputSchema`` for
each tool), the standalone client validates responses into them, the CLI renders
them, and ``scripts/gen_schema.py`` generates ``docs/mcp-schema.json`` from them.

Design rules:

- **Every tool response model has all fields defaulted**, so a bare
  ``Model(error=...)`` is valid and FastMCP can coerce the ``_safe_tool``
  catch-all dict (``{"error": ...}``) into the declared return type.
- Models tolerate unknown keys (Pydantic's default ``extra="ignore"``) so a
  newer server adding a field doesn't break an older client.
- No imports from the rest of ``shrike`` — keep this leaf-level to avoid cycles.
"""

from __future__ import annotations

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


class NoteTypeInfo(BaseModel):
    name: str
    id: int
    fields: list[str] = []
    type: str = "standard"
    templates: list[TemplateInfo] | None = None
    css: str | None = None


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
# Tool response models
# ============================================================================


class CollectionInfo(BaseModel):
    summary: Summary | None = None
    note_types: list[NoteTypeInfo] | None = None
    decks: list[DeckInfo] | None = None
    tags: list[str] | None = None
    stats: Stats | None = None
    error: str | None = None


class ListNotesResponse(BaseModel):
    notes: list[Note] = []
    total: int = 0
    limit: int = 0
    error: str | None = None


class SearchResultGroup(BaseModel):
    source: str
    matches: list[SearchMatch] = []


class SearchResponse(BaseModel):
    results: list[SearchResultGroup] = []
    message: str | None = None
    error: str | None = None


class UpsertNoteResult(BaseModel):
    status: str
    id: int | None = None
    index: int | None = None
    error: str | None = None
    neighbors: list[Neighbor] | None = None
    neighbors_unavailable: bool | None = None


class UpsertNotesResponse(BaseModel):
    results: list[UpsertNoteResult] = []
    message: str | None = None
    error: str | None = None


class NoteTypeResult(BaseModel):
    status: str
    id: int | None = None
    name: str | None = None
    index: int | None = None
    error: str | None = None


class UpsertNoteTypesResponse(BaseModel):
    results: list[NoteTypeResult] = []
    error: str | None = None


class DeleteNotesResponse(BaseModel):
    deleted: list[int] = []
    not_found: list[int] = []
    error: str | None = None


class DeleteNoteTypeResult(BaseModel):
    id: int
    name: str | None = None
    status: str
    error: str | None = None


class DeleteNoteTypesResponse(BaseModel):
    results: list[DeleteNoteTypeResult] = []
    error: str | None = None


# ============================================================================
# Server status / custom-endpoint models (not tool returns; client-side use)
# ============================================================================


class EmbeddingStatus(BaseModel):
    available: bool = False
    state: str | None = None
    pid: int | None = None
    url: str | None = None
    model: str | None = None


class IndexProgress(BaseModel):
    indexed: int = 0
    total: int = 0


class IndexStatus(BaseModel):
    state: str | None = None
    available: bool = False
    size: int = 0
    ndim: int | None = None
    path: str | None = None
    col_mod: int | None = None
    model_id: str | None = None
    progress: IndexProgress | None = None
    error: str | None = None


class ServerStatus(BaseModel):
    """The /status response, and the degraded shape daemon.server_status() yields.

    ``embedding`` and ``index`` are absent when synthesized from local daemon
    state (server alive but not yet responsive); ``responsive`` and ``log`` are
    filled in by the CLI rather than the server.
    """

    running: bool = False
    responsive: bool | None = None
    pid: int | None = None
    url: str | None = None
    collection: str | None = None
    log_level: str | None = None
    log_dir: str | None = None
    log: str | None = None
    started: str | None = None
    uptime: str | None = None
    embedding: EmbeddingStatus | None = None
    index: IndexStatus | None = None


class IndexRebuildResponse(BaseModel):
    status: str
    total: int | None = None
    size: int | None = None
    progress: IndexProgress | None = None
    error: str | None = None


class EmbeddingStartResponse(BaseModel):
    status: str
    embedding: EmbeddingStatus | None = None
    index: IndexStatus | None = None
    error: str | None = None


class EmbeddingStopResponse(BaseModel):
    status: str
    index: IndexStatus | None = None


class ShutdownResponse(BaseModel):
    status: str
    pid: int | None = None


class StopResponse(BaseModel):
    """Result of stopping the local daemon (daemon.stop_server)."""

    stopped: bool = False
    reason: str | None = None
    pid: int | None = None
    forced: bool | None = None
