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

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# Stable wire marker prefixing a tool's MCP isError text when the call failed
# because the collection couldn't be acquired (another process holds it, under
# cooperative locking). The single source of truth shared by the server
# (collection.py builds the message) and the dependency-light client (which maps
# it to CollectionBusyError). Not a response model: busy is a transport-level
# error class, orthogonal to every tool's response (the op never ran).
COLLECTION_BUSY_CODE = "collection_busy"

# The action exchange's protocol version — mirrors shrike-schemas'
# WIRE_PROTOCOL_VERSION (the schema contract test pins them equal). The
# exchange evolves additively: a breaking change to an action ships as a NEW
# action/tool name carrying its own types (upsert_notes_v2), so this bumps
# only when the exchange fabric itself breaks (envelope semantics, error
# taxonomy) — the backstop a future remote handshake checks. Reported in
# GET /status.
WIRE_PROTOCOL_VERSION = 1

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
    """Evidence that the query text occurs literally in a note.

    ``matched_fields``/``snippet`` are the field-text hit (the only case today). ``source`` names
    which *derived* text the literal match was found in — ``field`` today (``ocr``/``asr`` are a
    future seam; never VLM image-describe, which is embedding-only). For a non-field source ``ref``
    pins the single artifact that matched (a media filename); for the ``field`` source it stays
    ``None`` and ``matched_fields`` enumerates the fields instead. The ``source``/``ref`` seam lets
    a result say *where* an image/audio card's text matched.
    """

    matched_fields: list[str] = []
    snippet: str | None = None
    source: str = "field"
    ref: str | None = None


class FuzzyMatch(BaseModel):
    """Evidence that the query *approximately* matched a note's derived text.

    A trigram/typo-tolerant hit from the derived-text store (the ``fuzzy`` retrieval signal), for
    near-misses an exact substring search would miss (``protien`` → ``protein``). ``source`` is
    which derived text matched (``field`` today; ``ocr``/``asr`` are a future seam), ``ref`` the
    field name or media filename it hit, and ``snippet`` a window around the match — so an LLM/MCP
    client can see what an image/audio card actually is from the text that matched it.
    """

    source: str
    ref: str
    snippet: str | None = None


class SignalContribution(BaseModel):
    """One retrieval signal that contributed to a fused result, and at what rank.

    ``signal`` is the fusion signal's name — ``text`` / ``image`` for the per-modality semantic
    rankers (so the name *is* the matched-modality facet: ``image`` ⇒ "matched on the image"),
    ``exact`` for a literal substring hit, and later ``fuzzy`` / ``tag``. ``rank`` is the note's
    1-based position in that signal's own ranking; the signal's *unweighted* RRF term
    (``1/(k+rank)``) is derivable from it (the per-signal fusion weight is not carried in the
    response, so the full weighted contribution is not).
    """

    signal: str
    rank: int


class SearchMatch(Note):
    """A search result with per-mechanism match evidence.

    A hit can be semantically ranked, an exact-substring hit, or both. Each
    annotation below is independently optional and absent when its mechanism did
    not contribute — but a returned match always carries at least one. This is the
    extension point for future retrieval backends (n-gram / fuzzy / prefix):
    they add further optional evidence fields here, never a new param or tool.
    """

    # Semantic similarity (0-1); None when the hit matched only by exact substring
    # or the vector index was unavailable. Independent of `substring`.
    score: float | None = None
    # Present when the query text occurs literally in the note.
    substring: SubstringInfo | None = None
    # Present when the query trigram/typo-matched the note's derived text (the `fuzzy` signal).
    # Independent of `score`/`substring` — a hit can be any combination. Carries the source-aware
    # window (where in which derived text it matched).
    fuzzy: FuzzyMatch | None = None
    # Which signals surfaced this result, best (lowest) rank first. Always non-empty for a
    # returned match (it came from a fused hit). The unified provenance view over the fused ranking;
    # `score`/`substring` above stay as the per-signal detail. `signal: "image"` is the
    # matched-modality facet.
    provenance: list[SignalContribution] = []


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
    """A per-field editor-metadata update. Only the set attrs change."""

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


# -- media -------------------------------------------------------------------
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
        "with `data`; optional with `url`/`path` (derived from them if omitted). "
        "Anki resolves collisions, so the stored name may differ.",
    )
    data: str | None = Field(
        default=None,
        description="Base64-encoded file bytes. Exactly one of data/url/path.",
    )
    url: str | None = Field(
        default=None,
        description="http(s) URL the server fetches and stores. Exactly one of data/url/path.",
    )
    path: str | None = Field(
        default=None,
        description="Path to a file on the **server's** filesystem to store directly "
        "(zero-copy). Off by default — honored only when the server set --media-path-root "
        "(on a purely-local daemon) and the file is under one of those roots; rejected "
        "otherwise. Exactly one of data/url/path.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> StoreMediaItem:
        sources = [s for s in (self.data, self.url, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("provide exactly one of `data`, `url`, or `path`")
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
# genuine optional advisory (e.g. an index-building notice).
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
    # Result of changing a set of notes from one note type to another.
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
    # The two-tier live-search contract: "partial" means the
    # embedding-bearing signals (semantic + tag) were skipped because the
    # caller asked for the live tier — re-request with tier="full" to upgrade.
    # A response that is the final answer for its query/server state (full
    # tier, or nothing more would run) is "full".
    completeness: Literal["partial", "full"] = "full"
    # Echo of the request's `version` (client-side stale-response dropping:
    # the server is stateless per request; the client owns cancellation).
    version: int | None = None
    # Freshness advisory: True when the kernel was NOT settled as this search
    # ran — a write was still draining through the embed queue or a rebuild was
    # in flight, so the index/derived store may lag the collection and a
    # just-written note can be missing. The collection is always current, so an
    # exact/lexical hit is unaffected; only the semantic ranking can be stale.
    # Default False = settled. A standalone bool, not paired with the human
    # advisory in `message` (that slot is already the home for the prose
    # explanation) — keeping the machine-checkable flag independent of any text.
    # The caller decides: serve the result, or settle()/retry for a fresh one.
    stale: bool = False


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


# -- collection_prune --------------------------------------------------------
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
    dry_run: bool = False
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


# -- media responses ---------------------------------------------------------


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


class ImportPackageResponse(BaseModel):
    """The result of importing an `.apkg`/`.colpkg` — per-bucket note
    counts from anki's importer, mirroring `ImportResponse.Log`.

    `new` notes were added; `updated` were same-GUID notes the importer
    refreshed (governed by the `update_notes` condition); `duplicate` matched an
    existing note and were skipped; `conflicting` couldn't be merged;
    `first_field_match` matched an existing note by first field; `missing_notetype`
    / `missing_deck` couldn't resolve their notetype/deck; `empty_first_field`
    were rejected as empty. `found_notes` is the total the package contained.
    Counts, not note-id lists. `reindexed` reports whether the import triggered an
    index reconcile (False when no embedding is configured)."""

    new: int = 0
    updated: int = 0
    duplicate: int = 0
    conflicting: int = 0
    first_field_match: int = 0
    missing_notetype: int = 0
    missing_deck: int = 0
    empty_first_field: int = 0
    found_notes: int = 0
    reindexed: bool = False


# ============================================================================
# Server status / custom-endpoint models (not tool returns; client-side use)
# ============================================================================


class EmbeddingRunning(BaseModel):
    """A live embedding service. ``available`` is False when the /health
    probe fails while the process is up (the full fields ride along either
    way); ``url``/``model`` stay optional for older wire shapes and backends
    without them."""

    state: Literal["running"] = "running"
    available: bool = False
    pid: int | None = None
    url: str | None = None
    model: str | None = None
    # The effective execution provider (onnx backend; the accelerator that actually loaded,
    # so a silent CPU fallback is visible). Absent for llama, which has no provider concept.
    provider: str | None = None
    # Whether the startup batch-safety probe found the model batches deterministically.
    # ``batch`` is "batched" or "serial"; both absent for a backend that doesn't report them.
    batch_safe: bool | None = None
    batch: Literal["serial", "batched"] | None = None
    # The modalities this space embeds — what the running backend advertises
    # (text, or text+image for CLIP). Optional for older wire shapes.
    modalities: list[str] | None = None


class EmbeddingDown(BaseModel):
    """No live service — never started, stopped, or failed to start. Carries no
    pid/url/model, so the down states can't masquerade as running."""

    state: Literal["stopped", "failed", "not_configured"]
    available: Literal[False] = False


EmbeddingStatus = Annotated[EmbeddingRunning | EmbeddingDown, Field(discriminator="state")]


class IndexProgress(BaseModel):
    indexed: int = 0
    total: int = 0


class IndexModalityStat(BaseModel):
    """One per-modality sub-index's size/ndim.

    The aggregate ``size``/``ndim`` on the index collapse the per-modality
    sub-indexes (sum / the text modality's width), so a two-space (text+image)
    collection can't surface that its image sub-index is 512-dim while text is
    768-dim. This row carries each sub-index's own figures. ``ndim`` is ``None``
    for a sub-index with no vectors yet (width not set)."""

    modality: str
    size: int = 0
    ndim: int | None = None


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
    # Per-(non-text-)modality activation-gate calibration {modality: {n, mean, std}}; absent
    # until a multimodal index is calibrated (text-only / uncalibrated indexes have none).
    activation: dict[str, dict[str, float]] | None = None
    # Per-modality sub-index breakdown: each sub-index's own size/ndim,
    # which the aggregate size/ndim above can't express (text 768-dim, image
    # 512-dim under CLIP). Text-first. Empty list on older payloads / a server
    # that doesn't report it.
    modalities: list[IndexModalityStat] = []


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


class DerivedStatus(BaseModel):
    """The derived-text store's self-report — the FTS5 trigram sidecar.

    Unlike the vector index this is a flat shape, not a discriminated union: the store has no
    per-state-only fields (no build progress to report, no persisted error variant — a failed build
    drops back to ``unavailable`` and lookups fall back), so a single model with a ``state`` tag is
    honest. ``fts5`` is False when the runtime's SQLite lacks FTS5/trigram (the store is inert and
    search falls back to the linear scan); ``available`` adds "a build has run" on top.
    """

    state: Literal["unavailable", "building", "ready", "error"] = "unavailable"
    available: bool = False
    fts5: bool = False
    size: int = 0
    path: str | None = None
    col_mod: int | None = None


class RecognitionEngineStatus(BaseModel):
    """One attached recognition engine's self-report.

    ``state`` is ``ready`` (attached, sweeping/idle) or ``error`` (the engine's
    dependency is missing or a sweep failed) — the same lifecycle enum the
    index/derived stores use. ``backend`` is the construction kind (``apple``,
    ``describe-remote``); ``fingerprint`` is the engine identity when known.
    """

    state: Literal["unavailable", "building", "ready", "error"] = "ready"
    backend: str | None = None
    fingerprint: str | None = None


class CoverageCell(StrEnum):
    """How one (query-modality → target-modality) pair is reachable.

    The honest cross-modal contract a caller needs to know what a query can
    actually retrieve:

    - ``native``: a single live embedding space embeds BOTH the query and the
      target modality, so a query of the one retrieves content of the other in
      that shared space (text→text whenever a text space is up; text→image when
      a CLIP/omni space serves both).
    - ``via_derived_text``: the target isn't natively embeddable from the query,
      but a recognizer derives TEXT from it into the text space, so the (text)
      query reaches it there — text→image when an ``ocr`` or ``describe`` engine
      is attached (OCR text / VLM prose lands in the text vector space),
      text→audio when an ``asr`` engine is attached. Weaker than ``native``: it
      searches the derived text, not the media's own content.
    - ``unavailable``: neither — the target can't be reached from this query.
    """

    NATIVE = "native"
    VIA_DERIVED_TEXT = "via_derived_text"
    UNAVAILABLE = "unavailable"


class CoverageRow(BaseModel):
    """One query modality's reachability to each target modality.

    Every cell is a ``CoverageCell`` — there is no "absent" target, only an
    ``unavailable`` one, so the shape is stable for clients regardless of which
    spaces/recognizers came up."""

    text: CoverageCell = CoverageCell.UNAVAILABLE
    image: CoverageCell = CoverageCell.UNAVAILABLE
    audio: CoverageCell = CoverageCell.UNAVAILABLE


class CoverageMatrix(BaseModel):
    """The cross-modal coverage matrix: for each query modality, a
    ``CoverageRow`` naming how each target modality is reachable.

    Derived from the live embedding spaces (the ``native`` cells) plus the
    attached, ready recognizers (the ``via_derived_text`` cells). A typed
    matrix rather than a flat ``{modality: bool}`` so the surface tells a caller
    e.g. that text→audio is reachable only via ASR-derived text, not natively.
    With embedding down every cell is ``unavailable``; with one text space up,
    text→text is ``native`` and the rest follow from the recognizers."""

    text: CoverageRow = CoverageRow()
    image: CoverageRow = CoverageRow()
    audio: CoverageRow = CoverageRow()


class DedupStats(BaseModel):
    """Rolling dedup best-match statistics: one sample per search query group —
    the best SEMANTIC cosine among its matches, or a `no_match` tick when none
    ranked. The calibration feedstock for the dedup threshold, deliberately
    separate from the search-gate calibration (different population).
    `buckets[i]` counts best-scores in [i/20, (i+1)/20); 20 buckets over [0, 1].
    """

    samples: int = 0
    no_match: int = 0
    buckets: list[int] = []


class CollectionStatus(BaseModel):
    """One collection's state in a multi-collection daemon's ``/status``.

    A row per known collection: the daemon's boot/default collection plus every
    registered profile. ``name`` is the routing handle (the registry profile
    name, or a sentinel for an unregistered boot collection); ``is_default``
    marks the active default the operational routes act on and a bare call
    resolves to. ``registered`` is whether it's in the profile registry;
    ``active`` is whether a harness is currently assembled for it (lazily, on
    first route — so a registered-but-never-routed collection is
    ``active=False``). ``held`` is whether it currently holds its collection
    lock (cooperative mode releases when idle; ``None`` when not assembled).
    ``index_state``/``col_mod`` describe its namespaced index (``None`` when not
    assembled — nothing has been opened to read them from)."""

    name: str
    path: str
    registered: bool
    is_default: bool = False
    active: bool = False
    held: bool | None = None
    index_state: str | None = None
    col_mod: int | None = None


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
    # The action exchange's protocol version — a future remote client
    # checks this before speaking. ``ge=0`` mirrors the Rust ``u32``.
    wire_protocol_version: int = Field(ge=0)
    pid: int
    url: str
    collection: str
    log_level: str
    log_dir: str
    uptime: str | None = None
    log: str | None = None
    # The PRIMARY embedding space's health — kept for back-compat (every
    # existing consumer reads ``embedding``). ``embedding_spaces`` below is the
    # full per-space list; this is ``embedding_spaces[0]`` when any space
    # is live.
    embedding: EmbeddingStatus
    # Per-space embedding health: one entry per configured embedder — the
    # primary runtime plus every secondary space. A multi-space profile
    # (e.g. a text space + a text+image CLIP space) reports each here, keyed in
    # the CLI by its modalities. Single-space servers report a one-element list;
    # empty on older payloads (read ``embedding`` then).
    embedding_spaces: list[EmbeddingStatus] = []
    index: IndexStatus
    # Derived-text store: the FTS5 trigram sidecar backing substring/fuzzy lexical search.
    # Defaulted so older payloads (and a build without FTS5 support) validate.
    derived: DerivedStatus = DerivedStatus()
    # Collection-lock state: the locking mode and whether the collection is
    # currently held open. In the default permanent-hold mode it's always held;
    # in cooperative mode it's released when idle. Defaulted so older payloads
    # (and the permanent-mode common case) validate.
    locking: Literal["permanent", "cooperative"] = "permanent"
    collection_held: bool = True
    # Dedup/activation best-match statistics — None until the first search
    # records a sample (and on payloads from older servers). The sampler rides
    # the search path, not the upsert response.
    dedup: DedupStats | None = None
    # Per-engine recognition state: a map keyed by source
    # (``ocr``/``vlm``), each row {state, backend, fingerprint}. An EMPTY map
    # means nothing is attached — distinct from an attached-but-errored engine,
    # which is a present row with state=error. Defaulted (empty) so older
    # payloads validate.
    recognition: dict[str, RecognitionEngineStatus] = {}
    # The cross-modal coverage matrix: for each (query, target)
    # modality pair, how the target is reachable — ``native`` (one space embeds
    # both), ``via_derived_text`` (a recognizer derives text from the target
    # into the text space), or ``unavailable``. Derived from the live spaces and
    # attached recognizers; None on payloads from older servers (which sent the
    # flat ``{modality: bool}`` shape).
    coverage: CoverageMatrix | None = None
    # Multi-collection routing: one row per known collection — the
    # daemon's boot/default collection plus every registered profile — with its
    # held/index/col_mod state. None on a single-collection server / older
    # payloads (the top-level embedding/index/derived/collection_held fields
    # describe the DEFAULT collection, which the operational routes act on; tool
    # calls route per-selector). The top-level fields stay for back-compat.
    collections: list[CollectionStatus] | None = None


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
    # POST /reload — closed and re-opened the collection.
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


# -- the action exchange error envelope --------------------------------------
# The UI edge (POST /actions/{name}) is schema-first WITHOUT the MCP JSON-RPC
# envelope, so a failure needs its own one wire shape. ActionError is that
# shape — defined once in shrike-schemas (canonical) and mirrored here. The
# `code` is a small, stable taxonomy that mirrors the MCP edge's error split
# (a caller mistake vs contention vs an internal bug) onto HTTP status codes;
# the route layer (server.py) maps the actions' transport-neutral errors —
# ToolInputError / CollectionBusyError / NativeBusyError / an unknown name /
# anything else — onto these. `message` is non-leaking: for INTERNAL_ERROR it
# is a fixed generic sentence (the real cause + traceback went to the log via
# _safe_tool), for the caller-actionable codes it carries the actionable text.
#
# Like NoteValidationReason, ActionErrorCode is a *field-level* enum (not a
# standalone catalog entry on either side); its shape is contract-tested
# through ActionError.
class ActionErrorCode(StrEnum):
    INPUT_ERROR = "input_error"
    COLLECTION_BUSY = "collection_busy"
    UNKNOWN_ACTION = "unknown_action"
    INTERNAL_ERROR = "internal_error"


class ActionError(BaseModel):
    code: ActionErrorCode
    message: str


# -- the collection/profile registry enumeration -----------------------------
# `list_profiles` lets an agent discover which collections this server knows
# about (by friendly name) and which one is the active default — the read half
# of the multi-collection surface. Selection-as-routing (a per-call `collection`
# selector) is the routing capstone; this action only enumerates. These
# are host-side config shapes (like ServerStatus), not kernel wire types, so
# they live here and not in shrike-schemas — the kernel never produces them.
class ProfileEntry(BaseModel):
    """One registered collection profile: a friendly ``name`` and its
    collection ``path``. ``is_default`` marks the active default — the profile
    the per-call selector resolves to when no selector is passed.

    Per-profile embedding/cache overrides exist in the registry (config) but
    are deliberately not surfaced here: they're consumed by routing /
    namespacing, not actionable through enumeration."""

    name: str
    path: str
    is_default: bool = False


class ListProfilesResponse(BaseModel):
    """The registry enumeration: the registered profiles and the active-default
    name (``None`` when none is set — e.g. after the default was removed among
    several profiles)."""

    profiles: list[ProfileEntry] = Field(default_factory=list)
    default: str | None = None


# -- export to an Anki package -----------------------------------------------
# `export_package` writes a .apkg/.colpkg. The kernel op returns the
# ExportPackageResult (note_count + the on-disk path it wrote); the host action
# wraps it into ExportPackageResponse, adding `bytes`/`format` and — for the
# default no-output_path case — a download `url` instead of a server-local
# `path`. Both shapes are canonical in shrike-schemas; these mirror them (the
# contract test pins the pair).
class ExportPackageResult(BaseModel):
    """The kernel export-op outcome: notes written + the path the package
    landed at. Internal wire — the host action wraps it for the tool response."""

    note_count: int = Field(ge=0)  # mirrors the Rust ``u32``
    out_path: str


# The tool response is a discriminated union on ``delivery`` (the house style:
# make illegal states unrepresentable — a client never reads a ``path`` that
# isn't there, or a ``url`` that wasn't produced). ``path`` when the operator
# opted into a contained server-local ``output_path``; ``url`` (the default)
# when the server wrote a temp package it serves over HTTP — never base64,
# mirroring fetch_media's "GET the url for the bytes".
class ExportPackagePath(BaseModel):
    delivery: Literal["path"]
    note_count: int = Field(ge=0)  # Rust ``u32``
    bytes: int = Field(ge=0)  # Rust ``u64``
    format: str  # "apkg" | "colpkg"
    path: str


class ExportPackageUrl(BaseModel):
    delivery: Literal["url"]
    note_count: int = Field(ge=0)  # Rust ``u32``
    bytes: int = Field(ge=0)  # Rust ``u64``
    format: str
    url: str


ExportPackageResponse = Annotated[
    ExportPackagePath | ExportPackageUrl, Field(discriminator="delivery")
]
