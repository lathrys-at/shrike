//! Rust-canonical wire contracts.
//!
//! Every Shrike tool request/response and server-status shape, as serde types
//! mirroring `src/shrike/schemas.py` — which remains the Python *binding*,
//! kept honest by the CI contract test (Rust-emitted JSON Schema ≡ normalized
//! Pydantic schema, plus instance round-trips through `roundtrip`).
//!
//! Mapping rules (the wire is FastMCP's `model_dump(mode="json")`):
//! - `X | None` → `Option<X>`, serialized as explicit `null` (Pydantic includes
//!   None fields), tolerated absent on input (`#[serde(default)]`).
//! - Discriminated unions → internally-tagged enums (`#[serde(tag = ...)]`).
//!   A Pydantic variant tagged `Literal["created", "updated"]` becomes two
//!   enum variants sharing a payload shape — same wire, same schema semantics.
//! - `Literal[True]`-style fields that are *not* a union's tag use the
//!   [`literals`] types (const-valued, schema `const`).
//! - Unknown keys are ignored (Pydantic's `extra="ignore"`; serde's default).
//! - Pure Rust — NO pyo3; bound to Python in shrike-pyo3.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

pub mod literals;

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::literals::{LiteralFalse, LiteralTrue, ReloadedLiteral};

/// Stable wire marker prefixing a tool's MCP isError text when the call failed
/// because the collection couldn't be acquired. Mirrors
/// `schemas.COLLECTION_BUSY_CODE`.
pub const COLLECTION_BUSY_CODE: &str = "collection_busy";

fn default_limit() -> i64 {
    50
}

fn default_true() -> bool {
    true
}

fn default_source_field() -> String {
    "field".to_owned()
}

fn default_standard() -> String {
    "standard".to_owned()
}

// ============================================================================
// Request models (tool inputs)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A card template in a note-type create/update request.
pub struct TemplateInput {
    /// Template name.
    pub name: String,
    /// Front-side (question) template HTML.
    pub front: String,
    /// Back-side (answer) template HTML.
    pub back: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A note in an `upsert_notes` request (id present = update, absent = create).
pub struct NoteInput {
    #[serde(default)]
    /// Note id; present updates that note, absent creates one.
    pub id: Option<i64>,
    #[serde(default)]
    /// Target deck (create only).
    pub deck: Option<String>,
    #[serde(default)]
    /// Note type name (create only).
    pub note_type: Option<String>,
    #[serde(default)]
    /// Field name → value.
    pub fields: Option<BTreeMap<String, String>>,
    #[serde(default)]
    /// Tags to set on the note.
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A note type in an `upsert_note_types` request (id present = update).
pub struct NoteTypeInput {
    #[serde(default)]
    /// Note-type id; present updates it, absent creates one.
    pub id: Option<i64>,
    #[serde(default)]
    /// Note-type name.
    pub name: Option<String>,
    #[serde(default)]
    /// Field names, in order.
    pub fields: Option<Vec<String>>,
    #[serde(default)]
    /// Card templates.
    pub templates: Option<Vec<TemplateInput>>,
    #[serde(default)]
    /// Shared template CSS.
    pub css: Option<String>,
    #[serde(default)]
    /// Whether this is a cloze note type.
    pub is_cloze: Option<bool>,
}

// ============================================================================
// Shared nested models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A note as returned by the read surfaces (`list_notes`, search).
pub struct Note {
    /// Note id.
    pub id: i64,
    /// Note type name.
    pub note_type: String,
    /// Deck name.
    pub deck: String,
    #[serde(default)]
    /// The note's tags.
    pub tags: Vec<String>,
    /// Last-modified timestamp.
    pub modified: String,
    /// Independent projection: present in "full" mode, omitted in "meta" mode.
    #[serde(default)]
    pub content: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// Exact-substring match evidence on a search result.
pub struct SubstringInfo {
    #[serde(default)]
    /// Field names that contained the substring.
    pub matched_fields: Vec<String>,
    #[serde(default)]
    /// A snippet around the match, if available.
    pub snippet: Option<String>,
    #[serde(default = "default_source_field")]
    /// Where the matched text came from (`field`, later `ocr`/`asr`).
    pub source: String,
    #[serde(default)]
    /// The field name or media filename the text came from.
    pub r#ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// Fuzzy (trigram/typo) match evidence on a search result.
pub struct FuzzyMatch {
    /// Where the matched text came from (`field`, later `ocr`/`asr`).
    pub source: String,
    /// The field name or media filename the text came from.
    pub r#ref: String,
    #[serde(default)]
    /// A snippet around the match, if available.
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// One search signal's contribution to a fused result (provenance).
pub struct SignalContribution {
    /// The signal name (`text`, `exact`, `image`, `tag`, `fuzzy`).
    pub signal: String,
    /// This note's rank within that signal's ranking.
    pub rank: i64,
}

/// A search result with per-mechanism match evidence (Pydantic: `SearchMatch(Note)`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SearchMatch {
    #[serde(flatten)]
    /// The matched note (flattened onto the wire).
    pub note: Note,
    #[serde(default)]
    /// Fused relevance score; `None` for an exact-only hit.
    pub score: Option<f64>,
    #[serde(default)]
    /// Exact-substring evidence, if the exact signal matched.
    pub substring: Option<SubstringInfo>,
    #[serde(default)]
    /// Fuzzy-match evidence, if the fuzzy signal matched.
    pub fuzzy: Option<FuzzyMatch>,
    #[serde(default)]
    /// Which signals surfaced this result, with per-signal ranks.
    pub provenance: Vec<SignalContribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A card template as returned in note-type detail.
pub struct TemplateInfo {
    /// Template name.
    pub name: String,
    /// Front-side template HTML.
    pub front: String,
    /// Back-side template HTML.
    pub back: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A note-type field's editor metadata.
pub struct FieldDetail {
    /// Field name.
    pub name: String,
    /// Editor font family.
    pub font: String,
    /// Editor font size in px.
    pub size: i64,
    /// Editor field description/hint.
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A note type's templates/CSS/fields detail (the `detail` projection).
pub struct NoteTypeDetail {
    /// The note type's card templates.
    pub templates: Vec<TemplateInfo>,
    /// Shared template CSS.
    pub css: String,
    #[serde(default)]
    /// Per-field editor metadata.
    pub fields: Vec<FieldDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A note type in `collection_info` (with optional detail).
pub struct NoteTypeInfo {
    /// Note-type name.
    pub name: String,
    /// Note-type id.
    pub id: i64,
    #[serde(default)]
    /// Field names, in order.
    pub fields: Vec<String>,
    #[serde(default = "default_standard")]
    /// Kind: `standard` or `cloze`.
    pub r#type: String,
    #[serde(default)]
    /// Templates/CSS/field detail; present only when requested.
    pub detail: Option<NoteTypeDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A deck in `collection_info`.
pub struct DeckInfo {
    /// Deck name.
    pub name: String,
    /// Deck id.
    pub id: i64,
    #[serde(default)]
    /// Number of notes in the deck.
    pub note_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The collection summary (paths, timestamps, top-level counts).
pub struct Summary {
    /// Collection file path.
    pub path: String,
    /// Collection creation timestamp.
    pub created: String,
    /// Collection last-modified timestamp.
    pub modified: String,
    /// Total note count.
    pub notes: i64,
    /// Total card count.
    pub cards: i64,
    /// Total deck count.
    pub decks: i64,
    /// Total note-type count.
    pub note_types: i64,
    /// Total tag count.
    pub tags: i64,
    /// Cards due today.
    pub due_today: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// Per-deck note/due counts in `Stats`.
pub struct DeckStat {
    #[serde(default)]
    /// Notes in the deck.
    pub notes: i64,
    #[serde(default)]
    /// Cards due in the deck.
    pub due: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// Collection statistics (the `stats` section of `collection_info`).
pub struct Stats {
    #[serde(default)]
    /// Total note count.
    pub total_notes: i64,
    #[serde(default)]
    /// Total card count.
    pub total_cards: i64,
    #[serde(default)]
    /// Cards due today.
    pub cards_due_today: i64,
    #[serde(default)]
    /// New (unreviewed) cards.
    pub new_cards: i64,
    #[serde(default)]
    /// Per-deck note/due breakdown, keyed by deck name.
    pub decks_summary: BTreeMap<String, DeckStat>,
}

// ============================================================================
// Per-item result variants (discriminated unions)
// ============================================================================

/// Why a candidate note cannot be added (Anki's NoteFieldsCheckResult + the
/// structural problems caught before it).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum NoteValidationReason {
    Duplicate,
    Empty,
    MissingCloze,
    NotetypeNotCloze,
    FieldNotCloze,
    UnknownNoteType,
    UnknownField,
}

/// What a dry-run validated note *would* have done.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum UpsertAction {
    Create,
    Update,
}

/// The only skip reason (`on_duplicate="skip"`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum SkipReason {
    Duplicate,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// One note's outcome in an `upsert_notes` response (status-tagged).
pub enum UpsertNoteResult {
    /// The note was created.
    Created {
        /// The created/updated note's id.
        id: i64,
    },
    /// The note was updated.
    Updated {
        /// The created/updated note's id.
        id: i64,
    },
    /// A dry-run outcome: validated, nothing written.
    Ok {
        /// The item's index in the request batch.
        index: i64,
        /// What a dry-run would have done (create/update).
        action: UpsertAction,
    },
    /// The note was skipped (a duplicate under `on_duplicate="skip"`).
    Skipped {
        /// The item's index in the request batch.
        index: i64,
        /// Why the item was skipped.
        reason: SkipReason,
    },
    /// The note failed validation or write.
    Error {
        /// The item's index in the request batch.
        index: i64,
        /// The failure message.
        error: String,
        #[serde(default)]
        /// The machine-readable validation reason, when applicable.
        reason: Option<NoteValidationReason>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// One note type's outcome in `upsert_note_types` (status-tagged).
pub enum NoteTypeResult {
    /// The note type was created.
    Created {
        /// The created note-type id.
        id: i64,
        /// The note-type name.
        name: String,
    },
    /// The note type was updated.
    Updated {
        /// The updated note-type id.
        id: i64,
        /// The note-type name.
        name: String,
    },
    /// The item failed; `index` is its batch position.
    Error {
        /// The item's index in the request batch.
        index: i64,
        /// The failure message.
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
/// A by-name field edit op for `update_note_type_fields` (op-tagged).
pub enum FieldOp {
    /// Add a field (optionally at `position`).
    Add {
        /// The field name the op addresses.
        name: String,
        // 0-based insert position; mirror the Python `ge=0` so the advertised
        // schema declares the bound it enforces.
        #[serde(default)]
        #[schemars(range(min = 0))]
        /// 0-based insert position; absent appends.
        position: Option<i64>,
    },
    /// Remove a field by name.
    Remove {
        /// The field name the op addresses.
        name: String,
    },
    /// Rename a field.
    Rename {
        /// The field name the op addresses.
        name: String,
        /// The field's new name.
        new_name: String,
    },
    /// Move a field to `position`.
    Reposition {
        /// The field name the op addresses.
        name: String,
        #[schemars(range(min = 0))]
        /// 0-based target position.
        position: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// Per-field editor metadata to set (`update_note_type_field_metadata`).
pub struct FieldMetadataInput {
    /// The field to update.
    pub name: String,
    #[serde(default)]
    /// Editor font family; absent leaves it unchanged.
    pub font: Option<String>,
    // Edit-time font size in px; mirror the Python `ge=1` so the advertised
    // schema declares the bound it enforces.
    #[serde(default)]
    #[schemars(range(min = 1))]
    /// Editor font size in px; absent leaves it unchanged.
    pub size: Option<i64>,
    #[serde(default)]
    /// Editor description; absent leaves it unchanged.
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The result of `update_note_type_fields`.
pub struct UpdateNoteTypeFieldsResponse {
    /// The note-type id.
    pub id: i64,
    /// The note-type name.
    pub name: String,
    /// The resulting field names, in order.
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The result of `update_note_type_field_metadata`.
pub struct UpdateNoteTypeFieldMetadataResponse {
    /// The note-type id.
    pub id: i64,
    /// The note-type name.
    pub name: String,
    /// Field names whose metadata changed.
    pub fields_updated: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
/// A by-name template edit op for `update_note_type_templates` (op-tagged).
pub enum TemplateOp {
    /// Add a template (optionally at `position`).
    Add {
        /// The template name the op addresses.
        name: String,
        /// Front-side template HTML.
        front: String,
        /// Back-side template HTML.
        back: String,
        // 0-based insert position; mirror the Python `ge=0`.
        #[serde(default)]
        #[schemars(range(min = 0))]
        /// 0-based insert position; absent appends.
        position: Option<i64>,
    },
    /// Remove a template by name.
    Remove {
        /// The template name the op addresses.
        name: String,
    },
    /// Rename a template.
    Rename {
        /// The template name the op addresses.
        name: String,
        /// The template's new name.
        new_name: String,
    },
    /// Move a template to `position`.
    Reposition {
        /// The template name the op addresses.
        name: String,
        #[schemars(range(min = 0))]
        /// 0-based target position.
        position: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The result of `update_note_type_templates`.
pub struct UpdateNoteTypeTemplatesResponse {
    /// The note-type id.
    pub id: i64,
    /// The note-type name.
    pub name: String,
    /// The resulting template names, in order.
    pub templates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The result of `find_replace_note_types`.
pub struct FindReplaceNoteTypesResponse {
    /// The note-type id.
    pub id: i64,
    /// The note-type name.
    pub name: String,
    /// Number of replacements made.
    pub replacements: i64,
    /// Templates whose text changed.
    pub templates_changed: Vec<String>,
    /// Whether the shared CSS changed.
    pub css_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// A deck in an `upsert_decks` request (id present = rename/reparent).
pub struct DeckInput {
    #[serde(default)]
    /// Deck id; present renames/reparents, absent creates.
    pub id: Option<i64>,
    /// The deck name (target name on rename).
    pub name: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// One `store_media` item: exactly one of `data`/`url`/`path`.
pub struct StoreMediaItem {
    #[serde(default)]
    /// Target filename (required with `data`).
    pub filename: Option<String>,
    #[serde(default)]
    /// Base64-encoded bytes.
    pub data: Option<String>,
    #[serde(default)]
    /// A URL the server fetches (SSRF-guarded).
    pub url: Option<String>,
    #[serde(default)]
    /// A server-local path (gated; off by default).
    pub path: Option<String>,
}

impl StoreMediaItem {
    /// Pydantic's `model_validator`: exactly one source, and `data` needs a
    /// `filename`. Serde can't express cross-field rules, so callers (the action
    /// layer) validate explicitly.
    ///
    /// # Errors
    ///
    /// Returns an error message if not exactly one of `data`/`url`/`path` is
    /// set, or if `data` is given without a non-empty `filename`.
    pub fn validate(&self) -> Result<(), String> {
        let sources = [&self.data, &self.url, &self.path]
            .iter()
            .filter(|s| s.is_some())
            .count();
        if sources != 1 {
            return Err("provide exactly one of `data`, `url`, or `path`".to_owned());
        }
        if self.data.is_some() && self.filename.as_deref().is_none_or(str::is_empty) {
            return Err(
                "`filename` (with an extension) is required when `data` is given".to_owned(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// One deck's outcome in `upsert_decks` (status-tagged).
pub enum UpsertDeckResult {
    /// The deck was created.
    Created {
        /// The created/updated deck's id.
        id: i64,
        /// The deck name.
        name: String,
    },
    /// The deck was renamed/reparented.
    Updated {
        /// The created/updated deck's id.
        id: i64,
        /// The deck name.
        name: String,
    },
    /// The item failed.
    Error {
        /// The item's index in the request batch.
        index: i64,
        #[serde(default)]
        /// The deck name, when known.
        name: Option<String>,
        /// The failure message.
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// One note type's outcome in `delete_note_types` (status-tagged).
pub enum DeleteNoteTypeResult {
    /// The note type was deleted.
    Deleted {
        /// The note-type id.
        id: i64,
        /// The note-type name.
        name: String,
    },
    /// No note type with that id.
    NotFound {
        /// The note-type id.
        id: i64,
    },
    /// Deletion failed (e.g. the note type is in use).
    Error {
        /// The note-type id.
        id: i64,
        /// The note-type name.
        name: String,
        /// The failure message.
        error: String,
    },
}

// ============================================================================
// Tool response models
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `collection_info` response (each section is caller-selectable).
pub struct CollectionInfo {
    #[serde(default)]
    /// Top-level summary, when requested.
    pub summary: Option<Summary>,
    #[serde(default)]
    /// Note types, when requested.
    pub note_types: Option<Vec<NoteTypeInfo>>,
    #[serde(default)]
    /// Decks, when requested.
    pub decks: Option<Vec<DeckInfo>>,
    #[serde(default)]
    /// Tags, when requested.
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    /// Statistics, when requested.
    pub stats: Option<Stats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `list_notes`/`collection_query` response.
pub struct ListNotesResponse {
    #[serde(default)]
    /// The matched notes (up to `limit`).
    pub notes: Vec<Note>,
    #[serde(default)]
    /// Total matches before the limit.
    pub total: i64,
    #[serde(default = "default_limit")]
    /// The applied limit.
    pub limit: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `migrate_note_type` response.
pub struct MigrateNoteTypeResponse {
    #[serde(default)]
    /// Ids of the migrated notes.
    pub changed: Vec<i64>,
    /// Source note-type name.
    pub from_note_type: String,
    /// Target note-type name.
    pub to_note_type: String,
    #[serde(default)]
    /// Source fields with no mapping (content lost).
    pub dropped_fields: Vec<String>,
    #[serde(default)]
    /// Target fields nothing mapped into.
    pub new_empty_fields: Vec<String>,
    #[serde(default)]
    /// Whether this was a preview (nothing written).
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// One query's results in a `search_notes` response.
pub struct SearchResultGroup {
    /// The query this group answers.
    pub source: String,
    #[serde(default)]
    /// The ranked matches for the query.
    pub matches: Vec<SearchMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `search_notes` response (one group per query).
pub struct SearchResponse {
    #[serde(default)]
    /// Per-query result groups.
    pub results: Vec<SearchResultGroup>,
    #[serde(default)]
    /// An optional advisory (e.g. index still building).
    pub message: Option<String>,
    /// The as-you-type completeness contract: "partial" = the
    /// embedding-bearing signals were skipped at the caller's request
    /// (mode="lexical"); "full" = the final answer for this query/server state.
    #[serde(default)]
    pub completeness: Completeness,
    /// Echo of the request's `version` (client-side stale-response dropping).
    #[serde(default)]
    pub version: Option<i64>,
    /// Freshness advisory: `true` when the kernel was not settled as the search
    /// ran (a write still draining through the embed queue, or a rebuild in
    /// flight), so the semantic ranking may lag the collection. The result is
    /// served regardless; the caller decides to use it or retry. The collection
    /// is always current, so exact/fuzzy hits are never stale.
    #[serde(default)]
    pub stale: bool,
}

/// `SearchResponse.completeness` — mirrors the Pydantic
/// `Literal["partial", "full"]`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum Completeness {
    Partial,
    #[default]
    Full,
}

impl Default for SearchResponse {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            message: None,
            completeness: Completeness::Full,
            version: None,
            stale: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `upsert_notes` response.
pub struct UpsertNotesResponse {
    #[serde(default)]
    /// Per-item outcomes.
    pub results: Vec<UpsertNoteResult>,
    #[serde(default)]
    /// Whether this was a preview (nothing written).
    pub dry_run: bool,
    #[serde(default)]
    /// An optional advisory.
    pub message: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `upsert_note_types` response.
pub struct UpsertNoteTypesResponse {
    #[serde(default)]
    /// Per-item outcomes.
    pub results: Vec<NoteTypeResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `delete_notes` response.
pub struct DeleteNotesResponse {
    #[serde(default)]
    /// Ids that were deleted.
    pub deleted: Vec<i64>,
    #[serde(default)]
    /// Requested ids that didn't exist.
    pub not_found: Vec<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `delete_note_types` response.
pub struct DeleteNoteTypesResponse {
    #[serde(default)]
    /// Per-item outcomes.
    pub results: Vec<DeleteNoteTypeResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `update_note_tags` response.
pub struct UpdateNoteTagsResponse {
    #[serde(default)]
    /// Number of notes whose tags changed.
    pub notes_modified: i64,
    #[serde(default)]
    /// Requested ids that didn't exist.
    pub not_found: Vec<i64>,
    #[serde(default)]
    /// An optional advisory.
    pub message: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `rename_tag` response.
pub struct RenameTagResponse {
    #[serde(default)]
    /// Number of notes whose tags changed.
    pub notes_modified: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The unused-tags cleanup result in `collection_prune`.
pub struct PruneUnusedTags {
    #[serde(default)]
    /// Number of unused tags cleared.
    pub removed: i64,
    #[serde(default)]
    /// The tags that were (or would be) cleared.
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The empty-notes cleanup result in `collection_prune`.
pub struct PruneEmptyNotes {
    #[serde(default)]
    /// Ids of the empty notes removed (or that would be).
    pub removed: Vec<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The empty-cards cleanup result in `collection_prune`.
pub struct PruneEmptyCards {
    #[serde(default)]
    /// Number of empty cards removed.
    pub cards_removed: i64,
    #[serde(default)]
    /// Notes orphaned by card removal and deleted.
    pub notes_deleted: Vec<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The unused-media cleanup result in `collection_prune`.
pub struct PruneUnusedMedia {
    #[serde(default)]
    /// Number of unused media files trashed.
    pub removed: i64,
    #[serde(default)]
    /// The media filenames that were (or would be) trashed.
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `collection_prune` response (each section present only if its cleanup ran).
pub struct CollectionPruneResponse {
    #[serde(default = "default_true")]
    /// Whether this was a preview (nothing changed).
    pub dry_run: bool,
    #[serde(default)]
    /// Unused-tags cleanup, if it ran.
    pub unused_tags: Option<PruneUnusedTags>,
    #[serde(default)]
    /// Empty-notes cleanup, if it ran.
    pub empty_notes: Option<PruneEmptyNotes>,
    #[serde(default)]
    /// Empty-cards cleanup, if it ran.
    pub empty_cards: Option<PruneEmptyCards>,
    #[serde(default)]
    /// Unused-media cleanup, if it ran.
    pub unused_media: Option<PruneUnusedMedia>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `upsert_decks` response.
pub struct UpsertDecksResponse {
    #[serde(default)]
    /// Per-item outcomes.
    pub results: Vec<UpsertDeckResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `delete_decks` response.
pub struct DeleteDecksResponse {
    #[serde(default)]
    /// Deck names that were deleted.
    pub deleted: Vec<String>,
    #[serde(default)]
    /// Requested names that didn't exist.
    pub not_found: Vec<String>,
    #[serde(default)]
    /// Decks not deleted because they weren't empty.
    pub not_empty: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// One `store_media` item's outcome (status-tagged).
pub enum StoreMediaResult {
    /// The file was stored.
    Stored {
        /// The item's index in the request batch.
        index: i64,
        /// The stored filename (use this — Anki may rename on collision).
        filename: String,
        #[serde(default)]
        /// The detected MIME type, if known.
        mime: Option<String>,
        /// Stored size in bytes.
        size_bytes: i64,
        #[serde(default)]
        /// True if an identical file already existed (no new write).
        deduped: bool,
    },
    /// The item failed.
    Error {
        /// The item's index in the request batch.
        index: i64,
        #[serde(default)]
        /// The requested filename, echoed on error.
        filename: Option<String>,
        /// The failure message.
        error: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `store_media` response.
pub struct StoreMediaResponse {
    #[serde(default)]
    /// Per-item outcomes.
    pub results: Vec<StoreMediaResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// One `fetch_media` item's outcome (status-tagged; never returns bytes).
pub enum MediaFetchResult {
    /// The file exists; GET `url` (or read `path`) for bytes.
    Found {
        /// The media filename.
        filename: String,
        #[serde(default)]
        /// The server's `GET /media/<name>` URL, if servable.
        url: Option<String>,
        /// The server-side path of the file.
        path: String,
        #[serde(default)]
        /// The detected MIME type, if known.
        mime: Option<String>,
        /// File size in bytes.
        size_bytes: i64,
    },
    /// No such media file.
    Missing {
        /// The media filename.
        filename: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `fetch_media` response.
pub struct FetchMediaResponse {
    #[serde(default)]
    /// Per-item outcomes.
    pub results: Vec<MediaFetchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// One media file in a `list_media` response.
pub struct MediaFileInfo {
    /// The media filename.
    pub filename: String,
    #[serde(default)]
    /// The server's `GET /media/<name>` URL, if servable.
    pub url: Option<String>,
    #[serde(default)]
    /// The detected MIME type, if known.
    pub mime: Option<String>,
    /// File size in bytes.
    pub size_bytes: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `list_media` response.
pub struct ListMediaResponse {
    /// The collection's media directory.
    pub media_dir: String,
    #[serde(default)]
    /// Number of files returned.
    pub count: i64,
    #[serde(default)]
    /// The media files.
    pub files: Vec<MediaFileInfo>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `delete_media` response.
pub struct DeleteMediaResponse {
    #[serde(default)]
    /// Filenames moved to Anki's trash.
    pub deleted: Vec<String>,
    #[serde(default)]
    /// Requested filenames that didn't exist.
    pub not_found: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The read-only `collection_check` media diagnostics.
pub struct CollectionCheckResponse {
    /// The collection's media directory.
    pub media_dir: String,
    #[serde(default)]
    /// Media files referenced by no note.
    pub unused: Vec<String>,
    #[serde(default)]
    /// Referenced media files that are absent.
    pub missing: Vec<String>,
    #[serde(default)]
    /// Notes referencing a missing media file.
    pub missing_media_notes: Vec<i64>,
    #[serde(default)]
    /// Whether the media trash is non-empty.
    pub have_trash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// One before/after preview row in `find_replace_notes`.
pub struct FindReplaceSample {
    /// The note id.
    pub id: i64,
    /// The field that changed.
    pub field: String,
    /// The field value before replacement.
    pub before: String,
    /// The field value after replacement.
    pub after: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `find_replace_notes` response.
pub struct FindReplaceResponse {
    #[serde(default)]
    /// Number of notes changed (or that would change).
    pub notes_changed: i64,
    #[serde(default)]
    /// Whether this was a preview (nothing written).
    pub dry_run: bool,
    #[serde(default)]
    /// Before/after preview rows.
    pub samples: Vec<FindReplaceSample>,
}

/// The result of importing an `.apkg`/`.colpkg` — per-bucket note counts
/// from anki's importer (`ImportResponse.Log`) + whether the import reconciled
/// the index. The per-bucket mirror of `shrike.schemas.ImportPackageResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ImportPackageResponse {
    #[serde(default)]
    /// Notes newly added.
    pub new: i64,
    #[serde(default)]
    /// Existing notes updated.
    pub updated: i64,
    #[serde(default)]
    /// Notes skipped as duplicates.
    pub duplicate: i64,
    #[serde(default)]
    /// Notes skipped due to a conflict.
    pub conflicting: i64,
    #[serde(default)]
    /// Notes matched by first field.
    pub first_field_match: i64,
    #[serde(default)]
    /// Notes skipped for a missing note type.
    pub missing_notetype: i64,
    #[serde(default)]
    /// Notes skipped for a missing deck.
    pub missing_deck: i64,
    #[serde(default)]
    /// Notes skipped for an empty first field.
    pub empty_first_field: i64,
    #[serde(default)]
    /// Total notes found in the package.
    pub found_notes: i64,
    #[serde(default)]
    /// Whether the import reconciled the vector index.
    pub reindexed: bool,
}

// ============================================================================
// Server status / custom-endpoint models
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum BatchMode {
    Serial,
    Batched,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
/// An embedding space's health (state-tagged).
pub enum EmbeddingStatus {
    /// The embedding service is running.
    Running {
        #[serde(default)]
        /// Whether embeds can currently be served.
        available: bool,
        #[serde(default)]
        /// The backend PID, if it runs a subprocess.
        pid: Option<i64>,
        #[serde(default)]
        /// The backend endpoint URL, if any.
        url: Option<String>,
        #[serde(default)]
        /// The loaded model identifier.
        model: Option<String>,
        #[serde(default)]
        /// The actually-loaded compute provider (e.g. CPU/CoreML).
        provider: Option<String>,
        #[serde(default)]
        /// Whether the model is proven safe to batch.
        batch_safe: Option<bool>,
        #[serde(default)]
        /// Whether embeds run batched or serially.
        batch: Option<BatchMode>,
        /// The modalities this space embeds.
        #[serde(default)]
        modalities: Option<Vec<String>>,
    },
    /// The service is stopped.
    Stopped {
        #[serde(default)]
        /// Always false in this (non-running) state.
        available: LiteralFalse,
    },
    /// The service failed to start.
    Failed {
        #[serde(default)]
        /// Always false in this (non-running) state.
        available: LiteralFalse,
    },
    /// No embedder is configured for this space.
    NotConfigured {
        #[serde(default)]
        /// Always false in this (non-running) state.
        available: LiteralFalse,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// Build progress for the vector index.
pub struct IndexProgress {
    #[serde(default)]
    /// Notes embedded so far.
    pub indexed: i64,
    #[serde(default)]
    /// Total notes to embed.
    pub total: i64,
}

/// One per-modality sub-index's size/ndim. Mirror of Pydantic's
/// `IndexModalityStat`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct IndexModalityStat {
    /// The modality name (`text`, `image`).
    pub modality: String,
    #[serde(default)]
    /// Vectors in this sub-index.
    pub size: i64,
    #[serde(default)]
    /// This sub-index's dimensionality; `None` if empty.
    pub ndim: Option<i64>,
}

/// On-disk index contents shared across build states (Pydantic's `_IndexBase`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct IndexBase {
    #[serde(default)]
    /// Whether the index can currently serve searches.
    pub available: bool,
    #[serde(default)]
    /// Total vectors (the note count).
    pub size: i64,
    #[serde(default)]
    /// The text modality's dimensionality; `None` before any vectors.
    pub ndim: Option<i64>,
    #[serde(default)]
    /// The on-disk index path.
    pub path: Option<String>,
    #[serde(default)]
    /// The `col.mod` the index was last built at.
    pub col_mod: Option<i64>,
    #[serde(default)]
    /// The model fingerprint the vectors were built with.
    pub model_id: Option<String>,
    #[serde(default)]
    /// Per-modality activation-gate calibration stats.
    pub activation: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    /// Per-modality sub-index breakdown: each sub-index's own size/ndim.
    #[serde(default)]
    pub modalities: Vec<IndexModalityStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
/// The vector index's state (state-tagged).
pub enum IndexStatus {
    /// The embedding service isn't running; on-disk vectors only.
    Unavailable {
        #[serde(flatten)]
        /// The shared on-disk index contents.
        base: IndexBase,
    },
    /// A build/rebuild is in progress.
    Building {
        #[serde(flatten)]
        /// The shared on-disk index contents.
        base: IndexBase,
        /// Build progress.
        progress: IndexProgress,
    },
    /// The index is built and serving.
    Ready {
        #[serde(flatten)]
        /// The shared on-disk index contents.
        base: IndexBase,
    },
    /// The last build failed.
    Error {
        #[serde(flatten)]
        /// The shared on-disk index contents.
        base: IndexBase,
        /// The build-failure message.
        error: String,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum DerivedState {
    #[default]
    Unavailable,
    Building,
    Ready,
    Error,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The derived-text (FTS5) store's status.
pub struct DerivedStatus {
    #[serde(default)]
    /// The store's lifecycle state.
    pub state: DerivedState,
    #[serde(default)]
    /// Whether lexical lookups can be served.
    pub available: bool,
    #[serde(default)]
    /// Whether the runtime SQLite has FTS5 + the trigram tokenizer.
    pub fts5: bool,
    #[serde(default)]
    /// Indexed row count.
    pub size: i64,
    #[serde(default)]
    /// The sidecar database path.
    pub path: Option<String>,
    #[serde(default)]
    /// The `col.mod` the store was last reconciled to.
    pub col_mod: Option<i64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum LockingMode {
    #[default]
    Permanent,
    Cooperative,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum RecognitionState {
    Unavailable,
    Building,
    // A present engine row defaults to ready (the harness constructs it with an
    // explicit state; this is the deserialization default for an older/partial
    // payload).
    #[default]
    Ready,
    Error,
}

/// One attached recognition engine's self-report. The Rust mirror
/// of `shrike.schemas.RecognitionEngineStatus`; `ServerStatus.recognition` is
/// a map of these keyed by source (`ocr`/`vlm`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RecognitionEngineStatus {
    #[serde(default)]
    /// The engine's lifecycle state.
    pub state: RecognitionState,
    #[serde(default)]
    /// The engine backend name.
    pub backend: Option<String>,
    #[serde(default)]
    /// The engine's model fingerprint.
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// Dedup best-match statistics surfaced on `/status`.
pub struct DedupStats {
    #[serde(default)]
    /// Number of neighbor lookups sampled.
    pub samples: i64,
    #[serde(default)]
    /// Lookups that found no neighbor.
    pub no_match: i64,
    #[serde(default)]
    /// A histogram of best-match scores.
    pub buckets: Vec<i64>,
}

/// One collection's state in a multi-collection daemon's `/status`: a row
/// per known collection (the boot/default plus every registered profile). The
/// per-collection mirror of `shrike.schemas.CollectionStatus`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CollectionStatus {
    /// The profile/collection name.
    pub name: String,
    /// The collection file path.
    pub path: String,
    /// Whether it's a registered profile (vs the boot default).
    pub registered: bool,
    #[serde(default)]
    /// Whether it's the active default.
    pub is_default: bool,
    #[serde(default)]
    /// Whether it's the currently-open collection.
    pub active: bool,
    #[serde(default)]
    /// Whether its collection lock is currently held.
    pub held: Option<bool>,
    #[serde(default)]
    /// The index state for this collection, if known.
    pub index_state: Option<String>,
    #[serde(default)]
    /// The collection's `col.mod`, if known.
    pub col_mod: Option<i64>,
}

/// How one (query-modality → target-modality) pair is reachable. The
/// Rust mirror of `shrike.schemas.CoverageCell`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum CoverageCell {
    // A single live space embeds both the query and target modality.
    Native,
    // A recognizer derives text from the target into the text space.
    ViaDerivedText,
    // Neither — the target can't be reached from this query (the default for a
    // partial/older payload).
    #[default]
    Unavailable,
}

/// One query modality's reachability to each target modality. The Rust
/// mirror of `shrike.schemas.CoverageRow`; every cell is a `CoverageCell` (no
/// absent target, only an `Unavailable` one).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CoverageRow {
    #[serde(default)]
    /// How the text modality is reachable from this query modality.
    pub text: CoverageCell,
    #[serde(default)]
    /// How the image modality is reachable.
    pub image: CoverageCell,
    #[serde(default)]
    /// How the audio modality is reachable.
    pub audio: CoverageCell,
}

/// The cross-modal coverage matrix: per query modality, a `CoverageRow`
/// naming how each target modality is reachable. The Rust mirror of
/// `shrike.schemas.CoverageMatrix`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CoverageMatrix {
    #[serde(default)]
    /// Reachability when the query is text.
    pub text: CoverageRow,
    #[serde(default)]
    /// Reachability when the query is an image.
    pub image: CoverageRow,
    #[serde(default)]
    /// Reachability when the query is audio.
    pub audio: CoverageRow,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `GET /status` payload — the running server's full self-report.
pub struct ServerStatus {
    #[serde(default)]
    /// Always true (the server answered).
    pub running: LiteralTrue,
    /// The action exchange's protocol version.
    pub wire_protocol_version: u32,
    /// The server PID.
    pub pid: i64,
    /// The server's base URL.
    pub url: String,
    /// The open collection's path.
    pub collection: String,
    /// The active log level.
    pub log_level: String,
    /// The log directory.
    pub log_dir: String,
    #[serde(default)]
    /// How long the server has been up.
    pub uptime: Option<String>,
    #[serde(default)]
    /// A recent log tail, if requested.
    pub log: Option<String>,
    /// The primary embedding space's health.
    pub embedding: EmbeddingStatus,
    /// Per-space embedding health: one entry per configured embedder (the
    /// primary plus every secondary space). `embedding` above stays the primary
    /// for back-compat; this is the full list. Empty on older payloads.
    #[serde(default)]
    pub embedding_spaces: Vec<EmbeddingStatus>,
    /// The vector index state.
    pub index: IndexStatus,
    #[serde(default)]
    /// The derived-text store status.
    pub derived: DerivedStatus,
    #[serde(default)]
    /// The collection-lock mode (permanent/cooperative).
    pub locking: LockingMode,
    #[serde(default = "default_true")]
    /// Whether the collection lock is currently held.
    pub collection_held: bool,
    /// Dedup/activation best-match statistics — None until the first search
    /// records a sample.
    #[serde(default)]
    pub dedup: Option<DedupStats>,
    /// Per-engine recognition state: a map keyed by source
    /// (`ocr`/`vlm`), each row {state, backend, fingerprint}. An EMPTY map is
    /// "nothing attached" (distinct from an attached-but-errored engine).
    #[serde(default)]
    pub recognition: std::collections::BTreeMap<String, RecognitionEngineStatus>,
    /// The cross-modal coverage matrix: for each (query, target)
    /// modality pair, how the target is reachable — `native`, `via_derived_text`
    /// (a recognizer derives text from the target into the text space), or
    /// `unavailable`. None on payloads from older servers (the flat shape).
    #[serde(default)]
    pub coverage: Option<CoverageMatrix>,
    /// Multi-collection routing: one row per known collection (the
    /// boot/default plus every registered profile). None on a single-collection
    /// server / older payloads; the top-level fields describe the default
    /// collection (which the operational routes act on).
    #[serde(default)]
    pub collections: Option<Vec<CollectionStatus>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// The `POST /index/rebuild` outcome (status-tagged).
pub enum IndexRebuildResponse {
    /// A rebuild started over `total` notes.
    Started {
        /// Total notes to embed.
        total: i64,
    },
    /// The index is already current (`size` vectors).
    Complete {
        /// The current vector count.
        size: i64,
    },
    /// A build is already in progress.
    AlreadyBuilding {
        /// The in-progress build's progress.
        progress: IndexProgress,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// The `POST /index/save` outcome (status-tagged).
pub enum IndexSaveResponse {
    /// The index was flushed (`size` vectors, `pending` unsaved before).
    Saved {
        /// The vector count after the flush.
        size: i64,
        /// Unsaved changes that were pending before the flush.
        pending: i64,
    },
    /// Nothing to save (no vectors).
    Empty,
    /// Refused — a build is in progress.
    Building {
        /// The in-progress build's progress.
        progress: IndexProgress,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// The `POST /embedding/start` outcome (status-tagged).
pub enum EmbeddingStartResponse {
    /// The service started.
    Started {
        /// The embedding space's resulting health.
        embedding: EmbeddingStatus,
        /// The index state after attaching.
        index: IndexStatus,
    },
    /// The service was already running.
    AlreadyRunning {
        /// The embedding space's resulting health.
        embedding: EmbeddingStatus,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// The `POST /embedding/stop` outcome (status-tagged).
pub enum EmbeddingStopResponse {
    /// The service stopped; the index is now unavailable.
    Stopped {
        /// The index state after stopping (now unavailable).
        index: IndexStatus,
    },
    /// The service wasn't running.
    NotRunning,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `POST /shutdown` acknowledgement.
pub struct ShutdownResponse {
    /// The shutdown status string.
    pub status: String,
    /// The server PID that is shutting down.
    pub pid: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
/// The `POST /reload` outcome.
pub struct ReloadResponse {
    #[serde(default)]
    /// Always `reloaded`.
    pub status: ReloadedLiteral,
    /// The reopened collection's `col.mod`.
    pub col_mod: i64,
    #[serde(default)]
    /// Whether a background rebuild was triggered by drift.
    pub rebuilding: bool,
}

/// One registered collection profile. The registry name is a friendly
/// handle only — index identity keys on the collection file path, never the
/// name. `is_default` marks the active default (the profile the per-call
/// selector resolves to when none is passed).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ProfileEntry {
    /// The profile's friendly name.
    pub name: String,
    /// The collection file path (the index identity).
    pub path: String,
    #[serde(default)]
    /// Whether it's the active default.
    pub is_default: bool,
}

/// The collection/profile registry enumeration: the registered profiles
/// and the active-default name (`None` when none is set). Read-only — selection
/// as routing is the capstone.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ListProfilesResponse {
    #[serde(default)]
    /// The registered profiles.
    pub profiles: Vec<ProfileEntry>,
    #[serde(default)]
    /// The active-default profile name, if set.
    pub default: Option<String>,
}

/// The kernel's export-op outcome: the count of notes written and the
/// on-disk path the package landed at. Internal wire — the kernel op returns
/// this; the host action wraps it into [`ExportPackageResponse`] (adding the
/// download `url` / `bytes`, which the kernel doesn't know about).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ExportPackageResult {
    /// Notes written to the package.
    pub note_count: u32,
    /// Where the package landed on disk.
    pub out_path: String,
}

/// The export tool's response: the package the export produced, handed
/// back as a server-local `path` (when the operator opted into a contained
/// `output_path`) OR a downloadable `url` (the default — the server wrote a
/// temp file; never base64, mirroring `fetch_media`). Discriminated on
/// `delivery` so a client can't read a `path` that isn't there.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "delivery", rename_all = "snake_case")]
pub enum ExportPackageResponse {
    /// The server wrote the package to a contained server-local path; the
    /// caller (sharing the disk) reads it there.
    Path {
        /// Notes written to the package.
        note_count: u32,
        /// Package size in bytes.
        bytes: u64,
        /// "apkg" or "colpkg".
        format: String,
        /// The server-local path of the package.
        path: String,
    },
    /// The server wrote a temp package and serves it at `url` (GET it; the
    /// temp file is reaped after download / on a TTL / at shutdown).
    Url {
        /// Notes written to the package.
        note_count: u32,
        /// Package size in bytes.
        bytes: u64,
        /// "apkg" or "colpkg".
        format: String,
        /// The download URL (GET it).
        url: String,
    },
}

/// Discriminated on the bool `stopped` (string tags only in serde, so this is
/// an untagged union whose variants self-select via the literal-bool types).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
pub enum StopResponse {
    /// The server stopped (or was already stopped).
    Succeeded {
        #[serde(default)]
        /// Always true in the success variant.
        stopped: LiteralTrue,
        #[serde(default)]
        /// The stopped server's PID, if known.
        pid: Option<i64>,
        #[serde(default)]
        /// Whether a forceful kill was needed.
        forced: bool,
    },
    /// The stop attempt failed.
    Failed {
        /// Always false in the failure variant.
        stopped: LiteralFalse,
        /// Why the stop failed.
        reason: String,
    },
}

// ============================================================================
// The action exchange error envelope
// ============================================================================

/// The machine-readable class of an actions-over-HTTP failure.
///
/// The UI edge (`POST /actions/{name}`) maps the transport-neutral error
/// contract the actions raise — and that `_safe_tool` re-raises — onto a small,
/// stable taxonomy carried in the [`ActionError`] body, paired with an HTTP
/// status. It deliberately mirrors the MCP edge's split (a `ToolInputError` is
/// the caller's mistake; `collection_busy` is contention; everything else is an
/// internal bug whose detail stays in the log, never on the wire) without
/// reusing MCP's JSON-RPC envelope.
///
/// Like [`NoteValidationReason`] this is a *field-level* enum (the `code` of
/// `ActionError`), not a standalone catalog entry — its shape is contract-tested
/// through `ActionError`. The codes (and their HTTP status): `input_error` (400,
/// a caller mistake — a `ToolInputError` or argument-validation failure);
/// `collection_busy` (409, contention under cooperative locking — the op
/// never ran, retryable); `unknown_action` (404, no such action name);
/// `internal_error` (500, an unexpected server bug — detail logged, never
/// returned, so it can't leak to a UI client).
///
/// No per-variant doc comments: schemars then renders a plain string `enum`,
/// matching Pydantic's str-Enum (the contract normalizer compares them).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
// A plain string-enum: documenting individual variants makes schemars emit a
// `oneOf` of described consts instead of a flat `enum`, which diverges from
// the Pydantic str-Enum the schema contract test compares against (see
// `ActionErrorCode` and test_schema_contract.py). The type doc is harmless;
// the variants stay undocumented by design.
#[allow(missing_docs)]
pub enum ActionErrorCode {
    InputError,
    CollectionBusy,
    UnknownAction,
    InternalError,
}

/// The one error envelope every `POST /actions/{name}` failure returns.
///
/// Defined once here (the wire contract is shrike-schemas verbatim) and mirrored
/// by `shrike.schemas.ActionError`. `message` is a non-leaking human string: for
/// an `internal_error` it is a fixed, generic sentence (the real cause is in the
/// server log); for the caller-actionable codes it carries the actionable text.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ActionError {
    /// The machine-readable error class.
    pub code: ActionErrorCode,
    /// A non-leaking human-readable message.
    pub message: String,
}

// ============================================================================
// The catalog: every wire type by its Python name
// ============================================================================

macro_rules! catalog {
    ($(($name:literal, $ty:ty)),* $(,)?) => {
        /// `(python_name, json_schema)` for every wire type — the contract
        /// test's Rust side.
        pub fn schema_catalog() -> Vec<(&'static str, String)> {
            vec![$((
                $name,
                serde_json::to_string(&schemars::schema_for!($ty))
                    .expect("schema serializes"),
            )),*]
        }

        /// Deserialize `json` as the named type and re-serialize it — the
        /// instance-level wire-parity probe (parse + emit through Rust).
        ///
        /// # Errors
        ///
        /// Returns an error if `name` is not a known wire type, or if `json`
        /// fails to deserialize as that type or re-serialize.
        pub fn roundtrip(name: &str, json: &str) -> Result<String, String> {
            match name {
                $($name => {
                    let value: $ty =
                        serde_json::from_str(json).map_err(|e| e.to_string())?;
                    serde_json::to_string(&value).map_err(|e| e.to_string())
                })*
                _ => Err(format!("unknown schema type: {name}")),
            }
        }
    };
}

/// The action exchange's protocol version — the compatibility story,
/// decided while there is one consumer:
///
/// - **The exchange evolves additively.** A breaking change to an action's
///   contract ships as a NEW action name (`upsert_notes_v2`) with its own
///   types alongside the old — at this layer that's an addition, so old
///   clients never see it. New union variants count as breaking for an old
///   consumer of an EXISTING action (exhaustive tagged-union parses), which
///   is exactly why the name-versioning discipline exists.
/// - **This constant is the backstop**, bumped only when the exchange fabric
///   itself breaks (envelope semantics, error taxonomy, FFI conventions) —
///   never for per-action evolution. A future remote handshake (thin client,
///   relay) refuses on mismatch; `/status` reports it today.
/// - The MCP tool surface (external clients, no handshake possible) rides
///   the same discipline: a breaking tool change is a new tool name carrying
///   its new schema types; the old tool keeps its old types while served.
///
/// The Python mirror (`shrike.schemas.WIRE_PROTOCOL_VERSION`) is pinned
/// equal by the schema contract test.
pub const WIRE_PROTOCOL_VERSION: u32 = 1;

catalog![
    ("TemplateInput", TemplateInput),
    ("NoteInput", NoteInput),
    ("NoteTypeInput", NoteTypeInput),
    ("Note", Note),
    ("SubstringInfo", SubstringInfo),
    ("FuzzyMatch", FuzzyMatch),
    ("SignalContribution", SignalContribution),
    ("SearchMatch", SearchMatch),
    ("TemplateInfo", TemplateInfo),
    ("FieldDetail", FieldDetail),
    ("NoteTypeDetail", NoteTypeDetail),
    ("NoteTypeInfo", NoteTypeInfo),
    ("DeckInfo", DeckInfo),
    ("Summary", Summary),
    ("DeckStat", DeckStat),
    ("Stats", Stats),
    ("UpsertNoteResult", UpsertNoteResult),
    ("NoteTypeResult", NoteTypeResult),
    ("FieldOp", FieldOp),
    ("FieldMetadataInput", FieldMetadataInput),
    ("UpdateNoteTypeFieldsResponse", UpdateNoteTypeFieldsResponse),
    (
        "UpdateNoteTypeFieldMetadataResponse",
        UpdateNoteTypeFieldMetadataResponse
    ),
    ("TemplateOp", TemplateOp),
    (
        "UpdateNoteTypeTemplatesResponse",
        UpdateNoteTypeTemplatesResponse
    ),
    ("FindReplaceNoteTypesResponse", FindReplaceNoteTypesResponse),
    ("DeckInput", DeckInput),
    ("StoreMediaItem", StoreMediaItem),
    ("UpsertDeckResult", UpsertDeckResult),
    ("DeleteNoteTypeResult", DeleteNoteTypeResult),
    ("CollectionInfo", CollectionInfo),
    ("ListNotesResponse", ListNotesResponse),
    ("MigrateNoteTypeResponse", MigrateNoteTypeResponse),
    ("SearchResultGroup", SearchResultGroup),
    ("SearchResponse", SearchResponse),
    ("UpsertNotesResponse", UpsertNotesResponse),
    ("UpsertNoteTypesResponse", UpsertNoteTypesResponse),
    ("DeleteNotesResponse", DeleteNotesResponse),
    ("DeleteNoteTypesResponse", DeleteNoteTypesResponse),
    ("UpdateNoteTagsResponse", UpdateNoteTagsResponse),
    ("RenameTagResponse", RenameTagResponse),
    ("PruneUnusedTags", PruneUnusedTags),
    ("PruneEmptyNotes", PruneEmptyNotes),
    ("PruneEmptyCards", PruneEmptyCards),
    ("PruneUnusedMedia", PruneUnusedMedia),
    ("CollectionPruneResponse", CollectionPruneResponse),
    ("UpsertDecksResponse", UpsertDecksResponse),
    ("DeleteDecksResponse", DeleteDecksResponse),
    ("StoreMediaResult", StoreMediaResult),
    ("StoreMediaResponse", StoreMediaResponse),
    ("MediaFetchResult", MediaFetchResult),
    ("FetchMediaResponse", FetchMediaResponse),
    ("MediaFileInfo", MediaFileInfo),
    ("ListMediaResponse", ListMediaResponse),
    ("DeleteMediaResponse", DeleteMediaResponse),
    ("CollectionCheckResponse", CollectionCheckResponse),
    ("FindReplaceSample", FindReplaceSample),
    ("FindReplaceResponse", FindReplaceResponse),
    ("ImportPackageResponse", ImportPackageResponse),
    ("EmbeddingStatus", EmbeddingStatus),
    ("IndexProgress", IndexProgress),
    ("IndexModalityStat", IndexModalityStat),
    ("IndexStatus", IndexStatus),
    ("DedupStats", DedupStats),
    ("RecognitionEngineStatus", RecognitionEngineStatus),
    ("DerivedStatus", DerivedStatus),
    ("CollectionStatus", CollectionStatus),
    ("CoverageRow", CoverageRow),
    ("CoverageMatrix", CoverageMatrix),
    ("ServerStatus", ServerStatus),
    ("IndexRebuildResponse", IndexRebuildResponse),
    ("IndexSaveResponse", IndexSaveResponse),
    ("EmbeddingStartResponse", EmbeddingStartResponse),
    ("EmbeddingStopResponse", EmbeddingStopResponse),
    ("ShutdownResponse", ShutdownResponse),
    ("ReloadResponse", ReloadResponse),
    ("StopResponse", StopResponse),
    ("ActionError", ActionError),
    ("ProfileEntry", ProfileEntry),
    ("ListProfilesResponse", ListProfilesResponse),
    ("ExportPackageResult", ExportPackageResult),
    ("ExportPackageResponse", ExportPackageResponse),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_complete_and_emits_schemas() {
        let catalog = schema_catalog();
        assert!(catalog.len() >= 70, "catalog has {} entries", catalog.len());
        for (name, schema) in &catalog {
            let parsed: serde_json::Value =
                serde_json::from_str(schema).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(parsed.is_object(), "{name} schema is not an object");
        }
    }

    #[test]
    fn tagged_union_wire_shape() {
        let created = UpsertNoteResult::Created { id: 7 };
        let json = serde_json::to_string(&created).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["status"], "created");
        let back: UpsertNoteResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back, created);
    }

    #[test]
    fn multi_value_tag_variants_roundtrip() {
        for status in ["created", "updated"] {
            let json = format!(r#"{{"status":"{status}","id":1,"name":"Basic"}}"#);
            let parsed: NoteTypeResult = serde_json::from_str(&json).unwrap();
            let emitted: serde_json::Value =
                serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
            assert_eq!(emitted["status"], status);
        }
    }

    #[test]
    fn options_serialize_as_explicit_null() {
        // The wire is Pydantic's model_dump: None fields present as null.
        let note = Note {
            id: 1,
            note_type: "Basic".into(),
            deck: "D".into(),
            tags: vec![],
            modified: "2026-01-01".into(),
            content: None,
        };
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&note).unwrap()).unwrap();
        assert!(value.as_object().unwrap().contains_key("content"));
        assert!(value["content"].is_null());
    }

    #[test]
    fn flattened_search_match_inlines_note_fields() {
        let json = r#"{"id":1,"note_type":"Basic","deck":"D","tags":[],"modified":"m",
                       "content":null,"score":0.5,"substring":null,"fuzzy":null,
                       "provenance":[{"signal":"text","rank":1}]}"#;
        let m: SearchMatch = serde_json::from_str(json).unwrap();
        assert_eq!(m.note.id, 1);
        assert_eq!(m.score, Some(0.5));
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(value["id"], 1); // flattened, not nested under "note"
    }

    #[test]
    fn note_roundtrip_lossless_full_and_meta() {
        // The search path rests on `read_notes_batch` parsing each `note_dicts`
        // wire dict via `from_value::<Note>` losslessly — the dict IS a
        // serialized `Note`. Pin that round-trip value-for-value across the
        // edges the search path hydrates: full mode, meta mode (no content), an
        // embedded 0x1f field separator, multibyte unicode, and empty tags.
        let full = Note {
            id: 42,
            note_type: "Basic".into(),
            deck: "Deck::Sub".into(),
            tags: vec!["t1".into(), "t2".into()],
            modified: "2026-01-01T00:00:00".into(),
            content: Some(BTreeMap::from([
                // An embedded 0x1f survives the JSON round-trip verbatim (it is
                // not a JSON metacharacter); multibyte unicode survives too.
                ("Front".into(), "a\u{1f}b — 日本語 🎴".into()),
                ("Back".into(), String::new()),
            ])),
        };
        let meta = Note {
            tags: vec![],
            content: None,
            ..full.clone()
        };
        for note in [&full, &meta] {
            let json = serde_json::to_string(note).unwrap();
            let back: Note = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, note, "Note must round-trip value-for-value");
        }
    }

    #[test]
    fn searchmatch_wire_shape_absent_equals_null() {
        // The exact equivalence the search path rests on: the typed edge emits
        // score/substring/fuzzy as explicit null when None, AND a wire object
        // that OMITS those keys (the older shape, where they were never set)
        // deserializes to the IDENTICAL struct — so old-omit ≡ new-null once
        // each side passes through `SearchMatch` (de)serialization.
        let none = SearchMatch {
            note: Note {
                id: 1,
                note_type: "Basic".into(),
                deck: "D".into(),
                tags: vec![],
                modified: "m".into(),
                content: None,
            },
            score: None,
            substring: None,
            fuzzy: None,
            provenance: vec![],
        };
        // (1) None serializes as explicit null (keys present), the model_dump wire.
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&none).unwrap()).unwrap();
        let obj = value.as_object().unwrap();
        for key in ["score", "substring", "fuzzy"] {
            assert!(obj.contains_key(key), "{key} key must be present");
            assert!(value[key].is_null(), "{key} must serialize as null");
        }
        // (2) A wire object OMITTING those keys deserializes to the same struct.
        let omitted = r#"{"id":1,"note_type":"Basic","deck":"D","tags":[],
                          "modified":"m","content":null,"provenance":[]}"#;
        let from_omitted: SearchMatch = serde_json::from_str(omitted).unwrap();
        assert_eq!(from_omitted, none, "omitted keys must equal explicit null");
    }

    #[test]
    fn bool_tagged_stop_response_discriminates() {
        let ok: StopResponse =
            serde_json::from_str(r#"{"stopped":true,"pid":42,"forced":false}"#).unwrap();
        assert!(matches!(ok, StopResponse::Succeeded { pid: Some(42), .. }));
        let no: StopResponse =
            serde_json::from_str(r#"{"stopped":false,"reason":"not running"}"#).unwrap();
        assert!(matches!(no, StopResponse::Failed { .. }));
    }

    #[test]
    fn store_media_item_cross_field_validation() {
        let both = StoreMediaItem {
            data: Some("x".into()),
            url: Some("http://e".into()),
            ..Default::default()
        };
        assert!(both.validate().is_err());
        let neither = StoreMediaItem::default();
        assert!(neither.validate().is_err());
        let data_no_name = StoreMediaItem {
            data: Some("x".into()),
            ..Default::default()
        };
        assert!(data_no_name.validate().is_err());
        let ok = StoreMediaItem {
            data: Some("x".into()),
            filename: Some("a.png".into()),
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn unknown_keys_are_ignored() {
        // Pydantic's extra="ignore": a newer server's field doesn't break us.
        let json = r#"{"signal":"text","rank":1,"brand_new_field":123}"#;
        assert!(serde_json::from_str::<SignalContribution>(json).is_ok());
    }

    // The numeric-bound parity the advertised schema must declare: the
    // Python side enforces these with `ge=`; without the schemars `range`, the
    // schema served via MCP `tools/list` claimed any integer, so a client
    // following it strictly had inaccurate type info.
    fn schema_of(name: &str) -> serde_json::Value {
        let json = schema_catalog()
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no schema for {name}"))
            .1;
        serde_json::from_str(&json).unwrap()
    }

    /// The `minimum` declared on a property, unwrapping an Option's
    /// `[T, "null"]`/anyOf wrapper to the non-null branch.
    fn property_minimum(schema: &serde_json::Value, prop: &str) -> Option<i64> {
        let p = &schema["properties"][prop];
        if let Some(m) = p.get("minimum") {
            return m.as_i64();
        }
        // Option<T> renders as anyOf:[{...}, {"type":"null"}]; the bound lives
        // on the non-null branch.
        if let Some(branches) = p.get("anyOf").and_then(|b| b.as_array()) {
            for b in branches {
                if b.get("type").and_then(|t| t.as_str()) != Some("null") {
                    if let Some(m) = b.get("minimum") {
                        return m.as_i64();
                    }
                }
            }
        }
        None
    }

    fn tagged_variant<'a>(
        schema: &'a serde_json::Value,
        tag: &str,
        value: &str,
    ) -> &'a serde_json::Value {
        let branches = schema["oneOf"]
            .as_array()
            .or_else(|| schema["anyOf"].as_array())
            .expect("tagged union has oneOf/anyOf");
        branches
            .iter()
            .find(|b| {
                let t = &b["properties"][tag];
                t.get("const").and_then(|c| c.as_str()) == Some(value)
                    || t.get("enum")
                        .and_then(|e| e.as_array())
                        .map(|vs| vs.iter().any(|v| v.as_str() == Some(value)))
                        .unwrap_or(false)
            })
            .unwrap_or_else(|| panic!("no {tag}={value} variant"))
    }

    #[test]
    fn field_metadata_size_schema_declares_minimum_1() {
        let schema = schema_of("FieldMetadataInput");
        assert_eq!(property_minimum(&schema, "size"), Some(1));
    }

    #[test]
    fn field_op_position_schema_declares_minimum_0() {
        let schema = schema_of("FieldOp");
        let add = tagged_variant(&schema, "op", "add");
        assert_eq!(property_minimum(add, "position"), Some(0));
        let reposition = tagged_variant(&schema, "op", "reposition");
        assert_eq!(property_minimum(reposition, "position"), Some(0));
    }

    #[test]
    fn template_op_position_schema_declares_minimum_0() {
        let schema = schema_of("TemplateOp");
        let add = tagged_variant(&schema, "op", "add");
        assert_eq!(property_minimum(add, "position"), Some(0));
        let reposition = tagged_variant(&schema, "op", "reposition");
        assert_eq!(property_minimum(reposition, "position"), Some(0));
    }

    // ========================================================================
    // Adversarial wire-boundary tests (#742)
    // ========================================================================

    // SplitMix64 — the house deterministic PRNG (mirrors shrike-store's test
    // helper). No external dep; reproducible fuzz seeds.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// A structurally-plausible seed instance per catalog name. The fuzz
    /// mutates these toward malformed input; a seed need not be valid for the
    /// type (the property is "never panics", not "always parses").
    fn fuzz_seeds() -> &'static [&'static str] {
        &[
            r#"{"id":1,"note_type":"Basic","deck":"D","tags":["a"],"modified":"m","content":{"F":"v"}}"#,
            r#"{"status":"created","id":7}"#,
            r#"{"status":"error","index":0,"error":"boom","reason":"duplicate"}"#,
            r#"{"op":"add","name":"F","position":0}"#,
            r#"{"op":"reposition","name":"F","position":3}"#,
            r#"{"id":1,"note_type":"Basic","deck":"D","tags":[],"modified":"m","content":null,"score":0.5,"provenance":[{"signal":"text","rank":1}]}"#,
            r#"{"state":"running","available":true,"pid":42,"modalities":["text"]}"#,
            r#"{"state":"ready","available":true,"size":10,"ndim":384,"modalities":[]}"#,
            r#"{"stopped":true,"pid":9,"forced":false}"#,
            r#"{"delivery":"url","note_count":3,"bytes":99,"format":"apkg","url":"http://x"}"#,
            r#"{"code":"input_error","message":"bad"}"#,
            r#"{}"#,
            r#"[]"#,
            r#"null"#,
            r#"{"running":true,"wire_protocol_version":1,"pid":1,"url":"u","collection":"c","log_level":"info","log_dir":"d","embedding":{"state":"not_configured"},"index":{"state":"ready","available":true,"size":0},"derived":{}}"#,
        ]
    }

    /// Byte-level mutation: corrupt a seed in one of several adversarial ways.
    fn mutate(seed: &str, rng: &mut Rng) -> String {
        let mut bytes = seed.as_bytes().to_vec();
        if bytes.is_empty() {
            return seed.to_owned();
        }
        match rng.next_u64() % 6 {
            0 => {
                // Flip a byte.
                let i = (rng.next_u64() as usize) % bytes.len();
                bytes[i] ^= (rng.next_u64() & 0xff) as u8;
            }
            1 => {
                // Delete a byte.
                let i = (rng.next_u64() as usize) % bytes.len();
                bytes.remove(i);
            }
            2 => {
                // Insert a structural metacharacter.
                let i = (rng.next_u64() as usize) % (bytes.len() + 1);
                let injected =
                    [b'{', b'}', b'[', b']', b'"', b':', b',', 0x1f][(rng.next_u64() % 8) as usize];
                bytes.insert(i, injected);
            }
            3 => {
                // Splice in an adversarial token (null, huge int, unicode, nan).
                let tokens = [
                    "null",
                    "99999999999999999999999999",
                    "-9223372036854775809",
                    "\"日本語\\u0000\"",
                    "NaN",
                    "1e999",
                    "true",
                ];
                let tok = tokens[(rng.next_u64() % tokens.len() as u64) as usize];
                return format!("{seed}{tok}");
            }
            4 => {
                // Truncate.
                let cut = 1 + (rng.next_u64() as usize) % bytes.len();
                bytes.truncate(cut);
            }
            _ => {
                // Duplicate a region (unbalanced braces).
                let i = (rng.next_u64() as usize) % bytes.len();
                let dup = bytes[i];
                bytes.insert(i, dup);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[test]
    fn roundtrip_never_panics_on_mutated_input() {
        // The wire boundary's load-bearing robustness property: `roundtrip` is
        // the parse+emit probe the Python contract test drives over untrusted
        // JSON. It must ALWAYS terminate as Ok/Err — never unwind — for every
        // catalog name against arbitrarily-corrupted bytes, or a malformed
        // payload could crash the binding instead of erroring cleanly.
        let names: Vec<&str> = schema_catalog().into_iter().map(|(n, _)| n).collect();
        let mut rng = Rng::new(0xDEAD_BEEF_CAFE_F00D);
        for name in &names {
            for seed in fuzz_seeds() {
                for _ in 0..40 {
                    let mutated = mutate(seed, &mut rng);
                    // Must not panic; the result (Ok or Err) is irrelevant.
                    let _ = roundtrip(name, &mutated);
                }
            }
        }
    }

    #[test]
    fn roundtrip_unknown_name_is_err() {
        // The `_ =>` arm: an unrecognized type name is a clean Err, not a
        // panic — the binding dispatches names dynamically.
        assert!(roundtrip("NoSuchType", "{}").is_err());
        assert!(roundtrip("", "{}").is_err());
        // Case sensitivity: the catalog keys are exact.
        assert!(roundtrip("note", "{}").is_err());
    }

    #[test]
    fn status_tagged_union_rejects_missing_unknown_and_underspecified_tag() {
        // Every status/op-tagged union is exhaustively parsed by an old client;
        // a payload that doesn't name a known variant, or names one but omits
        // its payload, MUST error rather than silently parse to a wrong shape.

        // Missing tag entirely.
        assert!(roundtrip("UpsertNoteResult", r#"{"id":1}"#).is_err());
        // Unknown tag value.
        assert!(roundtrip("UpsertNoteResult", r#"{"status":"frobnicated","id":1}"#).is_err());
        // Known tag, payload field missing (created needs id).
        assert!(roundtrip("UpsertNoteResult", r#"{"status":"created"}"#).is_err());
        // Known tag, wrong-typed payload field.
        assert!(roundtrip("UpsertNoteResult", r#"{"status":"created","id":"seven"}"#).is_err());

        // NoteTypeResult (created/updated need id AND name).
        assert!(roundtrip("NoteTypeResult", r#"{"id":1,"name":"B"}"#).is_err());
        assert!(roundtrip("NoteTypeResult", r#"{"status":"created","id":1}"#).is_err());
        assert!(roundtrip(
            "NoteTypeResult",
            r#"{"status":"deleted","id":1,"name":"B"}"#
        )
        .is_err());

        // op-tagged FieldOp / TemplateOp.
        assert!(roundtrip("FieldOp", r#"{"name":"F"}"#).is_err());
        assert!(roundtrip("FieldOp", r#"{"op":"obliterate","name":"F"}"#).is_err());
        assert!(roundtrip("FieldOp", r#"{"op":"rename","name":"F"}"#).is_err()); // needs new_name
        assert!(roundtrip("TemplateOp", r#"{"op":"add","name":"T"}"#).is_err()); // needs front/back
        assert!(roundtrip("TemplateOp", r#"{"op":"reposition","name":"T"}"#).is_err()); // needs position

        // state-tagged EmbeddingStatus / IndexStatus.
        assert!(roundtrip("EmbeddingStatus", r#"{"available":true}"#).is_err());
        assert!(roundtrip("EmbeddingStatus", r#"{"state":"melting"}"#).is_err());
        assert!(roundtrip("IndexStatus", r#"{"state":"building","available":true}"#).is_err()); // needs progress

        // delivery-tagged ExportPackageResponse.
        assert!(roundtrip("ExportPackageResponse", r#"{"note_count":1}"#).is_err());
        assert!(roundtrip(
            "ExportPackageResponse",
            r#"{"delivery":"path","note_count":1,"bytes":2,"format":"apkg"}"#
        )
        .is_err()); // path variant needs `path`
    }

    #[test]
    fn untagged_stop_response_discriminates_on_stopped_literal() {
        // StopResponse is untagged and self-selects on the LiteralTrue/
        // LiteralFalse `stopped`. Succeeded defaults `stopped` to true, so a
        // bare object matches Succeeded (the intended "already stopped" read).
        // The adversarial edges are the cases that match NEITHER variant:
        //
        //  - stopped:false but no reason  -> not Succeeded (false ≠ true literal),
        //    not Failed (reason missing). Must error.
        assert!(roundtrip("StopResponse", r#"{"stopped":false}"#).is_err());
        //  - stopped:7 (non-bool)         -> matches neither literal-bool arm.
        assert!(roundtrip("StopResponse", r#"{"stopped":7}"#).is_err());
        //  - stopped:"false" (string)     -> not a bool, matches neither.
        assert!(roundtrip("StopResponse", r#"{"stopped":"false"}"#).is_err());
        // The well-formed Failed payload (stopped:false + reason) DOES parse.
        assert!(roundtrip("StopResponse", r#"{"stopped":false,"reason":"x"}"#).is_ok());
    }

    #[test]
    fn serde_default_fields_deserialize_to_documented_defaults() {
        // Omitting a `#[serde(default)]` field must yield the default the
        // default_* fns document — the binding and search path rely on these
        // exact values when an older/partial payload omits the key.

        // SubstringInfo.source -> "field" (default_source_field).
        let s: SubstringInfo = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(s.source, "field");
        assert!(s.matched_fields.is_empty());
        assert_eq!(s.snippet, None);

        // ListNotesResponse.limit -> 50 (default_limit).
        let l: ListNotesResponse = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(l.limit, 50);

        // NoteTypeInfo.type -> "standard" (default_standard).
        let nt: NoteTypeInfo = serde_json::from_str(r#"{"name":"B","id":1}"#).unwrap();
        assert_eq!(nt.r#type, "standard");

        // CollectionPruneResponse.dry_run -> true (default_true) — a missing
        // flag must default to the SAFE (preview) value, never to writing.
        let p: CollectionPruneResponse = serde_json::from_str(r#"{}"#).unwrap();
        assert!(p.dry_run);

        // ServerStatus.collection_held -> true (default_true).
        // SearchResponse.completeness -> Full (the safe "final answer" default).
        let sr: SearchResponse = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(sr.completeness, Completeness::Full);
        assert!(!sr.stale);
    }

    #[test]
    fn option_field_accepts_explicit_null_and_absent_identically() {
        // For Option fields, explicit `null` and an absent key must both yield
        // None — the model_dump wire emits null, older payloads omit; the
        // binding must treat them the same.
        let absent: SubstringInfo = serde_json::from_str(r#"{}"#).unwrap();
        let explicit: SubstringInfo =
            serde_json::from_str(r#"{"snippet":null,"ref":null}"#).unwrap();
        assert_eq!(absent, explicit);
        assert_eq!(explicit.snippet, None);
        assert_eq!(explicit.r#ref, None);

        // A non-Option defaulted Vec: absent and explicit [] coincide.
        let n_absent: Note =
            serde_json::from_str(r#"{"id":1,"note_type":"B","deck":"D","modified":"m"}"#).unwrap();
        let n_empty: Note =
            serde_json::from_str(r#"{"id":1,"note_type":"B","deck":"D","modified":"m","tags":[]}"#)
                .unwrap();
        assert_eq!(n_absent, n_empty);
        assert!(n_absent.tags.is_empty());
    }

    /// The set of property names a schema marks `required`. schemars omits the
    /// key entirely when nothing is required, which itself is the invariant for
    /// all-defaulted types.
    fn required_set(schema: &serde_json::Value) -> std::collections::BTreeSet<String> {
        schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn defaulted_and_optional_fields_are_never_schema_required() {
        // The schema/serde consistency invariant: a `#[serde(default)]` or
        // Option field is tolerated-absent on input, so it MUST NOT appear in
        // the schema's `required` array — else the advertised schema rejects
        // input the deserializer happily accepts (the MCP `tools/list` lie).
        // (defaulted/optional fields per type, asserted absent from required.)
        let cases: &[(&str, &[&str])] = &[
            ("Note", &["tags", "content"]),
            (
                "SubstringInfo",
                &["matched_fields", "snippet", "source", "ref"],
            ),
            (
                "SearchMatch",
                &[
                    "score",
                    "substring",
                    "fuzzy",
                    "provenance",
                    "tags",
                    "content",
                ],
            ),
            ("ListNotesResponse", &["notes", "total", "limit"]),
            ("NoteTypeInfo", &["fields", "type", "detail"]),
            ("StoreMediaItem", &["filename", "data", "url", "path"]),
            (
                "ServerStatus",
                &[
                    "embedding_spaces",
                    "derived",
                    "locking",
                    "collection_held",
                    "dedup",
                    "coverage",
                ],
            ),
            ("CollectionPruneResponse", &["dry_run", "unused_tags"]),
            ("FieldMetadataInput", &["font", "size", "description"]),
        ];
        for (name, defaulted) in cases {
            let schema = schema_of(name);
            let required = required_set(&schema);
            for field in *defaulted {
                assert!(
                    !required.contains(*field),
                    "{name}.{field} is `default`/Option but schema marks it required"
                );
            }
        }
    }

    #[test]
    fn required_fields_are_genuinely_required_at_deserialize() {
        // The converse of the invariant above: a field the schema marks
        // `required` must actually be rejected when absent — schema and serde
        // agree on the mandatory set, not just the optional one.
        let schema = schema_of("Note");
        let required = required_set(&schema);
        assert!(required.contains("id") && required.contains("note_type"));
        // Drop each required field in turn; deserialization must fail.
        assert!(
            serde_json::from_str::<Note>(r#"{"note_type":"B","deck":"D","modified":"m"}"#).is_err()
        );
        assert!(serde_json::from_str::<Note>(r#"{"id":1,"deck":"D","modified":"m"}"#).is_err());
        assert!(
            serde_json::from_str::<Note>(r#"{"id":1,"note_type":"B","modified":"m"}"#).is_err()
        );
        assert!(serde_json::from_str::<Note>(r#"{"id":1,"note_type":"B","deck":"D"}"#).is_err());
    }

    #[test]
    fn catalog_names_are_unique_and_schemas_are_titled_objects() {
        // The catalog is a registry the contract test indexes by name; a
        // duplicate name would silently shadow a type, and a schema without a
        // title can't be matched to its Pydantic mirror.
        let catalog = schema_catalog();
        let mut seen = std::collections::BTreeSet::new();
        for (name, schema) in &catalog {
            assert!(seen.insert(*name), "duplicate catalog name: {name}");
            let v: serde_json::Value = serde_json::from_str(schema).unwrap();
            assert!(v.is_object(), "{name} schema is not an object");
            assert!(
                v.get("title").and_then(|t| t.as_str()).is_some(),
                "{name} schema has no title"
            );
        }
    }

    #[test]
    fn catalog_and_roundtrip_registries_do_not_drift() {
        // schema_catalog! and roundtrip() are generated from the SAME macro
        // invocation, so every catalog name must be a name roundtrip() knows
        // (round-tripping its own emitted default value) and vice versa. A
        // drift here means a type advertised but not parseable, or parseable
        // but unadvertised — both break the Python contract test.
        for (name, _) in schema_catalog() {
            // An unknown name is the ONLY Err that is name-based; feeding `{}`
            // either parses (Ok) or fails on missing fields, but NEVER returns
            // the "unknown schema type" sentinel for a real catalog name.
            let res = roundtrip(name, "{}");
            if let Err(e) = res {
                assert!(
                    !e.contains("unknown schema type"),
                    "catalog name {name} is not known to roundtrip()"
                );
            }
        }
        // And the negative direction: a name NOT in the catalog is unknown.
        assert!(roundtrip("DefinitelyNotACatalogType", "{}")
            .unwrap_err()
            .contains("unknown schema type"));
    }

    #[test]
    fn wire_protocol_version_is_pinned_to_one() {
        // The backstop constant. The Python mirror
        // (`shrike.schemas.WIRE_PROTOCOL_VERSION`) is pinned EQUAL by the
        // schema contract test; bumping one without the other breaks the
        // handshake. Pinned here so a stray edit is a local failure too.
        assert_eq!(WIRE_PROTOCOL_VERSION, 1);
    }

    #[test]
    fn boundary_i64_ids_roundtrip_exactly() {
        // Note ids are Anki epoch-ms timestamps and can be large; the binding
        // must not lose precision at the i64 extremes (a f64 detour would).
        for id in [i64::MIN, i64::MAX, 0, -1, 9_007_199_254_740_993] {
            let note = Note {
                id,
                note_type: "B".into(),
                deck: "D".into(),
                tags: vec![],
                modified: "m".into(),
                content: None,
            };
            let json = serde_json::to_string(&note).unwrap();
            let back: Note = serde_json::from_str(&json).unwrap();
            assert_eq!(back.id, id, "i64 id must survive the wire exactly");
        }
        // An integer past i64::MAX must be REJECTED, not silently saturated.
        assert!(roundtrip(
            "Note",
            r#"{"id":9223372036854775808,"note_type":"B","deck":"D","modified":"m"}"#
        )
        .is_err());
    }

    #[test]
    fn nonfinite_f64_score_serializes_to_null_not_a_crash() {
        // serde_json cannot represent NaN/Infinity in JSON: it emits `null`
        // (the score is Option<f64>, so null deserializes back to None). Pin
        // this behavior — the search path's fused score may go non-finite, and
        // it must degrade to "no score" on the wire rather than panic or emit
        // invalid JSON that the Python side can't parse.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let m = SearchMatch {
                note: Note {
                    id: 1,
                    note_type: "B".into(),
                    deck: "D".into(),
                    tags: vec![],
                    modified: "m".into(),
                    content: None,
                },
                score: Some(bad),
                substring: None,
                fuzzy: None,
                provenance: vec![],
            };
            let json = serde_json::to_string(&m).unwrap();
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(value["score"].is_null(), "non-finite score must emit null");
            let back: SearchMatch = serde_json::from_str(&json).unwrap();
            assert_eq!(back.score, None, "null score round-trips to None");
        }
        // The `Infinity`/`NaN` JSON literals are NOT valid JSON on input —
        // they must be a clean parse Err, not accepted.
        assert!(roundtrip(
            "SearchMatch",
            r#"{"id":1,"note_type":"B","deck":"D","tags":[],"modified":"m","content":null,"score":Infinity}"#
        )
        .is_err());
    }

    #[test]
    fn literal_const_fields_reject_off_value_on_the_wire() {
        // ServerStatus.running is LiteralTrue; ReloadResponse.status is the
        // "reloaded" const. A payload carrying the wrong constant must error —
        // these fields are how a client confirms it's talking to the right
        // endpoint shape.
        assert!(roundtrip(
            "ReloadResponse",
            r#"{"status":"reloaded","col_mod":1,"rebuilding":false}"#
        )
        .is_ok());
        assert!(roundtrip(
            "ReloadResponse",
            r#"{"status":"reopened","col_mod":1,"rebuilding":false}"#
        )
        .is_err());
        // EmbeddingStatus::Stopped.available is LiteralFalse: an explicit true
        // contradicts the variant and must fail.
        assert!(roundtrip("EmbeddingStatus", r#"{"state":"stopped","available":true}"#).is_err());
        assert!(roundtrip(
            "EmbeddingStatus",
            r#"{"state":"stopped","available":false}"#
        )
        .is_ok());
    }
}
