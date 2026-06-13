//! Rust-canonical wire contracts (#330, kernel inversion S1).
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
//! - Pure Rust — NO pyo3 (epic #265 convention 5); bound to Python in shrike-py.

pub mod literals;

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::literals::{LiteralFalse, LiteralTrue, ReloadedLiteral};

/// Stable wire marker prefixing a tool's MCP isError text when the call failed
/// because the collection couldn't be acquired (#65). Mirrors
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
pub struct TemplateInput {
    pub name: String,
    pub front: String,
    pub back: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct NoteInput {
    #[serde(default)]
    pub id: Option<i64>,
    #[serde(default)]
    pub deck: Option<String>,
    #[serde(default)]
    pub note_type: Option<String>,
    #[serde(default)]
    pub fields: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct NoteTypeInput {
    #[serde(default)]
    pub id: Option<i64>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub fields: Option<Vec<String>>,
    #[serde(default)]
    pub templates: Option<Vec<TemplateInput>>,
    #[serde(default)]
    pub css: Option<String>,
    #[serde(default)]
    pub is_cloze: Option<bool>,
}

// ============================================================================
// Shared nested models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Note {
    pub id: i64,
    pub note_type: String,
    pub deck: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub modified: String,
    /// Independent projection: present in "full" mode, omitted in "meta" mode.
    #[serde(default)]
    pub content: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SubstringInfo {
    #[serde(default)]
    pub matched_fields: Vec<String>,
    #[serde(default)]
    pub snippet: Option<String>,
    #[serde(default = "default_source_field")]
    pub source: String,
    #[serde(default)]
    pub r#ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FuzzyMatch {
    pub source: String,
    pub r#ref: String,
    #[serde(default)]
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SignalContribution {
    pub signal: String,
    pub rank: i64,
}

/// A search result with per-mechanism match evidence (Pydantic: `SearchMatch(Note)`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SearchMatch {
    #[serde(flatten)]
    pub note: Note,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub substring: Option<SubstringInfo>,
    #[serde(default)]
    pub fuzzy: Option<FuzzyMatch>,
    #[serde(default)]
    pub provenance: Vec<SignalContribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Neighbor {
    pub id: i64,
    /// Cosine similarity when semantically ranked; None for a lexical-only
    /// near-verbatim hit (#206).
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Which signals surfaced the candidate (#208) — the search-provenance
    /// shape (#182): `text` (semantic) and/or `fuzzy` (lexical overlap).
    #[serde(default)]
    pub provenance: Vec<SignalContribution>,
}

/// One draft note's dedup outcome (#391 phase 1): the attached neighbor
/// candidates plus the calibration sample (`best` semantic cosine, None on
/// no-match) the host's dedup-stats recorder consumes. The internal wire of
/// the kernel's attach-neighbors action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpsertNeighbors {
    #[serde(default)]
    pub neighbors: Vec<Neighbor>,
    #[serde(default)]
    pub best: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct TemplateInfo {
    pub name: String,
    pub front: String,
    pub back: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FieldDetail {
    pub name: String,
    pub font: String,
    pub size: i64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct NoteTypeDetail {
    pub templates: Vec<TemplateInfo>,
    pub css: String,
    #[serde(default)]
    pub fields: Vec<FieldDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct NoteTypeInfo {
    pub name: String,
    pub id: i64,
    #[serde(default)]
    pub fields: Vec<String>,
    #[serde(default = "default_standard")]
    pub r#type: String,
    #[serde(default)]
    pub detail: Option<NoteTypeDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DeckInfo {
    pub name: String,
    pub id: i64,
    #[serde(default)]
    pub note_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Summary {
    pub path: String,
    pub created: String,
    pub modified: String,
    pub notes: i64,
    pub cards: i64,
    pub decks: i64,
    pub note_types: i64,
    pub tags: i64,
    pub due_today: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DeckStat {
    #[serde(default)]
    pub notes: i64,
    #[serde(default)]
    pub due: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Stats {
    #[serde(default)]
    pub total_notes: i64,
    #[serde(default)]
    pub total_cards: i64,
    #[serde(default)]
    pub cards_due_today: i64,
    #[serde(default)]
    pub new_cards: i64,
    #[serde(default)]
    pub decks_summary: BTreeMap<String, DeckStat>,
}

// ============================================================================
// Per-item result variants (discriminated unions)
// ============================================================================

/// Why a candidate note cannot be added (Anki's NoteFieldsCheckResult + the
/// structural problems caught before it).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
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
pub enum UpsertAction {
    Create,
    Update,
}

/// The only skip reason (`on_duplicate="skip"`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    Duplicate,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum UpsertNoteResult {
    Created {
        id: i64,
        #[serde(default)]
        neighbors: Vec<Neighbor>,
        #[serde(default)]
        neighbors_unavailable: bool,
    },
    Updated {
        id: i64,
        #[serde(default)]
        neighbors: Vec<Neighbor>,
        #[serde(default)]
        neighbors_unavailable: bool,
    },
    /// A dry-run outcome: validated, nothing written.
    Ok {
        index: i64,
        action: UpsertAction,
    },
    Skipped {
        index: i64,
        reason: SkipReason,
    },
    Error {
        index: i64,
        error: String,
        #[serde(default)]
        reason: Option<NoteValidationReason>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum NoteTypeResult {
    Created { id: i64, name: String },
    Updated { id: i64, name: String },
    Error { index: i64, error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum FieldOp {
    Add {
        name: String,
        #[serde(default)]
        position: Option<i64>,
    },
    Remove {
        name: String,
    },
    Rename {
        name: String,
        new_name: String,
    },
    Reposition {
        name: String,
        position: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FieldMetadataInput {
    pub name: String,
    #[serde(default)]
    pub font: Option<String>,
    #[serde(default)]
    pub size: Option<i64>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpdateNoteTypeFieldsResponse {
    pub id: i64,
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpdateNoteTypeFieldMetadataResponse {
    pub id: i64,
    pub name: String,
    pub fields_updated: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum TemplateOp {
    Add {
        name: String,
        front: String,
        back: String,
        #[serde(default)]
        position: Option<i64>,
    },
    Remove {
        name: String,
    },
    Rename {
        name: String,
        new_name: String,
    },
    Reposition {
        name: String,
        position: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpdateNoteTypeTemplatesResponse {
    pub id: i64,
    pub name: String,
    pub templates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FindReplaceNoteTypesResponse {
    pub id: i64,
    pub name: String,
    pub replacements: i64,
    pub templates_changed: Vec<String>,
    pub css_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DeckInput {
    #[serde(default)]
    pub id: Option<i64>,
    pub name: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct StoreMediaItem {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

impl StoreMediaItem {
    /// Pydantic's `model_validator`: exactly one source, and `data` needs a
    /// `filename`. Serde can't express cross-field rules, so callers (the S2
    /// action layer) validate explicitly.
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
pub enum UpsertDeckResult {
    Created {
        id: i64,
        name: String,
    },
    Updated {
        id: i64,
        name: String,
    },
    Error {
        index: i64,
        #[serde(default)]
        name: Option<String>,
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DeleteNoteTypeResult {
    Deleted {
        id: i64,
        name: String,
    },
    NotFound {
        id: i64,
    },
    Error {
        id: i64,
        name: String,
        error: String,
    },
}

// ============================================================================
// Tool response models
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CollectionInfo {
    #[serde(default)]
    pub summary: Option<Summary>,
    #[serde(default)]
    pub note_types: Option<Vec<NoteTypeInfo>>,
    #[serde(default)]
    pub decks: Option<Vec<DeckInfo>>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub stats: Option<Stats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ListNotesResponse {
    #[serde(default)]
    pub notes: Vec<Note>,
    #[serde(default)]
    pub total: i64,
    #[serde(default = "default_limit")]
    pub limit: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct MigrateNoteTypeResponse {
    #[serde(default)]
    pub changed: Vec<i64>,
    pub from_note_type: String,
    pub to_note_type: String,
    #[serde(default)]
    pub dropped_fields: Vec<String>,
    #[serde(default)]
    pub new_empty_fields: Vec<String>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SearchResultGroup {
    pub source: String,
    #[serde(default)]
    pub matches: Vec<SearchMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SearchResponse {
    #[serde(default)]
    pub results: Vec<SearchResultGroup>,
    #[serde(default)]
    pub message: Option<String>,
    /// The two-tier live-search contract (#181): "partial" = the
    /// embedding-bearing signals were skipped at the caller's request
    /// (tier="live"); "full" = the final answer for this query/server state.
    #[serde(default)]
    pub completeness: Completeness,
    /// Echo of the request's `version` (client-side stale-response dropping).
    #[serde(default)]
    pub version: Option<i64>,
}

/// `SearchResponse.completeness` (#181) — mirrors the Pydantic
/// `Literal["partial", "full"]`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
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
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpsertNotesResponse {
    #[serde(default)]
    pub results: Vec<UpsertNoteResult>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpsertNoteTypesResponse {
    #[serde(default)]
    pub results: Vec<NoteTypeResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DeleteNotesResponse {
    #[serde(default)]
    pub deleted: Vec<i64>,
    #[serde(default)]
    pub not_found: Vec<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DeleteNoteTypesResponse {
    #[serde(default)]
    pub results: Vec<DeleteNoteTypeResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpdateNoteTagsResponse {
    #[serde(default)]
    pub notes_modified: i64,
    #[serde(default)]
    pub not_found: Vec<i64>,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RenameTagResponse {
    #[serde(default)]
    pub notes_modified: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PruneUnusedTags {
    #[serde(default)]
    pub removed: i64,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PruneEmptyNotes {
    #[serde(default)]
    pub removed: Vec<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PruneEmptyCards {
    #[serde(default)]
    pub cards_removed: i64,
    #[serde(default)]
    pub notes_deleted: Vec<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PruneUnusedMedia {
    #[serde(default)]
    pub removed: i64,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CollectionPruneResponse {
    #[serde(default = "default_true")]
    pub dry_run: bool,
    #[serde(default)]
    pub unused_tags: Option<PruneUnusedTags>,
    #[serde(default)]
    pub empty_notes: Option<PruneEmptyNotes>,
    #[serde(default)]
    pub empty_cards: Option<PruneEmptyCards>,
    #[serde(default)]
    pub unused_media: Option<PruneUnusedMedia>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UpsertDecksResponse {
    #[serde(default)]
    pub results: Vec<UpsertDeckResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DeleteDecksResponse {
    #[serde(default)]
    pub deleted: Vec<String>,
    #[serde(default)]
    pub not_found: Vec<String>,
    #[serde(default)]
    pub not_empty: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StoreMediaResult {
    Stored {
        index: i64,
        filename: String,
        #[serde(default)]
        mime: Option<String>,
        size_bytes: i64,
        #[serde(default)]
        deduped: bool,
    },
    Error {
        index: i64,
        #[serde(default)]
        filename: Option<String>,
        error: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct StoreMediaResponse {
    #[serde(default)]
    pub results: Vec<StoreMediaResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MediaFetchResult {
    Found {
        filename: String,
        #[serde(default)]
        url: Option<String>,
        path: String,
        #[serde(default)]
        mime: Option<String>,
        size_bytes: i64,
    },
    Missing {
        filename: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FetchMediaResponse {
    #[serde(default)]
    pub results: Vec<MediaFetchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct MediaFileInfo {
    pub filename: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub mime: Option<String>,
    pub size_bytes: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ListMediaResponse {
    pub media_dir: String,
    #[serde(default)]
    pub count: i64,
    #[serde(default)]
    pub files: Vec<MediaFileInfo>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DeleteMediaResponse {
    #[serde(default)]
    pub deleted: Vec<String>,
    #[serde(default)]
    pub not_found: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CollectionCheckResponse {
    pub media_dir: String,
    #[serde(default)]
    pub unused: Vec<String>,
    #[serde(default)]
    pub missing: Vec<String>,
    #[serde(default)]
    pub missing_media_notes: Vec<i64>,
    #[serde(default)]
    pub have_trash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FindReplaceSample {
    pub id: i64,
    pub field: String,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FindReplaceResponse {
    #[serde(default)]
    pub notes_changed: i64,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub samples: Vec<FindReplaceSample>,
}

/// The result of importing an `.apkg`/`.colpkg` (#72) — per-bucket note counts
/// from anki's importer (`ImportResponse.Log`) + whether the import reconciled
/// the index. The per-bucket mirror of `shrike.schemas.ImportPackageResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ImportPackageResponse {
    #[serde(default)]
    pub new: i64,
    #[serde(default)]
    pub updated: i64,
    #[serde(default)]
    pub duplicate: i64,
    #[serde(default)]
    pub conflicting: i64,
    #[serde(default)]
    pub first_field_match: i64,
    #[serde(default)]
    pub missing_notetype: i64,
    #[serde(default)]
    pub missing_deck: i64,
    #[serde(default)]
    pub empty_first_field: i64,
    #[serde(default)]
    pub found_notes: i64,
    #[serde(default)]
    pub reindexed: bool,
}

// ============================================================================
// Server status / custom-endpoint models
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchMode {
    Serial,
    Batched,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum EmbeddingStatus {
    Running {
        #[serde(default)]
        available: bool,
        #[serde(default)]
        pid: Option<i64>,
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        provider: Option<String>,
        #[serde(default)]
        batch_safe: Option<bool>,
        #[serde(default)]
        batch: Option<BatchMode>,
        /// The modalities this space embeds (#498/#235).
        #[serde(default)]
        modalities: Option<Vec<String>>,
    },
    Stopped {
        #[serde(default)]
        available: LiteralFalse,
    },
    Failed {
        #[serde(default)]
        available: LiteralFalse,
    },
    NotConfigured {
        #[serde(default)]
        available: LiteralFalse,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct IndexProgress {
    #[serde(default)]
    pub indexed: i64,
    #[serde(default)]
    pub total: i64,
}

/// On-disk index contents shared across build states (Pydantic's `_IndexBase`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct IndexBase {
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub size: i64,
    #[serde(default)]
    pub ndim: Option<i64>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub col_mod: Option<i64>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub activation: Option<BTreeMap<String, BTreeMap<String, f64>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum IndexStatus {
    Unavailable {
        #[serde(flatten)]
        base: IndexBase,
    },
    Building {
        #[serde(flatten)]
        base: IndexBase,
        progress: IndexProgress,
    },
    Ready {
        #[serde(flatten)]
        base: IndexBase,
    },
    Error {
        #[serde(flatten)]
        base: IndexBase,
        error: String,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DerivedState {
    #[default]
    Unavailable,
    Building,
    Ready,
    Error,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DerivedStatus {
    #[serde(default)]
    pub state: DerivedState,
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub fts5: bool,
    #[serde(default)]
    pub size: i64,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub col_mod: Option<i64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LockingMode {
    #[default]
    Permanent,
    Cooperative,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
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

/// One attached recognition engine's self-report (#228/#485). The Rust mirror
/// of `shrike.schemas.RecognitionEngineStatus`; `ServerStatus.recognition` is
/// a map of these keyed by source (`ocr`/`vlm`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RecognitionEngineStatus {
    #[serde(default)]
    pub state: RecognitionState,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct DedupStats {
    #[serde(default)]
    pub samples: i64,
    #[serde(default)]
    pub no_match: i64,
    #[serde(default)]
    pub buckets: Vec<i64>,
}

/// One collection's state in a multi-collection daemon's `/status` (#68): a row
/// per known collection (the boot/default plus every registered profile). The
/// per-collection mirror of `shrike.schemas.CollectionStatus`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CollectionStatus {
    pub name: String,
    pub path: String,
    pub registered: bool,
    #[serde(default)]
    pub is_default: bool,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub held: Option<bool>,
    #[serde(default)]
    pub index_state: Option<String>,
    #[serde(default)]
    pub col_mod: Option<i64>,
}

/// How one (query-modality → target-modality) pair is reachable (#235). The
/// Rust mirror of `shrike.schemas.CoverageCell`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
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

/// One query modality's reachability to each target modality (#235). The Rust
/// mirror of `shrike.schemas.CoverageRow`; every cell is a `CoverageCell` (no
/// absent target, only an `Unavailable` one).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CoverageRow {
    #[serde(default)]
    pub text: CoverageCell,
    #[serde(default)]
    pub image: CoverageCell,
    #[serde(default)]
    pub audio: CoverageCell,
}

/// The cross-modal coverage matrix (#235): per query modality, a `CoverageRow`
/// naming how each target modality is reachable. The Rust mirror of
/// `shrike.schemas.CoverageMatrix`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CoverageMatrix {
    #[serde(default)]
    pub text: CoverageRow,
    #[serde(default)]
    pub image: CoverageRow,
    #[serde(default)]
    pub audio: CoverageRow,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ServerStatus {
    #[serde(default)]
    pub running: LiteralTrue,
    /// The action exchange's protocol version (#392).
    pub wire_protocol_version: u32,
    pub pid: i64,
    pub url: String,
    pub collection: String,
    pub log_level: String,
    pub log_dir: String,
    #[serde(default)]
    pub uptime: Option<String>,
    #[serde(default)]
    pub log: Option<String>,
    pub embedding: EmbeddingStatus,
    pub index: IndexStatus,
    #[serde(default)]
    pub derived: DerivedStatus,
    #[serde(default)]
    pub locking: LockingMode,
    #[serde(default = "default_true")]
    pub collection_held: bool,
    /// Dedup best-match statistics (#207) — None until the first upsert with
    /// neighbors runs.
    #[serde(default)]
    pub dedup: Option<DedupStats>,
    /// Per-engine recognition state (#228/#485): a map keyed by source
    /// (`ocr`/`vlm`), each row {state, backend, fingerprint}. An EMPTY map is
    /// "nothing attached" (distinct from an attached-but-errored engine).
    #[serde(default)]
    pub recognition: std::collections::BTreeMap<String, RecognitionEngineStatus>,
    /// The cross-modal coverage matrix (#498/#235): for each (query, target)
    /// modality pair, how the target is reachable — `native`, `via_derived_text`
    /// (a recognizer derives text from the target into the text space), or
    /// `unavailable`. None on payloads from older servers (the flat shape).
    #[serde(default)]
    pub coverage: Option<CoverageMatrix>,
    /// Multi-collection routing (#68): one row per known collection (the
    /// boot/default plus every registered profile). None on a single-collection
    /// server / older payloads; the top-level fields describe the default
    /// collection (which the operational routes act on).
    #[serde(default)]
    pub collections: Option<Vec<CollectionStatus>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IndexRebuildResponse {
    Started { total: i64 },
    Complete { size: i64 },
    AlreadyBuilding { progress: IndexProgress },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IndexSaveResponse {
    Saved { size: i64, pending: i64 },
    Empty,
    Building { progress: IndexProgress },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EmbeddingStartResponse {
    Started {
        embedding: EmbeddingStatus,
        index: IndexStatus,
    },
    AlreadyRunning {
        embedding: EmbeddingStatus,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EmbeddingStopResponse {
    Stopped { index: IndexStatus },
    NotRunning,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ShutdownResponse {
    pub status: String,
    pub pid: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ReloadResponse {
    #[serde(default)]
    pub status: ReloadedLiteral,
    pub col_mod: i64,
    #[serde(default)]
    pub rebuilding: bool,
}

/// One registered collection profile (#66). The registry name is a friendly
/// handle only — index identity keys on the collection file path, never the
/// name. `is_default` marks the active default (the profile the per-call
/// selector resolves to when none is passed).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ProfileEntry {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub is_default: bool,
}

/// The collection/profile registry enumeration (#66): the registered profiles
/// and the active-default name (`None` when none is set). Read-only — selection
/// as routing is the capstone (#68).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ListProfilesResponse {
    #[serde(default)]
    pub profiles: Vec<ProfileEntry>,
    #[serde(default)]
    pub default: Option<String>,
}

/// The kernel's export-op outcome (#71): the count of notes written and the
/// on-disk path the package landed at. Internal wire — the kernel op returns
/// this; the host action wraps it into [`ExportPackageResponse`] (adding the
/// download `url` / `bytes`, which the kernel doesn't know about).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ExportPackageResult {
    pub note_count: u32,
    pub out_path: String,
}

/// The export tool's response (#71): the package the export produced, handed
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
        note_count: u32,
        bytes: u64,
        /// "apkg" or "colpkg".
        format: String,
        path: String,
    },
    /// The server wrote a temp package and serves it at `url` (GET it; the
    /// temp file is reaped after download / on a TTL / at shutdown).
    Url {
        note_count: u32,
        bytes: u64,
        format: String,
        url: String,
    },
}

/// Discriminated on the bool `stopped` (string tags only in serde, so this is
/// an untagged union whose variants self-select via the literal-bool types).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
pub enum StopResponse {
    Succeeded {
        #[serde(default)]
        stopped: LiteralTrue,
        #[serde(default)]
        pid: Option<i64>,
        #[serde(default)]
        forced: bool,
    },
    Failed {
        stopped: LiteralFalse,
        reason: String,
    },
}

// ============================================================================
// The action exchange error envelope (#505)
// ============================================================================

/// The machine-readable class of an actions-over-HTTP failure (#505).
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
/// `collection_busy` (409, contention under cooperative locking #65 — the op
/// never ran, retryable); `unknown_action` (404, no such action name);
/// `internal_error` (500, an unexpected server bug — detail logged, never
/// returned, so it can't leak to a UI client).
///
/// No per-variant doc comments: schemars then renders a plain string `enum`,
/// matching Pydantic's str-Enum (the contract normalizer compares them).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionErrorCode {
    InputError,
    CollectionBusy,
    UnknownAction,
    InternalError,
}

/// The one error envelope every `POST /actions/{name}` failure returns (#505).
///
/// Defined once here (the wire contract is shrike-schemas verbatim) and mirrored
/// by `shrike.schemas.ActionError`. `message` is a non-leaking human string: for
/// an `internal_error` it is a fixed, generic sentence (the real cause is in the
/// server log); for the caller-actionable codes it carries the actionable text.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ActionError {
    pub code: ActionErrorCode,
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

/// The action exchange's protocol version (#392) — the compatibility story,
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
    ("Neighbor", Neighbor),
    ("UpsertNeighbors", UpsertNeighbors),
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
        let created = UpsertNoteResult::Created {
            id: 7,
            neighbors: vec![],
            neighbors_unavailable: false,
        };
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
        let json = r#"{"id":1,"score":0.9,"tags":[],"brand_new_field":123}"#;
        assert!(serde_json::from_str::<Neighbor>(json).is_ok());
    }
}
