//! The collection contract (#389 PR B): the typed surface the kernel and the
//! host bindings drive — anki never leaks through it (the canonical impl,
//! `shrike-collection`'s `CollectionCore`, keeps its protobuf adapter
//! private). Sequenced after #391 so every method speaks shrike-schemas
//! types, not JSON strings.

use std::collections::BTreeMap;

use serde_json::Value;
use shrike_error::NativeResult;
use shrike_schemas::{
    CollectionCheckResponse, CollectionInfo, CollectionPruneResponse, DeckInput,
    DeleteDecksResponse, DeleteMediaResponse, DeleteNoteTypeResult, FieldMetadataInput, FieldOp,
    FindReplaceNoteTypesResponse, ListMediaResponse, ListNotesResponse, MediaFetchResult,
    MigrateNoteTypeResponse, NoteInput, NoteTypeInput, NoteTypeResult, RenameTagResponse,
    StoreMediaResult, TemplateOp, UpdateNoteTagsResponse, UpdateNoteTypeFieldMetadataResponse,
    UpdateNoteTypeFieldsResponse, UpdateNoteTypeTemplatesResponse, UpsertDeckResult,
    UpsertNoteResult,
};

/// What `create_note` does about a first-field duplicate (the #77 policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicatePolicy {
    Error,
    Skip,
    Allow,
}

impl DuplicatePolicy {
    pub fn parse(s: &str) -> NativeResult<Self> {
        match s {
            "error" => Ok(Self::Error),
            "skip" => Ok(Self::Skip),
            "allow" => Ok(Self::Allow),
            other => Err(shrike_error::NativeError::invalid_input(format!(
                "on_duplicate must be error/skip/allow (got {other:?})"
            ))),
        }
    }
}

/// The per-note outcome of `create_note` (the upsert result union's spine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateOutcome {
    Created(i64),
    SkippedDuplicate,
}

/// The GUID-conflict / update condition for an imported note or notetype (#72)
/// — mirrors `anki_proto::import_export::ImportAnkiPackageUpdateCondition`. An
/// imported note with the same GUID as an existing one: `IfNewer` updates it
/// only when the incoming note is newer; `Always` always overwrites; `Never`
/// keeps the existing (skips the import). Brand-new notes always add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportUpdateCondition {
    IfNewer = 0,
    Always = 1,
    Never = 2,
}

impl ImportUpdateCondition {
    /// Parse the host's condition string (`if_newer`/`always`/`never`).
    pub fn parse(s: &str) -> NativeResult<Self> {
        match s {
            "if_newer" => Ok(Self::IfNewer),
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            other => Err(shrike_error::NativeError::invalid_input(format!(
                "update condition must be if_newer/always/never (got {other:?})"
            ))),
        }
    }
}

/// The import conflict/merge knobs Shrike exposes (#72). Defaults match anki
/// desktop and Shrike's authoring posture: same-GUID notes/notetypes update
/// only IF_NEWER, scheduling is NOT imported (Shrike manages cards, it does not
/// review), notetypes are not merged by name. `with_deck_configs` is deferred
/// (always false, not exposed) — so it is not a field here.
#[derive(Debug, Clone, Copy)]
pub struct ImportOptions {
    pub update_notes: ImportUpdateCondition,
    pub update_notetypes: ImportUpdateCondition,
    pub with_scheduling: bool,
    pub merge_notetypes: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            update_notes: ImportUpdateCondition::IfNewer,
            update_notetypes: ImportUpdateCondition::IfNewer,
            with_scheduling: false,
            merge_notetypes: false,
        }
    }
}

/// Per-bucket counts from an import (#72) — the summary of anki's
/// `ImportResponse.Log`. Counts, not note-id lists (the lists are too noisy for
/// the tool response; the buckets are what a caller acts on).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportSummary {
    pub new: usize,
    pub updated: usize,
    pub duplicate: usize,
    pub conflicting: usize,
    pub first_field_match: usize,
    pub missing_notetype: usize,
    pub missing_deck: usize,
    pub empty_first_field: usize,
    pub found_notes: usize,
}

impl ImportSummary {
    /// The JSON the binding hands the host (field names match the
    /// `ImportPackageResponse` wire model). Built with `serde_json::json!` so
    /// no `serde` derive dependency is needed.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "new": self.new,
            "updated": self.updated,
            "duplicate": self.duplicate,
            "conflicting": self.conflicting,
            "first_field_match": self.first_field_match,
            "missing_notetype": self.missing_notetype,
            "missing_deck": self.missing_deck,
            "empty_first_field": self.empty_first_field,
            "found_notes": self.found_notes,
        })
    }
}

/// One note as the collection serves it: id, type, raw fields, tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceNote {
    pub id: i64,
    pub notetype_id: i64,
    pub fields: Vec<String>,
    pub tags: Vec<String>,
}

/// One note's full raw field map: `(note_id, names, values)` — owned names
/// because this is the pyo3 wire shape the binding hands across.
pub type OwnedFieldRow = (i64, Vec<String>, Vec<String>);

/// One store_media item after the kernel's off-actor prepare (#490): byte
/// sources arrive fetched/decoded; `path` items pass through whole (their
/// gates are collection policy and run under the write); a failed prepare
/// carries its per-item error.
pub struct PreparedMedia {
    pub index: i64,
    /// The caller's `filename`, echoed on errors.
    pub filename: Option<String>,
    pub source: PreparedMediaSource,
}

pub enum PreparedMediaSource {
    /// Decoded base64 or a completed download; `name` already folds the
    /// URL-derived fallback.
    Bytes {
        name: String,
        data: Vec<u8>,
        content_type: Option<String>,
    },
    /// A server-local path item, gated under the write.
    Path { path: String },
    /// The prepare failed (bad base64, refused/failed download, invalid
    /// item); stored nothing.
    Failed { error: String },
}

/// The package format an export writes (#71). `.apkg` is the scoped,
/// shareable note package; `.colpkg` is a whole-collection backup (no scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageFormat {
    Apkg,
    Colpkg,
}

/// What an export covers (#71): the whole collection, one deck (by the
/// deck-ref convention — name / numeric id / `#id`), or an explicit note set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportScope {
    Whole,
    Deck(String),
    Notes(Vec<i64>),
}

/// One export request, resolved at the op layer (#71). The host has already
/// gated `out_path` (the path-safety check) before this reaches the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRequest {
    pub out_path: String,
    pub format: PackageFormat,
    pub scope: ExportScope,
    /// Include review/scheduling data (and, bound to it, deck configs). Ignored
    /// for `.colpkg` (a full backup always carries its scheduling).
    pub with_scheduling: bool,
    /// Bundle referenced media into the package.
    pub with_media: bool,
    /// Emit the legacy (pre-2.1.50) package format. Default false.
    pub legacy: bool,
}

/// The export outcome (#71): notes written + the on-disk path the package
/// landed at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOutcome {
    pub note_count: u32,
    pub out_path: String,
}

/// The collection store — Shrike's op layer over the note/deck/media state
/// of record. The canonical impl wraps anki via its protobuf service layer;
/// a remote impl proxies the same ops to a server that does.
///
/// Scheduling is the KERNEL's: every call runs on its collection task-actor
/// (FIFO by construction), so impls may block inside methods. The
/// `release`/`ensure_open`/`reopen` trio is the cooperative idle-release
/// lifecycle (#64); a store with no lock to share may no-op `release` and
/// report `ensure_open` = false.
pub trait Collection: Send + Sync {
    // ── lifecycle ────────────────────────────────────────────────────────
    fn close(&self) -> NativeResult<()>;
    /// Release the underlying resource, keeping the instance reusable.
    fn release(&self) -> NativeResult<()>;
    /// Re-acquire if (and only if) idle-released; true = a reopen happened.
    /// Contention surfaces as the BUSY error tier via `reopen`.
    fn ensure_open(&self) -> NativeResult<bool>;
    fn reopen(&self) -> NativeResult<()>;

    // ── reads ────────────────────────────────────────────────────────────
    /// The collection-modified watermark drift detection leans on.
    fn col_mod(&self) -> NativeResult<i64>;
    /// The impl's full search grammar → note ids (read-only).
    fn find_notes(&self, search: &str) -> NativeResult<Vec<i64>>;
    fn notetype_id(&self, name: &str) -> NativeResult<i64>;
    fn get_note(&self, note_id: i64) -> NativeResult<ServiceNote>;
    fn cards_of_note(&self, note_id: i64) -> NativeResult<Vec<i64>>;
    /// `(card_id, template_ordinal)` pairs for one note.
    fn card_ords_of_note(&self, note_id: i64) -> NativeResult<Vec<(i64, i64)>>;
    fn note_count(&self) -> NativeResult<usize>;
    /// Normalized embedding text per note (the `EMBED_TEXT_VERSION` scheme).
    fn note_texts(&self, note_ids: &[i64]) -> NativeResult<Vec<String>>;
    /// `(note_id, embed_text, image_names)` — the embed pipeline's input.
    fn note_embed_inputs(&self, note_ids: &[i64]) -> NativeResult<Vec<(i64, String, Vec<String>)>>;
    /// `(note_id, source, ref, text)` rows for the derived-store build.
    fn derived_field_rows(
        &self,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, String, String)>>;
    /// `(note_id, image_names)` for notes that reference images at all —
    /// the recognition sweep's scoped read (#445).
    fn note_image_refs(&self) -> NativeResult<Vec<(i64, Vec<String>)>>;
    /// `(note_id, sound_names)` for notes that reference `[sound:…]` audio at
    /// all — the ASR recognition sweep's scoped read (#485), the audio twin of
    /// [`Self::note_image_refs`].
    fn note_sound_refs(&self) -> NativeResult<Vec<(i64, Vec<String>)>>;
    /// `(note_id, tags)` for every tagged note (the tag-centroid feed).
    fn note_tag_rows(&self) -> NativeResult<Vec<(i64, Vec<String>)>>;
    /// Whether ANY of the ids carries a tag (the cheap membership probe).
    fn any_tagged(&self, note_ids: &[i64]) -> NativeResult<bool>;
    fn note_field_map(&self, note_ids: &[i64]) -> NativeResult<Vec<OwnedFieldRow>>;
    /// The embedding-text normalization applied to one raw field value.
    fn normalize_text(&self, value: &str) -> NativeResult<String>;
    /// Deck reference (name / id / `#id`) → canonical name; None = an
    /// explicit id matching no deck.
    fn resolve_deck_ref(&self, reference: &str) -> NativeResult<Option<String>>;
    /// The raw search escape hatch, list_notes-shaped.
    fn query(
        &self,
        search: &str,
        with_fields: bool,
        limit: usize,
    ) -> NativeResult<ListNotesResponse>;
    #[allow(clippy::too_many_arguments)]
    fn list_notes(
        &self,
        ids: Option<&[i64]>,
        deck: Option<&str>,
        tags: Option<&[String]>,
        note_type: Option<&str>,
        modified_since: Option<i64>,
        with_fields: bool,
        limit: usize,
    ) -> NativeResult<ListNotesResponse>;
    /// Wire-shaped note dicts (the read actions' assembly unit).
    fn note_dicts(&self, note_ids: &[i64], with_fields: bool) -> NativeResult<Vec<Value>>;
    fn collection_info(
        &self,
        sections: &[String],
        detail_names: &[String],
    ) -> NativeResult<CollectionInfo>;

    // ── note writes ──────────────────────────────────────────────────────
    fn create_note(
        &self,
        notetype_id: i64,
        deck_id: i64,
        fields: &[String],
        tags: &[String],
        policy: DuplicatePolicy,
    ) -> NativeResult<CreateOutcome>;
    fn update_note(
        &self,
        note_id: i64,
        fields: &[String],
        tags: Option<&[String]>,
    ) -> NativeResult<()>;
    fn delete_notes(&self, note_ids: &[i64]) -> NativeResult<usize>;
    fn upsert_notes(
        &self,
        notes: &[NoteInput],
        policy: DuplicatePolicy,
        dry_run: bool,
    ) -> NativeResult<Vec<UpsertNoteResult>>;
    /// Import an `.apkg`/`.colpkg` package (#72). MUTATES the collection (bumps
    /// `col.mod`), so the kernel op MUST follow with a drift reconcile and MUST
    /// NOT advance the index watermark first. Returns per-bucket counts.
    fn import_package(
        &self,
        package_path: &str,
        options: ImportOptions,
    ) -> NativeResult<ImportSummary>;
    /// Anki-grammar find/replace over a note set: the anki-reported change
    /// count plus the diffed changed-id set (kernel-internal maintenance
    /// data — the reindex tail — never the wire).
    #[allow(clippy::too_many_arguments)]
    fn find_replace_notes(
        &self,
        note_ids: &[i64],
        search: &str,
        replacement: &str,
        regex: bool,
        match_case: bool,
        field_name: Option<&str>,
    ) -> NativeResult<(usize, Vec<i64>)>;

    // ── tags + decks ─────────────────────────────────────────────────────
    fn update_note_tags(
        &self,
        note_ids: &[i64],
        set_tags: Option<&[String]>,
        add: &[String],
        remove: &[String],
    ) -> NativeResult<UpdateNoteTagsResponse>;
    fn rename_tag(&self, old: &str, new: &str, note_ids: &[i64])
        -> NativeResult<RenameTagResponse>;
    fn upsert_decks(&self, decks: &[DeckInput]) -> NativeResult<Vec<UpsertDeckResult>>;
    fn delete_decks(&self, refs: &[String]) -> NativeResult<DeleteDecksResponse>;

    // ── note types ───────────────────────────────────────────────────────
    fn upsert_note_types(&self, note_types: &[NoteTypeInput]) -> NativeResult<Vec<NoteTypeResult>>;
    fn update_note_type_fields(
        &self,
        note_type_name: &str,
        operations: &[FieldOp],
    ) -> NativeResult<UpdateNoteTypeFieldsResponse>;
    fn update_note_type_templates(
        &self,
        note_type_name: &str,
        operations: &[TemplateOp],
    ) -> NativeResult<UpdateNoteTypeTemplatesResponse>;
    #[allow(clippy::too_many_arguments)]
    fn find_and_replace_note_types(
        &self,
        note_type_name: &str,
        search: &str,
        replacement: &str,
        regex: bool,
        match_case: bool,
        front: bool,
        back: bool,
        css: bool,
    ) -> NativeResult<FindReplaceNoteTypesResponse>;
    fn update_note_type_field_metadata(
        &self,
        note_type_name: &str,
        updates: &[FieldMetadataInput],
    ) -> NativeResult<UpdateNoteTypeFieldMetadataResponse>;
    fn migrate_note_type(
        &self,
        note_ids: &[i64],
        new_note_type: &str,
        field_map: &BTreeMap<String, String>,
        template_map: &BTreeMap<String, String>,
        dry_run: bool,
    ) -> NativeResult<MigrateNoteTypeResponse>;
    fn delete_note_types(&self, ids: &[i64]) -> NativeResult<Vec<DeleteNoteTypeResult>>;

    // ── media + maintenance ──────────────────────────────────────────────
    fn store_media_bytes(
        &self,
        filename: Option<&str>,
        data: &[u8],
        content_type: Option<&str>,
    ) -> NativeResult<StoreMediaResult>;
    /// The write half of the kernel's re-homed store (#490): byte sources
    /// arrive prepared; `path` items run their gates here.
    fn store_prepared_media(
        &self,
        prepared: &[PreparedMedia],
        path_roots: &[String],
    ) -> NativeResult<Vec<StoreMediaResult>>;
    fn fetch_media(&self, filenames: &[String]) -> NativeResult<Vec<MediaFetchResult>>;
    fn list_media(
        &self,
        pattern: Option<&str>,
        limit: Option<usize>,
    ) -> NativeResult<ListMediaResponse>;
    fn delete_media(&self, filenames: &[String]) -> NativeResult<DeleteMediaResponse>;
    fn media_check(&self) -> NativeResult<CollectionCheckResponse>;
    /// The #89 cleanups; the removed-note-id list rides out of band for the
    /// kernel's sidecar tail, never the wire.
    fn prune(
        &self,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> NativeResult<(CollectionPruneResponse, Vec<i64>)>;
    /// Export the collection (or a scope of it) to an Anki package (#71).
    /// Read-only on the data; holds the collection for the package write, so
    /// the kernel runs it on the actor like every other op. The caller has
    /// already gated `out_path`.
    fn export_package(&self, request: &ExportRequest) -> NativeResult<ExportOutcome>;
}
