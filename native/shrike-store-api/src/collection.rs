//! The collection contract (#389 PR B): the typed surface the kernel and the
//! host bindings drive — anki never leaks through it (the canonical impl,
//! `shrike-collection`'s `CollectionCore`, keeps its protobuf adapter
//! private). Sequenced after #391 so every method speaks shrike-schemas
//! types, not JSON strings.

use std::collections::BTreeMap;

use serde_json::Value;
use shrike_ffi::NativeResult;
use shrike_schemas::{
    CollectionCheckResponse, CollectionInfo, CollectionPruneResponse, DeckInput,
    DeleteDecksResponse, DeleteMediaResponse, DeleteNoteTypeResult, FieldMetadataInput, FieldOp,
    FindReplaceNoteTypesResponse, ListMediaResponse, ListNotesResponse, MediaFetchResult,
    MigrateNoteTypeResponse, NoteInput, NoteTypeInput, NoteTypeResult, RenameTagResponse,
    StoreMediaItem, StoreMediaResult, TemplateOp, UpdateNoteTagsResponse,
    UpdateNoteTypeFieldMetadataResponse, UpdateNoteTypeFieldsResponse,
    UpdateNoteTypeTemplatesResponse, UpsertDeckResult, UpsertNoteResult,
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
            other => Err(shrike_ffi::NativeError::invalid_input(format!(
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

/// One note as the collection serves it: id, type, raw fields, tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceNote {
    pub id: i64,
    pub notetype_id: i64,
    pub fields: Vec<String>,
    pub tags: Vec<String>,
}

/// One note's full raw field map: `(note_id, names, values)`.
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
    /// Anki-grammar find/replace over a note set; returns the apply JSON
    /// (`{"notes_changed", "changed_ids"}`) — the one surface #391 left
    /// stringly, mirrored as-is (typing it is a contract-neutral follow-up).
    #[allow(clippy::too_many_arguments)]
    fn find_replace_notes(
        &self,
        note_ids: &[i64],
        search: &str,
        replacement: &str,
        regex: bool,
        match_case: bool,
        field_name: Option<&str>,
    ) -> NativeResult<String>;

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
    /// The standalone sequential batch (prepare + write per item).
    fn store_media_items(
        &self,
        items: &[StoreMediaItem],
        allow_private_fetch: bool,
        path_roots: &[String],
    ) -> NativeResult<Vec<StoreMediaResult>>;
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
}
