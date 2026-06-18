//! The collection contract: the typed
//! surface the kernel and the host bindings drive — anki never leaks through
//! it (the canonical impl, this crate's `CollectionCore`, keeps its protobuf
//! adapter private). Every method speaks shrike-schemas types, not JSON strings.
//!
//! Lives in `shrike-collection` (not the store-contract crate) because this
//! crate is the SOLE implementer and every consumer (kernel/pyo3/cabi) already
//! depends on it — homing the trait beside its only impl removes the edge a
//! separate contract crate would have forced.

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

/// What `create_note` does about a first-field duplicate (the duplicate policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicatePolicy {
    /// Report the duplicate as an error; do not write the note (the default).
    Error,
    /// Skip the duplicate silently (the note is not written).
    Skip,
    /// Write the note anyway, allowing the duplicate.
    Allow,
}

impl DuplicatePolicy {
    /// Parse the host's policy string (`error`/`skip`/`allow`).
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error if `s` is none of the three accepted
    /// values.
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
    /// The note was written; carries its new note id.
    Created(i64),
    /// A first-field duplicate was skipped under [`DuplicatePolicy::Skip`].
    SkippedDuplicate,
}

/// The GUID-conflict / update condition for an imported note or notetype
/// — mirrors `anki_proto::import_export::ImportAnkiPackageUpdateCondition`. An
/// imported note with the same GUID as an existing one: `IfNewer` updates it
/// only when the incoming note is newer; `Always` always overwrites; `Never`
/// keeps the existing (skips the import). Brand-new notes always add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportUpdateCondition {
    /// Update the existing note/notetype only when the incoming one is newer.
    IfNewer = 0,
    /// Always overwrite the existing note/notetype with the incoming one.
    Always = 1,
    /// Keep the existing note/notetype; skip the incoming one.
    Never = 2,
}

impl ImportUpdateCondition {
    /// Parse the host's condition string (`if_newer`/`always`/`never`).
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error if `s` is none of the three accepted
    /// values.
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

/// The import conflict/merge knobs Shrike exposes. Defaults match anki
/// desktop and Shrike's authoring posture: same-GUID notes/notetypes update
/// only IF_NEWER, scheduling is NOT imported (Shrike manages cards, it does not
/// review), notetypes are not merged by name. `with_deck_configs` is deferred
/// (always false, not exposed) — so it is not a field here.
#[derive(Debug, Clone, Copy)]
pub struct ImportOptions {
    /// How to resolve a same-GUID note already in the collection.
    pub update_notes: ImportUpdateCondition,
    /// How to resolve a same-GUID notetype already in the collection.
    pub update_notetypes: ImportUpdateCondition,
    /// Import the package's review/scheduling data (default false — Shrike
    /// manages cards, it does not review).
    pub with_scheduling: bool,
    /// Merge notetypes by name rather than treating them as distinct.
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

/// Per-bucket counts from an import — the summary of anki's
/// `ImportResponse.Log`. Counts, not note-id lists (the lists are too noisy for
/// the tool response; the buckets are what a caller acts on).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportSummary {
    /// Brand-new notes added.
    pub new: usize,
    /// Existing notes updated (a same-GUID match the condition allowed).
    pub updated: usize,
    /// Notes skipped as duplicates (identical to an existing note).
    pub duplicate: usize,
    /// Notes skipped because of a same-GUID conflict the condition rejected.
    pub conflicting: usize,
    /// Notes skipped because their first field matched an existing note.
    pub first_field_match: usize,
    /// Notes skipped because their notetype was missing from the package.
    pub missing_notetype: usize,
    /// Notes skipped because their deck was missing from the package.
    pub missing_deck: usize,
    /// Notes skipped because their first field was empty.
    pub empty_first_field: usize,
    /// Total notes found in the package.
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
    /// The note id.
    pub id: i64,
    /// The id of the note's notetype.
    pub notetype_id: i64,
    /// The raw field values, in field order.
    pub fields: Vec<String>,
    /// The note's tags.
    pub tags: Vec<String>,
}

/// One note's full raw field map: `(note_id, names, values)` — owned names
/// because this is the pyo3 wire shape the binding hands across.
pub type OwnedFieldRow = (i64, Vec<String>, Vec<String>);

// `PreparedMedia`/`PreparedMediaSource` — the interface between the inbound
// "acquire + validate untrusted bytes" half (`shrike-media`) and this
// crate's store-write tail (`store_prepared_media`). Defined in shrike-media
// (the floor crate the kernel fans the prepare onto) and re-exported here so
// `shrike_collection::PreparedMedia` keeps working for every consumer.
pub use shrike_media::{PreparedMedia, PreparedMediaSource};

/// The package format an export writes. `.apkg` is the scoped,
/// shareable note package; `.colpkg` is a whole-collection backup (no scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageFormat {
    /// `.apkg` — the scoped, shareable note package.
    Apkg,
    /// `.colpkg` — a whole-collection backup (no scope).
    Colpkg,
}

/// What an export covers: the whole collection, one deck (by the
/// deck-ref convention — name / numeric id / `#id`), or an explicit note set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportScope {
    /// The whole collection.
    Whole,
    /// One deck, by the deck-ref convention (name / numeric id / `#id`).
    Deck(String),
    /// An explicit set of note ids.
    Notes(Vec<i64>),
}

/// One export request, resolved at the op layer. The host has already
/// gated `out_path` (the path-safety check) before this reaches the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRequest {
    /// The on-disk path the package is written to (host-gated before this).
    pub out_path: String,
    /// The package format to write.
    pub format: PackageFormat,
    /// What the export covers.
    pub scope: ExportScope,
    /// Include review/scheduling data (and, bound to it, deck configs). Ignored
    /// for `.colpkg` (a full backup always carries its scheduling).
    pub with_scheduling: bool,
    /// Bundle referenced media into the package.
    pub with_media: bool,
    /// Emit the legacy (pre-2.1.50) package format. Default false.
    pub legacy: bool,
}

/// The export outcome: notes written + the on-disk path the package
/// landed at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOutcome {
    /// The number of notes written into the package.
    pub note_count: u32,
    /// The on-disk path the package landed at.
    pub out_path: String,
}

/// The collection store — Shrike's op layer over the note/deck/media state
/// of record. The canonical impl wraps anki via its protobuf service layer;
/// a remote impl proxies the same ops to a server that does.
///
/// Scheduling is the KERNEL's: every call runs on its collection task-actor
/// (FIFO by construction), so impls may block inside methods. The
/// `release`/`ensure_open`/`reopen` trio is the cooperative idle-release
/// lifecycle; a store with no lock to share may no-op `release` and
/// report `ensure_open` = false.
pub trait Collection: Send + Sync {
    // ── lifecycle ────────────────────────────────────────────────────────
    /// Close the collection, releasing its lock for good.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying store fails to close cleanly.
    fn close(&self) -> NativeResult<()>;
    /// Release the underlying resource, keeping the instance reusable.
    ///
    /// # Errors
    ///
    /// Returns an error if releasing the underlying resource fails.
    fn release(&self) -> NativeResult<()>;
    /// Re-acquire if (and only if) idle-released; true = a reopen happened.
    /// Contention surfaces as the BUSY error tier via `reopen`.
    ///
    /// # Errors
    ///
    /// Returns the BUSY error tier (via `reopen`) when another process holds
    /// the collection, or any other re-acquire failure.
    fn ensure_open(&self) -> NativeResult<bool>;
    /// Re-open the collection after an idle release.
    ///
    /// # Errors
    ///
    /// Returns the BUSY error tier when another process holds the collection,
    /// or any other open failure.
    fn reopen(&self) -> NativeResult<()>;

    // ── reads ────────────────────────────────────────────────────────────
    /// The collection-modified watermark drift detection leans on.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not open or the read fails.
    fn col_mod(&self) -> NativeResult<i64>;
    /// The impl's full search grammar → note ids (read-only).
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed search expression, or if the read
    /// fails.
    fn find_notes(&self, search: &str) -> NativeResult<Vec<i64>>;
    /// Resolve a notetype name to its id.
    ///
    /// # Errors
    ///
    /// Returns an error if no notetype has that name, or the read fails.
    fn notetype_id(&self, name: &str) -> NativeResult<i64>;
    /// Read one note by id.
    ///
    /// # Errors
    ///
    /// Returns an error if no note has that id, or the read fails.
    fn get_note(&self, note_id: i64) -> NativeResult<ServiceNote>;
    /// The card ids belonging to one note.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn cards_of_note(&self, note_id: i64) -> NativeResult<Vec<i64>>;
    /// `(card_id, template_ordinal)` pairs for one note.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn card_ords_of_note(&self, note_id: i64) -> NativeResult<Vec<(i64, i64)>>;
    /// The total note count in the collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn note_count(&self) -> NativeResult<usize>;
    /// Normalized embedding text per note (the `EMBED_TEXT_VERSION` scheme).
    ///
    /// # Errors
    ///
    /// Returns an error if any id is missing or the read/normalization fails.
    fn note_texts(&self, note_ids: &[i64]) -> NativeResult<Vec<String>>;
    /// `(note_id, embed_text, image_names)` — the embed pipeline's input.
    ///
    /// # Errors
    ///
    /// Returns an error if any id is missing or the read fails.
    fn note_embed_inputs(&self, note_ids: &[i64]) -> NativeResult<Vec<(i64, String, Vec<String>)>>;
    /// `(note_id, source, ref, text)` rows for the derived-store build.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn derived_field_rows(
        &self,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, String, String)>>;
    /// `(note_id, image_names)` for notes that reference images at all —
    /// the recognition sweep's scoped read.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn note_image_refs(&self) -> NativeResult<Vec<(i64, Vec<String>)>>;
    /// `(note_id, sound_names)` for notes that reference `[sound:…]` audio at
    /// all — the ASR recognition sweep's scoped read, the audio twin of
    /// [`Self::note_image_refs`].
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn note_sound_refs(&self) -> NativeResult<Vec<(i64, Vec<String>)>>;
    /// `(note_id, tags)` for every tagged note (the tag-centroid feed).
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn note_tag_rows(&self) -> NativeResult<Vec<(i64, Vec<String>)>>;
    /// Whether ANY of the ids carries a tag (the cheap membership probe).
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn any_tagged(&self, note_ids: &[i64]) -> NativeResult<bool>;
    /// The full raw field map for a set of notes.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn note_field_map(&self, note_ids: &[i64]) -> NativeResult<Vec<OwnedFieldRow>>;
    /// The embedding-text normalization applied to one raw field value.
    ///
    /// # Errors
    ///
    /// Returns an error if normalization fails.
    fn normalize_text(&self, value: &str) -> NativeResult<String>;
    /// Deck reference (name / id / `#id`) → canonical name; None = an
    /// explicit id matching no deck.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn resolve_deck_ref(&self, reference: &str) -> NativeResult<Option<String>>;
    /// The raw search escape hatch, list_notes-shaped.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed search expression, or if the read
    /// fails.
    fn query(
        &self,
        search: &str,
        with_fields: bool,
        limit: usize,
    ) -> NativeResult<ListNotesResponse>;
    /// Filter notes by the structured criteria (all ANDed), list-shaped.
    ///
    /// # Errors
    ///
    /// Returns an error if a referenced deck/notetype is unknown, or the read
    /// fails.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn note_dicts(&self, note_ids: &[i64], with_fields: bool) -> NativeResult<Vec<Value>>;
    /// The collection structure/stats for the requested sections.
    ///
    /// # Errors
    ///
    /// Returns an error if an unknown section/detail name is requested, or the
    /// read fails.
    fn collection_info(
        &self,
        sections: &[String],
        detail_names: &[String],
    ) -> NativeResult<CollectionInfo>;

    // ── note writes ──────────────────────────────────────────────────────
    /// Create one note under the duplicate policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the note is structurally invalid (empty first
    /// field, broken cloze), if it is a first-field duplicate under
    /// [`DuplicatePolicy::Error`], or if the write fails.
    fn create_note(
        &self,
        notetype_id: i64,
        deck_id: i64,
        fields: &[String],
        tags: &[String],
        policy: DuplicatePolicy,
    ) -> NativeResult<CreateOutcome>;
    /// Update one note's fields (and, optionally, tags).
    ///
    /// # Errors
    ///
    /// Returns an error if the note does not exist, the fields do not match
    /// the notetype, or the write fails.
    fn update_note(
        &self,
        note_id: i64,
        fields: &[String],
        tags: Option<&[String]>,
    ) -> NativeResult<()>;
    /// Permanently delete notes by id; returns the count removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn delete_notes(&self, note_ids: &[i64]) -> NativeResult<usize>;
    /// Create or update notes in bulk under the duplicate policy, with a
    /// per-item result union (one failure does not sink the batch).
    ///
    /// # Errors
    ///
    /// Returns an error only for a whole-batch failure (e.g. the collection is
    /// not open); per-note validation/duplicate failures ride the returned
    /// result union, not this `Result`.
    fn upsert_notes(
        &self,
        notes: &[NoteInput],
        policy: DuplicatePolicy,
        dry_run: bool,
    ) -> NativeResult<Vec<UpsertNoteResult>>;
    /// Import an `.apkg`/`.colpkg` package. MUTATES the collection (bumps
    /// `col.mod`), so the kernel op MUST follow with a drift reconcile and MUST
    /// NOT advance the index watermark first. Returns per-bucket counts.
    ///
    /// # Errors
    ///
    /// Returns an error if the package is missing/unreadable/malformed, or the
    /// import fails.
    fn import_package(
        &self,
        package_path: &str,
        options: ImportOptions,
    ) -> NativeResult<ImportSummary>;
    /// Anki-grammar find/replace over a note set: the anki-reported change
    /// count plus the diffed changed-id set (kernel-internal maintenance
    /// data — the reindex tail — never the wire).
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid regex (when `regex`), or if the write
    /// fails.
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
    /// Edit tags on a note set: `set_tags` (replace) XOR `add`/`remove`.
    ///
    /// # Errors
    ///
    /// Returns an error if `set_tags` is combined with `add`/`remove`, or if
    /// the write fails.
    fn update_note_tags(
        &self,
        note_ids: &[i64],
        set_tags: Option<&[String]>,
        add: &[String],
        remove: &[String],
    ) -> NativeResult<UpdateNoteTagsResponse>;
    /// Rename a tag collection-wide or on a note set (exact match).
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn rename_tag(&self, old: &str, new: &str, note_ids: &[i64])
        -> NativeResult<RenameTagResponse>;
    /// Create or rename/reparent decks in bulk (id = rename); decks never
    /// merge.
    ///
    /// # Errors
    ///
    /// Returns an error if a rename targets an existing deck name, or the
    /// write fails (per-deck outcomes ride the returned result vec).
    fn upsert_decks(&self, decks: &[DeckInput]) -> NativeResult<Vec<UpsertDeckResult>>;
    /// Delete decks by ref, only if empty (else reported `not_empty`).
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn delete_decks(&self, refs: &[String]) -> NativeResult<DeleteDecksResponse>;

    // ── note types ───────────────────────────────────────────────────────
    /// Create or update notetype definitions in bulk (position-replace).
    ///
    /// # Errors
    ///
    /// Returns an error if a definition is invalid, if an existing field name
    /// would move position (use the by-identity tools instead), or the write
    /// fails.
    fn upsert_note_types(&self, note_types: &[NoteTypeInput]) -> NativeResult<Vec<NoteTypeResult>>;
    /// Edit a notetype's fields by name (add/remove/rename/reposition).
    ///
    /// # Errors
    ///
    /// Returns an error if the notetype is unknown, the op sequence is unsound
    /// (validated against a simulated name list before any primitive runs), or
    /// the write fails.
    fn update_note_type_fields(
        &self,
        note_type_name: &str,
        operations: &[FieldOp],
    ) -> NativeResult<UpdateNoteTypeFieldsResponse>;
    /// Edit a notetype's card templates by name (add/remove/rename/reposition).
    ///
    /// # Errors
    ///
    /// Returns an error if the notetype is unknown, the op sequence is unsound
    /// (validated before any primitive runs), or the write fails.
    fn update_note_type_templates(
        &self,
        note_type_name: &str,
        operations: &[TemplateOp],
    ) -> NativeResult<UpdateNoteTypeTemplatesResponse>;
    /// Find/replace text in one notetype's template HTML + CSS.
    ///
    /// # Errors
    ///
    /// Returns an error if the notetype is unknown, the regex is invalid (when
    /// `regex`), or the write fails.
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
    /// Set a notetype's per-field editor metadata (font/size/description).
    ///
    /// # Errors
    ///
    /// Returns an error if the notetype or a named field is unknown, or the
    /// write fails.
    fn update_note_type_field_metadata(
        &self,
        note_type_name: &str,
        updates: &[FieldMetadataInput],
    ) -> NativeResult<UpdateNoteTypeFieldMetadataResponse>;
    /// Change notes' notetype via a field/template name map (#migrate).
    ///
    /// # Errors
    ///
    /// Returns an error if the target notetype is unknown, the maps name
    /// unknown fields/templates, two sources map to one target, the notes do
    /// not all share one source type, or the write fails.
    fn migrate_note_type(
        &self,
        note_ids: &[i64],
        new_note_type: &str,
        field_map: &BTreeMap<String, String>,
        template_map: &BTreeMap<String, String>,
        dry_run: bool,
    ) -> NativeResult<MigrateNoteTypeResponse>;
    /// Delete notetypes by id, only if unused (per-id outcomes in the result).
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn delete_note_types(&self, ids: &[i64]) -> NativeResult<Vec<DeleteNoteTypeResult>>;

    // ── media + maintenance ──────────────────────────────────────────────
    /// Store one media file from bytes (Anki resolves dedup/collisions).
    ///
    /// # Errors
    ///
    /// Returns an error if `filename` is missing/extensionless, or the write
    /// fails.
    fn store_media_bytes(
        &self,
        filename: Option<&str>,
        data: &[u8],
        content_type: Option<&str>,
    ) -> NativeResult<StoreMediaResult>;
    /// The write half of the kernel's re-homed store: byte sources
    /// arrive prepared; `path` items run their gates here.
    ///
    /// # Errors
    ///
    /// Returns an error for a whole-batch failure; per-item failures (a
    /// blocked `path`, a write error) ride the returned result vec.
    fn store_prepared_media(
        &self,
        prepared: &[PreparedMedia],
        path_roots: &[String],
    ) -> NativeResult<Vec<StoreMediaResult>>;
    /// Locate media files (per-item found/missing union); never returns bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn fetch_media(&self, filenames: &[String]) -> NativeResult<Vec<MediaFetchResult>>;
    /// List media filenames, optionally glob-filtered, with a limit.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    fn list_media(
        &self,
        pattern: Option<&str>,
        limit: Option<usize>,
    ) -> NativeResult<ListMediaResponse>;
    /// Delete media by name into Anki's recoverable trash (no ref-check).
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn delete_media(&self, filenames: &[String]) -> NativeResult<DeleteMediaResponse>;
    /// Read-only media diagnostics (unused/missing media, trash state).
    ///
    /// # Errors
    ///
    /// Returns an error if the media check fails.
    fn media_check(&self) -> NativeResult<CollectionCheckResponse>;
    /// The collection cleanups; the removed-note-id list rides out of band for
    /// the kernel's sidecar tail, never the wire.
    ///
    /// # Errors
    ///
    /// Returns an error if the cleanup write fails.
    fn prune(
        &self,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> NativeResult<(CollectionPruneResponse, Vec<i64>)>;
    /// Export the collection (or a scope of it) to an Anki package.
    /// Read-only on the data; holds the collection for the package write, so
    /// the kernel runs it on the actor like every other op. The caller has
    /// already gated `out_path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the scope resolves to nothing, the package cannot
    /// be written to `out_path`, or the export fails.
    fn export_package(&self, request: &ExportRequest) -> NativeResult<ExportOutcome>;
}
