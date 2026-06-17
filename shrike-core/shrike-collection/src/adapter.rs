//! The ONE anki-coupled module (#278's adapter-isolation rule).
//!
//! Everything in this file talks to anki exclusively through its **protobuf
//! service layer** — `Backend::run_service_method(service, method, bytes)` and
//! `Backend::run_db_command_bytes(json)`, the exact surface pylib's rsbridge
//! binds — never the bare crate API (#277 verdict review, binding constraint).
//! An anki tag bump is churn in this file only: re-extract the index tables
//! from the generated dispatcher, run the tripwire tests, done.
//!
//! The `(service, method)` indices below are extracted from the pinned tag's
//! generated dispatcher (`OUT_DIR/backend.rs` at 25.09.4). They are part of the
//! generated contract (proto declaration order); the tripwire tests in this
//! crate call every indexed RPC against a real temp collection, so a silent
//! index shuffle on a tag bump fails loudly instead of corrupting calls.

use anki::backend::{init_backend, Backend};
use prost::Message;
use shrike_ffi::{NativeError, NativeResult};

// In test builds every dispatch is recorded, so the method-constant
// coverage tripwire (#394) can assert each declared index is genuinely
// exercised against a real collection — the interim gate until the
// indices derive from anki's descriptors at build time.
#[cfg(test)]
pub(crate) static DISPATCHED_METHODS: std::sync::Mutex<std::collections::BTreeSet<(u32, u32)>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

#[cfg(test)]
fn record_dispatch(service: u32, method: u32) {
    DISPATCHED_METHODS
        .lock()
        .expect("dispatch recorder poisoned")
        .insert((service, method));
}

// ── service indices (Backend dispatcher, tag 25.09.4) ───────────────────────
const SVC_COLLECTION: u32 = 3;
const SVC_CARDS: u32 = 5;
const SVC_DECKS: u32 = 7;
const SVC_NOTETYPES: u32 = 23;
const SVC_NOTES: u32 = 25;
const SVC_CARD_RENDERING: u32 = 27;
const SVC_SEARCH: u32 = 29;
// import_export (#71/#72). NOT a runtime-spinning service: its export/import
// methods are `with_col` calls (no sync/network), so dispatching it is safe.
const SVC_IMPORT_EXPORT: u32 = 37;
const SVC_MEDIA: u32 = 39;
const SVC_TAGS: u32 = 43;

// ── method indices ───────────────────────────────────────────────────────────
const COLLECTION_OPEN: u32 = 0;
const COLLECTION_CLOSE: u32 = 1;

const CARDS_REMOVE_CARDS: u32 = 2;
const CARDS_SET_DECK: u32 = 3;

const DECKS_NEW_DECK: u32 = 0;
const DECKS_ADD_DECK: u32 = 1;
const DECKS_DECK_TREE: u32 = 4;
const DECKS_GET_DECK_ID_BY_NAME: u32 = 7;
const DECKS_GET_DECK_NAMES: u32 = 13;
const DECKS_REMOVE_DECKS: u32 = 16;
const DECKS_RENAME_DECK: u32 = 18;

const NOTETYPES_ADD_NOTETYPE_LEGACY: u32 = 2;
const NOTETYPES_UPDATE_NOTETYPE_LEGACY: u32 = 3;
const NOTETYPES_GET_STOCK_NOTETYPE_LEGACY: u32 = 5;
const NOTETYPES_GET_NOTETYPE: u32 = 6;
const NOTETYPES_GET_NOTETYPE_LEGACY: u32 = 7;
const NOTETYPES_GET_NOTETYPE_NAMES: u32 = 8;
const NOTETYPES_REMOVE_NOTETYPE: u32 = 11;
const NOTETYPES_CHANGE_NOTETYPE: u32 = 15;

const NOTES_NEW_NOTE: u32 = 0;
const NOTES_ADD_NOTE: u32 = 1;
const NOTES_UPDATE_NOTES: u32 = 5;
const NOTES_GET_NOTE: u32 = 6;
const NOTES_REMOVE_NOTES: u32 = 7;
const NOTES_FIELDS_CHECK: u32 = 11;
const NOTES_CARDS_OF_NOTE: u32 = 12;

// NB: card_rendering is the one service whose BACKEND-level dispatcher is a
// merged table (its backend-specific methods come first, the collection-level
// methods renumbered after) — strip_html is a backend method at index 0, NOT
// the collection-level 10 (which lands on render_markdown). The tripwire test
// pins this: a wrong index here dispatches to a different VALID method.
const CARD_RENDERING_STRIP_HTML: u32 = 0;
const CARD_RENDERING_GET_EMPTY_CARDS: u32 = 5;

const SEARCH_SEARCH_NOTES: u32 = 2;
const SEARCH_FIND_AND_REPLACE: u32 = 5;

// import_export methods (the MERGED backend dispatcher, tag 25.09.4): the
// backend-level methods come first (ImportCollectionPackage=0,
// ExportCollectionPackage=1), then the collection-level ones renumbered after
// (ImportAnkiPackage=2, GetPresets=3, ExportAnkiPackage=4, …). #71 uses the two
// export methods; #72 adds the import-anki-package one (a merge import; the
// destructive import_collection_package=0 restore is the deferred #552).
const IMPORT_EXPORT_EXPORT_COLLECTION_PACKAGE: u32 = 1;
const IMPORT_EXPORT_IMPORT_ANKI_PACKAGE: u32 = 2;
const IMPORT_EXPORT_EXPORT_ANKI_PACKAGE: u32 = 4;

const MEDIA_CHECK_MEDIA: u32 = 0;
const MEDIA_ADD_MEDIA_FILE: u32 = 1;
const MEDIA_TRASH_MEDIA_FILES: u32 = 2;

const TAGS_CLEAR_UNUSED_TAGS: u32 = 0;
const TAGS_ALL_TAGS: u32 = 1;
const TAGS_RENAME_TAGS: u32 = 6;
const TAGS_ADD_NOTE_TAGS: u32 = 7;
const TAGS_REMOVE_NOTE_TAGS: u32 = 8;

/// The duplicate-check states `note_fields_check` reports (mirrors
/// `anki_proto::notes::note_fields_check_response::State`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldsState {
    Normal,
    Empty,
    Duplicate,
    MissingCloze,
    NotetypeNotCloze,
    FieldNotCloze,
    Unknown(i32),
}

impl FieldsState {
    fn from_i32(value: i32) -> Self {
        match value {
            0 => FieldsState::Normal,
            1 => FieldsState::Empty,
            2 => FieldsState::Duplicate,
            3 => FieldsState::MissingCloze,
            4 => FieldsState::NotetypeNotCloze,
            5 => FieldsState::FieldNotCloze,
            other => FieldsState::Unknown(other),
        }
    }
}

/// Build the import summary (#72) from anki's `ImportResponse.Log` — the
/// counts the `ImportSummary` carries. Kept here (adjacent to the RPC) since it
/// reads the anki proto; the type itself lives in shrike-store-api (the trait
/// contract).
fn import_summary_from_log(
    log: anki_proto::import_export::import_response::Log,
) -> shrike_store_api::ImportSummary {
    shrike_store_api::ImportSummary {
        new: log.new.len(),
        updated: log.updated.len(),
        duplicate: log.duplicate.len(),
        conflicting: log.conflicting.len(),
        first_field_match: log.first_field_match.len(),
        missing_notetype: log.missing_notetype.len(),
        missing_deck: log.missing_deck.len(),
        empty_first_field: log.empty_first_field.len(),
        found_notes: log.found_notes as usize,
    }
}

pub use shrike_store_api::ServiceNote;

fn proto_to_note(n: anki_proto::notes::Note) -> ServiceNote {
    ServiceNote {
        id: n.id,
        notetype_id: n.notetype_id,
        fields: n.fields,
        tags: n.tags,
    }
}

pub struct ServiceAdapter {
    backend: Backend,
}

impl ServiceAdapter {
    pub fn new() -> NativeResult<Self> {
        let init = anki_proto::backend::BackendInit {
            preferred_langs: vec!["en".to_string()],
            ..Default::default()
        };
        let mut buf = Vec::new();
        init.encode(&mut buf)
            .map_err(|e| NativeError::internal(format!("encode init: {e}")))?;
        let backend = init_backend(&buf)
            .map_err(|e| NativeError::unavailable(format!("backend init: {e}")))?;
        Ok(Self { backend })
    }

    /// One service-layer call: encode → dispatch → decode-or-error.
    fn call<Req: Message, Resp: Message + Default>(
        &self,
        service: u32,
        method: u32,
        request: &Req,
    ) -> NativeResult<Resp> {
        let mut buf = Vec::new();
        request
            .encode(&mut buf)
            .map_err(|e| NativeError::internal(format!("encode request: {e}")))?;
        #[cfg(test)]
        record_dispatch(service, method);
        let out = self
            .backend
            .run_service_method(service, method, &buf)
            .map_err(|err_bytes| decode_backend_error(&err_bytes))?;
        Resp::decode(out.as_slice())
            .map_err(|e| NativeError::internal(format!("decode response: {e}")))
    }

    // ── lifecycle ────────────────────────────────────────────────────────────

    pub fn open_collection(
        &self,
        collection_path: &str,
        media_folder: &str,
        media_db: &str,
    ) -> NativeResult<()> {
        let req = anki_proto::collection::OpenCollectionRequest {
            collection_path: collection_path.to_string(),
            media_folder_path: media_folder.to_string(),
            media_db_path: media_db.to_string(),
        };
        let _: anki_proto::generic::Empty = self.call(SVC_COLLECTION, COLLECTION_OPEN, &req)?;
        Ok(())
    }

    pub fn close_collection(&self) -> NativeResult<()> {
        let req = anki_proto::collection::CloseCollectionRequest {
            downgrade_to_schema11: false,
        };
        let _: anki_proto::generic::Empty = self.call(SVC_COLLECTION, COLLECTION_CLOSE, &req)?;
        Ok(())
    }

    // ── raw db reads (the DBProxy surface pylib itself uses) ─────────────────

    /// Run one read-only SQL query through the DB proxy, returning JSON rows.
    /// The same surface pylib's `DBProxy.all()` uses; read-only by Shrike
    /// convention (every write goes through a service RPC).
    pub fn db_rows(&self, sql: &str) -> NativeResult<Vec<Vec<serde_json::Value>>> {
        let req = serde_json::json!({
            "kind": "query",
            "sql": sql,
            "args": [],
            "first_row_only": false,
        });
        let out = self
            .backend
            .run_db_command_bytes(req.to_string().as_bytes())
            .map_err(|err_bytes| decode_backend_error(&err_bytes))?;
        serde_json::from_slice(&out).map_err(|e| NativeError::internal(format!("db response: {e}")))
    }

    /// `col.mod` — the drift watermark. The service layer has no RPC for the
    /// raw stamp; pylib reads it through the DB proxy, so we do exactly that.
    pub fn col_mod(&self) -> NativeResult<i64> {
        let rows = self.db_rows("select mod from col")?;
        rows.first()
            .and_then(|r| r.first())
            .and_then(|v| v.as_i64())
            .ok_or_else(|| NativeError::internal("unexpected db row shape for col.mod".to_string()))
    }

    // ── search ───────────────────────────────────────────────────────────────

    /// The full Anki search grammar, unordered (the wrapper's find_notes).
    pub fn search_notes(&self, search: &str) -> NativeResult<Vec<i64>> {
        let req = anki_proto::search::SearchRequest {
            search: search.to_string(),
            ..Default::default()
        };
        let resp: anki_proto::search::SearchResponse =
            self.call(SVC_SEARCH, SEARCH_SEARCH_NOTES, &req)?;
        Ok(resp.ids)
    }

    // ── notetypes ────────────────────────────────────────────────────────────

    pub fn notetype_names(&self) -> NativeResult<Vec<(i64, String)>> {
        let req = anki_proto::generic::Empty::default();
        let resp: anki_proto::notetypes::NotetypeNames =
            self.call(SVC_NOTETYPES, NOTETYPES_GET_NOTETYPE_NAMES, &req)?;
        Ok(resp.entries.into_iter().map(|e| (e.id, e.name)).collect())
    }

    /// The full notetype proto (fields with editor metadata, templates, css,
    /// cloze-ness) — the read surface's serialization source.
    pub fn notetype(&self, notetype_id: i64) -> NativeResult<anki_proto::notetypes::Notetype> {
        let req = anki_proto::notetypes::NotetypeId { ntid: notetype_id };
        self.call(SVC_NOTETYPES, NOTETYPES_GET_NOTETYPE, &req)
    }

    // ── decks ────────────────────────────────────────────────────────────────

    /// Deck id for an exact full name, or None (pylib's `id_for_name`; the
    /// service maps a missing name to NotFound, folded to None here).
    pub fn deck_id_by_name(&self, name: &str) -> NativeResult<Option<i64>> {
        let req = anki_proto::generic::String {
            val: name.to_string(),
        };
        match self.call::<_, anki_proto::decks::DeckId>(SVC_DECKS, DECKS_GET_DECK_ID_BY_NAME, &req)
        {
            Ok(resp) => Ok(Some(resp.did)),
            Err(e) if e.kind == shrike_ffi::ErrorKind::InvalidInput => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Create a normal deck with `name` (pylib's `add_normal_deck_with_name`:
    /// a fresh default deck proto, renamed, added). Returns the new id.
    pub fn add_deck(&self, name: &str) -> NativeResult<i64> {
        let mut deck: anki_proto::decks::Deck = self.call(
            SVC_DECKS,
            DECKS_NEW_DECK,
            &anki_proto::generic::Empty::default(),
        )?;
        deck.name = name.to_string();
        let resp: anki_proto::collection::OpChangesWithId =
            self.call(SVC_DECKS, DECKS_ADD_DECK, &deck)?;
        Ok(resp.id)
    }

    pub fn rename_deck(&self, deck_id: i64, new_name: &str) -> NativeResult<()> {
        let req = anki_proto::decks::RenameDeckRequest {
            deck_id,
            new_name: new_name.to_string(),
        };
        let _: anki_proto::collection::OpChanges = self.call(SVC_DECKS, DECKS_RENAME_DECK, &req)?;
        Ok(())
    }

    pub fn remove_decks(&self, deck_ids: &[i64]) -> NativeResult<()> {
        let req = anki_proto::decks::DeckIds {
            dids: deck_ids.to_vec(),
        };
        let _: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_DECKS, DECKS_REMOVE_DECKS, &req)?;
        Ok(())
    }

    /// Every deck's (id, full name) — pylib's `all_names_and_ids()` call shape
    /// (keep the empty default deck, include filtered decks).
    pub fn deck_names(&self) -> NativeResult<Vec<(i64, String)>> {
        let req = anki_proto::decks::GetDeckNamesRequest {
            skip_empty_default: false,
            include_filtered: true,
        };
        let resp: anki_proto::decks::DeckNames =
            self.call(SVC_DECKS, DECKS_GET_DECK_NAMES, &req)?;
        Ok(resp.entries.into_iter().map(|e| (e.id, e.name)).collect())
    }

    /// The scheduler's due tree (pylib's `sched.deck_due_tree()`): `now` must
    /// be the current epoch seconds so due counts are computed (0 skips them).
    pub fn deck_tree(&self, now: i64) -> NativeResult<anki_proto::decks::DeckTreeNode> {
        let req = anki_proto::decks::DeckTreeRequest { now };
        self.call(SVC_DECKS, DECKS_DECK_TREE, &req)
    }

    // ── tags ─────────────────────────────────────────────────────────────────

    pub fn all_tags(&self) -> NativeResult<Vec<String>> {
        let req = anki_proto::generic::Empty::default();
        let resp: anki_proto::generic::StringList = self.call(SVC_TAGS, TAGS_ALL_TAGS, &req)?;
        Ok(resp.vals)
    }

    /// Collection-wide tag rename (prefix semantics: renames children like
    /// `old::sub`, never the substring `old-ish`) — pylib's `tags.rename`.
    pub fn rename_tags(&self, old: &str, new: &str) -> NativeResult<usize> {
        let req = anki_proto::tags::RenameTagsRequest {
            current_prefix: old.to_string(),
            new_prefix: new.to_string(),
        };
        let resp: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_TAGS, TAGS_RENAME_TAGS, &req)?;
        Ok(resp.count as usize)
    }

    /// Add space-separated `tags` to every note in `note_ids` (bulk_add).
    pub fn add_note_tags(&self, note_ids: &[i64], tags: &str) -> NativeResult<usize> {
        let req = anki_proto::tags::NoteIdsAndTagsRequest {
            note_ids: note_ids.to_vec(),
            tags: tags.to_string(),
        };
        let resp: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_TAGS, TAGS_ADD_NOTE_TAGS, &req)?;
        Ok(resp.count as usize)
    }

    /// Remove space-separated `tags` from every note in `note_ids` (bulk_remove).
    pub fn remove_note_tags(&self, note_ids: &[i64], tags: &str) -> NativeResult<usize> {
        let req = anki_proto::tags::NoteIdsAndTagsRequest {
            note_ids: note_ids.to_vec(),
            tags: tags.to_string(),
        };
        let resp: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_TAGS, TAGS_REMOVE_NOTE_TAGS, &req)?;
        Ok(resp.count as usize)
    }

    // ── cards ────────────────────────────────────────────────────────────────

    pub fn cards_of_note(&self, note_id: i64) -> NativeResult<Vec<i64>> {
        let req = anki_proto::notes::NoteId { nid: note_id };
        let resp: anki_proto::cards::CardIds = self.call(SVC_NOTES, NOTES_CARDS_OF_NOTE, &req)?;
        Ok(resp.cids)
    }

    pub fn set_card_deck(&self, card_ids: &[i64], deck_id: i64) -> NativeResult<()> {
        let req = anki_proto::cards::SetDeckRequest {
            card_ids: card_ids.to_vec(),
            deck_id,
        };
        let _: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_CARDS, CARDS_SET_DECK, &req)?;
        Ok(())
    }

    // ── find & replace / notetype removal ────────────────────────────────────

    /// Anki's own find_and_replace over note fields (Rust regex, undo-able).
    /// Empty `field_name` means all fields. Returns the changed-note count.
    #[allow(clippy::too_many_arguments)]
    pub fn find_and_replace(
        &self,
        note_ids: &[i64],
        search: &str,
        replacement: &str,
        regex: bool,
        match_case: bool,
        field_name: Option<&str>,
    ) -> NativeResult<usize> {
        let req = anki_proto::search::FindAndReplaceRequest {
            nids: note_ids.to_vec(),
            search: search.to_string(),
            replacement: replacement.to_string(),
            regex,
            match_case,
            field_name: field_name.unwrap_or("").to_string(),
        };
        let resp: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_SEARCH, SEARCH_FIND_AND_REPLACE, &req)?;
        Ok(resp.count as usize)
    }

    pub fn remove_notetype(&self, notetype_id: i64) -> NativeResult<()> {
        let req = anki_proto::notetypes::NotetypeId { ntid: notetype_id };
        let _: anki_proto::collection::OpChanges =
            self.call(SVC_NOTETYPES, NOTETYPES_REMOVE_NOTETYPE, &req)?;
        Ok(())
    }

    // ── notetype JSON (schema11) RPCs — pylib's update_dict/new_field path ───
    //
    // The note-type structural ops (#76) port operates on the schema11 JSON
    // dicts through the SAME legacy RPCs pylib's ModelManager uses
    // (update_dict → update_notetype_legacy, new_field → a stock-Basic clone),
    // so the ord-based data/card migration semantics are identical by
    // construction — not re-derived against the proto representation.

    fn json_call<Req: Message>(
        &self,
        method: u32,
        request: &Req,
    ) -> NativeResult<serde_json::Value> {
        let resp: anki_proto::generic::Json = self.call(SVC_NOTETYPES, method, request)?;
        serde_json::from_slice(&resp.json)
            .map_err(|e| NativeError::internal(format!("notetype json: {e}")))
    }

    /// The stock Basic notetype as a schema11 dict (the donor pylib's
    /// `models.new` / `new_field` / `new_template` clone from).
    pub fn stock_notetype_legacy(&self) -> NativeResult<serde_json::Value> {
        let req = anki_proto::notetypes::StockNotetype::default(); // kind 0 = Basic
        self.json_call(NOTETYPES_GET_STOCK_NOTETYPE_LEGACY, &req)
    }

    pub fn notetype_legacy(&self, notetype_id: i64) -> NativeResult<serde_json::Value> {
        let req = anki_proto::notetypes::NotetypeId { ntid: notetype_id };
        self.json_call(NOTETYPES_GET_NOTETYPE_LEGACY, &req)
    }

    /// Add a schema11 notetype dict; returns the new id (pylib's `models.add`).
    pub fn add_notetype_legacy(&self, notetype: &serde_json::Value) -> NativeResult<i64> {
        let req = anki_proto::generic::Json {
            json: notetype.to_string().into_bytes(),
        };
        let resp: anki_proto::collection::OpChangesWithId =
            self.call(SVC_NOTETYPES, NOTETYPES_ADD_NOTETYPE_LEGACY, &req)?;
        Ok(resp.id)
    }

    /// Persist a mutated schema11 notetype dict (pylib's `update_dict` — the
    /// single write behind every structural op; Anki migrates note data/cards
    /// from the `ord` markers).
    pub fn update_notetype_legacy(&self, notetype: &serde_json::Value) -> NativeResult<()> {
        let req = anki_proto::generic::Json {
            json: notetype.to_string().into_bytes(),
        };
        let _: anki_proto::collection::OpChanges =
            self.call(SVC_NOTETYPES, NOTETYPES_UPDATE_NOTETYPE_LEGACY, &req)?;
        Ok(())
    }

    /// Anki's history-safe note-type migration (pylib's `models.change`).
    pub fn change_notetype(
        &self,
        req: &anki_proto::notetypes::ChangeNotetypeRequest,
    ) -> NativeResult<()> {
        let _: anki_proto::collection::OpChanges =
            self.call(SVC_NOTETYPES, NOTETYPES_CHANGE_NOTETYPE, req)?;
        Ok(())
    }

    /// One write through the DB proxy. Exists ONLY for the pylib-mirroring
    /// `set_schema_modified` bump before `change_notetype` (pylib itself does
    /// `update col set scm=?` via this proxy — its `execute` is literally an
    /// alias of the query path, so this is the same `kind: "query"` call);
    /// every other write goes through a service RPC and reads stay on
    /// `db_rows`.
    pub fn db_execute(&self, sql: &str, args: &[serde_json::Value]) -> NativeResult<()> {
        let req = serde_json::json!({
            "kind": "query",
            "sql": sql,
            "args": args,
            "first_row_only": false,
        });
        self.backend
            .run_db_command_bytes(req.to_string().as_bytes())
            .map_err(|err_bytes| decode_backend_error(&err_bytes))?;
        Ok(())
    }

    // ── media (#70 port) ─────────────────────────────────────────────────────

    /// Store bytes under (a collision-resolved variant of) `desired_name`;
    /// returns the ACTUAL name Anki chose (pylib's `media.write_data`).
    pub fn add_media_file(&self, desired_name: &str, data: &[u8]) -> NativeResult<String> {
        let req = anki_proto::media::AddMediaFileRequest {
            desired_name: desired_name.to_string(),
            data: data.to_vec(),
        };
        let resp: anki_proto::generic::String = self.call(SVC_MEDIA, MEDIA_ADD_MEDIA_FILE, &req)?;
        Ok(resp.val)
    }

    /// Move media files to Anki's recoverable trash.
    pub fn trash_media_files(&self, fnames: &[String]) -> NativeResult<()> {
        let req = anki_proto::media::TrashMediaFilesRequest {
            fnames: fnames.to_vec(),
        };
        let _: anki_proto::generic::Empty = self.call(SVC_MEDIA, MEDIA_TRASH_MEDIA_FILES, &req)?;
        Ok(())
    }

    /// Anki's media check (unused/missing/missing-notes/trash state).
    pub fn check_media(&self) -> NativeResult<anki_proto::media::CheckMediaResponse> {
        self.call(
            SVC_MEDIA,
            MEDIA_CHECK_MEDIA,
            &anki_proto::generic::Empty::default(),
        )
    }

    // ── import/export (#71/#72) ─────────────────────────────────────────────

    /// Export an `.apkg` (the modern Rust exporter, `ExportAnkiPackage`):
    /// whole-collection or deck/note-scoped, with optional scheduling/media.
    /// Returns the exported note count (anki's `generic.UInt32`). The
    /// collection is held for the whole `with_col` export — the caller routes
    /// this through the collection actor so it serializes like every write.
    pub fn export_anki_package(
        &self,
        out_path: &str,
        with_scheduling: bool,
        with_media: bool,
        legacy: bool,
        limit: anki_proto::import_export::ExportLimit,
    ) -> NativeResult<u32> {
        let req = anki_proto::import_export::ExportAnkiPackageRequest {
            out_path: out_path.to_string(),
            options: Some(anki_proto::import_export::ExportAnkiPackageOptions {
                with_scheduling,
                // Deck configs ride with scheduling — they are meaningless
                // without it (an apkg with no review data has no use for the
                // deck's scheduling config), so we bind them together rather
                // than expose a second knob that only matters when the first is on.
                with_deck_configs: with_scheduling,
                with_media,
                legacy,
            }),
            limit: Some(limit),
        };
        let resp: anki_proto::generic::UInt32 =
            self.call(SVC_IMPORT_EXPORT, IMPORT_EXPORT_EXPORT_ANKI_PACKAGE, &req)?;
        Ok(resp.val)
    }

    /// Export a `.colpkg` (whole-collection backup, `ExportCollectionPackage`).
    /// No scoping — a colpkg is the entire collection. Optionally includes media.
    pub fn export_collection_package(
        &self,
        out_path: &str,
        include_media: bool,
        legacy: bool,
    ) -> NativeResult<()> {
        let req = anki_proto::import_export::ExportCollectionPackageRequest {
            out_path: out_path.to_string(),
            include_media,
            legacy,
        };
        let _: anki_proto::generic::Empty = self.call(
            SVC_IMPORT_EXPORT,
            IMPORT_EXPORT_EXPORT_COLLECTION_PACKAGE,
            &req,
        )?;
        Ok(())
    }

    /// Import an `.apkg`/`.colpkg` via anki's modern Rust importer
    /// (`import_anki_package`) — a MERGE into the open collection (notes added/
    /// updated), NOT the destructive whole-collection restore (that is the
    /// separate `import_collection_package`, deferred to #552). MUTATES the
    /// collection (bumps `col.mod`), so the caller MUST drive a drift reconcile
    /// afterward (never advance the index watermark — the col_mod bump is the
    /// signal). Returns per-bucket counts.
    pub fn import_anki_package(
        &self,
        package_path: &str,
        options: shrike_store_api::ImportOptions,
    ) -> NativeResult<shrike_store_api::ImportSummary> {
        let req = anki_proto::import_export::ImportAnkiPackageRequest {
            package_path: package_path.to_string(),
            options: Some(anki_proto::import_export::ImportAnkiPackageOptions {
                merge_notetypes: options.merge_notetypes,
                update_notes: options.update_notes as i32,
                update_notetypes: options.update_notetypes as i32,
                with_scheduling: options.with_scheduling,
                // Deferred (#72 scope): not exposed; anki's default is false.
                with_deck_configs: false,
            }),
        };
        let resp: anki_proto::import_export::ImportResponse =
            self.call(SVC_IMPORT_EXPORT, IMPORT_EXPORT_IMPORT_ANKI_PACKAGE, &req)?;
        Ok(import_summary_from_log(resp.log.unwrap_or_default()))
    }

    // ── maintenance ──────────────────────────────────────────────────────────

    pub fn get_empty_cards(&self) -> NativeResult<anki_proto::card_rendering::EmptyCardsReport> {
        self.call(
            SVC_CARD_RENDERING,
            CARD_RENDERING_GET_EMPTY_CARDS,
            &anki_proto::generic::Empty::default(),
        )
    }

    pub fn remove_cards(&self, card_ids: &[i64]) -> NativeResult<()> {
        let req = anki_proto::cards::RemoveCardsRequest {
            card_ids: card_ids.to_vec(),
        };
        // remove_cards returns OpChangesWithCount (service.rs) — decoding the
        // wrong message here produced a wire-type error the ported pytest
        // suite caught; the Rust round-trip lacked an empty-CARD case.
        let _: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_CARDS, CARDS_REMOVE_CARDS, &req)?;
        Ok(())
    }

    pub fn clear_unused_tags(&self) -> NativeResult<usize> {
        let resp: anki_proto::collection::OpChangesWithCount = self.call(
            SVC_TAGS,
            TAGS_CLEAR_UNUSED_TAGS,
            &anki_proto::generic::Empty::default(),
        )?;
        Ok(resp.count as usize)
    }

    // ── card rendering ───────────────────────────────────────────────────────

    /// Anki's own HTML→text (NORMAL mode: drop tags + `<img>`, unescape
    /// entities) — the SAME RPC pylib's `anki.utils.strip_html` calls, so the
    /// embed-text normalization is byte-identical to the Python facade's by
    /// construction.
    pub fn strip_html(&self, text: &str) -> NativeResult<String> {
        let req = anki_proto::card_rendering::StripHtmlRequest {
            text: text.to_string(),
            mode: anki_proto::card_rendering::strip_html_request::Mode::Normal as i32,
        };
        let resp: anki_proto::generic::String =
            self.call(SVC_CARD_RENDERING, CARD_RENDERING_STRIP_HTML, &req)?;
        Ok(resp.val)
    }

    // ── notes ────────────────────────────────────────────────────────────────

    pub fn new_note(&self, notetype_id: i64) -> NativeResult<ServiceNote> {
        let req = anki_proto::notetypes::NotetypeId { ntid: notetype_id };
        let resp: anki_proto::notes::Note = self.call(SVC_NOTES, NOTES_NEW_NOTE, &req)?;
        Ok(proto_to_note(resp))
    }

    pub fn add_note(&self, note: &ServiceNote, deck_id: i64) -> NativeResult<i64> {
        let req = anki_proto::notes::AddNoteRequest {
            note: Some(service_note_to_proto(note)),
            deck_id,
        };
        let resp: anki_proto::notes::AddNoteResponse =
            self.call(SVC_NOTES, NOTES_ADD_NOTE, &req)?;
        Ok(resp.note_id)
    }

    pub fn get_note(&self, note_id: i64) -> NativeResult<ServiceNote> {
        let req = anki_proto::notes::NoteId { nid: note_id };
        let resp: anki_proto::notes::Note = self.call(SVC_NOTES, NOTES_GET_NOTE, &req)?;
        Ok(proto_to_note(resp))
    }

    /// Set the exact tag list on many notes in ONE read + ONE `UpdateNotes`
    /// write (#445/#716): one batched DB read for the current note rows, then
    /// one transaction + one undo entry — instead of the get+update round trip
    /// and a journal commit per note (the 1000-note tag-set op previously paid
    /// 3 RPCs and an fsync each). The read side used to fan out to one
    /// `GetNote` RPC per note (the N+1 the "ONE call" framing hid), but the
    /// service layer has no batched `GetNotes`; the DB proxy does (the same
    /// `db_rows` surface `col_mod`/the prune reads use), so the whole read is
    /// one `SELECT … WHERE id IN (…)` round trip.
    ///
    /// `UpdateNotes` re-loads each note from storage by id (anki's
    /// `update_note_inner`); `note_differs_from_db` is only a skip-if-identical
    /// short-circuit, NOT a guard — a row whose `guid`/`notetype_id`/`fields`
    /// differ from storage is APPLIED/written through (a changed notetype even
    /// regenerates cards), and anki re-stamps `mtime`/`usn` itself. So a wrong
    /// value here would silently CORRUPT the note (it does not error) —
    /// therefore the row we send MUST carry each note's *current*
    /// guid/notetype_id/fields verbatim (only `tags` is overwritten), exactly
    /// as the prior `GetNote` fetch did. `flds` is anki's 0x1f-separated field
    /// blob, split the same way anki's `split_fields` (and our `typed_notes`)
    /// does.
    ///
    /// Contract: callers must pre-filter to existing ids — an absent id is
    /// silently skipped here (the `IN (…)` read just omits it; no per-note
    /// `GetNote`-style not-found error), mirroring the old per-note skip.
    pub fn set_note_tags_bulk(&self, note_ids: &[i64], tags: &[String]) -> NativeResult<usize> {
        if note_ids.is_empty() {
            return Ok(0);
        }
        // Integer ids (never user strings) → safe to inline, like the other
        // db_rows reads in this crate; no proxy parameterization needed.
        let id_list = crate::read::ids_sql_list(note_ids);
        let rows = self.db_rows(&format!(
            "select id, guid, mid, mod, usn, flds from notes where id in ({id_list})"
        ))?;
        let mut notes = Vec::with_capacity(rows.len());
        for row in rows {
            let (Some(id), Some(guid), Some(mid), Some(modt), Some(usn), Some(flds)) = (
                row.first().and_then(serde_json::Value::as_i64),
                row.get(1).and_then(serde_json::Value::as_str),
                row.get(2).and_then(serde_json::Value::as_i64),
                row.get(3).and_then(serde_json::Value::as_i64),
                row.get(4).and_then(serde_json::Value::as_i64),
                row.get(5).and_then(serde_json::Value::as_str),
            ) else {
                return Err(NativeError::internal(
                    "unexpected db row shape for notes".to_string(),
                ));
            };
            notes.push(anki_proto::notes::Note {
                id,
                guid: guid.to_string(),
                notetype_id: mid,
                mtime_secs: modt as u32,
                usn: usn as i32,
                tags: tags.to_vec(),
                fields: flds.split('\u{1f}').map(str::to_string).collect(),
            });
        }
        if notes.is_empty() {
            return Ok(0);
        }
        let count = notes.len();
        let req = anki_proto::notes::UpdateNotesRequest {
            notes,
            skip_undo_entry: false,
        };
        let _: anki_proto::collection::OpChanges =
            self.call(SVC_NOTES, NOTES_UPDATE_NOTES, &req)?;
        Ok(count)
    }

    pub fn update_note(&self, note: &ServiceNote) -> NativeResult<()> {
        // Round-trip through get_note so untouched proto fields (mtime/usn)
        // stay authoritative; we overwrite only fields/tags.
        let mut current: anki_proto::notes::Note = self.call(
            SVC_NOTES,
            NOTES_GET_NOTE,
            &anki_proto::notes::NoteId { nid: note.id },
        )?;
        current.fields = note.fields.clone();
        current.tags = note.tags.clone();
        let req = anki_proto::notes::UpdateNotesRequest {
            notes: vec![current],
            skip_undo_entry: false,
        };
        let _: anki_proto::collection::OpChanges =
            self.call(SVC_NOTES, NOTES_UPDATE_NOTES, &req)?;
        Ok(())
    }

    pub fn remove_notes(&self, note_ids: &[i64]) -> NativeResult<usize> {
        let req = anki_proto::notes::RemoveNotesRequest {
            note_ids: note_ids.to_vec(),
            card_ids: vec![],
        };
        let resp: anki_proto::collection::OpChangesWithCount =
            self.call(SVC_NOTES, NOTES_REMOVE_NOTES, &req)?;
        Ok(resp.count as usize)
    }

    /// Anki's own add-note validation — the #77 duplicate rule's source of truth.
    pub fn fields_check(&self, note: &ServiceNote) -> NativeResult<FieldsState> {
        let req = service_note_to_proto(note);
        let resp: anki_proto::notes::NoteFieldsCheckResponse =
            self.call(SVC_NOTES, NOTES_FIELDS_CHECK, &req)?;
        Ok(FieldsState::from_i32(resp.state))
    }
}

fn service_note_to_proto(note: &ServiceNote) -> anki_proto::notes::Note {
    anki_proto::notes::Note {
        id: note.id,
        notetype_id: note.notetype_id,
        fields: note.fields.clone(),
        tags: note.tags.clone(),
        ..Default::default()
    }
}

/// Decode the service layer's error bytes (`anki_proto::backend::BackendError`)
/// into the shared native taxonomy: invalid-input kinds map to the expected
/// tier, everything else is unavailable/internal.
fn decode_backend_error(bytes: &[u8]) -> NativeError {
    match anki_proto::backend::BackendError::decode(bytes) {
        Ok(err) => {
            use anki_proto::backend::backend_error::Kind;
            let kind = Kind::try_from(err.kind).unwrap_or(Kind::UndoEmpty);
            // Anki wraps interpolated values in Unicode isolation marks
            // (U+2068/U+2069) for bidi safety in its own UI; in an API error
            // they're invisible garbage — strip at the source so every
            // surface (tool errors, logs, exceptions) is clean.
            let err = anki_proto::backend::BackendError {
                message: err.message.replace(['\u{2068}', '\u{2069}'], ""),
                ..err
            };
            match kind {
                Kind::InvalidInput | Kind::NotFoundError | Kind::Exists | Kind::SearchError => {
                    NativeError::invalid_input(err.message)
                }
                Kind::DbError | Kind::Interrupted => NativeError::unavailable(err.message),
                _ => NativeError::internal(err.message),
            }
        }
        Err(_) => NativeError::internal("undecodable backend error"),
    }
}

// ── runtime-singularity pin (#374 design 9; revisited #503) ──────────────────
// anki's Backend owns a LAZY tokio runtime whose only initializer is
// `runtime_handle()`, consumed solely by the sync/AnkiWeb/AnkiHub services.
// Shrike dispatches none of those services TODAY, so anki's runtime stays cold
// and the kernel's owned runtime (#374) is the only one alive. This test pins
// exactly that: not one of the runtime-spinning service indices appears in the
// dispatched set.
//
// #503 settled what happens when sync support DOES land (#33/#362 wakes these
// services): the invariant the kernel guarantees is NOT "one runtime" but
// "sync ops never run on a runtime worker thread". anki's sync paths
// `block_on`, which panics from any runtime-worker thread regardless of which
// runtime owns it — so the fix is the `spawn_blocking` dispatch discipline in
// `shrike_kernel::runtime` (pinned by its `sync_dispatch_pin` panic-repro
// test), NOT a runtime-handle-injection patch to anki (rejected: the anki
// patch mechanism is Bazel-only, so it would fork sync behaviour across build
// lanes — see docs/decisions.md). This test stays as the today-true pin that
// none of those services is on a Shrike call path yet.
// (Backend dispatcher, tag 25.09.4: sync=41, ankiweb=45, ankihub=47 — none
// may appear in the service indices above.)
#[cfg(test)]
mod runtime_singularity {
    /// The dispatcher indices of anki's runtime-spinning services at the
    /// pinned tag. Bump alongside the SVC_* table on a tag bump.
    const SVC_SYNC: u32 = 41;
    const SVC_ANKIWEB: u32 = 45;
    const SVC_ANKIHUB: u32 = 47;

    #[test]
    fn no_runtime_spinning_service_is_dispatched() {
        // Compile-time-ish pin: every SVC_* this adapter dispatches, by value.
        let dispatched = [
            super::SVC_COLLECTION,
            super::SVC_CARDS,
            super::SVC_DECKS,
            super::SVC_NOTETYPES,
            super::SVC_NOTES,
            super::SVC_CARD_RENDERING,
            super::SVC_SEARCH,
            super::SVC_IMPORT_EXPORT,
            super::SVC_MEDIA,
            super::SVC_TAGS,
        ];
        for svc in [SVC_SYNC, SVC_ANKIWEB, SVC_ANKIHUB] {
            assert!(
                !dispatched.contains(&svc),
                "service {svc} spins anki's internal runtime — Shrike must never dispatch it"
            );
        }
        // And the source itself: no other SVC_ constant exists outside the
        // dispatched list + this pin's own three (a new one must be reviewed
        // against this pin). The needle is built dynamically so the filter
        // line can't match itself.
        let needle = format!("const {}_", "SVC");
        let source = include_str!("adapter.rs");
        let declared = source
            .lines()
            .filter(|l| l.trim_start().starts_with(&needle) && l.contains(": u32 ="))
            .count();
        assert_eq!(
            declared,
            dispatched.len() + 3,
            "a new SVC_ index was added — review it against the runtime-singularity pin"
        );
    }
}
