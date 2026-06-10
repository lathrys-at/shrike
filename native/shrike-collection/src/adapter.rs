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

// ── service indices (Backend dispatcher, tag 25.09.4) ───────────────────────
const SVC_COLLECTION: u32 = 3;
const SVC_DECKS: u32 = 7;
const SVC_NOTETYPES: u32 = 23;
const SVC_NOTES: u32 = 25;
const SVC_CARD_RENDERING: u32 = 27;
const SVC_SEARCH: u32 = 29;
const SVC_TAGS: u32 = 43;

// ── method indices ───────────────────────────────────────────────────────────
const COLLECTION_OPEN: u32 = 0;
const COLLECTION_CLOSE: u32 = 1;

const DECKS_DECK_TREE: u32 = 4;
const DECKS_GET_DECK_NAMES: u32 = 13;

const NOTETYPES_GET_NOTETYPE: u32 = 6;
const NOTETYPES_GET_NOTETYPE_NAMES: u32 = 8;

const NOTES_NEW_NOTE: u32 = 0;
const NOTES_ADD_NOTE: u32 = 1;
const NOTES_UPDATE_NOTES: u32 = 5;
const NOTES_GET_NOTE: u32 = 6;
const NOTES_REMOVE_NOTES: u32 = 7;
const NOTES_FIELDS_CHECK: u32 = 11;

// NB: card_rendering is the one service whose BACKEND-level dispatcher is a
// merged table (its backend-specific methods come first, the collection-level
// methods renumbered after) — strip_html is a backend method at index 0, NOT
// the collection-level 10 (which lands on render_markdown). The tripwire test
// pins this: a wrong index here dispatches to a different VALID method.
const CARD_RENDERING_STRIP_HTML: u32 = 0;

const SEARCH_SEARCH_NOTES: u32 = 2;

const TAGS_ALL_TAGS: u32 = 1;

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

/// A note as the service layer sees it (the slice the vertical ops need).
#[derive(Debug, Clone)]
pub struct ServiceNote {
    pub id: i64,
    pub notetype_id: i64,
    pub fields: Vec<String>,
    pub tags: Vec<String>,
}

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
