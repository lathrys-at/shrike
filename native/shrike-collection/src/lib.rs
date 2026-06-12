//! Rust collection core over anki's protobuf service layer (#278, slice 1).
//!
//! Slice-1 architecture (this crate is PR 1 of the slice's series):
//!
//! - **`adapter`** — the ONE anki-coupled module. Everything reaches anki
//!   through `Backend::run_service_method` / `run_db_command_bytes` (the exact
//!   rsbridge surface pylib binds) with `anki_proto` messages — never the bare
//!   crate API (#277 verdict review, binding). Tag bumps are churn here only.
//! - **`CollectionCore`** (below) — Shrike's op layer, written against the
//!   adapter. This PR carries the vertical slice: open/close, the `col.mod`
//!   watermark, the full search grammar, note read/create/update/delete, and
//!   the #77 duplicate policy on create. Later PRs in the series extend the op
//!   inventory (tracked on #278); the **wholesale facade cutover stays off**
//!   until coverage is complete — the hard safety rule (never co-manage one
//!   collection from two cores in a process) forbids per-op fallback, so the
//!   core is reachable only through the parity harness until then.
//!
//! Pin policy: the anki git tag equals the pip wheel version; bumped together.

mod adapter;
pub mod embed_text;
mod media;
pub mod media_fetch;
mod note_types;
mod read;
mod write;

pub use adapter::{FieldsState, ServiceAdapter};
pub use embed_text::{extract_image_refs, EMBED_TEXT_VERSION};
pub use media::media_name_from_url;
use shrike_ffi::{NativeError, NativeResult};

// Canonical homes moved to the store contract (#389); re-exported so the
// pre-trait import paths keep working.
pub use shrike_store_api::{
    Collection, CreateOutcome, DuplicatePolicy, OwnedFieldRow, PreparedMedia, PreparedMediaSource,
    ServiceNote,
};

/// Shrike's collection core, slice-1 vertical. One instance owns one open
/// collection (instance-per-collection, no global state), mirroring the
/// CollectionWrapper lifecycle it will eventually back.
pub struct CollectionCore {
    adapter: ServiceAdapter,
    collection_path: String,
    media_dir: String,
    /// Cooperative idle-release state (#64): set by `release`, cleared by
    /// `reopen` — `ensure_open` re-acquires on demand so an op that lands
    /// while released self-heals instead of erroring CollectionNotOpen.
    released: std::sync::atomic::AtomicBool,
}

impl CollectionCore {
    /// Open (creating if needed) a collection. `media_folder`/`media_db` are
    /// derived from the collection path exactly like anki's Python does.
    pub fn open(collection_path: &str) -> NativeResult<Self> {
        let adapter = ServiceAdapter::new()?;
        let base = collection_path
            .strip_suffix(".anki2")
            .unwrap_or(collection_path);
        let media_dir = format!("{base}.media");
        adapter.open_collection(collection_path, &media_dir, &format!("{base}.media.db2"))?;
        Ok(Self {
            released: std::sync::atomic::AtomicBool::new(false),
            adapter,
            collection_path: collection_path.to_string(),
            media_dir,
        })
    }

    pub fn close(&self) -> NativeResult<()> {
        self.adapter.close_collection()
    }

    /// Release the collection (cooperative idle-release, #64): close, keeping
    /// the instance reusable via [`reopen`]. Already-closed is a no-op.
    pub fn release(&self) -> NativeResult<()> {
        let _ = self.adapter.close_collection();
        self.released
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Re-acquire if (and only if) idle-released — the open-on-demand half of
    /// cooperative locking, run before every serialized job. Returns whether
    /// a reopen happened. Contention surfaces as the BUSY tier via `reopen`.
    pub fn ensure_open(&self) -> NativeResult<bool> {
        if !self.released.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(false);
        }
        self.reopen()?;
        Ok(true)
    }

    /// Re-acquire after a release (#64/#79). The file opened fine at boot, so
    /// a failure here is overwhelmingly lock contention (another process —
    /// usually Anki desktop — holds it), not corruption: it surfaces as the
    /// BUSY tier, mirroring the Python wrapper's contextual classification.
    /// The caller decides whether to retry; nothing here waits.
    pub fn reopen(&self) -> NativeResult<()> {
        self.released
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let _ = self.adapter.close_collection(); // a half-open handle is fine to close
        let base = self
            .collection_path
            .strip_suffix(".anki2")
            .unwrap_or(&self.collection_path);
        self.adapter
            .open_collection(
                &self.collection_path,
                &format!("{base}.media"),
                &format!("{base}.media.db2"),
            )
            .map_err(|e| {
                NativeError::busy(format!(
                    "the collection is in use by another process: {}",
                    e.message
                ))
            })
    }

    /// The collection-modified watermark Shrike's drift detection leans on.
    pub fn col_mod(&self) -> NativeResult<i64> {
        self.adapter.col_mod()
    }

    /// The full Anki search grammar → note ids (read-only).
    pub fn find_notes(&self, search: &str) -> NativeResult<Vec<i64>> {
        self.adapter.search_notes(search)
    }

    /// Resolve a notetype by name (case-sensitive, like the Python wrapper).
    pub fn notetype_id(&self, name: &str) -> NativeResult<i64> {
        self.notetype_id_opt(name)?
            .ok_or_else(|| NativeError::invalid_input(format!("unknown note type: {name}")))
    }

    pub(crate) fn notetype_id_opt(&self, name: &str) -> NativeResult<Option<i64>> {
        Ok(self
            .adapter
            .notetype_names()?
            .into_iter()
            .find(|(_, n)| n == name)
            .map(|(id, _)| id))
    }

    pub(crate) fn notetype_name(&self, notetype_id: i64) -> NativeResult<String> {
        Ok(self
            .adapter
            .notetype_names()?
            .into_iter()
            .find(|(id, _)| *id == notetype_id)
            .map(|(_, n)| n)
            .unwrap_or_else(|| "Unknown".to_string()))
    }

    pub(crate) fn notetype_field_names(&self, notetype_id: i64) -> NativeResult<Vec<String>> {
        Ok(self
            .adapter
            .notetype(notetype_id)?
            .fields
            .into_iter()
            .map(|f| f.name)
            .collect())
    }

    pub fn get_note(&self, note_id: i64) -> NativeResult<ServiceNote> {
        self.adapter.get_note(note_id)
    }

    /// Create a note under the #77 policy: Anki's own `fields_check` runs
    /// first; structural problems (empty first field, broken cloze) are always
    /// errors, a first-field duplicate is governed by `policy`.
    pub fn create_note(
        &self,
        notetype_id: i64,
        deck_id: i64,
        fields: &[String],
        tags: &[String],
        policy: DuplicatePolicy,
    ) -> NativeResult<CreateOutcome> {
        let mut note = self.adapter.new_note(notetype_id)?;
        for (i, value) in fields.iter().enumerate() {
            if i < note.fields.len() {
                note.fields[i] = value.clone();
            }
        }
        note.tags = tags.to_vec();

        match self.adapter.fields_check(&note)? {
            FieldsState::Normal => {}
            FieldsState::Duplicate => match policy {
                DuplicatePolicy::Allow => {}
                DuplicatePolicy::Skip => return Ok(CreateOutcome::SkippedDuplicate),
                DuplicatePolicy::Error => {
                    return Err(NativeError::invalid_input(
                        "duplicate: a note with this first field already exists".to_string(),
                    ));
                }
            },
            FieldsState::Empty => {
                return Err(NativeError::invalid_input(
                    "first field is empty".to_string(),
                ));
            }
            other => {
                return Err(NativeError::invalid_input(format!(
                    "note failed validation: {other:?}"
                )));
            }
        }

        let id = self.adapter.add_note(&note, deck_id)?;
        Ok(CreateOutcome::Created(id))
    }

    /// Replace a note's fields/tags (the update half of upsert; existence
    /// errors surface as invalid_input from the service layer).
    pub fn update_note(
        &self,
        note_id: i64,
        fields: &[String],
        tags: Option<&[String]>,
    ) -> NativeResult<()> {
        let mut note = self.adapter.get_note(note_id)?;
        for (i, value) in fields.iter().enumerate() {
            if i < note.fields.len() {
                note.fields[i] = value.clone();
            }
        }
        if let Some(tags) = tags {
            note.tags = tags.to_vec();
        }
        self.adapter.update_note(&note)
    }

    pub fn delete_notes(&self, note_ids: &[i64]) -> NativeResult<usize> {
        self.adapter.remove_notes(note_ids)
    }

    /// The card ids generated by one note (lowest-ordinal first is not
    /// guaranteed; callers needing deck-of-note order sort upstream).
    pub fn cards_of_note(&self, note_id: i64) -> NativeResult<Vec<i64>> {
        self.adapter.cards_of_note(note_id)
    }

    /// `(card_id, template_ordinal)` per card of one note — the identity the
    /// template data-safety tests assert on.
    pub fn card_ords_of_note(&self, note_id: i64) -> NativeResult<Vec<(i64, i64)>> {
        Ok(self
            .adapter
            .db_rows(&format!("select id, ord from cards where nid = {note_id}"))?
            .into_iter()
            .filter_map(|r| Some((r.first()?.as_i64()?, r.get(1)?.as_i64()?)))
            .collect())
    }
}

/// The store contract (#389): every method forwards to the inherent impl,
/// so the concrete core keeps its full API while the kernel and the host
/// bindings consume `dyn Collection`.
#[allow(clippy::use_self)]
mod contract {
    use super::{CollectionCore, CreateOutcome, DuplicatePolicy, NativeResult};
    use crate::{OwnedFieldRow, PreparedMedia, ServiceNote};
    use serde_json::Value;
    use shrike_schemas::{
        CollectionCheckResponse, CollectionInfo, CollectionPruneResponse, DeckInput,
        DeleteDecksResponse, DeleteMediaResponse, DeleteNoteTypeResult, FieldMetadataInput,
        FieldOp, FindReplaceNoteTypesResponse, ListMediaResponse, ListNotesResponse,
        MediaFetchResult, MigrateNoteTypeResponse, NoteInput, NoteTypeInput, NoteTypeResult,
        RenameTagResponse, StoreMediaItem, StoreMediaResult, TemplateOp, UpdateNoteTagsResponse,
        UpdateNoteTypeFieldMetadataResponse, UpdateNoteTypeFieldsResponse,
        UpdateNoteTypeTemplatesResponse, UpsertDeckResult, UpsertNoteResult,
    };
    use std::collections::BTreeMap;

    impl shrike_store_api::Collection for CollectionCore {
        fn close(&self) -> NativeResult<()> {
            Self::close(self)
        }
        fn release(&self) -> NativeResult<()> {
            Self::release(self)
        }
        fn ensure_open(&self) -> NativeResult<bool> {
            Self::ensure_open(self)
        }
        fn reopen(&self) -> NativeResult<()> {
            Self::reopen(self)
        }
        fn col_mod(&self) -> NativeResult<i64> {
            Self::col_mod(self)
        }
        fn find_notes(&self, search: &str) -> NativeResult<Vec<i64>> {
            Self::find_notes(self, search)
        }
        fn notetype_id(&self, name: &str) -> NativeResult<i64> {
            Self::notetype_id(self, name)
        }
        fn get_note(&self, note_id: i64) -> NativeResult<ServiceNote> {
            Self::get_note(self, note_id)
        }
        fn cards_of_note(&self, note_id: i64) -> NativeResult<Vec<i64>> {
            Self::cards_of_note(self, note_id)
        }
        fn card_ords_of_note(&self, note_id: i64) -> NativeResult<Vec<(i64, i64)>> {
            Self::card_ords_of_note(self, note_id)
        }
        fn note_count(&self) -> NativeResult<usize> {
            Self::note_count(self)
        }
        fn note_texts(&self, note_ids: &[i64]) -> NativeResult<Vec<String>> {
            Self::note_texts(self, note_ids)
        }
        fn note_embed_inputs(
            &self,
            note_ids: &[i64],
        ) -> NativeResult<Vec<(i64, String, Vec<String>)>> {
            Self::note_embed_inputs(self, note_ids)
        }
        fn derived_field_rows(
            &self,
            note_ids: &[i64],
        ) -> NativeResult<Vec<(i64, String, String, String)>> {
            Self::derived_field_rows(self, note_ids)
        }
        fn note_image_refs(&self) -> NativeResult<Vec<(i64, Vec<String>)>> {
            Self::note_image_refs(self)
        }
        fn note_tag_rows(&self) -> NativeResult<Vec<(i64, Vec<String>)>> {
            Self::note_tag_rows(self)
        }
        fn any_tagged(&self, note_ids: &[i64]) -> NativeResult<bool> {
            Self::any_tagged(self, note_ids)
        }
        fn note_field_map(&self, note_ids: &[i64]) -> NativeResult<Vec<OwnedFieldRow>> {
            Self::note_field_map(self, note_ids)
        }
        fn normalize_text(&self, value: &str) -> NativeResult<String> {
            Self::normalize_text(self, value)
        }
        fn resolve_deck_ref(&self, reference: &str) -> NativeResult<Option<String>> {
            Self::resolve_deck_ref(self, reference)
        }
        fn query(
            &self,
            search: &str,
            with_fields: bool,
            limit: usize,
        ) -> NativeResult<ListNotesResponse> {
            Self::query(self, search, with_fields, limit)
        }
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
        ) -> NativeResult<ListNotesResponse> {
            Self::list_notes(
                self,
                ids,
                deck,
                tags,
                note_type,
                modified_since,
                with_fields,
                limit,
            )
        }
        fn note_dicts(&self, note_ids: &[i64], with_fields: bool) -> NativeResult<Vec<Value>> {
            Self::note_dicts(self, note_ids, with_fields)
        }
        fn collection_info(
            &self,
            sections: &[String],
            detail_names: &[String],
        ) -> NativeResult<CollectionInfo> {
            Self::collection_info(self, sections, detail_names)
        }
        fn create_note(
            &self,
            notetype_id: i64,
            deck_id: i64,
            fields: &[String],
            tags: &[String],
            policy: DuplicatePolicy,
        ) -> NativeResult<CreateOutcome> {
            Self::create_note(self, notetype_id, deck_id, fields, tags, policy)
        }
        fn update_note(
            &self,
            note_id: i64,
            fields: &[String],
            tags: Option<&[String]>,
        ) -> NativeResult<()> {
            Self::update_note(self, note_id, fields, tags)
        }
        fn delete_notes(&self, note_ids: &[i64]) -> NativeResult<usize> {
            Self::delete_notes(self, note_ids)
        }
        fn upsert_notes(
            &self,
            notes: &[NoteInput],
            policy: DuplicatePolicy,
            dry_run: bool,
        ) -> NativeResult<Vec<UpsertNoteResult>> {
            Self::upsert_notes(self, notes, policy, dry_run)
        }
        fn find_replace_notes(
            &self,
            note_ids: &[i64],
            search: &str,
            replacement: &str,
            regex: bool,
            match_case: bool,
            field_name: Option<&str>,
        ) -> NativeResult<String> {
            Self::find_replace_notes(
                self,
                note_ids,
                search,
                replacement,
                regex,
                match_case,
                field_name,
            )
        }
        fn update_note_tags(
            &self,
            note_ids: &[i64],
            set_tags: Option<&[String]>,
            add: &[String],
            remove: &[String],
        ) -> NativeResult<UpdateNoteTagsResponse> {
            Self::update_note_tags(self, note_ids, set_tags, add, remove)
        }
        fn rename_tag(
            &self,
            old: &str,
            new: &str,
            note_ids: &[i64],
        ) -> NativeResult<RenameTagResponse> {
            Self::rename_tag(self, old, new, note_ids)
        }
        fn upsert_decks(&self, decks: &[DeckInput]) -> NativeResult<Vec<UpsertDeckResult>> {
            Self::upsert_decks(self, decks)
        }
        fn delete_decks(&self, refs: &[String]) -> NativeResult<DeleteDecksResponse> {
            Self::delete_decks(self, refs)
        }
        fn upsert_note_types(
            &self,
            note_types: &[NoteTypeInput],
        ) -> NativeResult<Vec<NoteTypeResult>> {
            Self::upsert_note_types(self, note_types)
        }
        fn update_note_type_fields(
            &self,
            note_type_name: &str,
            operations: &[FieldOp],
        ) -> NativeResult<UpdateNoteTypeFieldsResponse> {
            Self::update_note_type_fields(self, note_type_name, operations)
        }
        fn update_note_type_templates(
            &self,
            note_type_name: &str,
            operations: &[TemplateOp],
        ) -> NativeResult<UpdateNoteTypeTemplatesResponse> {
            Self::update_note_type_templates(self, note_type_name, operations)
        }
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
        ) -> NativeResult<FindReplaceNoteTypesResponse> {
            Self::find_and_replace_note_types(
                self,
                note_type_name,
                search,
                replacement,
                regex,
                match_case,
                front,
                back,
                css,
            )
        }
        fn update_note_type_field_metadata(
            &self,
            note_type_name: &str,
            updates: &[FieldMetadataInput],
        ) -> NativeResult<UpdateNoteTypeFieldMetadataResponse> {
            Self::update_note_type_field_metadata(self, note_type_name, updates)
        }
        fn migrate_note_type(
            &self,
            note_ids: &[i64],
            new_note_type: &str,
            field_map: &BTreeMap<String, String>,
            template_map: &BTreeMap<String, String>,
            dry_run: bool,
        ) -> NativeResult<MigrateNoteTypeResponse> {
            Self::migrate_note_type(
                self,
                note_ids,
                new_note_type,
                field_map,
                template_map,
                dry_run,
            )
        }
        fn delete_note_types(&self, ids: &[i64]) -> NativeResult<Vec<DeleteNoteTypeResult>> {
            Self::delete_note_types(self, ids)
        }
        fn store_media_bytes(
            &self,
            filename: Option<&str>,
            data: &[u8],
            content_type: Option<&str>,
        ) -> NativeResult<StoreMediaResult> {
            Self::store_media_bytes(self, filename, data, content_type)
        }
        fn store_media_items(
            &self,
            items: &[StoreMediaItem],
            allow_private_fetch: bool,
            path_roots: &[String],
        ) -> NativeResult<Vec<StoreMediaResult>> {
            Self::store_media_items(self, items, allow_private_fetch, path_roots)
        }
        fn store_prepared_media(
            &self,
            prepared: &[PreparedMedia],
            path_roots: &[String],
        ) -> NativeResult<Vec<StoreMediaResult>> {
            Self::store_prepared_media(self, prepared, path_roots)
        }
        fn fetch_media(&self, filenames: &[String]) -> NativeResult<Vec<MediaFetchResult>> {
            Self::fetch_media(self, filenames)
        }
        fn list_media(
            &self,
            pattern: Option<&str>,
            limit: Option<usize>,
        ) -> NativeResult<ListMediaResponse> {
            Self::list_media(self, pattern, limit)
        }
        fn delete_media(&self, filenames: &[String]) -> NativeResult<DeleteMediaResponse> {
            Self::delete_media(self, filenames)
        }
        fn media_check(&self) -> NativeResult<CollectionCheckResponse> {
            Self::media_check(self)
        }
        fn prune(
            &self,
            unused_tags: bool,
            empty_notes: bool,
            empty_cards: bool,
            unused_media: bool,
            dry_run: bool,
        ) -> NativeResult<(CollectionPruneResponse, Vec<i64>)> {
            Self::prune(
                self,
                unused_tags,
                empty_notes,
                empty_cards,
                unused_media,
                dry_run,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    //! The slice-1 parity floor AND the index tripwires: every hardcoded
    //! (service, method) pair is exercised against a real temp collection, so
    //! a tag bump that shuffles the generated dispatcher fails these tests
    //! instead of corrupting calls.

    use super::*;

    /// Test shim: drive the typed upsert with a JSON literal, assert on the
    /// serialized results (the pre-#391 call shape the assertions were
    /// written against).
    fn upsert_json(
        core: &CollectionCore,
        notes_json: &str,
        on_duplicate: &str,
        dry_run: bool,
    ) -> serde_json::Value {
        let notes: Vec<shrike_schemas::NoteInput> = serde_json::from_str(notes_json).unwrap();
        let policy = DuplicatePolicy::parse(on_duplicate).unwrap();
        let results = core.upsert_notes(&notes, policy, dry_run).unwrap();
        serde_json::to_value(&results).unwrap()
    }

    /// The deck counterpart (#391): JSON literals in, the typed op's
    /// serialized results out, keeping the pre-typed assertions verbatim.
    fn upsert_decks_json(core: &CollectionCore, decks_json: &str) -> serde_json::Value {
        let decks: Vec<shrike_schemas::DeckInput> = serde_json::from_str(decks_json).unwrap();
        let results = core.upsert_decks(&decks).unwrap();
        serde_json::to_value(&results).unwrap()
    }

    /// The note-type counterparts (#391): JSON literals in, the typed ops'
    /// serialized results out, keeping the pre-typed assertions verbatim.
    fn note_types_json(core: &CollectionCore, json_str: &str) -> serde_json::Value {
        let inputs: Vec<shrike_schemas::NoteTypeInput> = serde_json::from_str(json_str).unwrap();
        serde_json::to_value(core.upsert_note_types(&inputs).unwrap()).unwrap()
    }

    fn field_ops_json(
        core: &CollectionCore,
        name: &str,
        ops_json: &str,
    ) -> shrike_ffi::NativeResult<serde_json::Value> {
        let ops: Vec<shrike_schemas::FieldOp> = serde_json::from_str(ops_json).unwrap();
        Ok(serde_json::to_value(core.update_note_type_fields(name, &ops)?).unwrap())
    }

    fn template_ops_json(
        core: &CollectionCore,
        name: &str,
        ops_json: &str,
    ) -> shrike_ffi::NativeResult<serde_json::Value> {
        let ops: Vec<shrike_schemas::TemplateOp> = serde_json::from_str(ops_json).unwrap();
        Ok(serde_json::to_value(core.update_note_type_templates(name, &ops)?).unwrap())
    }

    fn field_metadata_json(
        core: &CollectionCore,
        name: &str,
        updates_json: &str,
    ) -> serde_json::Value {
        let updates: Vec<shrike_schemas::FieldMetadataInput> =
            serde_json::from_str(updates_json).unwrap();
        serde_json::to_value(
            core.update_note_type_field_metadata(name, &updates)
                .unwrap(),
        )
        .unwrap()
    }

    fn migrate_json(
        core: &CollectionCore,
        ids: &[i64],
        to: &str,
        fmap_json: &str,
        tmap_json: &str,
        dry_run: bool,
    ) -> serde_json::Value {
        let fmap: std::collections::BTreeMap<String, String> =
            serde_json::from_str(fmap_json).unwrap();
        let tmap: std::collections::BTreeMap<String, String> = if tmap_json.is_empty() {
            Default::default()
        } else {
            serde_json::from_str(tmap_json).unwrap()
        };
        serde_json::to_value(
            core.migrate_note_type(ids, to, &fmap, &tmap, dry_run)
                .unwrap(),
        )
        .unwrap()
    }

    fn temp_core() -> (CollectionCore, std::path::PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "shrike-collection-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("collection.anki2");
        (CollectionCore::open(path.to_str().unwrap()).unwrap(), dir)
    }

    const DEFAULT_DECK: i64 = 1;

    #[test]
    fn open_create_search_read_update_delete_round_trip() {
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();

        let outcome = core
            .create_note(
                basic,
                DEFAULT_DECK,
                &["front text".into(), "back text".into()],
                &["tag-a".into()],
                DuplicatePolicy::Error,
            )
            .unwrap();
        let CreateOutcome::Created(nid) = outcome else {
            panic!("expected create")
        };

        let found = core.find_notes("deck:*").unwrap();
        assert_eq!(found, vec![nid]);
        let note = core.get_note(nid).unwrap();
        assert_eq!(note.fields[0], "front text");
        assert_eq!(note.tags, vec!["tag-a".to_string()]);

        core.update_note(nid, &["front text".into(), "new back".into()], None)
            .unwrap();
        assert_eq!(core.get_note(nid).unwrap().fields[1], "new back");

        assert_eq!(core.delete_notes(&[nid]).unwrap(), 1);
        assert!(core.find_notes("deck:*").unwrap().is_empty());
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn duplicate_policy_matrix() {
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();
        let fields = vec!["same front".to_string(), "back".to_string()];
        core.create_note(basic, DEFAULT_DECK, &fields, &[], DuplicatePolicy::Error)
            .unwrap();

        // error (the default policy): reported, not written
        let err = core
            .create_note(basic, DEFAULT_DECK, &fields, &[], DuplicatePolicy::Error)
            .unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::InvalidInput);
        // skip: not written, reported as skipped
        let skipped = core
            .create_note(basic, DEFAULT_DECK, &fields, &[], DuplicatePolicy::Skip)
            .unwrap();
        assert_eq!(skipped, CreateOutcome::SkippedDuplicate);
        // allow: written anyway
        let allowed = core
            .create_note(basic, DEFAULT_DECK, &fields, &[], DuplicatePolicy::Allow)
            .unwrap();
        assert!(matches!(allowed, CreateOutcome::Created(_)));
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 2);
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn empty_first_field_always_errors() {
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();
        let err = core
            .create_note(
                basic,
                DEFAULT_DECK,
                &["".into(), "back".into()],
                &[],
                DuplicatePolicy::Allow, // policy never overrides structural errors
            )
            .unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::InvalidInput);
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn col_mod_advances_on_write() {
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();
        let before = core.col_mod().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        core.create_note(
            basic,
            DEFAULT_DECK,
            &["a".into(), "b".into()],
            &[],
            DuplicatePolicy::Error,
        )
        .unwrap();
        let after = core.col_mod().unwrap();
        assert!(after >= before, "col.mod must not move backwards");
        assert!(after > 0);
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn read_surface_round_trip() {
        // Tripwires for the step-2 (service, method) indices — deck_names,
        // deck_tree, all_tags, notetype, strip_html, db_rows — plus the
        // shape of the ported readers, all against a real temp collection.
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();
        let CreateOutcome::Created(nid) = core
            .create_note(
                basic,
                DEFAULT_DECK,
                &[
                    "the <b>mitochondria</b>&nbsp;powerhouse".into(),
                    "energy of the cell<br>line2".into(),
                ],
                &["bio".into()],
                DuplicatePolicy::Error,
            )
            .unwrap()
        else {
            panic!("create failed")
        };

        // note_texts: cloze revealed, HTML stripped, NBSP folded, block tag
        // spaced, "Name: text" render.
        let texts = core.note_texts(&[nid, 999]).unwrap();
        assert_eq!(
            texts[0],
            "Front: the mitochondria powerhouse\nBack: energy of the cell line2"
        );
        assert_eq!(texts[1], ""); // missing id → empty at position

        // Cloze reveal + sound/math wrappers through the REAL service
        // stripper (fields_check forbids cloze markup in a Basic note, so the
        // normalization is pinned directly).
        assert_eq!(
            core.normalize_text("{{c1::energy::hint}} of [sound:x.mp3] \\(E\\)")
                .unwrap(),
            "energy of E"
        );

        // note_embed_inputs + derived_field_rows shapes.
        let inputs = core.note_embed_inputs(&[nid]).unwrap();
        assert_eq!(inputs[0].0, nid);
        assert!(inputs[0].2.is_empty()); // no images
        let rows = core.derived_field_rows(&[nid]).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].1, "field");
        assert_eq!(rows[0].2, "Front");

        // list_notes: tag filter, full fields, wire shape (typed since #391
        // phase 2; asserted through the host-edge wire view — plain serde).
        let listed = serde_json::to_value(
            core.list_notes(None, None, Some(&["bio".into()]), None, None, true, 50)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(listed["total"], 1);
        let note = &listed["notes"][0];
        assert_eq!(note["id"], nid);
        assert_eq!(note["note_type"], "Basic");
        assert_eq!(note["deck"], "Default");
        assert_eq!(note["tags"][0], "bio");
        assert!(note["content"]["Front"]
            .as_str()
            .unwrap()
            .contains("mitochondria"));
        assert!(note["modified"].as_str().unwrap().ends_with("+00:00"));

        // deck reference forms: name, id, #id, unknown #id.
        let by_id = serde_json::to_value(
            core.list_notes(None, Some("1"), None, None, None, false, 50)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(by_id["total"], 1);
        let unknown = serde_json::to_value(
            core.list_notes(None, Some("#424242"), None, None, None, false, 50)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(unknown["total"], 0);

        // No filter at all → the expected-input error tier.
        let err = core
            .list_notes(None, None, None, None, None, false, 50)
            .unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::InvalidInput);

        // collection_info: all sections, summary/stats/decks coherent.
        let info = serde_json::to_value(
            core.collection_info(&["all".to_string()], &["Basic".to_string()])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(info["summary"]["notes"], 1);
        assert_eq!(info["summary"]["path"], core.collection_path.as_str());
        assert!(info["tags"].as_array().unwrap().iter().any(|t| t == "bio"));
        let basic_nt = info["note_types"]
            .as_array()
            .unwrap()
            .iter()
            .find(|nt| nt["name"] == "Basic")
            .unwrap();
        assert_eq!(basic_nt["type"], "standard");
        assert_eq!(basic_nt["fields"][0], "Front");
        assert!(basic_nt["detail"]["templates"][0]["front"]
            .as_str()
            .unwrap()
            .contains("{{Front}}"));
        let default_deck = info["decks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["name"] == "Default")
            .unwrap();
        assert_eq!(default_deck["note_count"], 1);
        assert_eq!(info["stats"]["total_notes"], 1);
        assert_eq!(info["stats"]["decks_summary"]["Default"]["notes"], 1);

        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn read_wire_is_plain_serde_with_explicit_nulls() {
        // #391 phase 2 (the to_wire retirement): ONE wire convention — plain
        // serde of the schema types, where an unset `Option` is an explicit
        // `null`, never a pruned key (the Pydantic shape the schema contract
        // test pins). Shape-level, deliberately not byte-level: every
        // consumer revalidates through the Pydantic models, so the contract
        // is "parses back into the schema type with the same content".
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();
        let CreateOutcome::Created(nid) = core
            .create_note(
                basic,
                DEFAULT_DECK,
                &["alpha".into(), "beta".into()],
                &["t1".into()],
                DuplicatePolicy::Error,
            )
            .unwrap()
        else {
            panic!("create failed")
        };

        // Meta mode: `content` is an explicit null on the wire, and the
        // payload round-trips losslessly into the schema type.
        let meta = core
            .list_notes(Some(&[nid]), None, None, None, None, false, 50)
            .unwrap();
        let wire = serde_json::to_string(&meta).unwrap();
        let value: serde_json::Value = serde_json::from_str(&wire).unwrap();
        assert_eq!(value["notes"][0]["content"], serde_json::Value::Null);
        let back: shrike_schemas::ListNotesResponse = serde_json::from_str(&wire).unwrap();
        assert!(back.notes[0].content.is_none());
        assert_eq!(back.notes[0].id, nid);
        assert_eq!(back.total, 1);

        // `query` rides the same response shape.
        let queried = core.query("tag:t1", false, 10).unwrap();
        let qvalue = serde_json::to_value(&queried).unwrap();
        assert_eq!(qvalue["notes"][0]["content"], serde_json::Value::Null);
        assert_eq!(qvalue["limit"], 10);

        // collection_info: an unrequested section is an explicit null, a
        // requested one an object — and the payload validates back.
        let info = core
            .collection_info(&["summary".into(), "decks".into()], &[])
            .unwrap();
        let wire = serde_json::to_string(&info).unwrap();
        let value: serde_json::Value = serde_json::from_str(&wire).unwrap();
        assert!(value["summary"].is_object());
        assert!(value["decks"].is_array());
        assert_eq!(value["stats"], serde_json::Value::Null);
        assert_eq!(value["note_types"], serde_json::Value::Null);
        let back: shrike_schemas::CollectionInfo = serde_json::from_str(&wire).unwrap();
        assert_eq!(back.summary.as_ref().unwrap().notes, 1);
        assert!(back.stats.is_none());

        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn note_types_round_trip() {
        // Step-4 tripwires (the legacy schema11 RPCs) + the ported note-type
        // ops, against a real temp collection — including the data-safety
        // property the #76/#99 history demands: note data survives renames
        // and identity-based moves.
        let (core, dir) = temp_core();

        // Create a custom type (cloze flag off), then a note carrying data.
        let create = serde_json::json!([{
            "name": "Custom",
            "fields": ["A", "B", "C"],
            "templates": [{"name": "Card 1", "front": "{{A}}", "back": "{{B}}"}],
            "css": ".card { color: red; }",
        }]);
        let created = note_types_json(&core, &create.to_string());
        assert_eq!(created[0]["status"], "created");
        let custom_id = created[0]["id"].as_i64().unwrap();
        let CreateOutcome::Created(nid) = core
            .create_note(
                custom_id,
                DEFAULT_DECK,
                &["a-data".into(), "b-data".into(), "c-data".into()],
                &[],
                DuplicatePolicy::Error,
            )
            .unwrap()
        else {
            panic!("create failed")
        };

        // Positional replace: rename-in-place + append is sound and data-safe.
        let update = serde_json::json!([{
            "id": custom_id,
            "fields": ["A2", "B", "C", "D"],
        }]);
        let updated = note_types_json(&core, &update.to_string());
        assert_eq!(updated[0]["status"], "updated");
        assert_eq!(
            core.get_note(nid).unwrap().fields,
            vec!["a-data", "b-data", "c-data", ""]
        );
        // A move is refused with the pointer to the identity tool.
        let bad = serde_json::json!([{"id": custom_id, "fields": ["B", "A2", "C", "D"]}]);
        let rejected = note_types_json(&core, &bad.to_string());
        assert_eq!(rejected[0]["status"], "error");
        assert!(rejected[0]["error"]
            .as_str()
            .unwrap()
            .contains("update_note_type_fields"));

        // Identity ops: a true move + a non-trailing remove migrate data by
        // identity (a-data follows A2; b-data is dropped with B).
        let ops = serde_json::json!([
            {"op": "reposition", "name": "A2", "position": 2},
            {"op": "remove", "name": "B"},
        ]);
        let result = field_ops_json(&core, "Custom", &ops.to_string()).unwrap();
        assert_eq!(result["fields"], serde_json::json!(["C", "A2", "D"]));
        assert_eq!(
            core.get_note(nid).unwrap().fields,
            vec!["c-data", "a-data", ""]
        );
        // Invalid op sequence changes nothing (atomic).
        let bad_ops = serde_json::json!([
            {"op": "rename", "name": "C", "new_name": "C2"},
            {"op": "remove", "name": "Ghost"},
        ]);
        let err = field_ops_json(&core, "Custom", &bad_ops.to_string()).unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::InvalidInput);
        assert_eq!(
            core.notetype_field_names(custom_id).unwrap(),
            vec!["C", "A2", "D"]
        );

        // Template identity ops: add + rename (pure label change).
        let tops = serde_json::json!([
            {"op": "add", "name": "Card 2", "front": "{{C}}", "back": "{{A2}}"},
            {"op": "rename", "name": "Card 1", "new_name": "Primary"},
        ]);
        let tresult = template_ops_json(&core, "Custom", &tops.to_string()).unwrap();
        assert_eq!(
            tresult["templates"],
            serde_json::json!(["Primary", "Card 2"])
        );

        // find_and_replace_note_types: literal + regex with a Python group ref.
        let fr = core
            .find_and_replace_note_types(
                "Custom",
                "color: red",
                "color: blue",
                false,
                true,
                true,
                true,
                true,
            )
            .unwrap();
        assert_eq!(fr.replacements, 1);
        assert!(fr.css_changed);
        let fr2 = core
            .find_and_replace_note_types(
                "Custom",
                r"\{\{(A2)\}\}",
                r"<b>{{\1}}</b>",
                true,
                true,
                true,
                false,
                false,
            )
            .unwrap();
        assert_eq!(fr2.replacements, 1);
        let info = serde_json::to_value(
            core.collection_info(&["note_types".to_string()], &["Custom".to_string()])
                .unwrap(),
        )
        .unwrap();
        let custom = info["note_types"]
            .as_array()
            .unwrap()
            .iter()
            .find(|nt| nt["name"] == "Custom")
            .unwrap();
        assert!(custom["detail"]["css"].as_str().unwrap().contains("blue"));

        // Field metadata: set + atomic validation.
        let meta = serde_json::json!([{"name": "A2", "font": "Courier", "size": 14}]);
        let mresult = field_metadata_json(&core, "Custom", &meta.to_string());
        assert_eq!(mresult["fields_updated"], serde_json::json!(["A2"]));

        // migrate_note_type: Custom -> Basic, dropping a field; dry_run first.
        let fmap = serde_json::json!({"A2": "Front", "C": "Back"}).to_string();
        let dry = migrate_json(&core, &[nid], "Basic", &fmap, "", true);
        assert_eq!(dry["dropped_fields"], serde_json::json!(["D"]));
        assert_eq!(dry["dry_run"], true);
        // dry run changed nothing
        assert_eq!(core.get_note(nid).unwrap().notetype_id, custom_id);
        let applied = migrate_json(&core, &[nid], "Basic", &fmap, "", false);
        assert_eq!(applied["to_note_type"], "Basic");
        let migrated = core.get_note(nid).unwrap();
        let basic = core.notetype_id("Basic").unwrap();
        assert_eq!(migrated.notetype_id, basic);
        assert_eq!(migrated.fields, vec!["a-data", "c-data"]);

        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn note_type_results_wire_shape() {
        // Pin the host-parsed wire of the typed note-type returns (#391):
        // status tags and field names exactly as the Pydantic models expect.
        use shrike_schemas::{
            DeleteNoteTypeResult, FindReplaceNoteTypesResponse, MigrateNoteTypeResponse,
            NoteTypeResult, UpdateNoteTypeFieldMetadataResponse, UpdateNoteTypeFieldsResponse,
            UpdateNoteTypeTemplatesResponse,
        };

        let v = serde_json::to_value(NoteTypeResult::Created {
            id: 5,
            name: "T".into(),
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"status": "created", "id": 5, "name": "T"})
        );
        let v = serde_json::to_value(NoteTypeResult::Updated {
            id: 5,
            name: "T".into(),
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"status": "updated", "id": 5, "name": "T"})
        );
        let v = serde_json::to_value(NoteTypeResult::Error {
            index: 1,
            error: "boom".into(),
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"status": "error", "index": 1, "error": "boom"})
        );

        let v = serde_json::to_value(UpdateNoteTypeFieldsResponse {
            id: 1,
            name: "T".into(),
            fields: vec!["A".into()],
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"id": 1, "name": "T", "fields": ["A"]})
        );
        let v = serde_json::to_value(UpdateNoteTypeTemplatesResponse {
            id: 1,
            name: "T".into(),
            templates: vec!["C1".into()],
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"id": 1, "name": "T", "templates": ["C1"]})
        );
        let v = serde_json::to_value(UpdateNoteTypeFieldMetadataResponse {
            id: 1,
            name: "T".into(),
            fields_updated: vec!["A".into()],
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"id": 1, "name": "T", "fields_updated": ["A"]})
        );
        let v = serde_json::to_value(FindReplaceNoteTypesResponse {
            id: 1,
            name: "T".into(),
            replacements: 2,
            templates_changed: vec!["C1".into()],
            css_changed: false,
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "id": 1, "name": "T", "replacements": 2,
                "templates_changed": ["C1"], "css_changed": false
            })
        );
        let v = serde_json::to_value(MigrateNoteTypeResponse {
            changed: vec![9],
            from_note_type: "A".into(),
            to_note_type: "B".into(),
            dropped_fields: vec!["X".into()],
            new_empty_fields: vec![],
            dry_run: true,
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "changed": [9], "from_note_type": "A", "to_note_type": "B",
                "dropped_fields": ["X"], "new_empty_fields": [], "dry_run": true
            })
        );

        let v = serde_json::to_value(DeleteNoteTypeResult::Deleted {
            id: 3,
            name: "T".into(),
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"status": "deleted", "id": 3, "name": "T"})
        );
        let v = serde_json::to_value(DeleteNoteTypeResult::NotFound { id: 3 }).unwrap();
        assert_eq!(v, serde_json::json!({"status": "not_found", "id": 3}));
        let v = serde_json::to_value(DeleteNoteTypeResult::Error {
            id: 3,
            name: "T".into(),
            error: "in use".into(),
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"status": "error", "id": 3, "name": "T", "error": "in use"})
        );
    }

    #[test]
    fn media_and_prune_round_trip() {
        // Step-5a tripwires (media + maintenance RPCs) + the ported ops.
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();

        // Store: bytes in, Anki-resolved name out; collision dedups/renames.
        // Serialize through the wire types so the assertions also pin the
        // tagged-union shape the host parses.
        let stored = serde_json::to_value(
            core.store_media_bytes(Some("pic.png"), b"PNGDATA", None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stored["status"], "stored");
        assert_eq!(stored["filename"], "pic.png");
        assert_eq!(stored["deduped"], false);
        let same = serde_json::to_value(
            core.store_media_bytes(Some("pic.png"), b"PNGDATA", None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(same["filename"], "pic.png"); // identical content → same name
        let diff = serde_json::to_value(
            core.store_media_bytes(Some("pic.png"), b"OTHERDATA", None)
                .unwrap(),
        )
        .unwrap();
        assert_ne!(diff["filename"], "pic.png"); // different content → suffixed
        assert_eq!(diff["deduped"], false); // different content: renamed, not deduped

        // fetch/list with the traversal guard + glob.
        let fetched = serde_json::to_value(
            core.fetch_media(&["pic.png".into(), "../pic.png".into(), "ghost.png".into()])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(fetched[0]["status"], "found");
        assert_eq!(fetched[0]["mime"], "image/png");
        assert_eq!(fetched[1]["status"], "found"); // basename guard resolves it
        assert_eq!(fetched[2]["status"], "missing");
        let listing = core.list_media(Some("pic*"), None).unwrap();
        assert_eq!(listing.count, 2);

        // A note referencing pic.png; the other file is unused.
        core.create_note(
            basic,
            DEFAULT_DECK,
            &["<img src=\"pic.png\">".into(), "kept".into()],
            &["usedtag".into()],
            DuplicatePolicy::Error,
        )
        .unwrap();
        // An empty note (no text, no media) for the prune.
        let empty_batch = serde_json::json!([
            {"note_type": "Basic", "deck": "Default",
             "fields": {"Front": "<b> </b>&nbsp;", "Back": ""}}
        ]);
        // fields_check calls this EMPTY, so create it via allow + raw create.
        let raw = upsert_json(&core, &empty_batch.to_string(), "allow", false);
        assert_eq!(raw[0]["status"], "error"); // structurally empty is never written
                                               // Insert a genuinely empty-able note: text now, blanked by update.
        let CreateOutcome::Created(empty_nid) = core
            .create_note(
                basic,
                DEFAULT_DECK,
                &["temp".into(), "".into()],
                &["onlytag".into()],
                DuplicatePolicy::Error,
            )
            .unwrap()
        else {
            panic!("create failed")
        };
        core.update_note(empty_nid, &["<b> </b>&nbsp;".into(), "".into()], None)
            .unwrap();

        // media check sees the unused file.
        let check = core.media_check().unwrap();
        assert_eq!(check.unused.len(), 1);
        assert_ne!(check.unused[0], "pic.png");

        // Dry-run prune: previews everything, mutates nothing.
        let (preview, _) = core.prune(true, true, true, true, true).unwrap();
        assert!(preview.dry_run);
        assert_eq!(preview.empty_notes.unwrap().removed, vec![empty_nid]);
        assert_eq!(preview.unused_media.unwrap().removed, 1);
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 2);

        // Apply: empty note gone (its tag freed and cleared), media trashed.
        let (applied, removed_note_ids) = core.prune(true, true, true, true, false).unwrap();
        assert_eq!(removed_note_ids, vec![empty_nid]);
        assert!(applied
            .unused_tags
            .unwrap()
            .tags
            .iter()
            .any(|t| t == "onlytag"));
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 1);
        assert_eq!(core.list_media(None, None).unwrap().count, 1);

        // delete_media: trash + echo, not_found for ghosts.
        let deleted = core
            .delete_media(&["pic.png".into(), "nope.png".into()])
            .unwrap();
        assert_eq!(deleted.deleted, vec!["pic.png"]);
        assert_eq!(deleted.not_found, vec!["nope.png"]);

        // The byte-source size cap (the path source is deliberately uncapped).
        let oversize = vec![0u8; crate::media_fetch::MEDIA_MAX_BYTES + 1];
        assert!(core
            .store_media_bytes(Some("big.bin"), &oversize, None)
            .is_err());

        // store_media_items: typed input, per-item errors never sink the
        // batch (a sourceless item fails its own slot only).
        let items = vec![
            shrike_schemas::StoreMediaItem {
                filename: Some("from-batch.png".into()),
                data: Some("QkFUQ0g=".into()), // b64("BATCH")
                ..Default::default()
            },
            shrike_schemas::StoreMediaItem::default(),
        ];
        let batch = core.store_media_items(&items, false, &[]).unwrap();
        assert!(matches!(
            batch[0],
            shrike_schemas::StoreMediaResult::Stored { index: 0, .. }
        ));
        assert!(matches!(
            &batch[1],
            shrike_schemas::StoreMediaResult::Error { index: 1, .. }
        ));

        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn busy_surface_release_reopen() {
        // The #64/#65 contention story: a second holder makes reopen BUSY;
        // releasing hands the lock over; reopening after the holder leaves
        // succeeds. (The second core never successfully co-manages the
        // collection — it exists to HOLD the lock, which is the scenario the
        // busy tier is for.)
        let (core, dir) = temp_core();
        let path = core.collection_path.clone();
        let basic = core.notetype_id("Basic").unwrap();

        // release → another core can open (cooperative time-slicing).
        core.release().unwrap();
        let holder = CollectionCore::open(&path).unwrap();

        // reopen while held → the BUSY tier, message intact.
        let err = core.reopen().unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::Busy);
        assert!(err.message.contains("in use by another process"));

        // holder leaves → reopen succeeds and ops work again.
        holder.close().unwrap();
        core.reopen().unwrap();
        core.create_note(
            basic,
            DEFAULT_DECK,
            &["after reopen".into(), "b".into()],
            &[],
            DuplicatePolicy::Error,
        )
        .unwrap();
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 1);

        // release is idempotent; double reopen is fine.
        core.release().unwrap();
        core.release().unwrap();
        core.reopen().unwrap();

        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn write_surface_round_trip() {
        // Tripwires for the step-3 (service, method) indices and the ported
        // write ops, against a real temp collection.
        let (core, dir) = temp_core();

        // Named-fields upsert: create (auto-creating a nested deck), with
        // the #77 policy + reason vocabulary.
        let batch = serde_json::json!([
            {"note_type": "Basic", "deck": "Science::Physics",
             "fields": {"Front": "alpha", "Back": "beta"}, "tags": ["t1"]},
            {"note_type": "Basic", "deck": "Science::Physics",
             "fields": {"Front": "alpha", "Back": "dup"}},
            {"note_type": "Nope", "deck": "D", "fields": {"Front": "x"}},
            {"note_type": "Basic", "deck": "D", "fields": {"Bogus": "x"}},
        ]);
        let results = upsert_json(&core, &batch.to_string(), "skip", false);
        assert_eq!(results[0]["status"], "created");
        let nid = results[0]["id"].as_i64().unwrap();
        assert_eq!(results[1]["status"], "skipped");
        assert_eq!(results[1]["reason"], "duplicate");
        assert_eq!(results[2]["reason"], "unknown_note_type");
        assert_eq!(results[3]["reason"], "unknown_field");
        // The nested deck was auto-created; the bogus items created nothing.
        assert!(core
            .adapter
            .deck_id_by_name("Science::Physics")
            .unwrap()
            .is_some());
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 1);

        // dry_run validates but writes nothing.
        let dry = serde_json::json!([
            {"note_type": "Basic", "deck": "DryDeck", "fields": {"Front": "new", "Back": "b"}}
        ]);
        let dry_results = upsert_json(&core, &dry.to_string(), "error", true);
        assert_eq!(dry_results[0]["status"], "ok");
        assert_eq!(dry_results[0]["action"], "create");
        assert!(core.adapter.deck_id_by_name("DryDeck").unwrap().is_none());
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 1);

        // Update: partial fields, tags, deck move; type change refused.
        let update = serde_json::json!([
            {"id": nid, "fields": {"Back": "new back"}, "tags": ["t2"], "deck": "Default"}
        ]);
        let up_results = upsert_json(&core, &update.to_string(), "error", false);
        assert_eq!(up_results[0]["status"], "updated");
        let note = core.get_note(nid).unwrap();
        assert_eq!(
            note.fields,
            vec!["alpha".to_string(), "new back".to_string()]
        );
        assert_eq!(note.tags, vec!["t2".to_string()]);
        assert_eq!(core.find_notes("\"deck:Default\"").unwrap(), vec![nid]);

        // Tags: add/remove (remove-before-add), set replace, not_found.
        let tags_result = core
            .update_note_tags(&[nid, 999], None, &["x1".into()], &["t2".into()])
            .unwrap();
        assert_eq!(tags_result.notes_modified, 1);
        assert_eq!(tags_result.not_found, vec![999]);
        assert_eq!(core.get_note(nid).unwrap().tags, vec!["x1".to_string()]);
        core.update_note_tags(&[nid], Some(&["fresh".into()]), &[], &[])
            .unwrap();
        assert_eq!(core.get_note(nid).unwrap().tags, vec!["fresh".to_string()]);

        // rename_tag: exact on a note set, then collection-wide.
        assert_eq!(
            core.rename_tag("fresh", "renamed", &[nid])
                .unwrap()
                .notes_modified,
            1
        );
        assert_eq!(
            core.get_note(nid).unwrap().tags,
            vec!["renamed".to_string()]
        );
        assert_eq!(
            core.rename_tag("renamed", "global", &[])
                .unwrap()
                .notes_modified,
            1
        );

        // Decks: upsert rename + clash, delete empty-only. Serialize through
        // the wire types so the assertions also pin the tagged-union shape.
        let physics = core
            .adapter
            .deck_id_by_name("Science::Physics")
            .unwrap()
            .unwrap();
        let deck_results = upsert_decks_json(
            &core,
            &serde_json::json!([
                {"id": physics, "name": "Science::Mechanics"},
                {"name": "Empty::Leaf"},
            ])
            .to_string(),
        );
        assert_eq!(deck_results[0]["status"], "updated");
        assert_eq!(deck_results[1]["status"], "created");
        let clash_results = upsert_decks_json(
            &core,
            &serde_json::json!([{"id": physics, "name": "Default"}]).to_string(),
        );
        assert_eq!(clash_results[0]["status"], "error");

        let del = core
            .delete_decks(&["Empty::Leaf".into(), "Default".into(), "Ghost".into()])
            .unwrap();
        assert_eq!(del.deleted, vec!["Empty::Leaf"]);
        assert_eq!(del.not_empty, vec!["Default"]); // holds the note's card
        assert_eq!(del.not_found, vec!["Ghost"]);

        // find_replace_notes: literal apply + changed-id diff.
        let fr: serde_json::Value = serde_json::from_str(
            &core
                .find_replace_notes(&[nid], "alpha", "omega", false, true, None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(fr["notes_changed"], 1);
        assert_eq!(fr["changed_ids"][0], nid);
        assert_eq!(core.get_note(nid).unwrap().fields[0], "omega");

        // delete_note_types: in-use error / not_found; (no unused stock type
        // is guaranteed, so the deleted path is covered by the binding tests).
        let basic = core.notetype_id("Basic").unwrap();
        let dnt = serde_json::to_value(core.delete_note_types(&[basic, 12345]).unwrap()).unwrap();
        assert_eq!(dnt[0]["status"], "error");
        assert_eq!(dnt[1]["status"], "not_found");
        // An unused stock type deletes cleanly.
        let cloze = core.notetype_id("Cloze").unwrap();
        let dnt2 = serde_json::to_value(core.delete_note_types(&[cloze]).unwrap()).unwrap();
        assert_eq!(dnt2[0]["status"], "deleted");

        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn full_search_grammar_works() {
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();
        core.create_note(
            basic,
            DEFAULT_DECK,
            &["alpha".into(), "beta".into()],
            &["mytag".into()],
            DuplicatePolicy::Error,
        )
        .unwrap();
        assert_eq!(core.find_notes("tag:mytag").unwrap().len(), 1);
        assert_eq!(core.find_notes("tag:nope").unwrap().len(), 0);
        assert_eq!(core.find_notes("alpha").unwrap().len(), 1);
        // A malformed expression is the expected-input error tier.
        let err = core.find_notes("added:notanumber").unwrap_err();
        assert_eq!(err.kind, shrike_ffi::ErrorKind::InvalidInput);
        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }
    /// #394 (interim gate): every hand-transcribed `(service, method)` index
    /// in adapter.rs must be EXERCISED against a real collection — a bumped
    /// anki whose dispatcher reordered would shift indices silently if any
    /// constant escaped its tripwire. The test self-scans the constants from
    /// the source (the SVC_ pin's pattern), drives the whole public surface
    /// once, and asserts the dispatch recorder saw every pair. Build-time
    /// derivation from anki's descriptors remains the preferred end-state.
    ///
    /// The recorder fires at dispatch (before the call returns): this gate is
    /// REACHABILITY; the sibling round-trip tests validate responses. Note
    /// the parser treats every bare `const X: u32` in adapter.rs as a method
    /// index (SVC_-prefixed ones as services) — an unrelated u32 const there
    /// panics this test loudly rather than passing falsely.
    #[test]
    fn every_method_constant_is_dispatched_by_the_surface() {
        // Parse `const NAME: u32 = N;` declarations out of adapter.rs.
        let src = include_str!("adapter.rs");
        let mut svc: std::collections::BTreeMap<&str, u32> = Default::default();
        let mut methods: Vec<(String, u32)> = Vec::new();
        for line in src.lines() {
            let Some(rest) = line.trim().strip_prefix("const ") else {
                continue;
            };
            let Some((name, value)) = rest.split_once(": u32 = ") else {
                continue;
            };
            let Ok(value) = value.trim_end_matches(';').parse::<u32>() else {
                continue;
            };
            if let Some(s) = name.strip_prefix("SVC_") {
                svc.insert(Box::leak(s.to_string().into_boxed_str()), value);
            } else {
                methods.push((name.to_string(), value));
            }
        }
        assert!(svc.len() >= 9, "service constants parsed: {svc:?}");
        assert!(
            methods.len() >= 30,
            "method constants parsed: {}",
            methods.len()
        );
        // Longest-prefix service resolution (CARD_RENDERING before CARDS).
        let mut prefixes: Vec<(&str, u32)> = svc.iter().map(|(k, v)| (*k, *v)).collect();
        prefixes.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
        let expected: Vec<(String, u32, u32)> = methods
            .iter()
            .map(|(name, m)| {
                let (_, s) = prefixes
                    .iter()
                    .find(|(p, _)| name.starts_with(&format!("{p}_")))
                    .unwrap_or_else(|| panic!("no service prefix for {name}"));
                (name.clone(), *s, *m)
            })
            .collect();

        // Drive the whole public surface once.
        let (core, dir) = temp_core();
        let parsed = upsert_json(
            &core,
            r#"[{"note_type":"Basic","deck":"Drive","fields":{"Front":"alpha <b>one</b>","Back":"a"}},
                {"note_type":"Basic","deck":"Drive","fields":{"Front":"beta two","Back":"b"}}]"#,
            "error",
            false,
        );
        let id_a = parsed[0]["id"].as_i64().unwrap();
        let id_b = parsed[1]["id"].as_i64().unwrap();
        // Duplicate-checked create (fields_check) + an update (deck move →
        // set_card_deck) + a plain field update.
        upsert_json(
            &core,
            r#"[{"note_type":"Basic","deck":"Drive","fields":{"Front":"alpha <b>one</b>","Back":"dupe"}}]"#,
            "skip",
            false,
        );
        upsert_json(
            &core,
            &format!(
                r#"[{{"id":{id_a},"deck":"Drive::Moved","fields":{{"Front":"alpha edited","Back":"a"}},"tags":["keep"]}}]"#
            ),
            "allow",
            false,
        );
        core.get_note(id_a).unwrap();
        core.cards_of_note(id_a).unwrap();
        core.note_texts(&[id_a]).unwrap();
        core.find_notes("deck:Drive*").unwrap();
        core.find_replace_notes(&[id_a], "edited", "patched", false, true, None)
            .unwrap();
        core.update_note_tags(&[id_a], None, &["fresh".into()], &[])
            .unwrap();
        core.update_note_tags(&[id_a], None, &[], &["fresh".into()])
            .unwrap();
        core.rename_tag("keep", "kept", &[]).unwrap();
        core.collection_info(&["all".into()], &["Basic".into()])
            .unwrap();

        // Decks: create, rename, delete-empty (and id-by-name resolution).
        upsert_decks_json(&core, r#"[{"name":"Spare"}]"#);
        let decks = upsert_decks_json(&core, r#"[{"name":"Spare2"}]"#);
        let spare2 = decks[0]["id"].as_i64().unwrap();
        upsert_decks_json(&core, &format!(r#"[{{"id":{spare2},"name":"Spare3"}}]"#));
        core.delete_decks(&["Spare3".to_string()]).unwrap();

        // Note types: stock create, positional update, identity ops, template
        // text rewrite, field metadata, migration, delete-unused.
        note_types_json(
            &core,
            r#"[{"name":"DriveType","fields":["F","B"],"templates":[{"name":"Card 1","front":"{{F}}","back":"{{B}}"}],"css":".card{}"}]"#,
        );
        field_ops_json(&core, "DriveType", r#"[{"op":"add","name":"C"}]"#).unwrap();
        template_ops_json(
            &core,
            "DriveType",
            r#"[{"op":"rename","name":"Card 1","new_name":"Card One"}]"#,
        )
        .unwrap();
        core.find_and_replace_note_types(
            "DriveType",
            ".card",
            ".kard",
            false,
            true,
            false,
            false,
            true,
        )
        .unwrap();
        field_metadata_json(
            &core,
            "DriveType",
            r#"[{"name":"F","description":"front"}]"#,
        );
        migrate_json(
            &core,
            &[id_b],
            "DriveType",
            r#"{"Front":"F","Back":"B"}"#,
            "",
            false,
        );
        // An empty CARD must BECOME empty (Anki never creates one): add a
        // template on C, give the migrated note a C value (the card
        // generates), then clear it — the existing card now renders empty
        // and the prune's sweep genuinely dispatches CARDS_REMOVE_CARDS.
        template_ops_json(
            &core,
            "DriveType",
            r#"[{"op":"add","name":"Empty","front":"{{C}}","back":"x"}]"#,
        )
        .unwrap();
        upsert_json(
            &core,
            &format!(r#"[{{"id":{id_b},"fields":{{"F":"beta two","B":"b","C":"temp"}}}}]"#),
            "allow",
            false,
        );
        upsert_json(
            &core,
            &format!(r#"[{{"id":{id_b},"fields":{{"F":"beta two","B":"b","C":""}}}}]"#),
            "allow",
            false,
        );
        note_types_json(
            &core,
            r#"[{"name":"Unused","fields":["X"],"templates":[{"name":"Card 1","front":"{{X}}","back":"{{X}}"}],"css":""}]"#,
        );
        let unused_id = core.notetype_id("Unused").unwrap();
        core.delete_note_types(&[unused_id]).unwrap();

        // Media: store, list, fetch, check, trash; then the prune sweep
        // (unused tags + empty notes/cards + unused media → get_empty_cards,
        // remove_cards, clear_unused_tags, trash_files).
        core.store_media_bytes(Some("drive.png"), b"drive bytes", None)
            .unwrap();
        core.list_media(None, None).unwrap();
        core.fetch_media(&["drive.png".to_string()]).unwrap();
        core.media_check().unwrap();
        core.delete_media(&["drive.png".to_string()]).unwrap();
        core.prune(true, true, true, true, false).unwrap();

        core.delete_notes(&[id_a]).unwrap();
        core.close().unwrap();

        let seen = adapter::DISPATCHED_METHODS
            .lock()
            .expect("dispatch recorder poisoned")
            .clone();
        let missing: Vec<&str> = expected
            .iter()
            .filter(|(_, s, m)| !seen.contains(&(*s, *m)))
            .map(|(name, ..)| name.as_str())
            .collect();
        assert!(
            missing.is_empty(),
            "method constants never dispatched by the public surface: {missing:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
