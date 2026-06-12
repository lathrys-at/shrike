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

pub use adapter::{FieldsState, ServiceAdapter, ServiceNote};
pub use embed_text::{extract_image_refs, EMBED_TEXT_VERSION};
use shrike_ffi::{NativeError, NativeResult};

/// What `create_note` did about a first-field duplicate (mirrors the Python
/// upsert's `on_duplicate` policy surface).
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
            other => Err(NativeError::invalid_input(format!(
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
        // phase 2; asserted through the host-edge wire view).
        let listed = shrike_schemas::to_wire_value(
            &core
                .list_notes(None, None, Some(&["bio".into()]), None, None, true, 50)
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
        let by_id = shrike_schemas::to_wire_value(
            &core
                .list_notes(None, Some("1"), None, None, None, false, 50)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(by_id["total"], 1);
        let unknown = shrike_schemas::to_wire_value(
            &core
                .list_notes(None, Some("#424242"), None, None, None, false, 50)
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
        let info = shrike_schemas::to_wire_value(
            &core
                .collection_info(&["all".to_string()], &["Basic".to_string()])
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
    fn read_wire_bytes_match_the_legacy_hand_built_wire() {
        // #391 phase 2 byte pin: the typed read responses, serialized through
        // the host-edge wire helper, must be byte-identical to the
        // `serde_json::Value` trees the pre-seam code hand-built — compact,
        // key-sorted (Value's map is a BTreeMap; no preserve_order), the
        // `content` key ABSENT in meta mode (never an explicit null), and
        // only the requested collection_info sections present. The oracles
        // below reproduce the legacy `json!` assembly verbatim.
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

        // Full mode.
        let full = core
            .list_notes(Some(&[nid]), None, None, None, None, true, 50)
            .unwrap();
        let n = &full.notes[0];
        let legacy = serde_json::json!({
            "notes": [{
                "id": n.id,
                "note_type": "Basic",
                "deck": "Default",
                "tags": ["t1"],
                "modified": n.modified.clone(),
                "content": {"Front": "alpha", "Back": "beta"},
            }],
            "total": 1,
            "limit": 50,
        })
        .to_string();
        assert_eq!(shrike_schemas::to_wire_json(&full).unwrap(), legacy);

        // Meta mode: no "content" key at all.
        let meta = core
            .list_notes(Some(&[nid]), None, None, None, None, false, 50)
            .unwrap();
        let wire = shrike_schemas::to_wire_json(&meta).unwrap();
        let legacy = serde_json::json!({
            "notes": [{
                "id": n.id,
                "note_type": "Basic",
                "deck": "Default",
                "tags": ["t1"],
                "modified": n.modified.clone(),
            }],
            "total": 1,
            "limit": 50,
        })
        .to_string();
        assert_eq!(wire, legacy);
        assert!(!wire.contains("content"));

        // `query` rides the same serialization (the shared response shape).
        let queried = core.query("tag:t1", false, 10).unwrap();
        assert_eq!(
            shrike_schemas::to_wire_json(&queried).unwrap(),
            legacy.replace("\"limit\":50", "\"limit\":10")
        );

        // collection_info: a section subset — unrequested sections absent.
        let info = core
            .collection_info(&["summary".into(), "decks".into()], &[])
            .unwrap();
        let s = info.summary.as_ref().unwrap();
        let legacy = serde_json::json!({
            "summary": {
                "path": s.path.clone(),
                "created": s.created.clone(),
                "modified": s.modified.clone(),
                "notes": 1,
                "cards": 1,
                "decks": 1,
                "note_types": s.note_types,
                "tags": 1,
                "due_today": s.due_today,
            },
            "decks": [{"name": "Default", "id": 1, "note_count": 1}],
        })
        .to_string();
        let wire = shrike_schemas::to_wire_json(&info).unwrap();
        assert_eq!(wire, legacy);
        assert!(!wire.contains("stats") && !wire.contains("null"));

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
        let created: serde_json::Value =
            serde_json::from_str(&core.upsert_note_types(&create.to_string()).unwrap()).unwrap();
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
        let updated: serde_json::Value =
            serde_json::from_str(&core.upsert_note_types(&update.to_string()).unwrap()).unwrap();
        assert_eq!(updated[0]["status"], "updated");
        assert_eq!(
            core.get_note(nid).unwrap().fields,
            vec!["a-data", "b-data", "c-data", ""]
        );
        // A move is refused with the pointer to the identity tool.
        let bad = serde_json::json!([{"id": custom_id, "fields": ["B", "A2", "C", "D"]}]);
        let rejected: serde_json::Value =
            serde_json::from_str(&core.upsert_note_types(&bad.to_string()).unwrap()).unwrap();
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
        let result: serde_json::Value = serde_json::from_str(
            &core
                .update_note_type_fields("Custom", &ops.to_string())
                .unwrap(),
        )
        .unwrap();
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
        let err = core
            .update_note_type_fields("Custom", &bad_ops.to_string())
            .unwrap_err();
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
        let tresult: serde_json::Value = serde_json::from_str(
            &core
                .update_note_type_templates("Custom", &tops.to_string())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            tresult["templates"],
            serde_json::json!(["Primary", "Card 2"])
        );

        // find_and_replace_note_types: literal + regex with a Python group ref.
        let fr: serde_json::Value = serde_json::from_str(
            &core
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
                .unwrap(),
        )
        .unwrap();
        assert_eq!(fr["replacements"], 1);
        assert_eq!(fr["css_changed"], true);
        let fr2: serde_json::Value = serde_json::from_str(
            &core
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
                .unwrap(),
        )
        .unwrap();
        assert_eq!(fr2["replacements"], 1);
        let info = shrike_schemas::to_wire_value(
            &core
                .collection_info(&["note_types".to_string()], &["Custom".to_string()])
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
        let mresult: serde_json::Value = serde_json::from_str(
            &core
                .update_note_type_field_metadata("Custom", &meta.to_string())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(mresult["fields_updated"], serde_json::json!(["A2"]));

        // migrate_note_type: Custom -> Basic, dropping a field; dry_run first.
        let fmap = serde_json::json!({"A2": "Front", "C": "Back"}).to_string();
        let dry: serde_json::Value = serde_json::from_str(
            &core
                .migrate_note_type(&[nid], "Basic", &fmap, "", true)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(dry["dropped_fields"], serde_json::json!(["D"]));
        assert_eq!(dry["dry_run"], true);
        // dry run changed nothing
        assert_eq!(core.get_note(nid).unwrap().notetype_id, custom_id);
        let applied: serde_json::Value = serde_json::from_str(
            &core
                .migrate_note_type(&[nid], "Basic", &fmap, "", false)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(applied["to_note_type"], "Basic");
        let migrated = core.get_note(nid).unwrap();
        let basic = core.notetype_id("Basic").unwrap();
        assert_eq!(migrated.notetype_id, basic);
        assert_eq!(migrated.fields, vec!["a-data", "c-data"]);

        core.close().unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn media_and_prune_round_trip() {
        // Step-5a tripwires (media + maintenance RPCs) + the ported ops.
        let (core, dir) = temp_core();
        let basic = core.notetype_id("Basic").unwrap();

        // Store: bytes in, Anki-resolved name out; collision dedups/renames.
        let stored: serde_json::Value = serde_json::from_str(
            &core
                .store_media_bytes(Some("pic.png"), b"PNGDATA", None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stored["filename"], "pic.png");
        assert_eq!(stored["deduped"], false);
        let same: serde_json::Value = serde_json::from_str(
            &core
                .store_media_bytes(Some("pic.png"), b"PNGDATA", None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(same["filename"], "pic.png"); // identical content → same name
        let diff: serde_json::Value = serde_json::from_str(
            &core
                .store_media_bytes(Some("pic.png"), b"OTHERDATA", None)
                .unwrap(),
        )
        .unwrap();
        assert_ne!(diff["filename"], "pic.png"); // different content → suffixed
        assert_eq!(diff["deduped"], false); // different content: renamed, not deduped

        // fetch/list with the traversal guard + glob.
        let fetched: serde_json::Value = serde_json::from_str(
            &core
                .fetch_media(&["pic.png".into(), "../pic.png".into(), "ghost.png".into()])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(fetched[0]["status"], "found");
        assert_eq!(fetched[0]["mime"], "image/png");
        assert_eq!(fetched[1]["status"], "found"); // basename guard resolves it
        assert_eq!(fetched[2]["status"], "missing");
        let listing: serde_json::Value =
            serde_json::from_str(&core.list_media(Some("pic*"), None).unwrap()).unwrap();
        assert_eq!(listing["count"], 2);

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
        let check: serde_json::Value = serde_json::from_str(&core.media_check().unwrap()).unwrap();
        let unused: Vec<&str> = check["unused"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(unused.len(), 1);
        assert_ne!(unused[0], "pic.png");

        // Dry-run prune: previews everything, mutates nothing.
        let preview: serde_json::Value =
            serde_json::from_str(&core.prune(true, true, true, true, true).unwrap()).unwrap();
        assert_eq!(preview["dry_run"], true);
        assert_eq!(preview["empty_notes"]["removed"][0], empty_nid);
        assert_eq!(preview["unused_media"]["removed"], 1);
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 2);

        // Apply: empty note gone (its tag freed and cleared), media trashed.
        let applied: serde_json::Value =
            serde_json::from_str(&core.prune(true, true, true, true, false).unwrap()).unwrap();
        assert_eq!(applied["removed_note_ids"][0], empty_nid);
        assert!(applied["unused_tags"]["tags"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "onlytag"));
        assert_eq!(core.find_notes("deck:*").unwrap().len(), 1);
        let listing_after: serde_json::Value =
            serde_json::from_str(&core.list_media(None, None).unwrap()).unwrap();
        assert_eq!(listing_after["count"], 1);

        // delete_media: trash + echo, not_found for ghosts.
        let deleted: serde_json::Value = serde_json::from_str(
            &core
                .delete_media(&["pic.png".into(), "nope.png".into()])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(deleted["deleted"][0], "pic.png");
        assert_eq!(deleted["not_found"][0], "nope.png");

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
        let (modified, not_found) = core
            .update_note_tags(&[nid, 999], None, &["x1".into()], &["t2".into()])
            .unwrap();
        assert_eq!((modified, not_found), (1, vec![999]));
        assert_eq!(core.get_note(nid).unwrap().tags, vec!["x1".to_string()]);
        core.update_note_tags(&[nid], Some(&["fresh".into()]), &[], &[])
            .unwrap();
        assert_eq!(core.get_note(nid).unwrap().tags, vec!["fresh".to_string()]);

        // rename_tag: exact on a note set, then collection-wide.
        assert_eq!(core.rename_tag("fresh", "renamed", &[nid]).unwrap(), 1);
        assert_eq!(
            core.get_note(nid).unwrap().tags,
            vec!["renamed".to_string()]
        );
        assert_eq!(core.rename_tag("renamed", "global", &[]).unwrap(), 1);

        // Decks: upsert rename + clash, delete empty-only.
        let physics = core
            .adapter
            .deck_id_by_name("Science::Physics")
            .unwrap()
            .unwrap();
        let deck_batch = serde_json::json!([
            {"id": physics, "name": "Science::Mechanics"},
            {"name": "Empty::Leaf"},
        ]);
        let deck_results: serde_json::Value =
            serde_json::from_str(&core.upsert_decks(&deck_batch.to_string()).unwrap()).unwrap();
        assert_eq!(deck_results[0]["status"], "updated");
        assert_eq!(deck_results[1]["status"], "created");
        let clash = serde_json::json!([{"id": physics, "name": "Default"}]);
        let clash_results: serde_json::Value =
            serde_json::from_str(&core.upsert_decks(&clash.to_string()).unwrap()).unwrap();
        assert_eq!(clash_results[0]["status"], "error");

        let del: serde_json::Value = serde_json::from_str(
            &core
                .delete_decks(&["Empty::Leaf".into(), "Default".into(), "Ghost".into()])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(del["deleted"][0], "Empty::Leaf");
        assert_eq!(del["not_empty"][0], "Default"); // holds the note's card
        assert_eq!(del["not_found"][0], "Ghost");

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
        let dnt: serde_json::Value =
            serde_json::from_str(&core.delete_note_types(&[basic, 12345]).unwrap()).unwrap();
        assert_eq!(dnt["results"][0]["status"], "error");
        assert_eq!(dnt["results"][1]["status"], "not_found");
        // An unused stock type deletes cleanly.
        let cloze = core.notetype_id("Cloze").unwrap();
        let dnt2: serde_json::Value =
            serde_json::from_str(&core.delete_note_types(&[cloze]).unwrap()).unwrap();
        assert_eq!(dnt2["results"][0]["status"], "deleted");

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
}
