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

pub use adapter::{FieldsState, ServiceAdapter, ServiceNote};
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
}

impl CollectionCore {
    /// Open (creating if needed) a collection. `media_folder`/`media_db` are
    /// derived from the collection path exactly like anki's Python does.
    pub fn open(collection_path: &str) -> NativeResult<Self> {
        let adapter = ServiceAdapter::new()?;
        let base = collection_path
            .strip_suffix(".anki2")
            .unwrap_or(collection_path);
        adapter.open_collection(
            collection_path,
            &format!("{base}.media"),
            &format!("{base}.media.db2"),
        )?;
        Ok(Self { adapter })
    }

    pub fn close(&self) -> NativeResult<()> {
        self.adapter.close_collection()
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
        self.adapter
            .notetype_names()?
            .into_iter()
            .find(|(_, n)| n == name)
            .map(|(id, _)| id)
            .ok_or_else(|| NativeError::invalid_input(format!("unknown note type: {name}")))
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
}

#[cfg(test)]
mod tests {
    //! The slice-1 parity floor AND the index tripwires: every hardcoded
    //! (service, method) pair is exercised against a real temp collection, so
    //! a tag bump that shuffles the generated dispatcher fails these tests
    //! instead of corrupting calls.

    use super::*;

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
