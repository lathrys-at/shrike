//! The write surface (#278 series, step 3): the named-fields upsert batch
//! (create/update with the #77 policy + dry_run), tags, decks,
//! find_replace_notes, delete_note_types — ports of the CollectionWrapper
//! methods of the same names, result-shape-identical JSON where the Python
//! side returns dicts (the tests/native parity harness compares them).

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use shrike_ffi::{NativeError, NativeResult};

use shrike_schemas::{
    DeckInput, DeleteDecksResponse, DeleteNoteTypeResult, NoteInput, NoteValidationReason,
    RenameTagResponse, SkipReason, UpdateNoteTagsResponse, UpsertAction, UpsertDeckResult,
    UpsertNoteResult,
};

use shrike_store_api::{ImportOptions, ImportSummary};

use crate::adapter::FieldsState;
use crate::{CollectionCore, DuplicatePolicy};

// Mirrors collection.py's _STRUCTURAL_PROBLEMS / _DUPLICATE_MESSAGE exactly
// (the parity test compares result dicts verbatim).
const DUPLICATE_MESSAGE: &str = "The first field duplicates an existing note of this type.";

fn structural_problem(state: FieldsState) -> Option<(NoteValidationReason, &'static str)> {
    match state {
        FieldsState::Empty => Some((NoteValidationReason::Empty, "The first field is empty.")),
        FieldsState::MissingCloze => Some((
            NoteValidationReason::MissingCloze,
            "No cloze deletions ({{c1::...}}) were found in the cloze field.",
        )),
        FieldsState::NotetypeNotCloze => Some((
            NoteValidationReason::NotetypeNotCloze,
            "Cloze syntax was used but the note type is not a cloze type.",
        )),
        FieldsState::FieldNotCloze => Some((
            NoteValidationReason::FieldNotCloze,
            "A cloze deletion is in a field that is not the cloze field.",
        )),
        _ => None,
    }
}

/// Python's `list[str]` repr (`['Front', 'Back']`) — the error strings are
/// compared verbatim against the wrapper's, so the rendering must match.
fn py_list_repr(items: &[String]) -> String {
    let inner = items
        .iter()
        .map(|s| format!("'{s}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn ids_csv(ids: &[i64]) -> String {
    ids.iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Per-batch lookup memo (#445): one upsert batch repeatedly resolved the
/// same notetype inventory per item — a full notetype-names list per created
/// note, field-name + type-name lookups per updated note. Scoped to one
/// `upsert_notes` call (this op never edits notetypes, so it can't go stale
/// mid-batch).
#[derive(Default)]
struct UpsertMemo {
    notetype_id_by_name: HashMap<String, Option<i64>>,
    notetype_meta_by_id: HashMap<i64, (Vec<String>, String)>,
}

impl CollectionCore {
    /// Import an `.apkg`/`.colpkg` package into the collection (#72).
    ///
    /// Delegates to anki's modern Rust importer via the service layer. MUTATES
    /// the collection — `col.mod` bumps — so the kernel op that calls this MUST
    /// follow with a drift reconcile (`reindex_if_needed`) and a derived
    /// rebuild, and must NOT advance the index watermark (the col_mod bump is
    /// the reconcile signal). Returns the per-bucket import summary.
    pub fn import_package(
        &self,
        package_path: &str,
        options: ImportOptions,
    ) -> NativeResult<ImportSummary> {
        self.adapter.import_anki_package(package_path, options)
    }

    /// The bulk upsert: each item is a typed [`NoteInput`] (`id`?,
    /// `note_type`?, `deck`?, `fields` map, `tags`?), per-item typed
    /// results out — `created`/`updated`/`ok`(dry_run)/`skipped`/`error`
    /// with the same `reason` vocabulary as `_upsert_notes`.
    pub fn upsert_notes(
        &self,
        notes: &[NoteInput],
        policy: DuplicatePolicy,
        dry_run: bool,
    ) -> NativeResult<Vec<UpsertNoteResult>> {
        let mut results: Vec<UpsertNoteResult> = Vec::new();
        let mut memo = UpsertMemo::default();
        for (index, note_input) in notes.iter().enumerate() {
            let result = if note_input.id.is_some() {
                self.update_note_named(note_input, index, dry_run, &mut memo)
            } else {
                self.create_note_named(note_input, index, policy, dry_run, &mut memo)
            };
            results.push(match result {
                Ok(r) => r,
                // Per-item try/except: one failure doesn't sink the batch.
                Err(e) => UpsertNoteResult::Error {
                    index: index as i64,
                    error: e.message,
                    reason: None,
                },
            });
        }
        Ok(results)
    }

    fn create_note_named(
        &self,
        note_input: &NoteInput,
        index: usize,
        policy: DuplicatePolicy,
        dry_run: bool,
        memo: &mut UpsertMemo,
    ) -> NativeResult<UpsertNoteResult> {
        let index = index as i64;
        let note_type_name = note_input
            .note_type
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| NativeError::invalid_input("note_type is required for new notes"))?;
        let deck_ref = note_input
            .deck
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| NativeError::invalid_input("deck is required for new notes"))?;
        let fields = note_input
            .fields
            .as_ref()
            .filter(|m| !m.is_empty())
            .ok_or_else(|| NativeError::invalid_input("fields is required for new notes"))?;

        let notetype_id = match memo.notetype_id_by_name.get(note_type_name) {
            Some(v) => *v,
            None => {
                let v = self.notetype_id_opt(note_type_name)?;
                memo.notetype_id_by_name
                    .insert(note_type_name.to_string(), v);
                v
            }
        };
        let Some(notetype_id) = notetype_id else {
            return Ok(UpsertNoteResult::Error {
                index,
                error: format!("Note type '{note_type_name}' not found"),
                reason: Some(NoteValidationReason::UnknownNoteType),
            });
        };

        // Resolve the deck reference (read-only: a plain not-yet-existing name
        // passes through and is auto-created on the write path below, so a
        // dry run still creates nothing).
        let Some(deck_name) = self.resolve_deck_ref(deck_ref)? else {
            return Err(NativeError::invalid_input(format!(
                "Deck '{deck_ref}' not found"
            )));
        };

        let mut note = self.adapter.new_note(notetype_id)?;
        let names = match memo.notetype_meta_by_id.get(&notetype_id) {
            Some((names, _)) => names.clone(),
            None => {
                let names = self.notetype_field_names(notetype_id)?;
                memo.notetype_meta_by_id
                    .insert(notetype_id, (names.clone(), note_type_name.to_string()));
                names
            }
        };
        for (field_name, value) in fields {
            let Some(pos) = names.iter().position(|n| n == field_name) else {
                return Ok(UpsertNoteResult::Error {
                    index,
                    error: format!(
                        "Field '{field_name}' not found in note type '{note_type_name}'. \
                         Available fields: {}",
                        py_list_repr(&names)
                    ),
                    reason: Some(NoteValidationReason::UnknownField),
                });
            };
            note.fields[pos] = value.clone();
        }
        if let Some(tags) = &note_input.tags {
            note.tags = tags.clone();
        }

        // Anki's own add-note validation, before any write (dry runs and real
        // runs classify identically).
        match self.adapter.fields_check(&note)? {
            FieldsState::Normal => {}
            FieldsState::Duplicate => match policy {
                DuplicatePolicy::Allow => {}
                DuplicatePolicy::Skip => {
                    return Ok(UpsertNoteResult::Skipped {
                        index,
                        reason: SkipReason::Duplicate,
                    });
                }
                DuplicatePolicy::Error => {
                    return Ok(UpsertNoteResult::Error {
                        index,
                        error: DUPLICATE_MESSAGE.to_string(),
                        reason: Some(NoteValidationReason::Duplicate),
                    });
                }
            },
            other => {
                // The fallback (an unmapped FieldsState) carries reason: None
                // — the typed wire has no "invalid" variant, and the message
                // still says what happened. (The old raw "invalid" string
                // would have failed response validation anyway.)
                let (reason, message) = match structural_problem(other) {
                    Some((reason, message)) => (Some(reason), message),
                    None => (None, "Note failed Anki's field validation."),
                };
                return Ok(UpsertNoteResult::Error {
                    index,
                    error: message.to_string(),
                    reason,
                });
            }
        }

        if dry_run {
            return Ok(UpsertNoteResult::Ok {
                index,
                action: UpsertAction::Create,
            });
        }

        let deck_id = match self.adapter.deck_id_by_name(&deck_name)? {
            Some(id) => id,
            None => self.adapter.add_deck(&deck_name)?,
        };
        let id = self.adapter.add_note(&note, deck_id)?;
        Ok(UpsertNoteResult::Created {
            id,
            neighbors: Vec::new(),
            neighbors_unavailable: false,
        })
    }

    fn update_note_named(
        &self,
        note_input: &NoteInput,
        index: usize,
        dry_run: bool,
        memo: &mut UpsertMemo,
    ) -> NativeResult<UpsertNoteResult> {
        let index = index as i64;
        let nid = note_input
            .id
            .ok_or_else(|| NativeError::invalid_input("id must be an integer"))?;
        let mut note = self
            .adapter
            .get_note(nid)
            .map_err(|_| NativeError::invalid_input(format!("Note {nid} not found")))?;

        let (names, current_type) = match memo.notetype_meta_by_id.get(&note.notetype_id) {
            Some(meta) => meta.clone(),
            None => {
                let meta = (
                    self.notetype_field_names(note.notetype_id)?,
                    self.notetype_name(note.notetype_id)?,
                );
                memo.notetype_meta_by_id
                    .insert(note.notetype_id, meta.clone());
                meta
            }
        };

        if let Some(requested) = note_input.note_type.as_deref() {
            if requested != current_type {
                return Err(NativeError::invalid_input(format!(
                    "Cannot change note type (current: '{current_type}', \
                     requested: '{requested}')"
                )));
            }
        }

        if let Some(fields) = &note_input.fields {
            for (field_name, value) in fields {
                let Some(pos) = names.iter().position(|n| n == field_name) else {
                    return Ok(UpsertNoteResult::Error {
                        index,
                        error: format!(
                            "Field '{field_name}' not found in note type '{current_type}'. \
                             Available fields: {}",
                            py_list_repr(&names)
                        ),
                        reason: Some(NoteValidationReason::UnknownField),
                    });
                };
                note.fields[pos] = value.clone();
            }
        }
        if let Some(tags) = &note_input.tags {
            note.tags = tags.clone();
        }

        // Resolve the deck reference BEFORE any write (#589): a bad ref — a
        // numeric/#id that resolves to no deck — must fail the item WITHOUT
        // having mutated the note. (The old order wrote the fields/tags first
        // and resolved the deck after, so a bad ref half-wrote the note and
        // bumped col.mod; create_note_named resolves the deck before its write,
        // and this mirrors that.) `resolve_deck_ref` is read-only — a
        // not-yet-existing plain name passes through and is auto-created on the
        // write path below, so a dry run still creates nothing.
        let deck_target = match note_input.deck.as_deref() {
            Some(deck_ref) => {
                let Some(deck_name) = self.resolve_deck_ref(deck_ref)? else {
                    return Err(NativeError::invalid_input(format!(
                        "Deck '{deck_ref}' not found"
                    )));
                };
                Some(deck_name)
            }
            None => None,
        };

        if dry_run {
            return Ok(UpsertNoteResult::Ok {
                index,
                action: UpsertAction::Update,
            });
        }

        self.adapter.update_note(&note)?;

        if let Some(deck_name) = deck_target {
            let deck_id = match self.adapter.deck_id_by_name(&deck_name)? {
                Some(id) => id,
                None => self.adapter.add_deck(&deck_name)?,
            };
            let card_ids = self.adapter.cards_of_note(nid)?;
            self.adapter.set_card_deck(&card_ids, deck_id)?;
        }

        Ok(UpsertNoteResult::Updated {
            id: nid,
            neighbors: Vec::new(),
            neighbors_unavailable: false,
        })
    }

    /// Edit tags on a note set — `set_tags` is a full replace (mutually
    /// exclusive with add/remove, validated by the caller); add/remove apply
    /// subtractively-then-additively.
    pub fn update_note_tags(
        &self,
        note_ids: &[i64],
        set_tags: Option<&[String]>,
        add: &[String],
        remove: &[String],
    ) -> NativeResult<UpdateNoteTagsResponse> {
        let existing: HashSet<i64> = self
            .adapter
            .search_notes(&format!("nid:{}", ids_csv(note_ids)))?
            .into_iter()
            .collect();
        let not_found: Vec<i64> = note_ids
            .iter()
            .filter(|i| !existing.contains(i))
            .copied()
            .collect();
        let targets: Vec<i64> = note_ids
            .iter()
            .filter(|i| existing.contains(i))
            .copied()
            .collect();

        if !targets.is_empty() {
            if let Some(set_tags) = set_tags {
                // One UpdateNotes call for the whole set (#445): the
                // per-note get+update loop paid 3 RPCs and a journal
                // commit per note at the 1000-note cap.
                self.adapter.set_note_tags_bulk(&targets, set_tags)?;
            } else {
                // Remove before add so a tag named in both ends up present.
                if !remove.is_empty() {
                    self.adapter.remove_note_tags(&targets, &remove.join(" "))?;
                }
                if !add.is_empty() {
                    self.adapter.add_note_tags(&targets, &add.join(" "))?;
                }
            }
        }
        Ok(UpdateNoteTagsResponse {
            notes_modified: targets.len() as i64,
            not_found,
            message: None,
        })
    }

    /// Rename a tag collection-wide (empty `note_ids`) or exactly on a note
    /// set (never substring: renaming `jp` never touches `jp-verbs`).
    pub fn rename_tag(
        &self,
        old: &str,
        new: &str,
        note_ids: &[i64],
    ) -> NativeResult<RenameTagResponse> {
        let notes_modified = if note_ids.is_empty() {
            self.adapter.rename_tags(old, new)?
        } else {
            let matching = self
                .adapter
                .search_notes(&format!("(nid:{}) tag:{old}", ids_csv(note_ids)))?;
            if !matching.is_empty() {
                self.adapter.remove_note_tags(&matching, old)?;
                self.adapter.add_note_tags(&matching, new)?;
            }
            matching.len()
        };
        Ok(RenameTagResponse {
            notes_modified: notes_modified as i64,
        })
    }

    /// Create or rename decks in bulk (id present = rename; never merges).
    /// Typed both directions (#391): per-item errors never sink the batch.
    pub fn upsert_decks(&self, decks: &[DeckInput]) -> NativeResult<Vec<UpsertDeckResult>> {
        let mut results: Vec<UpsertDeckResult> = Vec::new();
        for (index, deck) in decks.iter().enumerate() {
            results.push(match self.upsert_one_deck(deck) {
                Ok(r) => r,
                Err(e) => UpsertDeckResult::Error {
                    index: index as i64,
                    name: None,
                    error: e.message,
                },
            });
        }
        Ok(results)
    }

    fn upsert_one_deck(&self, deck: &DeckInput) -> NativeResult<UpsertDeckResult> {
        let name = Some(deck.name.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| NativeError::invalid_input("name is required"))?;
        if let Some(deck_id) = deck.id {
            let known: HashMap<i64, String> = self.adapter.deck_names()?.into_iter().collect();
            if !known.contains_key(&deck_id) {
                return Err(NativeError::invalid_input(format!(
                    "Deck {deck_id} not found"
                )));
            }
            if let Some(clash) = self.adapter.deck_id_by_name(name)? {
                if clash != deck_id {
                    return Err(NativeError::invalid_input(format!(
                        "A deck named '{name}' already exists"
                    )));
                }
            }
            self.adapter.rename_deck(deck_id, name)?;
            return Ok(UpsertDeckResult::Updated {
                id: deck_id,
                name: name.to_string(),
            });
        }
        if let Some(existing) = self.adapter.deck_id_by_name(name)? {
            return Ok(UpsertDeckResult::Updated {
                id: existing,
                name: name.to_string(),
            });
        }
        let new_id = self.adapter.add_deck(name)?;
        Ok(UpsertDeckResult::Created {
            id: new_id,
            name: name.to_string(),
        })
    }

    /// Delete decks by reference — only if empty (no cards in the deck or its
    /// subdecks). Result lists echo the caller's references.
    pub fn delete_decks(&self, refs: &[String]) -> NativeResult<DeleteDecksResponse> {
        let all: Vec<(i64, String)> = self.adapter.deck_names()?;
        let mut deleted: Vec<String> = Vec::new();
        let mut not_found: Vec<String> = Vec::new();
        let mut not_empty: Vec<String> = Vec::new();
        let mut to_remove: Vec<i64> = Vec::new();
        for reference in refs {
            let resolved = self.resolve_deck_ref(reference)?;
            let deck_id = match resolved {
                Some(name) => self.adapter.deck_id_by_name(&name)?,
                None => None,
            };
            let Some(deck_id) = deck_id else {
                not_found.push(reference.clone());
                continue;
            };
            // Card count incl. subdecks (and filtered-deck originals), via the
            // deck-name prefix — pylib's card_count(include_subdecks=True).
            let name = all
                .iter()
                .find(|(id, _)| *id == deck_id)
                .map(|(_, n)| n.clone())
                .unwrap_or_default();
            let prefix = format!("{name}::");
            let family: Vec<i64> = all
                .iter()
                .filter(|(_, n)| *n == name || n.starts_with(&prefix))
                .map(|(id, _)| *id)
                .collect();
            let family_csv = ids_csv(&family);
            let count = self
                .adapter
                .db_rows(&format!(
                    "select count() from cards where did in ({family_csv}) \
                     or odid in ({family_csv})"
                ))?
                .first()
                .and_then(|r| r.first())
                .and_then(Value::as_i64)
                .unwrap_or(0);
            if count > 0 {
                not_empty.push(reference.clone());
            } else {
                to_remove.push(deck_id);
                deleted.push(reference.clone());
            }
        }
        if !to_remove.is_empty() {
            self.adapter.remove_decks(&to_remove)?;
        }
        Ok(DeleteDecksResponse {
            deleted,
            not_found,
            not_empty,
        })
    }

    /// Apply a find/replace over a note set's fields via Anki's own
    /// find_and_replace, detecting the actually-changed notes by diffing the
    /// raw `flds` column before/after (note.mod is second-resolution). The
    /// dry-run *preview* (Python-side `apply_replacement` samples) stays in
    /// the host; this is the apply path. Returns
    /// `{"notes_changed": N, "changed_ids": [...]}` as JSON.
    #[allow(clippy::too_many_arguments)]
    pub fn find_replace_notes(
        &self,
        note_ids: &[i64],
        search: &str,
        replacement: &str,
        regex: bool,
        match_case: bool,
        field_name: Option<&str>,
    ) -> NativeResult<(usize, Vec<i64>)> {
        let flds_sql = format!(
            "select id, flds from notes where id in ({})",
            ids_csv(note_ids)
        );
        let snapshot = |rows: Vec<Vec<Value>>| -> HashMap<i64, String> {
            rows.into_iter()
                .filter_map(|r| Some((r.first()?.as_i64()?, r.get(1)?.as_str()?.to_string())))
                .collect()
        };
        let before = snapshot(self.adapter.db_rows(&flds_sql)?);
        let count = self.adapter.find_and_replace(
            note_ids,
            search,
            replacement,
            regex,
            match_case,
            field_name,
        )?;
        let after = snapshot(self.adapter.db_rows(&flds_sql)?);
        let changed_ids: Vec<i64> = note_ids
            .iter()
            .filter(|nid| after.get(nid) != before.get(nid))
            .copied()
            .collect();
        Ok((count, changed_ids))
    }

    /// Delete note types by id — only if unused. Typed per-item results
    /// (`deleted`/`not_found`/`error`), same vocabulary as `_delete_note_types`.
    pub fn delete_note_types(&self, ids: &[i64]) -> NativeResult<Vec<DeleteNoteTypeResult>> {
        let known: HashMap<i64, String> = self.adapter.notetype_names()?.into_iter().collect();
        let mut results: Vec<DeleteNoteTypeResult> = Vec::new();
        for nt_id in ids {
            let Some(name) = known.get(nt_id) else {
                results.push(DeleteNoteTypeResult::NotFound { id: *nt_id });
                continue;
            };
            let use_count = self
                .adapter
                .db_rows(&format!("select count() from notes where mid = {nt_id}"))?
                .first()
                .and_then(|r| r.first())
                .and_then(Value::as_i64)
                .unwrap_or(0);
            if use_count > 0 {
                results.push(DeleteNoteTypeResult::Error {
                    id: *nt_id,
                    name: name.clone(),
                    error: format!("Cannot delete: {use_count} note(s) use this type"),
                });
                continue;
            }
            self.adapter.remove_notetype(*nt_id)?;
            results.push(DeleteNoteTypeResult::Deleted {
                id: *nt_id,
                name: name.clone(),
            });
        }
        Ok(results)
    }
}
