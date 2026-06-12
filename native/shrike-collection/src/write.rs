//! The write surface (#278 series, step 3): the named-fields upsert batch
//! (create/update with the #77 policy + dry_run), tags, decks,
//! find_replace_notes, delete_note_types — ports of the CollectionWrapper
//! methods of the same names, result-shape-identical JSON where the Python
//! side returns dicts (the tests/native parity harness compares them).

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use shrike_ffi::{NativeError, NativeResult};

use crate::adapter::FieldsState;
use crate::CollectionCore;

// Mirrors collection.py's _STRUCTURAL_PROBLEMS / _DUPLICATE_MESSAGE exactly
// (the parity test compares result dicts verbatim).
const DUPLICATE_MESSAGE: &str = "The first field duplicates an existing note of this type.";

fn structural_problem(state: FieldsState) -> Option<(&'static str, &'static str)> {
    match state {
        FieldsState::Empty => Some(("empty", "The first field is empty.")),
        FieldsState::MissingCloze => Some((
            "missing_cloze",
            "No cloze deletions ({{c1::...}}) were found in the cloze field.",
        )),
        FieldsState::NotetypeNotCloze => Some((
            "notetype_not_cloze",
            "Cloze syntax was used but the note type is not a cloze type.",
        )),
        FieldsState::FieldNotCloze => Some((
            "field_not_cloze",
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
    /// The bulk upsert: each item is the wrapper's note-input dict (`id`?,
    /// `note_type`?, `deck`?, `fields` map, `tags`?), JSON in / per-item
    /// results JSON out — `created`/`updated`/`ok`(dry_run)/`skipped`/`error`
    /// with the same `reason` vocabulary as `_upsert_notes`.
    pub fn upsert_notes(
        &self,
        notes_json: &str,
        on_duplicate: &str,
        dry_run: bool,
    ) -> NativeResult<String> {
        let notes: Vec<Value> = serde_json::from_str(notes_json)
            .map_err(|e| NativeError::invalid_input(format!("notes must be a JSON list: {e}")))?;
        if !matches!(on_duplicate, "error" | "skip" | "allow") {
            return Err(NativeError::invalid_input(format!(
                "on_duplicate must be error/skip/allow (got {on_duplicate:?})"
            )));
        }
        let mut results: Vec<Value> = Vec::new();
        let mut memo = UpsertMemo::default();
        for (index, note_input) in notes.iter().enumerate() {
            let result = if note_input.get("id").is_some_and(|v| !v.is_null()) {
                self.update_note_named(note_input, index, dry_run, &mut memo)
            } else {
                self.create_note_named(note_input, index, on_duplicate, dry_run, &mut memo)
            };
            results.push(match result {
                Ok(r) => r,
                // Per-item try/except: one failure doesn't sink the batch.
                Err(e) => json!({"status": "error", "index": index, "error": e.message}),
            });
        }
        Ok(json!(results).to_string())
    }

    fn create_note_named(
        &self,
        note_input: &Value,
        index: usize,
        on_duplicate: &str,
        dry_run: bool,
        memo: &mut UpsertMemo,
    ) -> NativeResult<Value> {
        let note_type_name = note_input
            .get("note_type")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| NativeError::invalid_input("note_type is required for new notes"))?;
        let deck_ref = note_input
            .get("deck")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| NativeError::invalid_input("deck is required for new notes"))?;
        let fields = note_input
            .get("fields")
            .and_then(Value::as_object)
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
            return Ok(json!({
                "status": "error",
                "index": index,
                "error": format!("Note type '{note_type_name}' not found"),
                "reason": "unknown_note_type",
            }));
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
                return Ok(json!({
                    "status": "error",
                    "index": index,
                    "error": format!(
                        "Field '{field_name}' not found in note type '{note_type_name}'. \
                         Available fields: {}",
                        py_list_repr(&names)
                    ),
                    "reason": "unknown_field",
                }));
            };
            note.fields[pos] = value.as_str().unwrap_or_default().to_string();
        }
        if let Some(tags) = note_input.get("tags").and_then(Value::as_array) {
            note.tags = tags
                .iter()
                .map(|t| t.as_str().unwrap_or_default().to_string())
                .collect();
        }

        // Anki's own add-note validation, before any write (dry runs and real
        // runs classify identically).
        match self.adapter.fields_check(&note)? {
            FieldsState::Normal => {}
            FieldsState::Duplicate => match on_duplicate {
                "allow" => {}
                "skip" => {
                    return Ok(json!({"status": "skipped", "index": index, "reason": "duplicate"}));
                }
                _ => {
                    return Ok(json!({
                        "status": "error",
                        "index": index,
                        "error": DUPLICATE_MESSAGE,
                        "reason": "duplicate",
                    }));
                }
            },
            other => {
                let (reason, message) = structural_problem(other)
                    .unwrap_or(("invalid", "Note failed Anki's field validation."));
                return Ok(json!({
                    "status": "error",
                    "index": index,
                    "error": message,
                    "reason": reason,
                }));
            }
        }

        if dry_run {
            return Ok(json!({"status": "ok", "index": index, "action": "create"}));
        }

        let deck_id = match self.adapter.deck_id_by_name(&deck_name)? {
            Some(id) => id,
            None => self.adapter.add_deck(&deck_name)?,
        };
        let id = self.adapter.add_note(&note, deck_id)?;
        Ok(json!({"status": "created", "id": id}))
    }

    fn update_note_named(
        &self,
        note_input: &Value,
        index: usize,
        dry_run: bool,
        memo: &mut UpsertMemo,
    ) -> NativeResult<Value> {
        let nid = note_input
            .get("id")
            .and_then(Value::as_i64)
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

        if let Some(requested) = note_input.get("note_type").and_then(Value::as_str) {
            if requested != current_type {
                return Err(NativeError::invalid_input(format!(
                    "Cannot change note type (current: '{current_type}', \
                     requested: '{requested}')"
                )));
            }
        }

        if let Some(fields) = note_input.get("fields").and_then(Value::as_object) {
            for (field_name, value) in fields {
                let Some(pos) = names.iter().position(|n| n == field_name) else {
                    return Ok(json!({
                        "status": "error",
                        "index": index,
                        "error": format!(
                            "Field '{field_name}' not found in note type '{current_type}'. \
                             Available fields: {}",
                            py_list_repr(&names)
                        ),
                        "reason": "unknown_field",
                    }));
                };
                note.fields[pos] = value.as_str().unwrap_or_default().to_string();
            }
        }
        if let Some(tags) = note_input.get("tags").and_then(Value::as_array) {
            note.tags = tags
                .iter()
                .map(|t| t.as_str().unwrap_or_default().to_string())
                .collect();
        }

        if dry_run {
            return Ok(json!({"status": "ok", "index": index, "action": "update"}));
        }

        self.adapter.update_note(&note)?;

        if let Some(deck_ref) = note_input.get("deck").and_then(Value::as_str) {
            let Some(deck_name) = self.resolve_deck_ref(deck_ref)? else {
                return Err(NativeError::invalid_input(format!(
                    "Deck '{deck_ref}' not found"
                )));
            };
            let deck_id = match self.adapter.deck_id_by_name(&deck_name)? {
                Some(id) => id,
                None => self.adapter.add_deck(&deck_name)?,
            };
            let card_ids = self.adapter.cards_of_note(nid)?;
            self.adapter.set_card_deck(&card_ids, deck_id)?;
        }

        Ok(json!({"status": "updated", "id": nid}))
    }

    /// Edit tags on a note set — `set_tags` is a full replace (mutually
    /// exclusive with add/remove, validated by the caller); add/remove apply
    /// subtractively-then-additively. Returns `(notes_modified, not_found)`.
    pub fn update_note_tags(
        &self,
        note_ids: &[i64],
        set_tags: Option<&[String]>,
        add: &[String],
        remove: &[String],
    ) -> NativeResult<(usize, Vec<i64>)> {
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
        Ok((targets.len(), not_found))
    }

    /// Rename a tag collection-wide (empty `note_ids`) or exactly on a note
    /// set (never substring: renaming `jp` never touches `jp-verbs`).
    pub fn rename_tag(&self, old: &str, new: &str, note_ids: &[i64]) -> NativeResult<usize> {
        if note_ids.is_empty() {
            return self.adapter.rename_tags(old, new);
        }
        let matching = self
            .adapter
            .search_notes(&format!("(nid:{}) tag:{old}", ids_csv(note_ids)))?;
        if !matching.is_empty() {
            self.adapter.remove_note_tags(&matching, old)?;
            self.adapter.add_note_tags(&matching, new)?;
        }
        Ok(matching.len())
    }

    /// Create or rename decks in bulk (id present = rename; never merges).
    /// JSON in (list of `{name, id?}`) / per-item results JSON out.
    pub fn upsert_decks(&self, decks_json: &str) -> NativeResult<String> {
        let decks: Vec<Value> = serde_json::from_str(decks_json)
            .map_err(|e| NativeError::invalid_input(format!("decks must be a JSON list: {e}")))?;
        let mut results: Vec<Value> = Vec::new();
        for (index, deck) in decks.iter().enumerate() {
            results.push(match self.upsert_one_deck(deck) {
                Ok(r) => r,
                Err(e) => json!({"status": "error", "index": index, "error": e.message}),
            });
        }
        Ok(json!(results).to_string())
    }

    fn upsert_one_deck(&self, deck: &Value) -> NativeResult<Value> {
        let name = deck
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| NativeError::invalid_input("name is required"))?;
        if let Some(deck_id) = deck.get("id").and_then(Value::as_i64) {
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
            return Ok(json!({"status": "updated", "id": deck_id, "name": name}));
        }
        if let Some(existing) = self.adapter.deck_id_by_name(name)? {
            return Ok(json!({"status": "updated", "id": existing, "name": name}));
        }
        let new_id = self.adapter.add_deck(name)?;
        Ok(json!({"status": "created", "id": new_id, "name": name}))
    }

    /// Delete decks by reference — only if empty (no cards in the deck or its
    /// subdecks). Result lists echo the caller's references.
    pub fn delete_decks(&self, refs: &[String]) -> NativeResult<String> {
        let all: Vec<(i64, String)> = self.adapter.deck_names()?;
        let mut deleted: Vec<&str> = Vec::new();
        let mut not_found: Vec<&str> = Vec::new();
        let mut not_empty: Vec<&str> = Vec::new();
        let mut to_remove: Vec<i64> = Vec::new();
        for reference in refs {
            let resolved = self.resolve_deck_ref(reference)?;
            let deck_id = match resolved {
                Some(name) => self.adapter.deck_id_by_name(&name)?,
                None => None,
            };
            let Some(deck_id) = deck_id else {
                not_found.push(reference);
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
                not_empty.push(reference);
            } else {
                to_remove.push(deck_id);
                deleted.push(reference);
            }
        }
        if !to_remove.is_empty() {
            self.adapter.remove_decks(&to_remove)?;
        }
        Ok(json!({"deleted": deleted, "not_found": not_found, "not_empty": not_empty}).to_string())
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
    ) -> NativeResult<String> {
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
        Ok(json!({"notes_changed": count, "changed_ids": changed_ids}).to_string())
    }

    /// Delete note types by id — only if unused. Per-item results JSON,
    /// shape-identical to `_delete_note_types`.
    pub fn delete_note_types(&self, ids: &[i64]) -> NativeResult<String> {
        let known: HashMap<i64, String> = self.adapter.notetype_names()?.into_iter().collect();
        let mut results: Vec<Value> = Vec::new();
        for nt_id in ids {
            let Some(name) = known.get(nt_id) else {
                results.push(json!({"id": nt_id, "status": "not_found"}));
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
                results.push(json!({
                    "id": nt_id,
                    "name": name,
                    "status": "error",
                    "error": format!("Cannot delete: {use_count} note(s) use this type"),
                }));
                continue;
            }
            self.adapter.remove_notetype(*nt_id)?;
            results.push(json!({"id": nt_id, "name": name, "status": "deleted"}));
        }
        Ok(json!({"results": results}).to_string())
    }
}
