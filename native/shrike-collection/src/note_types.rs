//! Note-type operations (#278 series, step 4) — the port of
//! `shrike/note_types.py` plus `_migrate_note_type` from `collection.py`.
//!
//! Everything operates on the **schema11 JSON dicts** through the same legacy
//! RPCs pylib's ModelManager uses (`update_dict` → `update_notetype_legacy`,
//! `new_field`/`new_template` → stock-Basic clones), so the ord-based
//! data/card migration semantics are identical by construction. The
//! positional-vs-identity reconciliation (#76), the simulate-then-apply
//! atomicity, and the result shapes are ported verbatim (the tests/native
//! parity harness compares result dicts against the Python implementation).

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use shrike_ffi::{NativeError, NativeResult};

use crate::CollectionCore;

const MODEL_CLOZE: i64 = 1;

fn invalid(msg: impl Into<String>) -> NativeError {
    NativeError::invalid_input(msg)
}

/// Python's `list[str]` repr — error strings are compared verbatim.
fn py_str_list_repr(items: &[&str]) -> String {
    let inner = items
        .iter()
        .map(|s| format!("'{s}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn names_of(entries: &Value) -> Vec<String> {
    entries
        .as_array()
        .map(|a| {
            a.iter()
                .map(|e| e["name"].as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default()
}

impl CollectionCore {
    fn notetype_legacy_by_name(&self, name: &str) -> NativeResult<Option<Value>> {
        match self.notetype_id_opt(name)? {
            Some(id) => Ok(Some(self.adapter.notetype_legacy(id)?)),
            None => Ok(None),
        }
    }

    /// pylib `models.new_field`: the stock Basic's first field, renamed,
    /// ord cleared.
    fn new_field(&self, name: &str) -> NativeResult<Value> {
        let stock = self.adapter.stock_notetype_legacy()?;
        let mut field = stock["flds"][0].clone();
        field["name"] = json!(name);
        field["ord"] = Value::Null;
        Ok(field)
    }

    /// pylib `models.new_template`: stock Basic's first template, renamed,
    /// formats cleared, ord cleared.
    fn new_template(&self, name: &str) -> NativeResult<Value> {
        let stock = self.adapter.stock_notetype_legacy()?;
        let mut tmpl = stock["tmpls"][0].clone();
        tmpl["name"] = json!(name);
        tmpl["qfmt"] = json!("");
        tmpl["afmt"] = json!("");
        tmpl["ord"] = Value::Null;
        Ok(tmpl)
    }

    /// `upsert_note_types`: create or update note-type definitions in bulk
    /// (JSON in/out, per-item try/except, the position-keyed replace with the
    /// #76 unsound-move rejection).
    pub fn upsert_note_types(&self, note_types_json: &str) -> NativeResult<String> {
        let inputs: Vec<Value> = serde_json::from_str(note_types_json)
            .map_err(|e| invalid(format!("note_types must be a JSON list: {e}")))?;
        let mut results = Vec::new();
        for (i, nt_input) in inputs.iter().enumerate() {
            let result = if nt_input.get("id").is_some_and(|v| !v.is_null()) {
                self.update_note_type(nt_input)
            } else {
                self.create_note_type(nt_input)
            };
            results.push(match result {
                Ok(r) => r,
                Err(e) => json!({"status": "error", "index": i, "error": e.message}),
            });
        }
        Ok(json!(results).to_string())
    }

    fn create_note_type(&self, nt_input: &Value) -> NativeResult<Value> {
        let name = nt_input
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("name is required for new note types"))?;
        let fields = nt_input
            .get("fields")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("fields is required for new note types"))?;
        let templates = nt_input
            .get("templates")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("templates is required for new note types"))?;
        let css = nt_input
            .get("css")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("css is required for new note types"))?;

        if self.notetype_id_opt(name)?.is_some() {
            return Err(invalid(format!("Note type '{name}' already exists")));
        }

        // pylib models.new(name): a stock-Basic clone, renamed, id cleared,
        // field/template lists rebuilt from the inputs.
        let mut notetype = self.adapter.stock_notetype_legacy()?;
        notetype["name"] = json!(name);
        notetype["id"] = json!(0);
        if nt_input.get("is_cloze").and_then(Value::as_bool) == Some(true) {
            notetype["type"] = json!(MODEL_CLOZE);
        }
        notetype["css"] = json!(css);

        let mut flds = Vec::new();
        for field_name in fields {
            let field_name = field_name
                .as_str()
                .ok_or_else(|| invalid("fields must be a list of names"))?;
            flds.push(self.new_field(field_name)?);
        }
        notetype["flds"] = json!(flds);

        let mut tmpls = Vec::new();
        for tmpl_input in templates {
            let mut tmpl = self.new_template(
                tmpl_input
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("template name is required"))?,
            )?;
            tmpl["qfmt"] = tmpl_input.get("front").cloned().unwrap_or(json!(""));
            tmpl["afmt"] = tmpl_input.get("back").cloned().unwrap_or(json!(""));
            tmpls.push(tmpl);
        }
        notetype["tmpls"] = json!(tmpls);

        let id = self.adapter.add_notetype_legacy(&notetype)?;
        Ok(json!({"status": "created", "id": id, "name": name}))
    }

    fn update_note_type(&self, nt_input: &Value) -> NativeResult<Value> {
        let nt_id = nt_input
            .get("id")
            .and_then(Value::as_i64)
            .ok_or_else(|| invalid("id must be an integer"))?;
        let mut notetype = self
            .adapter
            .notetype_legacy(nt_id)
            .map_err(|_| invalid(format!("Note type with ID {nt_id} not found")))?;

        if let Some(is_cloze) = nt_input.get("is_cloze").and_then(Value::as_bool) {
            let current = notetype["type"].as_i64() == Some(MODEL_CLOZE);
            if is_cloze != current {
                return Err(invalid(
                    "Cannot change a note type between standard and cloze",
                ));
            }
        }
        if let Some(name) = nt_input.get("name").and_then(Value::as_str) {
            notetype["name"] = json!(name);
        }
        if let Some(css) = nt_input.get("css").and_then(Value::as_str) {
            notetype["css"] = json!(css);
        }
        if let Some(fields) = nt_input.get("fields").and_then(Value::as_array) {
            let names: Vec<String> = fields
                .iter()
                .map(|f| f.as_str().unwrap_or_default().to_string())
                .collect();
            self.set_fields_positional(&mut notetype, &names)?;
        }
        if let Some(templates) = nt_input.get("templates").and_then(Value::as_array) {
            self.set_templates_positional(&mut notetype, templates)?;
        }
        self.adapter.update_notetype_legacy(&notetype)?;
        Ok(json!({
            "status": "updated",
            "id": nt_id,
            "name": notetype["name"],
        }))
    }

    /// `_set_fields`: position-keyed whole-list replace, data-safe — reuse the
    /// existing field dicts in place (ords intact), append for added
    /// positions, drop the tail; refuse anything that moves an existing name.
    fn set_fields_positional(&self, notetype: &mut Value, names: &[String]) -> NativeResult<()> {
        let old_names = names_of(&notetype["flds"]);
        reject_unsound_positional_replace(
            &old_names,
            names,
            "field",
            "note data",
            "update_note_type_fields",
        )?;
        let old_flds = notetype["flds"].as_array().cloned().unwrap_or_default();
        let mut new = Vec::new();
        for (i, name) in names.iter().enumerate() {
            if i < old_flds.len() {
                let mut field = old_flds[i].clone();
                field["name"] = json!(name);
                new.push(field);
            } else {
                new.push(self.new_field(name)?);
            }
        }
        notetype["flds"] = json!(new);
        Ok(())
    }

    /// `_set_templates`: the template counterpart (cards keep their template
    /// by position; tail drops delete those templates' cards intentionally).
    fn set_templates_positional(
        &self,
        notetype: &mut Value,
        templates: &[Value],
    ) -> NativeResult<()> {
        let old_names = names_of(&notetype["tmpls"]);
        let new_names: Vec<String> = templates
            .iter()
            .map(|t| t["name"].as_str().unwrap_or_default().to_string())
            .collect();
        reject_unsound_positional_replace(
            &old_names,
            &new_names,
            "template",
            "cards (and their scheduling history)",
            "update_note_type_templates",
        )?;
        let old_tmpls = notetype["tmpls"].as_array().cloned().unwrap_or_default();
        let mut new = Vec::new();
        for (i, tmpl_input) in templates.iter().enumerate() {
            let mut tmpl = if i < old_tmpls.len() {
                old_tmpls[i].clone()
            } else {
                self.new_template(tmpl_input["name"].as_str().unwrap_or_default())?
            };
            tmpl["name"] = tmpl_input["name"].clone();
            tmpl["qfmt"] = tmpl_input.get("front").cloned().unwrap_or(json!(""));
            tmpl["afmt"] = tmpl_input.get("back").cloned().unwrap_or(json!(""));
            new.push(tmpl);
        }
        notetype["tmpls"] = json!(new);
        Ok(())
    }

    /// `update_note_type_fields`: identity-based ops (add/remove/rename/
    /// reposition by name), atomic via simulate-then-apply, one persist.
    pub fn update_note_type_fields(
        &self,
        note_type_name: &str,
        operations_json: &str,
    ) -> NativeResult<String> {
        let operations: Vec<Value> = serde_json::from_str(operations_json)
            .map_err(|e| invalid(format!("operations must be a JSON list: {e}")))?;
        let mut notetype = self
            .notetype_legacy_by_name(note_type_name)?
            .ok_or_else(|| invalid(format!("Note type '{note_type_name}' not found")))?;

        let mut sim = names_of(&notetype["flds"]);
        for (i, op) in operations.iter().enumerate() {
            simulate_struct_op(&mut sim, op, i, "field")?;
        }
        for op in &operations {
            self.apply_entry_op(&mut notetype, op, "flds")?;
        }
        self.adapter.update_notetype_legacy(&notetype)?;
        let final_names = names_of(&notetype["flds"]);
        Ok(json!({
            "id": notetype["id"],
            "name": note_type_name,
            "fields": final_names,
        })
        .to_string())
    }

    /// `update_note_type_templates`: the by-identity template counterpart.
    pub fn update_note_type_templates(
        &self,
        note_type_name: &str,
        operations_json: &str,
    ) -> NativeResult<String> {
        let operations: Vec<Value> = serde_json::from_str(operations_json)
            .map_err(|e| invalid(format!("operations must be a JSON list: {e}")))?;
        let mut notetype = self
            .notetype_legacy_by_name(note_type_name)?
            .ok_or_else(|| invalid(format!("Note type '{note_type_name}' not found")))?;

        let mut sim = names_of(&notetype["tmpls"]);
        for (i, op) in operations.iter().enumerate() {
            simulate_struct_op(&mut sim, op, i, "template")?;
        }
        for op in &operations {
            self.apply_entry_op(&mut notetype, op, "tmpls")?;
        }
        self.adapter.update_notetype_legacy(&notetype)?;
        let final_names = names_of(&notetype["tmpls"]);
        Ok(json!({
            "id": notetype["id"],
            "name": note_type_name,
            "templates": final_names,
        })
        .to_string())
    }

    /// Apply one validated op to the schema11 entry list (`flds`/`tmpls`).
    /// Mirrors pylib's primitives: every manipulation is pure list surgery
    /// with the existing entries' `ord` markers untouched — `update_dict`
    /// derives the data/card migration from them.
    fn apply_entry_op(&self, notetype: &mut Value, op: &Value, key: &str) -> NativeResult<()> {
        let entries = notetype[key]
            .as_array_mut()
            .ok_or_else(|| NativeError::internal("notetype entries not a list"))?;
        let kind = op["op"].as_str().unwrap_or_default();
        let name = op["name"].as_str().unwrap_or_default();
        let position = |sim_len: usize| -> usize {
            op.get("position")
                .and_then(Value::as_u64)
                .map_or(sim_len, |p| p as usize)
        };
        match kind {
            "add" => {
                let mut entry = if key == "flds" {
                    self.new_field(name)?
                } else {
                    let mut tmpl = self.new_template(name)?;
                    tmpl["qfmt"] = op.get("front").cloned().unwrap_or(json!(""));
                    tmpl["afmt"] = op.get("back").cloned().unwrap_or(json!(""));
                    tmpl
                };
                entry["name"] = json!(name);
                let entries_len = entries.len();
                let pos = position(entries_len).min(entries_len);
                entries.insert(pos, entry);
            }
            "remove" => {
                if let Some(idx) = entries
                    .iter()
                    .position(|e| e["name"].as_str() == Some(name))
                {
                    entries.remove(idx);
                }
            }
            "rename" => {
                if let Some(entry) = entries
                    .iter_mut()
                    .find(|e| e["name"].as_str() == Some(name))
                {
                    entry["name"] = op["new_name"].clone();
                }
            }
            "reposition" => {
                if let Some(idx) = entries
                    .iter()
                    .position(|e| e["name"].as_str() == Some(name))
                {
                    let entry = entries.remove(idx);
                    let entries_len = entries.len();
                    let pos = position(entries_len).min(entries_len);
                    entries.insert(pos, entry);
                }
            }
            _ => return Err(invalid(format!("unknown op: {kind:?}"))),
        }
        Ok(())
    }

    /// `find_and_replace_note_types`: literal-or-regex rewrite over one
    /// model's template HTML + shared CSS. Literal mode inserts the
    /// replacement verbatim; regex mode accepts Python-style group refs
    /// (`\1`, `\g<n>`), translated to the regex crate's `${n}`.
    #[allow(clippy::too_many_arguments)]
    pub fn find_and_replace_note_types(
        &self,
        note_type_name: &str,
        search: &str,
        replacement: &str,
        regex: bool,
        match_case: bool,
        front: bool,
        back: bool,
        css: bool,
    ) -> NativeResult<String> {
        let mut notetype = self
            .notetype_legacy_by_name(note_type_name)?
            .ok_or_else(|| invalid(format!("Note type '{note_type_name}' not found")))?;

        let pattern = if regex {
            let flagged = if match_case {
                search.to_string()
            } else {
                format!("(?i){search}")
            };
            fancy_regex::Regex::new(&flagged).map_err(|e| invalid(format!("invalid regex: {e}")))?
        } else {
            let escaped = fancy_regex::escape(search);
            let flagged = if match_case {
                escaped.into_owned()
            } else {
                format!("(?i){escaped}")
            };
            fancy_regex::Regex::new(&flagged).map_err(|e| invalid(format!("invalid regex: {e}")))?
        };
        let template = if regex {
            python_replacement_template(replacement)
        } else {
            // Literal: no group-ref interpretation — escape `$`.
            replacement.replace('$', "$$")
        };

        let sub = |value: &str| -> (String, usize) {
            // Count via find_iter (what replace_all walks anyway), then
            // expand the template in ONE replace_all — not a discarded
            // counting pass plus a second expansion pass (#382).
            let count = pattern.find_iter(value).filter(|m| m.is_ok()).count();
            if count == 0 {
                return (value.to_string(), 0);
            }
            let expanded = pattern.replace_all(value, template.as_str()).into_owned();
            (expanded, count)
        };

        let mut total = 0usize;
        let mut templates_changed: Vec<String> = Vec::new();
        if let Some(tmpls) = notetype["tmpls"].as_array_mut() {
            for tmpl in tmpls {
                let mut changed = 0;
                if front {
                    let (new, n) = sub(tmpl["qfmt"].as_str().unwrap_or_default());
                    if n > 0 {
                        tmpl["qfmt"] = json!(new);
                        changed += n;
                    }
                }
                if back {
                    let (new, n) = sub(tmpl["afmt"].as_str().unwrap_or_default());
                    if n > 0 {
                        tmpl["afmt"] = json!(new);
                        changed += n;
                    }
                }
                if changed > 0 {
                    templates_changed.push(tmpl["name"].as_str().unwrap_or_default().to_string());
                    total += changed;
                }
            }
        }
        let mut css_changed = false;
        if css {
            let (new, n) = sub(notetype["css"].as_str().unwrap_or_default());
            if n > 0 {
                notetype["css"] = json!(new);
                css_changed = true;
                total += n;
            }
        }
        if total > 0 {
            self.adapter.update_notetype_legacy(&notetype)?;
        }
        Ok(json!({
            "id": notetype["id"],
            "name": note_type_name,
            "replacements": total,
            "templates_changed": templates_changed,
            "css_changed": css_changed,
        })
        .to_string())
    }

    /// `update_note_type_field_metadata`: per-field editor metadata
    /// (font/size/description), validate-all-then-apply, one persist.
    pub fn update_note_type_field_metadata(
        &self,
        note_type_name: &str,
        updates_json: &str,
    ) -> NativeResult<String> {
        let updates: Vec<Value> = serde_json::from_str(updates_json)
            .map_err(|e| invalid(format!("updates must be a JSON list: {e}")))?;
        let mut notetype = self
            .notetype_legacy_by_name(note_type_name)?
            .ok_or_else(|| invalid(format!("Note type '{note_type_name}' not found")))?;

        let names: HashSet<String> = names_of(&notetype["flds"]).into_iter().collect();
        for (i, up) in updates.iter().enumerate() {
            let name = up["name"].as_str().unwrap_or_default();
            if !names.contains(name) {
                return Err(invalid(format!(
                    "update {i}: field '{name}' not in note type '{note_type_name}'"
                )));
            }
            if up.get("font").is_none_or(Value::is_null)
                && up.get("size").is_none_or(Value::is_null)
                && up.get("description").is_none_or(Value::is_null)
            {
                return Err(invalid(format!(
                    "update {i} (field '{name}'): set at least one of font, size, description"
                )));
            }
        }

        let mut updated: Vec<String> = Vec::new();
        if let Some(flds) = notetype["flds"].as_array_mut() {
            for up in &updates {
                let name = up["name"].as_str().unwrap_or_default();
                if let Some(field) = flds.iter_mut().find(|f| f["name"].as_str() == Some(name)) {
                    if let Some(font) = up.get("font").filter(|v| !v.is_null()) {
                        field["font"] = font.clone();
                    }
                    if let Some(size) = up.get("size").filter(|v| !v.is_null()) {
                        field["size"] = size.clone();
                    }
                    if let Some(description) = up.get("description").filter(|v| !v.is_null()) {
                        field["description"] = description.clone();
                    }
                    updated.push(name.to_string());
                }
            }
        }
        self.adapter.update_notetype_legacy(&notetype)?;
        Ok(json!({
            "id": notetype["id"],
            "name": note_type_name,
            "fields_updated": updated,
        })
        .to_string())
    }

    /// `_migrate_note_type`: change notes' note type via name maps, with the
    /// drop/new-empty reporting and the same validations; on apply, the same
    /// `change_notetype` RPC (and the pylib-mirroring scm bump) as
    /// `models.change`.
    pub fn migrate_note_type(
        &self,
        note_ids: &[i64],
        new_note_type: &str,
        field_map_json: &str,
        template_map_json: &str,
        dry_run: bool,
    ) -> NativeResult<String> {
        let field_map: Vec<(String, String)> =
            serde_json::from_str::<HashMap<String, String>>(field_map_json)
                .map_err(|e| invalid(format!("field_map must be a JSON object: {e}")))?
                .into_iter()
                .collect();
        let template_map: HashMap<String, String> = if template_map_json.is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(template_map_json)
                .map_err(|e| invalid(format!("template_map must be a JSON object: {e}")))?
        };

        // One (id, mid) query (#445): the per-note get_note loop paid a full
        // note-proto RPC per note just to learn the shared source type and
        // validate existence.
        let mut mid_of: HashMap<i64, i64> = HashMap::new();
        if !note_ids.is_empty() {
            let sql = format!(
                "select id, mid from notes where id in ({})",
                crate::read::ids_sql_list(note_ids)
            );
            for r in self.adapter.db_rows(&sql)? {
                if let (Some(id), Some(mid)) = (
                    r.first().and_then(Value::as_i64),
                    r.get(1).and_then(Value::as_i64),
                ) {
                    mid_of.insert(id, mid);
                }
            }
        }
        if let Some(missing) = note_ids.iter().find(|nid| !mid_of.contains_key(nid)) {
            return Err(invalid(format!("Note not found: {missing}")));
        }
        let source_mids: HashSet<i64> = mid_of.values().copied().collect();
        if source_mids.len() != 1 {
            return Err(invalid(
                "All notes must currently share one note type to migrate together.",
            ));
        }
        let source_id = *source_mids.iter().next().expect("one mid");
        let source = self.adapter.notetype_legacy(source_id)?;
        let target = self
            .notetype_legacy_by_name(new_note_type)?
            .ok_or_else(|| invalid(format!("Note type '{new_note_type}' not found")))?;
        if target["id"] == source["id"] {
            return Err(invalid(format!(
                "Notes already use note type '{new_note_type}'."
            )));
        }
        let source_name = source["name"].as_str().unwrap_or_default().to_string();

        let ord_map = |nt: &Value, key: &str| -> Vec<(String, i64)> {
            nt[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|e| {
                            (
                                e["name"].as_str().unwrap_or_default().to_string(),
                                e["ord"].as_i64().unwrap_or(0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let src_fields = ord_map(&source, "flds");
        let tgt_fields = ord_map(&target, "flds");
        let src_lookup: HashMap<&str, i64> =
            src_fields.iter().map(|(n, o)| (n.as_str(), *o)).collect();
        let tgt_lookup: HashMap<&str, i64> =
            tgt_fields.iter().map(|(n, o)| (n.as_str(), *o)).collect();

        if field_map.is_empty() {
            return Err(invalid("field_map is required and must be non-empty"));
        }
        for (old, new) in &field_map {
            if !src_lookup.contains_key(old.as_str()) {
                return Err(invalid(format!(
                    "Source field '{old}' not in note type '{source_name}'"
                )));
            }
            if !tgt_lookup.contains_key(new.as_str()) {
                return Err(invalid(format!(
                    "Target field '{new}' not in note type '{new_note_type}'"
                )));
            }
        }
        let targets: Vec<&String> = field_map.iter().map(|(_, v)| v).collect();
        let mut ambiguous: Vec<&str> = targets
            .iter()
            .filter(|t| targets.iter().filter(|u| u == t).count() > 1)
            .map(|t| t.as_str())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        ambiguous.sort_unstable();
        if !ambiguous.is_empty() {
            return Err(invalid(format!(
                "Multiple source fields map to the same target field(s): {}",
                py_str_list_repr(&ambiguous)
            )));
        }

        let mapped: HashMap<&str, &str> = field_map
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let dropped_fields: Vec<&str> = src_fields
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| !mapped.contains_key(n))
            .collect();
        let mapped_targets: HashSet<&str> = mapped.values().copied().collect();
        let new_empty_fields: Vec<&str> = tgt_fields
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| !mapped_targets.contains(n))
            .collect();

        // template map validation (optional).
        let mut cmap: Option<HashMap<i64, Option<i64>>> = None;
        if !template_map.is_empty() {
            let src_tmpls = ord_map(&source, "tmpls");
            let tgt_tmpls = ord_map(&target, "tmpls");
            let src_t: HashMap<&str, i64> =
                src_tmpls.iter().map(|(n, o)| (n.as_str(), *o)).collect();
            let tgt_t: HashMap<&str, i64> =
                tgt_tmpls.iter().map(|(n, o)| (n.as_str(), *o)).collect();
            for (old, new) in &template_map {
                if !src_t.contains_key(old.as_str()) {
                    return Err(invalid(format!(
                        "Source template '{old}' not in note type '{source_name}'"
                    )));
                }
                if !tgt_t.contains_key(new.as_str()) {
                    return Err(invalid(format!(
                        "Target template '{new}' not in note type '{new_note_type}'"
                    )));
                }
            }
            cmap = Some(
                src_tmpls
                    .iter()
                    .map(|(name, ord)| (*ord, template_map.get(name).map(|t| tgt_t[t.as_str()])))
                    .collect(),
            );
        }

        let changed: Vec<i64> = note_ids.to_vec();
        let result = json!({
            "changed": changed,
            "from_note_type": source_name,
            "to_note_type": target["name"],
            "dropped_fields": dropped_fields,
            "new_empty_fields": new_empty_fields,
            "dry_run": dry_run,
        });
        if dry_run {
            return Ok(result.to_string());
        }

        // pylib models.change: fmap {src_ord: tgt_ord|None} inverted into a
        // target-indexed list of source ords (-1 = nothing maps in).
        let fmap: HashMap<i64, Option<i64>> = src_fields
            .iter()
            .map(|(name, ord)| (*ord, mapped.get(name.as_str()).map(|t| tgt_lookup[*t])))
            .collect();
        let new_fields = convert_legacy_map(&fmap, tgt_fields.len());
        let is_cloze = source["type"].as_i64() == Some(MODEL_CLOZE)
            || target["type"].as_i64() == Some(MODEL_CLOZE);
        let new_templates = match (&cmap, is_cloze) {
            (Some(cmap), false) => {
                convert_legacy_map(cmap, target["tmpls"].as_array().map_or(0, Vec::len))
            }
            _ => Vec::new(),
        };

        // pylib mod_schema → set_schema_modified (scm = ms timestamp), then
        // the change RPC with the current schema stamp.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        self.adapter
            .db_execute("update col set scm=?", &[json!(now_ms)])?;
        let scm = self
            .adapter
            .db_rows("select scm from col")?
            .first()
            .and_then(|r| r.first())
            .and_then(Value::as_i64)
            .unwrap_or(0);
        self.adapter
            .change_notetype(&anki_proto::notetypes::ChangeNotetypeRequest {
                note_ids: changed.clone(),
                new_fields,
                new_templates,
                old_notetype_id: source_id,
                new_notetype_id: target["id"].as_i64().unwrap_or(0),
                current_schema: scm,
                old_notetype_name: source_name,
                is_cloze,
            })?;
        Ok(result.to_string())
    }
}

/// pylib `_convert_legacy_map`: invert {old_ord → new_ord|None} into a list
/// indexed by new ord carrying the old ord (or -1).
fn convert_legacy_map(old_to_new: &HashMap<i64, Option<i64>>, new_count: usize) -> Vec<i32> {
    let new_to_old: HashMap<i64, i64> = old_to_new
        .iter()
        .filter_map(|(old, new)| new.map(|n| (n, *old)))
        .collect();
    (0..new_count as i64)
        .map(|idx| new_to_old.get(&idx).map_or(-1, |v| *v as i32))
        .collect()
}

/// Python-style replacement template (`\1`, `\g<n>`) → regex-crate `${n}`,
/// with `$` escaped so it can't be misread as a group ref.
fn python_replacement_template(replacement: &str) -> String {
    let mut out = String::with_capacity(replacement.len());
    let mut chars = replacement.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '$' {
            out.push_str("$$");
        } else if c == '\\' {
            match chars.peek() {
                Some(d) if d.is_ascii_digit() => {
                    let mut num = String::new();
                    while let Some(d) = chars.peek().filter(|d| d.is_ascii_digit()) {
                        num.push(*d);
                        chars.next();
                    }
                    out.push_str(&format!("${{{num}}}"));
                }
                Some('g') => {
                    // \g<name-or-number>
                    chars.next();
                    if chars.peek() == Some(&'<') {
                        chars.next();
                        let mut name = String::new();
                        for d in chars.by_ref() {
                            if d == '>' {
                                break;
                            }
                            name.push(d);
                        }
                        out.push_str(&format!("${{{name}}}"));
                    } else {
                        out.push('g');
                    }
                }
                Some('\\') => {
                    chars.next();
                    out.push('\\');
                }
                _ => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `_reject_unsound_positional_replace` — wording kept verbatim.
fn reject_unsound_positional_replace(
    old: &[String],
    new: &[String],
    what: &str,
    mislabels: &str,
    mover_tool: &str,
) -> NativeResult<()> {
    let old_index: HashMap<&str, usize> = old
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    for (i, name) in new.iter().enumerate() {
        if let Some(&old_pos) = old_index.get(name.as_str()) {
            if old_pos != i {
                let what_cap = format!("{}{}", what[..1].to_uppercase(), &what[1..]);
                return Err(invalid(format!(
                    "{what_cap} '{name}' would move from position {old_pos} to {i}. \
                     upsert_note_types replaces {what}s by position — it can only rename \
                     a {what} in place, append new {what}s, or drop trailing {what}s; \
                     moving, inserting, or removing a non-trailing {what} this way would \
                     silently mislabel {mislabels}. Use {mover_tool} \
                     (reposition / add / remove / rename) for that."
                )));
            }
        }
    }
    Ok(())
}

/// `_simulate_struct_op` — validate one op against a simulated name list.
fn simulate_struct_op(sim: &mut Vec<String>, op: &Value, i: usize, what: &str) -> NativeResult<()> {
    let kind = op["op"].as_str().unwrap_or_default();
    match kind {
        "add" => {
            let name = op["name"].as_str().unwrap_or_default().to_string();
            if sim.contains(&name) {
                return Err(invalid(format!(
                    "op {i} (add): {what} '{name}' already exists"
                )));
            }
            match op.get("position").filter(|v| !v.is_null()) {
                None => sim.push(name),
                Some(p) => {
                    let pos = p.as_i64().unwrap_or(-1);
                    if pos < 0 || pos as usize > sim.len() {
                        return Err(invalid(format!(
                            "op {i} (add): position {pos} out of range 0..{}",
                            sim.len()
                        )));
                    }
                    sim.insert(pos as usize, name);
                }
            }
        }
        "remove" => {
            let name = op["name"].as_str().unwrap_or_default();
            let Some(idx) = sim.iter().position(|n| n == name) else {
                return Err(invalid(format!(
                    "op {i} (remove): {what} '{name}' not found"
                )));
            };
            if sim.len() == 1 {
                return Err(invalid(format!(
                    "op {i} (remove): a note type must keep at least one {what}"
                )));
            }
            sim.remove(idx);
        }
        "rename" => {
            let name = op["name"].as_str().unwrap_or_default();
            let new = op["new_name"].as_str().unwrap_or_default().to_string();
            let Some(idx) = sim.iter().position(|n| n == name) else {
                return Err(invalid(format!(
                    "op {i} (rename): {what} '{name}' not found"
                )));
            };
            if new != name && sim.contains(&new) {
                return Err(invalid(format!(
                    "op {i} (rename): {what} '{new}' already exists"
                )));
            }
            sim[idx] = new;
        }
        "reposition" => {
            let name = op["name"].as_str().unwrap_or_default().to_string();
            let pos = op["position"].as_i64().unwrap_or(-1);
            let Some(idx) = sim.iter().position(|n| *n == name) else {
                return Err(invalid(format!(
                    "op {i} (reposition): {what} '{name}' not found"
                )));
            };
            if pos < 0 || pos as usize >= sim.len() {
                return Err(invalid(format!(
                    "op {i} (reposition): position {pos} out of range 0..{}",
                    sim.len() - 1
                )));
            }
            sim.remove(idx);
            sim.insert(pos as usize, name);
        }
        other => {
            return Err(invalid(format!("op {i}: unknown op {other:?}")));
        }
    }
    Ok(())
}
