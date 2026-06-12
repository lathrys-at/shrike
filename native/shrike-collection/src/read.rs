//! The read surface (#278 series, step 2): collection_info, list_notes +
//! note serialization, and the embedding-text readers — ports of the
//! CollectionWrapper methods of the same names (the tests/native parity
//! harness compares against the pip core).
//!
//! Typed since #391 phase 2: the read ops return the canonical
//! `shrike-schemas` types directly — no JSON-string assembly here.
//! Serialization to the host wire happens exactly once, at the binding edge,
//! with plain serde (an unset `Option` is an explicit `null` — the one wire
//! convention, the Pydantic shape the schema contract test pins).

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use shrike_ffi::{NativeError, NativeResult};
use shrike_schemas::{
    CollectionInfo, DeckInfo, DeckStat, FieldDetail, ListNotesResponse, Note, NoteTypeDetail,
    NoteTypeInfo, Stats, Summary, TemplateInfo,
};

use crate::{embed_text, CollectionCore};

/// `datetime.fromtimestamp(secs, tz=UTC).isoformat()` for whole seconds:
/// `YYYY-MM-DDTHH:MM:SS+00:00`. (Civil-from-days per Howard Hinnant.)
fn iso_utc(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}+00:00")
}

/// `strftime("%Y-%m-%d")` on a UTC timestamp.
fn date_utc(secs: i64) -> String {
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    format!("{y:04}-{mo:02}-{d:02}")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// One note's `(note_id, field_names, field_values)`. Field names are
/// shared per notetype (#445): the old per-note `Vec<String>` clone copied
/// every field name once per note — ~400k string allocations per rebuild
/// read at 100k notes.
type NoteFieldRow = (i64, std::sync::Arc<Vec<String>>, Vec<String>);

/// The binding-facing field-map row: owned names (the pyo3 wire shape).
pub type OwnedFieldRow = (i64, Vec<String>, Vec<String>);

pub(crate) fn ids_sql_list(ids: &[i64]) -> String {
    ids.iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

impl CollectionCore {
    fn strip_fn(&self) -> impl Fn(&str) -> NativeResult<String> + '_ {
        move |s: &str| self.adapter.strip_html(s)
    }

    /// `(note_id, field_names, field_values)` per existing note, in input
    /// order, from one query over the notes table (ids absent are skipped) —
    /// the port of `CollectionWrapper._note_field_rows`.
    fn note_field_rows(&self, note_ids: &[i64]) -> NativeResult<Vec<NoteFieldRow>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "select id, mid, flds from notes where id in ({})",
            ids_sql_list(note_ids)
        );
        let mut rows: HashMap<i64, (i64, String)> = HashMap::new();
        for r in self.adapter.db_rows(&sql)? {
            let (Some(id), Some(mid), Some(flds)) = (
                r.first().and_then(Value::as_i64),
                r.get(1).and_then(Value::as_i64),
                r.get(2).and_then(Value::as_str),
            ) else {
                return Err(NativeError::internal(
                    "unexpected notes row shape".to_string(),
                ));
            };
            rows.insert(id, (mid, flds.to_string()));
        }
        let mut field_names: HashMap<i64, std::sync::Arc<Vec<String>>> = HashMap::new();
        let mut out = Vec::new();
        for nid in note_ids {
            let Some((mid, flds)) = rows.get(nid) else {
                continue;
            };
            let names = match field_names.get(mid) {
                Some(n) => std::sync::Arc::clone(n),
                None => {
                    let nt = self.adapter.notetype(*mid)?;
                    let names: std::sync::Arc<Vec<String>> =
                        std::sync::Arc::new(nt.fields.into_iter().map(|f| f.name).collect());
                    field_names.insert(*mid, std::sync::Arc::clone(&names));
                    names
                }
            };
            let values: Vec<String> = flds.split('\u{1f}').map(str::to_string).collect();
            out.push((*nid, names, values));
        }
        Ok(out)
    }

    /// Normalized embedding text per note id, "" for missing ids (positions
    /// preserved; a REPEATED id carries its text on the first occurrence
    /// only — the move-out assembly (#445) replaced a full per-note text
    /// clone, and no caller passes duplicates).
    /// The port of `CollectionWrapper.note_texts`.
    pub fn note_texts(&self, note_ids: &[i64]) -> NativeResult<Vec<String>> {
        let strip = self.strip_fn();
        let mut rendered: HashMap<i64, String> = HashMap::new();
        for (nid, names, values) in self.note_field_rows(note_ids)? {
            rendered.insert(nid, embed_text::render_embed_text(&names, &values, &strip)?);
        }
        Ok(note_ids
            .iter()
            .map(|nid| rendered.remove(nid).unwrap_or_default())
            .collect())
    }

    /// Per-note embedding input `(note_id, text, image_names)` — the
    /// multimodal counterpart (`_note_embed_inputs`): text from the shared
    /// render, image names from the RAW field values, de-duplicated in order.
    pub fn note_embed_inputs(
        &self,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, Vec<String>)>> {
        let strip = self.strip_fn();
        let mut by_id: HashMap<i64, (String, Vec<String>)> = HashMap::new();
        for (nid, names, values) in self.note_field_rows(note_ids)? {
            let text = embed_text::render_embed_text(&names, &values, &strip)?;
            let mut images: Vec<String> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            for value in &values {
                for name in embed_text::extract_image_refs(value) {
                    if seen.insert(name.clone()) {
                        images.push(name);
                    }
                }
            }
            by_id.insert(nid, (text, images));
        }
        Ok(note_ids
            .iter()
            .map(|nid| {
                // Move-out, not clone (#445): the rebuild path assembled a
                // second full copy of every note's text here. Same repeated-
                // id caveat as `note_texts` (no caller passes duplicates).
                let (text, images) = by_id.remove(nid).unwrap_or_default();
                (*nid, text, images)
            })
            .collect())
    }

    /// `(note_id, "field", field_name, raw_value)` for non-empty fields — what
    /// the derived-text store ingests (`derived_field_rows`).
    pub fn derived_field_rows(
        &self,
        note_ids: &[i64],
    ) -> NativeResult<Vec<(i64, String, String, String)>> {
        let mut out = Vec::new();
        for (nid, names, values) in self.note_field_rows(note_ids)? {
            for (name, value) in names.iter().zip(values.iter()) {
                if !value.trim().is_empty() {
                    out.push((nid, "field".to_string(), name.clone(), value.clone()));
                }
            }
        }
        Ok(out)
    }

    /// `(note_id, image_names)` for every note whose raw fields reference an
    /// `<img` tag — the recognition sweep's pending-set source (#445): the
    /// sweep previously rendered the FULL collection's embedding inputs
    /// (notetype lookups + normalization + strip per field) once per batch
    /// and discarded the text. One SQL pass with an ASCII `lower()`
    /// pre-filter — exactly the extractor's own ASCII-case-insensitive
    /// probe, so the filter can never skip a note the extractor would
    /// return names for — then the raw-field extractor (same per-field
    /// extraction + in-order dedupe as `note_embed_inputs`).
    pub fn note_image_refs(&self) -> NativeResult<Vec<(i64, Vec<String>)>> {
        let mut out = Vec::new();
        for r in self
            .adapter
            .db_rows("select id, flds from notes where instr(lower(flds), '<img') > 0")?
        {
            let (Some(id), Some(flds)) = (
                r.first().and_then(Value::as_i64),
                r.get(1).and_then(Value::as_str),
            ) else {
                return Err(NativeError::internal(
                    "unexpected notes row shape".to_string(),
                ));
            };
            let mut images: Vec<String> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            for value in flds.split('\u{1f}') {
                for name in embed_text::extract_image_refs(value) {
                    if seen.insert(name.clone()) {
                        images.push(name);
                    }
                }
            }
            if !images.is_empty() {
                out.push((id, images));
            }
        }
        Ok(out)
    }

    /// `(note_id, [leaf tags])` for every tagged note — the tag-centroid
    /// layer's membership source (#179): ONE pass over `notes.tags` (Anki
    /// keeps it space-delimited), exact leaf strings; hierarchy roll-up is
    /// the consumer's prefix aggregation.
    pub fn note_tag_rows(&self) -> NativeResult<Vec<(i64, Vec<String>)>> {
        let mut out = Vec::new();
        for r in self
            .adapter
            .db_rows("select id, tags from notes where tags != ''")?
        {
            let (Some(id), Some(tags)) = (
                r.first().and_then(Value::as_i64),
                r.get(1).and_then(Value::as_str),
            ) else {
                return Err(NativeError::internal(
                    "unexpected notes tag row shape".to_string(),
                ));
            };
            let leaves: Vec<String> = tags.split_whitespace().map(str::to_string).collect();
            if !leaves.is_empty() {
                out.push((id, leaves));
            }
        }
        Ok(out)
    }

    /// Whether ANY of `note_ids` currently carries a tag — the SQL half of
    /// the tag-centroid relevance probe (#445): one scoped aggregate lets an
    /// untagged write op skip the O(tagged-notes) recompute entirely.
    pub fn any_tagged(&self, note_ids: &[i64]) -> NativeResult<bool> {
        if note_ids.is_empty() {
            return Ok(false);
        }
        let sql = format!(
            "select exists(select 1 from notes where tags != '' and id in ({}))",
            ids_sql_list(note_ids)
        );
        let rows = self.adapter.db_rows(&sql)?;
        rows.first()
            .and_then(|r| r.first())
            .and_then(Value::as_i64)
            .map(|n| n != 0)
            .ok_or_else(|| NativeError::internal("unexpected exists row shape".to_string()))
    }

    /// Total note count via one SQL aggregate (#445): the tag-centroid
    /// refresh previously ran `find_notes("")` — materializing every note id
    /// through a protobuf SearchResponse — just to take `.len()`.
    pub fn note_count(&self) -> NativeResult<usize> {
        let rows = self.adapter.db_rows("select count(*) from notes")?;
        rows.first()
            .and_then(|r| r.first())
            .and_then(Value::as_i64)
            .map(|n| n as usize)
            .ok_or_else(|| NativeError::internal("unexpected count row shape".to_string()))
    }

    /// The raw Anki search escape hatch (`collection_query`): the full grammar
    /// straight to search, `list_notes`-shaped (`total` = the full match
    /// count before `limit`).
    pub fn query(
        &self,
        search: &str,
        with_fields: bool,
        limit: usize,
    ) -> NativeResult<ListNotesResponse> {
        let note_ids = self.adapter.search_notes(search)?;
        let total = note_ids.len();
        let take: Vec<i64> = note_ids.into_iter().take(limit).collect();
        Ok(ListNotesResponse {
            notes: self.typed_notes(&take, with_fields)?,
            total: total as i64,
            limit: limit as i64,
        })
    }

    /// Per note: the FULL raw field map `(note_id, [(name, value)...])` —
    /// unlike `derived_field_rows`, empty fields are included (`note_field_map`
    /// feeds substring_info + the find/replace preview, which want every
    /// field). Missing ids are absent.
    pub fn note_field_map(&self, note_ids: &[i64]) -> NativeResult<Vec<OwnedFieldRow>> {
        // Owned names on this surface (the pyo3 binding's wire shape); the
        // Arc sharing is an internal property of `note_field_rows` (#445).
        Ok(self
            .note_field_rows(note_ids)?
            .into_iter()
            .map(|(nid, names, values)| (nid, names.as_ref().clone(), values))
            .collect())
    }

    /// One field value through the embedding normalization — the parity-test
    /// surface for byte-identity against the Python normalizer.
    pub fn normalize_text(&self, value: &str) -> NativeResult<String> {
        embed_text::normalize_for_embedding(value, &self.strip_fn())
    }

    /// Map a deck reference (name, numeric id, `#id`) to a deck name —
    /// `_resolve_deck_ref`. None = an explicit id matching no deck.
    pub fn resolve_deck_ref(&self, reference: &str) -> NativeResult<Option<String>> {
        let name_of = |id: i64| -> NativeResult<Option<String>> {
            Ok(self
                .adapter
                .deck_names()?
                .into_iter()
                .find(|(did, _)| *did == id)
                .map(|(_, name)| name))
        };
        if let Some(digits) = reference.strip_prefix('#') {
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                return name_of(digits.parse::<i64>().unwrap_or(0));
            }
        }
        if !reference.is_empty() && reference.bytes().all(|b| b.is_ascii_digit()) {
            if let Some(name) = name_of(reference.parse::<i64>().unwrap_or(0))? {
                return Ok(Some(name));
            }
            return Ok(Some(reference.to_string()));
        }
        Ok(Some(reference.to_string()))
    }

    /// Structured filters → notes, shape-identical to the wrapper's
    /// `list_notes` (`{notes, total, limit}`). `modified_since` is an
    /// epoch-seconds cutoff (the host parses ISO).
    /// Divergence from the Python original: "no filter given" raises
    /// invalid_input here instead of returning an `{"error": ...}` dict (the
    /// facade owns that wire shape).
    #[allow(clippy::too_many_arguments)]
    pub fn list_notes(
        &self,
        ids: Option<&[i64]>,
        deck: Option<&str>,
        tags: Option<&[String]>,
        note_type: Option<&str>,
        modified_since: Option<i64>,
        with_fields: bool,
        limit: usize,
    ) -> NativeResult<ListNotesResponse> {
        // ids-only fast path (mirrors _get_notes_by_ids).
        if let Some(ids) = ids {
            if deck.is_none() && tags.is_none() && note_type.is_none() && modified_since.is_none() {
                let take: Vec<i64> = ids.iter().take(limit).copied().collect();
                let notes = self.typed_notes(&take, with_fields)?;
                let total = notes.len() as i64;
                return Ok(ListNotesResponse {
                    notes,
                    total,
                    limit: limit as i64,
                });
            }
        }

        // _build_scope_query.
        let mut parts: Vec<String> = Vec::new();
        if let Some(deck) = deck {
            match self.resolve_deck_ref(deck)? {
                None => {
                    return Ok(ListNotesResponse {
                        notes: Vec::new(),
                        total: 0,
                        limit: limit as i64,
                    });
                }
                Some(resolved) => parts.push(format!("\"deck:{resolved}\"")),
            }
        }
        if let Some(tags) = tags {
            for tag in tags {
                match tag.strip_prefix('-') {
                    Some(neg) => parts.push(format!("-tag:{neg}")),
                    None => parts.push(format!("tag:{tag}")),
                }
            }
        }
        if let Some(nt) = note_type {
            parts.push(format!("\"note:{nt}\""));
        }
        let mut combined = if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        };
        if let Some(ids) = ids {
            let id_query = format!("nid:{}", ids_sql_list(ids));
            combined = Some(match combined {
                Some(c) => format!("{id_query} {c}"),
                None => id_query,
            });
        }
        let combined = match combined {
            Some(c) => c,
            None if modified_since.is_some() => "deck:*".to_string(),
            None => {
                return Err(NativeError::invalid_input(
                    "At least one filter is required".to_string(),
                ));
            }
        };

        let mut note_ids = self.adapter.search_notes(&combined)?;
        if let Some(cutoff) = modified_since {
            let recent: HashSet<i64> = self
                .adapter
                .db_rows(&format!("select id from notes where mod >= {cutoff}"))?
                .into_iter()
                .filter_map(|r| r.first().and_then(Value::as_i64))
                .collect();
            note_ids.retain(|nid| recent.contains(nid));
        }
        let total = note_ids.len();
        note_ids.truncate(limit);
        Ok(ListNotesResponse {
            notes: self.typed_notes(&note_ids, with_fields)?,
            total: total as i64,
            limit: limit as i64,
        })
    }

    /// The typed notes as internal-wire `Value` dicts — the kernel's search
    /// assembly annotates candidates in place (`substring`/`score`/...), so it
    /// wants mutable JSON objects. Plain serde (#391 to_wire retirement): a
    /// meta-mode note carries an explicit `"content": null`, never a dropped
    /// key — every consumer reads via `.get(..)` + `as_*`, which treat `Null`
    /// exactly like absent.
    pub fn note_dicts(&self, note_ids: &[i64], with_fields: bool) -> NativeResult<Vec<Value>> {
        self.typed_notes(note_ids, with_fields)?
            .iter()
            .map(|note| {
                serde_json::to_value(note)
                    .map_err(|e| NativeError::internal(format!("note wire shape: {e}")))
            })
            .collect()
    }

    /// Read many notes in a fixed number of queries — `_notes_to_dicts`:
    /// note rows + first-card decks via the DB proxy, names from the
    /// notetype/deck services; input order kept, missing ids skipped.
    fn typed_notes(&self, note_ids: &[i64], with_fields: bool) -> NativeResult<Vec<Note>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        let id_list = ids_sql_list(note_ids);
        let mut note_rows: HashMap<i64, (i64, String, String, i64)> = HashMap::new();
        for r in self.adapter.db_rows(&format!(
            "select id, mid, tags, flds, mod from notes where id in ({id_list})"
        ))? {
            let (Some(id), Some(mid), Some(tags), Some(flds), Some(modified)) = (
                r.first().and_then(Value::as_i64),
                r.get(1).and_then(Value::as_i64),
                r.get(2).and_then(Value::as_str),
                r.get(3).and_then(Value::as_str),
                r.get(4).and_then(Value::as_i64),
            ) else {
                return Err(NativeError::internal(
                    "unexpected notes row shape".to_string(),
                ));
            };
            note_rows.insert(id, (mid, tags.to_string(), flds.to_string(), modified));
        }
        // First card's deck per note (lowest ord).
        let mut deck_by_nid: HashMap<i64, i64> = HashMap::new();
        for r in self.adapter.db_rows(&format!(
            "select nid, did from cards where nid in ({id_list}) order by ord"
        ))? {
            if let (Some(nid), Some(did)) = (
                r.first().and_then(Value::as_i64),
                r.get(1).and_then(Value::as_i64),
            ) {
                deck_by_nid.entry(nid).or_insert(did);
            }
        }
        let deck_names: HashMap<i64, String> = self.adapter.deck_names()?.into_iter().collect();

        let mut model_cache: HashMap<i64, (String, Vec<String>)> = HashMap::new();
        let mut out = Vec::new();
        for nid in note_ids {
            let Some((mid, tags, flds, modified)) = note_rows.get(nid) else {
                continue;
            };
            let (name, field_names) = match model_cache.get(mid) {
                Some(c) => c.clone(),
                None => {
                    let cached = match self.adapter.notetype(*mid) {
                        Ok(nt) => (
                            nt.name.clone(),
                            nt.fields.into_iter().map(|f| f.name).collect(),
                        ),
                        Err(_) => ("Unknown".to_string(), Vec::new()),
                    };
                    model_cache.insert(*mid, cached.clone());
                    cached
                }
            };
            let deck = deck_by_nid
                .get(nid)
                .and_then(|did| deck_names.get(did).cloned())
                .unwrap_or_else(|| "Default".to_string());
            let content = with_fields.then(|| {
                field_names
                    .iter()
                    .cloned()
                    .zip(flds.split('\u{1f}').map(str::to_string))
                    .collect()
            });
            out.push(Note {
                id: *nid,
                note_type: name,
                deck,
                tags: tags.split_whitespace().map(str::to_string).collect(),
                modified: iso_utc(*modified),
                content,
            });
        }
        Ok(out)
    }

    /// `collection_info` — the wrapper's sectioned info, typed: a requested
    /// section is `Some`, an unrequested one `None` (the wire helper omits
    /// it, exactly like the hand-built dict it replaced).
    /// `sections` mirrors `include` (`"all"` expands; empty = summary).
    pub fn collection_info(
        &self,
        sections: &[String],
        detail_names: &[String],
    ) -> NativeResult<CollectionInfo> {
        const ALL: [&str; 5] = ["summary", "note_types", "decks", "tags", "stats"];
        let sections: Vec<&str> = if sections.iter().any(|s| s == "all") {
            ALL.to_vec()
        } else if sections.is_empty() {
            vec!["summary"]
        } else {
            sections.iter().map(String::as_str).collect()
        };
        let detail: HashSet<&str> = detail_names.iter().map(String::as_str).collect();
        let mut result = CollectionInfo::default();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let tree = if sections.iter().any(|s| *s == "summary" || *s == "stats") {
            Some(self.adapter.deck_tree(now)?)
        } else {
            None
        };
        let counts = if sections.iter().any(|s| *s == "decks" || *s == "stats") {
            Some(self.note_counts_by_deck()?)
        } else {
            None
        };

        for section in &sections {
            match *section {
                "summary" => {
                    let tree = tree.as_ref().expect("tree computed for summary");
                    result.summary = Some(self.info_summary(tree)?);
                }
                "note_types" => {
                    result.note_types = Some(self.info_note_types(&detail)?);
                }
                "decks" => {
                    let counts = counts.as_ref().expect("counts computed for decks");
                    let decks: Vec<DeckInfo> = self
                        .adapter
                        .deck_names()?
                        .into_iter()
                        .map(|(id, name)| {
                            let count = counts.get(&name).copied().unwrap_or(0);
                            DeckInfo {
                                name,
                                id,
                                note_count: count as i64,
                            }
                        })
                        .collect();
                    result.decks = Some(decks);
                }
                "tags" => {
                    result.tags = Some(self.adapter.all_tags()?);
                }
                "stats" => {
                    let tree = tree.as_ref().expect("tree computed for stats");
                    let counts = counts.as_ref().expect("counts computed for stats");
                    result.stats = Some(self.info_stats(tree, counts)?);
                }
                other => {
                    return Err(NativeError::invalid_input(format!(
                        "unknown collection_info section: {other}"
                    )));
                }
            }
        }
        Ok(result)
    }

    fn count_scalar(&self, sql: &str) -> NativeResult<i64> {
        self.adapter
            .db_rows(sql)?
            .first()
            .and_then(|r| r.first())
            .and_then(Value::as_i64)
            .ok_or_else(|| NativeError::internal(format!("scalar query shape: {sql}")))
    }

    fn info_summary(&self, tree: &anki_proto::decks::DeckTreeNode) -> NativeResult<Summary> {
        let total_due: i64 = tree
            .children
            .iter()
            .map(|t| i64::from(t.review_count) + i64::from(t.learn_count))
            .sum();
        let crt = self.count_scalar("select crt from col")?;
        let mod_ms = self.adapter.col_mod()?;
        Ok(Summary {
            path: self.collection_path.clone(),
            created: date_utc(crt),
            modified: iso_utc(mod_ms / 1000),
            notes: self.count_scalar("select count() from notes")?,
            cards: self.count_scalar("select count() from cards")?,
            decks: self.adapter.deck_names()?.len() as i64,
            note_types: self.adapter.notetype_names()?.len() as i64,
            tags: self.adapter.all_tags()?.len() as i64,
            due_today: total_due,
        })
    }

    fn info_note_types(&self, detail: &HashSet<&str>) -> NativeResult<Vec<NoteTypeInfo>> {
        let mut out: Vec<NoteTypeInfo> = Vec::new();
        for (id, _name) in self.adapter.notetype_names()? {
            let nt = self.adapter.notetype(id)?;
            let is_cloze = nt.config.as_ref().is_some_and(|c| {
                c.kind == anki_proto::notetypes::notetype::config::Kind::Cloze as i32
            });
            let entry_detail = detail.contains(nt.name.as_str()).then(|| {
                let templates: Vec<TemplateInfo> = nt
                    .templates
                    .iter()
                    .map(|t| {
                        let cfg = t.config.clone().unwrap_or_default();
                        TemplateInfo {
                            name: t.name.clone(),
                            front: cfg.q_format,
                            back: cfg.a_format,
                        }
                    })
                    .collect();
                let fields: Vec<FieldDetail> = nt
                    .fields
                    .iter()
                    .map(|f| {
                        let cfg = f.config.clone().unwrap_or_default();
                        FieldDetail {
                            name: f.name.clone(),
                            font: cfg.font_name,
                            size: i64::from(cfg.font_size),
                            description: cfg.description,
                        }
                    })
                    .collect();
                let css = nt
                    .config
                    .as_ref()
                    .map(|c| c.css.clone())
                    .unwrap_or_default();
                NoteTypeDetail {
                    templates,
                    css,
                    fields,
                }
            });
            out.push(NoteTypeInfo {
                name: nt.name,
                id: nt.id,
                fields: nt.fields.into_iter().map(|f| f.name).collect(),
                r#type: if is_cloze { "cloze" } else { "standard" }.to_owned(),
                detail: entry_detail,
            });
        }
        Ok(out)
    }

    /// Note count per deck (including subdecks and filtered-deck originals),
    /// one pass — `_note_counts_by_deck`.
    fn note_counts_by_deck(&self) -> NativeResult<HashMap<String, usize>> {
        let id_to_name: HashMap<i64, String> = self.adapter.deck_names()?.into_iter().collect();
        let mut nids_by_deck: HashMap<i64, HashSet<i64>> = HashMap::new();
        for r in self.adapter.db_rows(
            "select nid, did from cards union select nid, odid from cards where odid != 0",
        )? {
            if let (Some(nid), Some(did)) = (
                r.first().and_then(Value::as_i64),
                r.get(1).and_then(Value::as_i64),
            ) {
                nids_by_deck.entry(did).or_default().insert(nid);
            }
        }
        let mut rolled: HashMap<String, HashSet<i64>> = HashMap::new();
        for (did, nids) in &nids_by_deck {
            let Some(name) = id_to_name.get(did) else {
                continue;
            };
            let parts: Vec<&str> = name.split("::").collect();
            for i in 1..=parts.len() {
                rolled
                    .entry(parts[..i].join("::"))
                    .or_default()
                    .extend(nids.iter().copied());
            }
        }
        Ok(id_to_name
            .into_values()
            .map(|name| {
                let count = rolled.get(&name).map_or(0, HashSet::len);
                (name, count)
            })
            .collect())
    }

    fn info_stats(
        &self,
        tree: &anki_proto::decks::DeckTreeNode,
        note_counts: &HashMap<String, usize>,
    ) -> NativeResult<Stats> {
        let total_due: i64 = tree
            .children
            .iter()
            .map(|t| i64::from(t.review_count) + i64::from(t.learn_count))
            .sum();
        let total_new: i64 = tree.children.iter().map(|t| i64::from(t.new_count)).sum();

        let mut decks_summary = std::collections::BTreeMap::new();
        fn walk(
            node: &anki_proto::decks::DeckTreeNode,
            prefix: &str,
            counts: &HashMap<String, usize>,
            out: &mut std::collections::BTreeMap<String, DeckStat>,
        ) {
            let name = if prefix.is_empty() {
                node.name.clone()
            } else {
                format!("{prefix}::{}", node.name)
            };
            let due = i64::from(node.review_count) + i64::from(node.learn_count);
            out.insert(
                name.clone(),
                DeckStat {
                    notes: counts.get(&name).copied().unwrap_or(0) as i64,
                    due,
                },
            );
            for child in &node.children {
                walk(child, &name, counts, out);
            }
        }
        for top in &tree.children {
            walk(top, "", note_counts, &mut decks_summary);
        }
        Ok(Stats {
            total_notes: self.count_scalar("select count() from notes")?,
            total_cards: self.count_scalar("select count() from cards")?,
            cards_due_today: total_due,
            new_cards: total_new,
            decks_summary,
        })
    }
}
