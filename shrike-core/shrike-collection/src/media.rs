//! Media + maintenance (#278 series, step 5a) — the LOCAL halves of the #70
//! media tools (store-from-bytes, fetch/list/delete, media check) and the #89
//! collection_prune, ported from CollectionWrapper. The URL-fetch path (the
//! SSRF guard + IP pinning) is deliberately NOT here — it is trust-boundary
//! code and lands as its own step (5b) under the security-review gate.

use serde_json::Value;
use shrike_error::NativeResult;
use shrike_schemas::{
    CollectionCheckResponse, CollectionPruneResponse, DeleteMediaResponse, ListMediaResponse,
    MediaFetchResult, MediaFileInfo, PruneEmptyCards, PruneEmptyNotes, PruneUnusedMedia,
    PruneUnusedTags, StoreMediaResult,
};

use crate::{embed_text, CollectionCore};

// The basename path-traversal guard + the extension<->MIME map live in the
// inbound-media crate (`shrike-media`, #711) — the one home both the store
// write tail (here) and the kernel's fetch/decode path read. Imported, not
// re-defined.
use shrike_media::{guess_mime, mime_extension, safe_media_name};

/// `_path_within_any_root`: containment on resolved real paths (canonicalize
/// collapses `..` and resolves symlinks on both sides), component-aware (the
/// `/srv/media-evil` vs `/srv/media` prefix bug can't happen on `Path`
/// components).
pub fn path_within_any_root(path: &str, roots: &[String]) -> bool {
    let Ok(target) = std::fs::canonicalize(path) else {
        return false;
    };
    roots.iter().any(|root| {
        std::fs::canonicalize(root)
            .map(|r| target.starts_with(&r))
            .unwrap_or(false)
    })
}

/// The byte-source size cap — caller-supplied/downloaded bytes only; a
/// server-local `path` inside an operator-configured root is deliberately
/// uncapped.
fn check_media_size(len: usize) -> NativeResult<()> {
    if len > shrike_media::MEDIA_MAX_BYTES {
        return Err(shrike_error::NativeError::invalid_input(format!(
            "file exceeds the {}-byte limit",
            shrike_media::MEDIA_MAX_BYTES
        )));
    }
    Ok(())
}

pub use crate::contract::{PreparedMedia, PreparedMediaSource};

impl CollectionCore {
    /// The shared write tail of every store path — the full `_write_one_media`
    /// semantics: extension derived from `content_type` when the name lacks
    /// one, basename-sanitized, Anki collision handling, `deduped` = identical
    /// content already existed (the caller must use the RETURNED filename).
    /// `index` echoes the caller's batch position. The size cap lives with
    /// the byte sources ([`check_media_size`]), not here.
    fn write_media_bytes(
        &self,
        index: i64,
        mut name: String,
        data: &[u8],
        content_type: Option<&str>,
    ) -> NativeResult<StoreMediaResult> {
        // Derive an extension from the HTTP type when the name lacks one
        // (pylib's add_extension_based_on_mime map).
        let basename_has_ext = name.rsplit('/').next().unwrap_or(&name).contains('.');
        if name.is_empty() || !basename_has_ext {
            if let Some(ct) = content_type {
                if let Some(ext) = mime_extension(ct) {
                    if name.is_empty() {
                        name = "media".to_string();
                    }
                    name.push_str(ext);
                }
            }
        }
        let safe = safe_media_name(&name);
        if safe.is_empty() {
            return Err(shrike_error::NativeError::invalid_input(
                "could not determine a filename",
            ));
        }
        let existed = std::path::Path::new(&self.media_dir).join(&safe).is_file();
        let stored = self.adapter.add_media_file(&safe, data)?;
        Ok(StoreMediaResult::Stored {
            index,
            mime: guess_mime(&stored).map(str::to_string),
            size_bytes: data.len() as i64,
            deduped: existed && stored == safe,
            filename: stored,
        })
    }

    /// Store one media item from prepared bytes. The single-item entry knows
    /// no batch position, so `index` is 0 — the host overwrites it with the
    /// caller's.
    pub fn store_media_bytes(
        &self,
        filename: Option<&str>,
        data: &[u8],
        content_type: Option<&str>,
    ) -> NativeResult<StoreMediaResult> {
        check_media_size(data.len())?;
        self.write_media_bytes(
            0,
            filename.unwrap_or_default().to_string(),
            data,
            content_type,
        )
    }

    /// The server-local `path` source (zero-copy intent; the #164/#170
    /// gates): honored only with configured roots, after `..`/symlink
    /// resolution, contained in one of them.
    fn store_path_item(
        &self,
        path: &str,
        index: i64,
        path_roots: &[String],
    ) -> NativeResult<StoreMediaResult> {
        use shrike_error::{ErrorKind, NativeError, ResultExt};
        if path_roots.is_empty() {
            return Err(NativeError::invalid_input(
                "server-local paths are not enabled (set --media-path-root on a \
                 purely-local daemon)",
            ));
        }
        let target = std::fs::canonicalize(path)
            .map_err(|_| NativeError::invalid_input(format!("file not found: {path}")))?;
        let target_str = target.to_string_lossy().to_string();
        if !path_within_any_root(&target_str, path_roots) {
            return Err(NativeError::invalid_input(
                "path is outside the configured media root(s)",
            ));
        }
        if !target.is_file() {
            return Err(NativeError::invalid_input(format!(
                "file not found: {path}"
            )));
        }
        let base = safe_media_name(&target_str);
        let data = std::fs::read(&target).context(ErrorKind::InvalidInput, "read failed")?;
        self.write_media_bytes(index, base, &data, None)
    }

    /// The write half of the kernel's re-homed store (#391): byte sources
    /// were fetched/decoded off-actor on the kernel's blocking pool; `path`
    /// items run their gates here (containment is collection policy). One
    /// collection job per batch; per-item errors never sink it.
    pub fn store_prepared_media(
        &self,
        prepared: &[PreparedMedia],
        path_roots: &[String],
    ) -> NativeResult<Vec<StoreMediaResult>> {
        let mut results: Vec<StoreMediaResult> = Vec::new();
        for p in prepared {
            let result = match &p.source {
                PreparedMediaSource::Bytes {
                    name,
                    data,
                    content_type,
                } => check_media_size(data.len()).and_then(|()| {
                    self.write_media_bytes(p.index, name.clone(), data, content_type.as_deref())
                }),
                PreparedMediaSource::Path { path } => {
                    self.store_path_item(path, p.index, path_roots)
                }
                PreparedMediaSource::Failed { error } => {
                    Err(shrike_error::NativeError::invalid_input(error.clone()))
                }
            };
            results.push(result.unwrap_or_else(|e| StoreMediaResult::Error {
                index: p.index,
                filename: p.filename.clone(),
                error: e.message,
            }));
        }
        Ok(results)
    }

    /// Resolve filenames to where their bytes live — never the bytes
    /// (`_fetch_media`; the host layer fills the serving `url`).
    pub fn fetch_media(&self, filenames: &[String]) -> NativeResult<Vec<MediaFetchResult>> {
        let mut results: Vec<MediaFetchResult> = Vec::new();
        for fn_ in filenames {
            let safe = safe_media_name(fn_);
            let path = if safe.is_empty() {
                None
            } else {
                let p = std::path::Path::new(&self.media_dir).join(&safe);
                std::fs::metadata(&p)
                    .ok()
                    .filter(|m| m.is_file())
                    .map(|m| (p, m.len()))
            };
            results.push(match path {
                None => MediaFetchResult::Missing {
                    filename: fn_.clone(),
                },
                Some((p, size)) => MediaFetchResult::Found {
                    path: p.to_string_lossy().to_string(),
                    url: None,
                    mime: guess_mime(&safe).map(str::to_string),
                    size_bytes: size as i64,
                    filename: safe,
                },
            });
        }
        Ok(results)
    }

    /// List media files (sorted, optional glob `pattern`, optional limit) —
    /// `_list_media`. The host layer fills each file's serving `url`.
    pub fn list_media(
        &self,
        pattern: Option<&str>,
        limit: Option<usize>,
    ) -> NativeResult<ListMediaResponse> {
        let mut entries: Vec<(String, u64)> = Vec::new();
        if let Ok(read_dir) = std::fs::read_dir(&self.media_dir) {
            for entry in read_dir.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(pattern) = pattern {
                    if !glob_match(pattern, &name) {
                        continue;
                    }
                }
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        entries.push((name, meta.len()));
                    }
                }
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let count = entries.len();
        if let Some(limit) = limit {
            entries.truncate(limit);
        }
        let files: Vec<MediaFileInfo> = entries
            .into_iter()
            .map(|(name, size)| MediaFileInfo {
                url: None,
                mime: guess_mime(&name).map(str::to_string),
                size_bytes: size as i64,
                filename: name,
            })
            .collect();
        Ok(ListMediaResponse {
            media_dir: self.media_dir.clone(),
            count: count as i64,
            files,
        })
    }

    /// Move existing media files to Anki's recoverable trash (`_delete_media`;
    /// result lists echo the caller's references).
    pub fn delete_media(&self, filenames: &[String]) -> NativeResult<DeleteMediaResponse> {
        let mut deleted: Vec<String> = Vec::new();
        let mut not_found: Vec<String> = Vec::new();
        let mut to_trash: Vec<String> = Vec::new();
        for fn_ in filenames {
            let safe = safe_media_name(fn_);
            let exists =
                !safe.is_empty() && std::path::Path::new(&self.media_dir).join(&safe).is_file();
            if exists {
                to_trash.push(safe);
                deleted.push(fn_.clone());
            } else {
                not_found.push(fn_.clone());
            }
        }
        if !to_trash.is_empty() {
            self.adapter.trash_media_files(&to_trash)?;
        }
        Ok(DeleteMediaResponse { deleted, not_found })
    }

    /// Read-only media diagnostics (`_media_check`).
    pub fn media_check(&self) -> NativeResult<CollectionCheckResponse> {
        let report = self.adapter.check_media()?;
        Ok(CollectionCheckResponse {
            media_dir: self.media_dir.clone(),
            unused: report.unused,
            missing: report.missing,
            missing_media_notes: report.missing_media_notes,
            have_trash: report.have_trash,
        })
    }

    /// `_find_empty_notes`: ids whose every field is blank (no text AND no
    /// media), via the ported `field_is_blank` over the raw notes table.
    fn find_empty_notes(&self) -> NativeResult<Vec<i64>> {
        let strip = |s: &str| self.adapter.strip_html(s);
        let mut empty = Vec::new();
        for row in self.adapter.db_rows("select id, flds from notes")? {
            let (Some(id), Some(flds)) = (
                row.first().and_then(Value::as_i64),
                row.get(1).and_then(Value::as_str),
            ) else {
                continue;
            };
            let mut all_blank = true;
            for value in flds.split('\u{1f}') {
                if !embed_text::field_is_blank(value, &strip)? {
                    all_blank = false;
                    break;
                }
            }
            if all_blank {
                empty.push(id);
            }
        }
        Ok(empty)
    }

    /// `_unused_tag_names`: registered tags no note carries (hierarchical,
    /// case-insensitive — `a` is used when a note has `a::b`).
    fn unused_tag_names(&self) -> NativeResult<Vec<String>> {
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        for row in self.adapter.db_rows("select distinct tags from notes")? {
            let Some(tagstr) = row.first().and_then(Value::as_str) else {
                continue;
            };
            for tag in tagstr.split_whitespace() {
                let lower = tag.to_lowercase();
                let parts: Vec<&str> = lower.split("::").collect();
                for i in 1..=parts.len() {
                    used.insert(parts[..i].join("::"));
                }
            }
        }
        Ok(self
            .adapter
            .all_tags()?
            .into_iter()
            .filter(|t| !used.contains(&t.to_lowercase()))
            .collect())
    }

    /// `_prune` (#89): the four cleanups, preview-by-default at the tool
    /// layer. Returns the typed response plus the removed-note-id list out of
    /// band (kernel-internal — the host's index maintenance, never the wire).
    pub fn prune(
        &self,
        unused_tags: bool,
        empty_notes: bool,
        empty_cards: bool,
        unused_media: bool,
        dry_run: bool,
    ) -> NativeResult<(CollectionPruneResponse, Vec<i64>)> {
        let mut response = CollectionPruneResponse {
            dry_run,
            unused_tags: None,
            empty_notes: None,
            empty_cards: None,
            unused_media: None,
        };
        let mut removed_note_ids: Vec<i64> = Vec::new();

        let mut empty_note_ids: Vec<i64> = Vec::new();
        if empty_notes {
            empty_note_ids = self.find_empty_notes()?;
            if !dry_run && !empty_note_ids.is_empty() {
                self.adapter.remove_notes(&empty_note_ids)?;
            }
            removed_note_ids.extend(&empty_note_ids);
            response.empty_notes = Some(PruneEmptyNotes {
                removed: empty_note_ids.clone(),
            });
        }

        if empty_cards {
            let report = self.adapter.get_empty_cards()?;
            let card_ids: Vec<i64> = report
                .notes
                .iter()
                .flat_map(|n| n.card_ids.iter().copied())
                .collect();
            let mut notes_deleted: Vec<i64> = report
                .notes
                .iter()
                .filter(|n| n.will_delete_note)
                .map(|n| n.note_id)
                .collect();
            if dry_run {
                // Empty notes go first on apply — don't double-list them.
                let already: std::collections::HashSet<i64> =
                    empty_note_ids.iter().copied().collect();
                notes_deleted.retain(|nid| !already.contains(nid));
            } else if !card_ids.is_empty() {
                self.adapter.remove_cards(&card_ids)?;
            }
            removed_note_ids.extend(&notes_deleted);
            response.empty_cards = Some(PruneEmptyCards {
                cards_removed: card_ids.len() as i64,
                notes_deleted,
            });
        }

        if unused_tags {
            let names = self.unused_tag_names()?;
            if !dry_run && !names.is_empty() {
                self.adapter.clear_unused_tags()?;
            }
            response.unused_tags = Some(PruneUnusedTags {
                removed: names.len() as i64,
                tags: names,
            });
        }

        if unused_media {
            // Last, so an apply catches media orphaned by the deletions above.
            let media_files = self.adapter.check_media()?.unused;
            if !dry_run && !media_files.is_empty() {
                self.adapter.trash_media_files(&media_files)?;
            }
            response.unused_media = Some(PruneUnusedMedia {
                removed: media_files.len() as i64,
                files: media_files,
            });
        }

        Ok((response, removed_note_ids))
    }
}

/// fnmatch-style glob over a filename: `*`, `?`, and `[...]` classes — the
/// subset `fnmatch.fnmatch` provides for `list_media`'s `pattern`
/// (case-sensitive: media names are exact on every platform Shrike serves).
fn glob_match(pattern: &str, name: &str) -> bool {
    fn inner(p: &[char], n: &[char]) -> bool {
        match p.first() {
            None => n.is_empty(),
            Some('*') => (0..=n.len()).any(|skip| inner(&p[1..], &n[skip..])),
            Some('?') => !n.is_empty() && inner(&p[1..], &n[1..]),
            Some('[') => {
                let Some(end) = p.iter().position(|c| *c == ']').filter(|e| *e > 1) else {
                    // unterminated class: literal '['
                    return !n.is_empty() && n[0] == '[' && inner(&p[1..], &n[1..]);
                };
                let Some(first) = n.first() else { return false };
                let (negate, class) = if p[1] == '!' {
                    (true, &p[2..end])
                } else {
                    (false, &p[1..end])
                };
                let mut matched = false;
                let mut i = 0;
                while i < class.len() {
                    if i + 2 < class.len() && class[i + 1] == '-' {
                        if *first >= class[i] && *first <= class[i + 2] {
                            matched = true;
                        }
                        i += 3;
                    } else {
                        if *first == class[i] {
                            matched = true;
                        }
                        i += 1;
                    }
                }
                if matched != negate {
                    inner(&p[end + 1..], &n[1..])
                } else {
                    false
                }
            }
            Some(c) => !n.is_empty() && n[0] == *c && inner(&p[1..], &n[1..]),
        }
    }
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    inner(&p, &n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_media_name_guards_traversal() {
        assert_eq!(safe_media_name("../../etc/passwd"), "passwd");
        assert_eq!(safe_media_name("a\\b\\c.png"), "c.png");
        assert_eq!(safe_media_name(".."), "");
        assert_eq!(safe_media_name("dir/"), "dir");
        assert_eq!(safe_media_name("plain.png"), "plain.png");
        // Whitespace-only (or whitespace-wrapped dots) is no name at all (#382).
        assert_eq!(safe_media_name("   "), "");
        assert_eq!(safe_media_name(" .. "), "");
        assert_eq!(safe_media_name("a/  "), "");
    }

    #[test]
    fn glob_matches_fnmatch_subset() {
        assert!(glob_match("*.png", "a.png"));
        assert!(!glob_match("*.png", "a.jpg"));
        assert!(glob_match("img-?.png", "img-1.png"));
        assert!(glob_match("img-[0-9].png", "img-7.png"));
        assert!(!glob_match("img-[!0-9].png", "img-7.png"));
        assert!(glob_match("*", "anything"));
    }
}
