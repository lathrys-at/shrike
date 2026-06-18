//! Collection/deck export to an Anki package.
//!
//! The op layer over the two anki exporters the `adapter` wraps: the modern
//! `.apkg` exporter (`ExportAnkiPackage` — whole-collection, deck, or
//! note-scoped, with optional scheduling/media) and the `.colpkg`
//! whole-collection backup (`ExportCollectionPackage`). Scope resolution (a
//! deck reference → deck id, the existing `resolve_deck_ref` convention) lives
//! here, not in the adapter — the adapter stays a thin proto shim.
//!
//! Export is read-only on the collection's data but holds it for the whole
//! `with_col` write of the package file, so the kernel routes it through the
//! collection task-actor (it serializes against other ops on that collection,
//! exactly like a write — export is exclusive by nature).
//!
//! The request/scope/format/outcome types are the store contract's
//! (`crate::contract`), shared with the kernel and any other store impl.

use crate::contract::{ExportOutcome, ExportRequest, ExportScope, PackageFormat};
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};

use crate::CollectionCore;

impl CollectionCore {
    /// Export the collection (or a scope of it) to `req.out_path`. The caller
    /// has already gated the path (the host's path-safety check); this trusts
    /// it and performs the anki export. `.colpkg` rejects any non-whole scope —
    /// it is a whole-collection backup by definition.
    ///
    /// **Symlink-safe write.** anki's exporters write to the exact path
    /// handed in with create/truncate and NO `O_NOFOLLOW`, following a symlink
    /// at that path. On a shared-host export root another local user could
    /// redirect this operator-privileged write outside the root by planting a
    /// symlink — at the requested basename, OR (the subtler door) at the temp
    /// name if it were predictable. The host's parent-dir gate catches only a
    /// symlinked *parent*. So anki is never handed an attacker-influenceable
    /// path:
    ///
    /// 1. mkdtemp a **securely-random, exclusively-created** subdir inside the
    ///    target's parent (`tempfile` — random name + `O_EXCL`/mkdir semantics,
    ///    so a pre-planted entry fails the create; a dir can't be a symlink the
    ///    write follows). Same filesystem as the target → the rename below is
    ///    atomic.
    /// 2. Export the package to a fixed name *inside* that server-owned dir
    ///    (nothing there is attacker-controlled).
    /// 3. Atomically `rename` the package out onto the requested basename.
    ///    `rename` replaces the directory entry without following a symlink at
    ///    the target, so a planted `out.apkg` symlink is replaced by the real
    ///    file, not written through — and the swap is crash-safe.
    /// 4. Drop the temp dir.
    ///
    /// Both symlink doors (basename and temp name) are closed.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error if a `.colpkg` is given a non-whole
    /// scope, and any error from re-acquiring the collection, creating the
    /// temp dir, the anki export RPC, or the atomic rename onto `out_path`.
    pub fn export_package(&self, req: &ExportRequest) -> NativeResult<ExportOutcome> {
        self.ensure_open()?;
        if matches!(req.format, PackageFormat::Colpkg) && !matches!(req.scope, ExportScope::Whole) {
            return Err(NativeError::invalid_input(
                "a .colpkg is a whole-collection backup and cannot be scoped to a \
                 deck or notes — export a .apkg for a subset, or drop the scope",
            ));
        }
        let final_path = std::path::Path::new(&req.out_path);
        let parent = final_path.parent().ok_or_else(|| {
            NativeError::invalid_input(format!("export path has no parent dir: {:?}", req.out_path))
        })?;
        // The parent must exist (the host gate already required it). Create it
        // defensively so the mkdtemp below can't fail on a missing dir.
        std::fs::create_dir_all(parent).context(ErrorKind::Internal, "export parent dir")?;

        // (1) A securely-random, exclusively-created temp dir in the parent.
        // `tempfile` removes it on drop, so any error path cleans up too.
        let temp_dir = tempfile::Builder::new()
            .prefix(".shrike-export-")
            .tempdir_in(parent)
            .context(ErrorKind::Internal, "export temp dir")?;
        // (2) Export into a fixed name inside the server-owned dir.
        let temp_path = temp_dir.path().join("package");
        let temp_str = temp_path.to_str().ok_or_else(|| {
            NativeError::internal(format!(
                "non-UTF-8 export temp path: {}",
                temp_path.display()
            ))
        })?;
        let note_count = self.export_to(req, temp_str)?;
        // (3) Atomically move the package out onto the requested basename.
        std::fs::rename(&temp_path, final_path).map_err(|e| {
            NativeError::internal(format!("finalize export ({}): {e}", req.out_path))
        })?;
        // (4) temp_dir drops here, removing the now-empty server-owned dir.
        Ok(ExportOutcome {
            note_count,
            out_path: req.out_path.clone(),
        })
    }

    /// Run the anki export to `out` (already a server-controlled temp path),
    /// dispatching on format. Returns the exported note count.
    fn export_to(&self, req: &ExportRequest, out: &str) -> NativeResult<u32> {
        match req.format {
            PackageFormat::Colpkg => {
                // anki's colpkg exporter reads the media folder unconditionally
                // (it walks it to build the package's media map), and errors if
                // it's absent. A real anki profile always has the folder;
                // Shrike's is lazily created on first media write, so a
                // collection that never stored media has none yet. Create it
                // (empty) so a media-free collection still backs up — matching
                // anki desktop, where the folder always exists.
                std::fs::create_dir_all(&self.media_dir).map_err(|e| {
                    NativeError::internal(format!("ensure media dir for colpkg export: {e}"))
                })?;
                // A colpkg carries the whole collection; report its note total
                // for a uniform `note_count`. Read it BEFORE the export — anki's
                // `export_collection_package` CONSUMES the open collection
                // (`guard.take()`), leaving the backend with none, so a read
                // after would hit CollectionNotOpen.
                let note_count = self.adapter.search_notes("")?.len() as u32;
                self.adapter
                    .export_collection_package(out, req.with_media, req.legacy)?;
                // Re-open: the colpkg export took the collection out of the
                // backend, so the next op on this core must find it open again
                // (the cooperative-lock `released` flag is untouched by anki's
                // internal take, so `ensure_open` wouldn't re-acquire — reopen
                // explicitly).
                self.reopen()?;
                Ok(note_count)
            }
            PackageFormat::Apkg => {
                let limit = self.resolve_export_limit(&req.scope)?;
                self.adapter.export_anki_package(
                    out,
                    req.with_scheduling,
                    req.with_media,
                    req.legacy,
                    limit,
                )
            }
        }
    }

    /// Map an [`ExportScope`] to anki's `ExportLimit` oneof. A deck ref resolves
    /// to a deck id (the `resolve_deck_ref` → `deck_id_by_name` convention); an
    /// unknown deck or an empty note set is a clean input error rather than a
    /// silently-empty package.
    fn resolve_export_limit(
        &self,
        scope: &ExportScope,
    ) -> NativeResult<anki_proto::import_export::ExportLimit> {
        use anki_proto::import_export::{export_limit::Limit, ExportLimit};
        let limit = match scope {
            ExportScope::Whole => Limit::WholeCollection(anki_proto::generic::Empty {}),
            ExportScope::Deck(reference) => {
                let name = self.resolve_deck_ref(reference)?.ok_or_else(|| {
                    NativeError::invalid_input(format!("no deck matches {reference:?}"))
                })?;
                let deck_id = self
                    .adapter
                    .deck_id_by_name(&name)?
                    .ok_or_else(|| NativeError::invalid_input(format!("no deck named {name:?}")))?;
                Limit::DeckId(deck_id)
            }
            ExportScope::Notes(ids) => {
                if ids.is_empty() {
                    return Err(NativeError::invalid_input(
                        "note-scoped export needs at least one note id",
                    ));
                }
                Limit::NoteIds(anki_proto::notes::NoteIds {
                    note_ids: ids.clone(),
                })
            }
        };
        Ok(ExportLimit { limit: Some(limit) })
    }
}
