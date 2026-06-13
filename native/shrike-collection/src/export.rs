//! Collection/deck export to an Anki package (#71).
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
//! (`shrike_store_api`), shared with the kernel and any other store impl.

use shrike_ffi::{NativeError, NativeResult};
use shrike_store_api::{ExportOutcome, ExportRequest, ExportScope, PackageFormat};

use crate::CollectionCore;

impl CollectionCore {
    /// Export the collection (or a scope of it) to `req.out_path`. The caller
    /// has already gated the path (the host's path-safety check); this trusts
    /// it and performs the anki export. `.colpkg` rejects any non-whole scope —
    /// it is a whole-collection backup by definition.
    pub fn export_package(&self, req: &ExportRequest) -> NativeResult<ExportOutcome> {
        self.ensure_open()?;
        match req.format {
            PackageFormat::Colpkg => {
                if !matches!(req.scope, ExportScope::Whole) {
                    return Err(NativeError::invalid_input(
                        "a .colpkg is a whole-collection backup and cannot be scoped to a \
                         deck or notes — export a .apkg for a subset, or drop the scope",
                    ));
                }
                // anki's colpkg exporter reads the media folder unconditionally
                // (it walks it to build the package's media map), and errors if
                // it's absent. A real anki profile always has the folder; ours
                // is lazily created on first media write, so a collection that
                // never stored media has none yet. Create it (empty) so a media-
                // free collection still backs up — matching anki desktop, where
                // the folder always exists.
                std::fs::create_dir_all(&self.media_dir).map_err(|e| {
                    NativeError::internal(format!("ensure media dir for colpkg export: {e}"))
                })?;
                // A colpkg carries the whole collection; report its note total
                // for a uniform `note_count`. Read it BEFORE the export — anki's
                // `export_collection_package` CONSUMES the open collection
                // (`guard.take()`), leaving the backend with none, so a read
                // after would hit CollectionNotOpen.
                let note_count = self.adapter.search_notes("")?.len() as u32;
                self.adapter.export_collection_package(
                    &req.out_path,
                    req.with_media,
                    req.legacy,
                )?;
                // Re-open: the colpkg export took the collection out of the
                // backend, so the next op on this core must find it open again
                // (our cooperative-lock `released` flag is untouched by anki's
                // internal take, so `ensure_open` wouldn't re-acquire — reopen
                // explicitly).
                self.reopen()?;
                Ok(ExportOutcome {
                    note_count,
                    out_path: req.out_path.clone(),
                })
            }
            PackageFormat::Apkg => {
                let limit = self.resolve_export_limit(&req.scope)?;
                let note_count = self.adapter.export_anki_package(
                    &req.out_path,
                    req.with_scheduling,
                    req.with_media,
                    req.legacy,
                    limit,
                )?;
                Ok(ExportOutcome {
                    note_count,
                    out_path: req.out_path.clone(),
                })
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
