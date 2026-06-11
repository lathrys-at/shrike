//! The action core (#331, kernel inversion S2) — slice 1: the read surface.
//!
//! Each action is the *whole* tool body: parameter normalization, the
//! collection-core call, and validation into the Rust-canonical response type
//! (#330) — so the typed contract guards the internal wire, not just the MCP
//! edge. Python's `actions.py` shrinks to a binding per re-homed action:
//! typed signature (FastMCP's inputSchema source) + context assembly + the
//! completion-log fragment.
//!
//! Actions are synchronous over `&CollectionCore`: the transitional harness
//! invokes them on its collection worker thread through the shrike-py
//! per-action bindings (the same serialization every collection op rides);
//! the kernel's async layer (S3, #332) will drive the same bodies through
//! [`crate::SerializedCollection`]. No threading, no runtime assumption here
//! (#308/#310).

use serde::de::DeserializeOwned;

use shrike_collection::CollectionCore;
use shrike_ffi::{NativeError, NativeResult};
use shrike_schemas::{CollectionInfo, ListNotesResponse};

/// The actions this module has re-homed (the registry seam: the Python
/// binding asserts its forwarding list against this, so the two sides can't
/// drift silently).
pub const REHOMED_ACTIONS: &[&str] = &["collection_info", "list_notes", "collection_query"];

/// Parse a core-emitted JSON payload into its canonical response type.
///
/// A parse failure here is a *bug* (the core and the schema disagree), not
/// caller input — surfaced as an internal error with the type named.
fn validate<T: DeserializeOwned>(name: &str, json: &str) -> NativeResult<T> {
    serde_json::from_str(json).map_err(|e| {
        NativeError::internal(format!(
            "{name}: core payload does not match the schema: {e}"
        ))
    })
}

/// `collection_info` — sectioned collection structure/stats.
///
/// `include` mirrors the tool param (empty = summary, `"all"` expands);
/// `note_type_details` selects which note types carry their full definition.
pub fn collection_info(
    core: &CollectionCore,
    include: &[String],
    note_type_details: &[String],
) -> NativeResult<CollectionInfo> {
    let raw = core.collection_info(include, note_type_details)?;
    validate("CollectionInfo", &raw)
}

/// Structured filters for [`list_notes`]. `modified_since_epoch` is an
/// epoch-seconds cutoff — ISO-8601 parsing stays host-side (a deliberate
/// divergence recorded on the core's `list_notes`).
#[derive(Debug, Clone, Default)]
pub struct ListNotesParams {
    pub ids: Option<Vec<i64>>,
    pub deck: Option<String>,
    pub tags: Option<Vec<String>>,
    pub note_type: Option<String>,
    pub modified_since_epoch: Option<i64>,
    pub with_fields: bool,
    pub limit: usize,
}

/// `list_notes` — filter/retrieve notes (filters ANDed; at least one given,
/// enforced by the core as invalid input).
pub fn list_notes(
    core: &CollectionCore,
    params: &ListNotesParams,
) -> NativeResult<ListNotesResponse> {
    let raw = core.list_notes(
        params.ids.as_deref(),
        params.deck.as_deref(),
        params.tags.as_deref(),
        params.note_type.as_deref(),
        params.modified_since_epoch,
        params.with_fields,
        params.limit,
    )?;
    validate("ListNotesResponse", &raw)
}

/// `collection_query` — a raw Anki search expression (the read-only escape
/// hatch, #97). A malformed expression is invalid input (isolation marks
/// already stripped by the core's error decoding).
pub fn collection_query(
    core: &CollectionCore,
    query: &str,
    with_fields: bool,
    limit: usize,
) -> NativeResult<ListNotesResponse> {
    let raw = core.query(query, with_fields, limit)?;
    validate("ListNotesResponse", &raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_collection() -> (std::path::PathBuf, CollectionCore) {
        let dir = std::env::temp_dir().join(format!(
            "shrike-kernel-actions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.anki2");
        let core = CollectionCore::open(path.to_str().unwrap()).unwrap();
        (dir, core)
    }

    fn add_note(core: &CollectionCore, front: &str, back: &str) -> i64 {
        let req = serde_json::json!([
            {"note_type": "Basic", "deck": "D", "fields": {"Front": front, "Back": back}}
        ]);
        let out = core.upsert_notes(&req.to_string(), "allow", false).unwrap();
        let results: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(results[0]["status"], "created", "{results}");
        results[0]["id"].as_i64().unwrap()
    }

    #[test]
    fn collection_info_returns_typed_sections() {
        let (_dir, core) = temp_collection();
        add_note(&core, "Q", "A");
        let info = collection_info(&core, &["summary".into(), "decks".into()], &[]).unwrap();
        let summary = info.summary.expect("summary requested");
        assert_eq!(summary.notes, 1);
        assert!(info.decks.is_some());
        assert!(info.stats.is_none()); // not requested
        core.close().unwrap();
    }

    #[test]
    fn list_notes_filters_and_validates() {
        let (_dir, core) = temp_collection();
        let id = add_note(&core, "mitochondria", "powerhouse");
        add_note(&core, "momentum", "mass times velocity");
        let resp = list_notes(
            &core,
            &ListNotesParams {
                deck: Some("D".into()),
                with_fields: true,
                limit: 50,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(resp.total, 2);
        assert!(resp.notes.iter().any(|n| n.id == id));
        // with_fields=true → content present on every note.
        assert!(resp.notes.iter().all(|n| n.content.is_some()));
        core.close().unwrap();
    }

    #[test]
    fn list_notes_without_filters_is_invalid_input() {
        let (_dir, core) = temp_collection();
        let err = list_notes(
            &core,
            &ListNotesParams {
                limit: 50,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("input"),
            "{err:?}"
        );
        core.close().unwrap();
    }

    #[test]
    fn collection_query_runs_raw_expressions() {
        let (_dir, core) = temp_collection();
        add_note(&core, "the cell", "biology");
        let resp = collection_query(&core, "deck:D", false, 10).unwrap();
        assert_eq!(resp.total, 1);
        assert!(resp.notes[0].content.is_none()); // meta mode
        assert!(collection_query(&core, "prop:bogus(((", false, 10).is_err());
        core.close().unwrap();
    }

    #[test]
    fn rehomed_registry_names_the_slice() {
        assert_eq!(
            REHOMED_ACTIONS,
            &["collection_info", "list_notes", "collection_query"]
        );
    }
}
