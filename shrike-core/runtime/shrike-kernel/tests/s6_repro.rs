//! Regression test: the maintained write tail is BEST-EFFORT.
//!
//! A note committed to the collection must not have the WHOLE call fail because
//! a downstream embed/index/derived step erred (the committed-but-errored
//! window). A tail that `?`-propagated → MCP `isError` → the caller, told
//! a committed write "failed", retries → spurious Duplicate / double-write.
//!
//! Asserts the behaviour: the call returns `Ok` with the per-item results even
//! though the embedder is down, and a follow-up dry-run confirms the note WAS
//! committed (it now reports a duplicate). It pairs with the in-crate
//! `s6_best_effort_*` tests and the `s5_interleaved_*` / `s6_2_derived_*`
//! watermark tests.

use std::sync::Arc;

use futures::future::BoxFuture;
use shrike_error::{NativeError, NativeResult};
use shrike_kernel::runtime::testing;
use shrike_kernel::{Embedder, Kernel};

/// An embedder whose every embed fails — stands in for a transient backend
/// outage AFTER the collection write has already committed.
struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn embed(&self, _texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move { Err(NativeError::internal("embed backend down")) })
    }
    fn fingerprint(&self) -> Option<String> {
        Some("failing-embedder:v1".to_string())
    }
    fn dim(&self) -> Option<usize> {
        Some(64)
    }
}

fn temp_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "shrike-s6-repro-{}-{}",
        std::process::id(),
        C.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn upsert_wire_returns_ok_results_even_when_the_embed_tail_fails() {
    testing::run_with_sync(async {
        let dir = temp_dir();
        let kernel = Kernel::open(
            dir.join("c.anki2").to_str().unwrap(),
            dir.join("cache").to_str().unwrap(),
        )
        .await
        .unwrap();
        kernel.attach_embedder(Arc::new(FailingEmbedder), None);

        let notes_json = r#"[
            {"note_type": "Basic", "deck": "D",
             "fields": {"Front": "paris is the capital of france", "Back": "geo"}}
        ]"#;
        let notes: Vec<shrike_schemas::NoteInput> = serde_json::from_str(notes_json).unwrap();
        let result = kernel
            .upsert_notes_wire(notes, shrike_collection::DuplicatePolicy::Error, false)
            .await;

        // The tail is best-effort — the committed write returns
        // Ok with the per-item result, NOT an Err.
        let results = result.expect("committed write must return Ok despite a failed embed tail");
        assert_eq!(results.len(), 1, "one note in, one result out");
        assert!(
            matches!(results[0], shrike_schemas::UpsertNoteResult::Created { .. }),
            "the note was created (committed) — got {:?}",
            results[0]
        );

        // And it really WAS committed: a follow-up dry-run upsert of the same
        // first field reports a duplicate (Anki's collection-wide first-field
        // rule), which can only happen if the first call persisted the note.
        let again: Vec<shrike_schemas::NoteInput> = serde_json::from_str(notes_json).unwrap();
        let second = kernel
            .upsert_notes_wire(again, shrike_collection::DuplicatePolicy::Error, true)
            .await;
        let results = second.expect("dry-run validation should not error");
        let dup = results.iter().any(|r| {
            matches!(r, shrike_schemas::UpsertNoteResult::Error { reason, .. }
                if format!("{reason:?}").to_lowercase().contains("dup"))
        });
        assert!(
            dup,
            "the first note was committed (the best-effort tail did not roll it back)"
        );

        kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}
