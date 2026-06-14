//! S6-1 repro (preserved by lead; rev-S6 worktree reaped, Cargo.toml reverted).
//! Maintained write path: upsert_notes_wire returns Err when the embed tail
//! fails, even though the note was already committed (committed-but-errored
//! window). Violates CLAUDE.md "index update failure doesn't fail the tool call"
//! and diverges from best-effort siblings (collection_prune/migrate_note_type/metadata).
//! Place at native/shrike-kernel/tests/s6_repro.rs.
//! Needs kernel [dev-dependencies]: futures = {version="0.3",features=["executor"]}, shrike-ffi = {workspace=true}.
//! Run: cd native && CARGO_TARGET_DIR=$HOME/.cache/shrike-review-target/s6 cargo test -p shrike-kernel --test s6_repro -- --nocapture
//! Observed at fa54f8c: PASSED (characterizing — proves the window). For xfail handoff,
//! invert to assert Ok(committed results) so it fails today / XPASS when fixed.

use std::sync::Arc;

use futures::future::BoxFuture;
use shrike_ffi::{NativeError, NativeResult};
use shrike_kernel::{block_on, Embedder, Kernel};

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
fn upsert_wire_errors_even_though_the_note_was_written() {
    block_on(async {
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
            .upsert_notes_wire(notes, shrike_store_api::DuplicatePolicy::Error, false)
            .await;
        assert!(result.is_err(), "tail did NOT propagate (contract honoured)");

        let again: Vec<shrike_schemas::NoteInput> = serde_json::from_str(notes_json).unwrap();
        let second = kernel
            .upsert_notes_wire(again, shrike_store_api::DuplicatePolicy::Error, true)
            .await;
        let results = second.expect("dry-run validation should not error");
        let dup = results.iter().any(|r| {
            matches!(r, shrike_schemas::UpsertNoteResult::Error { reason, .. }
                if format!("{reason:?}").to_lowercase().contains("dup"))
        });
        assert!(dup, "first note committed despite Err — committed-but-errored window");

        kernel.close().await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    });
}
