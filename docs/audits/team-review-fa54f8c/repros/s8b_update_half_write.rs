//! S8b-1 repro (preserved by lead; rev-S8b worktree reaped, reverted clean).
//! update_note_named commits fields BEFORE resolving the deck → a bad deck ref
//! half-writes the note, returns error. Cross-surface: in a multi-item batch,
//! advance_watermarks then pushes the index watermark past the half-write's
//! col_mod → next-boot drift never reconciles the stale vector (silent desync).
//! Insert into native/shrike-collection/src/lib.rs `mod tests` (reuses existing
//! upsert_json/temp_core/CreateOutcome/DuplicatePolicy helpers + serde_json).
//! Run: cd native && CARGO_TARGET_DIR=$HOME/.cache/shrike-review-target/s8b cargo test -p shrike-collection s8b_update_bad_deck_half_writes_fields -- --nocapture
//! Observed at fa54f8c: FAILED — fields = ["NEW front","NEW back"] (half-written) despite error result.

#[test]
fn s8b_update_bad_deck_half_writes_fields() {
    let (core, dir) = temp_core();
    let basic = core.notetype_id("Basic").unwrap();
    let CreateOutcome::Created(nid) = core
        .create_note(basic, DEFAULT_DECK,
            &["orig front".into(), "orig back".into()], &[], DuplicatePolicy::Error)
        .unwrap()
    else { panic!("create failed") };

    let upd = serde_json::json!([{
        "id": nid,
        "fields": {"Front": "NEW front", "Back": "NEW back"},
        "deck": "#999999999"
    }]);
    let results = upsert_json(&core, &upd.to_string(), "error", false);

    assert_eq!(results[0]["status"], "error", "item should report error");
    assert!(results[0]["error"].as_str().unwrap().contains("not found"));

    // PREDICTED DEFECT: fields already mutated despite the error result.
    let note = core.get_note(nid).unwrap();
    assert_eq!(note.fields,
        vec!["orig front".to_string(), "orig back".to_string()],
        "BUG: error result, but fields were half-written: {:?}", note.fields);

    core.close().unwrap();
    std::fs::remove_dir_all(dir).ok();
}
