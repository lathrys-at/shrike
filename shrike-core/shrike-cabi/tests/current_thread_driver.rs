//! The #504 current_thread-mode acceptance gate (the joint-review fix).
//!
//! A SEPARATE test binary on purpose: the runtime seam (`init_runtime`) is
//! process-global, so this proof must not share a process with the in-crate
//! suite (which runs on the lazily-installed DEFAULT multi-thread runtime).
//! Here we call `shrike_runtime_init` FIRST, so this process owns the
//! `current_thread` runtime + its dedicated driver thread.
//!
//! What it pins: under the `current_thread` mode the C ABI advertises, the
//! full open -> upsert -> search -> delete -> close flow runs and the
//! completion callbacks ACTUALLY FIRE. Before the driver fix, a
//! `current_thread` runtime had no thread driving it (every op is a
//! fire-and-forget `Handle::spawn`), so a `shrike_op` task was never polled
//! and its callback never fired — the flow would hang. This test would hang
//! (then fail on the recv timeout) on a regression that drops the driver.
//!
//! Lexical-only (no embedder): the public C ABI can't attach an embedder
//! (the kernel handle is opaque), so search relies on the lexical signal —
//! exactly like the kernel's own `current_thread.rs` proof. The point here is
//! that the callbacks FIRE under the driven current_thread runtime, not the
//! semantic ranking.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::time::Duration;

use shrike_cabi::{
    shrike_close, shrike_op, shrike_open, shrike_runtime_init, shrike_runtime_shutdown,
    ShrikeHandle,
};

/// The callback: send the borrowed JSON back through the channel whose Sender
/// is `user_data`. Fires once per op.
extern "C" fn collect(user_data: *mut c_void, result_json: *const c_char) {
    let tx = unsafe { Box::from_raw(user_data as *mut mpsc::Sender<String>) };
    let json = unsafe { CStr::from_ptr(result_json) }
        .to_str()
        .expect("UTF-8 JSON")
        .to_string();
    tx.send(json).expect("the receiver is alive");
}

fn user_data() -> (*mut c_void, mpsc::Receiver<String>) {
    let (tx, rx) = mpsc::channel::<String>();
    (Box::into_raw(Box::new(tx)) as *mut c_void, rx)
}

/// Receive the completion with a TIMEOUT — so a runtime that never drives the
/// op (the bug this test guards) fails as a timeout, not an infinite hang.
fn recv_envelope(rx: &mpsc::Receiver<String>) -> serde_json::Value {
    let raw = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("the completion callback fired within 20s (a current_thread runtime with no driver would hang here)");
    serde_json::from_str(&raw).unwrap()
}

fn open(collection: &str, cache: &str) -> *mut ShrikeHandle {
    let c_col = CString::new(collection).unwrap();
    let c_cache = CString::new(cache).unwrap();
    let (ud, rx) = user_data();
    unsafe { shrike_open(c_col.as_ptr(), c_cache.as_ptr(), collect, ud) };
    let v = recv_envelope(&rx);
    let ptr: usize = v
        .get("ok")
        .and_then(|s| s.as_str())
        .unwrap_or_else(|| panic!("open errored: {v}"))
        .parse()
        .expect("handle pointer integer");
    ptr as *mut ShrikeHandle
}

fn op(handle: *const ShrikeHandle, action: &str, params: &str) -> serde_json::Value {
    let c_action = CString::new(action).unwrap();
    let c_params = CString::new(params).unwrap();
    let (ud, rx) = user_data();
    unsafe { shrike_op(handle, c_action.as_ptr(), c_params.as_ptr(), collect, ud) };
    recv_envelope(&rx)
}

fn ok(v: serde_json::Value) -> serde_json::Value {
    if let Some(e) = v.get("error") {
        panic!("op errored: {e}");
    }
    v.get("ok").expect("ok").clone()
}

#[test]
fn full_flow_fires_callbacks_under_the_current_thread_driver() {
    // FIRST runtime touch in this process: install current_thread + its driver.
    assert!(
        shrike_runtime_init(),
        "shrike_runtime_init must install the current_thread runtime first in this process"
    );
    // A second init is a no-op false (one runtime, one driver).
    assert!(!shrike_runtime_init(), "a second runtime_init is a no-op");

    let dir = std::env::temp_dir().join(format!("shrike-cabi-ct-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("c.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();

    // open — the callback must fire (the op is driven by the parked driver).
    let handle = open(&collection, &cache);
    assert!(!handle.is_null());

    // upsert one note (lexical-only; no embedder on the public ABI).
    let notes = serde_json::json!({
        "notes": [{
            "note_type": "Basic",
            "deck": "Default",
            "fields": { "Front": "single threaded mitochondria", "Back": "powerhouse" }
        }],
        "on_duplicate": "error",
        "dry_run": false
    });
    let results = ok(op(handle, "upsert_notes", &notes.to_string()));
    let arr = results.as_array().expect("result array");
    assert_eq!(arr.len(), 1);
    assert_eq!(
        arr[0].get("status").and_then(|s| s.as_str()),
        Some("created"),
        "the note was created under current_thread mode: {arr:?}"
    );
    let nid = arr[0].get("id").and_then(|v| v.as_i64()).expect("note id");

    // collection_info — one note.
    let info = ok(op(handle, "collection_info", "{}"));
    assert_eq!(info.get("note_count").and_then(|v| v.as_i64()), Some(1));

    // search — the lexical signal finds the note (the callback fires, which is
    // the property under test).
    let hits = ok(op(
        handle,
        "search",
        &serde_json::json!({ "query": "mitochondria", "top_k": 5 }).to_string(),
    ));
    let hits = hits.as_array().expect("hit array");
    assert!(!hits.is_empty(), "lexical search found the note");
    assert_eq!(hits[0][0].as_i64(), Some(nid), "the note is the top hit");

    // delete — the maintained op (#604) returns {deleted, not_found}; the note
    // leaves and the count returns to zero.
    let deleted = ok(op(
        handle,
        "delete_notes",
        &serde_json::json!({ "note_ids": [nid] }).to_string(),
    ));
    assert_eq!(deleted["deleted"], serde_json::json!([nid]));
    assert_eq!(deleted["not_found"], serde_json::json!([]));
    let info = ok(op(handle, "collection_info", "{}"));
    assert_eq!(info.get("note_count").and_then(|v| v.as_i64()), Some(0));

    // close — the spawn+std-channel bridge must complete under the driver
    // (it would hang if the driver weren't driving the spawned close).
    unsafe { shrike_close(handle) };

    // Teardown: stop the driver thread cleanly and join it.
    shrike_runtime_shutdown();
    // A second shutdown is a no-op.
    shrike_runtime_shutdown();

    std::fs::remove_dir_all(dir).ok();
}
