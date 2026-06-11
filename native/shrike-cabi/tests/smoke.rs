//! The C-ABI smoke, Rust-side (#333): drive the surface exactly as a C host
//! would — opaque handle, C strings, JSON in/out, last-error, free — running
//! open → upsert → search → close against a temp collection with zero Python
//! and zero embedder (lexical-only — no engine attached, no ort linked).

use std::ffi::{CStr, CString};

use shrike_cabi::{
    shrike_index_status_json, shrike_kernel_close, shrike_kernel_open, shrike_last_error,
    shrike_search, shrike_string_free, shrike_upsert_notes_json,
};

fn take_string(ptr: *mut std::ffi::c_char) -> String {
    assert!(!ptr.is_null(), "unexpected null (error: {:?})", unsafe {
        shrike_last_error()
            .as_ref()
            .map(|p| CStr::from_ptr(p).to_string_lossy().into_owned())
    });
    let out = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { shrike_string_free(ptr) };
    out
}

#[test]
fn open_upsert_search_close_as_a_c_host() {
    let dir = std::env::temp_dir().join(format!("shrike-cabi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let col = CString::new(dir.join("c.anki2").to_str().unwrap()).unwrap();
    let cache = CString::new(dir.join("cache").to_str().unwrap()).unwrap();

    let kernel = unsafe { shrike_kernel_open(col.as_ptr(), cache.as_ptr()) };
    assert!(!kernel.is_null());

    let notes = CString::new(
        serde_json::json!([
            {"note_type": "Basic", "deck": "Default",
             "fields": {"Front": "the mitochondria powerhouse", "Back": "atp"}},
            {"note_type": "Basic", "deck": "Default",
             "fields": {"Front": "newton laws of motion", "Back": "mechanics"}}
        ])
        .to_string(),
    )
    .unwrap();
    let policy = CString::new("error").unwrap();
    let results = take_string(unsafe {
        shrike_upsert_notes_json(kernel, notes.as_ptr(), policy.as_ptr(), false)
    });
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&results).unwrap();
    assert_eq!(parsed.len(), 2);
    assert!(parsed.iter().all(|r| r["status"] == "created"));
    let mito = parsed[0]["id"].as_i64().unwrap();

    // Lexical search finds the literal hit (no embedder attached).
    let query = CString::new("mitochondria powerhouse").unwrap();
    let hits = take_string(unsafe { shrike_search(kernel, query.as_ptr(), 5) });
    let rows: Vec<serde_json::Value> = serde_json::from_str(&hits).unwrap();
    assert_eq!(rows[0]["note_id"].as_i64().unwrap(), mito);

    let status = take_string(unsafe { shrike_index_status_json(kernel) });
    assert!(status.contains("\"state\""));

    // Bad input surfaces through last_error, never a crash.
    let bad = CString::new("not json").unwrap();
    let null = unsafe { shrike_upsert_notes_json(kernel, bad.as_ptr(), policy.as_ptr(), false) };
    assert!(null.is_null());
    let err = unsafe { CStr::from_ptr(shrike_last_error()) }.to_string_lossy();
    assert!(err.contains("JSON"), "got: {err}");

    assert_eq!(unsafe { shrike_kernel_close(kernel) }, 0);
    std::fs::remove_dir_all(dir).ok();
}
