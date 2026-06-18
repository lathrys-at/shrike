//! The no-driven-runtime guard: a kernel op with the driven runtime UNINSTALLED
//! fails through the callback, never aborts the process.
//!
//! A SEPARATE test binary on purpose: the runtime seam is process-global, so
//! this proof must run in its own process where `init_driven_runtime` was never
//! called (no host parked the driver threads). There is no lazy default — the
//! kernel's `handle()`/`block_on` panic when the runtime is absent — so
//! `shrike_open` (which `spawn_op`s the open) hits that panic. The FFI boundary
//! guard (`ffi_guard_completion`) catches it and reports ONE internal-error
//! completion: a setup error surfaces as a clean error envelope, not UB.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::time::Duration;

use shrike_cabi::{shrike_open, ShrikeHandle};

extern "C" fn collect(user_data: *mut c_void, result_json: *const c_char) {
    let tx = unsafe { Box::from_raw(user_data as *mut mpsc::Sender<String>) };
    let json = unsafe { CStr::from_ptr(result_json) }
        .to_str()
        .expect("UTF-8 JSON")
        .to_string();
    let _ = tx.send(json);
}

fn user_data() -> (*mut c_void, mpsc::Receiver<String>) {
    let (tx, rx) = mpsc::channel::<String>();
    (Box::into_raw(Box::new(tx)) as *mut c_void, rx)
}

#[test]
fn open_without_a_driven_runtime_is_an_error_envelope_not_a_panic() {
    // Deliberately NO shrike_runtime_init / drive threads: the runtime is not
    // installed.
    let dir = std::env::temp_dir().join(format!("shrike-cabi-no-rt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = CString::new(dir.join("c.anki2").to_string_lossy().into_owned()).unwrap();
    let cache = CString::new(dir.join("cache").to_string_lossy().into_owned()).unwrap();

    let (ud, rx) = user_data();
    unsafe { shrike_open(collection.as_ptr(), cache.as_ptr(), collect, ud) };

    // The open's spawn hits `handle()`, which panics without an installed
    // runtime; the FFI guard catches it and fires one internal-error completion.
    // (A bounded recv so a regression that hangs fails as a timeout, not a hang.)
    let envelope = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the open fired a completion (the FFI guard reports the panic)");
    let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
    assert_eq!(
        v.get("error")
            .and_then(|e| e.get("kind"))
            .and_then(|k| k.as_str()),
        Some("internal"),
        "a missing driven runtime surfaces as an internal-error envelope: {envelope}"
    );

    std::fs::remove_dir_all(&dir).ok();
    // Keep the type referenced so the import is load-bearing (the handle is never
    // produced — the op errored before returning one).
    let _: Option<*mut ShrikeHandle> = None;
}
