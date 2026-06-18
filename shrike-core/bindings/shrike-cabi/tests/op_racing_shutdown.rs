//! The concurrent-op race gate.
//!
//! A SEPARATE test binary on purpose: the runtime seam (`init_runtime`) is
//! process-global and terminally undriven after `shrike_runtime_shutdown`, so
//! this proof must run in its own process, never sharing one with the other
//! `current_thread`-mode tests. This file holds exactly ONE #[test] for that
//! reason.
//!
//! What it pins: an op issued CONCURRENTLY with `shrike_runtime_shutdown` must
//! always fire its callback within a bounded timeout — it either COMPLETES (it
//! was admitted before the shutdown flag store, so the drain waited for it) or
//! FAST-FAILS `unavailable` (it reached admission after the store) — but NEVER
//! orphans (callback never fires → hang). The strictly-after case is closed
//! elsewhere; this closes the admit↔store window it left open. Pre-fix this repro is
//! red-by-design (the op can be stranded on the dying runtime); post-fix it
//! passes deterministically. The test plays the host (committed drive threads).

mod host;

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::time::Duration;

use host::Host;
use shrike_cabi::{shrike_close, shrike_op, shrike_open, ShrikeHandle};

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

fn open(collection: &str, cache: &str) -> *mut ShrikeHandle {
    let c_col = CString::new(collection).unwrap();
    let c_cache = CString::new(cache).unwrap();
    let (ud, rx) = user_data();
    unsafe { shrike_open(c_col.as_ptr(), c_cache.as_ptr(), collect, ud) };
    let raw = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("open fires under the live driver");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ptr: usize = v
        .get("ok")
        .and_then(|s| s.as_str())
        .unwrap_or_else(|| panic!("open errored: {v}"))
        .parse()
        .expect("handle pointer integer");
    ptr as *mut ShrikeHandle
}

#[test]
fn op_racing_shutdown_always_fires_its_callback_bounded() {
    // Play the host: install + park the committed threads.
    let mut host = Host::start();
    let dir = std::env::temp_dir().join(format!("shrike-cabi-race-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("c.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();
    let handle = open(&collection, &cache);
    let h = handle as usize;

    // Fire an op from another thread at ~the same time shutdown runs on this
    // one. Build the raw user_data INSIDE the thread (a *mut c_void isn't Send).
    let (tx, rx) = mpsc::channel::<String>();
    let op_thread = std::thread::spawn(move || {
        let ud = Box::into_raw(Box::new(tx)) as *mut c_void;
        let c_action = CString::new("collection_info").unwrap();
        let c_params = CString::new("{}").unwrap();
        unsafe {
            shrike_op(
                h as *const ShrikeHandle,
                c_action.as_ptr(),
                c_params.as_ptr(),
                collect,
                ud,
            )
        };
    });
    // No sleep: maximise the chance the op lands inside the shutdown window.
    // shutdown_no_join so the drive threads stay live through the drain (joined
    // at the end); the drain itself is what waits for an admitted-before-store op.
    host.shutdown_no_join();
    op_thread
        .join()
        .expect("op thread returns (shrike_op itself never blocks)");

    // The callback MUST fire within the timeout — completed OR fast-failed,
    // never orphaned.
    let raw = rx.recv_timeout(Duration::from_secs(10)).expect(
        "an op racing shutdown must still fire its callback (complete or fast-fail), not hang",
    );
    let env: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ok = env.get("ok").is_some();
    let unavailable = env
        .get("error")
        .and_then(|e| e.get("kind"))
        .and_then(|k| k.as_str())
        == Some("unavailable");
    assert!(
        ok || unavailable,
        "op racing shutdown resolved to a SAFE outcome (ok or unavailable): {env}"
    );

    // Close fast-frees regardless (watchdog so a regression times out).
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let closer = std::thread::spawn(move || {
        unsafe { shrike_close(h as *mut ShrikeHandle) };
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("close returns, not hang");
    closer.join().expect("the closer thread completes");
    // Join the committed drive threads (they returned when the pools closed).
    host.join();
    std::fs::remove_dir_all(dir).ok();
}
