//! The post-shutdown fast-fail gate.
//!
//! A SEPARATE test binary on purpose: the runtime seam (`init_driven_runtime`) is
//! process-global AND, after `shrike_runtime_shutdown`, terminally undriven —
//! so this proof must run in its own process, never sharing one with the
//! `current_thread_driver` flow (which needs the driver alive for its whole
//! run). Cargo/Bazel run each `tests/*.rs` integration binary as its own
//! process, and this file holds exactly ONE test so nothing else in the
//! process trips over the terminal shutdown.
//!
//! What it pins: after `shrike_runtime_shutdown` the `current_thread` runtime
//! exists but is undriven (the host's drive threads have returned). Before the
//! fix, a later `shrike_op` spawned a task that was never polled, so its
//! completion callback NEVER fired, and `shrike_close` then blocked forever on
//! `rx.recv()` — a silent hang, realistic on a racy iOS suspend/teardown.
//! After the fix both FAST-FAIL: `shrike_op` reports an `unavailable` error
//! THROUGH THE CALLBACK, and `shrike_close` returns (frees the handle) instead
//! of blocking. This test would hang (then fail on the recv timeout) on a
//! regression that drops the post-shutdown guard. The test plays the host: it
//! shuts down (cabi drains + closes the pools) and joins its drive threads
//! BEFORE the post-shutdown assertions, so the runtime is provably undriven.

mod host;

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::time::Duration;

use host::Host;
use shrike_cabi::{shrike_close, shrike_op, shrike_open, ShrikeHandle};

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

fn open(collection: &str, cache: &str) -> *mut ShrikeHandle {
    let c_col = CString::new(collection).unwrap();
    let c_cache = CString::new(cache).unwrap();
    let (ud, rx) = user_data();
    unsafe { shrike_open(c_col.as_ptr(), c_cache.as_ptr(), collect, ud) };
    let raw = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("open's completion fires under the live driver");
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
fn post_shutdown_op_and_close_fast_fail_via_callback_not_hang() {
    // Play the host: install + park the committed threads (IO first, then probe,
    // then sync + compute).
    let host = Host::start();

    let dir = std::env::temp_dir().join(format!("shrike-cabi-postshutdown-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("c.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();

    // Open WHILE the driver is alive (open needs it to fire its completion).
    let handle = open(&collection, &cache);
    assert!(!handle.is_null());

    // Tear down AND join: cabi drains + closes the pools, the drive threads
    // return, the host joins them — the runtime is now provably undriven for the
    // post-shutdown assertions below.
    host.shutdown();

    // A post-shutdown op MUST fire its callback with an error (proving no
    // silent hang). The recv timeout is the test's teeth: pre-fix, the spawned
    // task is never polled and this recv would block forever.
    let (ud, rx) = user_data();
    let c_action = CString::new("collection_info").unwrap();
    let c_params = CString::new("{}").unwrap();
    unsafe { shrike_op(handle, c_action.as_ptr(), c_params.as_ptr(), collect, ud) };
    let raw = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("a post-shutdown op MUST fast-fail through the callback, not hang");
    let env: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        env.get("error")
            .and_then(|e| e.get("kind"))
            .and_then(|k| k.as_str()),
        Some("unavailable"),
        "a post-shutdown op is reported as an unavailable error: {env}"
    );
    assert!(
        env.get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .contains("shut down"),
        "the error explains the runtime is shut down: {env}"
    );

    // A post-shutdown close MUST return (free the handle) rather than block
    // forever on its spawn+recv bridge. Run it on a watchdog thread so a
    // regression (the hang) fails as a timeout instead of wedging the suite.
    let h = handle as usize;
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let closer = std::thread::spawn(move || {
        unsafe { shrike_close(h as *mut ShrikeHandle) };
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("a post-shutdown shrike_close MUST return, not hang on rx.recv()");
    closer.join().expect("the closer thread completes");

    std::fs::remove_dir_all(dir).ok();
}
