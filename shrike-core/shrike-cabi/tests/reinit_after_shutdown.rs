//! The #597 re-init-after-shutdown regression guard.
//!
//! A SEPARATE test binary on purpose: the runtime seam (`init_runtime`) is
//! process-global and, after `shrike_runtime_shutdown`, terminally undriven —
//! so this proof must run in its own process, never sharing one with
//! `post_shutdown.rs` or `current_thread_driver.rs`. This file holds exactly
//! ONE test for that reason.
//!
//! What it pins (the ACTUAL `OnceLock` behaviour, vs. an overclaim): the kernel
//! runtime is a `OnceLock`, so a second `shrike_runtime_init` finds it already
//! set, returns `false`, installs NO new driver, and never reaches the
//! clear-on-init `store(DRIVER_SHUTDOWN, false)`. So a re-init after shutdown is
//! a no-op: the flag stays `true`, dispatch is NOT re-enabled in-process, and a
//! subsequent `shrike_op`/`shrike_close` STILL fast-fails through the callback
//! (the runtime stays terminally undriven — the documented contract). A
//! regression that made re-init silently re-enable a dead runtime would flip
//! this test (the op would hang, then trip the recv timeout).

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::time::Duration;

use shrike_cabi::{
    shrike_close, shrike_op, shrike_open, shrike_runtime_init, shrike_runtime_shutdown,
    ShrikeHandle,
};

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
fn reinit_after_shutdown_is_a_noop_and_ops_still_fast_fail() {
    assert!(
        shrike_runtime_init(),
        "first init installs the current_thread runtime"
    );
    let dir = std::env::temp_dir().join(format!("shrike-cabi-reinit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("c.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();
    let handle = open(&collection, &cache);

    shrike_runtime_shutdown();

    // Re-init: the kernel runtime OnceLock is already set → no new driver,
    // returns false, and the clear-on-init never runs.
    let reinit = shrike_runtime_init();
    assert!(
        !reinit,
        "re-init is a no-op false (the kernel runtime OnceLock is already set) — \
         the clear-on-init never runs and the runtime stays undriven"
    );

    // So a post-(shutdown+reinit) op STILL fast-fails — dispatch was NOT
    // re-enabled (it can't be, in-process). No hang.
    let (ud, rx) = user_data();
    let c_action = CString::new("collection_info").unwrap();
    let c_params = CString::new("{}").unwrap();
    unsafe { shrike_op(handle, c_action.as_ptr(), c_params.as_ptr(), collect, ud) };
    let raw = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("op fast-fails (no dispatch re-enabled), never hangs");
    let env: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        env.get("error")
            .and_then(|e| e.get("kind"))
            .and_then(|k| k.as_str()),
        Some("unavailable"),
        "still unavailable after a no-op re-init: {env}"
    );

    // Close still fast-frees (watchdog so a regression times out).
    let h = handle as usize;
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let closer = std::thread::spawn(move || {
        unsafe { shrike_close(h as *mut ShrikeHandle) };
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("close returns, never hangs");
    closer.join().expect("the closer thread completes");
    std::fs::remove_dir_all(dir).ok();
}
