//! The re-init-after-shutdown regression guard.
//!
//! A SEPARATE test binary on purpose: the runtime seam (`init_runtime`) is
//! process-global and, after `shrike_runtime_shutdown`, terminally undriven —
//! so this proof must run in its own process, never sharing one with
//! `post_shutdown.rs` or `current_thread_driver.rs`. This file holds exactly
//! ONE test for that reason.
//!
//! What it pins (the load-bearing guarantee, not an impl detail): after a
//! shutdown, a re-init CANNOT resurrect the dead runtime. The kernel runtime +
//! mode are `OnceLock`s, so `init_driven_runtime` on the second call finds them
//! already set and installs nothing — and `shrike_runtime_init` returns
//! `is_driven()`, which is STILL `true` (the mode OnceLock is never reset). So
//! re-init returning `true` is NOT the guarantee; the guarantee is that
//! `DRIVER_SHUTDOWN` stays terminally `true` (never cleared in-process) →
//! `admit_op` returns false → a subsequent `shrike_op`/`shrike_close` STILL
//! fast-fails through the callback. Re-driving after shutdown is also a no-op
//! (the drive entries return at once on the already-tripped pools), so a host
//! cannot bring a dead runtime back. A regression that re-enabled dispatch on
//! re-init would flip this test (the op would hang, then trip the recv timeout).
//!
//! The test plays the host: install + park the committed threads, shut them
//! down + join, then re-init and prove ops still fast-fail.

mod host;

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::time::Duration;

use host::Host;
use shrike_cabi::{shrike_close, shrike_op, shrike_open, shrike_runtime_init, ShrikeHandle};

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
    // Play the host: install + park the committed threads.
    let host = Host::start();
    let dir = std::env::temp_dir().join(format!("shrike-cabi-reinit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("c.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();
    let handle = open(&collection, &cache);

    // Shut down + join: the runtime is now terminally undriven.
    host.shutdown();

    // Re-init: the kernel runtime + mode OnceLocks are already set → no new
    // install. `shrike_runtime_init` returns `is_driven()`, which stays `true`
    // (the mode OnceLock is never reset) — so the return value is NOT the
    // guarantee, and asserting `!reinit` would be wrong. The guarantee is that
    // dispatch is NOT re-enabled: DRIVER_SHUTDOWN stays terminally set, so ops
    // still fast-fail (asserted below). We pin the actual return so a future
    // change to it is a deliberate, visible decision.
    let reinit = shrike_runtime_init();
    assert!(
        reinit,
        "re-init returns is_driven() == true (the mode OnceLock is never reset); \
         this is NOT the guarantee — the guarantee is that ops still fast-fail"
    );

    // The load-bearing guarantee: a post-(shutdown+reinit) op STILL fast-fails —
    // dispatch was NOT re-enabled (it can't be, in-process; DRIVER_SHUTDOWN is
    // terminal). No hang. Re-driving would also be a no-op (the drive entries
    // return at once on the tripped pools), so we don't re-park — the runtime
    // stays dead and the op must fast-fail without any driver.
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
