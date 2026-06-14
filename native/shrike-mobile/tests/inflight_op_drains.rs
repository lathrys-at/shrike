//! The #637 drain gate (the "careful part"): an op already in flight when
//! shutdown begins must still COMPLETE, not be abandoned.
//!
//! A SEPARATE test binary (own process — the runtime seam is process-global and
//! terminally undriven after shutdown). This is the deterministic complement to
//! `op_racing_shutdown.rs`: `shrike_op` admits (increments INFLIGHT) and spawns
//! the op SYNCHRONOUSLY before it returns, so once `shrike_op` has returned the
//! op is provably in flight. Calling `shrike_runtime_shutdown` after that must
//! DRAIN it — drive it to completion and fire its callback with a SUCCESS
//! result — rather than exiting the driver and stranding it. A regression that
//! made the driver exit-on-notify (without draining) would surface the op as a
//! timeout (orphaned) or an `unavailable` fast-fail (abandoned), both failing
//! the success assertion here.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::time::Duration;

use shrike_mobile::{
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
fn an_inflight_op_completes_through_the_shutdown_drain() {
    assert!(
        shrike_runtime_init(),
        "first init installs the current_thread runtime"
    );
    let dir = std::env::temp_dir().join(format!("shrike-mobile-drain-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let collection = dir.join("c.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();
    let handle = open(&collection, &cache);
    let h = handle as usize;

    // Issue the op. `shrike_op` admits (INFLIGHT += 1) and spawns SYNCHRONOUSLY
    // before returning, so the op is provably in flight once this returns —
    // there is no window where shutdown could miss it.
    let (ud, rx) = user_data();
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

    // Now shut down. The driver MUST drain the in-flight op to completion
    // before exiting, so its callback fires with a SUCCESS result.
    shrike_runtime_shutdown();

    let raw = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the in-flight op MUST complete through the drain, not be abandoned/hang");
    let env: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert!(
        env.get("ok").is_some(),
        "an op admitted before shutdown is DRAINED to a successful completion, \
         not fast-failed or orphaned: {env}"
    );
    // The drained op really ran (collection_info on an empty collection).
    assert_eq!(
        env["ok"].get("note_count").and_then(|v| v.as_i64()),
        Some(0),
        "the drained op produced its real result: {env}"
    );

    // A post-shutdown op now fast-fails (the runtime is drained + undriven).
    let (ud2, rx2) = user_data();
    let c_action2 = CString::new("collection_info").unwrap();
    let c_params2 = CString::new("{}").unwrap();
    unsafe {
        shrike_op(
            h as *const ShrikeHandle,
            c_action2.as_ptr(),
            c_params2.as_ptr(),
            collect,
            ud2,
        )
    };
    let raw2 = rx2
        .recv_timeout(Duration::from_secs(10))
        .expect("a post-drain op fast-fails through the callback, not hang");
    let env2: serde_json::Value = serde_json::from_str(&raw2).unwrap();
    assert_eq!(
        env2.get("error")
            .and_then(|e| e.get("kind"))
            .and_then(|k| k.as_str()),
        Some("unavailable"),
        "after the drain the runtime is undriven; a new op fast-fails: {env2}"
    );

    // Close fast-frees (watchdog).
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let closer = std::thread::spawn(move || {
        unsafe { shrike_close(h as *mut ShrikeHandle) };
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("close returns, not hang");
    closer.join().expect("the closer thread completes");
    std::fs::remove_dir_all(dir).ok();
}
