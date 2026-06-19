//! End-to-end verification: the open -> upsert -> search -> delete ->
//! close flow driven through the C ABI from a Rust C-caller (mirroring the
//! kernel's `no_cpython_smoke` — same HashEmbedder shape, here reached via
//! the real `extern "C"` entry points), plus the `no_libpython`
//! link-property test that asserts no libpython references in the artifact.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc;
use std::sync::Arc;

use futures::future::BoxFuture;

use super::*;

// ── a Rust C-caller harness ─────────────────────────────────────────────────

/// The callback every C-ABI call gets: it receives a borrowed JSON string and
/// sends an owned copy back through the channel whose `Sender` is the
/// `user_data` pointer. One-shot per call (each op gets a fresh channel), so
/// the harness blocks until exactly that op's completion fires.
extern "C" fn collect(user_data: *mut c_void, result_json: *const c_char) {
    // SAFETY: every call below threads a `Box<Sender<String>>`'s raw pointer
    // as user_data; the callback fires exactly once per op, so reclaiming the
    // box here is correct.
    let tx = unsafe { Box::from_raw(user_data as *mut mpsc::Sender<String>) };
    let json = unsafe { CStr::from_ptr(result_json) }
        .to_str()
        .expect("the library only emits UTF-8 JSON")
        .to_string();
    tx.send(json).expect("the harness receiver is alive");
}

/// Leak a one-shot `Sender` as the `user_data` pointer the callback reclaims.
fn user_data() -> (*mut c_void, mpsc::Receiver<String>) {
    let (tx, rx) = mpsc::channel::<String>();
    let ptr = Box::into_raw(Box::new(tx)) as *mut c_void;
    (ptr, rx)
}

/// Drive `shrike_open` and return the handle pointer the callback yields. The
/// C-ABI ops `spawn_op` onto the runtime and complete via the callback, so the
/// kernel's shared driven fixture must be parking the driver threads — start it
/// here (idempotent, process-global) so every handle-opening test is driven.
fn open(collection: &str, cache: &str) -> *mut ShrikeHandle {
    shrike_kernel::runtime::testing::run_with_collection(async {});
    let c_col = CString::new(collection).unwrap();
    let c_cache = CString::new(cache).unwrap();
    let (ud, rx) = user_data();
    unsafe { shrike_open(c_col.as_ptr(), c_cache.as_ptr(), collect, ud) };
    let envelope = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("open completion fired (the driven runtime must drive the op)");
    let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
    let ptr_int: usize = v
        .get("ok")
        .and_then(|s| s.as_str())
        .unwrap_or_else(|| panic!("open errored: {envelope}"))
        .parse()
        .expect("the handle is a stringified pointer integer");
    ptr_int as *mut ShrikeHandle
}

/// Drive one `shrike_op` and return the parsed envelope.
fn op(handle: *const ShrikeHandle, action: &str, params: &str) -> serde_json::Value {
    let c_action = CString::new(action).unwrap();
    let c_params = CString::new(params).unwrap();
    let (ud, rx) = user_data();
    unsafe { shrike_op(handle, c_action.as_ptr(), c_params.as_ptr(), collect, ud) };
    let envelope = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("op completion fired (the driven runtime must drive the op)");
    serde_json::from_str(&envelope).unwrap()
}

/// Unwrap a success envelope's `ok` payload, panicking with the error on a
/// failure envelope.
fn ok(envelope: serde_json::Value) -> serde_json::Value {
    if let Some(err) = envelope.get("error") {
        panic!("op errored: {err}");
    }
    envelope
        .get("ok")
        .expect("a success envelope has ok")
        .clone()
}

// ── the HashEmbedder (the no_cpython_smoke shape, reused) ───────────────────

/// Deterministic embedder: a token-hash bag vector — similar texts share
/// tokens, no model/network/Python. The same shape the kernel's
/// `no_cpython_smoke` uses, so the semantic signal is meaningful in search.
struct HashEmbedder;

impl shrike_engine_api::Embedder for HashEmbedder {
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(async move {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0f32; 64];
                    for token in t.to_lowercase().split_whitespace() {
                        let mut h: u64 = 1469598103934665603;
                        for b in token.bytes() {
                            h ^= b as u64;
                            h = h.wrapping_mul(1099511628211);
                        }
                        v[(h % 64) as usize] += 1.0;
                    }
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                    v.iter().map(|x| x / norm).collect()
                })
                .collect())
        })
    }

    fn fingerprint(&self) -> Option<String> {
        Some("hash-embedder:v1".to_string())
    }

    fn dim(&self) -> Option<usize> {
        Some(64)
    }
}

fn temp_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "shrike-cabi-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The smoke: open -> upsert (semantic + lexical signals live) -> search ->
/// delete -> close, ENTIRELY through the C ABI, with zero CPython in the
/// process (the crate links no pyo3 — the layering check guarantees that
/// structurally; this proves the composition runs).
///
/// Driven by the kernel's shared test fixture (1 io + 1 sync + N compute): the
/// C-ABI ops `spawn_op` onto the runtime and block this thread on a callback, so
/// the committed driver threads must be parked. The HashEmbedder is attached to
/// the kernel behind the opaque handle — an in-crate-only reach (the public ABI
/// can't attach an embedder), which is why this smoke stays in-crate rather than
/// moving to an integration binary alongside the lexical-only driver proof.
#[test]
fn c_abi_open_upsert_search_delete_close() {
    let dir = temp_dir();
    let collection = dir.join("c.anki2").to_string_lossy().into_owned();
    let cache = dir.join("cache").to_string_lossy().into_owned();

    // `open` starts the kernel's driven fixture (parks the driver threads).
    let handle = open(&collection, &cache);
    assert!(!handle.is_null());

    // Attach the HashEmbedder directly to the kernel behind the handle, so
    // the `text` semantic signal contributes to search (the relay/remote
    // attach path is exercised by its own surface; here we want a live
    // embedder without a network).
    // SAFETY: `handle` is the live pointer from open, not yet closed.
    let kernel: Arc<Kernel> = unsafe { Arc::clone(&(*handle).kernel) };
    kernel.attach_embedder(Arc::new(HashEmbedder), None);
    shrike_kernel::runtime::testing::run_with_collection(async move {
        kernel.reindex_if_needed().await.unwrap()
    });

    // upsert: one note via the wire-shaped action.
    let notes = serde_json::json!({
        "notes": [{
            "note_type": "Basic",
            "deck": "Default",
            "fields": { "Front": "paris is the capital of france", "Back": "geo" }
        }],
        "on_duplicate": "error",
        "dry_run": false
    });
    let results = ok(op(handle, "upsert_notes", &notes.to_string()));
    let arr = results.as_array().expect("upsert returns a result array");
    assert_eq!(arr.len(), 1);
    let status = arr[0].get("status").and_then(|s| s.as_str());
    assert_eq!(status, Some("created"), "the note was created: {arr:?}");
    let nid = arr[0]
        .get("id")
        .and_then(|v| v.as_i64())
        .expect("a created note carries its id");

    // collection_info: one note now.
    let info = ok(op(handle, "collection_info", "{}"));
    assert_eq!(info.get("note_count").and_then(|v| v.as_i64()), Some(1));

    // search: the created note is the top hit, and a semantic (text) signal
    // contributes — the fused composition works end to end over the C ABI.
    let hits = ok(op(
        handle,
        "search",
        &serde_json::json!({ "query": "capital of france", "top_k": 5 }).to_string(),
    ));
    let hits = hits.as_array().expect("search returns a hit array");
    assert!(!hits.is_empty(), "search found the note");
    let top = &hits[0];
    assert_eq!(
        top[0].as_i64(),
        Some(nid),
        "the created note is the top hit"
    );
    let signals: Vec<String> = top[2]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s[0].as_str().unwrap().to_string())
        .collect();
    assert!(
        signals.iter().any(|s| s == "text"),
        "the semantic signal contributed: {signals:?}"
    );

    // delete: the maintained op returns {deleted, not_found}; the note
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

    // close: frees the handle (and drains the actor).
    unsafe { shrike_close(handle) };
    std::fs::remove_dir_all(dir).ok();
}

/// Misuse is reported through the callback, never a panic: a null handle, an
/// unknown action, and malformed params each yield an error envelope.
#[test]
fn c_abi_misuse_is_an_error_envelope_not_a_panic() {
    // Null handle.
    let env = op(std::ptr::null(), "search", "{}");
    assert_eq!(
        env.get("error")
            .and_then(|e| e.get("kind"))
            .and_then(|k| k.as_str()),
        Some("invalid_input")
    );

    // Unknown action against a real handle.
    let dir = temp_dir();
    let handle = open(
        &dir.join("c.anki2").to_string_lossy(),
        &dir.join("cache").to_string_lossy(),
    );
    let env = op(handle, "no_such_action", "{}");
    let kind = env
        .get("error")
        .and_then(|e| e.get("kind"))
        .and_then(|k| k.as_str());
    assert_eq!(kind, Some("invalid_input"));
    let msg = env
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or("");
    assert!(
        msg.contains("no_such_action"),
        "the error names the action: {msg}"
    );

    // Malformed params (search wants an object with a query).
    let env = op(handle, "search", "not json");
    assert_eq!(
        env.get("error")
            .and_then(|e| e.get("kind"))
            .and_then(|k| k.as_str()),
        Some("invalid_input")
    );

    unsafe { shrike_close(handle) };
    std::fs::remove_dir_all(dir).ok();
}

/// `shrike_close(null)` and `shrike_string_free(null)` are no-ops, not crashes.
#[test]
fn null_teardown_is_a_noop() {
    unsafe { shrike_close(std::ptr::null_mut()) };
    unsafe { shrike_string_free(std::ptr::null_mut()) };
}

/// A panic inside a guarded entry body is CAUGHT and reported as one
/// internal-error completion — it never unwinds across the FFI boundary (UB),
/// and the caller still gets exactly one callback. Exercises
/// `ffi_guard_completion` directly with a panicking body (the kernel ops don't
/// panic on bad input, so this is the way to prove the boundary).
#[test]
fn a_panic_in_a_guarded_body_becomes_one_error_completion() {
    let (ud, rx) = user_data();
    ffi_guard_completion("test", collect, ud, |_completion| {
        panic!("boom");
    });
    let envelope = rx.recv().expect("the panic was reported via the callback");
    let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
    assert_eq!(
        v.get("error")
            .and_then(|e| e.get("kind"))
            .and_then(|k| k.as_str()),
        Some("internal"),
        "a caught panic surfaces as an internal error: {envelope}"
    );
}

/// A guarded body that returns WITHOUT firing (neither `Now` nor `Spawned`
/// is impossible to express — the type forces a return — but a panic before
/// the return is the real "didn't fire" path, covered above). This test pins
/// the `Dispatch::Now` happy path fires exactly once (no double callback).
#[test]
fn a_now_dispatch_fires_exactly_once() {
    let (ud, rx) = user_data();
    ffi_guard_completion("test", collect, ud, |_completion| {
        super::Dispatch::Now(Ok("42".to_string()))
    });
    let envelope = rx.recv().expect("the Now outcome fired");
    let v: serde_json::Value = serde_json::from_str(&envelope).unwrap();
    assert_eq!(v.get("ok").and_then(|n| n.as_i64()), Some(42));
    // No second callback: the channel is empty (a double-fire would panic the
    // collect callback on the already-reclaimed Sender box).
    assert!(rx.try_recv().is_err(), "the completion fired exactly once");
}

// ── the no-libpython link property ──────────────────────────────────────────

/// Mirror of the kernel's `no_cpython_smoke` link guarantee, made a direct
/// artifact assertion ("the no-Python property
/// pinned by the link — no libpython anywhere in the artifact"). The test
/// binary STATICALLY links the whole `shrike-cabi` crate and its kernel
/// closure, so if any of it pulled in libpython, this binary would reference
/// it. We inspect the running test binary's symbols/load commands for any
/// Python reference and assert there is none.
///
/// The structural guarantee is the layering check (this crate is not in
/// `PYO3_ALLOWED`, so it cannot name pyo3); this test is the belt to that
/// suspenders — it would also catch a transitive crate that linked CPython by
/// some other route. It degrades to a skip only if no symbol tool is present
/// (never a false pass: absence of the tool is reported, not asserted-clean).
#[test]
fn no_libpython_in_the_artifact() {
    let exe = std::env::current_exe().expect("the test binary path");

    // Try the platform symbol/link inspectors in turn; the first that runs
    // governs. `nm` lists symbols (a Py_* import would show); `otool -L`
    // (macOS) and `ldd` (linux) list dynamic dependencies (a libpython load
    // command would show).
    let probes: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
        &[("nm", &["-u"]), ("otool", &["-L"])]
    } else {
        &[("nm", &["-D"]), ("ldd", &[])]
    };

    let mut inspected = false;
    for (tool, args) in probes {
        let mut cmd = std::process::Command::new(tool);
        cmd.args(*args).arg(&exe);
        let Ok(out) = cmd.output() else {
            continue; // tool absent — try the next
        };
        if !out.status.success() {
            continue;
        }
        inspected = true;
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
        .to_lowercase();
        // Any libpython load command, or any CPython API symbol, is a
        // boundary breach (pyo3 surfaces CPython API symbols too).
        assert!(
            !text.contains("libpython"),
            "{tool} found a libpython reference in the mobile artifact — the \
             no-CPython boundary is breached"
        );
        assert!(
            !text.contains("py_initialize") && !text.contains("pygilstate"),
            "{tool} found a CPython API symbol in the mobile artifact — the \
             no-CPython boundary is breached"
        );
        break;
    }

    if !inspected {
        // No tool ran — report it (the layering check still holds the
        // structural line; CI has nm/otool/ldd, so this is a local-only gap).
        eprintln!(
            "no_libpython: no symbol tool ({}) ran; relying on the layering \
             check for the structural guarantee",
            probes.iter().map(|(t, _)| *t).collect::<Vec<_>>().join("/")
        );
    }
}
