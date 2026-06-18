//! The Shrike C-ABI binding: the action exchange over a C ABI.
//!
//! A native Swift/Kotlin app embeds the kernel in-process and drives it
//! through these `extern "C"` functions — the shape the tokio pivot reserved
//! ("a future C/Swift layer adapts the action exchange with completion
//! callbacks").
//! **Zero CPython**: nothing here links libpython (pinned by the
//! `no_libpython` link-property test), so the no-Python-on-mobile property
//! is structural, not aspirational.
//!
//! ## Shape (mirrors `shrike-pyo3`'s `async_kernel.rs`, minus Python)
//!
//! - [`shrike_runtime_init`] installs a `current_thread` tokio runtime via
//!   [`shrike_kernel::init_runtime`] AND starts a dedicated driver thread
//!   parked in `Runtime::block_on` for the runtime's life (the suspension-aware
//!   single-async-thread mode the iOS lifecycle wants). The driver is
//!   load-bearing: a `current_thread` runtime polls `spawn_op`'d tasks ONLY
//!   while a thread drives it, so without it the async ops below would never
//!   run and their callbacks would never fire. [`shrike_runtime_shutdown`]
//!   stops + joins the driver at teardown (fuller suspension handling
//!   lands in a later slice).
//! - [`shrike_open`] / [`shrike_close`] manage a kernel handle (an opaque
//!   `Arc<Kernel>` boxed behind a raw pointer).
//! - [`shrike_op`] dispatches one named action — its params a JSON string —
//!   by spawning the kernel future onto the runtime ([`shrike_kernel::spawn_op`])
//!   and invoking the C completion callback when it resolves. Dropping the
//!   work is never an option here: `spawn_op`'s detach-not-abort contract
//!   means the op always runs; the callback is the only way to observe it.
//! - [`shrike_attach_remote_embedder`] composes the remote embeddings engine
//!   (route 2 async-direct, #721 S2: `RemoteEmbedder` -> `AsyncWithPolicy`, no
//!   `Blocking` adapter) into the kernel's `Arc<dyn Embedder>` slot, mirroring
//!   `native_embedder.rs::from_remote` minus the PyO3 wrappers.
//! - [`shrike_string_free`] returns ownership of a string this library
//!   allocated (the only strings the caller frees through this ABI are the
//!   ones it received from a callback, which are borrowed for the callback's
//!   duration and freed by this library — so in this slice the caller frees
//!   nothing it didn't allocate; the entry point exists for the typed-result
//!   surface a later slice adds).
//!
//! ## Marshaling rules
//!
//! As on the PyO3 side, only coarse, batched data crosses the C ABI: C strings
//! (UTF-8) and byte buffers, with structured payloads carried as a single JSON
//! string per call — never a live host object, callback handle, or per-item
//! crossing. Compute crates receive owned data; no host handle enters a worker
//! thread.
//!
//! ## Errors
//!
//! Every result the callback receives is a JSON object: `{"ok": <value>}` on
//! success, or `{"error": {"kind": "...", "message": "..."}}` on failure,
//! where `kind` is the [`shrike_error::ErrorKind`] discriminant string. (The
//! error's Rust-side `#[source]` chain stays native — only `kind`/`message`
//! cross.) A synchronous misuse (a null handle, a non-UTF-8 C string) is
//! reported the same way through the callback, never a panic across the FFI
//! boundary.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[cfg(feature = "anki-core")]
use std::sync::Arc;

#[cfg(feature = "anki-core")]
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};
#[cfg(feature = "anki-core")]
use shrike_kernel::Kernel;

/// Run the synchronous body of an `extern "C"` entry under [`catch_unwind`],
/// so a Rust panic NEVER unwinds across the FFI boundary (that is undefined
/// behavior). A caught panic is logged and absorbed; the function returns its
/// `on_panic` value. Async work spawned onto the kernel runtime is panic-safe
/// independently (tokio's task harness catches a panic at the task boundary,
/// where it also can't cross a C frame).
fn ffi_guard<R>(what: &str, on_panic: R, body: impl FnOnce() -> R) -> R {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            // The panic payload is not propagated (it could be any type); the
            // name localizes it. Tracing is best-effort and itself guarded by
            // catch_unwind's having already unwound to here.
            tracing::error!("panic caught at the {what} FFI boundary (absorbed)");
            on_panic
        }
    }
}

// ── the C completion callback ───────────────────────────────────────────────

/// The completion callback every async C-ABI entry takes. It receives the
/// caller's `user_data` verbatim and one **borrowed**, NUL-terminated JSON
/// string describing the outcome (`{"ok": …}` / `{"error": …}`). The pointer
/// is valid only for the duration of the call — the callee owns the buffer and
/// frees it after the callback returns, so a caller that needs the bytes
/// beyond the call must copy them (Swift `String(cString:)` / JNI `NewStringUTF`
/// both copy).
pub type ShrikeCallback = extern "C" fn(user_data: *mut c_void, result_json: *const c_char);

/// A `user_data` pointer ferried onto the runtime. Carrying a raw pointer
/// across the spawn boundary needs `Send`; the **caller** guarantees the
/// pointee outlives the op (the documented contract — Swift/Kotlin keep the
/// continuation/context alive until the callback fires), so this wrapper
/// asserts the `Send` the type system can't see.
#[cfg(feature = "anki-core")]
struct UserData(*mut c_void);
// SAFETY: the pointer is opaque to this library and only handed back to the
// caller's own callback on the runtime thread; the caller owns its lifetime
// and the contract requires it to outlive the in-flight op.
#[cfg(feature = "anki-core")]
unsafe impl Send for UserData {}

/// The callback as a `Send` payload (an `extern "C" fn` is a plain function
/// pointer, already `Copy`/`Send`, but pairing it with `UserData` keeps the
/// runtime closure's captures in one named place). `Copy` so the FFI guard
/// retains it to report a panic the body raised before handing it off.
#[cfg(feature = "anki-core")]
#[derive(Clone, Copy)]
struct Completion {
    callback: ShrikeCallback,
    user_data: UserData,
}

// `UserData` is a raw pointer wrapper; deriving Copy on Completion needs it.
#[cfg(feature = "anki-core")]
impl Clone for UserData {
    fn clone(&self) -> Self {
        *self
    }
}
#[cfg(feature = "anki-core")]
impl Copy for UserData {}

#[cfg(feature = "anki-core")]
impl Completion {
    /// Invoke the caller's callback with one outcome, borrowing a freshly
    /// allocated C string for the call and freeing it afterward. A JSON
    /// payload that somehow fails to hold a NUL (it can't — serde never emits
    /// an interior NUL) degrades to a fixed error string rather than panicking.
    fn fire(self, outcome: NativeResult<String>) {
        let json = outcome_json(outcome);
        let c = CString::new(json).unwrap_or_else(|_| {
            CString::new(r#"{"error":{"kind":"internal","message":"result held a NUL byte"}}"#)
                .expect("the static fallback has no NUL")
        });
        // The pointer is borrowed for the call only; `c` drops (frees) on
        // return, after the callback has copied what it needs.
        (self.callback)(self.user_data.0, c.as_ptr());
    }
}

/// What an async-shaped entry's guarded body resolves to: either it computed a
/// synchronous outcome the guard fires, or it spawned the op onto the runtime
/// (the spawned task fires the completion itself, so the guard must NOT fire
/// again). Making the hand-off explicit is what keeps "fired exactly once"
/// provable without a shared mutable flag.
#[cfg(feature = "anki-core")]
enum Dispatch {
    /// Fire this outcome synchronously (a validation error, or a sync op).
    Now(NativeResult<String>),
    /// The op was spawned; its task owns the completion and will fire it.
    Spawned,
}

/// Run an async-shaped `extern "C"` entry's body under [`catch_unwind`],
/// guaranteeing the caller's completion fires EXACTLY once. The body returns a
/// [`Dispatch`]: `Now(result)` fires here, `Spawned` defers to the runtime
/// task. A panic in the body is caught and reported as one internal-error
/// completion — so a panic never crosses the C boundary, and the caller always
/// gets exactly one callback. The `completion` is `Copy`, but only ONE of the
/// guard's fire (`Now`/panic) and the spawned task's fire ever runs, because
/// `Spawned` is the body's promise that it handed the op off.
///
/// The sync fire (the `Now`/panic arms) ALSO runs under `catch_unwind`: the
/// "no panic crosses the boundary" guarantee then holds structurally, not by
/// an invariant of `fire` — a panic during the fire (in principle only the C
/// callback, whose own `extern "C"` ABI contains it) is absorbed as a last
/// resort. The spawned tail's fire is panic-contained by tokio's task harness.
#[cfg(feature = "anki-core")]
fn ffi_guard_completion(
    what: &str,
    callback: ShrikeCallback,
    user_data: *mut c_void,
    body: impl FnOnce(Completion) -> Dispatch,
) {
    let completion = Completion {
        callback,
        user_data: UserData(user_data),
    };
    // Step 1: run the body under catch_unwind to decide WHAT to fire, WITHOUT
    // firing yet — so a body panic and a normal outcome flow to the same
    // single fire below. `None` = the body spawned the op (the task fires);
    // `Some(outcome)` = fire it here; a caught panic maps to `Some(internal)`.
    let to_fire: Option<NativeResult<String>> =
        match catch_unwind(AssertUnwindSafe(|| body(completion))) {
            Ok(Dispatch::Now(outcome)) => Some(outcome),
            Ok(Dispatch::Spawned) => None,
            Err(_) => {
                tracing::error!("panic caught at the {what} FFI boundary (reported via callback)");
                Some(Err(NativeError::internal(
                    "internal panic at the FFI boundary",
                )))
            }
        };
    // Step 2: fire AT MOST ONCE, under its own catch_unwind — so even a
    // fire-time panic (in principle only the C callback, whose own extern "C"
    // ABI contains it) can't cross this C frame, and there is no recovery
    // fire that could double-invoke the callback.
    if let Some(outcome) = to_fire {
        let _ = catch_unwind(AssertUnwindSafe(|| completion.fire(outcome)));
    }
}

/// Render an op outcome as the wire envelope the callback receives. The
/// success branch's `value` is **already JSON** (each action serializes its
/// own typed result), so it is spliced in raw, not re-encoded as a string.
#[cfg(feature = "anki-core")]
fn outcome_json(outcome: NativeResult<String>) -> String {
    match outcome {
        Ok(value) => format!(r#"{{"ok":{value}}}"#),
        Err(e) => {
            // serde_json escapes the message so a quote/newline in it can't
            // break the envelope.
            let msg = serde_json::to_string(&e.message)
                .unwrap_or_else(|_| "\"<unprintable>\"".to_string());
            format!(
                r#"{{"error":{{"kind":"{}","message":{}}}}}"#,
                e.kind().as_str(),
                msg
            )
        }
    }
}

// ── runtime ─────────────────────────────────────────────────────────────────

/// The `current_thread` runtime's DRIVER state.
///
/// A `current_thread` tokio runtime has no worker threads: a spawned task
/// makes progress ONLY while some thread is parked inside
/// [`tokio::runtime::Runtime::block_on`] driving it (tokio's documented
/// contract — `Handle::spawn`'d tasks are suspended once `block_on` returns,
/// and only `Runtime::block_on` drives the IO/timer drivers). Every async
/// C-ABI op here is fire-and-forget (`spawn_op` = `handle().spawn(...)` +
/// detach), so without a driver a `current_thread` runtime would never poll
/// them and the completion callback would NEVER fire.
///
/// So `shrike_runtime_init` installs a `current_thread` runtime AND starts one
/// dedicated OS thread parked in `shrike_kernel::block_on(park)` for the
/// runtime's whole life — the canonical "runtime on a dedicated thread, post
/// work via the Handle" mobile shape (one async-executing thread the host can
/// suspend/stop deterministically). `park` awaits this `Notify`; once
/// the runtime is driven, every `spawn_op` task — and `shrike_close`'s
/// spawned close — runs to completion. `shrike_runtime_shutdown` notifies and
/// joins the driver.
#[cfg(feature = "anki-core")]
struct Driver {
    shutdown: Arc<tokio::sync::Notify>,
    thread: std::thread::JoinHandle<()>,
}

#[cfg(feature = "anki-core")]
static DRIVER: std::sync::Mutex<Option<Driver>> = std::sync::Mutex::new(None);

/// Terminal "the dedicated driver is shutting down / shut down" flag.
///
/// Set (under [`ADMIT_LOCK`]) by [`shrike_runtime_shutdown`] **before** it wakes
/// the driver to drain. Once set, [`admit_op`] refuses new ops, so they
/// FAST-FAIL a completion through the callback instead of being dispatched onto
/// a runtime that is being torn down (the undriven `current_thread` runtime
/// would queue a task that is never polled — its callback would never fire and
/// [`shrike_close`]'s `rx.recv()` would block forever; a hang
/// realistic on a racy iOS suspend/teardown). Ops admitted BEFORE the store are
/// counted in [`INFLIGHT`] and the driver drains them to completion first.
///
/// It is distinct from "`DRIVER` is `None`": that is ALSO the pre-init state of
/// the default multi-thread lane, whose worker threads DO drive spawned tasks.
/// This flag is only ever true after a real driver shutdown, so the
/// multi-thread default lane (where `shrike_runtime_init` was never called) and
/// every pre-shutdown op are wholly unaffected.
#[cfg(feature = "anki-core")]
static DRIVER_SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// In-flight op accounting for the drain-on-shutdown. An op admitted by
/// [`admit_op`] increments this; [`finish_op`] decrements it when the op's task
/// (and its callback) has run. The driver's shutdown path drains until this is
/// zero, so an op that was admitted BEFORE shutdown still completes rather than
/// being stranded on the dying runtime.
#[cfg(feature = "anki-core")]
static INFLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Woken (via `notify_waiters`) each time [`finish_op`] brings [`INFLIGHT`] to
/// zero, so the driver's drain loop can re-check and exit.
#[cfg(feature = "anki-core")]
static DRAINED: std::sync::LazyLock<tokio::sync::Notify> =
    std::sync::LazyLock::new(tokio::sync::Notify::new);

/// Serializes the admission decision against shutdown. The op path takes
/// it to atomically "check the flag AND increment INFLIGHT"; shutdown takes it
/// to "set the flag" — so an op can never read the flag as `false` and then
/// increment AFTER shutdown has already snapshotted INFLIGHT and started
/// draining. (A brief, uncontended std mutex; never held across `.await`.)
#[cfg(feature = "anki-core")]
static ADMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Try to admit one op for dispatch onto the runtime. Returns `true` (and
/// counts the op in [`INFLIGHT`]) when the runtime is still driven; `false`
/// when the driver has been shut down — the caller then fast-fails through the
/// callback rather than spawning onto a dying/undriven runtime. The flag read +
/// the increment are under [`ADMIT_LOCK`], paired with the same lock in
/// [`shrike_runtime_shutdown`], so the admit↔shutdown race window is closed:
/// either this admits and the drain waits for the op, or it fast-fails.
#[cfg(feature = "anki-core")]
fn admit_op() -> bool {
    let _guard = ADMIT_LOCK.lock().expect("admit lock poisoned");
    if DRIVER_SHUTDOWN.load(std::sync::atomic::Ordering::Acquire) {
        return false;
    }
    INFLIGHT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    true
}

/// Mark one admitted op finished (after its task + callback ran). When this
/// brings [`INFLIGHT`] to zero it wakes the driver's drain loop.
#[cfg(feature = "anki-core")]
fn finish_op() {
    if INFLIGHT.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
        DRAINED.notify_waiters();
    }
}

/// Install a `current_thread` tokio runtime as the process kernel runtime AND
/// start its driver thread. Returns `true` if this call installed it, `false`
/// if a runtime was already installed (idempotent-friendly — a second call is
/// a no-op `false`, never a second driver).
///
/// If never called, the kernel lazily installs its DEFAULT multi-thread
/// runtime on first use (whose worker threads drive spawned tasks with no
/// dedicated driver) — fine for a host that doesn't want the single-thread
/// mode. A mobile host calls this once at startup for the suspension-aware
/// single-async-thread shape, and `shrike_runtime_shutdown` at teardown.
#[no_mangle]
pub extern "C" fn shrike_runtime_init() -> bool {
    ffi_guard("shrike_runtime_init", false, || {
        #[cfg(feature = "anki-core")]
        {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(_) => return false,
            };
            // Install first: a second init (or a prior lazy default) loses the
            // race and we must NOT start a driver for a runtime we don't own.
            if shrike_kernel::init_runtime(rt).is_err() {
                return false;
            }
            let shutdown = Arc::new(tokio::sync::Notify::new());
            let park = Arc::clone(&shutdown);
            // The dedicated driver: parked in Runtime::block_on (via the
            // kernel's block_on, which calls it on the installed runtime) for
            // the runtime's life, so every Handle::spawn'd op is driven.
            let thread = std::thread::Builder::new()
                .name("shrike-cabi-rt".to_string())
                .spawn(move || {
                    // Drive the runtime for its life. On the shutdown signal,
                    // DRAIN already-spawned ops before returning: a
                    // current_thread runtime only polls spawned tasks while a
                    // thread is inside block_on, so returning the instant we're
                    // notified would strand any op admitted just before shutdown
                    // (never polled, callback never fires). Instead, keep driving
                    // until INFLIGHT hits zero — every in-flight op runs to
                    // completion and fires its callback first. Ops admitted AFTER
                    // shutdown set the flag are fast-failed by `admit_op`, so they
                    // never enter INFLIGHT and can't keep the drain spinning.
                    shrike_kernel::block_on(async move {
                        park.notified().await;
                        loop {
                            // Register interest BEFORE the check so a
                            // concurrent finish_op's notify_waiters can't be
                            // lost between the load and the await.
                            let drained = DRAINED.notified();
                            if INFLIGHT.load(std::sync::atomic::Ordering::Acquire) == 0 {
                                break;
                            }
                            drained.await;
                        }
                    });
                })
                .expect("the driver thread spawns");
            *DRIVER.lock().expect("driver lock poisoned") = Some(Driver { shutdown, thread });
            // Clear the terminal shutdown flag for this freshly installed driver.
            // This is a DEFENSIVE no-op given the kernel runtime is a OnceLock: a
            // second `shrike_runtime_init` finds the runtime already set and
            // returns `false` BEFORE reaching here, so re-init does NOT re-enable
            // dispatch in-process — once shut down, the runtime stays terminally
            // undriven (the documented contract). The store only ever runs on the
            // FIRST successful init (where the flag is already false), and is
            // published after the driver so an op can't observe "not shut down"
            // before the driver exists.
            DRIVER_SHUTDOWN.store(false, std::sync::atomic::Ordering::Release);
            true
        }
        #[cfg(not(feature = "anki-core"))]
        {
            false
        }
    })
}

/// Stop the `current_thread` driver thread started by [`shrike_runtime_init`]
/// (teardown). Sets the shutdown flag, wakes the driver to DRAIN every
/// already-admitted op, and joins it; a no-op if no driver is running (the
/// default multi-thread mode, or already shut down). After this the
/// `current_thread` runtime is no longer driven.
///
/// **An op is never orphaned by shutdown — sequential OR concurrent.** An op
/// issued strictly after this has returned fast-fails through
/// the callback (the flag is set; the runtime is provably undriven). An op
/// issued CONCURRENTLY with this resolves to one of two safe outcomes, never a
/// hang: either it was admitted before the flag store and the driver drains it
/// to completion (its callback fires), or it reaches admission after the store
/// and fast-fails through the callback. The host should still drain-then-
/// shutdown for predictable results, but a racy iOS suspend/teardown can no
/// longer strand an op's callback.
#[no_mangle]
pub extern "C" fn shrike_runtime_shutdown() {
    ffi_guard("shrike_runtime_shutdown", (), || {
        #[cfg(feature = "anki-core")]
        if let Some(driver) = DRIVER.lock().expect("driver lock poisoned").take() {
            // Set the flag BEFORE notifying the driver, under ADMIT_LOCK
            // so it is atomic with respect to op admission. The ordering closes
            // the concurrent-op race: an op issued while shutdown runs either
            //   (a) was admitted before this store → it's counted in INFLIGHT,
            //       so the driver's drain loop waits for it to complete + fire;
            //   (b) reaches `admit_op` after this store → it fast-fails through
            //       the callback (the runtime is being torn down).
            // Neither outcome strands the op on a dying runtime.
            {
                let _guard = ADMIT_LOCK.lock().expect("admit lock poisoned");
                DRIVER_SHUTDOWN.store(true, std::sync::atomic::Ordering::Release);
            }
            // Wake the driver: it resolves `park`, then DRAINS every already-
            // admitted op (keeps driving until INFLIGHT == 0) before returning
            // from block_on and exiting. Join so teardown is deterministic and
            // the runtime is provably undriven once this returns.
            driver.shutdown.notify_one();
            let _ = driver.thread.join();
        }
    })
}

// ── open / close ────────────────────────────────────────────────────────────

/// An opaque kernel handle the caller holds across ops. It is an
/// `Arc<Kernel>` boxed behind a raw pointer; the caller treats it as opaque
/// and returns it to [`shrike_close`] exactly once.
#[cfg(feature = "anki-core")]
pub struct ShrikeHandle {
    kernel: Arc<Kernel>,
}

/// Open a collection + its sidecar stores under `cache_dir`, blocking on the
/// kernel runtime until the open completes, and invoke `callback` with an
/// opaque handle pointer encoded as `{"ok": <pointer-as-int>}` on success or
/// an error envelope on failure.
///
/// The handle is returned through the callback (not the return value) so the
/// open shares the one async completion shape every other op uses; the caller
/// reads the integer, casts it back to `*mut ShrikeHandle`, and passes it to
/// subsequent [`shrike_op`] / [`shrike_close`] calls.
///
/// # Safety
/// `collection_path` and `cache_dir` must be valid NUL-terminated C strings;
/// `callback` must be a non-null function pointer (a null `extern "C" fn` is
/// UB when invoked); `user_data` must outlive the in-flight open per the
/// callback contract.
#[cfg(feature = "anki-core")]
#[no_mangle]
pub unsafe extern "C" fn shrike_open(
    collection_path: *const c_char,
    cache_dir: *const c_char,
    callback: ShrikeCallback,
    user_data: *mut c_void,
) {
    // The synchronous prologue runs under catch_unwind; a caught panic (or a
    // non-firing path) reports one internal-error completion. The body returns
    // a Dispatch so "fired exactly once" is provable.
    ffi_guard_completion("shrike_open", callback, user_data, |completion| {
        let (collection, cache) = match (cstr(collection_path), cstr(cache_dir)) {
            (Ok(c), Ok(d)) => (c, d),
            _ => {
                return Dispatch::Now(Err(NativeError::invalid_input(
                    "collection_path and cache_dir must be valid UTF-8 C strings",
                )))
            }
        };
        // Admit the open like any other op: fast-fail if shut down,
        // otherwise count it so a concurrent shutdown drains it.
        if !admit_op() {
            return Dispatch::Now(Err(NativeError::unavailable(
                "the kernel runtime has been shut down; no further ops can run",
            )));
        }
        spawn_completion(completion, async move {
            let kernel = Kernel::open(&collection, &cache).await?;
            let handle = Box::new(ShrikeHandle {
                kernel: Arc::new(kernel),
            });
            // Hand the heap pointer back as a STRING-encoded integer (a quoted
            // JSON value): a 64-bit pointer exceeds JSON's safe-integer range,
            // so the caller round-trips it as a decimal string, not a number.
            let ptr = Box::into_raw(handle) as usize;
            to_json(&ptr.to_string())
        });
        Dispatch::Spawned
    });
}

/// Close a handle's collection (draining the actor) and free the handle.
/// Synchronous: closing is a teardown verb the host calls on its own thread
/// and waits for. Safe to pass a null pointer (a no-op). After this returns
/// the pointer is dangling — the caller must not reuse it.
///
/// # Safety
/// `handle` must be a pointer returned by [`shrike_open`] and not yet closed,
/// or null. No [`shrike_op`] / [`shrike_attach_remote_embedder`] call against
/// this handle may still have an unfired completion when `shrike_close` runs
/// — those ops borrow the handle's kernel, and close frees the handle. The
/// caller owns this drain-then-close ordering.
#[cfg(feature = "anki-core")]
#[no_mangle]
pub unsafe extern "C" fn shrike_close(handle: *mut ShrikeHandle) {
    ffi_guard("shrike_close", (), || {
        if handle.is_null() {
            return;
        }
        let handle = Box::from_raw(handle);
        // Post-shutdown fast-path: admit the close like any op. If
        // the driver is shut down, `admit_op` returns false and the spawn +
        // std-channel bridge below (which would block forever on rx.recv(),
        // since the spawned close is queued onto an undriven runtime and never
        // runs) is skipped. Free the handle without driving the close — the
        // runtime is dead and the process is tearing down, so the actor cannot
        // drain anyway; dropping the Box reclaims the memory and is the most we
        // can soundly do. When admitted, the close is counted in INFLIGHT so a
        // CONCURRENT shutdown DRAINS it (drives it to completion) rather than
        // stranding this thread on rx.recv() — the close path.
        if !admit_op() {
            drop(handle);
            return;
        }
        let kernel = Arc::clone(&handle.kernel);
        // Drive close to completion on the kernel runtime via the SPAWN +
        // std-channel bridge: spawn the close onto the
        // runtime (`spawn_op` — driven by the parked current_thread driver, or
        // by the default multi-thread workers if no driver was installed) and
        // block THIS C thread on a plain std channel for the result. This
        // never calls `Runtime::block_on` from the C thread, so it can't
        // contend with the driver thread's `block_on` on a current_thread
        // runtime — the one shape that would otherwise hang. Dropping the
        // returned future detaches our observation; the spawned close (and its
        // std-channel send) still runs to completion (spawn_op's contract).
        // `finish_op` runs in the task tail (a drop guard, so a panic in
        // close() can't leak the count) — pairing the `admit_op` above.
        let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
        drop(shrike_kernel::spawn_op(async move {
            struct FinishGuard;
            impl Drop for FinishGuard {
                fn drop(&mut self) {
                    finish_op();
                }
            }
            let _finish = FinishGuard;
            let _ = kernel.close().await;
            let _ = tx.send(());
            Ok(())
        }));
        // Block until the close ran (its actor drained). The send only fails
        // if the spawned task was torn down without running — which spawn_op
        // never does — so a recv error just means "proceed with teardown".
        let _ = rx.recv();
        // `handle` (the Box) drops here, after close completed.
    });
}

// ── the op exchange ─────────────────────────────────────────────────────────

/// Dispatch one named action against an open handle. `params_json` is the
/// action's argument object (its shape is per-action); the result is delivered
/// through `callback` as the JSON envelope.
///
/// Actions in this slice (the open -> upsert -> search -> close smoke):
/// - `upsert_notes` — `{"notes": [...], "on_duplicate": "error"|..., "dry_run": bool}`;
///   result is the `Vec<UpsertNoteResult>` JSON.
/// - `search` — `{"query": "...", "top_k": N}`; result is `[[note_id, score, [[signal, rank], ...]], ...]`.
/// - `delete_notes` — `{"note_ids": [...]}`; result is `{"deleted": [...],
///   "not_found": [...]}` (the maintained single-op kernel delete).
/// - `collection_info` — `{}`; result is `{"note_count": N}` (the read the
///   smoke verifies; the full info surface grows in a later slice).
///
/// An unknown action or malformed params is an `invalid_input` error through
/// the callback, never a panic.
///
/// # Safety
/// - `handle` must be a live handle from [`shrike_open`] that is NOT
///   concurrently or subsequently passed to [`shrike_close`] until this op's
///   completion has fired — closing a handle with an op in flight is a
///   use-after-free (the op borrows the handle's kernel; `shrike_close` frees
///   the handle). The caller owns this ordering.
/// - `action` and `params_json` must be valid NUL-terminated C strings.
/// - `callback` must be a non-null function pointer (a null `extern "C" fn`
///   is undefined behavior when invoked and cannot be defended against in
///   Rust's type system).
/// - `user_data` must outlive the in-flight op (the completion contract).
#[cfg(feature = "anki-core")]
#[no_mangle]
pub unsafe extern "C" fn shrike_op(
    handle: *const ShrikeHandle,
    action: *const c_char,
    params_json: *const c_char,
    callback: ShrikeCallback,
    user_data: *mut c_void,
) {
    ffi_guard_completion("shrike_op", callback, user_data, |completion| {
        // Validate the synchronous preconditions FIRST (no admission needed for
        // a request that never reaches the runtime), so a bad-input op doesn't
        // perturb the in-flight count.
        if handle.is_null() {
            return Dispatch::Now(Err(NativeError::invalid_input("handle is null")));
        }
        // Borrow the Arc without taking ownership of the box (the caller still
        // owns it and closes it later).
        let kernel = Arc::clone(&(*handle).kernel);
        let (action, params) = match (cstr(action), cstr(params_json)) {
            (Ok(a), Ok(p)) => (a, p),
            _ => {
                return Dispatch::Now(Err(NativeError::invalid_input(
                    "action and params_json must be valid UTF-8 C strings",
                )))
            }
        };
        // Admit the op: fast-fail through the callback if the driver
        // is shut down (the undriven current_thread runtime would queue a task
        // that never runs); otherwise count it in INFLIGHT so a concurrent
        // shutdown DRAINS it instead of stranding it.
        if !admit_op() {
            return Dispatch::Now(Err(NativeError::unavailable(
                "the kernel runtime has been shut down; no further ops can run",
            )));
        }
        spawn_completion(completion, dispatch(kernel, action, params));
        Dispatch::Spawned
    });
}

/// One fused search hit on the wire: `(note_id, fused_score, [(signal, rank)])`
/// — the same tuple shape the Python binding's `search` returns.
#[cfg(feature = "anki-core")]
type SearchHitWire = (i64, f64, Vec<(String, i64)>);

/// Route a named action to its kernel op, parsing `params_json` into the op's
/// typed arguments and serializing the typed result back to JSON. Kept off the
/// `extern "C"` surface so the dispatch logic is plain, testable Rust.
#[cfg(feature = "anki-core")]
async fn dispatch(kernel: Arc<Kernel>, action: String, params: String) -> NativeResult<String> {
    match action.as_str() {
        "upsert_notes" => {
            #[derive(serde::Deserialize)]
            struct Args {
                notes: Vec<shrike_schemas::NoteInput>,
                #[serde(default = "default_on_duplicate")]
                on_duplicate: String,
                #[serde(default)]
                dry_run: bool,
            }
            let args: Args = parse(&params)?;
            let policy = shrike_collection::DuplicatePolicy::parse(&args.on_duplicate)?;
            let results = kernel
                .upsert_notes_wire(args.notes, policy, args.dry_run)
                .await?;
            to_json(&results)
        }
        "search" => {
            #[derive(serde::Deserialize)]
            struct Args {
                query: String,
                #[serde(default = "default_top_k")]
                top_k: usize,
            }
            let args: Args = parse(&params)?;
            let hits = kernel.search(&args.query, args.top_k).await?;
            // The same `(note_id, score, signals)` tuple shape the Python
            // binding's `search` returns.
            let wire: Vec<SearchHitWire> = hits
                .into_iter()
                .map(|h| (h.note_id, h.score, h.signals))
                .collect();
            to_json(&wire)
        }
        "delete_notes" => {
            #[derive(serde::Deserialize)]
            struct Args {
                note_ids: Vec<i64>,
            }
            let args: Args = parse(&params)?;
            // The maintained kernel op returns {deleted, not_found} in
            // its single write job — the same shape the MCP `delete_notes`
            // action serves. Serialize it through as the result JSON.
            let response = kernel.delete_notes(args.note_ids).await?;
            to_json(&response)
        }
        "collection_info" => {
            // The read the smoke verifies: a scoped `note_count` (a COUNT
            // query, not a materialize-all-ids scan — "read only what the
            // op needs"). The full collection_info surface (note types, decks,
            // tags, stats) lands in a later slice.
            let count = kernel.collection().run(|core| core.note_count()).await??;
            to_json(&serde_json::json!({ "note_count": count }))
        }
        other => Err(NativeError::invalid_input(format!(
            "unknown action: {other}"
        ))),
    }
}

#[cfg(feature = "anki-core")]
fn default_on_duplicate() -> String {
    "error".to_string()
}

#[cfg(feature = "anki-core")]
fn default_top_k() -> usize {
    10
}

// ── the remote embedder slot ────────────────────────────────────────────────

/// Compose the remote-embeddings engine into one of the kernel's embed SPACES
/// — the relay-offload path (a desktop/DIY kernel over the relay) or any
/// OpenAI-compatible cloud endpoint. Mirrors `native_embedder.rs::from_remote`
/// minus the PyO3 wrappers: `RemoteEmbedder` -> `AsyncWithPolicy`
/// (host-assembled fingerprint/dim + the `safe_batch` text chunking) ->
/// `Arc<dyn Embedder>`. Route 2 async-direct (#721 S2): no `Blocking` adapter —
/// the kernel awaits the engine's reqwest IO on its runtime.
///
/// The embed slot is an ORDERED SET of spaces: a Swift/Kotlin caller
/// that wants a dedicated text space PLUS a separate platform vision space
/// calls this **once per space** (the resolved ABI decision — one attach per
/// space, not a batched attach). `space_key` is the space's CONTENT identity
/// (reorder-stable); two distinct keys are two distinct spaces, re-using
/// a key REPLACES that space in place. When `space_key` is null the kernel keys
/// off `fingerprint` (and, failing that, the endpoint's own identity), so a
/// single-space mobile host attaches exactly as before.
///
/// `base_url` is required; `api_key`, `model`, `fingerprint`, and `space_key`
/// may be null. `dim` of 0 means "unknown" (the engine probes). `safe_batch` of
/// 0 is treated as 1 (serial). Returns the error envelope through `callback`;
/// on success the callback receives `{"ok": null}`.
///
/// # Safety
/// - `handle` must be a live handle from [`shrike_open`] not concurrently or
///   subsequently passed to [`shrike_close`] until this call returns (it
///   borrows the handle's kernel).
/// - The string pointers must be valid NUL-terminated C strings or null where
///   permitted.
/// - `callback` must be a non-null function pointer (a null `extern "C" fn`
///   is UB when invoked).
/// - `user_data` must outlive the call (the completion contract).
#[cfg(all(feature = "anki-core", feature = "engine-remote"))]
#[no_mangle]
pub unsafe extern "C" fn shrike_attach_remote_embedder(
    handle: *const ShrikeHandle,
    base_url: *const c_char,
    api_key: *const c_char,
    model: *const c_char,
    fingerprint: *const c_char,
    space_key: *const c_char,
    dim: usize,
    safe_batch: usize,
    callback: ShrikeCallback,
    user_data: *mut c_void,
) {
    ffi_guard_completion(
        "shrike_attach_remote_embedder",
        callback,
        user_data,
        |_completion| {
            if handle.is_null() {
                return Dispatch::Now(Err(NativeError::invalid_input("handle is null")));
            }
            let kernel = Arc::clone(&(*handle).kernel);

            let base_url = match cstr(base_url) {
                Ok(s) => s,
                Err(_) => {
                    return Dispatch::Now(Err(NativeError::invalid_input(
                        "base_url must be a valid UTF-8 C string",
                    )))
                }
            };
            let api_key = cstr_opt(api_key);
            let model = cstr_opt(model);
            let fingerprint = cstr_opt(fingerprint);
            let space_key = cstr_opt(space_key);

            let result = (|| -> NativeResult<String> {
                let engine = shrike_engine::remote::RemoteEmbedder::new(
                    shrike_engine::remote::RemoteEmbedderConfig {
                        base_url,
                        api_key,
                        model,
                    },
                )?;
                // Route 2: the remote embedder is async-direct (#721 S2) — it
                // implements the async `Embedder` trait, so it attaches WITHOUT
                // the `Blocking` adapter. `AsyncWithPolicy` carries the host
                // fingerprint/dim and chunks the text path by `safe_batch` (the
                // async sibling of `WithPolicy` + `Blocking`'s chunk loop).
                let embedder: Arc<dyn shrike_engine_api::Embedder> =
                    Arc::new(shrike_engine_api::AsyncWithPolicy::new(
                        Arc::new(engine),
                        fingerprint.clone(),
                        (dim != 0).then_some(dim),
                        safe_batch,
                    ));
                // The explicit space key wins; otherwise the tuned engine's
                // fingerprint (the same key `attach_embedder` would derive).
                let key = space_key.or(fingerprint);
                kernel.attach_embedder_space(key, embedder, None);
                Ok("null".to_string())
            })();
            Dispatch::Now(result)
        },
    );
}

// ── string ownership ────────────────────────────────────────────────────────

/// Free a string this library handed the caller to own. In this slice the
/// callback strings are borrowed (freed by the callee after the callback
/// returns), so the caller frees nothing it received — this entry exists for
/// the typed-result surface a later slice adds, and is a no-op-safe on null.
///
/// # Safety
/// `s` must be a pointer this library returned for the caller to own, or null.
#[no_mangle]
pub unsafe extern "C" fn shrike_string_free(s: *mut c_char) {
    ffi_guard("shrike_string_free", (), || {
        if !s.is_null() {
            drop(CString::from_raw(s));
        }
    });
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Borrow a C string as an owned Rust `String`, erroring on null/non-UTF-8.
#[cfg(feature = "anki-core")]
unsafe fn cstr(p: *const c_char) -> NativeResult<String> {
    if p.is_null() {
        return Err(NativeError::invalid_input("null C string"));
    }
    CStr::from_ptr(p)
        .to_str()
        .map(|s| s.to_string())
        .map_err(|_| NativeError::invalid_input("C string was not valid UTF-8"))
}

/// An optional C string: null or non-UTF-8 -> `None` (used for the embedder's
/// optional knobs, where absence is the meaningful state, not an error).
#[cfg(all(feature = "anki-core", feature = "engine-remote"))]
unsafe fn cstr_opt(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok().map(|s| s.to_string())
}

/// Parse params JSON into an action's typed args, mapping a shape mismatch to
/// `invalid_input` (it is the caller's request, not a bug).
#[cfg(feature = "anki-core")]
fn parse<T: serde::de::DeserializeOwned>(params: &str) -> NativeResult<T> {
    serde_json::from_str(params).context(ErrorKind::InvalidInput, "params JSON")
}

/// Serialize a typed result to JSON, mapping a (never-expected) failure to an
/// internal error.
#[cfg(feature = "anki-core")]
fn to_json<T: serde::Serialize>(value: &T) -> NativeResult<String> {
    serde_json::to_string(value).context(ErrorKind::Internal, "serialize result")
}

/// Spawn an op onto the kernel runtime and fire the completion when it
/// resolves — the one place the spawn+callback composition lives (the C-ABI
/// counterpart of `async_kernel.rs::kernel_op`). The callback fire is folded
/// INTO the spawned task (its tail), so the work and its one observation
/// point live in the same future: dropping the returned wrapper detaches the
/// caller's observation but the task — op AND callback — still runs to
/// completion (`spawn_op`'s detach-not-abort contract). The wrapper itself
/// resolves to `()` and is dropped; the callback is the real result channel.
///
/// The caller MUST have already counted this op via [`admit_op`]; the spawned
/// task calls [`finish_op`] in its tail (after the callback fires) so the
/// shutdown drain waits for it to complete. `finish_op` runs even if the
/// inner future or the callback panics — tokio's task harness unwinds the task,
/// but the count must not leak, so it's released via a drop guard.
#[cfg(feature = "anki-core")]
fn spawn_completion(
    completion: Completion,
    fut: impl std::future::Future<Output = NativeResult<String>> + Send + 'static,
) {
    // Decrement INFLIGHT on the way out no matter how the task ends (normal or
    // a panic in the future/callback), so a panicked op can't wedge the drain.
    struct FinishGuard;
    impl Drop for FinishGuard {
        fn drop(&mut self) {
            finish_op();
        }
    }
    // The op's result is captured and the callback fired from inside the
    // spawned task, so this never needs the (crate-private) runtime handle.
    // We drop the returned future immediately — detach, never abort.
    drop(shrike_kernel::spawn_op(async move {
        let _finish = FinishGuard;
        completion.fire(fut.await);
        Ok(())
    }));
}

#[cfg(test)]
mod tests;
