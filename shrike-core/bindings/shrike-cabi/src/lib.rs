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
//! - The threading model mirrors the server (`shrike-pyo3` + `driven_runtime.py`):
//!   shrike-cabi IS shrike-core, so it spawns NO threads — it EXPOSES the
//!   kernel's blocking drive entries and the HOST commits + joins the OS
//!   threads (the iOS app's lifecycle owns thread count/affinity/QoS).
//!   [`shrike_runtime_init`] installs the driven `current_thread` runtime
//!   ([`shrike_kernel::init_driven_runtime`]) ONLY — no threads. The host then
//!   spawns one thread in [`shrike_drive_io`] (owns + drives tokio's IO/timer
//!   drivers and the async executor), calls [`shrike_runtime_probe`] (the
//!   startup barrier — it confirms the IO thread owns the drivers before the
//!   rest park, since tokio gives driver ownership to the first `block_on`
//!   caller), then spawns one thread in [`shrike_drive_sync`] (the serialized
//!   collection / anki-sync thread) and N in [`shrike_drive_compute`] (CPU
//!   engine compute + blocking-fs leaves; N the host's choice — the "N >= 2"
//!   engine overlap). The IO driver is load-bearing: a `current_thread` runtime
//!   polls `spawn_op`'d tasks ONLY while a thread drives it, so without it the
//!   async ops below would never run and their callbacks would never fire.
//!   [`shrike_runtime_shutdown`] drains every in-flight op (so its callback
//!   fires) then closes the kernel's pool queues — which makes the host's drive
//!   entries return so the host JOINS its own threads. `shrike_runtime_init` is
//!   required before any op; the full contract is on its doc-comment.
//!
//!   Resuming the process after an iOS suspension re-uses the host's committed
//!   threads — no re-init unless the host called [`shrike_runtime_shutdown`].
//!   The fuller foreground/background pause (quiescing the drivers on
//!   backgrounding) is #393, a later slice.
//! - [`shrike_open`] / [`shrike_close`] manage a kernel handle (an opaque
//!   `Arc<Kernel>` boxed behind a raw pointer).
//! - [`shrike_op`] dispatches one named action — its params a JSON string —
//!   by spawning the kernel future onto the runtime ([`shrike_kernel::spawn_op`])
//!   and invoking the C completion callback when it resolves. Dropping the
//!   work is never an option here: `spawn_op`'s detach-not-abort contract
//!   means the op always runs; the callback is the only way to observe it.
//! - [`shrike_attach_remote_embedder`] composes the remote embeddings engine
//!   (route 2 async-direct: `RemoteEmbedder` -> `AsyncWithPolicy`, no
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

// The driven model, mirroring the server (`shrike-pyo3` + `driven_runtime.py`):
// shrike-cabi IS shrike-core, so it spawns NO threads — it EXPOSES the kernel's
// blocking drive entries as a C ABI, and the HOST (the Swift/Kotlin app; the
// lifecycle tests, playing host) commits the OS threads and joins them.
//
// A `current_thread` tokio runtime has no worker threads: a spawned task makes
// progress ONLY while some thread is parked inside
// `tokio::runtime::Runtime::block_on` driving it (tokio's documented contract —
// `Handle::spawn`'d tasks are suspended once `block_on` returns). Every async
// C-ABI op here is fire-and-forget (`spawn_op` is a detached `handle().spawn`)
// with a C callback as its only observation point, so without a live IO driver
// the ops would never poll and their callbacks would never fire.
//
// So [`shrike_runtime_init`] installs the driven `current_thread` runtime ONLY
// (no threads); the host parks its committed threads via [`shrike_drive_io`]
// (×1), [`shrike_drive_sync`] (×1), and [`shrike_drive_compute`] (×N, N the
// host's choice). [`shrike_runtime_shutdown`] drains every in-flight op (so its
// callback fires) and then closes the kernel's pool queues; the drive entries
// return and the HOST joins its own threads.

/// How long [`shrike_runtime_shutdown`] waits for in-flight ops to drain before
/// it closes the pools regardless. A bounded drain mirrors the server's
/// committed-thread join timeout (`driven_runtime.py`): an unbounded wait would
/// let one hung op (e.g. a remote embed with no response) wedge teardown
/// forever, and on iOS the OS watchdog kills an unbounded wait anyway — so a
/// bounded drain that then proceeds is strictly safer than a host that can't
/// shut down. On timeout the remaining ops are stranded on the now-undriven
/// runtime (their callbacks won't fire), which is the lesser evil at teardown;
/// the host is expected to quiesce before shutdown for a clean drain.
#[cfg(feature = "anki-core")]
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Terminal "the runtime is shutting down / shut down" flag.
///
/// Set (under [`ADMIT_LOCK`]) at the head of [`shrike_runtime_shutdown`], before
/// the drain. Once set, [`admit_op`] refuses new ops, so they FAST-FAIL a
/// completion through the callback instead of being spawned onto a runtime that
/// is being torn down (an undriven `current_thread` runtime would queue a task
/// that is never polled — its callback would never fire and [`shrike_close`]'s
/// `rx.recv()` would block forever; a hang realistic on a racy iOS
/// suspend/teardown). Ops admitted BEFORE the store are counted in [`INFLIGHT`]
/// and the shutdown drains them to completion first.
///
/// It starts `false` — the state of the default multi-thread lane too (a
/// consumer that called neither [`shrike_runtime_init`] nor the drive entries,
/// whose lazy-default worker threads DO drive spawned tasks), so that lane and
/// every pre-shutdown op are wholly unaffected. It is never cleared in-process:
/// the kernel runtime + mode are `OnceLock`s, so re-driving after shutdown is a
/// no-op (the drive entries return at once on the already-tripped pools) and the
/// runtime stays terminally undriven — pinned by `reinit_after_shutdown`, which
/// asserts ops still fast-fail after a re-init, not that re-init "fails".
#[cfg(feature = "anki-core")]
static DRIVER_SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// In-flight op accounting for the drain-on-shutdown, paired with [`DRAIN_GATE`].
/// An op admitted by [`admit_op`] increments this; [`finish_op`] decrements it
/// AFTER the op's task has fired its callback (see `spawn_completion`). The
/// shutdown path waits for this to reach zero (bounded), so an op admitted
/// BEFORE shutdown still completes + fires its callback rather than being
/// stranded on the dying runtime.
#[cfg(feature = "anki-core")]
static INFLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The drain gate: a `std` condvar the teardown thread waits on for
/// `INFLIGHT == 0`. It is a `std` primitive ON PURPOSE — the teardown thread is
/// a C-ABI caller thread, never a runtime worker, so it must NOT `block_on` a
/// tokio `Notify` here (that would re-enter / contend with the IO driver's
/// `block_on` on the `current_thread` runtime — the one shape that hangs).
/// [`finish_op`] signals it from the op tail; the waiter loops on the predicate
/// under the mutex, so a wakeup that races the predicate isn't lost. The mutex
/// guards nothing but the condvar's wait/notify pairing — `INFLIGHT` stays the
/// source of truth for the lock-free `admit_op`/`finish_op` fast path, and the
/// waiter re-reads it under the lock each loop.
#[cfg(feature = "anki-core")]
static DRAIN_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(feature = "anki-core")]
static DRAINED: std::sync::Condvar = std::sync::Condvar::new();

/// Serializes the admission decision against shutdown. The op path takes
/// it to atomically "check the flag AND increment INFLIGHT"; shutdown takes it
/// to "set the flag" — so an op can never read the flag as `false` and then
/// increment AFTER shutdown has already snapshotted INFLIGHT and started
/// draining. (A brief, uncontended std mutex; never held across `.await`.)
#[cfg(feature = "anki-core")]
static ADMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Try to admit one op for dispatch onto the runtime. Returns `true` (and
/// counts the op in [`INFLIGHT`]) when the runtime is still driven; `false`
/// when the runtime has been shut down — the caller then fast-fails through the
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

/// Mark one admitted op finished — called from the op task's tail AFTER its
/// callback has fired (the `spawn_completion` ordering, which is what makes
/// `INFLIGHT == 0` mean "every callback has fired"). When this brings
/// [`INFLIGHT`] to zero it wakes the teardown thread's drain wait. The mutex is
/// taken only to publish the wakeup under the same lock the waiter loops under,
/// closing the lost-wakeup window (a bare decrement → notify could otherwise
/// slip between the waiter's predicate check and its `wait`).
#[cfg(feature = "anki-core")]
fn finish_op() {
    if INFLIGHT.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
        let _guard = DRAIN_GATE.lock().expect("drain gate poisoned");
        DRAINED.notify_all();
    }
}

/// Install the driven `current_thread` kernel runtime — install ONLY, no
/// threads (shrike-cabi is shrike-core; the HOST owns thread provisioning).
/// Returns whether the runtime is now driven: `true` on a fresh install (or a
/// benign re-call where driven was already installed), `false` if the lazy
/// default multi-thread runtime had already been pinned by an earlier op. The
/// caller MUST NOT park drive threads when this is `false` — they would have no
/// driven queues to consume and would return at once.
///
/// Mirrors `shrike-pyo3`'s `init_driven_runtime`: a second call finds the
/// runtime already set (the seam is set-once), so the first install stands and
/// this returns `is_driven()` either way. If never called, the kernel lazily
/// installs its DEFAULT multi-thread runtime on first use (whose worker threads
/// drive spawned tasks) — fine for a host that doesn't want the driven model.
///
/// # Host contract (the committed-thread lifecycle)
///
/// 1. `shrike_runtime_init()` — install the driven runtime.
/// 2. spawn one thread in [`shrike_drive_io`].
/// 3. [`shrike_runtime_probe`] — block until that IO thread is driving (it must
///    own the IO/timer drivers before the others park — see the probe's doc).
/// 4. spawn one thread in [`shrike_drive_sync`] and N in [`shrike_drive_compute`]
///    (N is the host's choice — the source of the "N >= 2" engine overlap).
/// 5. open / run ops / close via the rest of this ABI.
/// 6. [`shrike_runtime_shutdown`] — drains in-flight ops + closes the pools.
/// 7. the host JOINS the threads it spawned (they have returned by now).
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
            // The seam is set-once: an already-installed runtime returns Err
            // carrying the runtime back, so the first install stands. `is_driven`
            // reports whether that install was the driven one (a prior lazy
            // default → false → the host must not park drive threads).
            let _ = shrike_kernel::init_driven_runtime(rt);
            shrike_kernel::is_driven()
        }
        #[cfg(not(feature = "anki-core"))]
        {
            false
        }
    })
}

/// Park the calling thread as the driven runtime's IO/timer driver until
/// [`shrike_runtime_shutdown`] closes the pools. The host spawns ONE thread
/// here. Returns `true` on a clean return, `false` on a misuse (the runtime is
/// not in driven mode — the host called this without a successful
/// [`shrike_runtime_init`]). The C-ABI analog of `shrike-pyo3`'s `drive_io`.
///
/// This thread must be the FIRST to enter `block_on` (the host enforces it via
/// [`shrike_runtime_probe`]): tokio gives IO/timer-driver ownership to the first
/// `block_on` caller, and only the owner drives timers + IO continuously. If a
/// `drive_sync`/`drive_compute` thread won that race instead, this thread would
/// hook in as a non-owner and timers/IO would advance only while that other leaf
/// is parked in `recv`, not while it runs a job — flaky starvation.
#[no_mangle]
pub extern "C" fn shrike_drive_io() -> bool {
    ffi_guard("shrike_drive_io", false, || {
        #[cfg(feature = "anki-core")]
        {
            shrike_kernel::drive_io_until_shutdown().is_ok()
        }
        #[cfg(not(feature = "anki-core"))]
        {
            false
        }
    })
}

/// Park the calling thread as the serialized collection / anki-sync execution
/// thread until the pools are shut down. The host spawns ONE thread here, AFTER
/// [`shrike_runtime_probe`] confirms the IO driver owns the runtime. Returns
/// `true` on a clean return, `false` on a misuse (not driven mode). The C-ABI
/// analog of `shrike-pyo3`'s `drive_sync`. This thread is never a runtime
/// context, so anki's own `block_on` is legal here by construction.
#[no_mangle]
pub extern "C" fn shrike_drive_sync() -> bool {
    ffi_guard("shrike_drive_sync", false, || {
        #[cfg(feature = "anki-core")]
        {
            shrike_kernel::drive_sync().is_ok()
        }
        #[cfg(not(feature = "anki-core"))]
        {
            false
        }
    })
}

/// Park the calling thread as one of the N CPU-compute workers until the pools
/// are shut down. The host spawns N threads here (N its choice — the "N >= 2"
/// engine-overlap property), AFTER [`shrike_runtime_probe`]. Returns `true` on a
/// clean return, `false` on a misuse (not driven mode). The C-ABI analog of
/// `shrike-pyo3`'s `drive_compute`.
#[no_mangle]
pub extern "C" fn shrike_drive_compute() -> bool {
    ffi_guard("shrike_drive_compute", false, || {
        #[cfg(feature = "anki-core")]
        {
            shrike_kernel::drive_compute().is_ok()
        }
        #[cfg(not(feature = "anki-core"))]
        {
            false
        }
    })
}

/// Block the calling thread until the [`shrike_drive_io`] thread is driving the
/// runtime — the host's startup barrier. Returns `true` once it is driving;
/// `false` on a misuse (not driven mode, or the runtime has been shut down).
///
/// # Preconditions
///
/// Call ONCE at startup, AFTER spawning the [`shrike_drive_io`] thread and
/// BEFORE spawning the `drive_sync`/`drive_compute` threads. Calling it before
/// the IO thread is spawned would block until that thread appears (the barrier's
/// job, but the host must actually spawn it); calling it after shutdown returns
/// `false` at once (the guard below).
///
/// Why it exists: tokio gives IO/timer-driver ownership to the FIRST thread to
/// enter the runtime's `block_on`. [`shrike_drive_io`] must win that race. The
/// probe schedules a TRIVIAL executor-only task (`spawn_op` + a std channel, the
/// `shrike_close` bridge shape) and blocks until it completes — and a spawned
/// task only completes once some thread is inside `block_on` driving the
/// executor. At probe time only the IO thread has been spawned, so the probe
/// completing PROVES the IO thread is in its `block_on` and owns the drivers;
/// the host then parks the rest behind that guarantee.
///
/// The task is deliberately trivial — NOT an open: an open needs the
/// `drive_sync` thread (the anki collection actor), which isn't parked yet, so
/// it would hang. This task touches only the executor the IO thread drives.
#[no_mangle]
pub extern "C" fn shrike_runtime_probe() -> bool {
    ffi_guard("shrike_runtime_probe", false, || {
        #[cfg(feature = "anki-core")]
        {
            // Short-circuit on a not-driven OR shut-down runtime BEFORE the
            // spawn+recv. Post-shutdown the runtime is undriven and never
            // dropped (a OnceLock), so the spawned task would never run and its
            // `tx` would never send NOR drop — `rx.recv()` would block FOREVER,
            // not error. This guard is what makes "false when shut down" honest
            // (the probe is a startup-only barrier, so this is the teardown-race
            // safety net, not a hot path).
            if !shrike_kernel::is_driven()
                || DRIVER_SHUTDOWN.load(std::sync::atomic::Ordering::Acquire)
            {
                return false;
            }
            // Schedule an executor-only task and block on its std-channel reply
            // (the shrike_close bridge: never Runtime::block_on from this C
            // thread). Once the IO thread is in its block_on the task runs and
            // sends, so this returns true.
            let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
            drop(shrike_kernel::spawn_op(async move {
                let _ = tx.send(());
                Ok(())
            }));
            rx.recv().is_ok()
        }
        #[cfg(not(feature = "anki-core"))]
        {
            false
        }
    })
}

/// Shut the driven runtime down (teardown). Sets the shutdown flag (new ops
/// fast-fail), DRAINS every already-admitted op so its callback fires (bounded
/// by `DRAIN_TIMEOUT`), then closes the kernel's pool queues — which makes the
/// host's [`shrike_drive_io`]/[`shrike_drive_sync`]/[`shrike_drive_compute`]
/// calls return so the host can join its threads. A no-op when the runtime was
/// never installed in driven mode, or on a second call (the flag is idempotent
/// and the pools are already closed).
///
/// **cabi drains; the host joins.** This is the deliberate asymmetry with
/// `shrike-pyo3`, which exposes a bare `drive_pools_shutdown` because the server
/// drains its ops via the asyncio bridge (`manager.close()` awaits every
/// `kernel.close()`) BEFORE closing the pools. cabi has no bridge-await: its ops
/// are detached fire-and-forget with a C callback as their only observation, so
/// the drain is folded INTO this one call and the drain-before-pools-close
/// ordering can't be split apart by the host (closing the pools first would
/// strand an in-flight op's callback — a `current_thread` runtime stops polling
/// spawned tasks the instant the IO driver's `block_on` returns).
///
/// **An op is never orphaned by a clean shutdown — sequential OR concurrent.**
/// An op issued strictly after this has returned fast-fails through the callback
/// (the flag is set; the runtime is provably undriven). An op issued
/// CONCURRENTLY resolves to one of two safe outcomes, never a hang: either it
/// was admitted before the flag store and the drain waits for it to complete (its
/// callback fires), or it reaches admission after the store and fast-fails
/// through the callback. The host should still drain-then-shutdown for
/// predictable results — a racy iOS suspend/teardown can no longer strand a
/// callback within the drain window, and a hung op past the window loses only
/// itself (teardown still completes).
#[no_mangle]
pub extern "C" fn shrike_runtime_shutdown() {
    ffi_guard("shrike_runtime_shutdown", (), || {
        #[cfg(feature = "anki-core")]
        {
            // 1. Set the flag under ADMIT_LOCK so it is atomic with respect to op
            //    admission. The ordering closes the concurrent-op race: an op
            //    issued while shutdown runs either
            //      (a) was admitted before this store → it's counted in INFLIGHT,
            //          so the drain below waits for it to complete + fire;
            //      (b) reaches `admit_op` after this store → it fast-fails through
            //          the callback (the runtime is being torn down).
            // Idempotent: a second shutdown re-sets an already-set flag (a no-op)
            // and drains a zero count, then shutdown_driven_pools no-ops on the
            // already-taken senders.
            {
                let _guard = ADMIT_LOCK.lock().expect("admit lock poisoned");
                DRIVER_SHUTDOWN.store(true, std::sync::atomic::Ordering::Release);
            }
            // 2. Drain in-flight ops WHILE the IO driver is still live — it keeps
            //    polling the spawned tasks, so each fires its callback and then
            //    `finish_op` decrements INFLIGHT. We wait off the runtime on the
            //    std condvar (this C thread must not block_on the runtime), looped
            //    on the predicate so a wakeup can't be lost, and BOUNDED so one
            //    hung op can't wedge teardown forever (it then strands on the
            //    undriven runtime — the lesser evil; see DRAIN_TIMEOUT).
            drain_inflight(DRAIN_TIMEOUT);
            // 3. NOW close the pool queues + trip the IO driver's shutdown. Done
            //    after the drain so the IO thread was live for every callback.
            //    This makes the host's drive_* calls return; the host joins them.
            shrike_kernel::shutdown_driven_pools();
        }
    })
}

/// Wait for [`INFLIGHT`] to reach zero, bounded by `timeout`. The std condvar
/// waiter loops on the predicate under [`DRAIN_GATE`] (lost-wakeup-safe), and
/// returns early if the deadline passes — the caller then proceeds with
/// teardown regardless (see [`DRAIN_TIMEOUT`]).
#[cfg(feature = "anki-core")]
fn drain_inflight(timeout: std::time::Duration) {
    let deadline = std::time::Instant::now() + timeout;
    let mut guard = DRAIN_GATE.lock().expect("drain gate poisoned");
    while INFLIGHT.load(std::sync::atomic::Ordering::Acquire) != 0 {
        let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
            tracing::warn!("shutdown drain timed out with in-flight ops; proceeding to teardown");
            return;
        };
        let (g, _timed_out) = DRAINED
            .wait_timeout(guard, remaining)
            .expect("drain gate poisoned");
        guard = g;
    }
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
        // Post-shutdown fast-path: admit the close like any op. If the runtime
        // is shut down, `admit_op` returns false and the spawn + std-channel
        // bridge below (which would block forever on rx.recv(), since the
        // spawned close is queued onto an undriven runtime and never runs) is
        // skipped. Free the handle without driving the close — the runtime is
        // dead and the process is tearing down, so the actor cannot drain
        // anyway; dropping the Box reclaims the memory and is the most we can
        // soundly do. When admitted, the close is counted in INFLIGHT so a
        // CONCURRENT shutdown DRAINS it (drives it to completion) rather than
        // stranding this thread on rx.recv() — the close path.
        if !admit_op() {
            drop(handle);
            return;
        }
        let kernel = Arc::clone(&handle.kernel);
        // Drive close to completion on the kernel runtime via the SPAWN +
        // std-channel bridge: spawn the close onto the runtime (`spawn_op` —
        // driven by the committed drive_io thread, or by the default
        // multi-thread workers if no driven runtime was installed) and block
        // THIS C thread on a plain std channel for the result. This never calls
        // `Runtime::block_on` from the C thread, so it can't contend with the
        // IO driver's `block_on` on the current_thread runtime — the one shape
        // that would otherwise hang. Dropping the returned future detaches our
        // observation; the spawned close (and its std-channel send) still runs
        // to completion (spawn_op's contract). `finish_op` runs in the task tail
        // (a drop guard, so a panic in close() can't leak the count) — pairing
        // the `admit_op` above, and decrementing AFTER `tx.send` so a concurrent
        // shutdown's drain waits for this close to finish.
        let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
        drop(shrike_kernel::spawn_op(async move {
            struct FinishGuard;
            impl Drop for FinishGuard {
                fn drop(&mut self) {
                    finish_op();
                }
            }
            // Declared first ⇒ dropped last: close + signal the C thread, THEN
            // finish_op decrements (so a concurrent shutdown's drain only sees
            // INFLIGHT reach zero after this close has fully completed).
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
/// `Arc<dyn Embedder>`. Route 2 async-direct: no `Blocking` adapter —
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
                // Route 2: the remote embedder is async-direct — it
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
///
/// **Invariant — the callback fires BEFORE `finish_op` decrements INFLIGHT.**
/// This is what makes `INFLIGHT == 0` mean "every admitted op's callback has
/// fired", which the shutdown drain (`drain_inflight`) relies on. `_finish` is
/// declared FIRST so it drops LAST (Rust drops locals in reverse declaration
/// order): `completion.fire(...)` runs, then `_finish` drops and `finish_op`
/// decrements. Reordering the two — or decrementing INFLIGHT inside the future
/// before the fire — would let the drain observe zero, trip
/// `shutdown_driven_pools`, and strand the last callback on the undriven
/// runtime (the exact regression `inflight_op_drains` pins).
#[cfg(feature = "anki-core")]
fn spawn_completion(
    completion: Completion,
    fut: impl std::future::Future<Output = NativeResult<String>> + Send + 'static,
) {
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
        // Declared first ⇒ dropped last: fire the callback, THEN finish_op
        // decrements (the invariant in this fn's doc).
        let _finish = FinishGuard;
        completion.fire(fut.await);
        Ok(())
    }));
}

#[cfg(test)]
mod tests;
