//! The C-ABI binding surface (#333, S4): a Swift/Kotlin/C host embeds the
//! kernel with zero CPython and zero reimplemented logic.
//!
//! **Verdict shape (recorded on #333):** a thin manual `extern "C"` + JSON
//! surface, not UniFFI — the kernel's action convention is already
//! strings-in/JSON-out (epic convention 6), so the binding carries no type
//! vocabulary worth generating; what a generator would add (object graphs,
//! callback interfaces) is exactly what the injected-runtime model keeps OUT
//! of the kernel.
//!
//! **Executor handoff, v1:** the calling thread IS the executor. Every call
//! drives the kernel's runtime-agnostic future to completion on the caller
//! (`futures::executor::block_on` over the conforming inline
//! [`MutexExecutor`]) — no thread owned by the kernel or this layer, exactly
//! the #308 contract. An async/poll-based handoff (the host pumping futures
//! on its own scheduler, as the asyncio bridge does) is the #226 evolution;
//! the surface here stays valid alongside it.
//!
//! **Memory contract:** every returned `char*` is allocated here and MUST be
//! released with [`shrike_string_free`]. On failure, functions return null
//! (or a non-zero code) and the message is retrievable via
//! [`shrike_last_error`] (thread-local).
//!
//! **Log sink:** the host injects a callback ([`shrike_set_log_callback`]);
//! kernel `tracing` events forward to it (level, target, message) — the
//! pyo3-log pattern generalized, tracing-only observability preserved.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;

use futures::executor::block_on;

use shrike_kernel::{Kernel, MutexExecutor};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: String) {
    let c = CString::new(message.replace('\0', " "))
        .unwrap_or_else(|_| CString::new("error message unrepresentable").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// The most recent error message on THIS thread, or null. Valid until the
/// next failing call on the same thread; do not free.
#[no_mangle]
pub extern "C" fn shrike_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr())
    })
}

/// Release a string returned by this library.
///
/// # Safety
/// `s` must be null or a pointer returned by this library (CString::into_raw)
/// that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn shrike_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// An open kernel, opaque to the host.
pub struct ShrikeKernel {
    kernel: Kernel,
}

/// # Safety
/// `s` must be a valid NUL-terminated C string (or null, which errors).
unsafe fn arg_str<'a>(s: *const c_char, name: &str) -> Result<&'a str, ()> {
    if s.is_null() {
        set_last_error(format!("{name} must not be null"));
        return Err(());
    }
    match unsafe { CStr::from_ptr(s) }.to_str() {
        Ok(v) => Ok(v),
        Err(_) => {
            set_last_error(format!("{name} must be valid UTF-8"));
            Err(())
        }
    }
}

fn out_string(s: String) -> *mut c_char {
    CString::new(s.replace('\0', " "))
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// Open (creating if needed) the collection at `collection_path`, with the
/// kernel's sidecars under `cache_dir`. Returns an owned handle (release with
/// [`shrike_kernel_close`]) or null on failure (see [`shrike_last_error`]).
///
/// The embedded v1 opens with no embedding service attached: every collection
/// op and lexical search works; semantic search lights up when a host-side
/// embedder registers (the #342 slot — a C-level registration is the #226
/// follow-up).
///
/// # Safety
/// Both arguments must be valid NUL-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn shrike_kernel_open(
    collection_path: *const c_char,
    cache_dir: *const c_char,
) -> *mut ShrikeKernel {
    clear_last_error();
    let Ok(path) = (unsafe { arg_str(collection_path, "collection_path") }) else {
        return std::ptr::null_mut();
    };
    let Ok(cache) = (unsafe { arg_str(cache_dir, "cache_dir") }) else {
        return std::ptr::null_mut();
    };
    // The calling thread is the executor (see the module docs): inline,
    // serialized, nothing owned.
    let executor = Arc::new(MutexExecutor::default());
    match block_on(Kernel::open(path, cache, executor, None)) {
        Ok(kernel) => Box::into_raw(Box::new(ShrikeKernel { kernel })),
        Err(e) => {
            set_last_error(e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// The wire-shaped bulk upsert (named fields, create AND update, dry_run):
/// per-item results JSON, kernel-maintained. Returns an owned JSON string
/// (free with [`shrike_string_free`]) or null on failure.
///
/// # Safety
/// `handle` must come from [`shrike_kernel_open`]; strings NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn shrike_upsert_notes_json(
    handle: *mut ShrikeKernel,
    notes_json: *const c_char,
    on_duplicate: *const c_char,
    dry_run: bool,
) -> *mut c_char {
    clear_last_error();
    let Some(h) = (unsafe { handle.as_ref() }) else {
        set_last_error("handle must not be null".into());
        return std::ptr::null_mut();
    };
    let Ok(notes) = (unsafe { arg_str(notes_json, "notes_json") }) else {
        return std::ptr::null_mut();
    };
    let Ok(policy) = (unsafe { arg_str(on_duplicate, "on_duplicate") }) else {
        return std::ptr::null_mut();
    };
    match block_on(
        h.kernel
            .upsert_notes_json(notes.to_string(), policy.to_string(), dry_run),
    ) {
        Ok(results) => out_string(results),
        Err(e) => {
            set_last_error(e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Fused search: a JSON array of `{note_id, score, signals: [[name, rank]]}`
/// rows. Returns an owned JSON string or null on failure.
///
/// # Safety
/// `handle` must come from [`shrike_kernel_open`]; `query` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn shrike_search(
    handle: *mut ShrikeKernel,
    query: *const c_char,
    top_k: usize,
) -> *mut c_char {
    clear_last_error();
    let Some(h) = (unsafe { handle.as_ref() }) else {
        set_last_error("handle must not be null".into());
        return std::ptr::null_mut();
    };
    let Ok(q) = (unsafe { arg_str(query, "query") }) else {
        return std::ptr::null_mut();
    };
    match block_on(h.kernel.search(q, top_k)) {
        Ok(hits) => {
            let rows: Vec<serde_json::Value> = hits
                .into_iter()
                .map(|hit| {
                    serde_json::json!({
                        "note_id": hit.note_id,
                        "score": hit.score,
                        "signals": hit.signals,
                    })
                })
                .collect();
            out_string(serde_json::Value::Array(rows).to_string())
        }
        Err(e) => {
            set_last_error(e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// The index status block as JSON (state/size/stamps/progress).
///
/// # Safety
/// `handle` must come from [`shrike_kernel_open`].
#[no_mangle]
pub unsafe extern "C" fn shrike_index_status_json(handle: *mut ShrikeKernel) -> *mut c_char {
    let Some(h) = (unsafe { handle.as_ref() }) else {
        set_last_error("handle must not be null".into());
        return std::ptr::null_mut();
    };
    out_string(h.kernel.index().status().to_string())
}

/// Close the collection and release the handle. Always frees the handle,
/// even when the close itself reports an error (returns non-zero then).
///
/// # Safety
/// `handle` must come from [`shrike_kernel_open`] and not be used after.
#[no_mangle]
pub unsafe extern "C" fn shrike_kernel_close(handle: *mut ShrikeKernel) -> i32 {
    clear_last_error();
    if handle.is_null() {
        return 0;
    }
    let owned = unsafe { Box::from_raw(handle) };
    let _ = owned.kernel.index().save();
    match block_on(owned.kernel.close()) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(e.to_string());
            1
        }
    }
}

// ── log sink ─────────────────────────────────────────────────────────────────

/// The host's log callback: (level 1=ERROR..5=TRACE, target, message), both
/// strings valid only for the duration of the call.
pub type ShrikeLogCallback = extern "C" fn(level: u8, target: *const c_char, msg: *const c_char);

/// A plain subscriber forwarding every event to the host callback — no
/// registry, no filtering layers; the host's logging system does the rest.
struct SinkSubscriber {
    callback: ShrikeLogCallback,
}

impl tracing::Subscriber for SinkSubscriber {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let mut message = String::new();
        let mut visitor = MessageVisitor(&mut message);
        event.record(&mut visitor);
        let level = match *event.metadata().level() {
            tracing::Level::ERROR => 1,
            tracing::Level::WARN => 2,
            tracing::Level::INFO => 3,
            tracing::Level::DEBUG => 4,
            tracing::Level::TRACE => 5,
        };
        let target = CString::new(event.metadata().target().replace('\0', " "))
            .unwrap_or_else(|_| CString::new("shrike").unwrap());
        let msg =
            CString::new(message.replace('\0', " ")).unwrap_or_else(|_| CString::new("").unwrap());
        (self.callback)(level, target.as_ptr(), msg.as_ptr());
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

struct MessageVisitor<'a>(&'a mut String);

impl tracing::field::Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        if field.name() == "message" {
            let _ = write!(self.0, "{value:?}");
        } else {
            if !self.0.is_empty() {
                self.0.push(' ');
            }
            let _ = write!(self.0, "{}={value:?}", field.name());
        }
    }
}

/// Install the host's log sink as the global tracing subscriber. Call at most
/// once, before `shrike_kernel_open`; returns non-zero if a subscriber is
/// already installed.
#[no_mangle]
pub extern "C" fn shrike_set_log_callback(callback: ShrikeLogCallback) -> i32 {
    let subscriber = SinkSubscriber { callback };
    match tracing::subscriber::set_global_default(subscriber) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}
