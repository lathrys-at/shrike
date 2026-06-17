//! Shared plumbing for the Swift C ABI (macOS only): the one free function
//! and the guard type both shims (`imp.rs` OCR, `speech.rs` ASR) wrap
//! returned strings in.

use std::ffi::{c_char, CStr};

extern "C" {
    fn shrike_av_free_string(ptr: *mut c_char);
}

/// A Swift-allocated C string, freed on the Swift side's allocator when
/// dropped — never `libc::free` across the boundary.
pub(crate) struct SwiftString(*mut c_char);

impl SwiftString {
    /// Wrap a returned pointer; `None` for null (the Swift side's
    /// "unavailable" sentinel).
    pub(crate) fn wrap(ptr: *mut c_char) -> Option<Self> {
        (!ptr.is_null()).then_some(Self(ptr))
    }

    pub(crate) fn as_str(&self) -> &str {
        // The Swift side only ever emits UTF-8; a torn string degrades to
        // empty rather than panicking mid-batch.
        unsafe { CStr::from_ptr(self.0) }.to_str().unwrap_or("")
    }
}

impl Drop for SwiftString {
    fn drop(&mut self) {
        unsafe { shrike_av_free_string(self.0) }
    }
}

/// The wire shape from Swift: a `Recognition` plus an optional `error`
/// (a failed request — logged by the caller, the recognition flows on).
#[derive(serde::Deserialize)]
pub(crate) struct Wire {
    #[serde(default)]
    pub(crate) error: Option<String>,
    #[serde(flatten)]
    pub(crate) recognition: shrike_engine_api::Recognition,
}

/// Parse one wire payload, logging a carried `error`; unparseable JSON
/// degrades to the empty recognition — per-item failures never sink a
/// batch.
pub(crate) fn parse_wire(json: &SwiftString) -> shrike_engine_api::Recognition {
    match serde_json::from_str::<Wire>(json.as_str()) {
        Ok(wire) => {
            if let Some(error) = wire.error {
                tracing::warn!("recognition glue reported: {error}");
            }
            wire.recognition
        }
        Err(e) => {
            tracing::warn!("recognition glue returned unparseable JSON: {e}");
            crate::empty_recognition()
        }
    }
}
