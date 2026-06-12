//! The macOS implementation: a thin `extern "C"` shim over the Swift glue
//! (`swift/Recognize.swift`), which drives Apple's Swift-only
//! `RecognizeTextRequest` (macOS 15+). One C call per image, one JSON
//! `Recognition` back — `serde` parses it straight into the engine-api
//! types. Strings allocated in Swift are freed in Swift
//! (`shrike_av_free_string`); the guard type makes that structural.

use std::ffi::{c_char, CStr};

use shrike_engine_api::Recognition;
use shrike_ffi::NativeResult;

use crate::empty_recognition;

extern "C" {
    fn shrike_av_recognize_one(ptr: *const u8, len: usize) -> *mut c_char;
    fn shrike_av_fingerprint() -> *mut c_char;
    fn shrike_av_free_string(ptr: *mut c_char);
}

/// A Swift-allocated C string, freed on the Swift side's allocator when
/// dropped — never `libc::free` across the boundary.
struct SwiftString(*mut c_char);

impl SwiftString {
    /// Wrap a returned pointer; `None` for null (the Swift side's
    /// "unavailable" sentinel).
    fn wrap(ptr: *mut c_char) -> Option<Self> {
        (!ptr.is_null()).then_some(Self(ptr))
    }

    fn as_str(&self) -> &str {
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
/// (a failed request — logged here, the empty recognition flows on).
#[derive(serde::Deserialize)]
struct Wire {
    #[serde(default)]
    error: Option<String>,
    #[serde(flatten)]
    recognition: Recognition,
}

/// `apple-vision-swift:{revision}:macos{X.Y.Z}` — the hard cut from the
/// objc2 engine's `apple-vision:rev{N}` lineage (#398): the new API rides a
/// newer text model, so the changed identity deliberately re-derives all
/// OCR rows once. Below macOS 15 (the API floor) the Swift side returns
/// null and construction fails `unavailable`, like the off-macOS stub.
pub(crate) fn fingerprint() -> NativeResult<String> {
    let raw = unsafe { shrike_av_fingerprint() };
    match SwiftString::wrap(raw) {
        Some(s) => Ok(s.as_str().to_owned()),
        None => Err(crate::unavailable()),
    }
}

/// One image through the Swift glue. A failed request logs and yields the
/// empty recognition — per-item failures never sink a batch.
pub(crate) fn recognize_one(bytes: &[u8]) -> Recognition {
    let raw = unsafe { shrike_av_recognize_one(bytes.as_ptr(), bytes.len()) };
    let Some(json) = SwiftString::wrap(raw) else {
        return empty_recognition();
    };
    match serde_json::from_str::<Wire>(json.as_str()) {
        Ok(wire) => {
            if let Some(error) = wire.error {
                tracing::warn!("Vision request failed: {error}");
            }
            wire.recognition
        }
        Err(e) => {
            tracing::warn!("Vision glue returned unparseable JSON: {e}");
            empty_recognition()
        }
    }
}
