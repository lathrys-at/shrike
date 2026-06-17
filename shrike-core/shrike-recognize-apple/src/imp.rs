//! The macOS OCR implementation: a thin `extern "C"` shim over the Swift
//! glue (`swift/Recognize.swift`), which drives Apple's Swift-only
//! `RecognizeTextRequest` (macOS 15+). One C call per image, one JSON
//! `Recognition` back — `serde` parses it straight into the engine-api
//! types. Strings allocated in Swift are freed in Swift (the shared
//! `SwiftString` guard makes that structural).

use std::ffi::c_char;

use shrike_engine_api::Recognition;
use shrike_error::NativeResult;

use crate::empty_recognition;
use crate::glue::{parse_wire, SwiftString};

extern "C" {
    fn shrike_av_recognize_one(ptr: *const u8, len: usize) -> *mut c_char;
    fn shrike_av_fingerprint() -> *mut c_char;
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
    parse_wire(&json)
}
