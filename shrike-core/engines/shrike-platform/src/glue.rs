//! Shared Swift C-ABI plumbing (macOS only): the `SwiftString` memory guard and
//! the one free function both shims (OCR [`super::vision`], ASR [`super::speech`])
//! wrap returned strings in.
//!
//! GLUE ONLY: this crate knows no engine contract — it returns raw JSON
//! strings; the engine layer (`shrike-engine::apple`) parses them into the
//! engine-api types.

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

    /// Borrow the string as UTF-8. The Swift side only ever emits UTF-8; a
    /// torn string degrades to empty rather than panicking mid-batch.
    pub(crate) fn as_str(&self) -> &str {
        // SAFETY: `self.0` is non-null (checked in `wrap`, the only constructor)
        // and points to a NUL-terminated C string the Swift glue allocated and
        // has not freed (this `SwiftString` owns it until its `Drop`, which is
        // the only place it is freed). The borrow's lifetime is tied to `&self`,
        // so the pointer stays valid for the returned `&str`.
        unsafe { CStr::from_ptr(self.0) }.to_str().unwrap_or("")
    }

    /// Take ownership of the string as an owned `String` copy.
    pub(crate) fn into_string(self) -> String {
        self.as_str().to_owned()
    }
}

impl Drop for SwiftString {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a non-null pointer the Swift glue allocated (via
        // `wrap`, the only constructor, which rejects null) and which this guard
        // uniquely owns — `SwiftString` is neither `Clone` nor `Copy`, so it is
        // freed exactly once, here, on the Swift side's matching allocator (the
        // discipline that makes the cross-language ownership sound). The
        // `extern "C"` free is the Swift counterpart of the allocation.
        unsafe { shrike_av_free_string(self.0) }
    }
}
