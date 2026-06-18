//! The Apple Vision OCR glue: a thin `extern "C"` shim over the Swift glue
//! (`swift/Recognize.swift`), which drives Apple's Swift-only
//! `RecognizeTextRequest` (macOS 15+). One C call per image, one JSON string
//! back — GLUE ONLY: this layer returns the raw JSON; `shrike-engine::apple`
//! parses it into the engine-api types. Strings allocated in Swift are freed
//! in Swift (the shared `SwiftString` guard makes that structural). Off macOS
//! the calls return `None` (the engine layer's constructor fails first).

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::c_char;

    use crate::glue::SwiftString;

    extern "C" {
        fn shrike_av_recognize_one(ptr: *const u8, len: usize) -> *mut c_char;
        fn shrike_av_fingerprint() -> *mut c_char;
    }

    pub(super) fn fingerprint() -> Option<String> {
        // SAFETY: `shrike_av_fingerprint` is the Swift glue's C ABI export
        // (linked by build.rs / the bazel genrule); no args, returns a
        // Swift-allocated NUL-terminated C string or null. `SwiftString::wrap`
        // owns/frees a non-null return and maps null to `None`.
        let raw = unsafe { shrike_av_fingerprint() };
        SwiftString::wrap(raw).map(SwiftString::into_string)
    }

    pub(super) fn recognize_one(bytes: &[u8]) -> Option<String> {
        // SAFETY: `shrike_av_recognize_one` is the Swift glue's C ABI export. We
        // pass a (ptr, len) pair from a live `&[u8]` borrow — valid for `len`
        // bytes, read-only on the Swift side (it copies into a Vision image), for
        // the call's duration. It returns a Swift-allocated NUL-terminated C
        // string or null; `SwiftString::wrap` owns/frees a non-null return and
        // maps null to `None`.
        let raw = unsafe { shrike_av_recognize_one(bytes.as_ptr(), bytes.len()) };
        SwiftString::wrap(raw).map(SwiftString::into_string)
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub(super) fn fingerprint() -> Option<String> {
        None
    }

    pub(super) fn recognize_one(_bytes: &[u8]) -> Option<String> {
        // Unreachable through the engine layer (its constructor fails first);
        // kept total so the stub compiles the same call graph.
        None
    }
}

/// `apple-vision-swift:{revision}:macos{X.Y.Z}` raw from the Swift side, or
/// `None` below macOS 15 (the API floor) / off macOS — the engine layer maps
/// that to `unavailable`.
pub fn fingerprint() -> Option<String> {
    imp::fingerprint()
}

/// One image through the Swift glue → its raw JSON recognition string, or
/// `None` when the Swift side returns null (the engine layer degrades that to
/// the empty recognition — per-item failures never sink a batch).
pub fn recognize_one(bytes: &[u8]) -> Option<String> {
    imp::recognize_one(bytes)
}
