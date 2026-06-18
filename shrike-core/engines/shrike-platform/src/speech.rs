//! The macOS SpeechAnalyzer ASR glue: a thin `extern "C"` shim over the
//! Swift glue (`swift/Transcribe.swift`), same C-ABI pattern as the OCR half.
//! GLUE ONLY: returns raw JSON strings; `shrike-engine::apple` parses them.

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::{c_char, CString};

    use crate::glue::SwiftString;

    extern "C" {
        fn shrike_av_transcribe_one(
            ptr: *const u8,
            len: usize,
            mime: *const c_char,
            locale: *const c_char,
        ) -> *mut c_char;
        fn shrike_av_speech_fingerprint(locale: *const c_char) -> *mut c_char;
        fn shrike_av_speech_ensure_assets(locale: *const c_char) -> *mut c_char;
    }

    fn c_locale(locale: &str) -> CString {
        // A locale id never carries an interior NUL; degrade to the default
        // rather than panic if one ever does.
        CString::new(locale).unwrap_or_else(|_| CString::new("en-US").expect("static"))
    }

    /// `apple-speech:{resolved-locale}:macos{X.Y.Z}` raw from the Swift side, or
    /// `None` when the API is absent (pre-26 OS or pre-26 build SDK) or the
    /// locale is unsupported.
    pub(super) fn fingerprint(locale: &str) -> Option<String> {
        let locale = c_locale(locale);
        // SAFETY: `shrike_av_speech_fingerprint` is the Swift glue's C ABI export.
        // `locale.as_ptr()` is a valid NUL-terminated C string for the duration
        // of the call (`locale` outlives it). The Swift side only reads it. It
        // returns a Swift-allocated NUL-terminated C string or null;
        // `SwiftString::wrap` owns/frees a non-null return and maps null to None.
        let raw = unsafe { shrike_av_speech_fingerprint(locale.as_ptr()) };
        SwiftString::wrap(raw).map(SwiftString::into_string)
    }

    /// One audio item → its raw JSON recognition string, or `None` on a null
    /// Swift return (the engine layer degrades that to the empty recognition).
    pub(super) fn transcribe_one(bytes: &[u8], mime: Option<&str>, locale: &str) -> Option<String> {
        let locale = c_locale(locale);
        let mime = mime.and_then(|m| CString::new(m).ok());
        // SAFETY: `shrike_av_transcribe_one` is the Swift glue's C ABI export.
        // The (ptr, len) pair is from a live `&[u8]` borrow (valid for `len`
        // bytes, read-only, for the call's duration); `mime`/`locale` are valid
        // NUL-terminated C strings (or null for an absent mime — the Swift side
        // accepts null) that outlive the call. It returns a Swift-allocated C
        // string or null, owned/freed by `SwiftString`.
        let raw = unsafe {
            shrike_av_transcribe_one(
                bytes.as_ptr(),
                bytes.len(),
                mime.as_ref().map_or(std::ptr::null(), |m| m.as_ptr()),
                locale.as_ptr(),
            )
        };
        SwiftString::wrap(raw).map(SwiftString::into_string)
    }

    /// The asset-status JSON (`{"status": …, "error"?: …}`) raw from the Swift
    /// side, or `None` on a null return (the engine layer maps that to
    /// `unavailable`).
    pub(super) fn ensure_assets(locale: &str) -> Option<String> {
        let locale = c_locale(locale);
        // SAFETY: `shrike_av_speech_ensure_assets` is the Swift glue's C ABI
        // export. `locale.as_ptr()` is a valid NUL-terminated C string for the
        // call (it outlives it); read-only on the Swift side. Returns a
        // Swift-allocated C string or null, owned/freed by `SwiftString`.
        let raw = unsafe { shrike_av_speech_ensure_assets(locale.as_ptr()) };
        SwiftString::wrap(raw).map(SwiftString::into_string)
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub(super) fn fingerprint(_locale: &str) -> Option<String> {
        None
    }

    pub(super) fn transcribe_one(
        _bytes: &[u8],
        _mime: Option<&str>,
        _locale: &str,
    ) -> Option<String> {
        // Unreachable through the engine layer (its constructor fails first);
        // kept total so the stub compiles the same call graph.
        None
    }

    pub(super) fn ensure_assets(_locale: &str) -> Option<String> {
        None
    }
}

/// `apple-speech:{resolved-locale}:macos{X.Y.Z}`, or `None` when SpeechAnalyzer
/// isn't available (non-macOS, pre-26 OS or build SDK, unsupported locale).
pub fn fingerprint(locale: &str) -> Option<String> {
    imp::fingerprint(locale)
}

/// One audio item's raw JSON recognition, or `None` on a null Swift return.
pub fn transcribe_one(bytes: &[u8], mime: Option<&str>, locale: &str) -> Option<String> {
    imp::transcribe_one(bytes, mime, locale)
}

/// The locale's on-device asset-status JSON (`ready`/`installed`/`unsupported`,
/// possibly with an `error`), or `None` on a null Swift return. The ONE path
/// allowed to drive the (multi-hundred-MB) download — call it from operational
/// code, never a constructor.
pub fn ensure_assets(locale: &str) -> Option<String> {
    imp::ensure_assets(locale)
}
