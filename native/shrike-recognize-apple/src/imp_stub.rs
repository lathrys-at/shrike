//! The off-macOS stub: identical surface, `unavailable` at construction —
//! the same degrade-don't-crash a missing pyobjc gave the Python backend.

use shrike_engine_api::Recognition;
use shrike_ffi::NativeResult;

pub(crate) fn fingerprint() -> NativeResult<String> {
    Err(crate::unavailable())
}

pub(crate) fn recognize_one(_bytes: &[u8]) -> Recognition {
    // Unreachable through the public API (new() fails first); kept total so
    // the stub compiles the same call graph.
    crate::empty_recognition()
}
