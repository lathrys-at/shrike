//! Raw platform-bindings glue: the Swift/C-ABI bridges to Apple's
//! Vision OCR and SpeechAnalyzer ASR (Android later). **GLUE ONLY** — this
//! crate knows no engine contract (no `shrike-engine-api` dep): it exposes safe
//! Rust wrappers returning the recognizer's raw JSON strings + fingerprints,
//! and `shrike-engine::apple` parses them into the engine-api types and
//! implements `RecognizeMedia`. Keeping the glue here quarantines the swiftc
//! toolchain off the engine/server build graph (the engine crate stays
//! `build.rs`-free); the engine pulls this crate only when `engine-apple` is on.
//!
//! Off macOS the crate compiles to the same API with the calls returning `None`
//! — the workspace builds everywhere without platform surgery in the build graph
//! (build.rs no-ops, no Swift toolchain needed). Building on macOS requires full
//! Xcode (the Swift-only Vision module isn't in the Command Line Tools SDK);
//! running needs nothing extra.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

// The Swift C-ABI memory guard (the free fn) is macOS-only; off macOS the
// vision/speech shims are pure-Rust `None` stubs that never reference it.
#[cfg(target_os = "macos")]
mod glue;

pub mod speech;
pub mod vision;

/// Reserved for the Android JNI/NDK glue (behind `engine-android`) — the home
/// for the Kotlin bridge, mirroring how `vision`/`speech` host the Apple one.
#[cfg(feature = "engine-android")]
pub mod android {}

// Off-platform (non-macOS) the vision/speech shims are total `None` stubs. The
// engine layer (`shrike-engine::apple`) relies on that contract: its
// constructor probes `fingerprint()` and fails to `unavailable` when it is
// `None`, so a non-macOS build never dispatches recognition. These tests pin
// that contract on the platform CI actually runs the unit lane on (Linux); on
// macOS the same calls reach the Swift glue and are exercised by the engine
// suite instead, so the assertions are gated off it.
#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::{speech, vision};

    #[test]
    fn vision_is_unavailable_off_platform() {
        assert_eq!(vision::fingerprint(), None);
        assert_eq!(vision::recognize_one(b"not really an image"), None);
    }

    #[test]
    fn speech_is_unavailable_off_platform() {
        assert_eq!(speech::fingerprint("en-US"), None);
        assert_eq!(
            speech::transcribe_one(b"audio", Some("audio/wav"), "en-US"),
            None
        );
        assert_eq!(speech::ensure_assets("en-US"), None);
    }

    #[test]
    fn off_platform_calls_are_total_over_adversarial_input() {
        // The stubs must be panic-free for any input the engine layer could hand
        // them before it learns the engine is unavailable: empty/large byte
        // buffers, and locales with unusual content (an interior NUL is the
        // CString hazard the macOS path guards — off-platform it is simply
        // ignored, never a panic).
        assert_eq!(vision::recognize_one(&[]), None);
        assert_eq!(vision::recognize_one(&vec![0u8; 4096]), None);
        for locale in ["", "xx-YY", "e\0n", "ja_JP.UTF-8", "💥"] {
            assert_eq!(speech::fingerprint(locale), None);
            assert_eq!(speech::ensure_assets(locale), None);
            assert_eq!(speech::transcribe_one(&[], None, locale), None);
        }
    }
}
