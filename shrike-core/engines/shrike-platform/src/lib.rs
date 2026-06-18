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
