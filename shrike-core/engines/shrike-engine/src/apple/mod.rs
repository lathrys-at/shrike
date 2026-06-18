//! The Apple platform recognizers as engines (#342 P3/#410; feature
//! `engine-apple`): the `RecognizeMedia` impls for Vision OCR ([`vision`]) and
//! SpeechAnalyzer ASR ([`speech`]), over `shrike-platform`'s raw Swift/C-ABI
//! glue (#709). The glue returns raw JSON; THIS layer parses it into the
//! engine-api types and owns the engine contract — so `shrike-platform` carries
//! no `shrike-engine-api` dep, and the swiftc toolchain stays quarantined there
//! (this crate is `build.rs`-free).
//!
//! Route 1 of the engine contract: the platform C entries are synchronous (the
//! Swift side bridges the async APIs internally — safe because these run on the
//! kernel runtime's *blocking* pool via the `Blocking` adapter, disjoint from
//! Swift's cooperative executor), so the engines implement [`RecognizeMedia`]
//! (pure chunk compute). Each engine holds only its cached fingerprint (Vision/
//! analyzer objects are per-call), so it is naturally `Send + Sync`.
//!
//! Off macOS the constructors return `NativeError::unavailable` (the glue's
//! calls return `None`), so the workspace builds everywhere without platform
//! surgery.

use shrike_engine_api::Recognition;
use shrike_error::NativeError;

pub mod speech;
pub mod vision;

pub use speech::AppleSpeechTranscriber;
pub use vision::AppleVisionRecognizer;

/// The wire shape from the Swift glue: a [`Recognition`] plus an optional
/// `error` (a failed request — logged, the recognition flows on).
#[derive(serde::Deserialize)]
struct Wire {
    #[serde(default)]
    error: Option<String>,
    #[serde(flatten)]
    recognition: Recognition,
}

/// Parse one raw JSON payload from the glue into a [`Recognition`], logging a
/// carried `error`; unparseable JSON (or a `None` from the glue, handled by the
/// caller) degrades to the empty recognition — per-item failures never sink a
/// batch.
fn parse_wire(json: &str) -> Recognition {
    match serde_json::from_str::<Wire>(json) {
        Ok(wire) => {
            if let Some(error) = wire.error {
                tracing::warn!("recognition glue reported: {error}");
            }
            wire.recognition
        }
        Err(e) => {
            tracing::warn!("recognition glue returned unparseable JSON: {e}");
            empty_recognition()
        }
    }
}

fn empty_recognition() -> Recognition {
    Recognition {
        text: String::new(),
        confidence: 0.0,
        segments: Vec::new(),
    }
}

fn unavailable() -> NativeError {
    NativeError::unavailable("Apple Vision OCR is only available on macOS")
}

fn speech_unavailable() -> NativeError {
    NativeError::unavailable(
        "Apple SpeechAnalyzer ASR is only available on macOS 26+ (and a macOS 26 build SDK), \
         for a supported locale",
    )
}
