//! Apple Vision OCR as a native engine crate (#342 P3, the port of #221's
//! pyobjc backend): `VNRecognizeTextRequest` at accurate level with language
//! correction, per-line text + confidence + normalized top-left boxes — the
//! one-pass text+positions contract, 1:1 with the retired Python backend
//! (including the fingerprint format, so existing derived text never
//! spuriously re-derives on the swap).
//!
//! Route 1 of the engine contract: Vision's `performRequests:` is synchronous,
//! so the engine implements [`RecognizeMedia`] (pure chunk compute, no
//! execution assumptions) and the `Blocking` adapter moves it onto
//! the kernel runtime's blocking pool. The struct holds only plain config — every
//! Vision/Foundation object is created per call on whatever thread runs it —
//! so it is naturally `Send + Sync`.
//!
//! Off macOS the crate compiles to the same API with a constructor returning
//! `NativeError::unavailable` — the workspace builds everywhere without
//! platform surgery in the build graph.

use shrike_engine_api::{MediaItem, Recognition, RecognizeMedia};
use shrike_ffi::{NativeError, NativeResult};

#[cfg(target_os = "macos")]
mod imp;
#[cfg(not(target_os = "macos"))]
mod imp_stub;
#[cfg(not(target_os = "macos"))]
use imp_stub as imp;

/// The engine: stateless beyond its cached fingerprint (Vision objects are
/// per-call), so one instance serves concurrent lanes.
pub struct AppleVisionRecognizer {
    fingerprint: String,
}

impl AppleVisionRecognizer {
    /// Construct, failing `unavailable` where Vision isn't (non-macOS).
    pub fn new() -> NativeResult<Self> {
        Ok(Self {
            fingerprint: imp::fingerprint()?,
        })
    }

    /// The platform identity: `apple-vision:rev{N}:macos{X.Y[.Z]}` — request
    /// revision + OS version, byte-compatible with the Python backend's
    /// format (an OS upgrade re-derives, exactly like a model change
    /// rebuilds vectors).
    pub fn fingerprint_str(&self) -> &str {
        &self.fingerprint
    }

    /// Recognize one image; an unreadable/empty/failed item yields the empty
    /// recognition (text "", confidence 0) rather than an error — per-item
    /// failures never sink a batch (the [`Recognizer`] contract).
    ///
    /// [`Recognizer`]: shrike_engine_api::Recognizer
    pub fn recognize_one(&self, bytes: &[u8]) -> Recognition {
        if bytes.is_empty() {
            return empty_recognition();
        }
        imp::recognize_one(bytes)
    }
}

impl RecognizeMedia for AppleVisionRecognizer {
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>> {
        Ok(items.iter().map(|m| self.recognize_one(&m.bytes)).collect())
    }

    fn fingerprint(&self) -> Option<String> {
        Some(self.fingerprint.clone())
    }
}

pub(crate) fn empty_recognition() -> Recognition {
    Recognition {
        text: String::new(),
        confidence: 0.0,
        segments: Vec::new(),
    }
}

/// Round to 4 decimal places (the segments contract's box precision, matching
/// the Python backend's `round(v, 4)`).
#[cfg(target_os = "macos")]
pub(crate) fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

// Keep the unavailable error construction in one place for the stub.
#[allow(dead_code)]
pub(crate) fn unavailable() -> NativeError {
    NativeError::unavailable("Apple Vision OCR is only available on macOS")
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn stub_constructor_is_unavailable() {
        // .err(), not .unwrap_err(): the Ok type carries no Debug impl.
        let err = super::AppleVisionRecognizer::new()
            .err()
            .expect("the stub constructor must fail off macOS");
        assert!(err.to_string().contains("only available on macOS"));
    }

    #[cfg(target_os = "macos")]
    mod live {
        use super::super::*;

        fn engine() -> AppleVisionRecognizer {
            AppleVisionRecognizer::new().expect("Vision is available on macOS")
        }

        #[test]
        fn fingerprint_format_matches_python_backend() {
            let fp = engine().fingerprint_str().to_string();
            // apple-vision:rev{N}:macos{X.Y[.Z]} — and stable across calls.
            assert!(fp.starts_with("apple-vision:rev"), "{fp}");
            let macos = fp.split(":macos").nth(1).expect("macos segment");
            let parts: Vec<&str> = macos.split('.').collect();
            assert!(parts.len() >= 2, "{fp}");
            assert!(parts.iter().all(|p| p.parse::<u32>().is_ok()), "{fp}");
            assert!(
                !macos.ends_with(".0"),
                "trailing .0 patch must be elided for Python-backend parity: {fp}"
            );
            assert_eq!(engine().fingerprint_str(), fp);
        }

        #[test]
        fn empty_bytes_yield_empty_recognition() {
            let r = engine().recognize_one(b"");
            assert_eq!((r.text.as_str(), r.confidence), ("", 0.0));
            assert!(r.segments.is_empty());
        }

        #[test]
        fn garbage_bytes_yield_empty_recognition_not_error() {
            let r = engine().recognize_one(&[0u8; 64]);
            assert_eq!((r.text.as_str(), r.confidence), ("", 0.0));
        }

        #[test]
        fn reads_the_checked_in_fixture() {
            // A pre-rendered PNG (tests/fixtures) — black "ELECTRON TRANSPORT
            // CHAIN" on white, the same phrase the Python live tests render.
            let png = include_bytes!("../tests/fixtures/ocr_phrase.png");
            let r = engine().recognize_one(png);
            assert!(
                r.text.to_lowercase().contains("electron transport chain"),
                "got {:?}",
                r.text
            );
            assert!(r.confidence > 0.0 && r.confidence <= 1.0);
            let seg = r.segments.first().expect("segments retained");
            let bbox = seg.bbox.expect("box present");
            // Normalized, top-left origin: every coordinate in [0, 1].
            assert!(bbox.iter().all(|v| (0.0..=1.0).contains(v)), "{bbox:?}");
            // 4-dp rounding (the contract's precision).
            for v in bbox {
                assert!((v * 10_000.0 - (v * 10_000.0).round()).abs() < 1e-9);
            }
        }

        #[test]
        fn chunk_preserves_order_and_flattens_lines() {
            let png = include_bytes!("../tests/fixtures/ocr_phrase.png");
            let items = vec![
                MediaItem::from_named("a.png", png.to_vec()),
                MediaItem::untyped(Vec::new()),
            ];
            let out = engine().recognize_chunk(&items).unwrap();
            assert_eq!(out.len(), 2);
            assert!(!out[0].text.is_empty());
            assert!(out[1].text.is_empty());
            // Overall confidence is the mean of per-line confidences.
            let mean = out[0].segments.iter().map(|s| s.confidence).sum::<f64>()
                / out[0].segments.len() as f64;
            assert!((out[0].confidence - mean).abs() < 1e-9);
        }
    }
}
