//! Apple Vision OCR as a native engine (Swift glue in
//! `shrike-platform`): Apple's `RecognizeTextRequest` (macOS 15+,
//! Swift-only) at accurate level with language correction, per-line text +
//! confidence + normalized top-left boxes — the one-pass text+positions
//! contract. This layer parses `shrike-platform`'s raw JSON into the engine-api
//! types and implements [`RecognizeMedia`]; the platform crate owns the Swift
//! glue and the `SwiftString` memory discipline.

use shrike_engine_api::{MediaItem, Recognition, RecognizeMedia};
use shrike_error::NativeResult;

use super::{empty_recognition, parse_wire, unavailable};

/// The engine: stateless beyond its cached fingerprint (Vision objects are
/// per-call), so one instance serves concurrent lanes.
pub struct AppleVisionRecognizer {
    fingerprint: String,
}

impl AppleVisionRecognizer {
    /// Construct, failing `unavailable` where Vision isn't (non-macOS, below the
    /// macOS-15 API floor).
    ///
    /// # Errors
    ///
    /// Returns an `unavailable` error where Apple Vision is not available.
    pub fn new() -> NativeResult<Self> {
        Ok(Self {
            fingerprint: shrike_platform::vision::fingerprint().ok_or_else(unavailable)?,
        })
    }

    /// The platform identity: `apple-vision-swift:{revision}:macos{X.Y.Z}`
    /// — model revision + OS version (an OS upgrade re-derives, exactly
    /// like a model change rebuilds vectors). A deliberate hard cut from
    /// the objc2 engine's `apple-vision:rev{N}` lineage: the new
    /// API rides a newer text model, so all OCR rows re-derive once.
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
        match shrike_platform::vision::recognize_one(bytes) {
            Some(json) => parse_wire(&json),
            None => empty_recognition(),
        }
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
        use shrike_engine_api::MediaItem;

        fn engine() -> AppleVisionRecognizer {
            AppleVisionRecognizer::new().expect("Vision is available on macOS")
        }

        #[test]
        fn fingerprint_format() {
            let fp = engine().fingerprint_str().to_string();
            // apple-vision-swift:{revision}:macos{X.Y.Z} — stable across
            // calls; a hard cut from the objc2 lineage.
            assert!(fp.starts_with("apple-vision-swift:revision"), "{fp}");
            let macos = fp.split(":macos").nth(1).expect("macos segment");
            let parts: Vec<&str> = macos.split('.').collect();
            assert_eq!(parts.len(), 3, "un-elided X.Y.Z: {fp}");
            assert!(parts.iter().all(|p| p.parse::<u32>().is_ok()), "{fp}");
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
            // A pre-rendered PNG — black "ELECTRON TRANSPORT CHAIN" on white,
            // the same phrase the Python live tests render.
            let png = include_bytes!("ocr_phrase.png");
            let r = engine().recognize_one(png);
            assert!(
                r.text.to_lowercase().contains("electron transport chain"),
                "got {:?}",
                r.text
            );
            assert!(r.confidence > 0.0 && r.confidence <= 1.0);
            let seg = r.segments.first().expect("segments retained");
            let Some(shrike_engine_api::Locator::Bbox(bbox)) = seg.locator else {
                panic!("box present: {:?}", seg.locator);
            };
            // Normalized, top-left origin: every coordinate in [0, 1].
            assert!(bbox.iter().all(|v| (0.0..=1.0).contains(v)), "{bbox:?}");
            // 4-dp rounding (the contract's precision).
            for v in bbox {
                assert!((v * 10_000.0 - (v * 10_000.0).round()).abs() < 1e-9);
            }
        }

        #[test]
        fn chunk_preserves_order_and_flattens_lines() {
            let png = include_bytes!("ocr_phrase.png");
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
