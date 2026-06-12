//! Apple SpeechAnalyzer ASR as a second engine in this crate (#410):
//! on-device transcription (macOS 26+, Swift-only) behind the same Swift
//! C ABI pattern as the OCR half. Segments carry `Locator::Span`
//! (`[start_seconds, duration_seconds]`) — the time-axis counterpart of
//! OCR's boxes.
//!
//! Route 1 of the engine contract, like OCR: the C entries are synchronous
//! (the Swift side bridges the async analyzer internally; safe on the
//! kernel runtime's blocking pool), so the engine implements
//! [`RecognizeMedia`].
//!
//! **Two-state locale model**: `new(locale)` validates the locale is
//! *supported* (and that the API exists — macOS 26+ at runtime AND at
//! build-SDK time) but does NOT require the on-device model *installed* —
//! a constructor must never drive a multi-hundred-MB download.
//! [`AppleSpeechTranscriber::ensure_assets`] is the one explicit download
//! path; transcribing with assets missing yields empty recognitions whose
//! carried error the shim logs.
//!
//! Live transcription tests are opt-in via `SHRIKE_ASR_LIVE_TESTS=1` (they
//! may download assets and synthesize audio via `say`); default `cargo
//! test` stays fast and download-free.

use shrike_engine_api::{MediaItem, Recognition, RecognizeMedia};
use shrike_ffi::NativeResult;

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::{c_char, CString};

    use shrike_engine_api::Recognition;
    use shrike_ffi::NativeResult;

    use crate::glue::{parse_wire, SwiftString};

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

    /// `apple-speech:{resolved-locale}:macos{X.Y.Z}`, or `unavailable` when
    /// the API is absent (pre-26 OS or pre-26 build SDK) or the locale is
    /// unsupported.
    pub(super) fn fingerprint(locale: &str) -> NativeResult<String> {
        let locale = c_locale(locale);
        let raw = unsafe { shrike_av_speech_fingerprint(locale.as_ptr()) };
        match SwiftString::wrap(raw) {
            Some(s) => Ok(s.as_str().to_owned()),
            None => Err(crate::speech_unavailable()),
        }
    }

    pub(super) fn transcribe_one(bytes: &[u8], mime: Option<&str>, locale: &str) -> Recognition {
        let locale = c_locale(locale);
        let mime = mime.and_then(|m| CString::new(m).ok());
        let raw = unsafe {
            shrike_av_transcribe_one(
                bytes.as_ptr(),
                bytes.len(),
                mime.as_ref().map_or(std::ptr::null(), |m| m.as_ptr()),
                locale.as_ptr(),
            )
        };
        let Some(json) = SwiftString::wrap(raw) else {
            return crate::empty_recognition();
        };
        parse_wire(&json)
    }

    /// `ready` (already installed) / `installed` (downloaded now) /
    /// `unsupported`, or an error string from the installer.
    pub(super) fn ensure_assets(locale: &str) -> NativeResult<String> {
        let locale = c_locale(locale);
        let raw = unsafe { shrike_av_speech_ensure_assets(locale.as_ptr()) };
        let json = SwiftString::wrap(raw).ok_or_else(crate::speech_unavailable)?;
        #[derive(serde::Deserialize)]
        struct Status {
            status: String,
            #[serde(default)]
            error: Option<String>,
        }
        let parsed: Status = serde_json::from_str(json.as_str()).map_err(|e| {
            shrike_ffi::NativeError::internal(format!("asset status unparseable: {e}"))
        })?;
        if let Some(error) = parsed.error {
            return Err(shrike_ffi::NativeError::unavailable(format!(
                "speech asset install failed: {error}"
            )));
        }
        Ok(parsed.status)
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use shrike_engine_api::Recognition;
    use shrike_ffi::NativeResult;

    pub(super) fn fingerprint(_locale: &str) -> NativeResult<String> {
        Err(crate::speech_unavailable())
    }

    pub(super) fn transcribe_one(_bytes: &[u8], _mime: Option<&str>, _locale: &str) -> Recognition {
        // Unreachable through the public API (new() fails first); kept
        // total so the stub compiles the same call graph.
        crate::empty_recognition()
    }

    pub(super) fn ensure_assets(_locale: &str) -> NativeResult<String> {
        Err(crate::speech_unavailable())
    }
}

/// The ASR engine: a validated locale + its cached fingerprint (analyzer
/// objects are per-call), so one instance serves concurrent lanes.
pub struct AppleSpeechTranscriber {
    locale: String,
    fingerprint: String,
}

impl AppleSpeechTranscriber {
    /// Construct for a BCP-47 locale (`None` = `en-US`), failing
    /// `unavailable` where SpeechAnalyzer isn't (non-macOS, pre-26 OS or
    /// build SDK, unsupported locale). Never downloads assets — see
    /// [`Self::ensure_assets`].
    pub fn new(locale: Option<&str>) -> NativeResult<Self> {
        let locale = locale.unwrap_or("en-US").to_owned();
        let fingerprint = imp::fingerprint(&locale)?;
        Ok(Self {
            locale,
            fingerprint,
        })
    }

    /// The platform identity: `apple-speech:{resolved-locale}:macos{X.Y.Z}`
    /// — locale + OS version (no public model-version accessor exists, so
    /// the OS version is the honest proxy; an asset update without an OS
    /// bump won't re-derive — accepted and documented).
    pub fn fingerprint_str(&self) -> &str {
        &self.fingerprint
    }

    /// Ensure the locale's on-device model is installed: `ready` /
    /// `installed` / `unsupported`. The ONE path allowed to drive the
    /// download — call it from operational code (or the live tests), never
    /// from a constructor.
    pub fn ensure_assets(&self) -> NativeResult<String> {
        imp::ensure_assets(&self.locale)
    }

    /// Transcribe one audio item; an unreadable/empty/failed item yields
    /// the empty recognition rather than an error — per-item failures never
    /// sink a batch.
    pub fn transcribe_one(&self, bytes: &[u8], mime: Option<&str>) -> Recognition {
        if bytes.is_empty() {
            return crate::empty_recognition();
        }
        imp::transcribe_one(bytes, mime, &self.locale)
    }
}

impl RecognizeMedia for AppleSpeechTranscriber {
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>> {
        Ok(items
            .iter()
            .map(|m| self.transcribe_one(&m.bytes, m.mime.as_deref()))
            .collect())
    }

    fn fingerprint(&self) -> Option<String> {
        Some(self.fingerprint.clone())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_os = "macos"))]
    use super::*;

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn stub_constructor_is_unavailable() {
        let err = AppleSpeechTranscriber::new(None)
            .err()
            .expect("the stub constructor must fail off macOS");
        assert!(err.to_string().contains("only available"));
    }

    /// On macOS the constructor's verdict depends on the OS and the SDK the
    /// Swift half was built against: macOS 26+ → available; older → the
    /// same `unavailable` degrade as off-macOS. Both are valid outcomes for
    /// this tier; the live transcription tier below is opt-in.
    #[cfg(target_os = "macos")]
    mod live {
        use super::super::*;
        use shrike_engine_api::Locator;

        fn live_engine() -> Option<AppleSpeechTranscriber> {
            match AppleSpeechTranscriber::new(None) {
                Ok(engine) => Some(engine),
                Err(e) => {
                    assert!(
                        e.to_string().contains("only available"),
                        "constructor may only fail unavailable: {e}"
                    );
                    None
                }
            }
        }

        #[test]
        fn fingerprint_format_and_locale_resolution() {
            let Some(engine) = live_engine() else { return };
            let fp = engine.fingerprint_str().to_string();
            assert!(fp.starts_with("apple-speech:en"), "{fp}");
            let macos = fp.split(":macos").nth(1).expect("macos segment");
            assert_eq!(macos.split('.').count(), 3, "un-elided X.Y.Z: {fp}");
            assert_eq!(engine.fingerprint_str(), fp);
        }

        #[test]
        fn unsupported_locale_is_unavailable_at_construction() {
            if live_engine().is_none() {
                return;
            }
            let err = AppleSpeechTranscriber::new(Some("xx-XX"))
                .err()
                .expect("an unsupported locale must fail construction");
            assert!(err.to_string().contains("unavailable") || !err.to_string().is_empty());
        }

        #[test]
        fn empty_bytes_yield_empty_recognition() {
            let Some(engine) = live_engine() else { return };
            let r = engine.transcribe_one(b"", None);
            assert_eq!((r.text.as_str(), r.confidence), ("", 0.0));
            assert!(r.segments.is_empty());
        }

        /// The full live pass: may download the locale's model assets and
        /// synthesizes its fixture via `say` — opt-in only.
        fn live_opted_in() -> bool {
            std::env::var("SHRIKE_ASR_LIVE_TESTS").as_deref() == Ok("1")
        }

        fn synthesize(phrase: &str) -> Vec<u8> {
            // Unique per call — tests run concurrently in one process, so a
            // pid-keyed path would race.
            static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir();
            let path = dir.join(format!("shrike-asr-{}-{seq}.aiff", std::process::id()));
            let status = std::process::Command::new("say")
                .args(["-o", path.to_str().expect("tmp path is UTF-8"), phrase])
                .status()
                .expect("`say` ships with macOS");
            assert!(status.success(), "say failed");
            let bytes = std::fs::read(&path).expect("synthesized fixture readable");
            let _ = std::fs::remove_file(&path);
            bytes
        }

        #[test]
        fn transcribes_synthesized_speech_with_spans() {
            if !live_opted_in() {
                return;
            }
            let Some(engine) = live_engine() else { return };
            let status = engine.ensure_assets().expect("asset path reachable");
            assert!(
                ["ready", "installed"].contains(&status.as_str()),
                "assets not installable: {status}"
            );

            let bytes = synthesize("the electron transport chain");
            let r = engine.transcribe_one(&bytes, Some("audio/aiff"));
            assert!(
                r.text.to_lowercase().contains("electron transport chain"),
                "got {:?}",
                r.text
            );
            assert!(r.confidence > 0.0 && r.confidence <= 1.0);
            let seg = r.segments.first().expect("segments retained");
            let Some(Locator::Span([start, duration])) = seg.locator else {
                panic!("span present: {:?}", seg.locator);
            };
            assert!(start >= 0.0 && duration > 0.0, "{start} {duration}");
            // 4-dp rounding (the contract's precision).
            for v in [start, duration] {
                assert!((v * 10_000.0 - (v * 10_000.0).round()).abs() < 1e-9);
            }
        }

        #[test]
        fn garbage_bytes_yield_empty_recognition_not_error() {
            if !live_opted_in() {
                return;
            }
            let Some(engine) = live_engine() else { return };
            let r = engine.transcribe_one(&[0u8; 64], Some("audio/wav"));
            assert_eq!((r.text.as_str(), r.confidence), ("", 0.0));
        }

        #[test]
        fn chunk_preserves_order() {
            if !live_opted_in() {
                return;
            }
            let Some(engine) = live_engine() else { return };
            engine.ensure_assets().expect("asset path reachable");
            let spoken = synthesize("alpha first line");
            let items = vec![
                MediaItem::from_named("a.aiff", spoken),
                MediaItem::untyped(Vec::new()),
            ];
            let out = engine.recognize_chunk(&items).unwrap();
            assert_eq!(out.len(), 2);
            assert!(
                out[0].text.to_lowercase().contains("alpha"),
                "{:?}",
                out[0].text
            );
            assert!(out[1].text.is_empty());
        }
    }
}
