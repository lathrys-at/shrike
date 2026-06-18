//! Apple SpeechAnalyzer ASR as a native engine (#410; Swift glue in
//! `shrike-platform` since #709): on-device transcription (macOS 26+,
//! Swift-only) behind the same Swift C-ABI pattern as the OCR half. Segments
//! carry `Locator::Span` (`[start_seconds, duration_seconds]`) — the time-axis
//! counterpart of OCR's boxes. This layer parses `shrike-platform`'s raw JSON
//! into the engine-api types and implements [`RecognizeMedia`].
//!
//! Route 1 of the engine contract, like OCR: the platform C entries are
//! synchronous (the Swift side bridges the async analyzer internally; safe on
//! the kernel runtime's blocking pool).
//!
//! **Two-state locale model**: `new(locale)` validates the locale is *supported*
//! (and that the API exists — macOS 26+ at runtime AND at build-SDK time) but
//! does NOT require the on-device model *installed* — a constructor must never
//! drive a multi-hundred-MB download. [`AppleSpeechTranscriber::ensure_assets`]
//! is the one explicit download path; transcribing with assets missing yields
//! empty recognitions whose carried error the glue logs.
//!
//! Live transcription tests are opt-in via `SHRIKE_ASR_LIVE_TESTS=1` (they may
//! download assets and synthesize audio via `say`); default `cargo test` stays
//! fast and download-free.

use shrike_engine_api::{MediaItem, Recognition, RecognizeMedia};
use shrike_error::{NativeError, NativeResult};

use super::{empty_recognition, parse_wire, speech_unavailable};

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
    ///
    /// # Errors
    ///
    /// Returns an `unavailable` error where SpeechAnalyzer or the locale is
    /// not available.
    pub fn new(locale: Option<&str>) -> NativeResult<Self> {
        let locale = locale.unwrap_or("en-US").to_owned();
        let fingerprint =
            shrike_platform::speech::fingerprint(&locale).ok_or_else(speech_unavailable)?;
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
    ///
    /// # Errors
    ///
    /// Returns an `unavailable` error if the locale is unavailable, or an
    /// error if the asset-status JSON can't be retrieved or parsed.
    pub fn ensure_assets(&self) -> NativeResult<String> {
        let json =
            shrike_platform::speech::ensure_assets(&self.locale).ok_or_else(speech_unavailable)?;
        #[derive(serde::Deserialize)]
        struct Status {
            status: String,
            #[serde(default)]
            error: Option<String>,
        }
        let parsed: Status = serde_json::from_str(&json)
            .map_err(|e| NativeError::internal(format!("asset status unparseable: {e}")))?;
        if let Some(error) = parsed.error {
            return Err(NativeError::unavailable(format!(
                "speech asset install failed: {error}"
            )));
        }
        Ok(parsed.status)
    }

    /// Transcribe one audio item; an unreadable/empty/failed item yields
    /// the empty recognition rather than an error — per-item failures never
    /// sink a batch.
    pub fn transcribe_one(&self, bytes: &[u8], mime: Option<&str>) -> Recognition {
        if bytes.is_empty() {
            return empty_recognition();
        }
        match shrike_platform::speech::transcribe_one(bytes, mime, &self.locale) {
            Some(json) => parse_wire(&json),
            None => empty_recognition(),
        }
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
        use shrike_engine_api::{Locator, MediaItem};

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
            assert!(
                err.to_string().contains("supported locale"),
                "the unavailable error names the locale condition: {err}"
            );
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
