//! The recognition seam (#228/#218, kernel-side): OCR/ASR engines produce
//! TEXT (+ structure) from media bytes — the kernel's second injected
//! capability, the exact sibling of [`crate::Embedder`] and the second
//! concrete slice of #342's pluggable-engine architecture. The kernel never
//! knows which engine recognizes (Apple Vision via the harness, Tesseract, a
//! remote service); it consumes the trait, and the harness attaches an
//! implementation at assembly — or doesn't, and recognition is simply off.
//!
//! One pass, many consumers (#228's load-bearing requirement): a
//! [`Recognition`] retains both the flattened text AND the per-segment
//! structure (boxes for OCR; time spans for ASR later), so the index/lexical
//! consumers and the positional consumer (#230 occlusion) share one
//! invocation — never flatten-and-discard.

use futures::future::BoxFuture;

use shrike_ffi::NativeResult;

/// One recognized segment: a line/word for OCR (with an optional normalized
/// `[x, y, w, h]` box) — the shape generalizes to time spans for ASR (a
/// `bbox` of `[start, 0, duration, 0]` is deliberately NOT used; ASR adds its
/// own field when it lands, without breaking this).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    pub text: String,
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 4]>,
}

/// One media item's recognition: the flattened text (reading order), the
/// overall confidence (engine-defined aggregate), and the retained segments.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Recognition {
    pub text: String,
    pub confidence: f64,
    #[serde(default)]
    pub segments: Vec<Segment>,
}

/// The recognizer the harness injects — async like [`crate::Embedder`] (a
/// platform API or remote engine is genuinely asynchronous; CPU engines
/// return ready futures), with the same optional identity metadata.
pub trait Recognizer: Send + Sync + 'static {
    /// Recognize a batch of media items (bytes in, one [`Recognition`] per
    /// item, order-preserving). An unreadable item yields an empty
    /// recognition (text "", confidence 0) rather than failing the batch.
    fn recognize(&self, items: Vec<Vec<u8>>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>>;

    /// Stable engine identity (model/OS version) — a changed fingerprint
    /// invalidates derived text on the next pending sweep, exactly as the
    /// embedder fingerprint invalidates vectors.
    fn fingerprint(&self) -> Option<String> {
        None
    }
}

/// Recognizers share freely, like embedders.
impl<T: Recognizer> Recognizer for std::sync::Arc<T> {
    fn recognize(&self, items: Vec<Vec<u8>>) -> BoxFuture<'_, NativeResult<Vec<Recognition>>> {
        (**self).recognize(items)
    }

    fn fingerprint(&self) -> Option<String> {
        (**self).fingerprint()
    }
}

/// The gating policy (#199): which recognitions mint an OCR vector, and
/// which enter the lexical store at all. Confidence + substance separate
/// text-bearing images from pictorial ones automatically — no detector.
#[derive(Debug, Clone)]
pub struct RecognitionGate {
    /// Below this overall confidence, the recognition is noise — nothing is
    /// stored or embedded.
    pub min_confidence: f64,
    /// Fewer recognized characters than this is not substantive — store
    /// nothing (an icon label or watermark fragment isn't content).
    pub min_chars_lexical: usize,
    /// Minting an embedding vector demands more substance than lexical
    /// indexing (a short label is findable text but a meaningless vector).
    pub min_chars_vector: usize,
}

impl Default for RecognitionGate {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            min_chars_lexical: 4,
            min_chars_vector: 20,
        }
    }
}

/// What a recognition may feed, per the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOutcome {
    /// Below confidence or lexical substance: store nothing.
    Drop,
    /// Lexically indexable, but not substantive enough for a vector.
    Lexical,
    /// Both: a derived row AND a provenance-tagged text-space vector.
    LexicalAndVector,
}

impl RecognitionGate {
    pub fn judge(&self, recognition: &Recognition) -> GateOutcome {
        let chars = recognition.text.trim().chars().count();
        if recognition.confidence < self.min_confidence || chars < self.min_chars_lexical {
            return GateOutcome::Drop;
        }
        if chars < self.min_chars_vector {
            return GateOutcome::Lexical;
        }
        GateOutcome::LexicalAndVector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(text: &str, confidence: f64) -> Recognition {
        Recognition {
            text: text.to_string(),
            confidence,
            segments: Vec::new(),
        }
    }

    #[test]
    fn gate_separates_noise_labels_and_content() {
        let gate = RecognitionGate::default();
        // Low confidence → drop regardless of length.
        assert_eq!(
            gate.judge(&rec("plenty of perfectly good text here", 0.3)),
            GateOutcome::Drop
        );
        // Confident but tiny → drop (a watermark fragment).
        assert_eq!(gate.judge(&rec("ok", 0.9)), GateOutcome::Drop);
        // A short label: findable, not vector-worthy.
        assert_eq!(gate.judge(&rec("Mitochondrion", 0.9)), GateOutcome::Lexical);
        // Substantive: both consumers.
        assert_eq!(
            gate.judge(&rec(
                "The inner membrane hosts the electron transport chain",
                0.9
            )),
            GateOutcome::LexicalAndVector
        );
        // Whitespace doesn't count as substance.
        assert_eq!(gate.judge(&rec("   a   ", 0.9)), GateOutcome::Drop);
    }

    #[test]
    fn recognition_serde_round_trips_with_segments() {
        let r = Recognition {
            text: "label".into(),
            confidence: 0.8,
            segments: vec![Segment {
                text: "label".into(),
                confidence: 0.8,
                bbox: Some([0.1, 0.2, 0.3, 0.05]),
            }],
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: Recognition = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
        // A box-less segment omits the key (the ASR-friendly shape).
        let no_box = serde_json::to_string(&Segment {
            text: "t".into(),
            confidence: 1.0,
            bbox: None,
        })
        .unwrap();
        assert!(!no_box.contains("bbox"));
    }
}
