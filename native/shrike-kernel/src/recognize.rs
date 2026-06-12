//! Recognition gating (#199/#228) — the KERNEL's policy half of the
//! recognition capability. The engine contract ([`Recognizer`],
//! [`Recognition`], [`Segment`]) lives in `shrike-engine-api` (#342); what
//! stays here is what the kernel decides for itself: which recognitions are
//! substantive enough to store, and which mint a vector.
//!
//! [`Recognizer`]: shrike_engine_api::Recognizer
//! [`Recognition`]: shrike_engine_api::Recognition
//! [`Segment`]: shrike_engine_api::Segment

use shrike_engine_api::Recognition;

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
        if self.vector_worthy(&recognition.text) {
            GateOutcome::LexicalAndVector
        } else {
            GateOutcome::Lexical
        }
    }

    /// Whether recognized text is substantive enough to mint a vector — the
    /// rule `judge` applies, owned here so the re-judge from *stored* text
    /// (confidence already gated at ingest) asks the gate instead of
    /// re-deriving the char-count threshold (#382).
    pub fn vector_worthy(&self, text: &str) -> bool {
        text.trim().chars().count() >= self.min_chars_vector
    }
}

/// What one `recognize_pending` sweep did — the typed contract for the host's
/// background driver (#391). Counts ride only the variant where a batch was
/// actually sent; `Ran { recognized: 0, .. }` is the no-progress signal (an
/// unreadable window) the harness stops on.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SweepReport {
    /// No recognizer attached.
    Unavailable,
    /// Nothing pending — the sweep had no work.
    Idle,
    /// A batch was sent: `recognized` items reached the engine, `stored`
    /// cleared the gate, `remaining` are left beyond this window.
    Ran {
        recognized: usize,
        stored: usize,
        remaining: usize,
    },
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
    fn vector_worthy_matches_the_judge_threshold() {
        let gate = RecognitionGate::default();
        // Same rule judge applies: trimmed char count vs min_chars_vector.
        assert!(!gate.vector_worthy("Mitochondrion"));
        assert!(gate.vector_worthy("The inner membrane hosts the electron transport chain"));
        // Whitespace padding doesn't count as substance.
        assert!(!gate.vector_worthy(&format!("{}short{}", " ".repeat(30), " ".repeat(30))));
    }

    #[test]
    fn sweep_report_wire_shape() {
        // The host's driver parses this wire — pin it at the type's home so
        // a serde attribute change can't silently reshape it.
        let to = |r: &SweepReport| serde_json::to_string(r).unwrap();
        assert_eq!(to(&SweepReport::Unavailable), r#"{"status":"unavailable"}"#);
        assert_eq!(to(&SweepReport::Idle), r#"{"status":"idle"}"#);
        assert_eq!(
            to(&SweepReport::Ran {
                recognized: 2,
                stored: 1,
                remaining: 3
            }),
            r#"{"status":"ran","recognized":2,"stored":1,"remaining":3}"#
        );
    }
}
