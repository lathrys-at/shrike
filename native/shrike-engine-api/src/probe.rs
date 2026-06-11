//! Empirical batch-safety self-check for embedding engines (#342 P4, the
//! Rust port of `shrike/embed_batching.py` — shared engine policy, so every
//! host composes batch policy the same way).
//!
//! Some engines produce a *different* vector for a text depending on what
//! else is in its batch. The clearest case is dynamically-quantized int8
//! ONNX models: the runtime computes a per-tensor activation scale over the
//! whole batch tensor, so a batch-mate's content shifts every element
//! (~0.06). That breaks the index's invariant that a note's vector is a pure
//! function of its text — which is what makes a reconcile's end state
//! identical to a full rebuild. Non-quantized models (fp32/fp16) and
//! llama-server batch deterministically.
//!
//! Rather than guess from a model's quantization scheme, the host probes the
//! loaded model: embed a fixed set of varied texts **serially** (the
//! reference) and **all in one batch** — the largest, most heterogeneous
//! batch, maximizing any batch-variance — and compare. Match within the
//! tolerance → safe to batch *up to the probe-set size* (which `WithPolicy`
//! then carries as `safe_batch`); mismatch → 1 (serial).
//!
//! Two deliberate choices make the check trustworthy rather than wishful:
//!
//! - **The probe set is "spiked" for activation magnitude, not just
//!   length**: calm anchors mixed with deliberately spiky inputs (long,
//!   numeric/hex/code, symbol soup, degenerate repeated tokens,
//!   mixed-script/emoji, ALL CAPS) — int8 drift on a calm text is maximized
//!   by a spiky batch-mate. An fp model has no activation quant, so spiking
//!   only raises sensitivity to variant models, never false-positives a safe
//!   one. The set's sensitivity stays pinned by the >10×-tolerance drift
//!   test against the real int8 fixtures (`test_onnx_models.py`).
//! - **We probe the batch size we actually use**: one batch of all probe
//!   texts, and the host never batches larger — "proven safe" and "what we
//!   batch" are the same size.

use crate::EmbedText;
use shrike_ffi::{NativeError, NativeResult};

/// Tolerance for "batched == serial". Sits well above float-reduction noise
/// (llama-server ~4e-5, fp32 ONNX exactly 0) and far below dynamic-int8
/// batch drift (~0.06), so it cleanly separates safe from variant.
pub const BATCH_DRIFT_TOL: f64 = 1e-3;

/// How many times to (re)run the serial reference before giving up — a
/// single transient failure shouldn't condemn a session to serial.
pub const PROBE_ATTEMPTS: usize = 3;

/// Probe texts, spiked for activation magnitude (see module docs). The
/// content is irrelevant to the result — only whether a text's vector
/// changes with its batch-mates — but the spread of magnitudes is what makes
/// a variant model actually diverge. The set's *length* (64, the index's
/// `BATCH_SIZE` chunk) is also the batching ceiling.
pub const BATCH_PROBE_TEXTS: &[&str] = &[
    // Calm anchors (real-note-shaped, low activation range).
    "mitochondria are the powerhouse of the cell",
    "Spaced repetition strengthens long-term memory through timed review.",
    "Newton's second law relates force, mass, and acceleration.",
    "An integral accumulates a quantity over an interval.",
    "DNA encodes genetic information in sequences of nucleotides.",
    "Supply and demand determine prices in a competitive market.",
    "Mitochondrial DNA is inherited maternally in most animals.",
    "The boiling point of water at sea level is 100 degrees Celsius.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog repeatedly.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    // Long (a wide activation profile over many tokens).
    "the derivative measures the instantaneous rate of change of a function with respect to \
     its variable, while the definite integral accumulates the signed area under a curve over \
     an interval, and together by the fundamental theorem of calculus they form inverse \
     operations that underpin much of classical analysis and its applications in physics, \
     engineering, economics, and statistics across countless practical and theoretical settings",
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt \
     ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation",
    // Numbers / hex / code (outlier-prone token embeddings).
    "0xDEADBEEF 1234567890 SELECT * FROM t WHERE id=42 AND ratio=3.14159265;",
    "SELECT id, name FROM users WHERE created_at > '2020-01-01' ORDER BY name DESC;",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
    "kHz MHz GHz THz 3.0e8 m/s 6.022e23 1.602e-19 9.81 273.15 -40",
    // Markup / structured.
    "<html><body><h1>Title</h1><p>paragraph &amp; entity</p></body></html>",
    "user@example.com +1-555-0123 https://test.org/page#anchor?q=1&r=2",
    "https://example.com/path?q=1&r=2#frag 5f4dcc3b5aa765d61d8327deb882cf99",
    // Symbol / math soup.
    "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./",
    "Σ Δ Ω α β γ ∫ ∂ ∇ √ ∞ ≈ ≠ ≤ ≥ ± × ÷ → ⇒ ∈ ∀ ∃",
    // Degenerate / repeated tokens (a spike with a tiny activation range of its own).
    "the the the the the the the the the the the the",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    // Mixed script / emoji (rare tokens, large embedding norms).
    "café 日本語 Привет мир 🧠🔬🧬 naïve résumé Größe",
    "🎉🎊🥳 congratulations on completing the course 🏆✨🚀",
    "ПриветПривет こんにちは こんにちは 안녕하세요 你好世界",
    // ALL CAPS.
    "URGENT WARNING SYSTEM FAILURE IMMINENT EVACUATE THE BUILDING NOW",
    "ALL CAPS SHORT",
    // Short.
    "x",
    // --- second half (to reach the 64-text / batch-64 ceiling) ---
    // More calm anchors.
    "The speed of sound in dry air is about 343 meters per second.",
    "Shakespeare wrote thirty-seven plays and over one hundred fifty sonnets.",
    "Water is composed of two hydrogen atoms and one oxygen atom.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "A leap year occurs every four years to keep the calendar aligned.",
    "Insulin regulates the level of glucose in the bloodstream.",
    "The Great Wall of China stretches thousands of kilometers.",
    "Gravity causes objects to accelerate toward the center of the Earth.",
    "The human heart pumps roughly five liters of blood per minute.",
    "Tectonic plates drift slowly across the planet's molten mantle.",
    "Vaccines train the immune system to recognize specific pathogens.",
    "Electrons occupy discrete energy levels around an atomic nucleus.",
    "The periodic table organizes elements by their atomic number.",
    // More numbers / code.
    "def f(x): return x**2 + 3*x - 7  # a simple quadratic",
    "git commit -m 'fix: off-by-one in loop bound' && git push origin main",
    "3.141592653589793 2.718281828459045 1.618033988749895 0.5772156649",
    "IPv4 192.168.0.1 IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 :8080",
    // More symbol / math / structured.
    "∮ E·dl = -dΦ/dt    ∇×B = μ₀J + μ₀ε₀ ∂E/∂t",
    "{ \"key\": [1, 2, 3], \"nested\": { \"a\": true, \"b\": null } }",
    // More mixed script / emoji.
    "中文 العربية हिन्दी ไทย Ελληνικά עברית 한국어 русский язык",
    "😀😃😄😁😆😅😂🤣☺️😊😇🙂🙃😉😌😍🥰😘🤗",
    "Zürich Köln München São Paulo Bogotá Reykjavík İstanbul",
    // More long.
    "in computer science a hash table implements an associative array abstract data type, a \
     structure that maps keys to values using a hash function to compute an index into an array \
     of buckets from which the desired value can be found, offering average constant-time \
     complexity for insertion, deletion, and lookup under a good hash distribution and load",
    "a regular expression is a sequence of characters that specifies a search pattern in text, \
     used by string-searching algorithms for find and replace operations or input validation, \
     and supported with varying syntax across editors, command-line tools, and programming \
     languages from grep and sed to Perl, Python, and the Rust regex crate among many others",
    // More ALL CAPS.
    "BREAKING NEWS MARKETS RALLY AS INFLATION COOLS FASTER THAN EXPECTED",
    "TODO FIXME XXX HACK NOTE WARNING DEPRECATED REVIEW",
    // More degenerate / repeated.
    "na na na na na na na na na na na batman",
    "0000000000000000000000000000000000000000",
    // More short.
    "ok",
    "42",
    // Misc.
    "e = mc²   F = ma   PV = nRT   a² + b² = c²",
    "colorless green ideas sleep furiously",
];

fn owned_texts() -> Vec<String> {
    BATCH_PROBE_TEXTS.iter().map(|s| s.to_string()).collect()
}

/// The batch size proven safe (the probe-set size) or 1 (embed serially).
///
/// Embeds each probe text **alone** (the serial reference), then all in
/// **one batch**, and compares max-abs per element. The two failure modes
/// stay distinct: the serial reference is what the model must be able to do
/// at all — retried up to [`PROBE_ATTEMPTS`] times, persistent failure is an
/// error (`unavailable` — the host fails loud, e.g. a model needing an input
/// the engine doesn't supply); a **batch-only** failure (e.g. a graph fixed
/// to batch size 1) is *not* an error — it returns 1.
pub fn probe_max_safe_batch<E: EmbedText + ?Sized>(engine: &E) -> NativeResult<usize> {
    probe_with(engine, BATCH_DRIFT_TOL, PROBE_ATTEMPTS)
}

/// [`probe_max_safe_batch`] with explicit tolerance/attempts (tests).
pub fn probe_with<E: EmbedText + ?Sized>(
    engine: &E,
    tol: f64,
    attempts: usize,
) -> NativeResult<usize> {
    let texts = owned_texts();
    let mut reference: Option<Vec<Vec<f32>>> = None;
    let mut last_err: Option<NativeError> = None;
    for _ in 0..attempts.max(1) {
        match serial_reference(engine, &texts) {
            Ok(vectors) => {
                reference = Some(vectors);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let Some(reference) = reference else {
        let detail = last_err.map(|e| e.to_string()).unwrap_or_default();
        return Err(NativeError::unavailable(format!(
            "serial embedding failed after {attempts} attempt(s): {detail}"
        )));
    };
    // The model can embed serially. Does it also batch deterministically? A
    // batch-only failure degrades to serial rather than erroring.
    let Ok(batched) = engine.embed_chunk(&texts) else {
        return Ok(1);
    };
    if drift(&reference, &batched) <= tol {
        Ok(texts.len())
    } else {
        Ok(1)
    }
}

/// Max-abs serial-vs-batched drift over the probe set — for sensitivity
/// pins (the >10×-tolerance assertion against the real int8 fixtures).
pub fn max_probe_drift<E: EmbedText + ?Sized>(engine: &E) -> NativeResult<f64> {
    let texts = owned_texts();
    let reference = serial_reference(engine, &texts)?;
    let batched = engine.embed_chunk(&texts)?;
    Ok(drift(&reference, &batched))
}

fn serial_reference<E: EmbedText + ?Sized>(
    engine: &E,
    texts: &[String],
) -> NativeResult<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(texts.len());
    for t in texts {
        let mut v = engine.embed_chunk(std::slice::from_ref(t))?;
        let Some(first) = v.pop() else {
            return Err(NativeError::internal(
                "engine returned no vector for a text",
            ));
        };
        out.push(first);
    }
    Ok(out)
}

/// Max-abs element-wise difference, in f64 (mismatched shapes → infinite
/// drift, which correctly reads as batch-variant).
fn drift(reference: &[Vec<f32>], batched: &[Vec<f32>]) -> f64 {
    if reference.len() != batched.len() {
        return f64::INFINITY;
    }
    let mut max = 0.0f64;
    for (r, b) in reference.iter().zip(batched) {
        if r.len() != b.len() {
            return f64::INFINITY;
        }
        for (x, y) in r.iter().zip(b) {
            max = max.max((f64::from(*x) - f64::from(*y)).abs());
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrike_ffi::NativeResult;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Hash-deterministic per-text vectors, optionally batch-variant.
    struct Toy {
        batch_variant: bool,
        fail_serial: bool,
        fail_batched: bool,
        calls: AtomicUsize,
    }

    impl Default for Toy {
        fn default() -> Self {
            Self {
                batch_variant: false,
                fail_serial: false,
                fail_batched: false,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl EmbedText for Toy {
        fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if texts.len() == 1 && self.fail_serial {
                return Err(NativeError::unavailable("serial down"));
            }
            if texts.len() > 1 && self.fail_batched {
                return Err(NativeError::invalid_input("fixed batch-1 graph"));
            }
            let shift = if self.batch_variant && texts.len() > 1 {
                0.05 // dynamic-quant-style drift, well past the tolerance
            } else {
                0.0
            };
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32 / 1000.0 + shift, 1.0])
                .collect())
        }
    }

    #[test]
    fn probe_set_is_the_documented_ceiling() {
        assert_eq!(BATCH_PROBE_TEXTS.len(), 64);
    }

    #[test]
    fn safe_engine_probes_to_the_full_set() {
        let toy = Toy::default();
        assert_eq!(probe_max_safe_batch(&toy).unwrap(), BATCH_PROBE_TEXTS.len());
    }

    #[test]
    fn variant_engine_probes_to_serial() {
        let toy = Toy {
            batch_variant: true,
            ..Toy::default()
        };
        assert_eq!(probe_max_safe_batch(&toy).unwrap(), 1);
        assert!(max_probe_drift(&toy).unwrap() > BATCH_DRIFT_TOL * 10.0);
    }

    #[test]
    fn persistent_serial_failure_is_an_error() {
        let toy = Toy {
            fail_serial: true,
            ..Toy::default()
        };
        let err = probe_with(&toy, BATCH_DRIFT_TOL, 2).unwrap_err();
        assert!(err.to_string().contains("after 2 attempt(s)"), "{err}");
    }

    #[test]
    fn batch_only_failure_degrades_to_serial() {
        let toy = Toy {
            fail_batched: true,
            ..Toy::default()
        };
        assert_eq!(probe_max_safe_batch(&toy).unwrap(), 1);
    }
}
