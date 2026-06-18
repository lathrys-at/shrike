//! Empirical batch-safety self-check for embedding engines (mirrors
//! `shrike/embed_batching.py` — shared engine policy, so every
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

use crate::{EmbedImages, EmbedText, MediaItem};
use shrike_error::{NativeError, NativeResult};

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

// ── vision probe set ─────────────────────────────────────────────────────────
//
// The image analogue of `BATCH_PROBE_TEXTS`. int8 vision graphs drift under
// batching for the same reason text ones do — a per-tensor activation scale
// over the whole batch tensor — and that drift is driven by *activation
// magnitude*, not image count or size. So the set spans content that drives a
// wide pixel-activation range: flat extremes (all-black, all-white, a calm
// mid-grey anchor), saturated primaries, sharp high-frequency structure
// (checkerboard, stripes), smooth gradients, and per-channel-distinct fills.
// A calm anchor batched against a saturated/high-frequency neighbour is where
// int8 vision drift is maximized — the image mirror of the text set's "calm
// anchor + spiky batch-mate" rule. An fp graph has no activation quant, so its
// batched-vs-serial drift is exactly 0 regardless of content, exactly as for
// text.
//
// Each entry is a tiny RGB image built at probe time (no fixture files),
// encoded as an uncompressed BMP in pure Rust so this leaf crate gains no
// image-encode dependency. The set is heterogeneous and its *length* is the
// vision batching ceiling — sized to match the text set (64) so the CLIP
// engine's `min(text, vision)` never lowers a uniform-safe pair's batch (both
// probe to 64), collapsing to 1 only when a path is genuinely batch-variant.

/// How many synthetic probe images to generate (the vision batching ceiling).
/// Matched to the text set size (64) so a uniform-safe CLIP pair probes both
/// paths to the *same* ceiling — `min(text, vision)` then never lowers a safe
/// pair's batch, and collapses to 1 only when one path is genuinely variant.
/// The first ~15 indices are distinct hand-designed extremes; the rest are
/// hashed-deterministic broadband speckle (the `_` branch of `probe_pixel`),
/// each a distinct image, so the set stays heterogeneous to the full length.
const IMAGE_PROBE_COUNT: usize = 64;

/// Side length of each synthetic probe image (small — the probe only needs
/// content variety, not resolution; the engine resizes/crops anyway).
const IMAGE_PROBE_SIDE: u32 = 32;

/// Per-pixel RGB for probe image `idx` at `(x, y)` — varied content spanning a
/// wide activation range (see the module note above on the vision set).
fn probe_pixel(idx: usize, x: u32, y: u32) -> [u8; 3] {
    let s = IMAGE_PROBE_SIDE;
    match idx {
        0 => [0, 0, 0],       // flat black (min activation)
        1 => [255, 255, 255], // flat white (max activation)
        2 => [128, 128, 128], // calm mid-grey anchor
        3 => [255, 0, 0],     // saturated red
        4 => [0, 255, 0],     // saturated green
        5 => [0, 0, 255],     // saturated blue
        // High-frequency checkerboard (alternating extremes — spiky).
        6 => {
            if (x + y).is_multiple_of(2) {
                [255, 255, 255]
            } else {
                [0, 0, 0]
            }
        }
        // Fine vertical stripes.
        7 => {
            if x.is_multiple_of(2) {
                [255, 255, 0]
            } else {
                [0, 0, 255]
            }
        }
        // Horizontal gradient (smooth ramp across the activation range).
        8 => {
            let v = (x * 255 / s.max(1)) as u8;
            [v, v, v]
        }
        // Vertical gradient.
        9 => {
            let v = (y * 255 / s.max(1)) as u8;
            [v, v, v]
        }
        // Diagonal gradient, per-channel-distinct (exercises each plane apart).
        10 => {
            let v = ((x + y) * 255 / (2 * s).max(1)) as u8;
            [v, 255u8.saturating_sub(v), v / 2]
        }
        // Quadrant split (sharp regional contrast).
        11 => {
            let right = x >= s / 2;
            let bottom = y >= s / 2;
            match (right, bottom) {
                (false, false) => [255, 0, 0],
                (true, false) => [0, 255, 0],
                (false, true) => [0, 0, 255],
                (true, true) => [255, 255, 255],
            }
        }
        // Saturated magenta / cyan / yellow (secondary extremes).
        12 => [255, 0, 255],
        13 => [0, 255, 255],
        14 => [255, 255, 0],
        // Pseudo-random speckle (broadband, hashed-deterministic).
        _ => {
            let h = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503) ^ (idx as u32))
                .wrapping_mul(2246822519);
            [
                (h & 0xff) as u8,
                ((h >> 8) & 0xff) as u8,
                ((h >> 16) & 0xff) as u8,
            ]
        }
    }
}

/// One synthetic probe image, encoded as an uncompressed 24-bit BMP
/// [`MediaItem`]. BMP is emitted in pure Rust (a fixed 54-byte header + raw
/// BGR rows) so this contract crate stays a LEAF — no `image`/`png`
/// dependency to *encode* the probe set. Engines decode it via the `image`
/// crate, which reads 24-bit BMP losslessly, so the pixels the engine sees
/// are exactly `probe_pixel`'s.
fn probe_image(idx: usize) -> MediaItem {
    MediaItem::from_named(&format!("probe-{idx}.bmp"), encode_bmp(idx))
}

/// Uncompressed 24-bit BMP of `IMAGE_PROBE_SIDE`² pixels from [`probe_pixel`].
/// Bottom-up BGR rows padded to a 4-byte boundary (the BMP format) — minimal,
/// dependency-free, and round-trips losslessly through any BMP decoder.
fn encode_bmp(idx: usize) -> Vec<u8> {
    let s = IMAGE_PROBE_SIDE;
    let row_bytes = (s * 3) as usize;
    let padding = (4 - row_bytes % 4) % 4;
    let stride = row_bytes + padding;
    let pixel_data = stride * s as usize;
    let file_size = 54 + pixel_data;

    let mut out = Vec::with_capacity(file_size);
    // -- BITMAPFILEHEADER (14 bytes) --
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    out.extend_from_slice(&54u32.to_le_bytes()); // pixel-data offset
                                                 // -- BITMAPINFOHEADER (40 bytes) --
    out.extend_from_slice(&40u32.to_le_bytes()); // header size
    out.extend_from_slice(&(s as i32).to_le_bytes()); // width
    out.extend_from_slice(&(s as i32).to_le_bytes()); // height (+ = bottom-up)
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&24u16.to_le_bytes()); // bits per pixel
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB (no compression)
    out.extend_from_slice(&(pixel_data as u32).to_le_bytes()); // image size
    out.extend_from_slice(&2835i32.to_le_bytes()); // x ppm (~72 dpi)
    out.extend_from_slice(&2835i32.to_le_bytes()); // y ppm
    out.extend_from_slice(&0u32.to_le_bytes()); // palette colours
    out.extend_from_slice(&0u32.to_le_bytes()); // important colours
                                                // -- pixel rows, bottom-up, BGR + per-row padding --
    for y in (0..s).rev() {
        for x in 0..s {
            let [r, g, b] = probe_pixel(idx, x, y);
            out.extend_from_slice(&[b, g, r]);
        }
        out.extend(std::iter::repeat_n(0u8, padding));
    }
    out
}

/// The synthetic vision probe set (BMP-encoded [`MediaItem`]s).
fn owned_images() -> Vec<MediaItem> {
    (0..IMAGE_PROBE_COUNT).map(probe_image).collect()
}

/// The synthetic vision probe set as raw encoded bytes — the canonical set the
/// Python host sources (mirroring `BATCH_PROBE_TEXTS`), so native and Python
/// hosts probe the *same* images. One BMP per entry; decoders read them
/// losslessly.
pub fn batch_probe_images() -> Vec<Vec<u8>> {
    (0..IMAGE_PROBE_COUNT).map(encode_bmp).collect()
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
///
/// # Errors
///
/// Returns an `unavailable` [`NativeError`] if the serial reference embed fails
/// on every one of [`PROBE_ATTEMPTS`] attempts.
pub fn probe_max_safe_batch<E: EmbedText + ?Sized>(engine: &E) -> NativeResult<usize> {
    probe_with(engine, BATCH_DRIFT_TOL, PROBE_ATTEMPTS)
}

/// [`probe_max_safe_batch`] with explicit tolerance/attempts (tests).
///
/// # Errors
///
/// Returns an `unavailable` [`NativeError`] if the serial reference embed fails
/// on every one of `attempts` attempts.
pub fn probe_with<E: EmbedText + ?Sized>(
    engine: &E,
    tol: f64,
    attempts: usize,
) -> NativeResult<usize> {
    let texts = owned_texts();
    probe_chunks(&texts, |items| engine.embed_chunk(items), tol, attempts)
}

/// Max-abs serial-vs-batched drift over the probe set — for sensitivity
/// pins (the >10×-tolerance assertion against the real int8 fixtures).
///
/// # Errors
///
/// Returns the engine's error if either the serial reference or the batched
/// embed fails.
pub fn max_probe_drift<E: EmbedText + ?Sized>(engine: &E) -> NativeResult<f64> {
    let texts = owned_texts();
    let reference = serial_reference(&texts, |items| engine.embed_chunk(items))?;
    let batched = engine.embed_chunk(&texts)?;
    Ok(drift(&reference, &batched))
}

// ── the vision probe ─────────────────────────────────────────────────────────

/// The image analogue of [`probe_max_safe_batch`]: the batch size proven safe
/// for the *vision* path, or 1 (embed serially). Same tolerance discipline,
/// same two-failure-modes split — over the synthetic `owned_images` set.
///
/// `_resolve_files` only ever loads a matched-precision text+vision pair, so a
/// uniform export's text probe already predicts the vision path. This guards
/// the one case it can't: a **hand-assembled mixed-precision** pair (fp text +
/// int8 vision a user dropped on disk), where the vision graph batches
/// non-deterministically and the text probe would wrongly clear it. A host
/// runs both and takes `min(text_safe, vision_safe)`.
///
/// # Errors
///
/// Returns an `unavailable` [`NativeError`] if the serial reference embed fails
/// on every one of [`PROBE_ATTEMPTS`] attempts.
pub fn probe_image_max_safe_batch<E: EmbedImages + ?Sized>(engine: &E) -> NativeResult<usize> {
    probe_image_with(engine, BATCH_DRIFT_TOL, PROBE_ATTEMPTS)
}

/// [`probe_image_max_safe_batch`] with explicit tolerance/attempts (tests).
///
/// # Errors
///
/// Returns an `unavailable` [`NativeError`] if the serial reference embed fails
/// on every one of `attempts` attempts.
pub fn probe_image_with<E: EmbedImages + ?Sized>(
    engine: &E,
    tol: f64,
    attempts: usize,
) -> NativeResult<usize> {
    let images = owned_images();
    probe_chunks(
        &images,
        |items| engine.embed_image_chunk(items),
        tol,
        attempts,
    )
}

/// Max-abs serial-vs-batched drift over the vision probe set — the image
/// mirror of [`max_probe_drift`], for the vision sensitivity pin.
///
/// # Errors
///
/// Returns the engine's error if either the serial reference or the batched
/// embed fails.
pub fn max_probe_image_drift<E: EmbedImages + ?Sized>(engine: &E) -> NativeResult<f64> {
    let images = owned_images();
    let reference = serial_reference(&images, |items| engine.embed_image_chunk(items))?;
    let batched = engine.embed_image_chunk(&images)?;
    Ok(drift(&reference, &batched))
}

// ── the shared comparison core (text + image) ────────────────────────────────

/// The probe over any chunk-embed closure: embed `items` serially (the
/// reference, retried up to `attempts` times — persistent failure is an
/// `unavailable` error) and all in one batch, comparing within `tol`. Match →
/// the set size (safe to batch that far); batch-only failure or drift → 1.
/// The text and image entrypoints differ only in their item type and closure;
/// this core is identical, so both paths get the same retry/degrade discipline.
fn probe_chunks<T, F>(items: &[T], mut embed: F, tol: f64, attempts: usize) -> NativeResult<usize>
where
    F: FnMut(&[T]) -> NativeResult<Vec<Vec<f32>>>,
{
    let mut reference: Option<Vec<Vec<f32>>> = None;
    let mut last_err: Option<NativeError> = None;
    for _ in 0..attempts.max(1) {
        match serial_reference(items, &mut embed) {
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
    let Ok(batched) = embed(items) else {
        return Ok(1);
    };
    if drift(&reference, &batched) <= tol {
        Ok(items.len())
    } else {
        Ok(1)
    }
}

fn serial_reference<T, F>(items: &[T], mut embed: F) -> NativeResult<Vec<Vec<f32>>>
where
    F: FnMut(&[T]) -> NativeResult<Vec<Vec<f32>>>,
{
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let mut v = embed(std::slice::from_ref(item))?;
        let Some(first) = v.pop() else {
            return Err(NativeError::internal(
                "engine returned no vector for an item",
            ));
        };
        out.push(first);
    }
    Ok(out)
}

/// Max-abs element-wise difference, in f64 (mismatched shapes → infinite
/// drift, which correctly reads as batch-variant).
///
/// A `NaN` element on either side also reads as **infinite** drift: a
/// batch-variant model whose drift manifests as `NaN`/`inf` (the int8
/// magnitude extremes the spiked probe set is built to surface) must condemn
/// the model to serial, not slip through. `f64::max` *discards* a `NaN`
/// operand (it returns the non-`NaN` side), and `(finite - NaN).abs()` is
/// `NaN`, so a plain `max.max(diff)` would silently drop the `NaN` drift and
/// declare the model batch-safe — breaking the reconcile == rebuild invariant.
/// Treating any `NaN` as `f64::INFINITY` mirrors the mismatched-shape handling
/// above.
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
            if x.is_nan() || y.is_nan() {
                return f64::INFINITY;
            }
            max = max.max((f64::from(*x) - f64::from(*y)).abs());
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrike_error::NativeResult;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Hash-deterministic per-text vectors, optionally batch-variant.
    struct Toy {
        batch_variant: bool,
        /// Emit `NaN` for every element of a *batched* embed — the int8
        /// magnitude-extreme failure mode. Serial stays finite, so the model
        /// is batch-variant and must probe to serial.
        nan_batched: bool,
        fail_serial: bool,
        fail_batched: bool,
        calls: AtomicUsize,
    }

    impl Default for Toy {
        fn default() -> Self {
            Self {
                batch_variant: false,
                nan_batched: false,
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
            if self.nan_batched && texts.len() > 1 {
                return Ok(texts.iter().map(|_| vec![f32::NAN, 1.0]).collect());
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

    /// A model whose batched embed produces `NaN` (the int8 magnitude-extreme
    /// failure mode) is batch-**variant** and must probe to serial. Regression
    /// guard: a `max.max((finite - NaN).abs())` would swallow
    /// the `NaN` (Rust `f64::max` returns the non-`NaN` operand), so the model
    /// would be declared safe (`safe_batch = 64`) → a note's vector becomes
    /// batch-dependent → reconcile (small chunks) ≠ rebuild (64-chunks).
    #[test]
    fn nan_batched_engine_probes_to_serial() {
        // The floating-point facts a naive max would trip on: the diff
        // against a NaN is NaN, and `f64::max` discards a NaN operand — so the
        // running `max` would stay 0.0 (≤ BATCH_DRIFT_TOL, i.e. "safe"),
        // silently dropping the drift.
        assert!((1.0_f64 - f64::NAN).abs().is_nan());
        assert_eq!(0.0_f64.max(f64::NAN), 0.0);

        let toy = Toy {
            nan_batched: true,
            ..Toy::default()
        };
        assert_eq!(probe_max_safe_batch(&toy).unwrap(), 1);
        // The drift is now reported as infinite, not silently dropped to 0.
        let observed = max_probe_drift(&toy).unwrap();
        assert!(
            observed.is_infinite() && observed > 0.0,
            "expected +inf NaN drift, got {observed}"
        );
        assert!(observed > BATCH_DRIFT_TOL);
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

    // ── vision probe ─────────────────────────────────────────────────────

    /// The image analogue of [`Toy`]: per-image vectors keyed by byte length
    /// (the probe images differ, so serial vectors differ), optionally
    /// batch-variant. The int8 vision drift the real fix guards against is
    /// pinned here by the test-double pattern, exactly like the text probe.
    #[derive(Default)]
    struct ImageToy {
        batch_variant: bool,
        fail_serial: bool,
        fail_batched: bool,
    }

    impl EmbedImages for ImageToy {
        fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
            if images.len() == 1 && self.fail_serial {
                return Err(NativeError::unavailable("serial down"));
            }
            if images.len() > 1 && self.fail_batched {
                return Err(NativeError::invalid_input("fixed batch-1 vision graph"));
            }
            // int8-style activation drift: a batch-mate shifts every element.
            let shift = if self.batch_variant && images.len() > 1 {
                0.05
            } else {
                0.0
            };
            Ok(images
                .iter()
                .map(|im| vec![im.bytes.len() as f32 / 100_000.0 + shift, 1.0])
                .collect())
        }
    }

    #[test]
    fn image_probe_set_matches_the_text_ceiling() {
        // Sized to the text set so a uniform-safe pair's min(text, vision)
        // doesn't lower the batch.
        assert_eq!(IMAGE_PROBE_COUNT, BATCH_PROBE_TEXTS.len());
        assert_eq!(batch_probe_images().len(), IMAGE_PROBE_COUNT);
    }

    #[test]
    fn image_probe_set_is_nonempty_and_decodable() {
        let images = owned_images();
        assert_eq!(images.len(), IMAGE_PROBE_COUNT);
        // Every probe image is a real, decodable bitmap with the expected
        // dimensions — the engine's decoder must be able to read it.
        for (i, item) in images.iter().enumerate() {
            let img = image::load_from_memory(&item.bytes)
                .unwrap_or_else(|e| panic!("probe image {i} did not decode: {e}"));
            assert_eq!(
                (img.width(), img.height()),
                (IMAGE_PROBE_SIDE, IMAGE_PROBE_SIDE),
                "probe image {i} dimensions"
            );
        }
        // The set is heterogeneous — every image's content is distinct
        // (including the hashed-speckle tail), so each contributes its own
        // activation profile and the probe maximizes batch variance.
        let mut seen = std::collections::HashSet::new();
        for item in &images {
            assert!(
                seen.insert(item.bytes.clone()),
                "probe images must all differ"
            );
        }
    }

    #[test]
    fn safe_vision_engine_probes_to_the_full_set() {
        let toy = ImageToy::default();
        assert_eq!(probe_image_max_safe_batch(&toy).unwrap(), IMAGE_PROBE_COUNT);
    }

    #[test]
    fn variant_vision_engine_probes_to_serial() {
        let toy = ImageToy {
            batch_variant: true,
            ..Default::default()
        };
        assert_eq!(probe_image_max_safe_batch(&toy).unwrap(), 1);
        assert!(max_probe_image_drift(&toy).unwrap() > BATCH_DRIFT_TOL * 10.0);
    }

    #[test]
    fn persistent_vision_serial_failure_is_an_error() {
        let toy = ImageToy {
            fail_serial: true,
            ..Default::default()
        };
        let err = probe_image_with(&toy, BATCH_DRIFT_TOL, 2).unwrap_err();
        assert!(err.to_string().contains("after 2 attempt(s)"), "{err}");
    }

    #[test]
    fn vision_batch_only_failure_degrades_to_serial() {
        let toy = ImageToy {
            fail_batched: true,
            ..Default::default()
        };
        assert_eq!(probe_image_max_safe_batch(&toy).unwrap(), 1);
    }
}
