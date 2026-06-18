//! Generic image preprocessing (#707): the byte → pixels → CHW-tensor pipeline
//! the CLIP vision path needs, extracted out of `shrike-embed/clip.rs` so it is
//! reusable, independently testable, and free of any engine/kernel coupling.
//!
//! The pipeline mirrors the Python CLIP `_preprocess` (PIL) it replaced: decode
//! → resize shortest edge → center-crop → rescale to `[0,1]` → per-channel
//! normalize → channel-major (CHW) f32. The `image` crate's Catmull-Rom bicubic
//! stands in for PIL bicubic (a different cubic coefficient), so image vectors
//! are semantically equivalent but not bit-identical to the Python engine's —
//! the consuming engine namespaces its fingerprint (`clip-rs:`) accordingly and
//! folds in [`IMAGE_PREP_VERSION_RS`].
//!
//! # Acceleration
//!
//! **`accel`** (default): the normalize/CHW transform runs as contiguous
//! per-channel plane passes with the per-channel affine math hoisted to a fused
//! multiply-add (`p·(1/(255·std)) + (−mean/std)`), which the compiler
//! autovectorizes — the old strided HWC→CHW scatter did ~`3·crop²` scalar
//! divides per image (~150k at 224²). It is the *same* affine math as the scalar
//! reference, so it matches within the golden image-prep tolerance; off, the
//! transform degrades to a plain scalar loop (minimal-core). The resize stays
//! `image`'s scalar Catmull-Rom throughout.
//!
//! **A change to the normalize math that cannot match the scalar reference
//! within the golden tolerance must bump [`IMAGE_PREP_VERSION_RS`]** — which
//! invalidates every stored image vector, so it is never a free perf win. (A
//! SIMD resize backend was prototyped but deferred: `fast_image_resize`'s
//! Catmull-Rom differs from `image`'s by ~0.078 on a real downscale, far past
//! the tolerance, so it is genuinely vector-affecting and belongs behind a
//! consumer-fingerprint mechanism, not this PR — see #707's follow-up.)

use image::imageops::FilterType;
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};

mod bmp;
pub use bmp::encode_bmp;

/// Bump when this pipeline's pixel math changes — folded into the consuming
/// engine's fingerprint so a changed pixel space invalidates stored vectors
/// (the Rust counterpart of the Python engine's `IMAGE_PREP_VERSION`). An
/// accelerated path that cannot match the scalar reference within the golden
/// tolerance is a pixel-math change and MUST bump this.
pub const IMAGE_PREP_VERSION_RS: u32 = 1;

/// CLIP-style preprocessing parameters (the preprocessor config's scalars).
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessConfig {
    /// Shortest-edge resize target (preprocessor "size").
    pub resize: u32,
    /// Center-crop size (preprocessor "crop_size").
    pub crop: u32,
    /// Per-channel mean (rescaled-space, 3 channels).
    pub mean: [f32; 3],
    /// Per-channel std (3 channels).
    pub std: [f32; 3],
}

impl PreprocessConfig {
    /// Build from slices (the shape the FFI hands across), validating arity.
    pub fn from_slices(resize: u32, crop: u32, mean: &[f32], std: &[f32]) -> NativeResult<Self> {
        if mean.len() != 3 || std.len() != 3 {
            return Err(NativeError::invalid_input(
                "image_mean/image_std must each have 3 channels",
            ));
        }
        Ok(Self {
            resize,
            crop,
            mean: [mean[0], mean[1], mean[2]],
            std: [std[0], std[1], std[2]],
        })
    }
}

/// One image's bytes → CHW f32 (length `3·crop²`): decode, resize shortest edge,
/// center-crop, rescale to `[0,1]`, per-channel normalize, channel-major layout.
pub fn preprocess_to_chw(bytes: &[u8], cfg: &PreprocessConfig) -> NativeResult<Vec<f32>> {
    let c = cfg.crop as usize;
    let mut out = Vec::with_capacity(3 * c * c);
    preprocess_into_chw(&mut out, bytes, cfg)?;
    Ok(out)
}

/// Preprocess each image and **append** its CHW plane to `out` in order — the
/// flat `[N, 3, crop, crop]` tensor the vision graph feeds on. `out` is appended
/// to (not cleared), so a caller may pre-reserve `images.len()·3·crop²`.
pub fn preprocess_batch_into(
    out: &mut Vec<f32>,
    images: &[&[u8]],
    cfg: &PreprocessConfig,
) -> NativeResult<()> {
    let c = cfg.crop as usize;
    out.reserve(images.len() * 3 * c * c);
    for bytes in images {
        preprocess_into_chw(out, bytes, cfg)?;
    }
    Ok(())
}

/// Shortest-edge resize target: scale so `min(w, h) == resize`, the other edge
/// rounded (round-half-away-from-zero, the f64 `round`), floored at 1.
fn resize_dims(w: u32, h: u32, resize: u32) -> (u32, u32) {
    let scale = resize as f64 / w.min(h) as f64;
    (
        (w as f64 * scale).round().max(1.0) as u32,
        (h as f64 * scale).round().max(1.0) as u32,
    )
}

/// Decode → resize shortest edge → center-crop, returning the cropped RGB image.
/// The pixel stage shared by every normalize path.
fn decode_resize_crop(bytes: &[u8], cfg: &PreprocessConfig) -> NativeResult<image::RgbImage> {
    let img = image::load_from_memory(bytes)
        .context(ErrorKind::InvalidInput, "image decode failed")?
        .to_rgb8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(NativeError::invalid_input("empty image"));
    }
    let (nw, nh) = resize_dims(w, h, cfg.resize);
    let resized = resize_shortest_edge(&img, nw, nh)?;
    let c = cfg.crop;
    let left = (nw.saturating_sub(c)) / 2;
    let top = (nh.saturating_sub(c)) / 2;
    Ok(image::imageops::crop_imm(&resized, left, top, c, c).to_image())
}

/// Catmull-Rom resize to `(nw, nh)` via `image`'s bicubic — the scalar
/// resampler the golden tests pin (a SIMD backend that would change these
/// pixels is deferred to a consumer-fingerprinted follow-up; see the crate docs).
fn resize_shortest_edge(img: &image::RgbImage, nw: u32, nh: u32) -> NativeResult<image::RgbImage> {
    Ok(image::imageops::resize(img, nw, nh, FilterType::CatmullRom))
}

/// Preprocess one image and **append** its CHW plane (length `3·crop²`) to
/// `out` — the per-item unit [`preprocess_batch_into`] loops and the natural
/// call for a consumer assembling one flat tensor across a batch. The hot path:
/// the per-channel affine `(p/255 − mean)/std` is hoisted to FMA constants and
/// the strided HWC→CHW scatter is rewritten as three contiguous plane passes
/// (one per channel), each a tight loop the compiler autovectorizes.
pub fn preprocess_into_chw(
    out: &mut Vec<f32>,
    bytes: &[u8],
    cfg: &PreprocessConfig,
) -> NativeResult<()> {
    let cropped = decode_resize_crop(bytes, cfg)?;
    normalize_into_chw(out, &cropped, &cfg.mean, &cfg.std);
    Ok(())
}

/// The normalize + HWC→CHW transform, appended to `out`. Both the `accel` and
/// scalar builds compute the *same* per-channel affine value
/// `p·(1/(255·std)) + (−mean/std)`; they differ only in loop shape. `accel`
/// walks each channel plane contiguously (autovectorizable); the scalar
/// fallback is the plain reference for minimal-core. The reference math the
/// golden tests pin is `(p/255 − mean)/std`; the FMA form matches it within the
/// golden tolerance (no `IMAGE_PREP_VERSION_RS` bump — see the crate docs).
fn normalize_into_chw(
    out: &mut Vec<f32>,
    cropped: &image::RgbImage,
    mean: &[f32; 3],
    std: &[f32; 3],
) {
    let (w, h) = cropped.dimensions();
    let plane = (w * h) as usize;
    let pixels = cropped.as_raw(); // tightly-packed interleaved RGB, row-major
    let base = out.len();
    out.resize(base + 3 * plane, 0.0);

    #[cfg(feature = "accel")]
    for ch in 0..3usize {
        // Loop-invariant per-channel affine constants, hoisted out of the plane
        // pass: value = p·scale + bias, equal to (p/255 − mean)/std.
        let scale = 1.0f32 / (255.0 * std[ch]);
        let bias = -mean[ch] / std[ch];
        let dst = &mut out[base + ch * plane..base + (ch + 1) * plane];
        // Contiguous channel plane: gather this channel's byte from the strided
        // source, apply the affine, write sequentially. The destination write is
        // contiguous and the affine is branch-free FMA-shaped, so the compiler
        // autovectorizes it.
        for (i, slot) in dst.iter_mut().enumerate() {
            *slot = pixels[i * 3 + ch] as f32 * scale + bias;
        }
    }

    // The literal scalar reference for minimal-core (no autovectorization hint):
    // the exact `(p/255 − mean)/std`, strided HWC→CHW scatter. The `accel` path
    // above must match this within the golden tolerance (pinned by
    // `accel_matches_scalar_reference_within_tolerance`).
    #[cfg(not(feature = "accel"))]
    for ch in 0..3usize {
        for i in 0..plane {
            out[base + ch * plane + i] = (pixels[i * 3 + ch] as f32 / 255.0 - mean[ch]) / std[ch];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory PNG of an RGB image built per pixel — no fixture files.
    fn png_bytes(w: u32, h: u32, px: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(px(x, y)));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .expect("png encode");
        buf.into_inner()
    }

    fn cfg(resize: u32, crop: u32, mean: [f32; 3], std: [f32; 3]) -> PreprocessConfig {
        PreprocessConfig {
            resize,
            crop,
            mean,
            std,
        }
    }

    // Identity-size resizes below stay pixel-exact: at scale 1 the Catmull-Rom
    // kernel samples land on pixel centers (weight 1 at the knot, 0 at ±1/±2),
    // so the golden values are deterministic; tolerances cover f32↔u8 rounding.
    const TOL: f32 = 1e-3;

    #[test]
    fn solid_color_golden_chw_and_channel_normalization() {
        // 4×4 solid (255, 128, 0), no resize (shortest edge == resize), no
        // crop offset. Distinct per-channel mean/std prove the buffer is CHW
        // (channel-major planes) and that mean/std index by channel.
        let bytes = png_bytes(4, 4, |_, _| [255, 128, 0]);
        let mean = [0.5f32, 0.25, 0.2];
        let std = [0.5f32, 0.25, 0.1];
        let chw = preprocess_to_chw(&bytes, &cfg(4, 4, mean, std)).unwrap();
        assert_eq!(chw.len(), 3 * 4 * 4);
        let expected = [
            (255.0 / 255.0 - mean[0]) / std[0], // 1.0
            (128.0 / 255.0 - mean[1]) / std[1], // ≈ 1.00784
            (0.0 / 255.0 - mean[2]) / std[2],   // -2.0
        ];
        for ch in 0..3 {
            for (i, v) in chw[ch * 16..(ch + 1) * 16].iter().enumerate() {
                assert!(
                    (v - expected[ch]).abs() < TOL,
                    "channel {ch} index {i}: got {v}, want {}",
                    expected[ch]
                );
            }
        }
    }

    #[test]
    fn center_crop_picks_middle_columns_of_wide_image() {
        // 8×4, left half black, right half white; shortest edge 4 == resize →
        // identity resize, crop 4 → left = (8-4)/2 = 2: columns 2..6, i.e.
        // two black then two white columns.
        let bytes = png_bytes(8, 4, |x, _| if x < 4 { [0, 0, 0] } else { [255, 255, 255] });
        let chw = preprocess_to_chw(&bytes, &cfg(4, 4, [0.0; 3], [1.0; 3])).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let want = if x < 2 { 0.0 } else { 1.0 };
                let got = chw[y * 4 + x]; // channel 0 plane
                assert!(
                    (got - want).abs() < TOL,
                    "({x},{y}): got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn center_crop_picks_middle_rows_of_tall_image() {
        // 4×8, top half black, bottom half white; crop top = (8-4)/2 = 2 →
        // rows 2..6: two black then two white rows.
        let bytes = png_bytes(4, 8, |_, y| if y < 4 { [0, 0, 0] } else { [255, 255, 255] });
        let chw = preprocess_to_chw(&bytes, &cfg(4, 4, [0.0; 3], [1.0; 3])).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let want = if y < 2 { 0.0 } else { 1.0 };
                let got = chw[y * 4 + x];
                assert!(
                    (got - want).abs() < TOL,
                    "({x},{y}): got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn center_crop_floors_odd_offset() {
        // 7×4, column x = x·30; crop 4 → left = (7-4)/2 = 1 (1.5 floored):
        // cropped column x carries source column x+1.
        let v = |x: u32| (x * 30) as u8;
        let bytes = png_bytes(7, 4, |x, _| [v(x); 3]);
        let chw = preprocess_to_chw(&bytes, &cfg(4, 4, [0.0; 3], [1.0; 3])).unwrap();
        for x in 0..4u32 {
            let want = v(x + 1) as f32 / 255.0;
            let got = chw[x as usize];
            assert!(
                (got - want).abs() < TOL,
                "column {x}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn resize_dims_shortest_edge_rounding() {
        // Even scale: exact halving.
        assert_eq!(resize_dims(8, 4, 2), (4, 2));
        // Odd: 5 · (2/3) = 3.33… rounds to 3.
        assert_eq!(resize_dims(5, 3, 2), (3, 2));
        // Half rounds away from zero: 10 · (3/4) = 7.5 → 8.
        assert_eq!(resize_dims(10, 4, 3), (8, 3));
        // The shortest edge always lands exactly on `resize`.
        assert_eq!(resize_dims(3, 2, 1), (2, 1));
    }

    #[test]
    fn batch_appends_in_order() {
        let red = png_bytes(4, 4, |_, _| [255, 0, 0]);
        let green = png_bytes(4, 4, |_, _| [0, 255, 0]);
        let c = cfg(4, 4, [0.0; 3], [1.0; 3]);
        let mut out = Vec::new();
        preprocess_batch_into(&mut out, &[red.as_slice(), green.as_slice()], &c).unwrap();
        assert_eq!(out.len(), 2 * 3 * 16);
        // First image plane 0 (red channel) is all 1.0; second is all 0.0.
        assert!(out[0..16].iter().all(|&v| (v - 1.0).abs() < TOL));
        let second = 3 * 16;
        assert!(out[second..second + 16].iter().all(|&v| v.abs() < TOL));
    }

    /// The accelerated FMA path matches the literal `(p/255 − mean)/std` scalar
    /// reference within the golden tolerance, over a non-trivial gradient image
    /// (every byte value exercised) and asymmetric per-channel mean/std. This is
    /// the pin that justifies NOT bumping `IMAGE_PREP_VERSION_RS`.
    #[test]
    fn accel_matches_scalar_reference_within_tolerance() {
        // 16×16 gradient: distinct value per channel per pixel, spanning 0..=255.
        let bytes = png_bytes(16, 16, |x, y| {
            [
                ((x * 16 + y) % 256) as u8,
                ((y * 16 + x) % 256) as u8,
                ((x * x + y * y) % 256) as u8,
            ]
        });
        // Asymmetric per-channel mean/std (f32-exact literals); the test needs
        // distinct channels, not the literal CLIP constants.
        let mean = [0.5f32, 0.25, 0.375];
        let std = [0.25f32, 0.3125, 0.1875];
        let c = cfg(16, 16, mean, std);
        let got = preprocess_to_chw(&bytes, &c).unwrap();

        // The literal reference: decode the *same* cropped pixels, apply the
        // two-divide affine with no constant hoisting.
        let cropped = super::decode_resize_crop(&bytes, &c).unwrap();
        let (w, h) = cropped.dimensions();
        let (w, h) = (w as usize, h as usize);
        let mut max_abs = 0.0f32;
        for ch in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let p = cropped.get_pixel(x as u32, y as u32).0[ch] as f32;
                    let want = (p / 255.0 - mean[ch]) / std[ch];
                    let idx = ch * w * h + y * w + x;
                    max_abs = max_abs.max((got[idx] - want).abs());
                }
            }
        }
        assert!(
            max_abs < TOL,
            "accel vs scalar-reference max abs drift {max_abs} exceeds golden tolerance {TOL}"
        );
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = preprocess_to_chw(&[0u8, 1, 2, 3], &cfg(4, 4, [0.0; 3], [1.0; 3])).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }
}
