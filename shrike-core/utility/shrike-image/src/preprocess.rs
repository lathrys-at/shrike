//! The CLIP byte→pixels→CHW pipeline: decode → resize-shortest-edge →
//! center-crop → rescale/normalize → channel-major f32. The normalize/CHW
//! transform has an `accel` (default) and a scalar fallback; the resize is
//! `image`'s scalar Catmull-Rom throughout.

use image::imageops::FilterType;
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};

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
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `mean` or `std` is not exactly
    /// three channels.
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
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidInput`] if `bytes` is not a decodable image or
/// decodes to a zero-sized image.
pub fn preprocess_to_chw(bytes: &[u8], cfg: &PreprocessConfig) -> NativeResult<Vec<f32>> {
    let c = cfg.crop as usize;
    let mut out = Vec::with_capacity(3 * c * c);
    preprocess_into_chw(&mut out, bytes, cfg)?;
    Ok(out)
}

/// Preprocess each image and **append** its CHW plane to `out` in order — the
/// flat `[N, 3, crop, crop]` tensor the vision graph feeds on. `out` is appended
/// to (not cleared), so a caller may pre-reserve `images.len()·3·crop²`.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidInput`] on the first image that is not a
/// decodable image or decodes to a zero-sized image; `out` keeps the planes
/// appended for the images processed before the failure.
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
/// resampler the golden tests pin (a SIMD backend would change these pixels, so
/// it stays behind a consumer-fingerprint mechanism; see the crate docs).
fn resize_shortest_edge(img: &image::RgbImage, nw: u32, nh: u32) -> NativeResult<image::RgbImage> {
    Ok(image::imageops::resize(img, nw, nh, FilterType::CatmullRom))
}

/// Preprocess one image and **append** its CHW plane (length `3·crop²`) to
/// `out` — the per-item unit [`preprocess_batch_into`] loops and the natural
/// call for a consumer assembling one flat tensor across a batch. The hot path:
/// the per-channel affine `(p/255 − mean)/std` is hoisted to FMA constants and
/// the strided HWC→CHW scatter is rewritten as three contiguous plane passes
/// (one per channel), each a tight loop the compiler autovectorizes.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidInput`] if `bytes` is not a decodable image or
/// decodes to a zero-sized image.
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

    /// SplitMix64 — a tiny deterministic PRNG for the malformed-byte fuzz sweep,
    /// so the adversarial corpus is reproducible without a dev-dep.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// In-memory encode of an RGB image to any container format — synthesizes
    /// valid fixtures across the decode surface (BMP/GIF/PNG; JPEG is lossy so
    /// callers avoid pixel-exact goldens on it).
    fn encode_rgb(
        w: u32,
        h: u32,
        fmt: image::ImageFormat,
        px: impl Fn(u32, u32) -> [u8; 3],
    ) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(px(x, y)));
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, fmt)
            .expect("encode");
        buf.into_inner()
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
        let cropped = decode_resize_crop(&bytes, &c).unwrap();
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

    // ---- Adversarial: malformed / hostile decode (untrusted bytes from notes
    // and media; the decode boundary must reject, never panic / OOM / hang). ----

    /// Empty and 1-byte inputs are the degenerate untrusted case; the boundary
    /// must reject them as `InvalidInput`, not panic on an out-of-bounds header
    /// read.
    #[test]
    fn rejects_empty_and_single_byte() {
        let c = cfg(4, 4, [0.0; 3], [1.0; 3]);
        for bytes in [&[][..], &[0x00][..], &[0xFF][..]] {
            let err = preprocess_to_chw(bytes, &c).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidInput);
        }
    }

    /// A valid PNG cut at every truncation point (header-only through almost-whole)
    /// must never panic or read past the buffer. The `png` decoder tolerates a
    /// missing trailer and may return the rows it has, so the contract is total
    /// (Ok | Err) — and any Ok is a *bounded* full-target plane with finite
    /// values, never an out-of-bounds or partial buffer.
    #[test]
    fn truncated_png_at_every_cut_errs_never_panics() {
        let png = png_bytes(12, 9, |x, y| {
            [(x * 9 + y) as u8, (y * 7) as u8, (x ^ y) as u8]
        });
        let c = cfg(4, 4, [0.0; 3], [1.0; 3]);
        for cut in 1..png.len() {
            match preprocess_to_chw(&png[..cut], &c) {
                Ok(v) => {
                    assert_eq!(v.len(), 3 * 4 * 4, "cut {cut}: wrong plane size");
                    assert!(v.iter().all(|f| f.is_finite()), "cut {cut}: non-finite");
                }
                Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput, "cut {cut}"),
            }
        }
    }

    /// Each format's magic/signature followed by random garbage must be rejected,
    /// not trusted past the header into an unbounded read.
    #[test]
    fn signature_then_garbage_errs() {
        let c = cfg(4, 4, [0.0; 3], [1.0; 3]);
        let sigs: &[&[u8]] = &[
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A], // PNG
            &[0xFF, 0xD8, 0xFF, 0xE0],                         // JPEG
            b"BM",                                             // BMP
            b"GIF89a",                                         // GIF
            &[0x52, 0x49, 0x46, 0x46],                         // RIFF (WebP)
        ];
        for sig in sigs {
            let mut bytes = sig.to_vec();
            bytes.extend(std::iter::repeat_n(0xA5u8, 64));
            // Garbage past a real signature must not decode; whatever the kind,
            // it must be an error and must not panic.
            assert!(preprocess_to_chw(&bytes, &c).is_err());
        }
    }

    /// A BMP whose header declares ~100k×100k (a decompression-bomb shape) must
    /// be bounded by the decoder's allocation limits and rejected — never an
    /// unbounded allocation. Untrusted bytes cannot dictate allocation size.
    #[test]
    fn decompression_bomb_dimensions_are_bounded_err() {
        let (w, h): (i32, i32) = (100_000, 100_000);
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&0u32.to_le_bytes()); // file size (ignored)
        bmp.extend_from_slice(&0u32.to_le_bytes()); // reserved
        bmp.extend_from_slice(&54u32.to_le_bytes()); // pixel-data offset
        bmp.extend_from_slice(&40u32.to_le_bytes()); // DIB header size (BITMAPINFOHEADER)
        bmp.extend_from_slice(&w.to_le_bytes());
        bmp.extend_from_slice(&h.to_le_bytes());
        bmp.extend_from_slice(&1u16.to_le_bytes()); // planes
        bmp.extend_from_slice(&24u16.to_le_bytes()); // bpp
        bmp.extend_from_slice(&0u32.to_le_bytes()); // compression = none
        bmp.extend_from_slice(&0u32.to_le_bytes()); // image byte size
        bmp.extend_from_slice(&0i32.to_le_bytes()); // x ppm
        bmp.extend_from_slice(&0i32.to_le_bytes()); // y ppm
        bmp.extend_from_slice(&0u32.to_le_bytes()); // colors used
        bmp.extend_from_slice(&0u32.to_le_bytes()); // important colors
        let err = preprocess_to_chw(&bmp, &cfg(224, 224, [0.0; 3], [1.0; 3])).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    /// The fuzz sweep: thousands of random buffers and random truncations of a
    /// valid fixture. The contract under hostile bytes is total — every call
    /// returns `Ok | Err`, none panics. (A panic here unwinds the test and fails
    /// it, which is exactly the regression we guard.)
    #[test]
    fn fuzz_random_and_truncated_buffers_never_panic() {
        let c = cfg(8, 8, [0.48, 0.46, 0.41], [0.27, 0.26, 0.28]);
        let valid = png_bytes(20, 14, |x, y| {
            [(x * 13) as u8, (y * 11) as u8, (x + y) as u8]
        });
        let mut rng = Rng::new(0xDEAD_BEEF_CAFE_F00D);

        for _ in 0..4000 {
            let pick = rng.next_u64();
            let bytes: Vec<u8> = if pick & 1 == 0 {
                // Fully random buffer, length 0..=255.
                let len = (rng.next_u64() % 256) as usize;
                (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect()
            } else {
                // Random truncation of the valid fixture, optionally with a
                // trailing byte flipped to corrupt an otherwise-whole image.
                let cut = (rng.next_u64() as usize % valid.len()).max(1);
                let mut v = valid[..cut].to_vec();
                if pick & 2 == 0 && !v.is_empty() {
                    let i = rng.next_u64() as usize % v.len();
                    v[i] ^= (rng.next_u64() & 0xFF) as u8;
                }
                v
            };
            // The only assertion is "did not panic"; an Ok result is acceptable
            // (a corrupted-but-decodable image is fine — it's bounded output).
            let r = preprocess_to_chw(&bytes, &c);
            if let Ok(v) = r {
                assert_eq!(v.len(), 3 * 8 * 8);
                assert!(v.iter().all(|f| f.is_finite()));
            }
        }
    }

    // ---- Format / shape edge cases: channel count and output dims are an
    // invariant of the embedding contract regardless of input shape. ----

    /// A 1×1 image (the smallest legal image) produces the full target plane
    /// `3·crop²` with finite values — no divide-by-zero on the degenerate dims.
    #[test]
    fn one_by_one_image_yields_full_target_plane() {
        let bytes = png_bytes(1, 1, |_, _| [200, 100, 50]);
        let chw = preprocess_to_chw(&bytes, &cfg(4, 4, [0.0; 3], [1.0; 3])).unwrap();
        assert_eq!(chw.len(), 3 * 4 * 4);
        assert!(chw.iter().all(|f| f.is_finite()));
    }

    /// Extreme aspect ratios (1px-tall-wide and 1px-wide-tall) must still produce
    /// exactly the target dims; shortest-edge resize + center-crop never under- or
    /// over-fills the output tensor.
    #[test]
    fn extreme_aspect_ratios_yield_exact_target_dims() {
        let c = cfg(8, 8, [0.0; 3], [1.0; 3]);
        for (w, h) in [(64, 1), (1, 64), (128, 2), (2, 128)] {
            let bytes = png_bytes(w, h, |x, y| [(x as u8).wrapping_add(y as u8); 3]);
            let chw = preprocess_to_chw(&bytes, &c).unwrap();
            assert_eq!(chw.len(), 3 * 8 * 8, "dims {w}x{h}");
            assert!(chw.iter().all(|f| f.is_finite()));
        }
    }

    /// Grayscale and RGBA inputs both decode through `to_rgb8` to a 3-channel
    /// output of the target size: the pipeline output channel count is fixed at 3
    /// regardless of the source pixel type.
    #[test]
    fn grayscale_and_rgba_inputs_become_three_channel_output() {
        let c = cfg(4, 4, [0.0; 3], [1.0; 3]);

        // Grayscale: every output channel equals the luma value (rescaled).
        let gray = {
            let img = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([128]));
            let mut buf = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageLuma8(img)
                .write_to(&mut buf, image::ImageFormat::Png)
                .unwrap();
            buf.into_inner()
        };
        let g = preprocess_to_chw(&gray, &c).unwrap();
        assert_eq!(g.len(), 3 * 4 * 4);
        for &v in &g {
            assert!((v - 128.0 / 255.0).abs() < TOL);
        }

        // RGBA: alpha is dropped, RGB carried through (no premultiply).
        let rgba = {
            let img = image::RgbaImage::from_fn(4, 4, |_, _| image::Rgba([60, 120, 180, 7]));
            let mut buf = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut buf, image::ImageFormat::Png)
                .unwrap();
            buf.into_inner()
        };
        let a = preprocess_to_chw(&rgba, &c).unwrap();
        let want = [60.0 / 255.0, 120.0 / 255.0, 180.0 / 255.0];
        for ch in 0..3 {
            for v in &a[ch * 16..(ch + 1) * 16] {
                assert!((v - want[ch]).abs() < TOL, "ch {ch}: {v} != {}", want[ch]);
            }
        }
    }

    /// A valid image synthesized in BMP and GIF decodes to the same target plane
    /// as PNG — the decode surface spans the declared format feature set, not just
    /// PNG.
    #[test]
    fn bmp_and_gif_decode_to_target_plane() {
        let c = cfg(4, 4, [0.0; 3], [1.0; 3]);
        for fmt in [image::ImageFormat::Bmp, image::ImageFormat::Gif] {
            let bytes = encode_rgb(6, 6, fmt, |_, _| [90, 90, 90]);
            let chw = preprocess_to_chw(&bytes, &c).unwrap();
            assert_eq!(chw.len(), 3 * 4 * 4, "{fmt:?}");
            assert!(chw.iter().all(|v| (v - 90.0 / 255.0).abs() < TOL));
        }
    }

    // ---- Pixel-pipeline oracle / property tests. ----

    /// Determinism: the same bytes through the same config produce a bitwise
    /// identical vector. The embedding contract relies on this — a non-deterministic
    /// preprocess would silently invalidate the stored-vector fingerprint model.
    #[test]
    fn preprocess_is_deterministic() {
        let bytes = png_bytes(40, 27, |x, y| [(x * 6) as u8, (y * 9) as u8, (x * y) as u8]);
        let c = cfg(16, 14, [0.481, 0.457, 0.408], [0.268, 0.261, 0.275]);
        let a = preprocess_to_chw(&bytes, &c).unwrap();
        let b = preprocess_to_chw(&bytes, &c).unwrap();
        assert_eq!(a, b, "identical input must yield identical output");
    }

    /// Shape + finiteness contract over a realistic CLIP config (224×224, the
    /// real ViT input): output length is exactly `3·crop²` and every value is
    /// finite (no NaN/Inf could reach the vision graph).
    #[test]
    fn clip_output_shape_and_finiteness() {
        let bytes = png_bytes(300, 200, |x, y| {
            [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]
        });
        let mean = [0.481_454_8, 0.457_827_5, 0.408_210_7];
        let std = [0.268_629_5, 0.261_302_6, 0.275_777_1];
        let chw = preprocess_to_chw(&bytes, &cfg(224, 224, mean, std)).unwrap();
        assert_eq!(chw.len(), 3 * 224 * 224);
        assert!(chw.iter().all(|f| f.is_finite()));
    }

    /// Analytic oracle: a solid-color image normalizes to exactly
    /// `(color/255 − mean)/std` per channel everywhere, independent of resize/crop
    /// (a constant image is resize/crop invariant). Computes the expected value by
    /// hand and asserts within f32 epsilon over the whole plane.
    #[test]
    fn solid_color_matches_analytic_oracle_across_resize() {
        let color = [219u8, 17, 144];
        let mean = [0.481f32, 0.457, 0.408];
        let std = [0.268f32, 0.261, 0.275];
        // Source bigger than the crop so a real resize runs; the constant image
        // is invariant under it, so the oracle still holds exactly.
        let bytes = png_bytes(50, 80, |_, _| color);
        let chw = preprocess_to_chw(&bytes, &cfg(28, 24, mean, std)).unwrap();
        let plane = 24 * 24;
        assert_eq!(chw.len(), 3 * plane);
        for ch in 0..3 {
            let want = (color[ch] as f32 / 255.0 - mean[ch]) / std[ch];
            for v in &chw[ch * plane..(ch + 1) * plane] {
                assert!((v - want).abs() < TOL, "ch {ch}: {v} != {want}");
            }
        }
    }

    /// Resize is an identity-shaped operation on output dims: regardless of input
    /// size or aspect, the output is always exactly `3·crop²`. An image already at
    /// the target size round-trips its constant pixels unchanged (resize at scale
    /// 1 is pixel-exact for a constant image).
    #[test]
    fn resize_output_dims_invariant_and_identity_at_target() {
        let mean = [0.0; 3];
        let std = [1.0; 3];
        // Output dims invariant across wildly different inputs.
        for (w, h) in [(8, 8), (8, 32), (32, 8), (100, 7), (7, 100), (1, 1)] {
            let bytes = png_bytes(w, h, |_, _| [33, 66, 99]);
            let chw = preprocess_to_chw(&bytes, &cfg(8, 8, mean, std)).unwrap();
            assert_eq!(chw.len(), 3 * 8 * 8, "dims {w}x{h}");
        }
        // Already-at-target identity: 8×8 source, resize 8, crop 8 → no scaling,
        // no crop offset; a constant image is carried through pixel-exact.
        let bytes = png_bytes(8, 8, |_, _| [33, 66, 99]);
        let chw = preprocess_to_chw(&bytes, &cfg(8, 8, mean, std)).unwrap();
        let want = [33.0 / 255.0, 66.0 / 255.0, 99.0 / 255.0];
        for ch in 0..3 {
            for v in &chw[ch * 64..(ch + 1) * 64] {
                assert!((v - want[ch]).abs() < TOL);
            }
        }
    }

    /// `from_slices` rejects wrong-arity mean/std — the FFI arity guard on the
    /// config boundary (not bytes, but still untrusted shape from across the FFI).
    #[test]
    fn from_slices_rejects_wrong_arity() {
        assert!(PreprocessConfig::from_slices(224, 224, &[0.0, 0.0], &[1.0; 3]).is_err());
        assert!(PreprocessConfig::from_slices(224, 224, &[0.0; 3], &[1.0; 4]).is_err());
        assert!(PreprocessConfig::from_slices(224, 224, &[0.0; 3], &[1.0; 3]).is_ok());
    }
}
