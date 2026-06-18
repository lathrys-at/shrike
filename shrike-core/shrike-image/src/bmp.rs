//! A minimal, dependency-free 24-bit BMP encoder (#707). Moved out of
//! `shrike-engine-api`'s probe so format-encoding code lives in the image
//! utility crate rather than the contract leaf.
//!
//! Emits an uncompressed `BI_RGB` bitmap: a fixed 54-byte header (14-byte
//! `BITMAPFILEHEADER` + 40-byte `BITMAPINFOHEADER`) and bottom-up BGR rows
//! padded to a 4-byte boundary. It round-trips losslessly through any BMP
//! decoder, so the pixels a decoder sees are exactly the ones `pixel` returned.
//! The batch-safety probe uses it to synthesize its vision probe set without
//! pulling an image-encode dependency into the engine contract.

/// Uncompressed 24-bit BMP of `width × height` pixels, each from `pixel(x, y)`
/// (top-left origin → emitted bottom-up per the format). Panics never: any
/// `(width, height)` producing a representable buffer is encoded.
pub fn encode_bmp(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    let row_bytes = (width * 3) as usize;
    let padding = (4 - row_bytes % 4) % 4;
    let stride = row_bytes + padding;
    let pixel_data = stride * height as usize;
    let file_size = 54 + pixel_data;

    let mut out = Vec::with_capacity(file_size);
    // -- BITMAPFILEHEADER (14 bytes) --
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    out.extend_from_slice(&54u32.to_le_bytes()); // pixel-data offset
                                                 // -- BITMAPINFOHEADER (40 bytes) --
    out.extend_from_slice(&40u32.to_le_bytes()); // header size
    out.extend_from_slice(&(width as i32).to_le_bytes()); // width
    out.extend_from_slice(&(height as i32).to_le_bytes()); // height (+ = bottom-up)
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&24u16.to_le_bytes()); // bits per pixel
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB (no compression)
    out.extend_from_slice(&(pixel_data as u32).to_le_bytes()); // image size
    out.extend_from_slice(&2835i32.to_le_bytes()); // x ppm (~72 dpi)
    out.extend_from_slice(&2835i32.to_le_bytes()); // y ppm
    out.extend_from_slice(&0u32.to_le_bytes()); // palette colours
    out.extend_from_slice(&0u32.to_le_bytes()); // important colours
                                                // -- pixel rows, bottom-up, BGR + per-row padding --
    for y in (0..height).rev() {
        for x in 0..width {
            let [r, g, b] = pixel(x, y);
            out.extend_from_slice(&[b, g, r]);
        }
        out.extend(std::iter::repeat_n(0u8, padding));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The encoded BMP round-trips losslessly through the `image` decoder: the
    /// dimensions and every pixel come back exactly as `pixel` produced them.
    /// This is the home of the probe set's old decodability check. Gated on
    /// `preprocess` because it is the only config where `image` is a dep — the
    /// dependency-free encoder above is tested by `header_offsets_and_size`.
    #[cfg(feature = "preprocess")]
    #[test]
    fn bmp_round_trips_through_decoder() {
        let side = 17u32; // odd → exercises the row padding (51 % 4 != 0)
        let px = |x: u32, y: u32| -> [u8; 3] {
            [
                (x.wrapping_mul(13) ^ y) as u8,
                (y.wrapping_mul(7)) as u8,
                (x ^ y).wrapping_mul(5) as u8,
            ]
        };
        let bytes = encode_bmp(side, side, px);
        let decoded = image::load_from_memory(&bytes)
            .expect("encoded BMP must decode")
            .to_rgb8();
        assert_eq!(decoded.dimensions(), (side, side));
        for y in 0..side {
            for x in 0..side {
                assert_eq!(decoded.get_pixel(x, y).0, px(x, y), "pixel ({x},{y})");
            }
        }
    }

    #[test]
    fn header_offsets_and_size() {
        let bytes = encode_bmp(2, 2, |_, _| [1, 2, 3]);
        assert_eq!(&bytes[0..2], b"BM");
        // 2×3 = 6 bytes/row, padded to 8; ×2 rows = 16; +54 header = 70.
        assert_eq!(u32::from_le_bytes(bytes[2..6].try_into().unwrap()), 70);
        assert_eq!(bytes.len(), 70);
        assert_eq!(u32::from_le_bytes(bytes[10..14].try_into().unwrap()), 54);
    }
}
