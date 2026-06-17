//! Native CLIP dual-encoder engine (#271): two ort graphs + image preprocessing
//! via the `image` crate (replacing PIL/numpy in the Python engine).
//!
//! Mirrors `shrike/embedding_clip.py`: text graph (`input_ids` → `text_embeds`)
//! and vision graph (`pixel_values` → `image_embeds`) projecting into one shared
//! space; fixed 77-token context; resize-shortest-edge → center-crop → rescale →
//! normalize preprocessing parameterised by the model's preprocessor config
//! (parsed Python-side — only scalars cross the FFI). **Pixel parity caveat:**
//! the `image` crate's Catmull-Rom bicubic differs from PIL's bicubic (different
//! cubic coefficient), so image vectors are semantically equivalent but not
//! bit-identical to the Python engine's — the facade namespaces the fingerprint
//! (`clip-rs:`) accordingly (epic #265 convention 7).

use std::sync::Mutex;

use image::imageops::FilterType;
use shrike_engine_api::MediaItem;
use shrike_error::{NativeError, NativeResult};
use tokenizers::Tokenizer;

use crate::{l2_normalize_rows, GraphInput};

/// Bump when this pipeline's pixel math changes (folded into the facade's
/// fingerprint, like IMAGE_PREP_VERSION does for the Python engine).
pub const IMAGE_PREP_VERSION_RS: u32 = 1;

pub struct ClipEmbedderConfig {
    pub text_model_path: String,
    pub vision_model_path: String,
    pub tokenizer_path: String,
    /// Already resolved Python-side (intersected with available, CPU appended).
    pub providers: Vec<String>,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    /// Shortest-edge resize target (preprocessor "size").
    pub resize: u32,
    /// Center-crop size (preprocessor "crop_size").
    pub crop: u32,
    /// The fixed token context (77 for CLIP).
    pub context: usize,
}

pub struct ClipEmbedder {
    // Locked because ort 2.0.0-rc.12's run entrypoints all take `&mut self`
    // (see the note on `TextEmbedder::session`).
    text_session: Mutex<ort::session::Session>,
    vision_session: Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    text_inputs: Vec<GraphInput>,
    vision_input_name: String,
    mean: Vec<f32>,
    std: Vec<f32>,
    resize: u32,
    crop: u32,
    active_providers: Vec<String>,
    dim: Mutex<Option<usize>>,
}

impl ClipEmbedder {
    pub fn load(cfg: ClipEmbedderConfig) -> NativeResult<Self> {
        if cfg.image_mean.len() != 3 || cfg.image_std.len() != 3 {
            return Err(NativeError::invalid_input(
                "image_mean/image_std must each have 3 channels",
            ));
        }
        let (text_session, active) = crate::build_session(&cfg.text_model_path, &cfg.providers)?;
        let (vision_session, _) = crate::build_session(&cfg.vision_model_path, &cfg.providers)?;

        // Feed whichever of input_ids/attention_mask the text graph declares,
        // at its declared integer width (mirrors the Python engine's feed).
        let text_inputs = crate::graph_inputs(&text_session, &["input_ids", "attention_mask"]);
        let vision_input_name = vision_session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| NativeError::invalid_input("vision graph declares no inputs"))?;

        let mut tokenizer = Tokenizer::from_file(&cfg.tokenizer_path)
            .map_err(|e| NativeError::unavailable(format!("loading tokenizer: {e}")))?;
        // CLIP uses a fixed context: truncate and pad to exactly `context`
        // (pad token defaults match the Python tokenizers enable_padding()).
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: cfg.context,
                ..Default::default()
            }))
            .map_err(|e| NativeError::internal(format!("truncation: {e}")))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(cfg.context),
            ..Default::default()
        }));

        Ok(Self {
            text_session: Mutex::new(text_session),
            vision_session: Mutex::new(vision_session),
            tokenizer,
            text_inputs,
            vision_input_name,
            mean: cfg.image_mean,
            std: cfg.image_std,
            resize: cfg.resize,
            crop: cfg.crop,
            active_providers: active,
            dim: Mutex::new(None),
        })
    }

    pub fn active_providers(&self) -> &[String] {
        &self.active_providers
    }

    pub fn dim(&self) -> Option<usize> {
        *self.dim.lock().expect("dim lock poisoned")
    }

    /// Embed one chunk of texts (the unit the facade's probe/chunking build on).
    pub fn embed_text_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| NativeError::invalid_input(format!("tokenization failed: {e}")))?;
        let batch = encodings.len();
        let seq = encodings[0].get_ids().len();
        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|&v| v as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&v| v as i64));
        }

        let mut session = self
            .text_session
            .lock()
            .expect("text session lock poisoned");
        let mut feed: Vec<(String, ort::value::DynValue)> = Vec::new();
        for input in &self.text_inputs {
            let data = match input.name.as_str() {
                "input_ids" => &ids,
                "attention_mask" => &mask,
                _ => continue,
            };
            feed.push((
                input.name.clone(),
                crate::int_tensor(batch, seq, data, input.int32)?,
            ));
        }
        let outputs = session
            .run(feed)
            .map_err(|e| NativeError::invalid_input(format!("clip text run failed: {e}")))?;
        let vectors = crate::extract_2d(&outputs)?;
        let vectors = l2_normalize_rows(vectors);
        *self.dim.lock().expect("dim lock poisoned") = Some(vectors.ncols());
        Ok(vectors.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// Embed one chunk of images, each an encoded [`MediaItem`] (PNG/JPEG/…).
    /// The mime hint is unused today — the decoder sniffs magic bytes — but
    /// the typed item keeps the signature aligned with the engine contract.
    pub fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.crop as usize;
        let mut pixels = Vec::with_capacity(images.len() * 3 * c * c);
        for item in images {
            pixels.extend(self.preprocess(&item.bytes)?);
        }
        let tensor = ort::value::Tensor::from_array(([images.len(), 3, c, c], pixels))
            .map_err(|e| NativeError::internal(format!("pixel tensor: {e}")))?;

        let mut session = self
            .vision_session
            .lock()
            .expect("vision session lock poisoned");
        let feed: Vec<(String, ort::value::DynValue)> =
            vec![(self.vision_input_name.clone(), tensor.into_dyn())];
        let outputs = session
            .run(feed)
            .map_err(|e| NativeError::invalid_input(format!("clip vision run failed: {e}")))?;
        let vectors = crate::extract_2d(&outputs)?;
        let vectors = l2_normalize_rows(vectors);
        *self.dim.lock().expect("dim lock poisoned") = Some(vectors.ncols());
        Ok(vectors.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// Internal bytes-shaped entry, kept for the binding's probe path.
    pub fn embed_image_bytes_chunk(&self, images: Vec<Vec<u8>>) -> NativeResult<Vec<Vec<f32>>> {
        let items: Vec<MediaItem> = images.into_iter().map(MediaItem::untyped).collect();
        self.embed_image_chunk(&items)
    }

    /// CLIP preprocessing → CHW f32 (see [`preprocess`]).
    fn preprocess(&self, bytes: &[u8]) -> NativeResult<Vec<f32>> {
        preprocess(bytes, self.resize, self.crop, &self.mean, &self.std)
    }
}

/// Shortest-edge resize target: scale so min(w, h) == `resize`, the other edge
/// rounded (round-half-away-from-zero, the f64 `round`), floored at 1.
fn resize_dims(w: u32, h: u32, resize: u32) -> (u32, u32) {
    let scale = resize as f64 / w.min(h) as f64;
    (
        (w as f64 * scale).round().max(1.0) as u32,
        (h as f64 * scale).round().max(1.0) as u32,
    )
}

/// CLIP preprocessing → CHW f32: decode, resize shortest edge, center-crop,
/// rescale to [0,1], normalize per channel. Mirrors the Python `_preprocess`
/// (PIL), with the `image` crate's Catmull-Rom standing in for PIL bicubic.
/// A free function (not a method) so the golden tests cover it without
/// loading sessions.
fn preprocess(
    bytes: &[u8],
    resize: u32,
    crop: u32,
    mean: &[f32],
    std: &[f32],
) -> NativeResult<Vec<f32>> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| NativeError::invalid_input(format!("image decode failed: {e}")))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(NativeError::invalid_input("empty image"));
    }
    let (nw, nh) = resize_dims(w, h, resize);
    let resized = image::imageops::resize(&img, nw, nh, FilterType::CatmullRom);
    let c = crop;
    let left = (nw.saturating_sub(c)) / 2;
    let top = (nh.saturating_sub(c)) / 2;
    let cropped = image::imageops::crop_imm(&resized, left, top, c, c).to_image();

    let c = c as usize;
    let mut chw = vec![0.0f32; 3 * c * c];
    for (y, row) in cropped.rows().enumerate() {
        for (x, px) in row.enumerate() {
            for ch in 0..3 {
                chw[ch * c * c + y * c + x] = (px.0[ch] as f32 / 255.0 - mean[ch]) / std[ch];
            }
        }
    }
    Ok(chw)
}

/// The engine contract (#342, route 1): the dual encoder is one engine
/// implementing BOTH compute traits — text and image chunks project into the
/// shared CLIP space. Identity/batch policy come from the host (`WithPolicy`),
/// execution from an adapter lane; `safe_batch` stays the probed-by-host
/// default, exactly as for [`crate::TextEmbedder`].
impl shrike_engine_api::EmbedText for ClipEmbedder {
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        self.embed_text_chunk(texts)
    }

    fn dim(&self) -> Option<usize> {
        ClipEmbedder::dim(self)
    }
}

impl shrike_engine_api::EmbedImages for ClipEmbedder {
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        ClipEmbedder::embed_image_chunk(self, images)
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
        let chw = preprocess(&bytes, 4, 4, &mean, &std).unwrap();
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
        let chw = preprocess(&bytes, 4, 4, &[0.0; 3], &[1.0; 3]).unwrap();
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
        let chw = preprocess(&bytes, 4, 4, &[0.0; 3], &[1.0; 3]).unwrap();
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
        let chw = preprocess(&bytes, 4, 4, &[0.0; 3], &[1.0; 3]).unwrap();
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
}
