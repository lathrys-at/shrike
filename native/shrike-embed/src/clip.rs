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
use shrike_ffi::{NativeError, NativeResult};
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

    /// Embed one chunk of images, each given as encoded bytes (PNG/JPEG/...).
    pub fn embed_image_chunk(&self, images: &[Vec<u8>]) -> NativeResult<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.crop as usize;
        let mut pixels = Vec::with_capacity(images.len() * 3 * c * c);
        for bytes in images {
            pixels.extend(self.preprocess(bytes)?);
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

    /// CLIP preprocessing → CHW f32: decode, resize shortest edge, center-crop,
    /// rescale to [0,1], normalize per channel. Mirrors the Python `_preprocess`
    /// (PIL), with the `image` crate's Catmull-Rom standing in for PIL bicubic.
    fn preprocess(&self, bytes: &[u8]) -> NativeResult<Vec<f32>> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| NativeError::invalid_input(format!("image decode failed: {e}")))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 {
            return Err(NativeError::invalid_input("empty image"));
        }
        let scale = self.resize as f64 / w.min(h) as f64;
        let (nw, nh) = (
            (w as f64 * scale).round().max(1.0) as u32,
            (h as f64 * scale).round().max(1.0) as u32,
        );
        let resized = image::imageops::resize(&img, nw, nh, FilterType::CatmullRom);
        let c = self.crop;
        let left = (nw.saturating_sub(c)) / 2;
        let top = (nh.saturating_sub(c)) / 2;
        let cropped = image::imageops::crop_imm(&resized, left, top, c, c).to_image();

        let c = c as usize;
        let mut chw = vec![0.0f32; 3 * c * c];
        for (y, row) in cropped.rows().enumerate() {
            for (x, px) in row.enumerate() {
                for ch in 0..3 {
                    chw[ch * c * c + y * c + x] =
                        (px.0[ch] as f32 / 255.0 - self.mean[ch]) / self.std[ch];
                }
            }
        }
        Ok(chw)
    }
}
