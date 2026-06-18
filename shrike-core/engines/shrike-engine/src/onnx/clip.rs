//! Native CLIP dual-encoder engine: two ort graphs + image preprocessing
//! via the `shrike-image` utility crate (the byte→pixels→CHW pipeline).
//!
//! Mirrors `shrike/embedding_clip.py`: text graph (`input_ids` → `text_embeds`)
//! and vision graph (`pixel_values` → `image_embeds`) projecting into one shared
//! space; fixed 77-token context; resize-shortest-edge → center-crop → rescale →
//! normalize preprocessing parameterised by the model's preprocessor config
//! (parsed Python-side — only scalars cross the FFI). **Pixel parity caveat:**
//! the `image` crate's Catmull-Rom bicubic differs from PIL's bicubic (different
//! cubic coefficient), so image vectors are semantically equivalent but not
//! bit-identical to the Python engine's — the facade namespaces the fingerprint
//! (`clip-rs:`) accordingly.

use std::sync::Mutex;

use shrike_engine_api::MediaItem;
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};
use shrike_image::PreprocessConfig;
use tokenizers::Tokenizer;

use super::session::GraphInput;
use super::session::{build_session, extract_2d, graph_inputs, int_tensor, l2_normalize_rows};

/// The pixel-math version of the CLIP preprocessing pipeline, re-exported from
/// `shrike-image` (its owner) so the facade's fingerprint and the
/// Python binding keep reading `IMAGE_PREP_VERSION_RS` from this crate.
pub use shrike_image::IMAGE_PREP_VERSION_RS;

/// Construction parameters for [`ClipEmbedder`] (a CLIP text+vision ONNX pair).
pub struct ClipEmbedderConfig {
    /// Path to the text-encoder ONNX model.
    pub text_model_path: String,
    /// Path to the vision-encoder ONNX model.
    pub vision_model_path: String,
    /// Path to the CLIP tokenizer JSON.
    pub tokenizer_path: String,
    /// Already resolved Python-side (intersected with available, CPU appended).
    pub providers: Vec<String>,
    /// Per-channel pixel mean for normalization.
    pub image_mean: Vec<f32>,
    /// Per-channel pixel std for normalization.
    pub image_std: Vec<f32>,
    /// Shortest-edge resize target (preprocessor "size").
    pub resize: u32,
    /// Center-crop size (preprocessor "crop_size").
    pub crop: u32,
    /// The fixed token context (77 for CLIP).
    pub context: usize,
}

/// A CLIP dual-encoder embedding text and images into one shared space.
pub struct ClipEmbedder {
    // Locked because ort 2.0.0-rc.12's run entrypoints all take `&mut self`
    // (see the note on `super::text::TextEmbedder`'s session).
    text_session: Mutex<ort::session::Session>,
    vision_session: Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    text_inputs: Vec<GraphInput>,
    vision_input_name: String,
    /// The CLIP pixel-preprocessing parameters (shrike-image owns the pipeline).
    prep: PreprocessConfig,
    active_providers: Vec<String>,
    dim: Mutex<Option<usize>>,
}

impl ClipEmbedder {
    /// Load both ONNX sessions and the tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error if a model/tokenizer file can't be loaded or a session can't be built.
    pub fn load(cfg: ClipEmbedderConfig) -> NativeResult<Self> {
        let prep =
            PreprocessConfig::from_slices(cfg.resize, cfg.crop, &cfg.image_mean, &cfg.image_std)?;
        let (text_session, active) = build_session(&cfg.text_model_path, &cfg.providers)?;
        let (vision_session, _) = build_session(&cfg.vision_model_path, &cfg.providers)?;

        // Feed whichever of input_ids/attention_mask the text graph declares,
        // at its declared integer width (mirrors the Python engine's feed).
        let text_inputs = graph_inputs(&text_session, &["input_ids", "attention_mask"]);
        let vision_input_name = vision_session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| NativeError::invalid_input("vision graph declares no inputs"))?;

        let mut tokenizer = Tokenizer::from_file(&cfg.tokenizer_path)
            .context(ErrorKind::Unavailable, "loading tokenizer")?;
        // CLIP uses a fixed context: truncate and pad to exactly `context`
        // (pad token defaults match the Python tokenizers enable_padding()).
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: cfg.context,
                ..Default::default()
            }))
            .context(ErrorKind::Internal, "truncation")?;
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
            prep,
            active_providers: active,
            dim: Mutex::new(None),
        })
    }

    /// The execution providers actually registered, in order.
    pub fn active_providers(&self) -> &[String] {
        &self.active_providers
    }

    /// The embedding width once known (set by the first embed).
    ///
    /// # Panics
    ///
    /// Panics if the dim mutex is poisoned (a prior holder panicked).
    pub fn dim(&self) -> Option<usize> {
        *self.dim.lock().expect("dim lock poisoned")
    }

    /// Embed one chunk of texts (the unit the facade's probe/chunking build on).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the ONNX run fails.
    ///
    /// # Panics
    ///
    /// Panics if the session or dim mutex is poisoned.
    pub fn embed_text_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .context(ErrorKind::InvalidInput, "tokenization failed")?;
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
                int_tensor(batch, seq, data, input.int32)?,
            ));
        }
        let outputs = session
            .run(feed)
            .context(ErrorKind::InvalidInput, "clip text run failed")?;
        let vectors = extract_2d(&outputs)?;
        let vectors = l2_normalize_rows(vectors);
        *self.dim.lock().expect("dim lock poisoned") = Some(vectors.ncols());
        Ok(vectors.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// Embed one chunk of images, each an encoded [`MediaItem`] (PNG/JPEG/…).
    /// The mime hint is unused today — the decoder sniffs magic bytes — but
    /// the typed item keeps the signature aligned with the engine contract.
    ///
    /// # Errors
    ///
    /// Returns an error if an image can't be decoded/preprocessed or the ONNX run fails.
    ///
    /// # Panics
    ///
    /// Panics if the session or dim mutex is poisoned.
    pub fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.prep.crop as usize;
        let mut pixels = Vec::with_capacity(images.len() * 3 * c * c);
        for item in images {
            shrike_image::preprocess_into_chw(&mut pixels, &item.bytes, &self.prep)?;
        }
        let tensor = ort::value::Tensor::from_array(([images.len(), 3, c, c], pixels))
            .context(ErrorKind::Internal, "pixel tensor")?;

        let mut session = self
            .vision_session
            .lock()
            .expect("vision session lock poisoned");
        let feed: Vec<(String, ort::value::DynValue)> =
            vec![(self.vision_input_name.clone(), tensor.into_dyn())];
        let outputs = session
            .run(feed)
            .context(ErrorKind::InvalidInput, "clip vision run failed")?;
        let vectors = extract_2d(&outputs)?;
        let vectors = l2_normalize_rows(vectors);
        *self.dim.lock().expect("dim lock poisoned") = Some(vectors.ncols());
        Ok(vectors.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// Internal bytes-shaped entry, kept for the binding's probe path.
    ///
    /// # Errors
    ///
    /// Returns an error if an image can't be decoded/preprocessed or the ONNX run fails.
    pub fn embed_image_bytes_chunk(&self, images: Vec<Vec<u8>>) -> NativeResult<Vec<Vec<f32>>> {
        let items: Vec<MediaItem> = images.into_iter().map(MediaItem::untyped).collect();
        self.embed_image_chunk(&items)
    }
}

/// The engine contract (route 1): the dual encoder is one engine
/// implementing BOTH compute traits — text and image chunks project into the
/// shared CLIP space. Identity/batch policy come from the host (`WithPolicy`),
/// execution from an adapter lane; `safe_batch` stays the probed-by-host
/// default, exactly as for [`super::text::TextEmbedder`].
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
