//! The native text-embedding engine (#270): ort + tokenizers + pooling.
//!
//! The Rust counterpart of `shrike/embedding_onnx.py` — same model layout, same
//! tokenization (the Python `tokenizers` package wraps this same crate), same
//! pooling/normalization semantics, driven through the same onnxruntime shared
//! library (dlopened from the pinned Python wheel — see
//! [`super::session::init_runtime`]). The Python `OnnxBackend` facade selects
//! this engine for the `onnx-rs` backend kind; provider resolution policy
//! (intersect-with-available + warnings) stays Python-side, and this crate
//! receives the already-resolved provider list.

use std::sync::Mutex;

use ndarray::{s, Array1, Array2, Ix3};
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};
use tokenizers::Tokenizer;

use super::session::{build_session, int_tensor, l2_normalize};

/// Pooling strategies (mirrors `_POOLINGS` in embedding_onnx.py; "none" is
/// rejected Python-side before construction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Mean-pool the token embeddings.
    Mean,
    /// Take the CLS token embedding.
    Cls,
    /// Take the last token's embedding (last-token models).
    Last,
}

impl Pooling {
    /// Parse a pooling name (`mean`/`cls`/`last`).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] for any other value.
    pub fn parse(s: &str) -> NativeResult<Self> {
        match s {
            "mean" => Ok(Pooling::Mean),
            "cls" => Ok(Pooling::Cls),
            "last" => Ok(Pooling::Last),
            other => Err(NativeError::invalid_input(format!(
                "pooling must be one of mean/cls/last (got {other:?})"
            ))),
        }
    }
}

/// Construction parameters for [`TextEmbedder`] (a text-only ONNX encoder).
pub struct TextEmbedderConfig {
    /// Path to the ONNX model.
    pub model_path: String,
    /// Path to the tokenizer JSON.
    pub tokenizer_path: String,
    /// Already resolved Python-side (intersected with available, CPU appended).
    pub providers: Vec<String>,
    /// Token-pooling strategy.
    pub pooling: Pooling,
    /// Whether to L2-normalize the output (scale-only; not fingerprinted).
    pub normalize: bool,
    /// Tokenizer truncation length.
    pub max_length: usize,
}

/// Which of the standard sentence-transformers inputs the graph declares, and
/// at what integer width (some quantized exports declare int32 ids).
#[derive(Debug, Clone, Copy)]
struct InputSpec {
    wanted: bool,
    int32: bool,
}

struct GraphInputs {
    input_ids: InputSpec,
    attention_mask: InputSpec,
    token_type_ids: InputSpec,
    /// Required inputs outside the supplied set (e.g. position_ids) — reported
    /// at load so the failure mode matches the Python backend's start() check.
    unsupported: Vec<String>,
}

/// A text-only ONNX embedder (tokenize → run → pool → optional normalize).
pub struct TextEmbedder {
    /// Locked because ort 2.0.0-rc.12's run entrypoints all take `&mut self`
    /// (`Session::run<'s, …>(&'s mut self, …) -> Result<SessionOutputs<'s>>`,
    /// likewise `run_with_options`/`run_binding`/`run_async`) — re-checked for
    /// #384; drop the lock if a future ort makes `run` take `&self`.
    session: Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    inputs: GraphInputs,
    pooling: Pooling,
    normalize: bool,
    /// The execution providers actually registered on the session, in order.
    active_providers: Vec<String>,
    dim: Mutex<Option<usize>>,
}

impl TextEmbedder {
    /// Load the ONNX session and tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the model/tokenizer can't be loaded or the session built.
    pub fn load(cfg: TextEmbedderConfig) -> NativeResult<Self> {
        let (session, active) = build_session(&cfg.model_path, &cfg.providers)?;

        let inputs = Self::inspect_inputs(&session);

        let mut tokenizer = Tokenizer::from_file(&cfg.tokenizer_path)
            .context(ErrorKind::Unavailable, "loading tokenizer")?;
        // Pad-token resolution across conventions, mirroring the Python backend:
        // BERT/WordPiece "[PAD]", RoBERTa/BART BPE "<pad>", else id 0.
        let (pad_id, pad_token) = match tokenizer.token_to_id("[PAD]") {
            Some(id) => (id, "[PAD]".to_string()),
            None => match tokenizer.token_to_id("<pad>") {
                Some(id) => (id, "<pad>".to_string()),
                None => (0, "[PAD]".to_string()),
            },
        };
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id,
            pad_token,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: cfg.max_length,
                ..Default::default()
            }))
            .context(ErrorKind::Internal, "truncation")?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            inputs,
            pooling: cfg.pooling,
            normalize: cfg.normalize,
            active_providers: active,
            dim: Mutex::new(None),
        })
    }

    fn inspect_inputs(session: &ort::session::Session) -> GraphInputs {
        use ort::value::{TensorElementType, ValueType};

        let mut spec = GraphInputs {
            input_ids: InputSpec {
                wanted: false,
                int32: false,
            },
            attention_mask: InputSpec {
                wanted: false,
                int32: false,
            },
            token_type_ids: InputSpec {
                wanted: false,
                int32: false,
            },
            unsupported: Vec::new(),
        };
        for input in session.inputs() {
            let int32 = matches!(
                input.dtype(),
                ValueType::Tensor {
                    ty: TensorElementType::Int32,
                    ..
                }
            );
            match input.name() {
                "input_ids" => {
                    spec.input_ids = InputSpec {
                        wanted: true,
                        int32,
                    }
                }
                "attention_mask" => {
                    spec.attention_mask = InputSpec {
                        wanted: true,
                        int32,
                    }
                }
                "token_type_ids" => {
                    spec.token_type_ids = InputSpec {
                        wanted: true,
                        int32,
                    }
                }
                other => spec.unsupported.push(other.to_string()),
            }
        }
        spec
    }

    /// Inputs the graph declares that this engine can't supply (the Python
    /// backend surfaces these in its start()-time diagnostic).
    pub fn unsupported_inputs(&self) -> &[String] {
        &self.inputs.unsupported
    }

    /// The execution providers actually registered, in order.
    pub fn active_providers(&self) -> &[String] {
        &self.active_providers
    }

    /// The embedding width, once known (set by the first embed; readable
    /// beforehand only if the graph declares a static last output dim).
    ///
    /// # Panics
    ///
    /// Panics if the dim mutex is poisoned (a prior holder panicked).
    pub fn dim(&self) -> Option<usize> {
        *self.dim.lock().expect("dim lock poisoned")
    }

    /// Embed one chunk of texts as a single batch — the unit the Python
    /// facade's batch-safety probe and chunking build on.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the ONNX run fails.
    ///
    /// # Panics
    ///
    /// Panics if the session or dim mutex is poisoned.
    pub fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // The GIL is released across this whole call (the binding detaches);
        // this span is the harness's visibility into that window (#308).
        let span = tracing::debug_span!("embed.text_chunk", batch = texts.len());
        let _enter = span.enter();
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .context(ErrorKind::InvalidInput, "tokenization failed")?;

        let batch = encodings.len();
        let seq = encodings[0].get_ids().len();
        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        let mut type_ids = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|&v| v as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&v| v as i64));
            type_ids.extend(enc.get_type_ids().iter().map(|&v| v as i64));
        }

        let mask_arr = Array2::from_shape_vec((batch, seq), mask.clone())
            .context(ErrorKind::Internal, "mask shape")?;

        let mut session = self.session.lock().expect("session lock poisoned");
        let mut feed: Vec<(String, ort::value::DynValue)> = Vec::new();
        let push = |feed: &mut Vec<(String, ort::value::DynValue)>,
                    name: &str,
                    spec: InputSpec,
                    data: &[i64]|
         -> NativeResult<()> {
            if spec.wanted {
                feed.push((name.to_string(), int_tensor(batch, seq, data, spec.int32)?));
            }
            Ok(())
        };
        push(&mut feed, "input_ids", self.inputs.input_ids, &ids)?;
        push(
            &mut feed,
            "attention_mask",
            self.inputs.attention_mask,
            &mask,
        )?;
        push(
            &mut feed,
            "token_type_ids",
            self.inputs.token_type_ids,
            &type_ids,
        )?;

        let outputs = session
            .run(feed)
            .context(ErrorKind::InvalidInput, "onnx run failed")?;
        // The first output, by position — mirroring the Python backend's
        // `session.run(None, feed)[0]`. Pooling reads the borrowed view
        // directly: no owned copy of the full [B,S,H] token tensor (#384).
        let array = outputs[0]
            .try_extract_array::<f32>()
            .context(ErrorKind::Internal, "output extract")?;

        let vectors: Array2<f32> = match array.ndim() {
            3 => {
                let tokens = array
                    .into_dimensionality::<Ix3>()
                    .context(ErrorKind::Internal, "rank-3 view")?;
                pool(tokens, &mask_arr, self.pooling)
            }
            2 => array
                .into_dimensionality::<ndarray::Ix2>()
                .context(ErrorKind::Internal, "rank-2 view")?
                .to_owned(),
            n => {
                return Err(NativeError::invalid_input(format!(
                    "ONNX model first output has rank {n}; expected 2 (pooled) or 3 (tokens)"
                )));
            }
        };

        let vectors = if self.normalize {
            l2_normalize(vectors)
        } else {
            vectors
        };
        *self.dim.lock().expect("dim lock poisoned") = Some(vectors.ncols());
        Ok(vectors.rows().into_iter().map(|r| r.to_vec()).collect())
    }
}

/// The engine contract (#342, route 1): pure chunk-level compute — the host
/// supplies identity + batch policy (`WithPolicy`) and execution (an adapter
/// lane) at composition. `safe_batch` deliberately stays the trait default
/// (1): batch safety is *probed on the loaded model* by the host, never
/// asserted by the engine.
impl shrike_engine_api::EmbedText for TextEmbedder {
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        TextEmbedder::embed_chunk(self, texts)
    }

    fn dim(&self) -> Option<usize> {
        TextEmbedder::dim(self)
    }
}

/// Reduce token embeddings [B,S,H] to sentence vectors [B,H] (mirrors
/// `OnnxBackend._pool`).
fn pool(token_emb: ndarray::ArrayView3<f32>, mask: &Array2<i64>, pooling: Pooling) -> Array2<f32> {
    let (b, s, h) = token_emb.dim();
    match pooling {
        Pooling::Cls => token_emb.slice(s![.., 0, ..]).to_owned(),
        Pooling::Last => {
            let mut out = Array2::<f32>::zeros((b, h));
            for i in 0..b {
                let len: i64 = mask.row(i).sum();
                // The clamp is also the all-padding guard (the counterpart of
                // the mean arm's max(1e-9)): a zero mask row gives len-1 = -1,
                // clamped to token 0 — a deterministic (if meaningless) vector
                // for a degenerate row, never an out-of-bounds index.
                let idx = (len - 1).clamp(0, s as i64 - 1) as usize;
                out.row_mut(i).assign(&token_emb.slice(s![i, idx, ..]));
            }
            out
        }
        Pooling::Mean => {
            let mut out = Array2::<f32>::zeros((b, h));
            for i in 0..b {
                let mut summed = Array1::<f32>::zeros(h);
                let mut count = 0.0f32;
                for j in 0..s {
                    if mask[(i, j)] != 0 {
                        summed += &token_emb.slice(s![i, j, ..]);
                        count += 1.0;
                    }
                }
                let denom = count.max(1e-9);
                out.row_mut(i).assign(&(summed / denom));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pooling_parse() {
        assert_eq!(Pooling::parse("mean").unwrap(), Pooling::Mean);
        assert!(Pooling::parse("none").is_err());
    }

    #[test]
    fn mean_pooling_masks_padding() {
        // [1, 2 tokens, 2 dims]; second token is padding → mean == first token.
        let emb = array![[[1.0f32, 3.0], [100.0, 100.0]]];
        let mask = array![[1i64, 0]];
        let out = pool(emb.view(), &mask, Pooling::Mean);
        assert_eq!(out, array![[1.0f32, 3.0]]);
    }

    #[test]
    fn last_pooling_takes_last_real_token() {
        let emb = array![[[1.0f32, 1.0], [2.0, 2.0], [9.0, 9.0]]];
        let mask = array![[1i64, 1, 0]];
        let out = pool(emb.view(), &mask, Pooling::Last);
        assert_eq!(out, array![[2.0f32, 2.0]]);
    }

    #[test]
    fn last_pooling_all_padding_row_clamps_to_first_token() {
        // A zero mask row (degenerate, but reachable) clamps to token 0 —
        // pinned so the guard in the Last arm doesn't regress to a panic.
        let emb = array![[[7.0f32, 8.0], [9.0, 9.0]]];
        let mask = array![[0i64, 0]];
        let out = pool(emb.view(), &mask, Pooling::Last);
        assert_eq!(out, array![[7.0f32, 8.0]]);
    }
}
