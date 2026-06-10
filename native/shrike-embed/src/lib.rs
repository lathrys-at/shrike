//! Native embedding backends (#270): ort + tokenizers + pooling.
//!
//! The Rust counterpart of `shrike/embedding_onnx.py` — same model layout, same
//! tokenization (the Python `tokenizers` package wraps this same crate), same
//! pooling/normalization semantics, driven through the same onnxruntime shared
//! library (dlopened from the pinned Python wheel — see `init_runtime`). The
//! Python `OnnxBackend` facade selects this engine for the `onnx-rs` backend
//! kind; provider resolution policy (intersect-with-available + warnings) stays
//! Python-side, and this crate receives the already-resolved provider list.
//!
//! Pure Rust: no pyo3 (epic #265 convention 5) — bound to Python in `shrike-py`.

mod clip;

use std::path::Path;
use std::sync::Mutex;

pub use clip::{ClipEmbedder, ClipEmbedderConfig, IMAGE_PREP_VERSION_RS};

use ndarray::{s, Array1, Array2, ArrayD, Axis, Ix3};
use shrike_ffi::{NativeError, NativeResult};
use tokenizers::Tokenizer;

/// Pooling strategies (mirrors `_POOLINGS` in embedding_onnx.py; "none" is
/// rejected Python-side before construction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Mean,
    Cls,
    Last,
}

impl Pooling {
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

/// Initialise the ort runtime from a specific onnxruntime shared library.
///
/// Must be called once per process before the first session is created. The
/// path comes from the installed onnxruntime Python wheel (located by the
/// facade), so the native and Python backends run the *same* runtime build.
pub fn init_runtime(dylib_path: &str) -> NativeResult<()> {
    // commit() returns false when an environment is already committed — fine,
    // init is process-wide and idempotent for our single-dylib use.
    ort::init_from(dylib_path)
        .map_err(|e| NativeError::unavailable(format!("onnxruntime init failed: {e}")))?
        .commit();
    Ok(())
}

pub struct TextEmbedderConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    /// Already resolved Python-side (intersected with available, CPU appended).
    pub providers: Vec<String>,
    pub pooling: Pooling,
    pub normalize: bool,
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

pub struct TextEmbedder {
    session: Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    inputs: GraphInputs,
    pooling: Pooling,
    normalize: bool,
    /// The execution providers actually registered on the session, in order.
    active_providers: Vec<String>,
    dim: Mutex<Option<usize>>,
}

fn map_provider(name: &str) -> Option<ort::ep::ExecutionProviderDispatch> {
    use ort::ep;
    // onnxruntime-python provider names → ort EPs (the names the facade passes
    // are already resolved against onnxruntime.get_available_providers()).
    match name {
        "CPUExecutionProvider" => Some(ep::CPU::default().build()),
        "CoreMLExecutionProvider" => Some(ep::CoreML::default().build()),
        "CUDAExecutionProvider" => Some(ep::CUDA::default().build()),
        "TensorrtExecutionProvider" => Some(ep::TensorRT::default().build()),
        "DmlExecutionProvider" => Some(ep::DirectML::default().build()),
        _ => None,
    }
}

/// A graph input this engine supplies, at its declared integer width
/// (some quantized exports declare int32 ids).
pub(crate) struct GraphInput {
    pub(crate) name: String,
    pub(crate) int32: bool,
}

/// Build an ort session for a model with the (already Python-resolved)
/// execution providers registered. Returns the session and the providers
/// actually registered, in order.
pub(crate) fn build_session(
    model_path: &str,
    providers: &[String],
) -> NativeResult<(ort::session::Session, Vec<String>)> {
    if !Path::new(model_path).is_file() {
        return Err(NativeError::unavailable(format!(
            "ONNX model not found: {model_path}"
        )));
    }
    let mut builder = ort::session::Session::builder()
        .map_err(|e| NativeError::internal(format!("session builder: {e}")))?;
    let mut active: Vec<String> = Vec::new();
    for name in providers {
        // The list is already resolved Python-side; mapping is defence in
        // depth (an unmappable name degrades to CPU, which is always last).
        if let Some(ep) = map_provider(name) {
            builder = builder
                .with_execution_providers([ep])
                .map_err(|e| NativeError::unavailable(format!("provider {name}: {e}")))?;
            active.push(name.clone());
        }
    }
    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| NativeError::unavailable(format!("loading {model_path}: {e}")))?;
    Ok((session, active))
}

/// The graph's declared inputs among `supplied`, at their declared int widths.
pub(crate) fn graph_inputs(session: &ort::session::Session, supplied: &[&str]) -> Vec<GraphInput> {
    use ort::value::{TensorElementType, ValueType};

    session
        .inputs()
        .iter()
        .filter(|i| supplied.contains(&i.name()))
        .map(|i| GraphInput {
            name: i.name().to_string(),
            int32: matches!(
                i.dtype(),
                ValueType::Tensor {
                    ty: TensorElementType::Int32,
                    ..
                }
            ),
        })
        .collect()
}

/// An integer [batch, seq] tensor at the input's declared width.
pub(crate) fn int_tensor(
    batch: usize,
    seq: usize,
    data: &[i64],
    int32: bool,
) -> NativeResult<ort::value::DynValue> {
    let value = if int32 {
        let v32: Vec<i32> = data.iter().map(|&v| v as i32).collect();
        ort::value::Tensor::from_array(([batch, seq], v32))
            .map_err(|e| NativeError::internal(format!("int tensor: {e}")))?
            .into_dyn()
    } else {
        ort::value::Tensor::from_array(([batch, seq], data.to_vec()))
            .map_err(|e| NativeError::internal(format!("int tensor: {e}")))?
            .into_dyn()
    };
    Ok(value)
}

/// The first output as a [batch, dim] f32 matrix (already-pooled exports).
pub(crate) fn extract_2d(outputs: &ort::session::SessionOutputs<'_>) -> NativeResult<Array2<f32>> {
    let array: ArrayD<f32> = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| NativeError::internal(format!("output extract: {e}")))?
        .to_owned();
    array
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| NativeError::invalid_input(format!("expected a rank-2 output: {e}")))
}

/// Backwards-compat alias used by both engines.
pub(crate) use l2_normalize as l2_normalize_rows;

impl TextEmbedder {
    pub fn load(cfg: TextEmbedderConfig) -> NativeResult<Self> {
        let (session, active) = build_session(&cfg.model_path, &cfg.providers)?;

        let inputs = Self::inspect_inputs(&session);

        let mut tokenizer = Tokenizer::from_file(&cfg.tokenizer_path)
            .map_err(|e| NativeError::unavailable(format!("loading tokenizer: {e}")))?;
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
            .map_err(|e| NativeError::internal(format!("truncation: {e}")))?;

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

    pub fn active_providers(&self) -> &[String] {
        &self.active_providers
    }

    /// The embedding width, once known (set by the first embed; readable
    /// beforehand only if the graph declares a static last output dim).
    pub fn dim(&self) -> Option<usize> {
        *self.dim.lock().expect("dim lock poisoned")
    }

    /// Embed one chunk of texts as a single batch — the unit the Python
    /// facade's batch-safety probe and chunking build on.
    pub fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
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
        let mut type_ids = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|&v| v as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&v| v as i64));
            type_ids.extend(enc.get_type_ids().iter().map(|&v| v as i64));
        }

        let mask_arr = Array2::from_shape_vec((batch, seq), mask.clone())
            .map_err(|e| NativeError::internal(format!("mask shape: {e}")))?;

        let mut session = self.session.lock().expect("session lock poisoned");
        let mut feed: Vec<(String, ort::value::DynValue)> = Vec::new();
        let push = |feed: &mut Vec<(String, ort::value::DynValue)>,
                    name: &str,
                    spec: InputSpec,
                    data: &[i64]|
         -> NativeResult<()> {
            if !spec.wanted {
                return Ok(());
            }
            let value: ort::value::DynValue = if spec.int32 {
                let v32: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                ort::value::Tensor::from_array(([batch, seq], v32))
                    .map_err(|e| NativeError::internal(format!("tensor {name}: {e}")))?
                    .into_dyn()
            } else {
                ort::value::Tensor::from_array(([batch, seq], data.to_vec()))
                    .map_err(|e| NativeError::internal(format!("tensor {name}: {e}")))?
                    .into_dyn()
            };
            feed.push((name.to_string(), value));
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
            .map_err(|e| NativeError::invalid_input(format!("onnx run failed: {e}")))?;
        // The first output, by position — mirroring the Python backend's
        // `session.run(None, feed)[0]`.
        let array: ArrayD<f32> = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| NativeError::internal(format!("output extract: {e}")))?
            .to_owned();

        let vectors: Array2<f32> = match array.ndim() {
            3 => {
                let tokens = array
                    .into_dimensionality::<Ix3>()
                    .map_err(|e| NativeError::internal(format!("rank-3 view: {e}")))?;
                pool(&tokens, &mask_arr, self.pooling)
            }
            2 => array
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| NativeError::internal(format!("rank-2 view: {e}")))?,
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

/// Reduce token embeddings [B,S,H] to sentence vectors [B,H] (mirrors
/// `OnnxBackend._pool`).
fn pool(token_emb: &ndarray::Array3<f32>, mask: &Array2<i64>, pooling: Pooling) -> Array2<f32> {
    let (b, s, h) = token_emb.dim();
    match pooling {
        Pooling::Cls => token_emb.slice(s![.., 0, ..]).to_owned(),
        Pooling::Last => {
            let mut out = Array2::<f32>::zeros((b, h));
            for i in 0..b {
                let len: i64 = mask.row(i).sum();
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

/// L2-normalize each row (mirrors the numpy `clip(norm, 1e-12)` guard).
fn l2_normalize(mut vectors: Array2<f32>) -> Array2<f32> {
    for mut row in vectors.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|v| v / norm);
    }
    vectors
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
        let out = pool(&emb, &mask, Pooling::Mean);
        assert_eq!(out, array![[1.0f32, 3.0]]);
    }

    #[test]
    fn last_pooling_takes_last_real_token() {
        let emb = array![[[1.0f32, 1.0], [2.0, 2.0], [9.0, 9.0]]];
        let mask = array![[1i64, 1, 0]];
        let out = pool(&emb, &mask, Pooling::Last);
        assert_eq!(out, array![[2.0f32, 2.0]]);
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let v = l2_normalize(array![[3.0f32, 4.0]]);
        assert!((v[(0, 0)] - 0.6).abs() < 1e-6);
        assert!((v[(0, 1)] - 0.8).abs() < 1e-6);
    }
}
