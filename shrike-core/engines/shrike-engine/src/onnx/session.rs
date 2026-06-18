//! Shared ort plumbing for the onnx engines: runtime init, session
//! building with execution-provider resolution, the small input/tensor
//! helpers, and L2 normalization. Both the text engine ([`super::text`]) and
//! the CLIP dual-encoder ([`super::clip`]) build on these.

use std::path::Path;

use ndarray::{Array2, ArrayD, Axis};
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};

/// Initialise the ort runtime from a specific onnxruntime shared library.
///
/// Must be called once per process before the first session is created. The
/// path comes from the installed onnxruntime Python wheel (located by the
/// facade), so the native and Python backends run the *same* runtime build.
///
/// # Errors
///
/// Returns [`ErrorKind::Unavailable`] if the onnxruntime library can't be initialised.
pub fn init_runtime(dylib_path: &str) -> NativeResult<()> {
    // commit() returns false when an environment is already committed — fine,
    // init is process-wide and idempotent for our single-dylib use.
    ort::init_from(dylib_path)
        .context(ErrorKind::Unavailable, "onnxruntime init failed")?
        .commit();
    Ok(())
}

fn map_provider(name: &str) -> Option<ort::ep::ExecutionProviderDispatch> {
    use ort::ep;
    // onnxruntime-python provider names → ort EPs (the names the facade passes
    // are already resolved against onnxruntime.get_available_providers()).
    // The GPU EPs are feature-gated: a slim build with the
    // feature off degrades that name to None → CPU, which is always last.
    match name {
        "CPUExecutionProvider" => Some(ep::CPU::default().build()),
        "CoreMLExecutionProvider" => Some(ep::CoreML::default().build()),
        #[cfg(feature = "cuda")]
        "CUDAExecutionProvider" => Some(ep::CUDA::default().build()),
        #[cfg(feature = "tensorrt")]
        "TensorrtExecutionProvider" => Some(ep::TensorRT::default().build()),
        #[cfg(feature = "directml")]
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
    let mut builder =
        ort::session::Session::builder().context(ErrorKind::Internal, "session builder")?;
    let mut active: Vec<String> = Vec::new();
    for name in providers {
        // The list is already resolved Python-side; mapping is defence in
        // depth (an unmappable name degrades to CPU, which is always last).
        if let Some(ep) = map_provider(name) {
            // ort's builder errors embed the (non-'static) SessionBuilder, so
            // they aren't Into<BoxError> — render with Display, no source chain.
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
            .context(ErrorKind::Internal, "int tensor")?
            .into_dyn()
    } else {
        ort::value::Tensor::from_array(([batch, seq], data.to_vec()))
            .context(ErrorKind::Internal, "int tensor")?
            .into_dyn()
    };
    Ok(value)
}

/// The first output as a [batch, dim] f32 matrix (already-pooled exports).
pub(crate) fn extract_2d(outputs: &ort::session::SessionOutputs<'_>) -> NativeResult<Array2<f32>> {
    let array: ArrayD<f32> = outputs[0]
        .try_extract_array::<f32>()
        .context(ErrorKind::Internal, "output extract")?
        .to_owned();
    array
        .into_dimensionality::<ndarray::Ix2>()
        .context(ErrorKind::InvalidInput, "expected a rank-2 output")
}

/// L2-normalize each row (mirrors the numpy `clip(norm, 1e-12)` guard).
pub(crate) fn l2_normalize(mut vectors: Array2<f32>) -> Array2<f32> {
    for mut row in vectors.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|v| v / norm);
    }
    vectors
}

/// Alias used by both engines.
pub(crate) use l2_normalize as l2_normalize_rows;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn l2_normalize_unit_norm() {
        let v = l2_normalize(array![[3.0f32, 4.0]]);
        assert!((v[(0, 0)] - 0.6).abs() < 1e-6);
        assert!((v[(0, 1)] - 0.8).abs() < 1e-6);
    }
}
