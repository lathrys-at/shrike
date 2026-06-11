//! Feature-gated engine registration (#342 P5): the embedded host's
//! config→construct→attach, mirroring what the Python facades do — load the
//! engine, run the batch-safety probe (the engine-api port's first embedded
//! consumer), wrap host-assembled identity + policy in `WithPolicy`, adapt
//! with `Inline` (the calling-thread model this surface already uses), and
//! fill the kernel slot. Behind `--features engine-onnx` so the minimal C
//! host links no ort.

use std::ffi::c_char;
use std::sync::Arc;

use shrike_engine_api::{probe, Inline, WithPolicy};

use crate::{arg_str, clear_last_error, set_last_error, ShrikeKernel};

/// Attach the in-process ONNX text engine to an open kernel.
///
/// `ort_dylib_path` points at the onnxruntime shared library to load (the
/// load-dynamic linkage; pass null if a previous call already initialized
/// the process). `pooling` is `mean`/`cls`/`last`. `fingerprint` is the
/// host-assembled identity for index drift (null → engine-derived none —
/// the index then skips the model-change rule). The loaded model is probed
/// for batch safety before attach (serial when batch-variant).
///
/// Returns 0 on success, -1 on failure (see [`crate::shrike_last_error`]).
/// Follow up by driving a reindex (the attach is the #342 slot swap).
///
/// # Safety
/// `handle` must come from [`crate::shrike_kernel_open`]; strings
/// NUL-terminated (or null where documented).
#[no_mangle]
pub unsafe extern "C" fn shrike_attach_embedder_onnx(
    handle: *mut ShrikeKernel,
    ort_dylib_path: *const c_char,
    model_path: *const c_char,
    tokenizer_path: *const c_char,
    pooling: *const c_char,
    normalize: bool,
    max_length: usize,
    fingerprint: *const c_char,
) -> i32 {
    clear_last_error();
    let Some(h) = (unsafe { handle.as_ref() }) else {
        set_last_error("handle must not be null".into());
        return -1;
    };
    let Ok(model) = (unsafe { arg_str(model_path, "model_path") }) else {
        return -1;
    };
    let Ok(tokenizer) = (unsafe { arg_str(tokenizer_path, "tokenizer_path") }) else {
        return -1;
    };
    let Ok(pooling) = (unsafe { arg_str(pooling, "pooling") }) else {
        return -1;
    };
    let fingerprint = if fingerprint.is_null() {
        None
    } else {
        match unsafe { arg_str(fingerprint, "fingerprint") } {
            Ok(v) => Some(v.to_string()),
            Err(()) => return -1,
        }
    };
    if !ort_dylib_path.is_null() {
        let Ok(dylib) = (unsafe { arg_str(ort_dylib_path, "ort_dylib_path") }) else {
            return -1;
        };
        if let Err(e) = shrike_embed::init_runtime(dylib) {
            set_last_error(e.to_string());
            return -1;
        }
    }

    let pooling = match shrike_embed::Pooling::parse(pooling) {
        Ok(p) => p,
        Err(e) => {
            set_last_error(e.to_string());
            return -1;
        }
    };
    let engine = match shrike_embed::TextEmbedder::load(shrike_embed::TextEmbedderConfig {
        model_path: model.to_string(),
        tokenizer_path: tokenizer.to_string(),
        providers: vec!["CPUExecutionProvider".to_string()],
        pooling,
        normalize,
        max_length: max_length.max(1),
    }) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            set_last_error(e.to_string());
            return -1;
        }
    };
    // The batch-safety probe — same policy as every other host (#342 P4a).
    let safe_batch = match probe::probe_max_safe_batch(&engine) {
        Ok(n) => n,
        Err(e) => {
            set_last_error(e.to_string());
            return -1;
        }
    };
    let dim = shrike_embed::TextEmbedder::dim(&engine);
    let tuned = WithPolicy::new(engine, fingerprint, dim, safe_batch);
    h.kernel.attach_embedder(Arc::new(Inline(tuned)), None);
    0
}
