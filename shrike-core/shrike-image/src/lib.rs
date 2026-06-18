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
//! # Features
//!
//! - **`preprocess`** (default): the decode/resize/normalize pipeline
//!   ([`preprocess_to_chw`] & co.). It pulls the `image` crate, so a consumer
//!   that only needs the dependency-free BMP encoder ([`encode_bmp`]) depends
//!   with `default-features = false` and gains NO `image` dependency — that is
//!   how the engine-contract leaf (`shrike-engine-api`'s batch-safety probe)
//!   stays a leaf while reusing the encoder.
//! - **`accel`** (default, requires `preprocess`): the normalize/CHW transform
//!   runs as contiguous per-channel plane passes with the per-channel affine
//!   math hoisted to a fused multiply-add (`p·(1/(255·std)) + (−mean/std)`),
//!   which the compiler autovectorizes — the old strided HWC→CHW scatter did
//!   ~`3·crop²` scalar divides per image (~150k at 224²). It is the *same*
//!   affine math as the scalar reference, so it matches within the golden
//!   image-prep tolerance; off, the transform degrades to a plain scalar loop
//!   (minimal-core). The resize stays `image`'s scalar Catmull-Rom throughout.
//!
//! **A change to the normalize math that cannot match the scalar reference
//! within the golden tolerance must bump [`IMAGE_PREP_VERSION_RS`]** — which
//! invalidates every stored image vector, so it is never a free perf win. (A
//! SIMD resize backend was prototyped but deferred: `fast_image_resize`'s
//! Catmull-Rom differs from `image`'s by ~0.078 on a real downscale, far past
//! the tolerance, so it is genuinely vector-affecting and belongs behind a
//! consumer-fingerprint mechanism, not this crate's default — see #767.)

mod bmp;
pub use bmp::encode_bmp;

#[cfg(feature = "preprocess")]
mod preprocess;
#[cfg(feature = "preprocess")]
pub use preprocess::{
    preprocess_batch_into, preprocess_into_chw, preprocess_to_chw, PreprocessConfig,
};

/// Bump when this pipeline's pixel math changes — folded into the consuming
/// engine's fingerprint so a changed pixel space invalidates stored vectors
/// (the Rust counterpart of the Python engine's `IMAGE_PREP_VERSION`). An
/// accelerated path that cannot match the scalar reference within the golden
/// tolerance is a pixel-math change and MUST bump this. Always available (a
/// plain const, no `image` dependency) so a consumer can read it without the
/// `preprocess` feature.
pub const IMAGE_PREP_VERSION_RS: u32 = 1;
