//! The in-process ONNX engines (feature `onnx`): ort + tokenizers + pooling.
//!
//! - [`text`]: the sentence-transformers text embedder ([`TextEmbedder`]).
//! - [`clip`]: the CLIP dual-encoder ([`ClipEmbedder`]) — text + image into one
//!   shared space, image preprocessing via `shrike-image`.
//! - [`session`]: the shared ort plumbing (runtime init, session building +
//!   execution-provider resolution, tensor/normalization helpers).
//!
//! Pure Rust: no pyo3 (epic #265 convention 5) — bound to Python in
//! `shrike-pyo3`. The GPU execution providers are additive sub-features
//! (`cuda`/`tensorrt`/`directml`, each implying `onnx`, #384): the server
//! profile keeps the full set; slim builds drop the GPU EP glue.

pub mod clip;
pub mod session;
pub mod text;

pub use clip::{ClipEmbedder, ClipEmbedderConfig, IMAGE_PREP_VERSION_RS};
pub use session::init_runtime;
pub use text::{Pooling, TextEmbedder, TextEmbedderConfig};
