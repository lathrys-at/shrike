//! The Shrike engine crate: every engine-contract implementation, one
//! crate, feature-gated by family. The kernel never names it — it composes the
//! `shrike-engine-api` traits these engines implement (the kernel↔engine
//! firewall; `shrike-engine-api` stays a separate thin contract crate).
//!
//! - [`onnx`] (feature `onnx`): in-process ort engines — text + CLIP dual
//!   encoder. The GPU execution providers are additive sub-features
//!   (`cuda`/`tensorrt`/`directml`, each implying `onnx`).
//! - [`remote`] (feature `remote`): OpenAI-compatible HTTP engines — embeddings
//!   ([`remote::embed`]) and VLM describe ([`remote::describe`]) over a shared,
//!   SSRF-pinned async HTTP client ([`remote::http`]). Route 2 async-direct
//!   (#721 S2): these engines implement the async `Embedder`/`Recognizer` traits
//!   directly over `reqwest`, so the kernel awaits them on its runtime — no
//!   `Blocking` adapter.
//! - [`apple`] (feature `engine-apple`): the Apple Vision OCR + SpeechAnalyzer
//!   ASR `RecognizeMedia` impls, over `shrike-platform`'s raw Swift/C-ABI glue
//!   — the swiftc toolchain rides THERE (pulled only when `engine-apple` is on),
//!   so this crate stays `build.rs`-free. This layer parses the glue's raw JSON
//!   into engine-api types.
//!
//! Pure Rust — NO pyo3; bound to Python in
//! `shrike-pyo3`, to the C ABI in the mobile binding.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "remote")]
pub mod remote;

#[cfg(feature = "engine-apple")]
pub mod apple;
