//! The Shrike engine crate (#708): every engine-contract implementation, one
//! crate, feature-gated by family. The kernel never names it — it composes the
//! `shrike-engine-api` traits these engines implement (the kernel↔engine
//! firewall; `shrike-engine-api` stays a separate thin contract crate).
//!
//! - [`onnx`] (feature `onnx`): in-process ort engines — text + CLIP dual
//!   encoder. The GPU execution providers are additive sub-features
//!   (`cuda`/`tensorrt`/`directml`, each implying `onnx`).
//! - [`remote`] (feature `remote`): OpenAI-compatible HTTP engines — embeddings
//!   ([`remote::embed`]) and VLM describe ([`remote::describe`]) over a shared,
//!   SSRF-pinned HTTP client ([`remote::http`]). `ureq` (synchronous,
//!   runtime-less); the kernel's `Blocking` adapter moves each request onto the
//!   blocking pool.
//! - `apple` (feature `engine-apple`): the Apple Vision OCR + SpeechAnalyzer
//!   ASR `RecognizeMedia` impls land in #709 (slice 2), over `shrike-platform`'s
//!   raw Swift/C-ABI glue — the swiftc toolchain rides there (pulled only when
//!   `engine-apple` is on), so this crate stays `build.rs`-free.
//!
//! Pure Rust — NO pyo3 (epic #265 convention 5); bound to Python in
//! `shrike-pyo3`, to the C ABI in the mobile binding.

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "remote")]
pub mod remote;
