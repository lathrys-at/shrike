//! The OpenAI-compatible remote HTTP engines (feature `remote`): embeddings
//! ([`embed`]) and VLM describe ([`describe`]) over any endpoint, behind ONE
//! shared SSRF-pinned client ([`http`], the dedup of the two engines'
//! previously-copy-pasted trust-boundary code).
//!
//! **Route 2 — async-direct**: the engines implement engine-api's
//! async traits ([`shrike_engine_api::Embedder`]/[`shrike_engine_api::ImageEmbedder`]/
//! [`shrike_engine_api::Recognizer`]) DIRECTLY over an async `reqwest` client,
//! so the kernel awaits them on its runtime (no `Blocking` adapter, no parked
//! blocking-pool thread on a network wait). The SSRF pinning + the per-hop
//! same-host redirect re-vet live in [`shrike_network`] (the one audited home);
//! the engine policy (bounded retry, `Retry-After`, api-key validation,
//! item-level status) lives in [`http::RemoteHttpClient`].

pub mod describe;
pub mod embed;
pub mod http;

pub use describe::{
    compose_fingerprint, RemoteDescriber, RemoteDescriberConfig, DESCRIBE_PROMPT_V1,
    DESCRIBE_PROMPT_VERSION,
};
pub use embed::{LlamaProps, RemoteEmbedder, RemoteEmbedderConfig};
pub use http::ModelInfo;
