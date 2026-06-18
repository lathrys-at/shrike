//! The OpenAI-compatible remote HTTP engines (feature `remote`): embeddings
//! ([`embed`]) and VLM describe ([`describe`]) over any endpoint, behind ONE
//! shared SSRF-pinned client ([`http`], the #708 dedup of the two engines'
//! previously-copy-pasted trust-boundary code).
//!
//! Sync `ureq` (runtime-less); the kernel's `Blocking` adapter moves each
//! request onto the blocking pool. The async-first port (reqwest over
//! `shrike-network`'s async IP-pinned connector) is #721 — it swaps the
//! transport inside [`http::RemoteHttpClient`].

pub mod describe;
pub mod embed;
pub mod http;

pub use describe::{
    compose_fingerprint, RemoteDescriber, RemoteDescriberConfig, DESCRIBE_PROMPT_V1,
    DESCRIBE_PROMPT_VERSION,
};
pub use embed::{LlamaProps, RemoteEmbedder, RemoteEmbedderConfig};
pub use http::ModelInfo;
