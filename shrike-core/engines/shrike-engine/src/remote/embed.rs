//! The generic remote-embeddings engine (#342 P4): an OpenAI-compatible
//! embeddings endpoint — llama-server locally, a cloud embedding API with a
//! key, a service across a tailnet — as a **route-2 async-direct engine** (#721
//! S2): it implements engine-api's async [`Embedder`]/[`ImageEmbedder`] traits
//! directly over the async `reqwest` client, so the kernel awaits it on its
//! runtime (no `Blocking` adapter, no parked blocking-pool thread). The shared
//! SSRF/retry/api-key machinery lives in [`super::http`] (#708 dedup); this
//! module holds only the embeddings-specific dialects.
//!
//! Since #501 it also speaks llama.cpp's NATIVE multimodal dialect for
//! [`ImageEmbedder`]: `GET /props` advertises the loaded model's modalities
//! and the per-process `media_marker` (randomized each server start — it
//! must be read, never assumed), and media embeds ride
//! `POST /embeddings` with `{"content": {"prompt_string": <marker>,
//! "multimodal_data": ["<base64>"]}}` — NOT the OpenAI `/v1/embeddings`
//! shape, which stays the text path on every endpoint. The dialect is
//! probe-gated: an endpoint without `/props` (a cloud API) simply has no
//! media path, and a llama-server without the right mmproj refuses cleanly
//! before any payload is sent.
//!
//! Scope discipline: this crate **talks to an endpoint**, nothing else.
//! Launching/managing a llama-server subprocess is a different concern
//! (`shrike-llama-server`); fingerprint *assembly* (the `pool=`/`args=`/
//! `textprep=` policy suffixes) stays host-side — this crate only serves the
//! raw identity ingredients (`/v1/models` id + meta, `/props` capabilities).

use std::sync::OnceLock;
use std::time::Duration;

use base64::Engine as _;
use futures::future::BoxFuture;
use serde::Deserialize;
use shrike_engine_api::{Embedder, ImageEmbedder, MediaItem};
use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};
use tracing::Instrument as _;

use super::http::{ModelInfo, PostOutcome, RemoteHttpClient};

/// Per-request ceiling, matching the Python backend's httpx timeout.
const EMBED_TIMEOUT: Duration = Duration::from_secs(60);

/// Bounded retry on the embed path (the request is idempotent): cloud
/// endpoints 429/503 routinely (rate limits, cold scale-up), so a transient
/// failure must not sink the chunk. Mirrors the probe's small explicit
/// attempts loop.
const EMBED_ATTEMPTS: u32 = 3;

/// Embeds have NO item-level rejection: any non-transient status is an
/// endpoint/config problem that errors the chunk (unlike describe, where a 4xx
/// can condemn a single image). The shared retry loop takes this predicate.
fn no_item_level(_code: u16) -> bool {
    false
}

/// Construction parameters for [`RemoteEmbedder`] (embeddings over an HTTP endpoint).
pub struct RemoteEmbedderConfig {
    /// e.g. `http://127.0.0.1:8373` (no trailing slash needed).
    pub base_url: String,
    /// Optional bearer token — the API-key seam for cloud services
    /// (config-supplied; no key management here).
    pub api_key: Option<String>,
    /// Pin the model in each request body so a multi-model endpoint resolves
    /// the right one (a single-model llama-server ignores it). `None` omits
    /// the field.
    pub model: Option<String>,
    /// The per-request text-batch size — the host's probed + capped safe batch
    /// (#721 S2: a route-2 engine owns its own request chunking; the host passes
    /// the size it determined). `None`/0 = one request for the whole input.
    /// Images always ride one request per item (#501).
    pub batch_size: Option<usize>,
}

/// The engine: a thin, stateless HTTP client wrapper.
///
/// SSRF posture lives in [`RemoteHttpClient`] (#592): the client is pinned to
/// `base_url`'s resolved IP with auto-redirects OFF, and a redirect is followed
/// only when it stays on the SAME host.
pub struct RemoteEmbedder {
    http: RemoteHttpClient,
    model: Option<String>,
    /// The per-request text-batch size; `None` = the whole input in one request.
    batch_size: Option<usize>,
    /// The endpoint's resolved multimodal capabilities, cached after the first
    /// successful `GET /props` (#708): the marker + vision flag are per-process
    /// invariants of the endpoint, so the image path reads them ONCE rather than
    /// re-probing per chunk (the documented per-chunk round-trip fix). Only a
    /// SUCCESSFUL probe is cached — a text-only/absent `/props` falls through to
    /// re-probe, preserving the original error behaviour exactly.
    props: OnceLock<LlamaProps>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    #[serde(default)]
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

/// llama.cpp server capabilities from `GET /props` (#501): which modalities
/// the loaded model serves (an mmproj per modality), and the per-process
/// `media_marker` a multimodal prompt references. The marker is randomized
/// at every server start, so it is read here, never assumed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlamaProps {
    /// Whether the endpoint serves a vision modality.
    pub vision: bool,
    /// Whether the endpoint serves an audio modality.
    pub audio: bool,
    /// The per-process media marker a multimodal prompt references.
    pub media_marker: Option<String>,
}

impl RemoteEmbedder {
    /// Construction validates the API key (header-injection guard, in
    /// [`RemoteHttpClient::new`]) and pins the endpoint's IP.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is invalid or the endpoint host can't be resolved/pinned.
    pub fn new(cfg: RemoteEmbedderConfig) -> NativeResult<Self> {
        let http = RemoteHttpClient::new(&cfg.base_url, cfg.api_key, EMBED_TIMEOUT)?;
        Ok(Self {
            http,
            model: cfg.model,
            batch_size: cfg.batch_size.filter(|&n| n > 1),
            props: OnceLock::new(),
        })
    }

    /// `GET /health` is 200 — llama-server's readiness.
    pub async fn health_ok(&self) -> bool {
        self.http.health_ok().await
    }

    /// llama.cpp's `GET /props` capabilities, or `None` when the endpoint
    /// doesn't serve the route (a cloud API, an old build) — which means
    /// "no native multimodal dialect here", never an error. This is the raw
    /// probe; the image path caches it (see [`Self::resolved_props`]).
    pub async fn props(&self) -> Option<LlamaProps> {
        let body = self.http.get_json("/props").await?;
        let modalities = body.get("modalities")?;
        Some(LlamaProps {
            vision: modalities
                .get("vision")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            audio: modalities
                .get("audio")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            media_marker: body
                .get("media_marker")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }

    /// `/props`, read once and cached (#708): the marker + capabilities are
    /// per-process invariants of the endpoint, so the image path probes the
    /// first time and reuses thereafter instead of paying a round-trip per
    /// chunk. Only a SUCCESSFUL probe is cached — `None` (no `/props`) falls
    /// through to re-probe next chunk, identical to the original behaviour.
    async fn resolved_props(&self) -> Option<LlamaProps> {
        if let Some(cached) = self.props.get() {
            return Some(cached.clone());
        }
        let fresh = self.props().await?;
        // Race-tolerant: a concurrent probe may win the set; either value is the
        // same per-process invariant, so take whichever landed.
        let _ = self.props.set(fresh.clone());
        Some(self.props.get().cloned().unwrap_or(fresh))
    }

    /// The first `/v1/models` entry's id + meta (empty on any failure or
    /// shape mismatch — identity falls back to host config, never errors).
    pub async fn model_info(&self) -> ModelInfo {
        self.http.model_info().await
    }

    /// One `POST /v1/embeddings` request for a chunk of texts. Vectors are
    /// ordered by the response's own `index` rather than positional order — a
    /// cheap guard that survives a backend that doesn't preserve order (each
    /// note would otherwise silently get a batch-mate's vector).
    async fn embed_one_request(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        let mut payload = serde_json::json!({ "input": texts });
        if let Some(model) = &self.model {
            payload["model"] = serde_json::Value::String(model.clone());
        }
        let resp = match self
            .http
            .post_json_with_retry(
                "/v1/embeddings",
                &payload,
                EMBED_ATTEMPTS,
                EMBED_TIMEOUT,
                "embeddings",
                no_item_level,
            )
            .await?
        {
            PostOutcome::Response(resp) => resp,
            // embed's predicate never condemns an item, so this is unreachable.
            PostOutcome::ItemRejected => {
                return Err(NativeError::internal(
                    "embeddings request unexpectedly item-rejected",
                ))
            }
        };
        let body: EmbeddingsResponse = resp
            .into_json()
            .context(ErrorKind::Internal, "malformed embeddings response")?;
        if body.data.len() != texts.len() {
            return Err(NativeError::internal(format!(
                "endpoint returned {} embeddings for {} inputs",
                body.data.len(),
                texts.len()
            )));
        }
        let mut items = body.data;
        items.sort_by_key(|d| d.index);
        // All vectors must share one width — a mixed-width response is a
        // malformed service response (the index would reject or corrupt on
        // it). No expected dim lives in this crate (that's host policy, in
        // `WithPolicy`), so uniformity is the invariant asserted here.
        let width = items[0].embedding.len();
        if let Some(bad) = items.iter().find(|d| d.embedding.len() != width) {
            return Err(NativeError::internal(format!(
                "endpoint returned mixed-width embeddings ({width} vs {})",
                bad.embedding.len()
            )));
        }
        Ok(items.into_iter().map(|d| d.embedding).collect())
    }
}

impl Embedder for RemoteEmbedder {
    /// Embed the whole batch, chunked by the host's `batch_size` (one request
    /// per chunk, in order). A route-2 engine owns its own chunking (#721 S2):
    /// the old `Blocking` adapter's `safe_batch` chunk loop moves in here.
    fn embed(&self, texts: Vec<String>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(
            async move {
                if texts.is_empty() {
                    return Ok(Vec::new());
                }
                let chunk = self.batch_size.unwrap_or(texts.len()).max(1);
                let mut out = Vec::with_capacity(texts.len());
                for piece in texts.chunks(chunk) {
                    out.extend(self.embed_one_request(piece).await?);
                }
                Ok(out)
            }
            .instrument(tracing::debug_span!("embed.remote")),
        )
    }
}

/// llama.cpp's NATIVE `/embeddings` response: a bare array, each item's
/// `embedding` nested one level deeper than the OpenAI shape (a list of
/// pooled vectors, one per sequence — one sequence per request here).
#[derive(Deserialize)]
struct NativeEmbeddingItem {
    embedding: Vec<Vec<f32>>,
}

impl ImageEmbedder for RemoteEmbedder {
    /// One native `/embeddings` request **per item** (#501): media payloads
    /// are orders of magnitude heavier than text, so per-item requests keep
    /// the retry/backoff semantics simple and attribute a failure to the
    /// exact image. Capability-gated up front — a model without the vision
    /// mmproj refuses here with the actionable error instead of paying the
    /// payload upload and the server's own 500.
    fn embed_images(&self, images: Vec<MediaItem>) -> BoxFuture<'_, NativeResult<Vec<Vec<f32>>>> {
        Box::pin(
            async move {
                if images.is_empty() {
                    return Ok(Vec::new());
                }
                // `/props` is read ONCE at the first image chunk and cached (#708)
                // — the marker + capabilities are per-process invariants of the
                // endpoint, so a full reindex no longer re-probes per chunk.
                let props = self.resolved_props().await.ok_or_else(|| {
                    NativeError::unavailable(
                        "endpoint does not serve llama.cpp's /props — image embeddings need its \
                         native multimodal dialect (an OpenAI-style endpoint has no media path)",
                    )
                })?;
                if !props.vision {
                    return Err(NativeError::unavailable(
                        "the loaded model does not serve image embeddings — llama-server needs the \
                         model's vision mmproj loaded (--mmproj; managed.llama_server in config)",
                    ));
                }
                let marker = props.media_marker.ok_or_else(|| {
                    NativeError::internal("/props advertises vision but carries no media_marker")
                })?;
                let mut out = Vec::with_capacity(images.len());
                for item in &images {
                    out.push(self.embed_one_media(&marker, item).await?);
                }
                Ok(out)
            }
            .instrument(tracing::debug_span!("embed.remote_media")),
        )
    }
}

impl RemoteEmbedder {
    async fn embed_one_media(&self, marker: &str, item: &MediaItem) -> NativeResult<Vec<f32>> {
        let payload = serde_json::json!({
            "content": {
                "prompt_string": marker,
                "multimodal_data": [base64::engine::general_purpose::STANDARD.encode(&item.bytes)],
            }
        });
        let resp = match self
            .http
            .post_json_with_retry(
                "/embeddings",
                &payload,
                EMBED_ATTEMPTS,
                EMBED_TIMEOUT,
                "embeddings",
                no_item_level,
            )
            .await?
        {
            PostOutcome::Response(resp) => resp,
            PostOutcome::ItemRejected => {
                return Err(NativeError::internal(
                    "embeddings request unexpectedly item-rejected",
                ))
            }
        };
        let mut body: Vec<NativeEmbeddingItem> = resp
            .into_json()
            .context(ErrorKind::Internal, "malformed native embeddings response")?;
        // One sequence per request → exactly one outer item, exactly one
        // pooled inner vector. A multi-inner response means the server is
        // running `--pooling none` (per-token vectors), which would silently
        // index only the first token — reject it, the symmetry with
        // `embed_chunk`'s mixed-width guard. An empty inner is a zero-width
        // vector the index can't hold (the text path's `ndim == 0` guard).
        let item = body
            .pop()
            .ok_or_else(|| NativeError::internal("native embeddings response carried no vector"))?;
        if !body.is_empty() {
            return Err(NativeError::internal(format!(
                "native embeddings response carried {} items for one input",
                body.len() + 1
            )));
        }
        if item.embedding.len() != 1 {
            return Err(NativeError::internal(format!(
                "native embeddings response carried {} pooled vectors for one image \
                 (a multi-vector response means the server runs --pooling none)",
                item.embedding.len()
            )));
        }
        let vector = item.embedding.into_iter().next().unwrap();
        if vector.is_empty() {
            return Err(NativeError::internal(
                "native embeddings response carried a zero-width vector",
            ));
        }
        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::super::http::test_server::{canned_server, one_shot_server};
    use super::*;

    fn engine(base_url: String, model: Option<&str>, key: Option<&str>) -> RemoteEmbedder {
        RemoteEmbedder::new(RemoteEmbedderConfig {
            base_url,
            api_key: key.map(String::from),
            model: model.map(String::from),
            batch_size: None,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn embeds_sorting_by_response_index_and_pinning_model() {
        // Out-of-order `data` must land back in input order via `index`.
        let body = serde_json::json!({"data": [
            {"index": 1, "embedding": [2.0, 2.0]},
            {"index": 0, "embedding": [1.0, 1.0]},
        ]})
        .to_string();
        let (url, rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let out = engine(url, Some("minilm"), Some("sk-test"))
            .embed(vec!["a".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(out, vec![vec![1.0, 1.0], vec![2.0, 2.0]]);
        let raw = rx.recv().unwrap();
        assert!(raw.starts_with("POST /v1/embeddings"), "{raw}");
        assert!(raw.contains("\"model\":\"minilm\""), "model pinned: {raw}");
        // reqwest/hyper emits header names lowercased (HTTP headers are
        // case-insensitive); match case-insensitively on the bearer line.
        assert!(
            raw.to_ascii_lowercase()
                .contains("authorization: bearer sk-test"),
            "auth header: {raw}"
        );
    }

    #[tokio::test]
    async fn arity_mismatch_is_an_internal_error() {
        let body = serde_json::json!({"data": [{"index": 0, "embedding": [1.0]}]}).to_string();
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let err = engine(url, None, None)
            .embed(vec!["a".into(), "b".into()])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("1 embeddings for 2 inputs"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn retries_through_transient_503() {
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).to_string();
        let (url, rx) = canned_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "{}".into()),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let out = engine(url, None, None)
            .embed(vec!["a".into()])
            .await
            .unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0]]);
        // Both attempts actually reached the server.
        assert!(rx.recv().unwrap().starts_with("POST /v1/embeddings"));
        assert!(rx.recv().unwrap().starts_with("POST /v1/embeddings"));
    }

    #[tokio::test]
    async fn honors_retry_after_on_429() {
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).to_string();
        let (url, _rx) = canned_server(vec![
            (
                "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1",
                "{}".into(),
            ),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let started = std::time::Instant::now();
        let out = engine(url, None, None)
            .embed(vec!["a".into()])
            .await
            .unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0]]);
        // The server-requested 1s overrides the 250ms default backoff — the
        // lower bound proves the header was honored, and can't flake.
        assert!(
            started.elapsed() >= Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn bad_request_does_not_retry() {
        // A 200 is queued behind the 400 — a retry would succeed, so the
        // error proves the 400 failed immediately.
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).to_string();
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 400 Bad Request", "{}".into()),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let err = engine(url, None, None)
            .embed(vec!["a".into()])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("embeddings request failed"),
            "{err}"
        );
        assert!(!err.to_string().contains("attempt"), "{err}");
    }

    #[tokio::test]
    async fn exhausted_retries_map_to_unavailable() {
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "{}".to_string());
            3
        ]);
        let err = engine(url, None, None)
            .embed(vec!["a".into()])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("after 3 attempt(s)"), "{err}");
    }

    #[tokio::test]
    async fn mixed_width_response_is_an_internal_error() {
        let body = serde_json::json!({"data": [
            {"index": 0, "embedding": [1.0, 2.0]},
            {"index": 1, "embedding": [3.0]},
        ]})
        .to_string();
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let err = engine(url, None, None)
            .embed(vec!["a".into(), "b".into()])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("mixed-width embeddings (2 vs 1)"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn bad_api_key_rejected_at_construction() {
        for key in ["sk\r\nX-Injected: 1", " sk-test", "sk-test ", "sk\ttest"] {
            // `.err()` rather than `.unwrap_err()`: the engine is deliberately
            // not `Debug` (the struct holds the API key).
            let err = RemoteEmbedder::new(RemoteEmbedderConfig {
                base_url: "http://127.0.0.1:9".into(),
                api_key: Some(key.into()),
                model: None,
                batch_size: None,
            })
            .err()
            .expect("construction must reject the key");
            assert!(err.to_string().contains("api_key"), "{key:?}: {err}");
        }
    }

    #[tokio::test]
    async fn connection_refused_maps_to_unavailable() {
        // An unbound port: connection refused, no server.
        let err = engine("http://127.0.0.1:9".into(), None, None)
            .embed(vec!["a".into()])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("embeddings request failed"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn model_info_reads_id_and_meta_and_defaults_empty() {
        let body = serde_json::json!({"data": [{
            "id": "all-minilm",
            "meta": {"n_embd": 384, "n_params": 22713216},
        }]})
        .to_string();
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let info = engine(url, None, None).model_info().await;
        assert_eq!(info.id.as_deref(), Some("all-minilm"));
        assert_eq!(info.meta["n_embd"], 384);
        // And the graceful default on a down endpoint.
        assert_eq!(
            engine("http://127.0.0.1:9".into(), None, None)
                .model_info()
                .await,
            ModelInfo::default()
        );
    }

    #[tokio::test]
    async fn health_ok_only_on_200() {
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", "{}".into());
        assert!(engine(url, None, None).health_ok().await);
        assert!(
            !engine("http://127.0.0.1:9".into(), None, None)
                .health_ok()
                .await
        );
    }

    #[tokio::test]
    async fn empty_input_short_circuits() {
        // No server at all — must not even attempt a request.
        let out = engine("http://127.0.0.1:9".into(), None, None)
            .embed(vec![])
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn batch_size_chunks_into_multiple_requests() {
        // batch_size=2 over 3 inputs → two requests (2 then 1), order preserved.
        let r1 = serde_json::json!({"data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]})
        .to_string();
        let r2 = serde_json::json!({"data": [{"index": 0, "embedding": [3.0]}]}).to_string();
        let (url, rx) = canned_server(vec![("HTTP/1.1 200 OK", r1), ("HTTP/1.1 200 OK", r2)]);
        let eng = RemoteEmbedder::new(RemoteEmbedderConfig {
            base_url: url,
            api_key: None,
            model: None,
            batch_size: Some(2),
        })
        .unwrap();
        let out = eng
            .embed(vec!["a".into(), "b".into(), "c".into()])
            .await
            .unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0], vec![3.0]]);
        // Two distinct requests reached the server (the chunk split).
        assert!(rx.recv().unwrap().starts_with("POST /v1/embeddings"));
        assert!(rx.recv().unwrap().starts_with("POST /v1/embeddings"));
    }

    // ── SSRF redirect re-vet (#592) ─────────────────────────────────────────

    #[tokio::test]
    async fn cross_host_redirect_is_refused() {
        // The endpoint 30x-es to a DIFFERENT host (the SSRF vector: a public
        // endpoint redirecting you to cloud metadata / loopback). The 200 queued
        // behind it must NEVER be followed — the cross-host redirect is refused.
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).to_string();
        let (url, _rx) = canned_server(vec![
            (
                "HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/",
                "{}".into(),
            ),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let err = engine(url, None, None)
            .embed(vec!["a".into()])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("cross-host redirect"), "{err}");
    }

    #[tokio::test]
    async fn same_host_redirect_is_followed_repinned() {
        // A same-host redirect (relative Location) IS followed — and lands on
        // the same pinned host, so the 200 behind it is served.
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [7.0, 8.0]}]}).to_string();
        let (url, rx) = canned_server(vec![
            (
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: /v2/embeddings",
                "{}".into(),
            ),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let out = engine(url, None, None)
            .embed(vec!["a".into()])
            .await
            .unwrap();
        assert_eq!(out, vec![vec![7.0, 8.0]]);
        // First request hit /v1/embeddings, the redirect followed to /v2/.
        assert!(rx.recv().unwrap().starts_with("POST /v1/embeddings"));
        assert!(rx.recv().unwrap().starts_with("POST /v2/embeddings"));
    }

    // ── The llama.cpp native multimodal dialect (#501) ──────────────────────

    const PROPS_MM: &str =
        r#"{"modalities":{"vision":true,"audio":false},"media_marker":"<__media_X__>"}"#;
    const PROPS_TEXT_ONLY: &str =
        r#"{"modalities":{"vision":false,"audio":false},"media_marker":"<__media_X__>"}"#;

    #[tokio::test]
    async fn props_parses_modalities_and_marker_and_defaults_none() {
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", PROPS_MM.to_string());
        let props = engine(url, None, None).props().await.unwrap();
        assert!(props.vision && !props.audio);
        assert_eq!(props.media_marker.as_deref(), Some("<__media_X__>"));
        // No /props at all (connection refused) → None, not an error.
        assert!(engine("http://127.0.0.1:9".into(), None, None)
            .props()
            .await
            .is_none());
    }

    #[tokio::test]
    async fn image_chunk_rides_the_native_dialect() {
        // /props first, then one native /embeddings per item; the request
        // must carry the SERVER'S marker and the item's base64 bytes, and
        // the nested [[...]] vector unwraps to one f32 vector per item.
        let native = r#"[{"index":0,"embedding":[[1.0,2.0,3.0]]}]"#;
        let (url, rx) = canned_server(vec![
            ("HTTP/1.1 200 OK", PROPS_MM.to_string()),
            ("HTTP/1.1 200 OK", native.to_string()),
        ]);
        let out = engine(url, None, None)
            .embed_images(vec![MediaItem::untyped(b"pngbytes".to_vec())])
            .await
            .unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0, 3.0]]);
        let props_req = rx.recv().unwrap();
        assert!(props_req.starts_with("GET /props"), "{props_req}");
        let embed_req = rx.recv().unwrap();
        assert!(embed_req.starts_with("POST /embeddings "), "{embed_req}");
        assert!(
            embed_req.contains(r#""prompt_string":"<__media_X__>""#),
            "server marker echoed: {embed_req}"
        );
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"pngbytes");
        assert!(
            embed_req.contains(&format!(r#""multimodal_data":["{b64}"]"#)),
            "base64 payload: {embed_req}"
        );
    }

    #[tokio::test]
    async fn text_only_model_refuses_before_any_payload() {
        // /props says vision:false → the error names the mmproj fix and NO
        // embed request reaches the server (only the one canned response is
        // consumed; a second request would hang on the closed listener, so
        // the immediate error itself is the proof).
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", PROPS_TEXT_ONLY.to_string());
        let err = engine(url, None, None)
            .embed_images(vec![MediaItem::untyped(vec![1])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("mmproj"), "{err}");
    }

    #[tokio::test]
    async fn endpoint_without_props_has_no_media_path() {
        let err = engine("http://127.0.0.1:9".into(), None, None)
            .embed_images(vec![MediaItem::untyped(vec![1])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("/props"), "{err}");
    }

    #[tokio::test]
    async fn server_error_message_is_surfaced() {
        // The terminal error carries llama.cpp's own message (after the
        // bounded retries — 500 is transient by policy).
        let body = r#"{"error":{"code":500,"message":"Multimodal data provided, but model does not support multimodal requests.","type":"server_error"}}"#;
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 200 OK", PROPS_MM.to_string()),
            ("HTTP/1.1 500 Internal Server Error", body.to_string()),
            ("HTTP/1.1 500 Internal Server Error", body.to_string()),
            ("HTTP/1.1 500 Internal Server Error", body.to_string()),
        ]);
        let err = engine(url, None, None)
            .embed_images(vec![MediaItem::untyped(vec![1])])
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("does not support multimodal requests"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn multi_pooled_vector_response_is_rejected() {
        // `--pooling none` yields a per-token [[...],[...]] response; taking
        // [0] would silently index only the first token, so it's rejected
        // (symmetry with the text path's mixed-width guard).
        let native = r#"[{"index":0,"embedding":[[1.0,2.0],[3.0,4.0]]}]"#;
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 200 OK", PROPS_MM.to_string()),
            ("HTTP/1.1 200 OK", native.to_string()),
        ]);
        let err = engine(url, None, None)
            .embed_images(vec![MediaItem::untyped(vec![1])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("--pooling none"), "{err}");
    }

    #[tokio::test]
    async fn zero_width_vector_response_is_rejected() {
        let native = r#"[{"index":0,"embedding":[[]]}]"#;
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 200 OK", PROPS_MM.to_string()),
            ("HTTP/1.1 200 OK", native.to_string()),
        ]);
        let err = engine(url, None, None)
            .embed_images(vec![MediaItem::untyped(vec![1])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("zero-width"), "{err}");
    }
}
