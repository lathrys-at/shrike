//! The generic remote-embeddings engine (#342 P4): [`EmbedText`] over any
//! OpenAI-compatible embeddings endpoint — llama-server locally, a cloud
//! embedding API with a key, a service across a tailnet. Route 1 of the
//! engine contract: ureq is synchronous (no runtime), so the `Blocking`
//! adapter moves each request onto the runtime's blocking pool and network
//! calls never block a runtime worker.
//!
//! Since #501 it also speaks llama.cpp's NATIVE multimodal dialect for
//! [`EmbedImages`]: `GET /props` advertises the loaded model's modalities
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

use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;
use shrike_engine_api::{EmbedImages, EmbedText, MediaItem};
use shrike_ffi::{NativeError, NativeResult};

/// Per-request ceiling, matching the Python backend's httpx timeout.
const EMBED_TIMEOUT: Duration = Duration::from_secs(60);
const META_TIMEOUT: Duration = Duration::from_secs(5);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Bounded retry on the embed path (the request is idempotent): cloud
/// endpoints 429/503 routinely (rate limits, cold scale-up), so a transient
/// failure must not sink the chunk. Mirrors the probe's small explicit
/// attempts loop; this is a sync engine on the runtime's blocking pool, so
/// the backoff is a plain `std::thread::sleep`.
const EMBED_ATTEMPTS: u32 = 3;
const BACKOFF_BASE: Duration = Duration::from_millis(250);
/// A server-supplied `Retry-After` is honored but never trusted past this.
const RETRY_AFTER_CAP: Duration = Duration::from_secs(10);

/// 429 and 5xx are transient (rate limit, restart, overload); any other
/// status is a request problem a retry can't fix.
fn retryable_status(code: u16) -> bool {
    code == 429 || (500..600).contains(&code)
}

/// The `Retry-After` seconds form, capped. The HTTP-date form is ignored
/// (the default backoff covers it).
fn retry_after(resp: &ureq::Response) -> Option<Duration> {
    let secs: u64 = resp.header("retry-after")?.trim().parse().ok()?;
    Some(Duration::from_secs(secs).min(RETRY_AFTER_CAP))
}

/// Exponential: 250ms, 500ms, … for attempt 1, 2, …
fn backoff(attempt: u32) -> Duration {
    BACKOFF_BASE * 2u32.saturating_pow(attempt.saturating_sub(1))
}

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
}

/// The engine: a thin, stateless HTTP client (ureq agents are cheap and
/// `Send + Sync`; each call is one request).
pub struct RemoteEmbedder {
    base_url: String,
    api_key: Option<String>,
    model: Option<String>,
    agent: ureq::Agent,
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

/// The identity ingredients from `/v1/models`: the model id and its numeric
/// meta block (`n_params`, `n_embd`, …) as raw JSON for the host to fold
/// into its fingerprint policy. Both empty when the endpoint doesn't serve
/// them (an older llama.cpp, a minimal cloud API) — the host falls back to
/// its configured identity.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelInfo {
    pub id: Option<String>,
    pub meta: serde_json::Map<String, serde_json::Value>,
}

/// llama.cpp server capabilities from `GET /props` (#501): which modalities
/// the loaded model serves (an mmproj per modality), and the per-process
/// `media_marker` a multimodal prompt references. The marker is randomized
/// at every server start, so it is read here, never assumed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlamaProps {
    pub vision: bool,
    pub audio: bool,
    pub media_marker: Option<String>,
}

impl RemoteEmbedder {
    /// Construction validates the API key — it is interpolated into the
    /// `Authorization` header, so a control character (a pasted key with a
    /// stray newline) or leading/trailing whitespace must fail loudly here
    /// as `invalid_input`, not as a garbled or injected header later.
    pub fn new(cfg: RemoteEmbedderConfig) -> NativeResult<Self> {
        if let Some(key) = &cfg.api_key {
            if key.chars().any(char::is_control) || key.trim() != key {
                return Err(NativeError::invalid_input(
                    "api_key must not contain control characters or leading/trailing whitespace",
                ));
            }
        }
        Ok(Self {
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            api_key: cfg.api_key,
            model: cfg.model,
            agent: ureq::AgentBuilder::new().build(),
        })
    }

    fn request(&self, method: &str, path: &str, timeout: Duration) -> ureq::Request {
        let mut req = self
            .agent
            .request(method, &format!("{}{}", self.base_url, path))
            .timeout(timeout);
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        req
    }

    /// `GET /health` is 200 — llama-server's readiness; other services may
    /// not serve it (treated as not-healthy, the caller decides what that
    /// means for its lifecycle).
    pub fn health_ok(&self) -> bool {
        self.request("GET", "/health", HEALTH_TIMEOUT)
            .call()
            .map(|r| r.status() == 200)
            .unwrap_or(false)
    }

    /// llama.cpp's `GET /props` capabilities, or `None` when the endpoint
    /// doesn't serve the route (a cloud API, an old build) — which means
    /// "no native multimodal dialect here", never an error.
    pub fn props(&self) -> Option<LlamaProps> {
        let resp = self.request("GET", "/props", META_TIMEOUT).call().ok()?;
        let body = resp.into_json::<serde_json::Value>().ok()?;
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

    /// The first `/v1/models` entry's id + meta (empty on any failure or
    /// shape mismatch — identity falls back to host config, never errors).
    pub fn model_info(&self) -> ModelInfo {
        let Ok(resp) = self.request("GET", "/v1/models", META_TIMEOUT).call() else {
            return ModelInfo::default();
        };
        let Ok(body) = resp.into_json::<serde_json::Value>() else {
            return ModelInfo::default();
        };
        let Some(entry) = body.get("data").and_then(|d| d.get(0)) else {
            return ModelInfo::default();
        };
        ModelInfo {
            id: entry
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            meta: entry
                .get("meta")
                .and_then(|m| m.as_object())
                .cloned()
                .unwrap_or_default(),
        }
    }
}

impl RemoteEmbedder {
    /// The bounded-retry POST both dialects share. A refused/failed request
    /// is a service-availability problem (down, mid-restart, auth rejected),
    /// not an engine bug. 429/5xx and transport failures are transient →
    /// bounded retry; any other status fails immediately. A terminal status
    /// error appends the server's own `error.message` when the body carries
    /// one (llama.cpp's "model does not support multimodal requests", a
    /// cloud API's auth detail) — the actionable half of the failure.
    fn post_json_with_retry(
        &self,
        path: &str,
        payload: &serde_json::Value,
    ) -> NativeResult<ureq::Response> {
        let mut attempt = 1u32;
        loop {
            let err = match self.request("POST", path, EMBED_TIMEOUT).send_json(payload) {
                Ok(resp) => return Ok(resp),
                Err(e) => e,
            };
            let transient = match &err {
                ureq::Error::Status(code, _) => retryable_status(*code),
                ureq::Error::Transport(_) => true,
            };
            if !transient || attempt >= EMBED_ATTEMPTS {
                let attempts = if transient {
                    format!(" after {EMBED_ATTEMPTS} attempt(s)")
                } else {
                    String::new()
                };
                let detail = match err {
                    ureq::Error::Status(code, resp) => {
                        let msg = resp
                            .into_json::<serde_json::Value>()
                            .ok()
                            .and_then(|b| {
                                b.get("error")?.get("message")?.as_str().map(String::from)
                            })
                            .unwrap_or_default();
                        if msg.is_empty() {
                            format!("status {code}")
                        } else {
                            format!("status {code}: {msg}")
                        }
                    }
                    ureq::Error::Transport(t) => t.to_string(),
                };
                return Err(NativeError::unavailable(format!(
                    "embeddings request failed{attempts}: {detail}"
                )));
            }
            let delay = match &err {
                ureq::Error::Status(_, resp) => {
                    retry_after(resp).unwrap_or_else(|| backoff(attempt))
                }
                ureq::Error::Transport(_) => backoff(attempt),
            };
            tracing::warn!(
                attempt,
                max_attempts = EMBED_ATTEMPTS,
                delay_ms = delay.as_millis() as u64,
                error = %err,
                "transient embeddings failure; retrying"
            );
            std::thread::sleep(delay);
            attempt += 1;
        }
    }
}

impl EmbedText for RemoteEmbedder {
    /// One `POST /v1/embeddings` request per chunk. Vectors are ordered by
    /// the response's own `index` rather than positional order — a cheap
    /// guard that survives a backend that doesn't preserve order (each note
    /// would otherwise silently get a batch-mate's vector).
    fn embed_chunk(&self, texts: &[String]) -> NativeResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let span = tracing::debug_span!("embed.remote_chunk", batch = texts.len());
        let _enter = span.enter();
        let mut payload = serde_json::json!({ "input": texts });
        if let Some(model) = &self.model {
            payload["model"] = serde_json::Value::String(model.clone());
        }
        let resp = self.post_json_with_retry("/v1/embeddings", &payload)?;
        let body: EmbeddingsResponse = resp
            .into_json()
            .map_err(|e| NativeError::internal(format!("malformed embeddings response: {e}")))?;
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

/// llama.cpp's NATIVE `/embeddings` response: a bare array, each item's
/// `embedding` nested one level deeper than the OpenAI shape (a list of
/// pooled vectors, one per sequence — one sequence per request here).
#[derive(Deserialize)]
struct NativeEmbeddingItem {
    embedding: Vec<Vec<f32>>,
}

impl EmbedImages for RemoteEmbedder {
    /// One native `/embeddings` request **per item** (#501): media payloads
    /// are orders of magnitude heavier than text, so per-item requests keep
    /// the retry/backoff semantics simple and attribute a failure to the
    /// exact image. Capability-gated up front — a model without the vision
    /// mmproj refuses here with the actionable error instead of paying the
    /// payload upload and the server's own 500.
    fn embed_image_chunk(&self, images: &[MediaItem]) -> NativeResult<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let span = tracing::debug_span!("embed.remote_media_chunk", batch = images.len());
        let _enter = span.enter();
        // NOTE (#501 slice B): `props()` is an extra round-trip per chunk —
        // the `Blocking` image path now chunks by the host's image `safe_batch`
        // (#211), so a full reindex re-probes ⌈images/safe_batch⌉ times (one
        // chunk per kernel image batch when safe_batch ≥ the batch size, the
        // normal case). The marker + capabilities are per-process invariants of
        // the endpoint, so this belongs read-once at engine composition (the
        // attach path that lands with the config/facade wiring) and cached,
        // leaving only a cheap assertion here. Harmless to re-read meanwhile.
        let props = self.props().ok_or_else(|| {
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
        images
            .iter()
            .map(|item| self.embed_one_media(&marker, item))
            .collect()
    }
}

impl RemoteEmbedder {
    fn embed_one_media(&self, marker: &str, item: &MediaItem) -> NativeResult<Vec<f32>> {
        let payload = serde_json::json!({
            "content": {
                "prompt_string": marker,
                "multimodal_data": [base64::engine::general_purpose::STANDARD.encode(&item.bytes)],
            }
        });
        let resp = self.post_json_with_retry("/embeddings", &payload)?;
        let mut body: Vec<NativeEmbeddingItem> = resp.into_json().map_err(|e| {
            NativeError::internal(format!("malformed native embeddings response: {e}"))
        })?;
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
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// A canned HTTP server on an ephemeral port, serving one connection per
    /// response in sequence (each closes its connection, so a retry is a
    /// fresh accept): returns (base_url, a receiver yielding each raw
    /// request head+body). A `status_line` may carry extra header lines
    /// (`"HTTP/1.1 429 …\r\nRetry-After: 1"`).
    fn canned_server(responses: Vec<(&'static str, String)>) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for (status_line, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut head = String::new();
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if let Some(v) = line
                        .to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(str::trim)
                    {
                        content_length = v.parse().unwrap_or(0);
                    }
                    let done = line == "\r\n" || line == "\n" || line.is_empty();
                    head.push_str(&line);
                    if done {
                        break;
                    }
                }
                let mut req_body = vec![0u8; content_length];
                if content_length > 0 {
                    reader.read_exact(&mut req_body).unwrap();
                }
                head.push_str(&String::from_utf8_lossy(&req_body));
                let _ = tx.send(head);
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (format!("http://{addr}"), rx)
    }

    /// The one-request special case.
    fn one_shot_server(
        status_line: &'static str,
        body: String,
    ) -> (String, mpsc::Receiver<String>) {
        canned_server(vec![(status_line, body)])
    }

    fn engine(base_url: String, model: Option<&str>, key: Option<&str>) -> RemoteEmbedder {
        RemoteEmbedder::new(RemoteEmbedderConfig {
            base_url,
            api_key: key.map(String::from),
            model: model.map(String::from),
        })
        .unwrap()
    }

    #[test]
    fn embeds_sorting_by_response_index_and_pinning_model() {
        // Out-of-order `data` must land back in input order via `index`.
        let body = serde_json::json!({"data": [
            {"index": 1, "embedding": [2.0, 2.0]},
            {"index": 0, "embedding": [1.0, 1.0]},
        ]})
        .to_string();
        let (url, rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let out = engine(url, Some("minilm"), Some("sk-test"))
            .embed_chunk(&["a".into(), "b".into()])
            .unwrap();
        assert_eq!(out, vec![vec![1.0, 1.0], vec![2.0, 2.0]]);
        let raw = rx.recv().unwrap();
        assert!(raw.starts_with("POST /v1/embeddings"), "{raw}");
        assert!(raw.contains("\"model\":\"minilm\""), "model pinned: {raw}");
        assert!(
            raw.contains("Authorization: Bearer sk-test"),
            "auth header: {raw}"
        );
    }

    #[test]
    fn arity_mismatch_is_an_internal_error() {
        let body = serde_json::json!({"data": [{"index": 0, "embedding": [1.0]}]}).to_string();
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let err = engine(url, None, None)
            .embed_chunk(&["a".into(), "b".into()])
            .unwrap_err();
        assert!(
            err.to_string().contains("1 embeddings for 2 inputs"),
            "{err}"
        );
    }

    #[test]
    fn retries_through_transient_503() {
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).to_string();
        let (url, rx) = canned_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "{}".into()),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let out = engine(url, None, None).embed_chunk(&["a".into()]).unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0]]);
        // Both attempts actually reached the server.
        assert!(rx.recv().unwrap().starts_with("POST /v1/embeddings"));
        assert!(rx.recv().unwrap().starts_with("POST /v1/embeddings"));
    }

    #[test]
    fn honors_retry_after_on_429() {
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).to_string();
        let (url, _rx) = canned_server(vec![
            (
                "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1",
                "{}".into(),
            ),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let started = std::time::Instant::now();
        let out = engine(url, None, None).embed_chunk(&["a".into()]).unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0]]);
        // The server-requested 1s overrides the 250ms default backoff — the
        // lower bound proves the header was honored, and can't flake.
        assert!(
            started.elapsed() >= Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn bad_request_does_not_retry() {
        // A 200 is queued behind the 400 — a retry would succeed, so the
        // error proves the 400 failed immediately.
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).to_string();
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 400 Bad Request", "{}".into()),
            ("HTTP/1.1 200 OK", ok),
        ]);
        let err = engine(url, None, None)
            .embed_chunk(&["a".into()])
            .unwrap_err();
        assert!(
            err.to_string().contains("embeddings request failed"),
            "{err}"
        );
        assert!(!err.to_string().contains("attempt"), "{err}");
    }

    #[test]
    fn exhausted_retries_map_to_unavailable() {
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "{}".to_string());
            3
        ]);
        let err = engine(url, None, None)
            .embed_chunk(&["a".into()])
            .unwrap_err();
        assert!(err.to_string().contains("after 3 attempt(s)"), "{err}");
    }

    #[test]
    fn mixed_width_response_is_an_internal_error() {
        let body = serde_json::json!({"data": [
            {"index": 0, "embedding": [1.0, 2.0]},
            {"index": 1, "embedding": [3.0]},
        ]})
        .to_string();
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let err = engine(url, None, None)
            .embed_chunk(&["a".into(), "b".into()])
            .unwrap_err();
        assert!(
            err.to_string().contains("mixed-width embeddings (2 vs 1)"),
            "{err}"
        );
    }

    #[test]
    fn bad_api_key_rejected_at_construction() {
        for key in ["sk\r\nX-Injected: 1", " sk-test", "sk-test ", "sk\ttest"] {
            // `.err()` rather than `.unwrap_err()`: the engine is deliberately
            // not `Debug` (the struct holds the API key).
            let err = RemoteEmbedder::new(RemoteEmbedderConfig {
                base_url: "http://127.0.0.1:9".into(),
                api_key: Some(key.into()),
                model: None,
            })
            .err()
            .expect("construction must reject the key");
            assert!(err.to_string().contains("api_key"), "{key:?}: {err}");
        }
    }

    #[test]
    fn connection_refused_maps_to_unavailable() {
        // An unbound port: connection refused, no server.
        let err = engine("http://127.0.0.1:9".into(), None, None)
            .embed_chunk(&["a".into()])
            .unwrap_err();
        assert!(
            err.to_string().contains("embeddings request failed"),
            "{err}"
        );
    }

    #[test]
    fn model_info_reads_id_and_meta_and_defaults_empty() {
        let body = serde_json::json!({"data": [{
            "id": "all-minilm",
            "meta": {"n_embd": 384, "n_params": 22713216},
        }]})
        .to_string();
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let info = engine(url, None, None).model_info();
        assert_eq!(info.id.as_deref(), Some("all-minilm"));
        assert_eq!(info.meta["n_embd"], 384);
        // And the graceful default on a down endpoint.
        assert_eq!(
            engine("http://127.0.0.1:9".into(), None, None).model_info(),
            ModelInfo::default()
        );
    }

    #[test]
    fn health_ok_only_on_200() {
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", "{}".into());
        assert!(engine(url, None, None).health_ok());
        assert!(!engine("http://127.0.0.1:9".into(), None, None).health_ok());
    }

    #[test]
    fn empty_input_short_circuits() {
        // No server at all — must not even attempt a request.
        let out = engine("http://127.0.0.1:9".into(), None, None)
            .embed_chunk(&[])
            .unwrap();
        assert!(out.is_empty());
    }
    // ── The llama.cpp native multimodal dialect (#501) ──────────────────────

    const PROPS_MM: &str =
        r#"{"modalities":{"vision":true,"audio":false},"media_marker":"<__media_X__>"}"#;
    const PROPS_TEXT_ONLY: &str =
        r#"{"modalities":{"vision":false,"audio":false},"media_marker":"<__media_X__>"}"#;

    #[test]
    fn props_parses_modalities_and_marker_and_defaults_none() {
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", PROPS_MM.to_string());
        let props = engine(url, None, None).props().unwrap();
        assert!(props.vision && !props.audio);
        assert_eq!(props.media_marker.as_deref(), Some("<__media_X__>"));
        // No /props at all (connection refused) → None, not an error.
        assert!(engine("http://127.0.0.1:9".into(), None, None)
            .props()
            .is_none());
    }

    #[test]
    fn image_chunk_rides_the_native_dialect() {
        // /props first, then one native /embeddings per item; the request
        // must carry the SERVER'S marker and the item's base64 bytes, and
        // the nested [[...]] vector unwraps to one f32 vector per item.
        let native = r#"[{"index":0,"embedding":[[1.0,2.0,3.0]]}]"#;
        let (url, rx) = canned_server(vec![
            ("HTTP/1.1 200 OK", PROPS_MM.to_string()),
            ("HTTP/1.1 200 OK", native.to_string()),
        ]);
        let out = engine(url, None, None)
            .embed_image_chunk(&[MediaItem::untyped(b"pngbytes".to_vec())])
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

    #[test]
    fn text_only_model_refuses_before_any_payload() {
        // /props says vision:false → the error names the mmproj fix and NO
        // embed request reaches the server (only the one canned response is
        // consumed; a second request would hang on the closed listener, so
        // the immediate error itself is the proof).
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", PROPS_TEXT_ONLY.to_string());
        let err = engine(url, None, None)
            .embed_image_chunk(&[MediaItem::untyped(vec![1])])
            .unwrap_err();
        assert!(err.to_string().contains("mmproj"), "{err}");
    }

    #[test]
    fn endpoint_without_props_has_no_media_path() {
        let err = engine("http://127.0.0.1:9".into(), None, None)
            .embed_image_chunk(&[MediaItem::untyped(vec![1])])
            .unwrap_err();
        assert!(err.to_string().contains("/props"), "{err}");
    }

    #[test]
    fn server_error_message_is_surfaced() {
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
            .embed_image_chunk(&[MediaItem::untyped(vec![1])])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("does not support multimodal requests"),
            "{err}"
        );
    }

    #[test]
    fn multi_pooled_vector_response_is_rejected() {
        // `--pooling none` yields a per-token [[...],[...]] response; taking
        // [0] would silently index only the first token, so it's rejected
        // (symmetry with the text path's mixed-width guard).
        let native = r#"[{"index":0,"embedding":[[1.0,2.0],[3.0,4.0]]}]"#;
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 200 OK", PROPS_MM.to_string()),
            ("HTTP/1.1 200 OK", native.to_string()),
        ]);
        let err = engine(url, None, None)
            .embed_image_chunk(&[MediaItem::untyped(vec![1])])
            .unwrap_err();
        assert!(err.to_string().contains("--pooling none"), "{err}");
    }

    #[test]
    fn zero_width_vector_response_is_rejected() {
        let native = r#"[{"index":0,"embedding":[[]]}]"#;
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 200 OK", PROPS_MM.to_string()),
            ("HTTP/1.1 200 OK", native.to_string()),
        ]);
        let err = engine(url, None, None)
            .embed_image_chunk(&[MediaItem::untyped(vec![1])])
            .unwrap_err();
        assert!(err.to_string().contains("zero-width"), "{err}");
    }
}
