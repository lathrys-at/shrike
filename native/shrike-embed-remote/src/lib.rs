//! The generic remote-embeddings engine (#342 P4): [`EmbedText`] over any
//! OpenAI-compatible embeddings endpoint — llama-server locally, a cloud
//! embedding API with a key, a service across a tailnet. Route 1 of the
//! engine contract: ureq is synchronous (no runtime), so the host adapts the
//! engine onto its lane (`OnExecutor` over the asyncio pool in the Python
//! server) and network calls never block a polling thread.
//!
//! Scope discipline: this crate **talks to an endpoint**, nothing else.
//! Launching/managing a llama-server subprocess is a different concern
//! (`shrike-llama-server`); fingerprint *assembly* (the `pool=`/`args=`/
//! `textprep=` policy suffixes) stays host-side — this crate only serves the
//! raw identity ingredients (`/v1/models` id + meta).

use std::time::Duration;

use serde::Deserialize;
use shrike_engine_api::EmbedText;
use shrike_ffi::{NativeError, NativeResult};

/// Per-request ceiling, matching the Python backend's httpx timeout.
const EMBED_TIMEOUT: Duration = Duration::from_secs(60);
const META_TIMEOUT: Duration = Duration::from_secs(5);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

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

impl RemoteEmbedder {
    pub fn new(cfg: RemoteEmbedderConfig) -> Self {
        Self {
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            api_key: cfg.api_key,
            model: cfg.model,
            agent: ureq::AgentBuilder::new().build(),
        }
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
        let resp = self
            .request("POST", "/v1/embeddings", EMBED_TIMEOUT)
            .send_json(payload)
            .map_err(|e| {
                // A refused/failed request is a service-availability problem
                // (down, mid-restart, auth rejected), not an engine bug.
                NativeError::unavailable(format!("embeddings request failed: {e}"))
            })?;
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
        Ok(items.into_iter().map(|d| d.embedding).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// A one-request canned HTTP server on an ephemeral port: returns
    /// (base_url, a receiver yielding the raw request head+body).
    fn one_shot_server(
        status_line: &'static str,
        body: String,
    ) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
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
        });
        (format!("http://{addr}"), rx)
    }

    fn engine(base_url: String, model: Option<&str>, key: Option<&str>) -> RemoteEmbedder {
        RemoteEmbedder::new(RemoteEmbedderConfig {
            base_url,
            api_key: key.map(String::from),
            model: model.map(String::from),
        })
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
    fn http_error_maps_to_unavailable() {
        let (url, _rx) = one_shot_server("HTTP/1.1 503 Service Unavailable", "{}".into());
        let err = engine(url, None, None)
            .embed_chunk(&["a".into()])
            .unwrap_err();
        assert!(
            err.to_string().contains("embeddings request failed"),
            "{err}"
        );
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
}
