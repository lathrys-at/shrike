//! The remote VLM describe engine (#433): [`RecognizeMedia`] over any
//! OpenAI-compatible chat-completions endpoint with vision content parts —
//! llama-server with a multimodal GGUF locally (`--mmproj`, spawned via
//! `shrike-llama-server`'s `chat_mode`), a cloud service with a key
//! (OpenAI / OpenRouter / Gemini's OpenAI-compat endpoint accept the
//! identical shape; Anthropic's native API differs — reach Claude via an
//! OpenAI-compat gateway). Route 1 of the engine contract: ureq is
//! synchronous, so the `Blocking` adapter moves each request onto the
//! runtime's blocking pool.
//!
//! Scope discipline: this crate **talks to an endpoint**, nothing else —
//! the `shrike-embed-remote` posture. Fingerprint *assembly* is host
//! policy; this crate serves the raw ingredients (`/v1/models` id + meta)
//! plus [`compose_fingerprint`], the recipe the host applies (which folds
//! [`DESCRIBE_PROMPT_VERSION`] — the prompt is output-affecting exactly
//! like `EMBED_TEXT_VERSION` — and the projector name for a local vision
//! server, which `/v1/models` meta does NOT reflect).
//!
//! **Destination rule (settled in docs/decisions.md): VLM descriptions go
//! to the embedding space only, never the trigram index.** Today's
//! recognition pipeline ingests recognized text into the derived (lexical)
//! store unconditionally, so this engine is constructible and tested but
//! must NOT occupy a recognition slot until the kernel grows a per-engine
//! destination policy (#433 tracks that integration).
//!
//! Error semantics — the load-bearing split, proven against the kernel's
//! sweep: a **per-item permanent** failure (400/413/415/422 — *this image*
//! is bad) yields the empty recognition so one item never sinks a batch
//! (the kernel re-offers gate-dropped items, same as OCR's gated-out
//! cost); an **endpoint-level** failure (transport, auth, exhausted
//! transient retries) errors the whole chunk, because `recognize_pending`
//! aborts on a chunk error *before* persisting anything or advancing the
//! recognizer-fingerprint meta — everything stays pending and the next
//! sweep retries. Returning N empty recognitions for a down endpoint would
//! instead drain the backlog into stored nothing that never re-derives.

use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;
use shrike_engine_api::{MediaItem, Recognition, RecognizeMedia};
use shrike_ffi::{NativeError, NativeResult};

/// Bump whenever [`DESCRIBE_PROMPT_V1`] (or whichever template ships)
/// changes: the prompt is part of the output space, so a change must
/// re-derive — the describe analogue of `EMBED_TEXT_VERSION`.
pub const DESCRIBE_PROMPT_VERSION: u32 = 1;

/// The retrieval-oriented describe prompt: subject + image type + named
/// entities (the embedding signal), then a verbatim transcription of
/// visible text, with chat boilerplate suppressed (preamble pollutes the
/// embedding input). Plain prose by design — the destination is an
/// embedding, and small VLMs are unreliable JSON emitters.
pub const DESCRIBE_PROMPT_V1: &str = "Describe this image for search indexing. In 1-3 \
     sentences, state what the image shows: the main subject and any objects, the type of image \
     (photo, diagram, chart, screenshot, handwriting), and any notable entities by name. Then, \
     if the image contains any visible text, transcribe it verbatim. Output only the description \
     and transcription — no preamble, no markdown, no commentary.";

/// Per-request ceiling: describe is far slower than embed (a local VLM can
/// take tens of seconds per image on CPU).
const DESCRIBE_TIMEOUT: Duration = Duration::from_secs(120);
const META_TIMEOUT: Duration = Duration::from_secs(5);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Bounded retry on the describe path (the request is idempotent): cloud
/// endpoints 429/503 routinely. Mirrors `shrike-embed-remote` verbatim.
const DESCRIBE_ATTEMPTS: u32 = 3;
const BACKOFF_BASE: Duration = Duration::from_millis(250);
/// A server-supplied `Retry-After` is honored but never trusted past this.
const RETRY_AFTER_CAP: Duration = Duration::from_secs(10);

/// 429 and 5xx are transient (rate limit, restart, overload).
fn retryable_status(code: u16) -> bool {
    code == 429 || (500..600).contains(&code)
}

/// A status that condemns *this image*, not the endpoint: oversized,
/// unsupported, or malformed input. The item degrades to the empty
/// recognition; any other non-transient status is an endpoint/config
/// problem and errors the chunk.
fn item_level_status(code: u16) -> bool {
    matches!(code, 400 | 413 | 415 | 422)
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

pub struct RemoteDescriberConfig {
    /// e.g. `http://127.0.0.1:8090` (no trailing slash needed).
    pub base_url: String,
    /// Optional bearer token — the API-key seam for cloud services.
    pub api_key: Option<String>,
    /// Pin the model in each request body so a multi-model endpoint
    /// resolves the right one (a single-model llama-server ignores it).
    pub model: Option<String>,
    /// The describe template; defaults to [`DESCRIBE_PROMPT_V1`]. An
    /// override is host policy and must come with its own fingerprint
    /// suffix discipline.
    pub prompt: Option<String>,
    /// Response budget — long enough for description + transcription,
    /// short enough to bound latency/cost.
    pub max_tokens: u32,
    /// 0.0 for a reproducible describe pass.
    pub temperature: f32,
    /// The OpenAI `detail` knob (`"auto"`/`"low"`/`"high"`); `None` omits
    /// it (llama-server ignores it either way; `"low"` is a cloud cost
    /// lever).
    pub detail: Option<String>,
    /// Per-request ceiling override; `None` = [`DESCRIBE_TIMEOUT`].
    pub timeout: Option<Duration>,
}

impl Default for RemoteDescriberConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: None,
            model: None,
            prompt: None,
            max_tokens: 384,
            temperature: 0.0,
            detail: None,
            timeout: None,
        }
    }
}

/// The engine: a thin, stateless HTTP client (ureq agents are cheap and
/// `Send + Sync`; each call is one request).
pub struct RemoteDescriber {
    base_url: String,
    api_key: Option<String>,
    model: Option<String>,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    detail: Option<String>,
    timeout: Duration,
    agent: ureq::Agent,
}

/// The identity ingredients from `/v1/models` (same shape as
/// `shrike-embed-remote`): id + numeric meta, both empty when the endpoint
/// doesn't serve them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelInfo {
    pub id: Option<String>,
    pub meta: serde_json::Map<String, serde_json::Value>,
}

/// The host-side fingerprint recipe:
/// `describe:<id>[:meta=<n_params>/<n_embd>/<n_vocab>/<n_ctx_train>/<size>][:mmproj=<name>]:prompt=<N>`.
///
/// - `id` falls back to the configured model, then `"unknown"` (a minimal
///   cloud API may serve neither `/v1/models` nor meta).
/// - The meta segment folds llama-server's numeric block so a same-name
///   re-quantization invalidates; omitted when absent.
/// - `mmproj` is the projector the host launched the local server with —
///   folded because `/v1/models` meta describes the language model GGUF
///   only, so a projector swap would otherwise never re-derive. Absent for
///   cloud.
/// - `prompt=<N>` is unconditional: the template is output-affecting.
pub fn compose_fingerprint(
    info: &ModelInfo,
    configured_model: Option<&str>,
    mmproj: Option<&str>,
) -> String {
    let id = info.id.as_deref().or(configured_model).unwrap_or("unknown");
    let mut fp = format!("describe:{id}");
    let fields = ["n_params", "n_embd", "n_vocab", "n_ctx_train", "size"];
    if fields.iter().any(|f| info.meta.contains_key(*f)) {
        let joined = fields
            .iter()
            .map(|f| info.meta.get(*f).map_or("?".to_string(), |v| v.to_string()))
            .collect::<Vec<_>>()
            .join("/");
        fp.push_str(&format!(":meta={joined}"));
    }
    if let Some(mmproj) = mmproj {
        fp.push_str(&format!(":mmproj={mmproj}"));
    }
    fp.push_str(&format!(":prompt={DESCRIBE_PROMPT_VERSION}"));
    fp
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

impl RemoteDescriber {
    /// Construction validates the API key — it is interpolated into the
    /// `Authorization` header, so a control character or stray whitespace
    /// must fail loudly here as `invalid_input`, not as a garbled or
    /// injected header later. (Verbatim `shrike-embed-remote` discipline.)
    pub fn new(cfg: RemoteDescriberConfig) -> NativeResult<Self> {
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
            prompt: cfg.prompt.unwrap_or_else(|| DESCRIBE_PROMPT_V1.to_string()),
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            detail: cfg.detail,
            timeout: cfg.timeout.unwrap_or(DESCRIBE_TIMEOUT),
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
    /// not serve it (treated as not-healthy, the caller decides).
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

    /// One image through one chat-completions request. `Ok(empty)` for an
    /// item-level failure (empty bytes, 4xx the image caused, an empty
    /// caption); `Err` only for an endpoint-level failure.
    fn describe_one(&self, item: &MediaItem) -> NativeResult<Recognition> {
        if item.bytes.is_empty() {
            return Ok(empty_recognition());
        }
        let span = tracing::debug_span!("describe.remote_one", bytes = item.bytes.len());
        let _enter = span.enter();

        // The mime hint rides the data URL; `image/png` is the documented
        // pragmatic default (the dominant Anki case; llama-server sniffs
        // content regardless, and some cloud gateways reject non-image/*).
        let mime = item.mime.as_deref().unwrap_or("image/png");
        let data_url = format!(
            "data:{mime};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&item.bytes)
        );
        let mut image_url = serde_json::json!({ "url": data_url });
        if let Some(detail) = &self.detail {
            image_url["detail"] = serde_json::Value::String(detail.clone());
        }
        let mut payload = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": self.prompt },
                    { "type": "image_url", "image_url": image_url },
                ],
            }],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        });
        if let Some(model) = &self.model {
            payload["model"] = serde_json::Value::String(model.clone());
        }

        let mut attempt = 1u32;
        let resp = loop {
            let err = match self
                .request("POST", "/v1/chat/completions", self.timeout)
                .send_json(&payload)
            {
                Ok(resp) => break resp,
                Err(e) => e,
            };
            // Item-level: this image is unprocessable (oversized, rejected,
            // malformed) — degrade to the empty recognition, never sink the
            // batch, never retry (a bad image won't improve).
            if let ureq::Error::Status(code, _) = &err {
                if item_level_status(*code) {
                    tracing::warn!(status = code, "endpoint rejected an image; item skipped");
                    return Ok(empty_recognition());
                }
            }
            // Endpoint-level: transient (429/5xx/transport) retries bounded;
            // anything else (auth, bad route) errors the chunk immediately.
            let transient = match &err {
                ureq::Error::Status(code, _) => retryable_status(*code),
                ureq::Error::Transport(_) => true,
            };
            if !transient {
                return Err(NativeError::unavailable(format!(
                    "describe request failed: {err}"
                )));
            }
            if attempt >= DESCRIBE_ATTEMPTS {
                return Err(NativeError::unavailable(format!(
                    "describe request failed after {DESCRIBE_ATTEMPTS} attempt(s): {err}"
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
                max_attempts = DESCRIBE_ATTEMPTS,
                delay_ms = delay.as_millis() as u64,
                error = %err,
                "transient describe failure; retrying"
            );
            std::thread::sleep(delay);
            attempt += 1;
        };

        let body: ChatResponse = resp
            .into_json()
            .map_err(|e| NativeError::internal(format!("malformed describe response: {e}")))?;
        let text = body
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .trim()
            .to_string();
        if text.is_empty() {
            return Ok(empty_recognition());
        }
        Ok(Recognition {
            text,
            // The API reports no confidence; 1.0 on substance lets the
            // kernel's gate decide on text substance alone (the right
            // signal for a generated caption).
            confidence: 1.0,
            // A description locates nothing — no boxes, no spans.
            segments: Vec::new(),
        })
    }
}

fn empty_recognition() -> Recognition {
    Recognition {
        text: String::new(),
        confidence: 0.0,
        segments: Vec::new(),
    }
}

impl RecognizeMedia for RemoteDescriber {
    fn recognize_chunk(&self, items: &[MediaItem]) -> NativeResult<Vec<Recognition>> {
        items.iter().map(|m| self.describe_one(m)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// A canned HTTP server on an ephemeral port, one connection per queued
    /// response (ported from shrike-embed-remote's tests).
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

    fn one_shot_server(
        status_line: &'static str,
        body: String,
    ) -> (String, mpsc::Receiver<String>) {
        canned_server(vec![(status_line, body)])
    }

    fn chat_ok(text: &str) -> String {
        serde_json::json!({"choices": [{"message": {"role": "assistant", "content": text}}]})
            .to_string()
    }

    fn engine(base_url: String, model: Option<&str>, key: Option<&str>) -> RemoteDescriber {
        RemoteDescriber::new(RemoteDescriberConfig {
            base_url,
            api_key: key.map(String::from),
            model: model.map(String::from),
            ..Default::default()
        })
        .unwrap()
    }

    fn item(bytes: &[u8], name: &str) -> MediaItem {
        MediaItem::from_named(name, bytes.to_vec())
    }

    #[test]
    fn describes_pinning_request_shape() {
        let (url, rx) = one_shot_server("HTTP/1.1 200 OK", chat_ok("  A red square.  "));
        let out = engine(url, Some("smolvlm"), Some("sk-test"))
            .recognize_chunk(&[item(&[1, 2, 3], "a.png")])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "A red square.");
        assert_eq!(out[0].confidence, 1.0);
        assert!(out[0].segments.is_empty());
        let raw = rx.recv().unwrap();
        assert!(raw.starts_with("POST /v1/chat/completions"), "{raw}");
        // The exact data-URL prefix: mime from the filename, base64 payload.
        assert!(raw.contains("data:image/png;base64,AQID"), "{raw}");
        assert!(raw.contains("\"model\":\"smolvlm\""), "{raw}");
        assert!(raw.contains("\"temperature\":0"), "{raw}");
        assert!(raw.contains("\"max_tokens\":384"), "{raw}");
        assert!(raw.contains("Authorization: Bearer sk-test"), "{raw}");
        assert!(
            raw.contains("search indexing"),
            "the prompt rode along: {raw}"
        );
    }

    #[test]
    fn unnamed_item_defaults_to_png_mime() {
        let (url, rx) = one_shot_server("HTTP/1.1 200 OK", chat_ok("x"));
        let items = [MediaItem::untyped(vec![9])];
        engine(url, None, None).recognize_chunk(&items).unwrap();
        assert!(
            rx.recv().unwrap().contains("data:image/png;base64,"),
            "default mime"
        );
    }

    #[test]
    fn empty_caption_yields_empty_recognition() {
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", chat_ok("   "));
        let out = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap();
        assert_eq!((out[0].text.as_str(), out[0].confidence), ("", 0.0));
    }

    #[test]
    fn empty_bytes_short_circuit_without_a_request() {
        // No server at all — must not even attempt a request.
        let out = engine("http://127.0.0.1:9".into(), None, None)
            .recognize_chunk(&[MediaItem::untyped(Vec::new())])
            .unwrap();
        assert_eq!(out[0].text, "");
    }

    #[test]
    fn item_level_4xx_degrades_the_item_not_the_chunk() {
        // 400 → empty recognition; the queued 200 must NOT be consumed (no
        // retry of a condemned image).
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 413 Payload Too Large", "{}".into()),
            ("HTTP/1.1 200 OK", chat_ok("never sent")),
        ]);
        let out = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "big.png")])
            .unwrap();
        assert_eq!((out[0].text.as_str(), out[0].confidence), ("", 0.0));
    }

    #[test]
    fn endpoint_level_failure_errors_the_chunk() {
        // Connection refused: the chunk must Err so the kernel's sweep
        // leaves every item pending (N empty recognitions would burn the
        // backlog into stored nothing).
        let err = engine("http://127.0.0.1:9".into(), None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap_err();
        assert!(err.to_string().contains("describe request failed"), "{err}");
    }

    #[test]
    fn auth_failure_errors_the_chunk_not_the_item() {
        let (url, _rx) = one_shot_server("HTTP/1.1 401 Unauthorized", "{}".into());
        let err = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap_err();
        assert!(err.to_string().contains("describe request failed"), "{err}");
        assert!(
            !err.to_string().contains("attempt"),
            "no retry on 401: {err}"
        );
    }

    #[test]
    fn retries_through_transient_503() {
        let (url, rx) = canned_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "{}".into()),
            ("HTTP/1.1 200 OK", chat_ok("after retry")),
        ]);
        let out = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap();
        assert_eq!(out[0].text, "after retry");
        assert!(rx.recv().unwrap().starts_with("POST /v1/chat/completions"));
        assert!(rx.recv().unwrap().starts_with("POST /v1/chat/completions"));
    }

    #[test]
    fn honors_retry_after_on_429() {
        let (url, _rx) = canned_server(vec![
            (
                "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1",
                "{}".into(),
            ),
            ("HTTP/1.1 200 OK", chat_ok("ok")),
        ]);
        let started = std::time::Instant::now();
        let out = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap();
        assert_eq!(out[0].text, "ok");
        assert!(
            started.elapsed() >= Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn exhausted_retries_map_to_unavailable() {
        let (url, _rx) = canned_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "{}".to_string());
            3
        ]);
        let err = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap_err();
        assert!(err.to_string().contains("after 3 attempt(s)"), "{err}");
    }

    #[test]
    fn bad_api_key_rejected_at_construction() {
        for key in ["sk\r\nX-Injected: 1", " sk-test", "sk-test ", "sk\ttest"] {
            let err = RemoteDescriber::new(RemoteDescriberConfig {
                base_url: "http://127.0.0.1:9".into(),
                api_key: Some(key.into()),
                ..Default::default()
            })
            .err()
            .expect("construction must reject the key");
            assert!(err.to_string().contains("api_key"), "{key:?}: {err}");
        }
    }

    #[test]
    fn model_info_reads_id_and_meta_and_defaults_empty() {
        let body = serde_json::json!({"data": [{
            "id": "smolvlm-500m",
            "meta": {"n_params": 507000000, "n_embd": 960},
        }]})
        .to_string();
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", body);
        let info = engine(url, None, None).model_info();
        assert_eq!(info.id.as_deref(), Some("smolvlm-500m"));
        assert_eq!(info.meta["n_embd"], 960);
        assert_eq!(
            engine("http://127.0.0.1:9".into(), None, None).model_info(),
            ModelInfo::default()
        );
    }

    #[test]
    fn fingerprint_recipe_composition() {
        let mut meta = serde_json::Map::new();
        meta.insert("n_params".into(), 507_000_000u64.into());
        meta.insert("n_embd".into(), 960u64.into());
        meta.insert("size".into(), 437_000_000u64.into());
        let info = ModelInfo {
            id: Some("smolvlm-500m".into()),
            meta,
        };
        assert_eq!(
            compose_fingerprint(&info, None, Some("mmproj-q8.gguf")),
            "describe:smolvlm-500m:meta=507000000/960/?/?/437000000:mmproj=mmproj-q8.gguf:prompt=1"
        );
        // Minimal cloud: no /v1/models id, no meta → configured model.
        assert_eq!(
            compose_fingerprint(&ModelInfo::default(), Some("gpt-5-mini"), None),
            "describe:gpt-5-mini:prompt=1"
        );
        // Nothing at all → unknown (still prompt-versioned).
        assert_eq!(
            compose_fingerprint(&ModelInfo::default(), None, None),
            "describe:unknown:prompt=1"
        );
    }

    #[test]
    fn health_ok_only_on_200() {
        let (url, _rx) = one_shot_server("HTTP/1.1 200 OK", "{}".into());
        assert!(engine(url, None, None).health_ok());
        assert!(!engine("http://127.0.0.1:9".into(), None, None).health_ok());
    }

    /// A 64×64 solid-red PNG, generated once and inlined — the live tier's
    /// self-contained fixture (no cross-crate include_bytes!).
    #[rustfmt::skip]
    const RED_SQUARE_PNG: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 64, 0, 0, 0, 64,
        8, 2, 0, 0, 0, 37, 11, 230, 137, 0, 0, 0, 75, 73, 68, 65, 84, 120, 218, 237, 207, 65, 9,
        0, 0, 8, 0, 177, 235, 95, 90, 35, 248, 22, 6, 43, 176, 166, 94, 75, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 224, 178,
        169, 220, 240, 226, 96, 232, 185, 247, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];

    /// Live tier: point `SHRIKE_DESCRIBE_URL` at a running OpenAI-compatible
    /// vision endpoint (e.g. `llama-server -hf
    /// ggml-org/SmolVLM-500M-Instruct-GGUF -c 8192`); optional
    /// `SHRIKE_DESCRIBE_MODEL` / `SHRIKE_DESCRIBE_API_KEY`. Skipped when
    /// unset — default `cargo test` stays hermetic.
    #[test]
    fn live_endpoint_describes_the_fixture() {
        let Ok(url) = std::env::var("SHRIKE_DESCRIBE_URL") else {
            return;
        };
        let engine = RemoteDescriber::new(RemoteDescriberConfig {
            base_url: url,
            api_key: std::env::var("SHRIKE_DESCRIBE_API_KEY").ok(),
            model: std::env::var("SHRIKE_DESCRIBE_MODEL").ok(),
            ..Default::default()
        })
        .unwrap();
        let out = engine
            .recognize_chunk(&[item(RED_SQUARE_PNG, "red.png")])
            .unwrap();
        assert!(
            !out[0].text.is_empty(),
            "the live endpoint produced a description"
        );
        assert_eq!(out[0].confidence, 1.0);
        // And the identity ingredients are servable.
        let info = engine.model_info();
        let fp = compose_fingerprint(&info, None, None);
        assert!(fp.starts_with("describe:"), "{fp}");
        assert!(fp.ends_with(":prompt=1"), "{fp}");
    }
}
