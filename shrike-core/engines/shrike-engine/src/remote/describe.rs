//! The remote VLM describe engine (#433): [`RecognizeMedia`] over any
//! OpenAI-compatible chat-completions endpoint with vision content parts —
//! llama-server with a multimodal GGUF locally (`--mmproj`, spawned via
//! `shrike-llama-server`'s `chat_mode`), a cloud service with a key
//! (OpenAI / OpenRouter / Gemini's OpenAI-compat endpoint accept the
//! identical shape; Anthropic's native API differs — reach Claude via an
//! OpenAI-compat gateway). Route 1 of the engine contract: sync `ureq`, so
//! the `Blocking` adapter moves each request onto the runtime's blocking
//! pool. The shared SSRF/retry/api-key machinery lives in [`super::http`]
//! (#708 dedup); this module holds only the describe-specific dialect.
//!
//! Scope discipline: this crate **talks to an endpoint**, nothing else — the
//! [`super::embed`] posture. Fingerprint *assembly* is host policy; this
//! module serves the raw ingredients (`/v1/models` id + meta) plus
//! [`compose_fingerprint`], the recipe the host applies (which folds
//! [`DESCRIBE_PROMPT_VERSION`] — the prompt is output-affecting exactly like
//! `EMBED_TEXT_VERSION` — and the projector name for a local vision server,
//! which `/v1/models` meta does NOT reflect).
//!
//! **Destination rule (settled in docs/dev/decisions.md): VLM descriptions go
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
use shrike_error::{ErrorKind, NativeResult, ResultExt};

use super::http::{ModelInfo, PostOutcome, RemoteHttpClient};

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

/// Bounded retry on the describe path (the request is idempotent): cloud
/// endpoints 429/503 routinely. Mirrors [`super::embed`].
const DESCRIBE_ATTEMPTS: u32 = 3;

/// A status that condemns *this image*, not the endpoint: oversized,
/// unsupported, or malformed input. The item degrades to the empty
/// recognition; any other non-transient status is an endpoint/config
/// problem and errors the chunk.
fn item_level_status(code: u16) -> bool {
    matches!(code, 400 | 413 | 415 | 422)
}

/// Construction parameters for [`RemoteDescriber`] (VLM image→text over HTTP).
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

/// The engine: a thin, stateless HTTP client wrapper.
///
/// SSRF posture lives in [`RemoteHttpClient`] (#592, same as [`super::embed`]):
/// the agent is pinned to `base_url`'s resolved IP with auto-redirects OFF, and
/// a redirect is followed only when it stays on the SAME host.
pub struct RemoteDescriber {
    http: RemoteHttpClient,
    model: Option<String>,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    detail: Option<String>,
    timeout: Duration,
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
    /// Construction validates the API key (header-injection guard, in
    /// [`RemoteHttpClient::new`]) and pins the endpoint's IP.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is invalid or the endpoint host can't be resolved/pinned.
    pub fn new(cfg: RemoteDescriberConfig) -> NativeResult<Self> {
        let timeout = cfg.timeout.unwrap_or(DESCRIBE_TIMEOUT);
        let http = RemoteHttpClient::new(&cfg.base_url, cfg.api_key, timeout)?;
        Ok(Self {
            http,
            model: cfg.model,
            prompt: cfg.prompt.unwrap_or_else(|| DESCRIBE_PROMPT_V1.to_string()),
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            detail: cfg.detail,
            timeout,
        })
    }

    /// `GET /health` is 200 — llama-server's readiness.
    pub fn health_ok(&self) -> bool {
        self.http.health_ok()
    }

    /// The first `/v1/models` entry's id + meta (empty on any failure or
    /// shape mismatch — identity falls back to host config, never errors).
    pub fn model_info(&self) -> ModelInfo {
        self.http.model_info()
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
        // `item.mime` is the ENGINE routing-hint (`shrike_engine_api::mime_for_name`
        // via `MediaItem::from_named`) — describe-remote reuses that table by
        // construction, NOT shrike-media's store/response MIME (the two are
        // deliberately separate; see the #711 notes on both tables).
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

        // Item-level rejection (a 4xx the image caused) → empty recognition.
        let resp = match self.http.post_json_with_retry(
            "/v1/chat/completions",
            &payload,
            DESCRIBE_ATTEMPTS,
            self.timeout,
            "describe",
            item_level_status,
        )? {
            PostOutcome::Response(resp) => resp,
            PostOutcome::ItemRejected => return Ok(empty_recognition()),
        };

        let body: ChatResponse = resp
            .into_json()
            .context(ErrorKind::Internal, "malformed describe response")?;
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
    use super::super::http::test_server::{canned_server, one_shot_server};
    use super::*;

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

    // ── SSRF redirect re-vet (#592) ─────────────────────────────────────────

    #[test]
    fn cross_host_redirect_is_refused() {
        // The endpoint 30x-es to a DIFFERENT host (the SSRF vector). The 200
        // queued behind it must NEVER be followed — refused as a chunk error.
        let (url, _rx) = canned_server(vec![
            (
                "HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/",
                "{}".into(),
            ),
            ("HTTP/1.1 200 OK", chat_ok("leaked")),
        ]);
        let err = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap_err();
        assert!(err.to_string().contains("cross-host redirect"), "{err}");
    }

    #[test]
    fn same_host_redirect_is_followed_repinned() {
        // A same-host redirect IS followed and lands on the same pinned host.
        let (url, rx) = canned_server(vec![
            (
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: /v2/chat/completions",
                "{}".into(),
            ),
            ("HTTP/1.1 200 OK", chat_ok("after redirect")),
        ]);
        let out = engine(url, None, None)
            .recognize_chunk(&[item(&[1], "a.png")])
            .unwrap();
        assert_eq!(out[0].text, "after redirect");
        assert!(rx.recv().unwrap().starts_with("POST /v1/chat/completions"));
        assert!(rx.recv().unwrap().starts_with("POST /v2/chat/completions"));
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
