//! The shared, SSRF-pinned HTTP client both remote engines (`embed`, `describe`)
//! ride (#708 dedup): ONE copy of the trust-boundary code the two engines used
//! to carry verbatim — the API-key/header-injection validator, the IP-pin, the
//! same-host redirect re-vet (#592), and the bounded transient retry
//! (backoff + `Retry-After`).
//!
//! **Sync `ureq`** — runtime-less, on the kernel runtime's blocking pool via the
//! `Blocking` adapter. The transport (the pinned [`ureq::Agent`] + the per-URL
//! POST in [`RemoteHttpClient::post_one_url_with_retry`]) is the ONE swap-point
//! for the async-first port (#721: reqwest/hyper over `shrike-network`'s async
//! IP-pinned connector). #721 also must convert the engines' `.enter()` spans to
//! `.instrument(span)` — holding a span guard across an `.await` leaks it.
//!
//! SSRF posture (#592): `base_url` is operator-configured and trusted (a loopback
//! llama-server, a tailnet host, a cloud API the operator chose), so it is NOT
//! `is_global`-gated. But the agent is built with auto-redirects OFF and **pinned
//! to `base_url`'s resolved IP** (closing the DNS-rebinding TOCTOU), and a
//! redirect is followed only when it stays on the SAME host (a remote endpoint
//! that 30x-es you to a different host is the SSRF vector —
//! `shrike_network::same_host_redirect` refuses it). NOTE: the IP is pinned once
//! at construction; an endpoint whose address rotates mid-life (a cloud LB
//! draining a node) would surface as a transient failure → the bounded retry,
//! not silently wrong — reconstruct the client to re-pin.

use std::time::Duration;

use shrike_error::{NativeError, NativeResult};

/// Metadata round-trip ceiling (`/v1/models`, `/props`, `/health` reads).
pub(crate) const META_TIMEOUT: Duration = Duration::from_secs(5);
pub(crate) const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Bounded retry on an idempotent request: cloud endpoints 429/503 routinely
/// (rate limits, cold scale-up), so a transient failure must not sink the
/// request. This is a sync engine on the runtime's blocking pool, so the
/// backoff is a plain `std::thread::sleep`.
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

/// One bounded-retry POST's terminal outcome, parameterised so both engines
/// share the loop:
///
/// - `Response` — a non-3xx response (2xx, or a 4xx/5xx the caller maps).
/// - `ItemRejected` — a status the `item_level` predicate condemned as a
///   per-item problem (describe's 400/413/415/422). The caller degrades the
///   single item; embed's predicate never returns true, so it never sees this.
pub(crate) enum PostOutcome {
    Response(Box<ureq::Response>),
    ItemRejected,
}

/// The shared SSRF-pinned HTTP client: `base_url` + the parsed base host (for
/// the same-host redirect comparison) + the optional bearer key + the pinned
/// `ureq` agent. Cheap and `Send + Sync`; each call is one request. The struct
/// is deliberately NOT `Debug` — it holds the API key.
pub(crate) struct RemoteHttpClient {
    base_url: String,
    /// `base_url` parsed once, for the same-host redirect comparison.
    base: url::Url,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl RemoteHttpClient {
    /// Construct, validating the API key and pinning the connection to
    /// `base_url`'s resolved IP with auto-redirects OFF.
    ///
    /// The API key is interpolated into the `Authorization` header, so a
    /// control character (a pasted key with a stray newline) or
    /// leading/trailing whitespace must fail loudly here as `invalid_input`,
    /// not as a garbled or injected header later. `agent_timeout` is the
    /// agent-level backstop; each request sets its own via `.timeout()`.
    pub(crate) fn new(
        base_url: &str,
        api_key: Option<String>,
        agent_timeout: Duration,
    ) -> NativeResult<Self> {
        if let Some(key) = &api_key {
            if key.chars().any(char::is_control) || key.trim() != key {
                return Err(NativeError::invalid_input(
                    "api_key must not contain control characters or leading/trailing whitespace",
                ));
            }
        }
        let base_url = base_url.trim_end_matches('/').to_string();
        // SSRF posture (#592): pin the connection to base_url's resolved IP with
        // auto-redirects OFF. The agent-level timeout is a backstop; each request
        // sets its own via `.timeout()`.
        let (agent, base) = shrike_network::pinned_endpoint_agent(&base_url, agent_timeout)?;
        Ok(Self {
            base_url,
            base,
            api_key,
            agent,
        })
    }

    fn request(&self, method: &str, path: &str, timeout: Duration) -> ureq::Request {
        self.request_abs(method, &format!("{}{}", self.base_url, path), timeout)
    }

    /// Build a request to an ABSOLUTE url (the redirect loop sends to the
    /// validated same-host target, not a base-relative path).
    fn request_abs(&self, method: &str, url: &str, timeout: Duration) -> ureq::Request {
        let mut req = self.agent.request(method, url).timeout(timeout);
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        req
    }

    /// `GET /health` is 200 — llama-server's readiness; other services may
    /// not serve it (treated as not-healthy, the caller decides what that
    /// means for its lifecycle).
    pub(crate) fn health_ok(&self) -> bool {
        self.request("GET", "/health", HEALTH_TIMEOUT)
            .call()
            .map(|r| r.status() == 200)
            .unwrap_or(false)
    }

    /// A bare `GET <path>` for engine-specific metadata reads (e.g. llama.cpp's
    /// `/props`); `None` on any failure. The caller parses the JSON body.
    pub(crate) fn get_json(&self, path: &str) -> Option<serde_json::Value> {
        let resp = self.request("GET", path, META_TIMEOUT).call().ok()?;
        resp.into_json::<serde_json::Value>().ok()
    }

    /// The first `/v1/models` entry's id + meta (empty on any failure or
    /// shape mismatch — identity falls back to host config, never errors).
    pub(crate) fn model_info(&self) -> ModelInfo {
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

    /// The bounded-retry POST both engines share. Auto-redirects are OFF, so a
    /// 3xx surfaces as a response we follow ONLY if it stays on the SAME host
    /// (the pinned base host); a cross-host redirect is refused (the SSRF
    /// vector). Capped at `shrike_network::MAX_REDIRECTS`. Each URL gets its own
    /// transient-retry.
    ///
    /// `op` labels the error string ("embeddings"/"describe"); `timeout` is the
    /// per-request ceiling; `item_level` classifies a status as a per-item
    /// rejection (`PostOutcome::ItemRejected`) rather than an endpoint failure —
    /// embed passes `|_| false` (it has no item-level concept), describe passes
    /// its 400/413/415/422 predicate.
    pub(crate) fn post_json_with_retry(
        &self,
        path: &str,
        payload: &serde_json::Value,
        attempts: u32,
        timeout: Duration,
        op: &str,
        item_level: fn(u16) -> bool,
    ) -> NativeResult<PostOutcome> {
        let mut current = format!("{}{}", self.base_url, path);
        let mut from = self.base.clone();
        for _hop in 0..=shrike_network::MAX_REDIRECTS {
            let resp = match self
                .post_one_url_with_retry(&current, payload, attempts, timeout, op, item_level)?
            {
                PostOutcome::Response(resp) => resp,
                PostOutcome::ItemRejected => return Ok(PostOutcome::ItemRejected),
            };
            if (300..400).contains(&resp.status()) {
                let location = resp.header("location").ok_or_else(|| {
                    NativeError::unavailable("redirect response without a Location header")
                })?;
                let target = shrike_network::same_host_redirect(&from, location)?;
                current = target.to_string();
                from = target;
                continue;
            }
            return Ok(PostOutcome::Response(resp));
        }
        Err(NativeError::unavailable(format!(
            "too many redirects (>{})",
            shrike_network::MAX_REDIRECTS
        )))
    }

    /// One absolute URL's bounded-retry POST (the per-URL half the redirect
    /// loop drives). A 3xx is returned as `Response` for the caller to
    /// follow/refuse. **This is the transport seam #721 ports to async.**
    fn post_one_url_with_retry(
        &self,
        url: &str,
        payload: &serde_json::Value,
        attempts: u32,
        timeout: Duration,
        op: &str,
        item_level: fn(u16) -> bool,
    ) -> NativeResult<PostOutcome> {
        let mut attempt = 1u32;
        loop {
            let err = match self.request_abs("POST", url, timeout).send_json(payload) {
                Ok(resp) => return Ok(PostOutcome::Response(Box::new(resp))),
                // ureq surfaces a 3xx either as Ok (redirects disabled) or as
                // Error::Status depending on version — return it for the
                // redirect loop to follow/refuse, identically to the Ok case.
                Err(ureq::Error::Status(code, resp)) if (300..400).contains(&code) => {
                    return Ok(PostOutcome::Response(Box::new(resp)));
                }
                Err(e) => e,
            };
            // Item-level: this item is unprocessable (oversized, rejected,
            // malformed) — the caller degrades to the empty result, never sinks
            // the batch, never retries (a bad item won't improve). embed's
            // predicate is `|_| false`, so this branch is describe-only.
            if let ureq::Error::Status(code, _) = &err {
                if item_level(*code) {
                    tracing::warn!(status = code, "endpoint rejected an item; item skipped");
                    return Ok(PostOutcome::ItemRejected);
                }
            }
            let transient = match &err {
                ureq::Error::Status(code, _) => retryable_status(*code),
                ureq::Error::Transport(_) => true,
            };
            if !transient || attempt >= attempts {
                let suffix = if transient {
                    format!(" after {attempts} attempt(s)")
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
                    "{op} request failed{suffix}: {detail}"
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
                max_attempts = attempts,
                delay_ms = delay.as_millis() as u64,
                error = %err,
                "transient {op} failure; retrying"
            );
            std::thread::sleep(delay);
            attempt += 1;
        }
    }
}

/// A canned HTTP server on an ephemeral port, serving one connection per
/// response in sequence (each closes its connection, so a retry is a fresh
/// accept): returns (base_url, a receiver yielding each raw request head+body).
/// A `status_line` may carry extra header lines
/// (`"HTTP/1.1 429 …\r\nRetry-After: 1"`). Shared by both engines' test
/// modules — the SSRF/redirect/api-key/retry vectors run against ONE harness.
#[cfg(test)]
pub(crate) mod test_server {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    pub(crate) fn canned_server(
        responses: Vec<(&'static str, String)>,
    ) -> (String, mpsc::Receiver<String>) {
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
    pub(crate) fn one_shot_server(
        status_line: &'static str,
        body: String,
    ) -> (String, mpsc::Receiver<String>) {
        canned_server(vec![(status_line, body)])
    }
}
