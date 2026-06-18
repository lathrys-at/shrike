//! The shared, SSRF-pinned HTTP client both remote engines (`embed`, `describe`)
//! ride (#708 dedup): ONE copy of the trust-boundary code the two engines used
//! to carry verbatim — the API-key/header-injection validator, the IP-pin, the
//! same-host redirect re-vet (#592), and the bounded transient retry
//! (backoff + `Retry-After`).
//!
//! **Async `reqwest`** (#721 S2 — route 2): the engines implement engine-api's
//! async traits directly and the kernel awaits them on its runtime, so the
//! transport here is async and the backoff is `tokio::time::sleep` (no parked
//! thread). The SSRF pinning + the per-hop same-host redirect loop live in
//! [`shrike_network`] (the one audited home: [`shrike_network::pinned_endpoint_async_client`]
//! pins the connection, [`shrike_network::post_pinned_with_revet`] follows
//! redirects re-vetting each hop as same-host); the engine policy below owns the
//! send + retry + status handling.
//!
//! SSRF posture (#592): `base_url` is operator-configured and trusted (a loopback
//! llama-server, a tailnet host, a cloud API the operator chose), so it is NOT
//! `is_global`-gated. But the client is built with auto-redirects OFF and **pinned
//! to `base_url`'s resolved IP** (closing the DNS-rebinding TOCTOU), and a
//! redirect is followed only when it stays on the SAME host (a remote endpoint
//! that 30x-es you to a different host is the SSRF vector —
//! `shrike_network::same_host_redirect` refuses it). NOTE: the IP is pinned once
//! at construction; an endpoint whose address rotates mid-life (a cloud LB
//! draining a node) would surface as a transient failure → the bounded retry,
//! not silently wrong — reconstruct the client to re-pin.

use std::time::Duration;

use serde::de::DeserializeOwned;
use shrike_error::{NativeError, NativeResult};
use shrike_network::{post_pinned_with_revet, RevetStep};

/// Metadata round-trip ceiling (`/v1/models`, `/props`, `/health` reads).
pub(crate) const META_TIMEOUT: Duration = Duration::from_secs(5);
pub(crate) const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Bounded retry on an idempotent request: cloud endpoints 429/503 routinely
/// (rate limits, cold scale-up), so a transient failure must not sink the
/// request. The backoff rides `tokio::time::sleep` (the engines are async now —
/// route 2 — so a sleep never parks a thread).
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
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let secs: u64 = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
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
    /// The endpoint's reported model id, if any.
    pub id: Option<String>,
    /// The raw `meta` block (n_params, n_embd, …) for fingerprint policy.
    pub meta: serde_json::Map<String, serde_json::Value>,
}

/// One bounded-retry POST's terminal outcome, parameterised so both engines
/// share the loop:
///
/// - `Response` — a non-3xx response (2xx, or a 4xx/5xx the caller maps),
///   carrying the response's status + body bytes (already read off the wire).
/// - `ItemRejected` — a status the `item_level` predicate condemned as a
///   per-item problem (describe's 400/413/415/422). The caller degrades the
///   single item; embed's predicate never returns true, so it never sees this.
pub(crate) enum PostOutcome {
    Response(HttpResponse),
    ItemRejected,
}

/// A terminal (non-3xx) response after the redirect loop: the full body bytes
/// (reqwest's body is consumed async, so it is buffered here and the engines
/// deserialize from it synchronously). A `PostOutcome::Response` is always a
/// success body by the time it reaches the engine — the retry loop has already
/// mapped 4xx/5xx to an error or item-rejection — so only the body is carried
/// (the status is consumed inside the loop). The payloads are small JSON.
pub(crate) struct HttpResponse {
    body: Vec<u8>,
}

impl HttpResponse {
    /// Deserialize the JSON body into `T` (consuming the buffered body).
    pub(crate) fn into_json<T: DeserializeOwned>(self) -> Result<T, serde_json::Error> {
        serde_json::from_slice(&self.body)
    }
}

/// The shared SSRF-pinned HTTP client: `base_url` + the parsed base host (for
/// the same-host redirect comparison) + the optional bearer key + the pinned
/// async `reqwest` client. Cheap and `Send + Sync`; each call is one request.
/// The struct is deliberately NOT `Debug` — it holds the API key.
pub(crate) struct RemoteHttpClient {
    base_url: String,
    /// `base_url` parsed once, for the same-host redirect comparison.
    base: url::Url,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl RemoteHttpClient {
    /// Construct, validating the API key and pinning the connection to
    /// `base_url`'s resolved IP with auto-redirects OFF.
    ///
    /// The API key is interpolated into the `Authorization` header, so a
    /// control character (a pasted key with a stray newline) or
    /// leading/trailing whitespace must fail loudly here as `invalid_input`,
    /// not as a garbled or injected header later. `client_timeout` is the
    /// client-level backstop; each request also sets its own `.timeout()`.
    ///
    /// # Errors
    ///
    /// Returns an `InvalidInput` [`NativeError`] if the API key contains control
    /// characters or surrounding whitespace, or if `base_url` is not a valid
    /// http(s) URL whose host resolves; an `Unavailable` [`NativeError`] if the
    /// pinned client cannot be built.
    pub(crate) fn new(
        base_url: &str,
        api_key: Option<String>,
        client_timeout: Duration,
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
        // auto-redirects OFF. The client-level timeout is a backstop; each request
        // sets its own via `.timeout()`.
        let (client, base) =
            shrike_network::pinned_endpoint_async_client(&base_url, client_timeout)?;
        Ok(Self {
            base_url,
            base,
            api_key,
            client,
        })
    }

    /// Apply the bearer header (if any) and per-request timeout to a builder.
    fn authed(
        &self,
        builder: reqwest::RequestBuilder,
        timeout: Duration,
    ) -> reqwest::RequestBuilder {
        let builder = builder.timeout(timeout);
        if let Some(key) = &self.api_key {
            builder.header(reqwest::header::AUTHORIZATION, format!("Bearer {key}"))
        } else {
            builder
        }
    }

    /// `GET /health` is 200 — llama-server's readiness; other services may
    /// not serve it (treated as not-healthy, the caller decides what that
    /// means for its lifecycle).
    pub(crate) async fn health_ok(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        match self
            .authed(self.client.get(&url), HEALTH_TIMEOUT)
            .send()
            .await
        {
            Ok(resp) => resp.status().as_u16() == 200,
            Err(_) => false,
        }
    }

    /// A bare `GET <path>` for engine-specific metadata reads (e.g. llama.cpp's
    /// `/props`); `None` on any failure. The caller parses the JSON body.
    pub(crate) async fn get_json(&self, path: &str) -> Option<serde_json::Value> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .authed(self.client.get(&url), META_TIMEOUT)
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json::<serde_json::Value>().await.ok()
    }

    /// The first `/v1/models` entry's id + meta (empty on any failure or
    /// shape mismatch — identity falls back to host config, never errors).
    pub(crate) async fn model_info(&self) -> ModelInfo {
        let Some(body) = self.get_json("/v1/models").await else {
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
    /// vector). The per-hop same-host re-vet + cap live in
    /// [`shrike_network::post_pinned_with_revet`]; each URL gets its own
    /// transient-retry here.
    ///
    /// `op` labels the error string ("embeddings"/"describe"); `timeout` is the
    /// per-request ceiling; `item_level` classifies a status as a per-item
    /// rejection (`PostOutcome::ItemRejected`) rather than an endpoint failure —
    /// embed passes `|_| false` (it has no item-level concept), describe passes
    /// its 400/413/415/422 predicate.
    ///
    /// # Errors
    ///
    /// Returns an `Unavailable` [`NativeError`] on a transport failure, an
    /// exhausted transient retry, a refused cross-host redirect, or the redirect
    /// cap; the body is mapped by the caller.
    pub(crate) async fn post_json_with_retry(
        &self,
        path: &str,
        payload: &serde_json::Value,
        attempts: u32,
        timeout: Duration,
        op: &str,
        item_level: fn(u16) -> bool,
    ) -> NativeResult<PostOutcome> {
        let start = format!("{}{}", self.base_url, path);
        post_pinned_with_revet(start, self.base.clone(), |url| async move {
            // Each absolute URL gets the bounded transient-retry; a 3xx comes
            // back as RevetStep::Redirect for the shared loop to re-vet + follow,
            // a terminal outcome (2xx/mapped 4xx-5xx, or item-rejected) as Done.
            match self
                .post_one_url_with_retry(&url, payload, attempts, timeout, op, item_level)
                .await?
            {
                PostOneOutcome::Redirect(location) => Ok(RevetStep::Redirect(location)),
                PostOneOutcome::Response(resp) => Ok(RevetStep::Done(PostOutcome::Response(resp))),
                PostOneOutcome::ItemRejected => Ok(RevetStep::Done(PostOutcome::ItemRejected)),
            }
        })
        .await
    }

    /// One absolute URL's bounded-retry POST (the per-URL half the shared
    /// redirect loop drives). A 3xx is returned as `Redirect(location)` for the
    /// loop to re-vet/follow.
    ///
    /// # Errors
    ///
    /// Returns an `Unavailable` [`NativeError`] on a transport failure, an
    /// exhausted transient retry, or a 3xx without a `Location` header.
    async fn post_one_url_with_retry(
        &self,
        url: &str,
        payload: &serde_json::Value,
        attempts: u32,
        timeout: Duration,
        op: &str,
        item_level: fn(u16) -> bool,
    ) -> NativeResult<PostOneOutcome> {
        let mut attempt = 1u32;
        loop {
            let send = self
                .authed(self.client.post(url), timeout)
                .json(payload)
                .send()
                .await;

            // A transport error (connection refused, TLS, timeout) is transient
            // by policy; a response (any status) is inspected for its code.
            let resp = match send {
                Ok(resp) => resp,
                Err(e) => {
                    if attempt >= attempts {
                        return Err(NativeError::unavailable(format!(
                            "{op} request failed after {attempts} attempt(s): {e}"
                        )));
                    }
                    let delay = backoff(attempt);
                    tracing::warn!(
                        attempt,
                        max_attempts = attempts,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "transient {op} failure; retrying"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
            };
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();

            // 3xx: hand the Location to the redirect loop (auto-redirects OFF).
            if (300..400).contains(&status) {
                let location = headers
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        NativeError::unavailable("redirect response without a Location header")
                    })?
                    .to_string();
                return Ok(PostOneOutcome::Redirect(location));
            }

            // 2xx: read the body and return it for the caller to map.
            if status < 300 {
                let body = resp.bytes().await.map(|b| b.to_vec()).map_err(|e| {
                    NativeError::unavailable(format!("{op} response read failed: {e}"))
                })?;
                return Ok(PostOneOutcome::Response(HttpResponse { body }));
            }

            // Item-level: this item is unprocessable (oversized, rejected,
            // malformed) — the caller degrades to the empty result, never sinks
            // the batch, never retries (a bad item won't improve). embed's
            // predicate is `|_| false`, so this branch is describe-only.
            if item_level(status) {
                tracing::warn!(status, "endpoint rejected an item; item skipped");
                return Ok(PostOneOutcome::ItemRejected);
            }

            // A 4xx/5xx: transient (429/5xx) retries; anything else fails now.
            let transient = retryable_status(status);
            if !transient || attempt >= attempts {
                let detail = error_detail(resp, status).await;
                let suffix = if transient {
                    format!(" after {attempts} attempt(s)")
                } else {
                    String::new()
                };
                return Err(NativeError::unavailable(format!(
                    "{op} request failed{suffix}: {detail}"
                )));
            }
            let delay = retry_after(&headers).unwrap_or_else(|| backoff(attempt));
            tracing::warn!(
                attempt,
                max_attempts = attempts,
                delay_ms = delay.as_millis() as u64,
                status,
                "transient {op} failure; retrying"
            );
            tokio::time::sleep(delay).await;
            attempt += 1;
        }
    }
}

/// The per-URL POST's outcome before the redirect loop maps it onto
/// [`PostOutcome`]: a redirect to re-vet, a terminal response, or item-rejected.
enum PostOneOutcome {
    Redirect(String),
    Response(HttpResponse),
    ItemRejected,
}

/// The terminal error detail for a non-transient/exhausted status: the
/// endpoint's own `error.message` when it serves one, else `status N`.
async fn error_detail(resp: reqwest::Response, status: u16) -> String {
    let body = resp.bytes().await.map(|b| b.to_vec()).unwrap_or_default();
    let msg = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|b| b.get("error")?.get("message")?.as_str().map(String::from))
        .unwrap_or_default();
    if msg.is_empty() {
        format!("status {status}")
    } else {
        format!("status {status}: {msg}")
    }
}

/// A canned HTTP server on an ephemeral port, serving one connection per
/// response in sequence (each closes its connection, so a retry is a fresh
/// accept): returns (base_url, a receiver yielding each raw request head+body).
/// A `status_line` may carry extra header lines
/// (`"HTTP/1.1 429 …\r\nRetry-After: 1"`). Shared by both engines' test
/// modules — the SSRF/redirect/api-key/retry vectors run against ONE harness.
/// A plain blocking std-thread server: the engines' async tests drive it from a
/// tokio runtime, which is fine (the server thread blocks on `accept`, the test
/// awaits the client request).
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
