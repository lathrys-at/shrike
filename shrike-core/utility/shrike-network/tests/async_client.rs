//! Live-socket integration tests for the async IP-pinned client + the
//! centralized per-hop re-vet loops.
//!
//! These are the SSRF controls the user-triggered security review scrutinizes:
//! the connection is pinned to the vetted IP while `Host`/SNI carry the NAME;
//! auto-redirects are OFF and the consumer loops re-vet every hop (the redirect
//! SSRF vector); the body is size-capped WHILE streaming. The no-network construction
//! gates (scheme/host/resolution, the is_global gate) live in the crate's unit
//! tests; these need a tokio runtime + a local server, so they live out here.
//!
//! S2 centralized the consumer loops into `fetch_pinned_get` (untrusted-media
//! GET, is_global per hop) and `post_pinned_with_revet` (operator-endpoint POST,
//! same-host per hop). These tests drive the REAL helpers (not a reconstructed
//! loop) so the one audited SSRF surface is exactly what is exercised.

use std::net::IpAddr;
use std::time::Duration;

use shrike_network::{
    fetch_pinned_get, pinned_async_client, pinned_endpoint_async_client, post_pinned_with_revet,
    RevetStep, MAX_REDIRECTS,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const T: Duration = Duration::from_secs(5);
const CAP: usize = 1024 * 1024;

/// One canned HTTP/1.1 response, plus the request head the server captured.
struct Captured {
    /// The full request head (request line + headers) the server received.
    head: String,
}

/// A minimal async canned server: binds an ephemeral 127.0.0.1 port, serves
/// `responses` in order (one connection each, `Connection: close`), and reports
/// each request's head over a channel. Returns the bound port and a receiver of
/// the captured request heads.
fn canned_server(responses: Vec<String>) -> (u16, tokio::sync::mpsc::UnboundedReceiver<Captured>) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    // Bind synchronously so the caller has the port before the first request.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let listener = TcpListener::from_std(listener).unwrap();
        for response in responses {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let head = read_request_head(&mut stream).await;
            let _ = tx.send(Captured { head });
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.flush().await;
            // Drop closes the connection (we always send Connection: close).
        }
    });
    (port, rx)
}

/// Read just the request head (up to the blank line) — enough to assert on the
/// request line + `Host`. We don't need the body for these GETs.
async fn read_request_head(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match stream.read(&mut byte).await {
            Ok(0) => break,
            Ok(_) => {
                buf.push(byte[0]);
                if buf.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&buf).to_string()
}

/// A 200 OK with a JSON body of `body`.
fn ok_response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// A 302 to `location`.
fn redirect_response(location: &str) -> String {
    format!(
        "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
}

// ── the low-level client builders ──────────────────────────────────

#[tokio::test]
async fn pin_connects_to_vetted_ip_while_host_carries_the_name() {
    // Build a client pinning the NAME `pinned.test` to the loopback server's
    // address. The request URL uses the name + the server's port, so:
    //  - the socket goes to 127.0.0.1 (the only thing listening — if the pin
    //    failed, `pinned.test` would not resolve and the request would error);
    //  - the `Host` header (and TLS SNI, for https) carries `pinned.test:PORT`.
    let (port, mut rx) = canned_server(vec![ok_response("{\"ok\":true}")]);
    let loopback: IpAddr = "127.0.0.1".parse().unwrap();
    let client = pinned_async_client(loopback, "pinned.test", T).unwrap();

    let url = format!("http://pinned.test:{port}/v1/thing");
    let resp = client
        .get(&url)
        .send()
        .await
        .expect("request reaches the pinned socket");
    assert_eq!(resp.status(), 200);

    let captured = rx.recv().await.expect("server captured the request");
    // The Host header carries the NAME — not 127.0.0.1 — proving SNI/Host ride
    // the name while the socket went to the pinned IP.
    let host_line = captured
        .head
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("host:"))
        .expect("a Host header");
    assert!(
        host_line.to_ascii_lowercase().contains("pinned.test"),
        "Host must carry the name, got: {host_line:?}"
    );
    assert!(
        !host_line.contains("127.0.0.1"),
        "Host must NOT be rewritten to the pinned IP, got: {host_line:?}"
    );
}

#[tokio::test]
async fn endpoint_client_pins_loopback_and_round_trips() {
    // The operator-endpoint builder resolves loopback (no is_global gate) and
    // pins to it; a POST round-trips.
    let (port, mut rx) = canned_server(vec![ok_response("{\"data\":[]}")]);
    let base = format!("http://127.0.0.1:{port}");
    let (client, parsed) = pinned_endpoint_async_client(&base, T).unwrap();
    assert_eq!(parsed.host_str(), Some("127.0.0.1"));

    let resp = client
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"input": "hi"}))
        .send()
        .await
        .expect("endpoint round-trip");
    assert_eq!(resp.status(), 200);
    let captured = rx.recv().await.unwrap();
    assert!(
        captured.head.starts_with("POST /v1/embeddings"),
        "{:?}",
        captured.head
    );
}

#[tokio::test]
async fn auto_redirects_are_off_so_a_3xx_surfaces_to_the_caller() {
    // The whole SSRF control depends on the client NOT auto-following: a 302
    // must surface as a 302 the caller inspects + re-vets, never a transparent
    // jump to the Location.
    let (port, _rx) = canned_server(vec![redirect_response("http://evil.test/x")]);
    let loopback: IpAddr = "127.0.0.1".parse().unwrap();
    let client = pinned_async_client(loopback, "pinned.test", T).unwrap();
    let resp = client
        .get(format!("http://pinned.test:{port}/start"))
        .send()
        .await
        .unwrap();
    // The status is the redirect itself — not 200 from a followed hop.
    assert_eq!(resp.status(), 302);
    assert_eq!(
        resp.headers().get("location").unwrap().to_str().unwrap(),
        "http://evil.test/x"
    );
}

// ── post_pinned_with_revet: the operator-endpoint POST loop ─────────

/// Drive `post_pinned_with_revet` with a `reqwest` client doing the per-URL send
/// (the consumer's job): return `Redirect(location)` on a 3xx, `Done(status)`
/// otherwise — exactly the shape `RemoteHttpClient` uses.
async fn run_endpoint_loop(base: &str) -> shrike_error::NativeResult<reqwest::StatusCode> {
    let (client, base_url) = pinned_endpoint_async_client(base, T).unwrap();
    let start = format!("{base}/v1/thing");
    post_pinned_with_revet(start, base_url, move |url| {
        let client = client.clone();
        async move {
            let resp = client.get(&url).send().await.unwrap();
            let status = resp.status();
            if status.is_redirection() {
                let location = resp
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .unwrap()
                    .to_string();
                Ok(RevetStep::Redirect(location))
            } else {
                Ok(RevetStep::Done(status))
            }
        }
    })
    .await
}

#[tokio::test]
async fn endpoint_loop_refuses_cross_host_redirect() {
    // A cross-host 30x (the SSRF vector: a public endpoint redirecting you to
    // cloud metadata / loopback) is refused by the same-host re-vet.
    let (port, _rx) = canned_server(vec![redirect_response("http://169.254.169.254/latest/")]);
    let base = format!("http://127.0.0.1:{port}");
    let err = run_endpoint_loop(&base).await.unwrap_err();
    assert!(err.message.contains("cross-host redirect"), "{err:?}");
}

#[tokio::test]
async fn endpoint_loop_follows_same_host_to_the_final_response() {
    // A same-host redirect IS followed to the final 200, still on the pinned IP.
    let (port, _rx) = canned_server(vec![
        redirect_response("/v2/thing"),
        ok_response("{\"ok\":1}"),
    ]);
    let base = format!("http://127.0.0.1:{port}");
    let status = run_endpoint_loop(&base).await.unwrap();
    assert_eq!(status, 200);
}

#[tokio::test]
async fn endpoint_loop_caps_at_max_redirects() {
    // Every hop points to a same-host path, so same_host_redirect keeps
    // permitting it — only the cap stops the loop, surfacing the cap error.
    let responses: Vec<String> = (0..(MAX_REDIRECTS + 3))
        .map(|i| redirect_response(&format!("/hop/{}", i + 1)))
        .collect();
    let (port, _rx) = canned_server(responses);
    let base = format!("http://127.0.0.1:{port}");
    let err = run_endpoint_loop(&base).await.unwrap_err();
    assert!(err.message.contains("too many redirects"), "{err:?}");
}

// ── fetch_pinned_get: the untrusted-media GET loop ──────────────────

#[tokio::test]
async fn media_get_round_trips_and_reports_content_type() {
    // The happy path: a 200 with a content-type returns (bytes, Some(ct)).
    // allow_private=true reaches the loopback server (non-global, so the gate-ON
    // path would refuse it); the gate itself is exercised below.
    let (port, _rx) = canned_server(vec![ok_response("{\"ok\":true}")]);
    let url = format!("http://127.0.0.1:{port}/file.json");
    let (body, ct) = fetch_pinned_get(&url, true, T, CAP).await.unwrap();
    assert_eq!(body, b"{\"ok\":true}");
    assert_eq!(ct.as_deref(), Some("application/json"));
}

#[tokio::test]
async fn media_get_refuses_non_global_first_hop() {
    // The untrusted path (gate ON) refuses a loopback host before any socket —
    // resolve_public_ip rejects 127.0.0.1 (non-public). This is the first-hop
    // is_global gate; the per-hop re-vet on subsequent hops reuses the identical
    // resolve_public_ip call (the crate's unit tests pin the classifier corpus).
    let err = fetch_pinned_get("http://127.0.0.1:1/x", false, T, CAP)
        .await
        .unwrap_err();
    assert!(err.message.contains("non-public address"), "{err:?}");
}

#[tokio::test]
async fn media_get_refuses_non_http_scheme_and_hostless() {
    assert!(fetch_pinned_get("ftp://example.com/x", false, T, CAP)
        .await
        .is_err());
    assert!(fetch_pinned_get("not a url", false, T, CAP).await.is_err());
}

#[tokio::test]
async fn media_get_body_is_size_capped_while_streaming() {
    // The size cap applies WHILE reading (so an oversize body is rejected before
    // it is fully buffered). A 2 MiB body, a 1 MiB cap → the helper errors.
    let big = "x".repeat(2 * 1024 * 1024);
    let (port, _rx) = canned_server(vec![ok_response(&big)]);
    let url = format!("http://127.0.0.1:{port}/big");
    let err = fetch_pinned_get(&url, true, T, CAP).await.unwrap_err();
    assert!(
        err.message.contains("exceeds the") && err.message.contains("byte limit"),
        "{err:?}"
    );
}

#[tokio::test]
async fn media_get_follows_same_host_redirect() {
    // The media loop follows a relative same-host redirect manually (still
    // capped) and reads the final body.
    let (port, _rx) = canned_server(vec![redirect_response("/final"), ok_response("{\"v\":2}")]);
    let url = format!("http://127.0.0.1:{port}/start");
    let (body, _ct) = fetch_pinned_get(&url, true, T, CAP).await.unwrap();
    assert_eq!(body, b"{\"v\":2}");
}
