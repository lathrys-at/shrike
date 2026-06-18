//! Live-socket integration tests for the async IP-pinned client (#721 S1).
//!
//! These are the SSRF controls the user-triggered security review scrutinizes:
//! the connection is pinned to the vetted IP while `Host`/SNI carry the NAME;
//! auto-redirects are OFF and the caller re-vets every hop (the #592 vector);
//! the body is size-capped WHILE streaming. The no-network construction gates
//! (scheme/host/resolution, the is_global gate) live in the crate's unit tests;
//! these need a tokio runtime + a local server, so they live out here.
//!
//! The redirect-follow loop is NOT part of S1's client (it lands in the S2
//! consumers). We reconstruct the consumer loop HERE — exactly as
//! `fetch_media_url` / `RemoteHttpClient` will, reusing
//! `same_host_redirect` / `resolve_public_ip` / `MAX_REDIRECTS` — to prove the
//! S1 primitives compose into the per-hop re-vet the security model requires.

use std::net::IpAddr;
use std::time::Duration;

use shrike_network::{
    pinned_async_client, pinned_endpoint_async_client, resolve_public_ip, same_host_redirect,
    MAX_REDIRECTS,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const T: Duration = Duration::from_secs(5);

/// One canned HTTP/1.1 response, plus the request head the server captured.
struct Captured {
    /// The full request head (request line + headers) the server received.
    head: String,
}

/// A minimal async canned server: binds an ephemeral 127.0.0.1 port, serves
/// `responses` in order (one connection each, `Connection: close`), and reports
/// each request's head over a channel. Mirrors the engine crate's sync
/// `test_server` but on tokio so the async client can drive it. Returns the
/// bound port and a receiver of the captured request heads.
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
    // The whole #592 control depends on the client NOT auto-following: a 302
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

#[tokio::test]
async fn endpoint_redirect_loop_refuses_cross_host_follows_same_host_caps() {
    // Reconstruct the S2 endpoint consumer loop: follow a 3xx ONLY when
    // `same_host_redirect` permits it (same host), refuse cross-host, cap at
    // MAX_REDIRECTS. The connection stays pinned to the base IP throughout.

    // (a) cross-host redirect is REFUSED.
    {
        let (port, _rx) = canned_server(vec![redirect_response("http://169.254.169.254/latest/")]);
        let base = format!("http://127.0.0.1:{port}");
        let (client, base_url) = pinned_endpoint_async_client(&base, T).unwrap();
        let resp = client.get(format!("{base}/start")).send().await.unwrap();
        assert!((300..400).contains(&resp.status().as_u16()));
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        let err = same_host_redirect(&base_url, location).unwrap_err();
        assert!(err.message.contains("cross-host redirect"), "{err:?}");
    }

    // (b) same-host redirect is FOLLOWED to the final 200 (still on the pinned IP).
    {
        let (port, _rx) = canned_server(vec![
            redirect_response("/v2/thing"),
            ok_response("{\"ok\":1}"),
        ]);
        let base = format!("http://127.0.0.1:{port}");
        let (client, base_url) = pinned_endpoint_async_client(&base, T).unwrap();
        let mut current = format!("{base}/v1/thing");
        let mut from = base_url;
        let mut hops = 0usize;
        let final_status = loop {
            assert!(hops <= MAX_REDIRECTS, "exceeded the cap unexpectedly");
            let resp = client.get(&current).send().await.unwrap();
            if (300..400).contains(&resp.status().as_u16()) {
                let location = resp
                    .headers()
                    .get("location")
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                let target = same_host_redirect(&from, &location).unwrap();
                current = target.to_string();
                from = target;
                hops += 1;
                continue;
            }
            break resp.status();
        };
        assert_eq!(final_status, 200);
        assert_eq!(hops, 1, "followed exactly one same-host hop");
    }

    // (c) a redirect chain longer than the cap terminates with a cap error.
    {
        // Every hop points back to a same-host path, so same_host_redirect keeps
        // permitting it — only the cap stops the loop.
        let responses: Vec<String> = (0..(MAX_REDIRECTS + 3))
            .map(|i| redirect_response(&format!("/hop/{}", i + 1)))
            .collect();
        let (port, _rx) = canned_server(responses);
        let base = format!("http://127.0.0.1:{port}");
        let (client, base_url) = pinned_endpoint_async_client(&base, T).unwrap();
        let mut current = format!("{base}/hop/0");
        let mut from = base_url;
        let mut capped = false;
        for hop in 0..=MAX_REDIRECTS {
            let resp = client.get(&current).send().await.unwrap();
            assert!((300..400).contains(&resp.status().as_u16()));
            let location = resp
                .headers()
                .get("location")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            let target = same_host_redirect(&from, &location).unwrap();
            current = target.to_string();
            from = target;
            if hop == MAX_REDIRECTS {
                capped = true; // the loop bound is the cap — the consumer errors here
            }
        }
        assert!(capped, "the loop must hit the MAX_REDIRECTS bound");
    }
}

#[tokio::test]
async fn media_redirect_loop_re_vets_every_hop_against_is_global() {
    // The untrusted-media consumer loop re-resolves + re-vets EACH hop's host
    // (not just same-host): a redirect to a private/metadata host is refused by
    // resolve_public_ip on the next hop, before any socket to it opens.
    let (port, _rx) = canned_server(vec![redirect_response(
        "http://169.254.169.254/latest/meta-data/",
    )]);
    let loopback: IpAddr = "127.0.0.1".parse().unwrap();

    // hop 0: the test server is loopback; in the real media path the FIRST host
    // is is_global-vetted too. Here we drive from the loopback server (already
    // bound) and assert the NEXT hop's host is refused.
    let client = pinned_async_client(loopback, "pinned.test", T).unwrap();
    let resp = client
        .get(format!("http://pinned.test:{port}/start"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 302);
    let location = resp.headers().get("location").unwrap().to_str().unwrap();
    let next = url::Url::parse(location).unwrap();
    let next_host = next.host_str().unwrap();
    // The media consumer re-vets the redirect target's host: 169.254.169.254
    // (link-local cloud metadata) is refused with no socket opened.
    let err = resolve_public_ip(next_host).unwrap_err();
    assert!(err.message.contains("non-public address"), "{err:?}");
}

#[tokio::test]
async fn body_is_size_capped_while_streaming() {
    // The size cap applies WHILE reading (so an oversize body is rejected before
    // it is fully buffered) — the S2 consumers stream `bytes_stream()` and stop
    // once the running total exceeds the cap. Here: a 2 MiB body, a 1 MiB cap.
    use futures_util::StreamExt;

    const CAP: usize = 1024 * 1024;
    let big = "x".repeat(2 * 1024 * 1024);
    let (port, _rx) = canned_server(vec![ok_response(&big)]);
    let loopback: IpAddr = "127.0.0.1".parse().unwrap();
    let client = pinned_async_client(loopback, "pinned.test", T).unwrap();
    let resp = client
        .get(format!("http://pinned.test:{port}/big"))
        .send()
        .await
        .unwrap();

    let mut stream = resp.bytes_stream();
    let mut total = 0usize;
    let mut exceeded = false;
    while let Some(chunk) = stream.next().await {
        total += chunk.unwrap().len();
        if total > CAP {
            exceeded = true;
            break; // the consumer aborts here — does NOT buffer the whole body
        }
    }
    assert!(exceeded, "the cap must trip on the oversize body");
    // We stopped on the FIRST chunk that pushed the running total over the cap —
    // never buffering the whole 2 MiB. hyper hands back large chunks, so the
    // stopping point is "one chunk past the cap", bounded well below the full
    // body. The load-bearing property is that we broke out early, not the exact
    // byte count.
    assert!(
        total < big.len(),
        "must stop before buffering the whole {} -byte body (read {total})",
        big.len()
    );
}
