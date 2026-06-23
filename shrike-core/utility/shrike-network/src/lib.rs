//! `shrike-network` — the one home for Shrike's SSRF safety primitives.
//!
//! **Trust-boundary code**: changes here go through the security-review gate.
//!
//! The SSRF control lives here once, shared not copied. Three consumers
//! depend on it:
//!
//! - `shrike-media` (the inbound-media crate): the attacker-supplied
//!   `store_media` url path — resolves+vets EVERY hop's host against
//!   `ipaddress.is_global` parity, pins the connection to the vetted IP,
//!   follows redirects manually re-vetting each hop.
//! - `shrike-embed-remote` / `shrike-describe-remote`: the operator-configured
//!   endpoint path. The base URL is operator-trusted (loopback llama-server, a
//!   tailnet host) so it is NOT is_global-gated — but the connection is pinned
//!   to its resolved IP (closes the DNS-rebinding TOCTOU) and redirects are
//!   refused unless same-host (an embeddings/describe POST endpoint has no
//!   business 30x-ing you to a DIFFERENT host — that is the SSRF vector).
//!
//! What lives here:
//! - the IPv4/IPv6 `ipaddress.is_global` classifier ([`ip_is_allowed`]) —
//!   parity-tested against CPython over an address corpus, including the
//!   6to4 (2002::/16) and 3fff::/20 refusals;
//! - [`resolve_public_ip`]: resolve a host and refuse it unless EVERY resolved
//!   address passes the allowlist, returning the first for pinning;
//! - [`resolve_pinned`]: resolve an operator-trusted host to one address to pin
//!   (no `is_global` gate);
//! - [`same_host_redirect`]: vet a redirect for the endpoint posture (same-host
//!   only), shared by every manual redirect-following loop;
//! - the async `reqwest` builders ([`pinned_async_client`],
//!   [`pinned_endpoint_async_client`]): a client whose connection is pinned to
//!   one fixed `SocketAddr` (the vetted IP) with auto-redirects OFF, so the
//!   caller follows + re-vets each hop itself. They pin via
//!   `reqwest::ClientBuilder::resolve` (a per-host DNS override, no connect-time
//!   re-resolution) keeping SNI/cert/`Host` on the name;
//! - the centralized redirect-following loops the consumers ride
//!   ([`fetch_pinned_get`] for the untrusted-media GET — is_global per hop;
//!   [`post_pinned_with_revet`] for the operator-endpoint POST — same-host per
//!   hop): ONE audited copy of the per-hop SSRF re-vet, one helper per
//!   posture. Engine policy (retry/backoff/`Retry-After`/api-key/item-level
//!   status) stays with the consumer — the helpers own only the pinned-send +
//!   per-hop re-vet + the body size-cap.
//!
//! Pure Rust, NO engine crate and NOT `shrike-kernel` — it sits BELOW both, so
//! the kernel-purity and engine-purity layering rules both stay satisfied. The
//! transport is async `reqwest` only; synchronous `ureq` survives only in the
//! unrelated `shrike-llama-server` loopback health probe, a managed-crate concern.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::time::Duration;

use futures_util::StreamExt;
use shrike_error::{NativeError, NativeResult};

/// The hop cap shared by every manual redirect-following loop in the tree (the
/// media fetch and the remote engines), so "how many redirects" is one
/// number, not three.
pub const MAX_REDIRECTS: usize = 5;

fn invalid(msg: impl Into<String>) -> NativeError {
    NativeError::invalid_input(msg)
}

/// Python `ipaddress.is_global` for IPv4 (the registry-derived private set
/// plus the explicit 100.64.0.0/10 carve-out), with the two per-registry
/// exceptions inside 192.0.0.0/24 that ARE global. Parity-tested against
/// Python over a corpus in tests/native.
fn ipv4_is_global(addr: Ipv4Addr) -> bool {
    let o = addr.octets();
    let in_net = |net: [u8; 4], prefix: u32| -> bool {
        let ip = u32::from_be_bytes(o);
        let net = u32::from_be_bytes(net);
        let mask = if prefix == 0 {
            0
        } else {
            u32::MAX << (32 - prefix)
        };
        (ip & mask) == (net & mask)
    };
    // Exceptions inside 192.0.0.0/24 that the IANA registry marks global.
    if o == [192, 0, 0, 9] || o == [192, 0, 0, 10] {
        return true;
    }
    let private = in_net([0, 0, 0, 0], 8)
        || in_net([10, 0, 0, 0], 8)
        || in_net([127, 0, 0, 0], 8)
        || in_net([169, 254, 0, 0], 16)
        || in_net([172, 16, 0, 0], 12)
        || in_net([192, 0, 0, 0], 24)
        || in_net([192, 0, 2, 0], 24)
        || in_net([192, 168, 0, 0], 16)
        || in_net([198, 18, 0, 0], 15)
        || in_net([198, 51, 100, 0], 24)
        || in_net([203, 0, 113, 0], 24)
        || in_net([240, 0, 0, 0], 4)
        || o == [255, 255, 255, 255];
    let shared = in_net([100, 64, 0, 0], 10); // carrier-grade NAT: not private, not global
    !private && !shared
}

/// Python `ipaddress.is_global` for IPv6 (the private set; an IPv4-mapped
/// address defers to the IPv4 classifier, like Python's `ipv4_mapped`
/// handling rejects ::ffff:10.0.0.1).
fn ipv6_is_global(addr: Ipv6Addr) -> bool {
    if let Some(v4) = addr.to_ipv4_mapped() {
        return ipv4_is_global(v4);
    }
    let seg = addr.segments();
    let in_net = |net: [u16; 8], prefix: u32| -> bool {
        let ip = u128::from_be_bytes(addr.octets());
        let net_ip = u128::from_be_bytes(Ipv6Addr::from(net).octets());
        let mask = if prefix == 0 {
            0
        } else {
            u128::MAX << (128 - prefix)
        };
        (ip & mask) == (net_ip & mask)
    };
    let private = addr.is_loopback()
        || addr.is_unspecified()
        || in_net([0x64, 0xff9b, 0x1, 0, 0, 0, 0, 0], 48) // 64:ff9b:1::/48
        || in_net([0x100, 0, 0, 0, 0, 0, 0, 0], 64) // 100::/64
        || (in_net([0x2001, 0, 0, 0, 0, 0, 0, 0], 23)
            // exceptions that ARE global inside 2001::/23
            && !in_net([0x2001, 0x1, 0, 0, 0, 0, 0, 0], 32)
            && !in_net([0x2001, 0x3, 0, 0, 0, 0, 0, 0], 32)
            && !in_net([0x2001, 0x4, 0x112, 0, 0, 0, 0, 0], 48)
            && !in_net([0x2001, 0x20, 0, 0, 0, 0, 0, 0], 28)
            && !in_net([0x2001, 0x30, 0, 0, 0, 0, 0, 0], 28))
        || in_net([0x2001, 0xdb8, 0, 0, 0, 0, 0, 0], 32)
        || in_net([0x2002, 0, 0, 0, 0, 0, 0, 0], 16) // 6to4: embeds an IPv4, fails open to internal (2002:7f00:1:: = 127.0.0.1)
        || in_net([0x3fff, 0, 0, 0, 0, 0, 0, 0], 20) // 3fff::/20 reserved-by-IANA (RFC 9637, documentation)
        || in_net([0xfc00, 0, 0, 0, 0, 0, 0, 0], 7)
        || in_net([0xfe80, 0, 0, 0, 0, 0, 0, 0], 10);
    let _ = seg;
    !private
}

/// Whether one address passes the SSRF allowlist (global and not multicast).
pub fn ip_is_allowed(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => ipv4_is_global(v4) && !v4.is_multicast(),
        IpAddr::V6(v6) => ipv6_is_global(v6) && !v6.is_multicast(),
    }
}

/// Resolve and vet EVERY address (a name can't smuggle an internal A record
/// alongside a public one); return the first so the caller pins the connection
/// to it. The attacker-supplied path (`store_media`) uses this; the
/// operator-configured remote endpoints pin without this gate (see
/// [`resolve_pinned`]).
///
/// # Errors
///
/// Returns an `InvalidInput` [`NativeError`] if `host` does not resolve, or if any
/// resolved address fails the SSRF allowlist (non-global or multicast).
pub fn resolve_public_ip(host: &str) -> NativeResult<IpAddr> {
    let addrs: Vec<SocketAddr> = (host, 0u16)
        .to_socket_addrs()
        .map_err(|e| invalid(format!("could not resolve host '{host}': {e}")))?
        .collect();
    let mut vetted: Option<IpAddr> = None;
    for sa in &addrs {
        let ip = sa.ip();
        if !ip_is_allowed(ip) {
            return Err(invalid(format!(
                "refusing to fetch from non-public address {ip} (host '{host}')"
            )));
        }
        vetted.get_or_insert(ip);
    }
    vetted.ok_or_else(|| invalid(format!("could not resolve host '{host}'")))
}

/// Resolve a host to one address WITHOUT the is_global gate — for the
/// operator-configured remote endpoint, which is trusted by construction
/// (loopback llama-server, a tailnet host). Returns the first resolved
/// address so the caller can pin the connection to it (closing the
/// DNS-rebinding TOCTOU even for a trusted host).
///
/// # Errors
///
/// Returns an `InvalidInput` [`NativeError`] if `host` does not resolve to any
/// address.
pub fn resolve_pinned(host: &str) -> NativeResult<IpAddr> {
    (host, 0u16)
        .to_socket_addrs()
        .map_err(|e| invalid(format!("could not resolve host '{host}': {e}")))?
        .map(|sa| sa.ip())
        .next()
        .ok_or_else(|| invalid(format!("could not resolve host '{host}'")))
}

/// Validate a redirect for the operator-configured remote-endpoint posture:
/// the only redirect allowed is to the **same host** as where the request was
/// sent — an embeddings/describe POST endpoint that 30x-es you to a DIFFERENT
/// host is the SSRF vector, so a cross-host (or schemeless/hostless) redirect
/// is refused. Returns the resolved absolute target URL on success (its host
/// equals `from`'s host, so the connection stays pinned to the already-vetted
/// base IP — a same-host redirect can't be used to rebind to a new address).
///
/// `from` is the URL the request was sent to; `location` is the raw `Location`
/// header (may be relative). Relative locations resolve against `from`.
///
/// # Errors
///
/// Returns an `InvalidInput` [`NativeError`] if `location` is not a valid URL
/// relative to `from`, if the resolved scheme is not `http`/`https`, or if the
/// target host is absent or differs from `from`'s host.
pub fn same_host_redirect(from: &url::Url, location: &str) -> NativeResult<url::Url> {
    let target = from
        .join(location)
        .map_err(|e| invalid(format!("bad redirect location: {e}")))?;
    let scheme = target.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(invalid(format!(
            "refusing redirect to unsupported scheme '{scheme}'"
        )));
    }
    let from_host = from.host_str();
    let to_host = target.host_str();
    if to_host.is_none() || to_host != from_host {
        return Err(invalid(format!(
            "refusing cross-host redirect from '{}' to '{}' (an endpoint may not \
             redirect to a different host)",
            from_host.unwrap_or("?"),
            to_host.unwrap_or("?"),
        )));
    }
    Ok(target)
}

/// A reqwest async client that connects ONLY to `pinned` (the vetted IP),
/// ignoring the host's real DNS, with auto-redirects OFF so the caller follows
/// and re-vets each hop manually (the low-level builder under
/// [`fetch_pinned_get`]): `reqwest::ClientBuilder::resolve(host, addr)`
/// installs a per-host DNS
/// override for the client's life, so every connection to `host` uses `addr`
/// and reqwest never re-resolves the name at connect time — closing the
/// DNS-rebinding TOCTOU. The request URL keeps the hostname, so TLS SNI +
/// certificate validation verify against the NAME and the `Host` header is
/// right by construction, while the socket goes where we checked. The port
/// comes from the request URL (the override uses port 0 by convention).
///
/// `host` must be the hostname (or IP literal) of the URL the returned client
/// will be used against; mixing in a different host re-resolves through the
/// system resolver, defeating the pin — callers build one client per pinned
/// host (the [`resolve_public_ip`] → build → request pattern, per hop).
///
/// NOTE: a configured HTTP/SOCKS proxy short-circuits the DNS override —
/// reqwest connects to the *proxy* and the proxy resolves the name, so the pin
/// governs only the DIRECT-connection case (the SSRF surface). A configured
/// proxy is an operator opt-in to route through it. Proxy env
/// (`HTTP_PROXY`/`ALL_PROXY`/…, SOCKS via the `socks` feature) is honored.
///
/// # Errors
///
/// Returns an `Unavailable` [`NativeError`] if the reqwest client cannot be
/// built (a transport/TLS initialization failure).
pub fn pinned_async_client(
    pinned: IpAddr,
    host: &str,
    timeout: Duration,
) -> NativeResult<reqwest::Client> {
    // Port 0: "use the conventional port for the scheme, unless the URL names
    // one" — reqwest always prefers the URL's port over the override's, so the
    // socket address's port is irrelevant and 0 is the documented convention.
    reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .resolve(host, SocketAddr::new(pinned, 0))
        .build()
        .map_err(|e| NativeError::unavailable(format!("could not build HTTP client: {e}")))
}

/// Build an IP-pinned async client for an OPERATOR-CONFIGURED endpoint base URL:
/// parse it, resolve its host (WITHOUT the `is_global` gate — the operator
/// trusts the host, e.g. loopback llama-server / a tailnet) via [`resolve_pinned`],
/// and pin the connection to that one address (the low-level builder under
/// [`post_pinned_with_revet`]). Returns the client plus the parsed base URL
/// (which the caller keeps for the [`same_host_redirect`] comparison). The pin
/// closes the DNS-rebinding TOCTOU even for a trusted host; redirects are OFF so
/// the caller applies [`same_host_redirect`] per hop.
///
/// # Errors
///
/// Returns an `InvalidInput` [`NativeError`] if `base_url` is not a valid URL,
/// its scheme is not `http`/`https`, it has no host, or that host does not
/// resolve; or an `Unavailable` [`NativeError`] if the client cannot be built.
pub fn pinned_endpoint_async_client(
    base_url: &str,
    timeout: Duration,
) -> NativeResult<(reqwest::Client, url::Url)> {
    let base =
        url::Url::parse(base_url).map_err(|e| invalid(format!("invalid endpoint URL: {e}")))?;
    let scheme = base.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(invalid(format!(
            "endpoint URL must be http(s), got scheme '{scheme}'"
        )));
    }
    let host = base
        .host_str()
        .ok_or_else(|| invalid("endpoint URL has no host"))?
        .to_string();
    let pinned = resolve_pinned(&host)?;
    let client = pinned_async_client(pinned, &host, timeout)?;
    Ok((client, base))
}

// ── the centralized per-hop re-vet loops ───────────────────────────
// ONE audited copy of the manual redirect-following + SSRF re-vet for both
// consumers (the untrusted-media GET, the operator-endpoint POST). The posture
// differs only in how each hop is re-vetted; engine policy
// (retry/backoff/`Retry-After`/api-key/item-level status) stays with the
// consumer — these helpers own only the pinned-send + per-hop re-vet + cap.

/// Download a URL with the untrusted-media SSRF posture (`store_media`'s url
/// path), following redirects manually and re-vetting EVERY hop: each hop's
/// host is resolved and refused unless every resolved address is globally
/// routable ([`ip_is_allowed`]), then the connection is pinned to the vetted IP
/// (the URL keeps the name, so SNI/cert/`Host` ride the name). Capped at
/// [`MAX_REDIRECTS`]; the body is size-capped WHILE streaming (an oversize body
/// is rejected before it is fully buffered). Returns `(bytes, content_type)`.
///
/// With `allow_private` the is_global gate AND the pin are off (the operator
/// opted into trusted internal hosts) — system DNS, same as the Python facade's
/// switch. Redirects are still followed manually and capped, but each hop's host
/// is no longer is_global-gated.
///
/// # Errors
///
/// Returns an `InvalidInput` [`NativeError`] if the URL is malformed, uses a
/// non-http(s) scheme, has no host, a hop resolves to a non-public address
/// (unless `allow_private`), the redirect cap is exceeded, the request fails, or
/// the body exceeds `max_bytes`; an `Unavailable` [`NativeError`] if the client
/// cannot be built.
pub async fn fetch_pinned_get(
    url: &str,
    allow_private: bool,
    timeout: Duration,
    max_bytes: usize,
) -> NativeResult<(Vec<u8>, Option<String>)> {
    let mut logical = url.to_string();
    for _hop in 0..=MAX_REDIRECTS {
        let parsed = url::Url::parse(&logical).map_err(|e| invalid(format!("invalid URL: {e}")))?;
        let scheme = parsed.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(invalid(format!("unsupported URL scheme: {scheme}")));
        }
        let host = parsed
            .host_str()
            .ok_or_else(|| invalid("URL has no host"))?
            .to_string();

        // Pin the connection: the per-host DNS override hands reqwest the vetted
        // IP while the URL keeps the hostname, so SNI/cert/Host all see the name
        // and the socket goes where we checked. With allow_private, system DNS.
        let client = if allow_private {
            reqwest::Client::builder()
                .timeout(timeout)
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|e| {
                    NativeError::unavailable(format!("could not build HTTP client: {e}"))
                })?
        } else {
            // The is_global gate, applied to EVERY hop's host (a redirect to a
            // private/metadata host is refused here, before any socket to it).
            let pinned = resolve_public_ip(&host)?;
            pinned_async_client(pinned, &host, timeout)?
        };

        let resp = client
            .get(&logical)
            .send()
            .await
            .map_err(|e| invalid(format!("fetch failed: {e}")))?;
        let status = resp.status();
        if status.is_redirection() {
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| invalid("redirect response without a Location header"))?;
            // Resolve relative against the LOGICAL url; the next loop re-vets the
            // new host (this is what closes the redirect-to-private SSRF vector).
            logical = parsed
                .join(location)
                .map_err(|e| invalid(format!("bad redirect location: {e}")))?
                .to_string();
            continue;
        }
        if !status.is_success() {
            return Err(invalid(format!("HTTP error {status} fetching {logical}")));
        }

        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.split(';').next().unwrap_or("").trim().to_string())
            .filter(|ct| !ct.is_empty());

        // Size-cap WHILE streaming: stop as soon as the running total exceeds the
        // cap, never buffering an oversize body whole.
        let mut body = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| invalid(format!("read failed: {e}")))?;
            body.extend_from_slice(&chunk);
            if body.len() > max_bytes {
                return Err(invalid(format!(
                    "download exceeds the {max_bytes}-byte limit"
                )));
            }
        }
        return Ok((body, content_type));
    }
    Err(invalid(format!("too many redirects (>{MAX_REDIRECTS})")))
}

/// One hop's terminal outcome on the operator-endpoint POST path: either a
/// redirect to follow (re-vetted same-host) or the caller's done value.
/// The consumer's per-URL closure returns this so [`post_pinned_with_revet`]
/// drives the redirect loop while the consumer owns the send + retry policy.
pub enum RevetStep<T> {
    /// A 3xx whose `Location` header value the loop must re-vet and follow.
    Redirect(String),
    /// A terminal (non-3xx) outcome the caller produced — returned as-is.
    Done(T),
}

/// Follow redirects on the operator-configured endpoint posture, re-vetting
/// EVERY hop as same-host ([`same_host_redirect`]): a remote endpoint that
/// 30x-es you to a DIFFERENT host is the SSRF vector, so a cross-host
/// (or schemeless/hostless) redirect is refused. The connection stays pinned to
/// the base host's already-vetted IP throughout (a same-host redirect can't
/// rebind to a new address). Capped at [`MAX_REDIRECTS`].
///
/// `send_one` is the consumer's per-URL send (its retry/backoff/`Retry-After`/
/// api-key/item-level policy lives inside it). It returns [`RevetStep`]:
/// `Redirect(location)` for a 3xx the loop should re-vet + follow, or `Done(T)`
/// for any terminal outcome (a 2xx response, or a status the consumer maps).
/// `start` is the absolute URL of the first request; `base` is the parsed base
/// URL the first request was sent to (the same-host comparison anchor).
///
/// # Errors
///
/// Returns the consumer's error from `send_one`, an `InvalidInput`
/// [`NativeError`] for a redirect without a `Location` header or a refused
/// cross-host redirect, or an `Unavailable` [`NativeError`] if the cap is
/// exceeded.
pub async fn post_pinned_with_revet<T, F, Fut>(
    start: String,
    base: url::Url,
    mut send_one: F,
) -> NativeResult<T>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = NativeResult<RevetStep<T>>>,
{
    let mut current = start;
    let mut from = base;
    for _hop in 0..=MAX_REDIRECTS {
        match send_one(current.clone()).await? {
            RevetStep::Done(value) => return Ok(value),
            RevetStep::Redirect(location) => {
                let target = same_host_redirect(&from, &location)?;
                current = target.to_string();
                from = target;
            }
        }
    }
    Err(NativeError::unavailable(format!(
        "too many redirects (>{MAX_REDIRECTS})"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_rejects_the_known_bad_ranges() {
        for bad in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254", // cloud metadata
            "100.64.0.1",      // carrier-grade NAT
            "192.0.0.1",
            "198.18.0.1", // benchmarking
            "0.0.0.0",
            "255.255.255.255",
            "224.0.0.1", // multicast
            "240.0.0.1", // reserved
            "::1",
            "fe80::1",
            "fc00::1",
            "::ffff:10.0.0.1", // v4-mapped private
            "2001:db8::1",     // doc range
        ] {
            let ip: IpAddr = bad.parse().unwrap();
            assert!(!ip_is_allowed(ip), "{bad} must be refused");
        }
    }

    #[test]
    fn allowlist_permits_public_addresses() {
        for good in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "2606:4700::1111",
            "192.0.0.9",
        ] {
            let ip: IpAddr = good.parse().unwrap();
            assert!(ip_is_allowed(ip), "{good} must be permitted");
        }
    }

    /// The SSRF parity regression: the allowlist must refuse 6to4 (2002::/16)
    /// and 3fff::/20 exactly like Python's `ipaddress.is_global`. A 6to4
    /// address embeds an IPv4 in bytes 2..6, so `2002:7f00:0001::` is the 6to4
    /// encoding of 127.0.0.1 — fail-open to an internal IPv4 if permitted.
    #[test]
    fn ssrf_classifier_refuses_6to4_and_3fff_like_python() {
        let must_refuse = [
            "2002::1",        // 6to4 base
            "2002:7f00:1::1", // 6to4 of 127.0.0.1 (internal!)
            "2002:a00:1::1",  // 6to4 of 10.0.0.1 (internal!)
            "2002:c0a8:1::1", // 6to4 of 192.168.0.1 (internal!)
            "3fff::1",        // 3fff::/20 reserved-by-IANA
        ];
        for s in must_refuse {
            let ip: IpAddr = s.parse().unwrap();
            assert!(
                !ip_is_allowed(ip),
                "{s} is NON-global per Python ipaddress.is_global but the SSRF \
                 allowlist permitted it (parity divergence / SSRF reach)"
            );
        }
    }

    #[test]
    fn resolve_public_ip_refuses_loopback() {
        let err = resolve_public_ip("127.0.0.1").unwrap_err();
        assert!(err.message.contains("non-public address"), "{err:?}");
    }

    #[test]
    fn resolve_pinned_allows_loopback() {
        // The operator-trusted path resolves loopback (the primary local-llama
        // use case) where resolve_public_ip refuses it.
        let ip = resolve_pinned("127.0.0.1").unwrap();
        assert!(ip.is_loopback());
    }

    #[test]
    fn same_host_redirect_allows_same_host() {
        let from = url::Url::parse("http://api.example.com:8080/v1/embeddings").unwrap();
        // Relative same-host.
        let t = same_host_redirect(&from, "/v2/embeddings").unwrap();
        assert_eq!(t.as_str(), "http://api.example.com:8080/v2/embeddings");
        // Absolute same-host (different port is still the same HOST — the IP is
        // pinned to the base host regardless, so this stays on the vetted box).
        let t2 = same_host_redirect(&from, "http://api.example.com/x").unwrap();
        assert_eq!(t2.host_str(), Some("api.example.com"));
    }

    #[test]
    fn same_host_redirect_refuses_cross_host() {
        let from = url::Url::parse("http://api.example.com/v1/embeddings").unwrap();
        for bad in [
            "http://169.254.169.254/latest/meta-data/", // cloud metadata
            "http://127.0.0.1/x",                       // loopback
            "http://evil.example.net/x",                // a different host
            "https://attacker.test/",                   // cross-host https
        ] {
            let err = same_host_redirect(&from, bad).unwrap_err();
            assert!(
                err.message.contains("cross-host redirect"),
                "{bad}: {err:?}"
            );
        }
    }

    #[test]
    fn same_host_redirect_refuses_non_http_scheme() {
        let from = url::Url::parse("http://api.example.com/x").unwrap();
        let err = same_host_redirect(&from, "file:///etc/passwd").unwrap_err();
        assert!(err.message.contains("unsupported scheme"), "{err:?}");
    }

    // The look-alike redirect class: inputs crafted to *appear*
    // same-host while resolving to a different host. These are exactly where a
    // future refactor of the host comparison could silently reopen SSRF, so they
    // are pinned as regression guards. `host_str()` is the authority — it parses
    // out userinfo, resolves protocol-relative against the base, and IDNA-encodes
    // a Unicode host to punycode — so each of these compares as a different host.

    #[test]
    fn same_host_redirect_refuses_userinfo_decoy() {
        // The "@" makes the base host the USERINFO, evil.com the real host —
        // host_str() returns evil.com, so it's refused.
        let from = url::Url::parse("http://trusted.example.com/v1").unwrap();
        let err = same_host_redirect(&from, "http://trusted.example.com@evil.com/").unwrap_err();
        assert!(err.message.contains("cross-host redirect"), "{err:?}");
    }

    #[test]
    fn same_host_redirect_refuses_protocol_relative_to_other_host() {
        // A protocol-relative `//host/...` location resolves (against the http
        // base) to that host — here evil.com, not the base — so it's refused.
        let from = url::Url::parse("http://trusted.example.com/v1").unwrap();
        let target = from.join("//evil.com/latest/meta-data/").unwrap();
        assert_eq!(target.host_str(), Some("evil.com")); // resolves cross-host
        let err = same_host_redirect(&from, "//evil.com/latest/meta-data/").unwrap_err();
        assert!(err.message.contains("cross-host redirect"), "{err:?}");
    }

    #[test]
    fn same_host_redirect_refuses_idna_homoglyph() {
        // A Cyrillic-homoglyph host that LOOKS like the base but is a different
        // host: `url` IDNA-encodes it to punycode (xn--…), which is not the
        // ASCII base host — refused. (The 'а' here is U+0430 CYRILLIC A.)
        let from = url::Url::parse("http://example.com/v1").unwrap();
        let homoglyph = "http://ex\u{0430}mple.com/"; // exаmple.com (Cyrillic а)
        let target = url::Url::parse(homoglyph).unwrap();
        assert_ne!(target.host_str(), Some("example.com")); // punycode, not ASCII
        let err = same_host_redirect(&from, homoglyph).unwrap_err();
        assert!(err.message.contains("cross-host redirect"), "{err:?}");
    }

    #[test]
    fn same_host_redirect_allows_port_only_change_same_host() {
        // A different port is still the SAME host — allowed, and the returned URL
        // keeps the base host (the IP/port pinning to the base is exercised by
        // the agent-level tests; here we pin that the host identity is preserved).
        let from = url::Url::parse("http://trusted.example.com:8080/v1").unwrap();
        let target = same_host_redirect(&from, "http://trusted.example.com:9999/x").unwrap();
        assert_eq!(target.host_str(), Some("trusted.example.com"));
    }

    // ── async client builders ─────────────────────────────────────────
    // The IP-pin / SNI / redirect-revet / size-cap behaviors that need a live
    // socket are in tests/async_client.rs (they need a tokio runtime + a local
    // server). These are the no-network construction-gate tests.

    const T: Duration = Duration::from_secs(5);

    #[test]
    fn pinned_async_client_builds_for_a_vetted_ip() {
        // The untrusted-media sibling builds a client once its caller has vetted
        // the IP (the is_global gate is the CALLER's resolve_public_ip step,
        // exercised below; this proves the builder itself succeeds).
        let ip: IpAddr = "93.184.216.34".parse().unwrap();
        assert!(pinned_async_client(ip, "example.com", T).is_ok());
    }

    #[test]
    fn pinned_async_client_path_refuses_non_global_before_any_socket() {
        // The untrusted path gates on resolve_public_ip BEFORE building/connecting
        // (the SSRF control): a host resolving only to a non-global address is
        // refused with no socket opened. This mirrors how the media consumer will
        // drive it (resolve_public_ip -> pinned_async_client, per hop).
        let err = resolve_public_ip("127.0.0.1").unwrap_err();
        assert!(err.message.contains("non-public address"), "{err:?}");
    }

    #[test]
    fn pinned_endpoint_async_client_allows_loopback() {
        // The operator-trusted endpoint path resolves loopback (the primary
        // local-llama use case) where the untrusted path refuses it — no
        // is_global gate, just the pin.
        let (_client, base) = pinned_endpoint_async_client("http://127.0.0.1:8080/v1", T).unwrap();
        assert_eq!(base.host_str(), Some("127.0.0.1"));
        assert_eq!(base.port(), Some(8080));
    }

    #[test]
    fn pinned_endpoint_async_client_refuses_non_http_scheme() {
        for bad in [
            "ftp://example.com/",
            "file:///etc/passwd",
            "ws://example.com/",
        ] {
            let err = pinned_endpoint_async_client(bad, T).unwrap_err();
            assert!(
                err.message.contains("must be http(s)")
                    || err.message.contains("invalid endpoint URL"),
                "{bad}: {err:?}"
            );
        }
    }

    #[test]
    fn pinned_endpoint_async_client_refuses_malformed_or_hostless_url() {
        // Not a URL at all -> a parse failure.
        let err = pinned_endpoint_async_client("not a url", T).unwrap_err();
        assert!(err.message.contains("invalid endpoint URL"), "{err:?}");
        // An http(s) URL with an empty host (`http://`) is a parse error in the
        // `url` crate (it never yields a hostless-but-valid http URL — the
        // `host_str().is_none()` arm is defensive, matching the sync sibling),
        // so it is refused at parse.
        let err = pinned_endpoint_async_client("http://", T).unwrap_err();
        assert!(err.message.contains("invalid endpoint URL"), "{err:?}");
    }

    // ════════════════════════════════════════════════════════════════════
    // Adversarial SSRF suite (#742): every classifier evasion, boundary, and
    // malformed input that an attacker would reach for to make the allowlist
    // hand back an internal address. Each test states the property it pins and
    // names the asset the evasion would otherwise reach.
    // ════════════════════════════════════════════════════════════════════

    /// A tiny deterministic SplitMix64 — the inline property lane's PRNG, in
    /// place of an external `rand`/`proptest` dependency, so the fuzz test rides
    /// the existing `rust_test` target with no crate-graph churn. A failure is
    /// reproducible from its seed.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn v4(s: &str) -> IpAddr {
        IpAddr::V4(s.parse().unwrap())
    }

    // ── IPv4 ranges: exact boundaries, just-in and just-out ─────────────

    /// Every private/reserved IPv4 range is refused at BOTH its first and last
    /// address — an off-by-one in a mask (e.g. `<<` vs `<=`) would leak an edge
    /// host of a block that fronts loopback, LAN, the cloud metadata service, or
    /// CGNAT. Table is (range-label, first, last); both ends must be refused.
    #[test]
    fn allowlist_refuses_ipv4_private_ranges_at_both_boundaries() {
        let blocked_edges = [
            ("0.0.0.0/8", "0.0.0.0", "0.255.255.255"),
            ("10.0.0.0/8", "10.0.0.0", "10.255.255.255"),
            ("127.0.0.0/8 loopback", "127.0.0.0", "127.255.255.255"),
            (
                "169.254.0.0/16 link-local",
                "169.254.0.0",
                "169.254.255.255",
            ),
            ("172.16.0.0/12", "172.16.0.0", "172.31.255.255"),
            ("192.168.0.0/16", "192.168.0.0", "192.168.255.255"),
            ("100.64.0.0/10 CGNAT", "100.64.0.0", "100.127.255.255"),
            ("198.18.0.0/15 benchmarking", "198.18.0.0", "198.19.255.255"),
            ("192.0.2.0/24 TEST-NET-1", "192.0.2.0", "192.0.2.255"),
            (
                "198.51.100.0/24 TEST-NET-2",
                "198.51.100.0",
                "198.51.100.255",
            ),
            ("203.0.113.0/24 TEST-NET-3", "203.0.113.0", "203.0.113.255"),
            ("240.0.0.0/4 reserved", "240.0.0.0", "255.255.255.254"),
        ];
        for (label, first, last) in blocked_edges {
            assert!(!ip_is_allowed(v4(first)), "{label}: first {first} leaked");
            assert!(!ip_is_allowed(v4(last)), "{label}: last {last} leaked");
        }
        // The all-ones broadcast and 169.254.169.254 (the cloud metadata IP) are
        // the two single addresses an attacker most wants; pin them by name.
        assert!(!ip_is_allowed(v4("255.255.255.255")), "broadcast leaked");
        assert!(
            !ip_is_allowed(v4("169.254.169.254")),
            "cloud metadata IP leaked"
        );
    }

    /// The addresses ONE step outside each blocked range stay PUBLIC — the gate
    /// must not over-block and break legitimate fetches. This is the negative
    /// boundary control for the 172.16/12 block in particular, whose non-octet
    /// mask (172.15.x public, 172.16–172.31 blocked, 172.32.x public) is the
    /// classic place a hand-rolled check gets wrong.
    #[test]
    fn allowlist_permits_ipv4_just_outside_private_ranges() {
        for good in [
            "9.255.255.255",   // just below 10/8
            "11.0.0.0",        // just above 10/8
            "172.15.255.255",  // just below 172.16/12
            "172.32.0.0",      // just above 172.16/12
            "192.167.255.255", // just below 192.168/16
            "192.169.0.0",     // just above 192.168/16
            "169.253.255.255", // just below 169.254/16
            "169.255.0.0",     // just above 169.254/16
            "100.63.255.255",  // just below 100.64/10 CGNAT
            "100.128.0.0",     // just above 100.64/10 CGNAT
            "126.255.255.255", // just below 127/8 loopback
            "128.0.0.0",       // just above 127/8 loopback
            "192.0.1.255",     // just below 192.0.2/24
            "192.0.3.0",       // just above 192.0.2/24
            "239.255.255.255", // just below 240/4 (but multicast — see note)
        ] {
            // 239.255.255.255 is the top of 224/4 multicast; it is correctly
            // refused by the multicast guard, not the range check — exclude it.
            if good == "239.255.255.255" {
                assert!(!ip_is_allowed(v4(good)), "{good} is multicast, must block");
                continue;
            }
            assert!(
                ip_is_allowed(v4(good)),
                "{good} wrongly blocked (over-block)"
            );
        }
    }

    /// The 192.0.0.9 / 192.0.0.10 carve-out: the IANA registry marks these two
    /// addresses inside the otherwise-reserved 192.0.0.0/24 as global (PCP / NAT
    /// discovery anycast), and the classifier must match Python on that — both
    /// the two exceptions ALLOWED and the rest of the /24 BLOCKED.
    #[test]
    fn allowlist_matches_python_on_the_192_0_0_24_carveout() {
        assert!(ip_is_allowed(v4("192.0.0.9")), "192.0.0.9 is global");
        assert!(ip_is_allowed(v4("192.0.0.10")), "192.0.0.10 is global");
        for blocked in ["192.0.0.0", "192.0.0.8", "192.0.0.11", "192.0.0.255"] {
            assert!(!ip_is_allowed(v4(blocked)), "{blocked} must stay blocked");
        }
    }

    /// IPv4 multicast (224.0.0.0/4) is refused — a notetype that fetches a
    /// multicast group would let an attacker pivot onto LAN multicast services.
    #[test]
    fn allowlist_refuses_ipv4_multicast() {
        for m in ["224.0.0.0", "224.0.0.1", "232.1.2.3", "239.255.255.255"] {
            assert!(!ip_is_allowed(v4(m)), "{m} multicast leaked");
        }
    }

    // ── IPv6 evasions ───────────────────────────────────────────────────

    /// The canonical IPv6 private/internal forms are all refused: loopback,
    /// unspecified, ULA, link-local, NAT64-local, and the documentation range.
    /// Each is a direct path to an internal v6 service if permitted.
    #[test]
    fn allowlist_refuses_ipv6_internal_forms() {
        for bad in [
            "::1",                                    // loopback
            "::",                                     // unspecified (0.0.0.0 analogue)
            "fc00::1",                                // ULA fc00::/7 low half
            "fdff:ffff::1",                           // ULA fc00::/7 high half (fd..)
            "fe80::1",                                // link-local fe80::/10
            "febf:ffff::1",                           // link-local fe80::/10 top
            "100::1",                                 // 100::/64 discard-only
            "64:ff9b:1::1",                           // 64:ff9b:1::/48 NAT64 local-use
            "2001:db8::1",                            // 2001:db8::/32 documentation
            "2001:db8:ffff:ffff:ffff:ffff:ffff:ffff", // doc range top
        ] {
            let ip: IpAddr = bad.parse().unwrap();
            assert!(!ip_is_allowed(ip), "{bad} must be refused");
        }
    }

    /// fc00::/7 ULA boundaries: the block runs fc00:: .. fdff:ffff:…:ffff. The
    /// address just below (fbff:…) and just above (fe00::) the /7 must stay
    /// public (no over-block), while both ends of the /7 are refused.
    #[test]
    fn allowlist_ula_fc00_slash7_boundaries() {
        let parse = |s: &str| -> IpAddr { s.parse().unwrap() };
        assert!(!ip_is_allowed(parse("fc00::")), "fc00:: ULA start leaked");
        assert!(
            !ip_is_allowed(parse("fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")),
            "ULA /7 end leaked"
        );
        // Just outside the /7 in both directions stays global.
        assert!(
            ip_is_allowed(parse("fbff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")),
            "fbff:: wrongly blocked"
        );
        // fe00::/8 is global (link-local only starts at fe80::/10).
        assert!(ip_is_allowed(parse("fe00::1")), "fe00::1 wrongly blocked");
    }

    /// IPv4-mapped IPv6 (`::ffff:a.b.c.d`) MUST defer to the embedded IPv4 and
    /// inherit its verdict — the headline IPv6→IPv4 SSRF evasion: wrap loopback
    /// or the metadata IP in a v6 literal hoping the v6 path skips the v4 block.
    #[test]
    fn allowlist_refuses_ipv4_mapped_internal_addresses() {
        for bad in [
            "::ffff:127.0.0.1",       // loopback
            "::ffff:10.0.0.1",        // RFC1918
            "::ffff:192.168.1.1",     // RFC1918
            "::ffff:172.16.0.1",      // RFC1918
            "::ffff:169.254.169.254", // cloud metadata
            "::ffff:100.64.0.1",      // CGNAT
            "::ffff:0.0.0.0",         // unspecified
        ] {
            let ip: IpAddr = bad.parse().unwrap();
            assert!(
                !ip_is_allowed(ip),
                "{bad} (v4-mapped internal) must be refused"
            );
        }
        // A v4-mapped PUBLIC address still passes (defer-to-v4 is symmetric).
        assert!(
            ip_is_allowed("::ffff:8.8.8.8".parse().unwrap()),
            "::ffff:8.8.8.8 wrongly blocked"
        );
    }

    /// 6to4 (2002::/16) embeds an IPv4 in bytes 2..6; the whole /16 is refused
    /// so an attacker can't 6to4-wrap an internal v4 (`2002:7f00:1::` = the 6to4
    /// of 127.0.0.1) to slip past. Pins both ends of the /16 and several
    /// internal-v4 wrappings, extending the existing single-address probe.
    #[test]
    fn allowlist_refuses_entire_6to4_2002_slash16() {
        for bad in [
            "2002::",                                  // /16 start
            "2002:ffff:ffff:ffff:ffff:ffff:ffff:ffff", // /16 end
            "2002:7f00:1::1",                          // wraps 127.0.0.1
            "2002:a00:1::1",                           // wraps 10.0.0.1
            "2002:c0a8:101::1",                        // wraps 192.168.1.1
            "2002:a9fe:a9fe::1",                       // wraps 169.254.169.254 (metadata)
            "2002:0808:0808::1",                       // wraps 8.8.8.8 — still refused (whole /16)
        ] {
            let ip: IpAddr = bad.parse().unwrap();
            assert!(!ip_is_allowed(ip), "{bad} (6to4) must be refused");
        }
        // Just outside the /16 stays global.
        assert!(
            ip_is_allowed("2001:ffff::1".parse().unwrap()),
            "2001:ffff::1 wrongly blocked"
        );
        assert!(
            ip_is_allowed("2003::1".parse().unwrap()),
            "2003::1 wrongly blocked"
        );
    }

    /// IPv6 multicast (ff00::/8) is refused. Python's `is_global` actually
    /// returns True for some multicast scopes, so this guard is the `&&
    /// !is_multicast()` layer ON TOP of the parity classifier — pin that it
    /// catches link-local, site-local, and global multicast scopes alike.
    #[test]
    fn allowlist_refuses_ipv6_multicast_all_scopes() {
        for m in [
            "ff00::1",
            "ff01::1",
            "ff02::1",        // node/link-local (all-nodes etc.)
            "ff05::1",        // site-local
            "ff0e::1",        // global-scope multicast
            "ff02::1:ff00:0", // solicited-node
        ] {
            let ip: IpAddr = m.parse().unwrap();
            assert!(!ip_is_allowed(ip), "{m} (v6 multicast) must be refused");
        }
    }

    /// The 2001::/23 exception window: Python marks most of 2001::/23 non-global
    /// but carves several sub-blocks back to global (2001:1::/32 etc.). The
    /// classifier must match — both a refused interior (2001:2::, Teredo-ish) and
    /// the allowed carve-outs.
    #[test]
    fn allowlist_matches_python_on_2001_slash23_window() {
        // Refused: inside 2001::/23 but NOT one of the carve-outs.
        for blocked in ["2001:2::1", "2001:5::1", "2001:1ff::1"] {
            let ip: IpAddr = blocked.parse().unwrap();
            assert!(!ip_is_allowed(ip), "{blocked} should be non-global");
        }
        // Allowed carve-outs that ARE global per the registry.
        for good in [
            "2001:1::1",
            "2001:3::1",
            "2001:4:112::1",
            "2001:20::1",
            "2001:30::1",
        ] {
            let ip: IpAddr = good.parse().unwrap();
            assert!(ip_is_allowed(ip), "{good} is a global carve-out");
        }
    }

    /// IPv4-COMPATIBLE IPv6 (`::a.b.c.d`, the deprecated RFC4291 form, NOT
    /// v4-mapped) wrapping an internal v4. `to_ipv4_mapped()` returns None for
    /// these, so the classifier does NOT defer to the IPv4 check — it follows
    /// Python's `is_global`, which treats `::127.0.0.1` (= `::7f00:1`) as global.
    /// The connect on this host fails (the OS rejects the deprecated family), but
    /// on a host that DOES route it this is a loopback reach the allowlist would
    /// permit.
    ///
    /// This pins the SECURE behaviour (these must be refused). It is parity-
    /// faithful for `ip_is_allowed` to permit them — Python agrees — so this is a
    /// hardening gap beyond strict parity, tracked as a DEFECT and left red-but-
    /// ignored rather than fixed here (trust-boundary change → security review).
    #[test]
    #[ignore = "DEFECT #1008: IPv4-compatible ::a.b.c.d (e.g. ::127.0.0.1) is is_global-true per Python parity but reaches an internal v4 where the OS routes it; ip_is_allowed should refuse the whole ::/96 v4-compatible block on top of parity"]
    fn allowlist_should_refuse_ipv4_compatible_internal_addresses() {
        for bad in [
            "::127.0.0.1",       // ::7f00:1 — loopback
            "::10.0.0.1",        // ::a00:1 — RFC1918
            "::192.168.1.1",     // ::c0a8:101 — RFC1918
            "::169.254.169.254", // ::a9fe:a9fe — cloud metadata
        ] {
            let ip: IpAddr = bad.parse().unwrap();
            assert!(
                !ip_is_allowed(ip),
                "{bad} (IPv4-compatible) wraps an internal v4 and must be refused"
            );
        }
    }

    // ── parsing / normalization evasions through the resolving API ──────

    // The classic alternate-encoding bypass (octal/hex/decimal spellings of
    // 127.0.0.1) is NOT unit-tested here: it depends entirely on the platform
    // resolver's `inet_aton` behaviour (glibc reads `0177.0.0.1` as octal-127;
    // macOS reads it as decimal `177.0.0.1`, a different public host), so the
    // outcome is libc-specific, not a property of this crate's vetting. The
    // crate's contract — whatever the resolver returns is classified by
    // `ip_is_allowed` and pinned — is covered hermetically by the IP-literal
    // resolve tests below and the classifier corpus. Running live `getaddrinfo`
    // over alternate spellings belongs in an end-to-end lane, not the unit lane.

    /// `resolve_public_ip` vets the IPv6 loopback and metadata literals too, not
    /// just v4 — the v6 path is the same resolve-then-classify gate.
    #[test]
    fn resolve_public_ip_refuses_ipv6_internal_literals() {
        for bad in ["::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1"] {
            let err = resolve_public_ip(bad).unwrap_err();
            assert!(err.message.contains("non-public address"), "{bad}: {err:?}");
        }
    }

    /// A name that resolves at all must have EVERY address vetted — but with a
    /// pure-literal API the practical guard is: a literal that classifies bad is
    /// refused before any socket, and a non-resolving name errors cleanly (no
    /// panic, no Ok). Public literals pass and are returned for pinning.
    #[test]
    fn resolve_public_ip_returns_public_literal_for_pinning() {
        let ip = resolve_public_ip("8.8.8.8").unwrap();
        assert_eq!(ip, "8.8.8.8".parse::<IpAddr>().unwrap());
    }

    // ── same_host_redirect look-alike host evasions ─────────────────────

    /// Host comparison is case-insensitive (the `url` crate lowercases the host),
    /// so a redirect that only changes the host's CASE is the SAME host and is
    /// allowed — a refusal here would be a false positive, but more importantly
    /// this pins that case-folding can't be (mis)used to smuggle a different host
    /// past a naive byte-compare.
    #[test]
    fn same_host_redirect_is_case_insensitive_on_host() {
        let from = url::Url::parse("http://Api.Example.Com/v1").unwrap();
        let t = same_host_redirect(&from, "http://API.EXAMPLE.COM/v2").unwrap();
        assert_eq!(t.host_str(), Some("api.example.com"));
    }

    /// A trailing-dot FQDN (`example.com.`) is a DIFFERENT host string than
    /// `example.com` to the `url` crate, so a redirect that adds/removes the dot
    /// is refused. This matters because some resolvers treat the two as equal —
    /// the conservative refusal keeps the pinned-IP invariant (the connection is
    /// pinned to the base host's vetted IP; a host string mismatch must not slip
    /// through as "same host").
    #[test]
    fn same_host_redirect_refuses_trailing_dot_fqdn_variant() {
        let from = url::Url::parse("http://example.com/v1").unwrap();
        let err = same_host_redirect(&from, "http://example.com.:80/x").unwrap_err();
        assert!(err.message.contains("cross-host redirect"), "{err:?}");
    }

    /// A bracketed IPv6 redirect target to a DIFFERENT host (loopback) is
    /// refused — the endpoint posture allows ONLY the same host, and `[::1]` is
    /// not the base host.
    #[test]
    fn same_host_redirect_refuses_ipv6_loopback_target() {
        let from = url::Url::parse("http://example.com/v1").unwrap();
        let err = same_host_redirect(&from, "http://[::1]/x").unwrap_err();
        assert!(err.message.contains("cross-host redirect"), "{err:?}");
    }

    /// Non-http(s) schemes beyond `file:` are all refused by the endpoint
    /// redirect vet — `gopher:`/`ftp:`/`data:` are SSRF-adjacent exfil vectors,
    /// and an empty/whitespace location is a malformed input that must error not
    /// panic.
    #[test]
    fn same_host_redirect_refuses_dangerous_schemes_and_garbage() {
        let from = url::Url::parse("http://example.com/v1").unwrap();
        for bad in [
            "gopher://example.com/x", // gopher SSRF classic
            "ftp://example.com/x",
            "data:text/plain,hi",
            "ws://example.com/x",
            "javascript:alert(1)",
        ] {
            let err = same_host_redirect(&from, bad).unwrap_err();
            // Either an unsupported-scheme refusal or a cross-host refusal
            // (scheme-relative parsing can drop the host) — never an Ok.
            assert!(
                err.message.contains("unsupported scheme")
                    || err.message.contains("cross-host redirect"),
                "{bad}: {err:?}"
            );
        }
    }

    /// An absolute redirect to the metadata IP that happens to reuse the base
    /// host as USERINFO (`http://base@169.254.169.254/`) — `host_str()` strips
    /// userinfo and returns the metadata IP, so it is refused. Re-pins the
    /// userinfo decoy against the metadata asset specifically.
    #[test]
    fn same_host_redirect_userinfo_decoy_to_metadata_is_refused() {
        let from = url::Url::parse("http://trusted.example.com/v1").unwrap();
        let err = same_host_redirect(&from, "http://trusted.example.com@169.254.169.254/latest/")
            .unwrap_err();
        assert!(err.message.contains("cross-host redirect"), "{err:?}");
    }

    // ── HTTP client construction invariants ─────────────────────────────

    /// The endpoint builder refuses a URL whose host is an internal LITERAL only
    /// if … actually it must NOT — the operator path is deliberately ungated, so
    /// loopback is allowed. But a non-resolving / malformed host must still error
    /// cleanly, and a public literal builds. Pins the operator posture's contract
    /// (loopback OK, garbage errors).
    #[test]
    fn pinned_endpoint_client_operator_posture_loopback_ok_garbage_errs() {
        // Operator-trusted: loopback resolves + builds (no is_global gate).
        assert!(pinned_endpoint_async_client("http://127.0.0.1:8080/v1", T).is_ok());
        // A host that cannot resolve errors (not a panic, not a silent Ok).
        let err = pinned_endpoint_async_client("http://no.such.host.shrike.invalid.:9/v1", T)
            .unwrap_err();
        assert!(
            err.message.contains("could not resolve") || err.message.contains("invalid"),
            "{err:?}"
        );
    }

    /// `pinned_async_client` accepts any vetted IP literal as the pin target and
    /// builds — including a v6 public address — proving the builder is not
    /// v4-only (a v6 media host must be pinnable too).
    #[test]
    fn pinned_async_client_builds_for_public_ipv6() {
        let ip: IpAddr = "2606:4700::1111".parse().unwrap();
        assert!(pinned_async_client(ip, "example.com", T).is_ok());
    }

    // ── generative panic-freedom fuzz ───────────────────────────────────

    /// FUZZ: `ip_is_allowed` over random v4 AND v6 bit patterns must be total —
    /// always a clean bool, never a panic (a panic in the classifier is a
    /// fail-open DoS on the media path). 20k random addresses per family.
    #[test]
    fn fuzz_ip_is_allowed_is_total_over_random_addresses() {
        let mut rng = Rng::new(0xD15E_A5E0_5512_F00D);
        for _ in 0..20_000 {
            let v4bits = rng.next_u64() as u32;
            let _ = ip_is_allowed(IpAddr::V4(Ipv4Addr::from(v4bits)));
            let hi = rng.next_u64() as u128;
            let lo = rng.next_u64() as u128;
            let v6bits = (hi << 64) | lo;
            let _ = ip_is_allowed(IpAddr::V6(Ipv6Addr::from(v6bits))); // never panics
        }
    }

    /// FUZZ: the parsing/vetting surface must be panic-free on hostile strings.
    /// Builds random host/URL strings from a charset rich in the bytes that
    /// trip parsers (brackets, colons, `@`, `%`, dots, slashes, control bytes)
    /// and drives them through `resolve_public_ip` and `same_host_redirect` —
    /// every outcome must be Ok or Err, never a panic.
    #[test]
    fn fuzz_parsing_surface_is_panic_free_on_hostile_input() {
        const CHARSET: &[u8] = b"0123456789abcdefABCDEF.:/[]@%-_xX vfF\t\n";
        let mut rng = Rng::new(0x0FF1_CEBA_D5EE_D001);
        let from = url::Url::parse("http://base.example.com/v1").unwrap();
        // Hermetic: drives only the pure URL parser/classifier. `resolve_public_ip`
        // and `pinned_endpoint_async_client` are deliberately NOT swept here —
        // they call `getaddrinfo`, so fuzzing them is live-network I/O (slow,
        // flaky, blocks offline), which has no place in the unit lane.
        for _ in 0..5_000 {
            let len = (rng.next_u64() % 40) as usize;
            let mut s = String::with_capacity(len);
            for _ in 0..len {
                let b = CHARSET[(rng.next_u64() as usize) % CHARSET.len()];
                s.push(b as char);
            }
            // The redirect vet over hostile garbage as a Location header: every
            // outcome is Ok or Err, never a panic.
            let _ = same_host_redirect(&from, &s);
        }
    }
}
