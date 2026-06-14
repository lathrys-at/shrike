//! The SSRF-guarded media URL fetch (#278 step 5b; kernel-owned since
//! #389 B2) — the port of `_resolve_public_ip` / `_fetch_media_url` /
//! `_decode_media_b64` (`path_within_any_root` stayed with the collection's
//! write gates). **Trust-boundary code**: changes here go through the
//! security-review gate.
//!
//! Pure Rust networking, NO anki coupling and NO async runtime: `ureq` is a
//! synchronous client (no tokio, no owned threads — the #308 constraint).
//! The SSRF model mirrors the Python implementation exactly:
//!
//! - http/https only; every hop's host is resolved and refused unless every
//!   address is **globally routable** (an allowlist mirroring Python
//!   `ipaddress.is_global`, multicast rejected explicitly — parity-tested
//!   against the Python classifier over an address corpus);
//! - the connection is **pinned to the vetted IP** via a custom resolver, so
//!   the URL keeps the hostname (TLS SNI + certificate validation verify the
//!   name, the Host header is right by construction) while the socket
//!   connects to the address we vetted — closing the DNS-rebinding TOCTOU;
//! - redirects are followed **manually**, re-vetting every hop, capped at
//!   [`MAX_MEDIA_REDIRECTS`];
//! - the body is size-capped while streaming ([`MEDIA_MAX_BYTES`]).
//!
//! With `allow_private` the guard and pinning are off (the operator opted
//! into trusted internal hosts) — same switch as the Python facade.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};

use base64::Engine;
use shrike_ffi::{NativeError, NativeResult};
use shrike_schemas::StoreMediaItem;
use shrike_store_api::{PreparedMedia, PreparedMediaSource};
use url::Url;

pub use shrike_store_api::MEDIA_MAX_BYTES;
pub const URL_FETCH_TIMEOUT_SECS: u64 = 30;
pub const MAX_MEDIA_REDIRECTS: usize = 5;

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

/// `_resolve_public_ip`: resolve and vet EVERY address (a name can't smuggle
/// an internal A record alongside a public one); return the first so the
/// caller pins the connection to it.
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

/// `_decode_media_b64`: cap on the ENCODED length first (base64 is ~4/3 of
/// the payload, so the string length bounds the decoded size — no allocating
/// an oversize payload just to reject it).
pub fn decode_media_b64(data: &str) -> NativeResult<Vec<u8>> {
    if data.len() > MEDIA_MAX_BYTES / 3 * 4 + 4 {
        return Err(invalid(format!(
            "file exceeds the {MEDIA_MAX_BYTES}-byte limit"
        )));
    }
    let cleaned: String = data.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .map_err(|e| invalid(format!("invalid base64 data: {e}")))
}

/// `_fetch_media_url`: download into memory, returning (bytes, content_type).
pub fn fetch_media_url(url: &str, allow_private: bool) -> NativeResult<(Vec<u8>, Option<String>)> {
    let mut logical = url.to_string();
    for _hop in 0..=MAX_MEDIA_REDIRECTS {
        let parsed = Url::parse(&logical).map_err(|e| invalid(format!("invalid URL: {e}")))?;
        let scheme = parsed.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(invalid(format!("unsupported URL scheme: {scheme}")));
        }
        let host = parsed
            .host_str()
            .ok_or_else(|| invalid("URL has no host"))?
            .to_string();

        // Pin the connection: the resolver hands ureq the vetted IP while the
        // URL keeps the hostname, so SNI/cert/Host all see the name and the
        // socket goes where we checked. With allow_private, system resolution.
        let agent = if allow_private {
            ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(URL_FETCH_TIMEOUT_SECS))
                .redirects(0)
                .build()
        } else {
            let pinned = resolve_public_ip(&host)?;
            // The agent is rebuilt per hop and resolves only this URL, so the
            // effective port comes from the parsed URL itself — the netloc
            // string ureq hands the resolver doesn't split cleanly for
            // bracketed IPv6 literals (#382).
            let port = parsed.port_or_known_default().unwrap_or(0);
            ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(URL_FETCH_TIMEOUT_SECS))
                .redirects(0)
                .resolver(move |_netloc: &str| -> std::io::Result<Vec<SocketAddr>> {
                    Ok(vec![SocketAddr::new(pinned, port)])
                })
                .build()
        };

        // ureq surfaces 3xx either as Ok (redirects disabled) or Error::Status
        // depending on version details — handle both identically.
        let mut redirect_from: Option<ureq::Response> = None;
        let response = match agent.get(&logical).call() {
            Ok(resp) if (300..400).contains(&resp.status()) => {
                redirect_from = Some(resp);
                None
            }
            Ok(resp) => Some(resp),
            Err(ureq::Error::Status(code, resp)) if (300..400).contains(&code) => {
                redirect_from = Some(resp);
                None
            }
            Err(ureq::Error::Status(code, _)) => {
                return Err(invalid(format!("HTTP error {code} fetching {logical}")));
            }
            Err(e) => return Err(invalid(format!("fetch failed: {e}"))),
        };
        if let Some(resp) = redirect_from {
            let location = resp
                .header("location")
                .ok_or_else(|| invalid("redirect response without a Location header"))?;
            // resolve relative against the LOGICAL url; re-vet next loop
            logical = parsed
                .join(location)
                .map_err(|e| invalid(format!("bad redirect location: {e}")))?
                .to_string();
            continue;
        }
        let response = response.expect("non-redirect response present");

        let content_type = response
            .header("content-type")
            .map(|ct| ct.split(';').next().unwrap_or("").trim().to_string())
            .filter(|ct| !ct.is_empty());
        let mut reader = response.into_reader().take(MEDIA_MAX_BYTES as u64 + 1);
        let mut body = Vec::new();
        reader
            .read_to_end(&mut body)
            .map_err(|e| invalid(format!("read failed: {e}")))?;
        if body.len() > MEDIA_MAX_BYTES {
            return Err(invalid(format!(
                "download exceeds the {MEDIA_MAX_BYTES}-byte limit"
            )));
        }
        return Ok((body, content_type));
    }
    Err(invalid(format!(
        "too many redirects (>{MAX_MEDIA_REDIRECTS})"
    )))
}

/// The filename a URL's path implies (basename-sanitized via the same rule
/// the collection's media names use), for a `url` store item with no
/// explicit `filename`.
pub fn media_name_from_url(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    let base = parsed
        .path()
        .replace('\\', "/")
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_string();
    let checked = base.trim();
    if checked.is_empty() || checked == "." || checked == ".." {
        None
    } else {
        Some(base)
    }
}

/// One store_media item's prepare — validate, then decode/fetch the byte
/// source (path items pass through; their gates are collection policy and
/// run under the write). The ONE prepare both drivers share: the kernel's
/// concurrent op fans it onto the blocking pool, the binding's sequential
/// edge calls it inline.
pub fn prepare_media_item(
    index: i64,
    item: StoreMediaItem,
    allow_private_fetch: bool,
) -> PreparedMedia {
    let filename = item.filename.clone();
    let source = if let Err(e) = item.validate() {
        PreparedMediaSource::Failed { error: e }
    } else if let Some(path) = item.path {
        PreparedMediaSource::Path { path }
    } else if let Some(data) = item.data.as_deref() {
        match decode_media_b64(data) {
            Ok(bytes) => PreparedMediaSource::Bytes {
                name: item.filename.unwrap_or_default(),
                data: bytes,
                content_type: None,
            },
            Err(e) => PreparedMediaSource::Failed { error: e.message },
        }
    } else if let Some(url) = item.url.as_deref() {
        match fetch_media_url(url, allow_private_fetch) {
            Ok((bytes, ct)) => PreparedMediaSource::Bytes {
                name: item
                    .filename
                    .clone()
                    .or_else(|| media_name_from_url(url))
                    .unwrap_or_default(),
                data: bytes,
                content_type: ct,
            },
            Err(e) => PreparedMediaSource::Failed { error: e.message },
        }
    } else {
        // validate() guarantees one source; backstop message kept.
        PreparedMediaSource::Failed {
            error: "each item needs one of data, url, or path".to_string(),
        }
    };
    PreparedMedia {
        index,
        filename,
        source,
    }
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

    #[test]
    fn b64_cap_applies_before_decoding() {
        let oversize = "A".repeat(MEDIA_MAX_BYTES / 3 * 4 + 8);
        assert!(decode_media_b64(&oversize).is_err());
        assert_eq!(decode_media_b64("aGk=").unwrap(), b"hi");
        assert!(decode_media_b64("not base64!!").is_err());
    }

    #[test]
    fn scheme_and_host_validation() {
        assert!(fetch_media_url("ftp://example.com/x", false).is_err());
        assert!(fetch_media_url("not a url", false).is_err());
        // loopback never leaves the building, even before any socket opens
        let err = fetch_media_url("http://127.0.0.1:1/x", false).unwrap_err();
        assert!(err.message.contains("non-public address"));
    }
}
