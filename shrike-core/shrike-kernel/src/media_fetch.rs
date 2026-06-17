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

use base64::Engine;
use shrike_collection::{PreparedMedia, PreparedMediaSource};
use shrike_error::{NativeError, NativeResult};
use shrike_schemas::StoreMediaItem;
use url::Url;

// The SSRF classifier + the resolve-and-vet helper now live in the shared
// `shrike-network` crate (#592) so the remote engine crates use the SAME control.
// Re-exported here so the kernel's Python-facing binding
// (anki_core::media_ip_allowed) and any in-tree caller keep importing them from
// `media_fetch` unchanged — a pure move, store_media SSRF behavior + the parity
// corpus are byte-identical.
pub use shrike_network::{ip_is_allowed, resolve_public_ip};

pub use shrike_store::MEDIA_MAX_BYTES;
pub const URL_FETCH_TIMEOUT_SECS: u64 = 30;
/// The redirect hop cap (= `shrike_network::MAX_REDIRECTS`, kept as a named alias so
/// existing call sites and the error message that quotes it are unchanged).
pub const MAX_MEDIA_REDIRECTS: usize = shrike_network::MAX_REDIRECTS;

fn invalid(msg: impl Into<String>) -> NativeError {
    NativeError::invalid_input(msg)
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
        let timeout = std::time::Duration::from_secs(URL_FETCH_TIMEOUT_SECS);
        let agent = if allow_private {
            ureq::AgentBuilder::new()
                .timeout(timeout)
                .redirects(0)
                .build()
        } else {
            let pinned = resolve_public_ip(&host)?;
            // The agent is rebuilt per hop and resolves only this URL, so the
            // effective port comes from the parsed URL itself — the netloc
            // string ureq hands the resolver doesn't split cleanly for
            // bracketed IPv6 literals (#382).
            let port = parsed.port_or_known_default().unwrap_or(0);
            shrike_network::pinned_agent(pinned, port, timeout)
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
    use std::net::IpAddr;

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
