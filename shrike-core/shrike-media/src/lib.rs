//! `shrike-media` — the inbound/untrusted-media crate (#711, epic #703).
//!
//! The "acquire + validate untrusted bytes" half of media handling, cracked
//! out of `shrike-kernel`'s `media_fetch`. The *store* half (store/fetch/list/
//! delete/prune) stays in `shrike-collection` — it is inseparable from
//! `col.media`. This crate owns:
//!
//! - the SSRF-guarded media URL fetch ([`fetch_media_url`]) + base64 decode
//!   ([`decode_media_b64`]) + the [`MEDIA_MAX_BYTES`] cap;
//! - the per-item prepare ([`prepare_media_item`]) the kernel fans onto its
//!   blocking pool, producing a [`PreparedMedia`] the collection's write tail
//!   consumes;
//! - [`safe_media_name`], the basename path-traversal guard;
//! - the one extension→MIME home the store half reads
//!   ([`guess_mime`]/[`mime_extension`]).
//!
//! **Trust-boundary code**: changes here go through the security-review gate.
//! The SSRF model (origin of #164/#71/#72) is preserved exactly — it delegates
//! the primitives to `shrike-network` (the one home for the is_global
//! classifier + the IP-pinned agent), so the control is shared, not copied:
//!
//! - http/https only; every hop's host is resolved and refused unless every
//!   address is **globally routable** (`shrike_network::ip_is_allowed`, an
//!   allowlist mirroring Python `ipaddress.is_global`, parity-tested);
//! - the connection is **pinned to the vetted IP** via a custom resolver, so
//!   the URL keeps the hostname (TLS SNI + certificate validation verify the
//!   name, the Host header is right by construction) while the socket connects
//!   to the address we vetted — closing the DNS-rebinding TOCTOU;
//! - redirects are followed **manually**, re-vetting every hop, capped at
//!   [`MAX_MEDIA_REDIRECTS`];
//! - the body is size-capped while streaming ([`MEDIA_MAX_BYTES`]).
//!
//! With `allow_private` the guard and pinning are off (the operator opted into
//! trusted internal hosts) — same switch as the Python facade.
//!
//! Pure Rust, NO anki coupling and NO async runtime (`ureq` is synchronous —
//! the #308 constraint), NO engine crate and NOT `shrike-kernel`: it sits BELOW
//! both, so the kernel-purity and engine-purity layering rules both stay
//! satisfied (//shrike-core:layering_check).

use std::io::Read;

use base64::Engine;
use shrike_error::{NativeError, NativeResult};
use shrike_schemas::StoreMediaItem;
use url::Url;

// The SSRF classifier + the resolve-and-vet helper live in the shared
// `shrike-network` crate (#592) so the remote engine crates use the SAME
// control. Re-exported here so the in-tree callers (the kernel's Python-facing
// binding, any media caller) keep importing them from `shrike_media` unchanged.
pub use shrike_network::{ip_is_allowed, resolve_public_ip};

/// The byte-source size cap — caller-supplied/downloaded bytes only; a
/// server-local `path` inside an operator-configured root is deliberately
/// uncapped. The one policy value the collection write tail and the
/// fetch/decode caps must agree on, so it lives where both can see it (#711:
/// rehomed from shrike-store now that both sides depend on shrike-media).
pub const MEDIA_MAX_BYTES: usize = 64 * 1024 * 1024;

pub const URL_FETCH_TIMEOUT_SECS: u64 = 30;
/// The redirect hop cap (= `shrike_network::MAX_REDIRECTS`, kept as a named alias so
/// existing call sites and the error message that quotes it are unchanged).
pub const MAX_MEDIA_REDIRECTS: usize = shrike_network::MAX_REDIRECTS;

/// One store_media item after the kernel's off-actor prepare (#490): byte
/// sources arrive fetched/decoded; `path` items pass through whole (their
/// gates are collection policy and run under the write); a failed prepare
/// carries its per-item error. The interface between the acquire half (this
/// crate) and the store half (`shrike-collection`, which re-exports this).
pub struct PreparedMedia {
    pub index: i64,
    /// The caller's `filename`, echoed on errors.
    pub filename: Option<String>,
    pub source: PreparedMediaSource,
}

pub enum PreparedMediaSource {
    /// Decoded base64 or a completed download; `name` already folds the
    /// URL-derived fallback.
    Bytes {
        name: String,
        data: Vec<u8>,
        content_type: Option<String>,
    },
    /// A server-local path item, gated under the write.
    Path { path: String },
    /// The prepare failed (bad base64, refused/failed download, invalid
    /// item); stored nothing.
    Failed { error: String },
}

fn invalid(msg: impl Into<String>) -> NativeError {
    NativeError::invalid_input(msg)
}

/// `_safe_media_name`: reduce a caller-supplied name to a bare basename so it
/// can only resolve inside the media dir (path-traversal guard for
/// fetch/delete). Returns "" for a name that is only separators/dots — or
/// only whitespace around them, which the emptiness check would otherwise
/// pass (#382).
pub fn safe_media_name(name: &str) -> String {
    let normalized = name.replace('\\', "/");
    let trimmed = normalized.trim_end_matches('/');
    let base = trimmed.rsplit('/').next().unwrap_or("");
    let checked = base.trim();
    if checked.is_empty() || checked == "." || checked == ".." {
        String::new()
    } else {
        base.to_string()
    }
}

/// Best-effort MIME from the filename extension (the subset the Python
/// `mimetypes` table returns for media Anki actually stores). The store half's
/// fetch/list responses read this; the one home for the store/response
/// extension→MIME map the collection write paths key against (#711).
///
/// DELIBERATELY DISTINCT from `shrike_engine_api::mime_for_name` (#711): that
/// one is the engine routing-HINT (carries `heic`/`aiff`, omits `pdf`/`txt`/
/// `css`/`js`); these are the store/response MIME the fetch/list/write paths
/// serve. They are NOT to be merged: folding `mime_for_name` in here would
/// force shrike-engine-api (a LEAF, the kernel↔ort firewall) to depend on
/// shrike-media, pulling a media-fetch/SSRF edge into the engine contract.
/// (Chesterton's fence; the lead's #711 ruling — leaf purity > table-count==1.)
pub fn guess_mime(filename: &str) -> Option<&'static str> {
    let ext = filename.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "avif" => "image/avif",
        "ico" => "image/vnd.microsoft.icon",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "wav" => "audio/x-wav",
        "flac" => "audio/x-flac",
        "m4a" => "audio/mp4",
        "opus" => "audio/opus",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        "mov" => "video/quicktime",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" => "text/javascript",
        "json" => "application/json",
        _ => return None,
    })
}

/// pylib `media.add_extension_based_on_mime`'s type map: the extension a stored
/// file should carry given an HTTP `content_type` whose name lacks one.
pub fn mime_extension(content_type: &str) -> Option<&'static str> {
    Some(match content_type {
        "audio/mpeg" => ".mp3",
        "audio/ogg" => ".oga",
        "audio/opus" => ".opus",
        "audio/wav" => ".wav",
        "audio/webm" => ".weba",
        "audio/aac" => ".aac",
        "image/jpeg" => ".jpg",
        "image/png" => ".png",
        "image/svg+xml" => ".svg",
        "image/webp" => ".webp",
        "image/avif" => ".avif",
        _ => return None,
    })
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

    /// The #591 SSRF parity regression, pinned at the media boundary too: the
    /// allowlist must refuse 6to4 (2002::/16) and 3fff::/20 exactly like
    /// Python's `ipaddress.is_global`. A 6to4 address embeds an IPv4 in bytes
    /// 2..6, so `2002:7f00:0001::` is the 6to4 encoding of 127.0.0.1 —
    /// fail-open to an internal IPv4 on the attacker-supplied `store_media`
    /// url path if permitted.
    #[test]
    fn ssrf_classifier_refuses_6to4_and_3fff_like_python() {
        for bad in [
            "2002::1",        // 6to4 base
            "2002:7f00:1::1", // 6to4 of 127.0.0.1 (internal!)
            "2002:a00:1::1",  // 6to4 of 10.0.0.1 (internal!)
            "2002:c0a8:1::1", // 6to4 of 192.168.0.1 (internal!)
            "3fff::1",        // 3fff::/20 reserved-by-IANA
        ] {
            let ip: IpAddr = bad.parse().unwrap();
            assert!(
                !ip_is_allowed(ip),
                "{bad} is NON-global per Python ipaddress.is_global but the SSRF \
                 allowlist permitted it (parity divergence / SSRF reach)"
            );
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

    #[test]
    fn safe_media_name_guards_traversal() {
        assert_eq!(safe_media_name("../../etc/passwd"), "passwd");
        assert_eq!(safe_media_name("a\\b\\c.png"), "c.png");
        assert_eq!(safe_media_name(".."), "");
        assert_eq!(safe_media_name("dir/"), "dir");
        assert_eq!(safe_media_name("plain.png"), "plain.png");
        // Whitespace-only (or whitespace-wrapped dots) is no name at all (#382).
        assert_eq!(safe_media_name("   "), "");
        assert_eq!(safe_media_name(" .. "), "");
        assert_eq!(safe_media_name("a/  "), "");
    }

    #[test]
    fn mime_tables_cover_the_stored_kinds() {
        assert_eq!(guess_mime("a.png"), Some("image/png"));
        assert_eq!(guess_mime("clip.m4a"), Some("audio/mp4"));
        assert_eq!(guess_mime("doc.pdf"), Some("application/pdf"));
        assert_eq!(guess_mime("noext"), None);
        assert_eq!(mime_extension("image/png"), Some(".png"));
        assert_eq!(mime_extension("audio/ogg"), Some(".oga"));
        assert_eq!(mime_extension("application/unknown"), None);
    }
}
