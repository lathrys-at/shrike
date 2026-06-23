//! `shrike-media` — the inbound/untrusted-media crate.
//!
//! The "acquire + validate untrusted bytes" half of media handling. The *store*
//! half (store/fetch/list/delete/prune) stays in `shrike-collection` — it is
//! inseparable from `col.media`. This crate owns:
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
//! It delegates the SSRF primitives to `shrike-network` (the one home for the
//! is_global classifier + the IP-pinned agent), so the control is shared, not
//! copied:
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
//! Pure Rust, NO anki coupling, NO engine crate and NOT `shrike-kernel`: it
//! sits BELOW both, so the kernel-purity and engine-purity layering rules both
//! stay satisfied (//shrike-core:layering_check). The URL fetch is **async**:
//! it rides `shrike_network::fetch_pinned_get` (the centralized
//! IP-pinned, per-hop-re-vetting reqwest loop) instead of a blocking `ureq`
//! call, so `prepare_media_item` is an `async fn` the host drives on its
//! runtime (the kernel `tokio::spawn`s each item concurrently; the standalone
//! binding drives it via `block_on`) — no thread parked on a network wait.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use base64::Engine;
use shrike_error::{NativeError, NativeResult};
use shrike_schemas::StoreMediaItem;
use url::Url;

// The SSRF classifier + the resolve-and-vet helper live in the shared
// `shrike-network` crate so the remote engine crates use the SAME control.
// Re-exported here so the in-tree callers (the kernel's Python-facing binding,
// any media caller) can import them from `shrike_media`.
pub use shrike_network::{ip_is_allowed, resolve_public_ip};

/// The byte-source size cap — caller-supplied/downloaded bytes only; a
/// server-local `path` inside an operator-configured root is deliberately
/// uncapped. The one policy value the collection write tail and the
/// fetch/decode caps must agree on, so it lives where both can see it.
pub const MEDIA_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Per-request timeout for an inbound media URL fetch.
pub const URL_FETCH_TIMEOUT_SECS: u64 = 30;
/// The redirect hop cap (= `shrike_network::MAX_REDIRECTS`, a named alias the
/// call sites and the error message that quotes it use).
pub const MAX_MEDIA_REDIRECTS: usize = shrike_network::MAX_REDIRECTS;

/// One store_media item after the kernel's off-actor prepare: byte
/// sources arrive fetched/decoded; `path` items pass through whole (their
/// gates are collection policy and run under the write); a failed prepare
/// carries its per-item error. The interface between the acquire half (this
/// crate) and the store half (`shrike-collection`, which re-exports this).
pub struct PreparedMedia {
    /// The item's index in the caller's batch (echoed in results).
    pub index: i64,
    /// The caller's `filename`, echoed on errors.
    pub filename: Option<String>,
    /// The prepared source (bytes, a server-local path, or a failure).
    pub source: PreparedMediaSource,
}

/// The outcome of preparing one `store_media` item off the collection actor.
pub enum PreparedMediaSource {
    /// Decoded base64 or a completed download; `name` already folds the
    /// URL-derived fallback.
    Bytes {
        /// Final filename (already folds the URL-derived fallback).
        name: String,
        /// The fetched/decoded payload.
        data: Vec<u8>,
        /// MIME hint from the source, if any.
        content_type: Option<String>,
    },
    /// A server-local path item, gated under the write.
    Path {
        /// The server-local path to store from.
        path: String,
    },
    /// The prepare failed (bad base64, refused/failed download, invalid
    /// item); stored nothing.
    Failed {
        /// The per-item failure message.
        error: String,
    },
}

fn invalid(msg: impl Into<String>) -> NativeError {
    NativeError::invalid_input(msg)
}

/// `_safe_media_name`: reduce a caller-supplied name to a bare basename so it
/// can only resolve inside the media dir (path-traversal guard for
/// fetch/delete). Returns "" for a name that is only separators/dots — or
/// only whitespace around them, which the emptiness check would otherwise
/// pass.
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
/// extension→MIME map the collection write paths key against.
///
/// DELIBERATELY DISTINCT from `shrike_engine_api::mime_for_name`: that
/// one is the engine routing-HINT (carries `heic`/`aiff`, omits `pdf`/`txt`/
/// `css`/`js`); these are the store/response MIME the fetch/list/write paths
/// serve. They are NOT to be merged: folding `mime_for_name` in here would
/// force shrike-engine-api (a LEAF, the kernel↔ort firewall) to depend on
/// shrike-media, pulling a media-fetch/SSRF edge into the engine contract.
/// (Chesterton's fence: leaf purity > table-count==1.)
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
///
/// # Errors
///
/// Returns an `InvalidInput` [`NativeError`] if the encoded length exceeds the
/// [`MEDIA_MAX_BYTES`] limit or `data` is not valid base64.
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
///
/// The SSRF posture (http/https only; per-hop is_global re-vet; IP-pin keeping
/// SNI/cert/`Host` on the name; manual redirect following capped at
/// [`MAX_MEDIA_REDIRECTS`]; the body size-capped while streaming) lives in the
/// shared [`shrike_network::fetch_pinned_get`] — the ONE audited copy
/// every consumer rides. With `allow_private` the gate and pin are off (the
/// operator opted into trusted internal hosts).
///
/// # Errors
///
/// Returns an `InvalidInput` [`NativeError`] if the URL is malformed, uses a
/// non-http(s) scheme, has no host, resolves to a non-public address (unless
/// `allow_private`), exceeds the redirect cap, the request fails, or the body
/// exceeds [`MEDIA_MAX_BYTES`]; or an `Unavailable` [`NativeError`] if the HTTP
/// client cannot be built.
pub async fn fetch_media_url(
    url: &str,
    allow_private: bool,
) -> NativeResult<(Vec<u8>, Option<String>)> {
    let timeout = std::time::Duration::from_secs(URL_FETCH_TIMEOUT_SECS);
    shrike_network::fetch_pinned_get(url, allow_private, timeout, MEDIA_MAX_BYTES).await
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
/// run under the write). Decodes base64 inline; for a driver that wants the
/// CPU decode off its IO thread, [`prepare_media_item_with_decode`] takes an
/// injected decoder. The url path awaits [`fetch_media_url`].
pub async fn prepare_media_item(
    index: i64,
    item: StoreMediaItem,
    allow_private_fetch: bool,
) -> PreparedMedia {
    prepare_media_item_with_decode(index, item, allow_private_fetch, |data| async move {
        decode_media_b64(&data)
    })
    .await
}

/// [`prepare_media_item`] with the base64 decode injected as `decode`, so a
/// driver can move the CPU decode off the thread driving the async fetch.
/// The kernel routes `decode` through its compute pool while the url fetch rides
/// the IO driver; the standalone driver supplies an inline decoder (what
/// [`prepare_media_item`] does). `decode` runs only for a `data` item — `path`
/// and `url` items never call it.
pub async fn prepare_media_item_with_decode<F, Fut>(
    index: i64,
    item: StoreMediaItem,
    allow_private_fetch: bool,
    decode: F,
) -> PreparedMedia
where
    F: FnOnce(String) -> Fut,
    Fut: std::future::Future<Output = NativeResult<Vec<u8>>>,
{
    let filename = item.filename.clone();
    let source = if let Err(e) = item.validate() {
        PreparedMediaSource::Failed { error: e }
    } else if let Some(path) = item.path {
        PreparedMediaSource::Path { path }
    } else if let Some(data) = item.data {
        match decode(data).await {
            Ok(bytes) => PreparedMediaSource::Bytes {
                name: item.filename.unwrap_or_default(),
                data: bytes,
                content_type: None,
            },
            Err(e) => PreparedMediaSource::Failed { error: e.message },
        }
    } else if let Some(url) = item.url.as_deref() {
        match fetch_media_url(url, allow_private_fetch).await {
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
        // validate() guarantees one source; this is a backstop.
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
    use proptest::prelude::*;

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

    /// The SSRF parity regression, pinned at the media boundary too: the
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

    #[tokio::test]
    async fn scheme_and_host_validation() {
        assert!(fetch_media_url("ftp://example.com/x", false).await.is_err());
        assert!(fetch_media_url("not a url", false).await.is_err());
        // loopback never leaves the building, even before any socket opens
        let err = fetch_media_url("http://127.0.0.1:1/x", false)
            .await
            .unwrap_err();
        assert!(err.message.contains("non-public address"));
    }

    #[test]
    fn safe_media_name_guards_traversal() {
        assert_eq!(safe_media_name("../../etc/passwd"), "passwd");
        assert_eq!(safe_media_name("a\\b\\c.png"), "c.png");
        assert_eq!(safe_media_name(".."), "");
        assert_eq!(safe_media_name("dir/"), "dir");
        assert_eq!(safe_media_name("plain.png"), "plain.png");
        // Whitespace-only (or whitespace-wrapped dots) is no name at all.
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

    // ---- Adversarial sweeps (epic #740 / #742) -------------------------
    //
    // Every input below is *inbound and untrusted*: a `store_media` filename,
    // URL path, or base64 blob a caller hands the server. The tests pin the
    // EXACT behavior of the parsing/validation surface so a regression that
    // widens what we accept (or panics on a hostile string) is caught here.

    /// MIME guessing is case-insensitive on the extension: an attacker can't
    /// dodge or force a content type by varying case. `.PNG`/`.Png`/`.png`
    /// all map to `image/png`.
    #[test]
    fn guess_mime_is_case_insensitive_on_the_extension() {
        for name in ["a.PNG", "a.Png", "a.png", "a.pNg", "CLIP.M4A", "Doc.PDF"] {
            assert!(
                guess_mime(name).is_some(),
                "{name}: extension case must not change the MIME result"
            );
        }
        assert_eq!(guess_mime("a.PNG"), guess_mime("a.png"));
        assert_eq!(guess_mime("CLIP.M4A"), Some("audio/mp4"));
        assert_eq!(guess_mime("Doc.PDF"), Some("application/pdf"));
    }

    /// Only the LAST dot-segment is the extension, so a double-extension
    /// disguise (`image.png.exe`) is classified by its true trailing token —
    /// never the misleading inner one. Pins that we don't mistake `image.png.exe`
    /// for a PNG, and that an unknown trailing token yields the documented
    /// `None` fallback (NOT octet-stream — there is no permissive default).
    #[test]
    fn guess_mime_uses_only_the_last_extension() {
        assert_eq!(guess_mime("image.PNG.exe"), None);
        assert_eq!(guess_mime("archive.tar.gz"), None);
        // The real trailing extension still resolves through inner dots.
        assert_eq!(guess_mime("my.photo.backup.jpeg"), Some("image/jpeg"));
        // A known trailing token after a dotted prefix is honored.
        assert_eq!(guess_mime("a.b.c.png"), Some("image/png"));
    }

    /// Degenerate filename shapes the parser must classify as "no usable
    /// extension" -> `None`, never panic, never a default MIME. Empty string,
    /// trailing dot, and a bare unknown word all collapse to `None`.
    #[test]
    fn guess_mime_degenerate_names_yield_none_not_a_default() {
        for name in ["", "noext", "file.", "...", "a.unknownext", "a."] {
            assert_eq!(
                guess_mime(name),
                None,
                "{name:?}: a name with no known trailing extension must be None"
            );
        }
    }

    /// A leading-dot "hidden file" with no stem (`.png`) is treated as having
    /// extension `png` by `rsplit('.')`: the dotfile name IS the extension.
    /// Pinned so the behavior is a deliberate, documented choice rather than an
    /// accident — an attacker naming a payload `.png` gets `image/png`.
    #[test]
    fn guess_mime_leading_dot_hidden_file_is_treated_as_extension() {
        assert_eq!(guess_mime(".png"), Some("image/png"));
        assert_eq!(guess_mime(".jpeg"), Some("image/jpeg"));
        assert_eq!(guess_mime(".unknown"), None);
    }

    /// Unicode in the stem and path separators in the name don't break MIME
    /// guessing: only the trailing ASCII extension matters, and a non-ASCII
    /// extension token is simply unknown (-> None), never a panic.
    #[test]
    fn guess_mime_handles_unicode_and_separators_without_panic() {
        assert_eq!(guess_mime("\u{4f60}\u{597d}.png"), Some("image/png"));
        assert_eq!(guess_mime("dir/sub/clip.mp3"), Some("audio/mpeg"));
        // A non-ASCII "extension" lowercases to itself and is unknown.
        assert_eq!(guess_mime("file.\u{0444}"), None);
        // A pathologically long extension is just an unknown token.
        let long = format!("f.{}", "z".repeat(4096));
        assert_eq!(guess_mime(&long), None);
    }

    /// A char palette mixing letters, dots, separators, whitespace, NUL, a
    /// combining mark, and an astral code point — the junk an untrusted filename
    /// might carry. `png` chars are included so the sweep also lands on
    /// near-real extensions.
    fn fuzzed_filename() -> impl Strategy<Value = String> {
        prop::collection::vec(
            prop::sample::select(vec![
                'a',
                'Z',
                '.',
                '/',
                '\\',
                ' ',
                '\0',
                '\u{0301}',
                '\u{1F4A9}',
                'p',
                'n',
                'g',
            ]),
            0..24,
        )
        .prop_map(|cs| cs.into_iter().collect::<String>())
    }

    proptest! {
        /// Fuzz: `guess_mime` is total over arbitrary byte-ish strings — it
        /// returns Some/None but never panics or hangs, whatever junk an
        /// untrusted filename carries (control chars, dots, slashes, high code
        /// points).
        #[test]
        fn guess_mime_is_panic_free_over_fuzzed_names(s in fuzzed_filename()) {
            // Total function: any Some(&'static str) value is fine; the property
            // is "no panic", asserted by returning normally.
            let _ = guess_mime(&s);
        }
    }

    /// `guess_mime` is deterministic: same input, same output across calls.
    #[test]
    fn guess_mime_is_deterministic() {
        for name in ["a.png", "x.unknown", ".gif", "image.png.exe", ""] {
            assert_eq!(guess_mime(name), guess_mime(name));
        }
    }

    /// Valid base64 round-trips, including the empty payload and single-byte
    /// (double-padding) shapes, for the trusted-shape baseline the adversarial
    /// cases contrast against.
    #[test]
    fn b64_valid_inputs_round_trip() {
        assert_eq!(decode_media_b64("aGk=").unwrap(), b"hi");
        assert_eq!(decode_media_b64("QQ==").unwrap(), b"A");
        assert_eq!(decode_media_b64("").unwrap(), Vec::<u8>::new());
        // encode -> decode identity over a non-trivial payload.
        let payload: Vec<u8> = (0u8..=255).cycle().take(777).collect();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);
        assert_eq!(decode_media_b64(&encoded).unwrap(), payload);
    }

    /// Hostile base64: the STANDARD engine demands canonical padding, so a blob
    /// with missing, extra, or interior padding is REJECTED — not silently
    /// truncated to a "best effort" decode. Pins each as Err so a future engine
    /// swap to a lenient mode is caught (a lenient decoder is an integrity hole
    /// for untrusted media bytes).
    #[test]
    fn b64_non_canonical_padding_is_rejected() {
        for bad in [
            "aGk",       // missing padding
            "aGk==",     // over-padded
            "aGk=extra", // trailing junk after pad
            "QQ",        // missing padding (1-byte payload)
            "QQ=",       // wrong pad count
            "Zg==Zg==",  // two padded quanta concatenated
        ] {
            assert!(
                decode_media_b64(bad).is_err(),
                "{bad:?}: non-canonical/garbage base64 must be rejected"
            );
        }
    }

    /// Invalid-alphabet symbols are rejected: anything outside the STANDARD
    /// alphabet (`+`/`/` are the only specials) fails rather than being skipped.
    #[test]
    fn b64_invalid_alphabet_is_rejected() {
        for bad in [
            "not base64!!",
            "****",
            "@@@@",
            "aGk-",
            "aGk_",
            "\u{00e9}\u{00e9}",
        ] {
            assert!(
                decode_media_b64(bad).is_err(),
                "{bad:?}: non-alphabet symbols must be rejected"
            );
        }
    }

    /// The de-chunker strips ALL whitespace before decoding, so PEM-style line
    /// wrapping and data:-URI newlines decode to the same bytes as the joined
    /// form — even whitespace inside a quantum. This is the property data-URI /
    /// chunked uploads rely on.
    #[test]
    fn b64_strips_embedded_whitespace_pem_and_datauri_style() {
        let joined = "SGVsbG8sIHdvcmxkIQ=="; // "Hello, world!"
        let expected = b"Hello, world!".to_vec();
        let wrapped = "SGVsbG8s\nIHdvcm\r\nxkIQ==";
        let spaced = "SGVs bG8s IHdv cmxk IQ==";
        let tabbed = "\tSGVsbG8sIHdvcmxkIQ==\n";
        assert_eq!(decode_media_b64(joined).unwrap(), expected);
        assert_eq!(decode_media_b64(wrapped).unwrap(), expected);
        assert_eq!(decode_media_b64(spaced).unwrap(), expected);
        assert_eq!(decode_media_b64(tabbed).unwrap(), expected);
        // Pure whitespace is an empty payload, not an error.
        assert_eq!(decode_media_b64(" \n\t ").unwrap(), Vec::<u8>::new());
    }

    /// The cap is on the ENCODED length and is checked BEFORE whitespace is
    /// stripped (so the raw `data.len()` bounds it). A string longer than the
    /// threshold is rejected with the limit message without ever allocating a
    /// decode buffer. Pins the exact threshold `MEDIA_MAX_BYTES/3*4+4` boundary.
    #[test]
    fn b64_encoded_length_cap_boundary_is_exact() {
        let threshold = MEDIA_MAX_BYTES / 3 * 4 + 4;
        // At the threshold: not rejected by the cap (decode may still fail on
        // content, but it must not be the size error).
        let at = "A".repeat(threshold);
        match decode_media_b64(&at) {
            Ok(_) => {}
            Err(e) => assert!(
                !e.message.contains("limit"),
                "len==threshold must not trip the size cap, got: {}",
                e.message
            ),
        }
        // One byte over: rejected, and specifically by the size cap.
        let over = "A".repeat(threshold + 1);
        let err = decode_media_b64(&over).unwrap_err();
        assert!(
            err.message.contains("limit"),
            "len>threshold must trip the size cap, got: {}",
            err.message
        );
        // A huge whitespace-padded string is rejected on raw length BEFORE the
        // strip — i.e. whitespace can't be used to smuggle past the cap.
        let smuggle = " ".repeat(threshold + 1);
        let err = decode_media_b64(&smuggle).unwrap_err();
        assert!(
            err.message.contains("limit"),
            "raw length cap must apply before whitespace stripping"
        );
    }

    /// MEDIA_MAX_BYTES is a sane, finite policy value (a real megabyte-scale
    /// cap, not 0 and not effectively unbounded). Guards against an accidental
    /// edit that disables the cap.
    #[test]
    fn media_max_bytes_is_a_sane_cap() {
        assert_eq!(MEDIA_MAX_BYTES, 64 * 1024 * 1024);
        // Compile-time guards: a cap of 0 or an effectively-unbounded one would
        // disable the policy. Const-block so the check rides the build, not just
        // this test run.
        const {
            assert!(MEDIA_MAX_BYTES >= 1024 * 1024, "cap must be at least 1 MiB");
        }
        const {
            assert!(
                MEDIA_MAX_BYTES <= 1024 * 1024 * 1024,
                "cap must stay bounded well under 1 GiB"
            );
        }
    }

    /// A char palette mixing the base64 alphabet, padding, whitespace, and junk
    /// so the sweep hits canonical, near-miss, and garbage shapes.
    fn fuzzed_b64() -> impl Strategy<Value = String> {
        let palette: Vec<char> = "ABCXYZabcxyz0189+/=\n\r\t !@#*\u{00e9}".chars().collect();
        prop::collection::vec(prop::sample::select(palette), 0..40)
            .prop_map(|cs| cs.into_iter().collect::<String>())
    }

    proptest! {
        /// Fuzz: `decode_media_b64` is total over arbitrary strings — every
        /// input yields Ok or Err, never a panic, however hostile. This is the
        /// core promise for an untrusted base64 source.
        #[test]
        fn b64_decode_is_panic_free_over_fuzzed_strings(s in fuzzed_b64()) {
            // The property is panic-freedom; Ok|Err are both acceptable.
            let _ = decode_media_b64(&s);
        }
    }

    /// `safe_media_name` and `media_name_from_url` agree on the traversal guard:
    /// a URL whose path is a traversal sequence yields no derived name (None),
    /// mirroring `safe_media_name` collapsing the same to "". Untrusted URL
    /// paths can't smuggle `..`/empty basenames into the media dir.
    #[test]
    fn media_name_from_url_refuses_traversal_and_empty_basenames() {
        assert_eq!(
            media_name_from_url("http://h/../../etc/passwd").as_deref(),
            Some("passwd")
        );
        assert_eq!(media_name_from_url("http://h/").as_deref(), None);
        assert_eq!(media_name_from_url("http://h/a/b/"), Some("b".to_string()));
        assert_eq!(media_name_from_url("http://h/dir/.."), None);
        // A non-URL yields None rather than panicking.
        assert_eq!(media_name_from_url("not a url"), None);
        assert_eq!(media_name_from_url(""), None);
    }

    /// A char palette mixing letters, both separators, dots, whitespace, and a
    /// CJK code point — the hostile shapes a `store_media` name might carry.
    fn fuzzed_name() -> impl Strategy<Value = String> {
        let palette: Vec<char> = "ab/\\.. \tCD\u{4f60}".chars().collect();
        prop::collection::vec(prop::sample::select(palette), 0..20)
            .prop_map(|cs| cs.into_iter().collect::<String>())
    }

    proptest! {
        /// Fuzz: the two untrusted-name reducers never panic and never emit a
        /// name containing a path separator (the traversal invariant) over
        /// hostile input.
        #[test]
        fn name_reducers_never_emit_separators_over_fuzzed_input(s in fuzzed_name()) {
            let safe = safe_media_name(&s);
            prop_assert!(
                !safe.contains('/') && !safe.contains('\\'),
                "safe_media_name({:?}) leaked a separator: {:?}",
                s,
                safe
            );
            // media_name_from_url only parses absolute URLs; just assert no panic
            // and, when it returns a name, no separator leaked.
            if let Some(n) = media_name_from_url(&format!("http://h/{s}")) {
                prop_assert!(
                    !n.contains('/') && !n.contains('\\'),
                    "media_name_from_url leaked a separator for {:?}: {:?}",
                    s,
                    n
                );
            }
        }
    }
}
