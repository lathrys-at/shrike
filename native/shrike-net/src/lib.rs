//! `shrike-net` — the one home for Shrike's SSRF safety primitives (#592).
//!
//! **Trust-boundary code**: changes here go through the security-review gate.
//!
//! This crate was extracted from `shrike-kernel`'s `media_fetch` so the SSRF
//! control is shared, not copied. Three consumers depend on it:
//!
//! - `shrike-kernel` (`media_fetch`): the attacker-supplied `store_media` url
//!   path — resolves+vets EVERY hop's host against `ipaddress.is_global`
//!   parity, pins the connection to the vetted IP, follows redirects manually
//!   re-vetting each hop.
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
//!   6to4 (2002::/16) and 3fff::/20 refusals (#591);
//! - [`resolve_public_ip`]: resolve a host and refuse it unless EVERY resolved
//!   address passes the allowlist, returning the first for pinning;
//! - [`pinned_agent`]: a ureq agent whose resolver always hands back one fixed
//!   `SocketAddr` (the vetted IP), with auto-redirects OFF (`redirects(0)`) so
//!   the caller follows + re-vets each hop itself.
//!
//! Pure Rust, NO async runtime (`ureq` is synchronous), NO engine crate and
//! NOT `shrike-kernel` — it sits BELOW both, so the kernel-purity and
//! engine-purity layering rules both stay satisfied.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::time::Duration;

use shrike_ffi::{NativeError, NativeResult};

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
pub fn resolve_pinned(host: &str) -> NativeResult<IpAddr> {
    (host, 0u16)
        .to_socket_addrs()
        .map_err(|e| invalid(format!("could not resolve host '{host}': {e}")))?
        .map(|sa| sa.ip())
        .next()
        .ok_or_else(|| invalid(format!("could not resolve host '{host}'")))
}

/// A ureq agent that connects ONLY to `pinned` (the vetted IP) on `port`,
/// ignoring the netloc ureq would otherwise re-resolve, with auto-redirects
/// OFF so the caller follows + re-vets each hop manually. The URL keeps the
/// hostname, so TLS SNI + certificate validation verify against the name (the
/// Host header is right by construction) while the socket goes where we
/// checked — closing the DNS-rebinding TOCTOU.
pub fn pinned_agent(pinned: IpAddr, port: u16, timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(timeout)
        .redirects(0)
        .resolver(move |_netloc: &str| -> std::io::Result<Vec<SocketAddr>> {
            Ok(vec![SocketAddr::new(pinned, port)])
        })
        .build()
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

    /// The #591 SSRF parity regression, moved with the classifier: the
    /// allowlist must refuse 6to4 (2002::/16) and 3fff::/20 exactly like
    /// Python's `ipaddress.is_global`. A 6to4 address embeds an IPv4 in bytes
    /// 2..6, so `2002:7f00:0001::` is the 6to4 encoding of 127.0.0.1 —
    /// fail-open to an internal IPv4 if permitted.
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
}
