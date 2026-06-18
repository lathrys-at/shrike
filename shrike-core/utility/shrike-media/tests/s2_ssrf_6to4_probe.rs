//! SSRF classifier parity regression.
//! The Rust SSRF classifier diverged from Python's `ipaddress.is_global` on
//! 6to4 (2002::/16) and 3fff::/20 — Python refuses (non-global); the Rust
//! allowlist permitted. A 6to4 address embeds an IPv4 in bytes 2..6, so
//! `2002:7f00:0001::` is the 6to4 encoding of 127.0.0.1, fail-open to an
//! internal IPv4 on the attacker-supplied `store_media` url path.

use std::net::IpAddr;

use shrike_media::ip_is_allowed;

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
            "{s} is NON-global per Python ipaddress.is_global but the Rust SSRF \
             allowlist permitted it (parity divergence / SSRF reach)"
        );
    }
}
