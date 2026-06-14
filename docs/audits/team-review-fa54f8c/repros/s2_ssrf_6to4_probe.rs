//! S2 review scratch repro (preserved by lead; rev-S2 worktree reaped).
//! The Rust SSRF classifier diverges from Python's `ipaddress.is_global` on
//! 6to4 (2002::/16) and 3fff::/20 — Python refuses (non-global); the Rust
//! allowlist permits. A 6to4 address embeds an IPv4 in bytes 2..6, so
//! `2002:7f00:0001::` is the 6to4 encoding of 127.0.0.1.
//! Place at native/shrike-kernel/tests/s2_ssrf_6to4_probe.rs.
//! Run: CARGO_TARGET_DIR=$HOME/.cache/shrike-review-target/s2 \
//!   cargo test -p shrike-kernel --test s2_ssrf_6to4_probe -- --nocapture
//! Observed at fa54f8c: RED (allowlist permits 2002::1 etc.).

use std::net::IpAddr;

use shrike_kernel::media_fetch::ip_is_allowed;

#[test]
fn ssrf_classifier_refuses_6to4_and_3fff_like_python() {
    let must_refuse = [
        "2002::1",         // 6to4 base
        "2002:7f00:1::1",  // 6to4 of 127.0.0.1 (internal!)
        "2002:a00:1::1",   // 6to4 of 10.0.0.1 (internal!)
        "2002:c0a8:1::1",  // 6to4 of 192.168.0.1 (internal!)
        "3fff::1",         // 3fff::/20 reserved-by-IANA
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
// Fix direction: add 2002::/16 and 3fff::/20 to ipv6_is_global's private set in
// media_fetch.rs; add boundary addrs for both to IP_CORPUS in
// tests/native/test_media_url_fetch.py so the parity test guards them.
