"""Unit tests for the server's transport-security helpers (audit 1.1 / 1.2)."""

from __future__ import annotations

import ipaddress
import logging

import pytest

from shrike.server import _build_transport_security, _is_loopback


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"],
)
def test_loopback_hosts_are_recognized(host: str) -> None:
    assert _is_loopback(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.10", "10.0.0.5", "example.com", "::"],
)
def test_non_loopback_hosts_are_rejected(host: str) -> None:
    assert _is_loopback(host) is False


def test_transport_security_enabled_for_loopback() -> None:
    settings = _build_transport_security("127.0.0.1")
    assert settings is not None
    assert settings.enable_dns_rebinding_protection is True
    # Loopback hosts/origins only; the port is wildcarded.
    assert settings.allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert settings.allowed_origins == [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    ]


def test_transport_security_none_for_non_loopback() -> None:
    # A deliberately remote bind with no explicit allow-list has no fixed Host
    # set to validate against; protection is left to the network boundary /
    # (roadmap) auth layer. This preserves the historical --allow-remote default.
    assert _build_transport_security("0.0.0.0") is None
    assert _build_transport_security("192.168.1.10") is None


def test_extra_allowed_host_extends_loopback_list() -> None:
    # The reverse-proxy / VPN case: loopback bind, guard on, but additionally
    # trust the hostname the proxy forwards (e.g. a Tailscale name).
    settings = _build_transport_security(
        "127.0.0.1",
        allowed_hosts=["host.tailnet.ts.net"],
        allowed_origins=["https://host.tailnet.ts.net"],
    )
    assert settings is not None
    assert settings.allowed_hosts == [
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        "host.tailnet.ts.net",
    ]
    assert "https://host.tailnet.ts.net" in settings.allowed_origins


def test_no_dns_rebinding_protection_disables_guard_on_any_bind() -> None:
    # The network-is-the-boundary mode: guard off regardless of bind address,
    # even on loopback (Shrike behind a local reverse proxy / tailscale serve).
    assert _build_transport_security("127.0.0.1", disable=True) is None
    assert _build_transport_security("0.0.0.0", disable=True) is None


def test_explicit_allow_list_builds_guard_even_when_non_loopback() -> None:
    # A non-loopback bind WITH an explicit allow-list keeps the guard, trusting
    # only the named hosts (no loopback defaults, since the bind isn't loopback).
    settings = _build_transport_security("0.0.0.0", allowed_hosts=["proxy.internal"])
    assert settings is not None
    assert settings.allowed_hosts == ["proxy.internal"]
    assert settings.allowed_origins == []


def test_disable_takes_precedence_over_allow_list() -> None:
    # disable=True wins even when explicit hosts are given — the operator opted
    # out of the guard entirely.
    assert (
        _build_transport_security("127.0.0.1", allowed_hosts=["proxy.internal"], disable=True)
        is None
    )


def test_origins_only_non_loopback_is_fail_closed(caplog: pytest.LogCaptureFixture) -> None:
    # The footgun: a non-loopback bind given allowed_origins but NO allowed_hosts
    # builds a guard whose Host allow-list is empty — so every request's Host is
    # rejected (421) and the server answers nothing. Fail-*closed* (not a hole),
    # but a config trap, so it must warn at build time.
    with caplog.at_level(logging.WARNING, logger="shrike.server"):
        settings = _build_transport_security("0.0.0.0", allowed_origins=["https://app.example"])
    assert settings is not None
    assert settings.allowed_hosts == []  # nothing matches → 421 for everything
    assert settings.allowed_origins == ["https://app.example"]
    assert any("no allowed Host values" in r.message for r in caplog.records)


def test_loopback_guard_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    # The normal loopback path has a populated Host list — no footgun, no warning.
    with caplog.at_level(logging.WARNING, logger="shrike.server"):
        _build_transport_security("127.0.0.1")
    assert not any("no allowed Host values" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0:0:0:0:0:0:0:1", True),  # expanded ::1 — still loopback
        ("LOCALHOST", False),  # exact "localhost" match only; uppercase is NOT loopback
    ],
)
def test_is_loopback_stable_edges(host: str, expected: bool) -> None:
    # Pin the Shrike-controlled boundary so a refactor can't silently shift it.
    assert _is_loopback(host) is expected


def test_is_loopback_ipv4_mapped_delegates_to_stdlib() -> None:
    # ::ffff:127.0.0.1 — Shrike adds no special handling, it defers to
    # ipaddress.is_loopback, whose result for IPv4-mapped addresses is
    # *Python-version-dependent* (False <3.12.4, True after). Pin the delegation,
    # not a fixed bool, so the test is robust across the supported range.
    mapped = "::ffff:127.0.0.1"
    assert _is_loopback(mapped) is ipaddress.ip_address(mapped).is_loopback
