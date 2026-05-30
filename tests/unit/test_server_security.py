"""Unit tests for the server's transport-security helpers (audit 1.1 / 1.2)."""

from __future__ import annotations

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
    # A deliberately remote bind has no fixed Host allow-list; protection is
    # left to the (roadmap) auth layer rather than header validation.
    assert _build_transport_security("0.0.0.0") is None
    assert _build_transport_security("192.168.1.10") is None
