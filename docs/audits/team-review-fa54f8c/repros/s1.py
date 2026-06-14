"""Scratch repros for S1 (server/transport/trust-boundary) audit findings.
Preserved by lead from rev-S1 report (worktree reaped). Observed at fa54f8c: 5 passed.
These are CHARACTERIZING (assert current behavior). For the xfail handoff they will be
rewritten to assert the CORRECT behavior (so they fail today, XPASS when fixed).

Run with:
  SHRIKE_SKIP_NATIVE_STALE_CHECK=1 .venv/bin/python -m pytest \
    tests/unit/test_scratch_s1.py -q -p no:cacheprovider
"""

from __future__ import annotations

import pytest
from mcp.server.transport_security import TransportSecurityMiddleware

from shrike.pathsafety import is_loopback
from shrike.server import _build_transport_security


# FINDING S1-1: a non-127.0.0.1 loopback bind self-bricks (every request 421).
class TestLoopbackBindNot127001:
    def test_main_accepts_127_0_0_2_as_loopback(self) -> None:
        assert is_loopback("127.0.0.2") is True

    def test_guard_allowlist_omits_actual_bind_host(self) -> None:
        settings = _build_transport_security("127.0.0.2")
        assert settings is not None
        assert "127.0.0.2:*" not in settings.allowed_hosts

    def test_legitimate_client_to_127_0_0_2_is_rejected(self) -> None:
        settings = _build_transport_security("127.0.0.2")
        mw = TransportSecurityMiddleware(settings)
        assert mw._validate_host("127.0.0.2:8372") is False
        assert mw._validate_host("127.0.0.1:8372") is True


# FINDING S1-2: --no-dns-rebinding-protection does NOT disable the guard on /mcp.
class TestNoDnsRebindingMcpDivergence:
    def test_custom_route_guard_is_off(self) -> None:
        settings = _build_transport_security("127.0.0.1", disable=True)
        assert settings is None
        mw = TransportSecurityMiddleware(settings)
        assert mw.settings.enable_dns_rebinding_protection is False

    def test_mcp_endpoint_guard_is_silently_reenabled(self) -> None:
        from mcp.server.fastmcp import FastMCP

        app = FastMCP(
            "scratch",
            stateless_http=True,
            json_response=True,
            host="127.0.0.1",
            port=8372,
            transport_security=None,
        )
        ts = app.settings.transport_security
        assert ts is not None
        assert ts.enable_dns_rebinding_protection is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
