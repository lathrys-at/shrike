"""Integration tests for transport-security hardening (audit 1.1 / 1.2).

Covers the Host/Origin guard on the custom HTTP routes (which bypass the MCP
transport middleware) and the refuse-to-start guard for non-loopback binds.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from .conftest import ServerInfo

pytestmark = pytest.mark.integration


def _base_url(server: ServerInfo) -> str:
    return server.url.rsplit("/", 1)[0]


# Every custom HTTP route the server registers (each wrapped by `_guard`). A
# forged Host/Origin is rejected *before* the handler runs, so probing even the
# destructive POSTs is non-destructive (the server survives — asserted below).
_CUSTOM_ROUTES = [
    ("GET", "/status"),
    ("GET", "/media/probe.png"),  # added in #70
    ("POST", "/shutdown"),
    ("POST", "/index/rebuild"),
    ("POST", "/index/save"),
    ("POST", "/embedding/start"),
    ("POST", "/embedding/stop"),
    ("POST", "/reload"),
]


class TestEveryCustomRouteGuard:
    """`_guard` is applied per-route; assert it on *every* one, not just 2."""

    @pytest.mark.parametrize(("method", "path"), _CUSTOM_ROUTES)
    def test_route_rejects_cross_origin(self, server: ServerInfo, method: str, path: str) -> None:
        resp = httpx.request(
            method,
            f"{_base_url(server)}{path}",
            headers={"Origin": "http://evil.example.com"},
            timeout=5.0,
        )
        assert resp.status_code == 403
        # The guard ran before the handler — the server is still alive.
        assert httpx.get(f"{_base_url(server)}/status", timeout=5.0).status_code == 200

    @pytest.mark.parametrize(("method", "path"), _CUSTOM_ROUTES)
    def test_route_rejects_forged_host(self, server: ServerInfo, method: str, path: str) -> None:
        resp = httpx.request(
            method,
            f"{_base_url(server)}{path}",
            headers={"Host": "evil.example.com"},
            timeout=5.0,
        )
        assert resp.status_code == 421
        assert httpx.get(f"{_base_url(server)}/status", timeout=5.0).status_code == 200


class TestMcpEndpointGuard:
    """/mcp itself enforces the guard (pins create_mcp's transport_security=)."""

    def test_mcp_rejects_cross_origin(self, server: ServerInfo) -> None:
        resp = httpx.post(
            server.url,
            headers={
                "Origin": "http://evil.example.com",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=5.0,
        )
        assert resp.status_code == 403

    def test_mcp_rejects_forged_host(self, server: ServerInfo) -> None:
        resp = httpx.post(
            server.url,
            headers={
                "Host": "evil.example.com",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=5.0,
        )
        assert resp.status_code == 421


class TestOriginAndMethodEdges:
    def test_origin_null_rejected(self, server: ServerInfo) -> None:
        # `Origin: null` is what a sandboxed iframe / file:// page sends — must not
        # be treated as same-origin.
        resp = httpx.get(f"{_base_url(server)}/status", headers={"Origin": "null"}, timeout=5.0)
        assert resp.status_code == 403

    def test_no_origin_loopback_host_allowed(self, server: ServerInfo) -> None:
        # The native-client path (mcp-remote, CLI): no Origin header + a loopback
        # Host is allowed. Pinned explicitly so a future tightening can't silently
        # break native clients.
        resp = httpx.get(f"{_base_url(server)}/status", timeout=5.0)
        assert resp.status_code == 200

    @pytest.mark.parametrize("path", ["/shutdown", "/index/rebuild", "/reload"])
    def test_get_on_post_only_route_is_405(self, server: ServerInfo, path: str) -> None:
        # Destructive routes are POST-only, so `<img src=.../shutdown>` can't fire
        # them (a GET is 405, before any guard/handler).
        resp = httpx.get(f"{_base_url(server)}{path}", timeout=5.0)
        assert resp.status_code == 405
        assert httpx.get(f"{_base_url(server)}/status", timeout=5.0).status_code == 200


class TestEscapeHatchesFlipBehavior:
    """The disable flag must actually disable the guard (not just be wired)."""

    def test_no_dns_rebinding_protection_accepts_forged_headers(self, server_factory) -> None:
        srv = server_factory("nodns", extra_args=["--no-dns-rebinding-protection"])
        base = _base_url(srv)
        # A forged Origin/Host that would 403/421 by default is now accepted.
        assert (
            httpx.get(
                f"{base}/status", headers={"Origin": "http://evil.example.com"}, timeout=5.0
            ).status_code
            == 200
        )
        assert (
            httpx.get(
                f"{base}/status", headers={"Host": "evil.example.com"}, timeout=5.0
            ).status_code
            == 200
        )


class TestNonLoopbackGuard:
    """Binding to a non-loopback host requires an explicit opt-in."""

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.0.1"])
    def test_refuses_non_loopback_without_allow_remote(self, tmp_path: Path, host: str) -> None:
        log_dir = tmp_path / "logs"
        state_dir = tmp_path / "state"
        cache_dir = tmp_path / "cache"
        for d in (log_dir, state_dir, cache_dir):
            d.mkdir()

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "shrike.server",
                "--collection",
                str(tmp_path / "collection.anki2"),
                "--host",
                host,
                "--foreground",  # routes the refusal log to the console handler
                "--log-dir",
                str(log_dir),
                "--state-dir",
                str(state_dir),
                "--cache-dir",
                str(cache_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 1
        assert "Refusing to bind to non-loopback host" in (proc.stdout + proc.stderr)
