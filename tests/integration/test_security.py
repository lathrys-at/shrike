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


class TestCustomRouteGuard:
    """The custom routes get the same Host/Origin validation as /mcp."""

    def test_status_allows_loopback_request(self, server: ServerInfo) -> None:
        resp = httpx.get(f"{_base_url(server)}/status", timeout=5.0)
        assert resp.status_code == 200

    def test_status_rejects_cross_origin(self, server: ServerInfo) -> None:
        resp = httpx.get(
            f"{_base_url(server)}/status",
            headers={"Origin": "http://evil.example.com"},
            timeout=5.0,
        )
        assert resp.status_code == 403

    def test_status_rejects_forged_host(self, server: ServerInfo) -> None:
        # DNS-rebinding: the connection still lands on 127.0.0.1, but a forged
        # Host header (as a rebound domain would send) must be rejected.
        resp = httpx.get(
            f"{_base_url(server)}/status",
            headers={"Host": "evil.example.com"},
            timeout=5.0,
        )
        assert resp.status_code == 421

    def test_shutdown_rejected_without_killing_server(self, server: ServerInfo) -> None:
        # A no-preflight cross-origin POST to /shutdown must be refused — and the
        # server must still be alive afterwards.
        resp = httpx.post(
            f"{_base_url(server)}/shutdown",
            headers={"Origin": "http://evil.example.com"},
            timeout=5.0,
        )
        assert resp.status_code == 403
        # Server still up.
        assert httpx.get(f"{_base_url(server)}/status", timeout=5.0).status_code == 200


class TestNonLoopbackGuard:
    """Binding to a non-loopback host requires an explicit opt-in."""

    def test_refuses_non_loopback_without_allow_remote(self, tmp_path: Path) -> None:
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
                "0.0.0.0",
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
