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
    ("GET", "/media/probe.png"),
    ("POST", "/shutdown"),
    ("POST", "/index/rebuild"),
    ("POST", "/index/save"),
    ("POST", "/embedding/start"),
    ("POST", "/embedding/stop"),
    ("POST", "/reload"),
    ("POST", "/actions/collection_info"),  # the actions-over-HTTP edge
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

    @pytest.mark.parametrize(
        "path", ["/shutdown", "/index/rebuild", "/reload", "/actions/collection_info"]
    )
    def test_get_on_post_only_route_is_405(self, server: ServerInfo, path: str) -> None:
        # POST-only routes can't be fired by a no-preflight GET (`<img src=...>`):
        # a GET is 405, before any guard/handler. The /actions/* family is POST-
        # only too, so a write action can never ride a cross-origin <img>/<form>.
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


class TestEmbeddingStartInputRobustness:
    """`/embedding/start` best-effort-parses an arbitrary JSON body into
    `runtime.start(**overrides)`. Within the trust model it must not 500 on
    garbage: the body parse is suppressed and only *known* keys are forwarded
    (an unknown key reaching start as **overrides would TypeError → 500). On the
    shared server (no model configured) every variant lands on the clean 400
    "no model" path. Placed here, not in the llama-gated test_embedding.py, so it
    runs in the normal suite — it needs no embedding service."""

    @pytest.mark.parametrize(
        "body",
        [
            "not json at all",  # malformed → parse suppressed → overrides={}
            "[]",  # valid JSON but not a dict → ignored
            '{"unknown_key": 1, "another": [1, 2, 3]}',  # unknown keys filtered out
            '{"port": "not-an-int"}',  # wrong-typed known key (never used: no model)
            '{"model": null}',  # explicit null is skipped
        ],
    )
    def test_garbage_body_yields_clean_400_not_500(self, server: ServerInfo, body: str) -> None:
        resp = httpx.post(
            f"{_base_url(server)}/embedding/start",
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400, resp.text  # no model configured, handled cleanly
        assert httpx.get(f"{_base_url(server)}/status", timeout=5.0).status_code == 200


class TestActionsErrorEnvelope:
    """The actions-over-HTTP edge returns ONE error envelope, and it must
    not leak server internals on any error code. The envelope is
    `{"code": <taxonomy>, "message": <non-leaking>}`; the security-critical case
    is the 500, whose message is FIXED — the real exception + traceback go only
    to the log (via `_safe_tool`'s `logger.exception`), never to the wire."""

    def _post(self, server: ServerInfo, name: str, body: dict | None = None) -> httpx.Response:
        return httpx.post(
            f"{_base_url(server)}/actions/{name}",
            json=body if body is not None else {},
            timeout=10.0,
        )

    def test_unknown_action_is_404(self, server: ServerInfo) -> None:
        resp = self._post(server, "no_such_action")
        assert resp.status_code == 404
        body = resp.json()
        assert body == {"code": "unknown_action", "message": body["message"]}
        # The name is echoed (it's the caller's, not a server internal).
        assert "no_such_action" in body["message"]

    def test_input_error_is_400(self, server: ServerInfo) -> None:
        # list_notes with no filter raises ToolInputError → 400 input_error.
        resp = self._post(server, "list_notes", {})
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "input_error"
        assert "filter" in body["message"].lower()

    def test_out_of_range_arg_is_400_input_error(self, server: ServerInfo) -> None:
        # A bound the MCP tool's arg_model enforces (limit <= 200) must NOT be
        # bypassable on the UI edge: the same validation runs, surfacing a 400.
        resp = self._post(server, "list_notes", {"deck": "Default", "limit": 9999})
        assert resp.status_code == 400
        assert resp.json()["code"] == "input_error"

    def test_malformed_body_is_400_not_500(self, server: ServerInfo) -> None:
        resp = httpx.post(
            f"{_base_url(server)}/actions/collection_info",
            content="this is not json",
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "input_error"

    def test_non_object_body_is_400(self, server: ServerInfo) -> None:
        resp = httpx.post(
            f"{_base_url(server)}/actions/collection_info",
            content="[1, 2, 3]",
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "input_error"

    @pytest.mark.parametrize(
        ("name", "body"),
        [
            ("no_such_action", {}),  # 404
            ("list_notes", {}),  # 400 input_error
            ("list_notes", {"deck": "Default", "limit": 9999}),  # 400 validation
        ],
    )
    def test_error_bodies_never_leak_internals(
        self, server: ServerInfo, name: str, body: dict
    ) -> None:
        # No filesystem path, stack frame, or module internal in ANY error body.
        resp = self._post(server, name, body)
        assert resp.status_code in (400, 404, 409, 500)
        payload = resp.json()
        assert set(payload.keys()) == {"code", "message"}
        text = payload["message"]
        for leak in ("Traceback", "/Users/", "/home/", "site-packages", 'File "'):
            assert leak not in text, f"error message leaked {leak!r}: {text!r}"


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
