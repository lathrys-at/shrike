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


# The DATA-plane custom routes (each wrapped by the data `_guard`). A forged
# Host/Origin is rejected *before* the handler runs. The privileged control routes
# (/shutdown, /index/*, /embedding/*, /reload, full /status) are NOT on this
# listener at all — they live on the always-local control plane (see
# TestControlPlaneSplit), so they can't be probed here.
_CUSTOM_ROUTES = [
    ("GET", "/health"),
    ("GET", "/media/probe.png"),
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
        assert httpx.get(f"{_base_url(server)}/health", timeout=5.0).status_code == 200

    @pytest.mark.parametrize(("method", "path"), _CUSTOM_ROUTES)
    def test_route_rejects_forged_host(self, server: ServerInfo, method: str, path: str) -> None:
        resp = httpx.request(
            method,
            f"{_base_url(server)}{path}",
            headers={"Host": "evil.example.com"},
            timeout=5.0,
        )
        assert resp.status_code == 421
        assert httpx.get(f"{_base_url(server)}/health", timeout=5.0).status_code == 200


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
        resp = httpx.get(f"{_base_url(server)}/health", headers={"Origin": "null"}, timeout=5.0)
        assert resp.status_code == 403

    def test_no_origin_loopback_host_allowed(self, server: ServerInfo) -> None:
        # The native-client path (mcp-remote, CLI): no Origin header + a loopback
        # Host is allowed. Pinned explicitly so a future tightening can't silently
        # break native clients.
        resp = httpx.get(f"{_base_url(server)}/health", timeout=5.0)
        assert resp.status_code == 200

    def test_get_on_post_only_route_is_405(self, server: ServerInfo) -> None:
        # The data plane's POST-only route (/actions/*) can't be fired by a
        # no-preflight GET (`<img src=...>`): a GET is 405, before any handler, so a
        # write action can never ride a cross-origin <img>/<form>.
        resp = httpx.get(f"{_base_url(server)}/actions/collection_info", timeout=5.0)
        assert resp.status_code == 405
        assert httpx.get(f"{_base_url(server)}/health", timeout=5.0).status_code == 200


class TestEscapeHatchesFlipBehavior:
    """The disable flag must actually disable the data-plane guard (not just be
    wired). It governs the data plane only — the control plane stays always-local
    regardless (see TestControlPlaneSplit)."""

    def test_no_dns_rebinding_protection_accepts_forged_headers(self, server_factory) -> None:
        srv = server_factory("nodns", extra_args=["--no-dns-rebinding-protection"])
        base = _base_url(srv)
        # A forged Origin/Host that would 403/421 by default is now accepted.
        assert (
            httpx.get(
                f"{base}/health", headers={"Origin": "http://evil.example.com"}, timeout=5.0
            ).status_code
            == 200
        )
        assert (
            httpx.get(
                f"{base}/health", headers={"Host": "evil.example.com"}, timeout=5.0
            ).status_code
            == 200
        )


class TestControlPlaneSplit:
    """The privileged control routes live on a separate always-local listener, not
    the data plane. This is the structural guarantee that supersedes the per-request
    exec-override gate: a remote/proxied caller can't reach the embedding spawn (or
    shutdown/reload/index/full-status) at all, regardless of data-plane exposure."""

    # The routes that must NOT be reachable on the data listener.
    _CONTROL_ROUTES = [
        ("GET", "/status"),
        ("GET", "/metrics"),
        ("POST", "/shutdown"),
        ("POST", "/index/rebuild"),
        ("POST", "/index/save"),
        ("POST", "/embedding/start"),
        ("POST", "/embedding/stop"),
        ("POST", "/reload"),
    ]

    @pytest.mark.parametrize(("method", "path"), _CONTROL_ROUTES)
    def test_control_routes_absent_from_data_plane(
        self, server: ServerInfo, method: str, path: str
    ) -> None:
        # Not registered on the data app → 404 (route not found), never the handler.
        resp = httpx.request(method, f"{_base_url(server)}{path}", timeout=5.0)
        assert resp.status_code == 404, resp.text
        # And the server is unharmed (the /shutdown probe did nothing).
        assert httpx.get(f"{_base_url(server)}/health", timeout=5.0).status_code == 200

    @pytest.mark.parametrize(("method", "path"), _CONTROL_ROUTES)
    def test_control_routes_absent_even_with_remote_exposure(
        self, server_factory, method: str, path: str
    ) -> None:
        # The capability hole the split closes: with the data plane's guard disabled
        # (the behind-a-proxy posture), the control routes are STILL not on the
        # data listener — the structural guarantee doesn't depend on the guard.
        srv = server_factory("splitnodns", extra_args=["--no-dns-rebinding-protection"])
        resp = httpx.request(method, f"{_base_url(srv)}{path}", timeout=5.0)
        assert resp.status_code == 404, resp.text

    def test_control_routes_reachable_on_control_plane(self, server: ServerInfo) -> None:
        # Full /status is served on the control channel (UDS or loopback TCP).
        resp = server.control_request("GET", "/status", timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] is True
        assert "collection" in body  # the full diagnostics, not the minimal /health

        metrics = server.control_request("GET", "/metrics", timeout=5.0)
        assert metrics.status_code == 200
        assert metrics.headers["content-type"].startswith("text/plain")
        assert "# TYPE shrike_http_requests_total counter" in metrics.text
        assert "shrike_runtime_pool_queue_depth" in metrics.text
        assert "shrike_index_saver_pending" in metrics.text

    def test_metrics_scrapes_are_observational(self, server: ServerInfo) -> None:
        # Create one deterministic request sample, then prove scraping itself is
        # excluded from the request counter family.
        assert httpx.get(f"{_base_url(server)}/health", timeout=5.0).status_code == 200
        first = server.control_request("GET", "/metrics", timeout=5.0).text
        second = server.control_request("GET", "/metrics", timeout=5.0).text

        def request_lines(text: str) -> list[str]:
            return sorted(
                line for line in text.splitlines() if line.startswith("shrike_http_requests_total{")
            )

        assert request_lines(first) == request_lines(second)
        assert any('route="/health"' in line for line in request_lines(second))

    def test_health_is_minimal_and_leaks_nothing(self, server: ServerInfo) -> None:
        # The data plane's liveness probe carries running + wire version, and none
        # of the sensitive diagnostics the full control-plane /status does.
        body = httpx.get(f"{_base_url(server)}/health", timeout=5.0).json()
        assert body["running"] is True
        assert "wire_protocol_version" in body
        for leaky in ("pid", "collection", "log_dir", "url", "embedding", "index"):
            assert leaky not in body

    @pytest.mark.skipif(sys.platform == "win32", reason="UDS control socket is POSIX-only")
    def test_control_socket_is_owner_only(self, server: ServerInfo) -> None:
        import stat

        from shrike.platform import daemon

        # The socket lives in a short runtime dir (AF_UNIX length limit), its path
        # recorded in server.json — resolve it the way the client does.
        _, uds = daemon.control_channel(daemon.read_server_meta(Path(server.state_dir)))
        assert uds is not None
        sock = Path(uds)
        assert sock.exists()
        assert stat.S_IMODE(sock.stat().st_mode) == 0o600


class TestEmbeddingStartControlPlane:
    """`/embedding/start` is a control route. It best-effort-parses an
    arbitrary JSON body and must not 500 on garbage within the trust model: the
    parse is suppressed and only *known* keys are forwarded. With no model
    configured every variant lands on the clean 400 "no model" path. Driven over
    the control channel since the route isn't on the data listener."""

    @pytest.mark.parametrize(
        "body",
        [
            {"unknown_key": 1, "another": [1, 2, 3]},  # unknown keys filtered out
            {"port": "not-an-int"},  # wrong-typed known key (never used: no model)
            {"model": None},  # explicit null is skipped
            {"llama_server": "/tmp/whatever"},  # an exec override (allowed: local control plane)
        ],
    )
    def test_garbage_body_yields_clean_400_not_500(self, server: ServerInfo, body: dict) -> None:
        resp = server.control_request("POST", "/embedding/start", json=body, timeout=10.0)
        assert resp.status_code == 400, resp.text  # no model configured, handled cleanly
        # Not the exec-override gate's refusal — the local control plane forwards it.
        assert "Execution-shaping parameters" not in resp.text
        assert server.control_request("GET", "/status", timeout=5.0).status_code == 200


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
