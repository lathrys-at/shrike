"""Tests for the standalone shrike.client library."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from shrike.client import (
    ServerError,
    ServerHTTPError,
    ServerSpec,
    ServerStartError,
    ServerUnreachableError,
    ShrikeClient,
)


def _resp(status: int = 200, json_body: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {} if json_body is None else json_body
    return r


class TestCall:
    def test_success_returns_content(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        body = {"result": {"structuredContent": {"ok": 1}}}
        with patch("httpx.Client.post", return_value=_resp(200, body)):
            assert c._call("t") == {"ok": 1}

    def test_is_error_result_raises(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        body = {"result": {"isError": True, "content": [{"type": "text", "text": "bad note"}]}}
        with (
            patch("httpx.Client.post", return_value=_resp(200, body)),
            pytest.raises(ServerError, match="bad note"),
        ):
            c._call("t")

    def test_jsonrpc_error(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with (
            patch("httpx.Client.post", return_value=_resp(200, {"error": {"code": -1}})),
            pytest.raises(ServerError),
        ):
            c._call("t")

    def test_http_error_raises_typed(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with (
            patch("httpx.Client.post", return_value=_resp(500, {})),
            pytest.raises(ServerHTTPError) as ei,
        ):
            c._call("t")
        assert ei.value.status_code == 500

    def test_unreachable(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with (
            patch("httpx.Client.post", side_effect=httpx.ConnectError("down")),
            pytest.raises(ServerUnreachableError),
        ):
            c._call("t")


class TestCustomEndpoints:
    def test_http_error_surfaces_server_message(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with (
            patch("httpx.Client.request", return_value=_resp(400, {"error": "no model"})),
            pytest.raises(ServerHTTPError, match="no model"),
        ):
            c.embedding_start()

    def test_request_unreachable(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with (
            patch("httpx.Client.request", side_effect=httpx.ConnectError("down")),
            pytest.raises(ServerUnreachableError),
        ):
            c.index_rebuild()

    def test_embedding_start_drops_none(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        captured: dict = {}

        def req(method, url, *, json=None, timeout=None):  # type: ignore[no-untyped-def]
            captured["json"] = json
            return _resp(
                200,
                {
                    "status": "started",
                    "embedding": {"state": "not_configured"},
                    "index": {"state": "unavailable"},
                },
            )

        with patch("httpx.Client.request", side_effect=req):
            c.embedding_start(model="/m.gguf", port=None, threads=4)
        assert captured["json"] == {"model": "/m.gguf", "threads": 4}

    def test_server_status_none_on_unreachable(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("down")):
            assert c.server_status() is None

    def test_index_status_extracts_section(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        status = {
            "running": True,
            "wire_protocol_version": 1,
            "pid": 1,
            "url": "u",
            "collection": "c",
            "log_level": "info",
            "log_dir": "/l",
            "embedding": {"state": "not_configured"},
            "index": {"state": "ready", "size": 5, "ndim": 384},
        }
        with patch("httpx.Client.request", return_value=_resp(200, status)):
            idx = c.index_status()
            assert idx.state == "ready"
            assert idx.size == 5


class TestLifecycle:
    def test_is_alive_delegates_to_daemon(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with patch("shrike.client.daemon.is_server_alive", return_value=True):
            assert c.is_alive() is True

    def test_stop_delegates_to_daemon(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with patch("shrike.client.daemon.stop_server", return_value={"stopped": True}) as m:
            assert c.stop().stopped is True
            m.assert_called_once()

    def test_ensure_running_returns_when_already_alive(self) -> None:
        spec = ServerSpec(collection="/c.anki2", port=9001)
        c = ShrikeClient(spec.url, spec=spec, autostart=False)
        with (
            patch("shrike.client.daemon.is_server_alive", return_value=True),
            patch(
                "shrike.client.daemon.read_server_meta",
                return_value={"url": "http://127.0.0.1:9001/mcp"},
            ),
        ):
            assert c.ensure_running(spec) == "http://127.0.0.1:9001/mcp"

    def test_ensure_running_spawns_and_waits(self, tmp_path) -> None:
        spec = ServerSpec(collection="/c.anki2", port=9002, log_dir=str(tmp_path))
        c = ShrikeClient(spec.url, spec=spec, autostart=False)
        proc = MagicMock()
        proc.poll.return_value = None
        with (
            patch("shrike.client.daemon.is_server_alive", return_value=False),
            patch("shrike.client.daemon.cleanup_state"),
            patch("shrike.client.subprocess.Popen", return_value=proc) as popen,
            patch.object(ShrikeClient, "wait_until_ready", return_value={"running": True}),
        ):
            assert c.ensure_running(spec) == spec.url
        popen.assert_called_once()

    def test_spawn_leaves_state_dir_alone_when_log_dir_set(self, tmp_path) -> None:
        # #424: the darwin Bazel sandbox forbids creating the real platformdirs
        # state dir, and _spawn has no business touching it — the daemon side
        # (ServerLock, the meta/pid writers) creates it where it's used. With a
        # log_dir set, a spawn must not create the state dir at all.
        spec = ServerSpec(collection="/c.anki2", port=9007, log_dir=str(tmp_path / "logs"))
        c = ShrikeClient(spec.url, spec=spec, autostart=False)
        proc = MagicMock()
        proc.poll.return_value = None
        sentinel = tmp_path / "state-dir-must-stay-absent"
        with (
            patch("shrike.client.daemon.is_server_alive", return_value=False),
            patch("shrike.client.daemon.cleanup_state"),
            patch("shrike.client.daemon.STATE_DIR", sentinel),
            patch("shrike.client.subprocess.Popen", return_value=proc),
            patch.object(ShrikeClient, "wait_until_ready", return_value={"running": True}),
        ):
            assert c.ensure_running(spec) == spec.url
        assert not sentinel.exists()

    def test_ensure_running_raises_when_spawn_exits(self, tmp_path) -> None:
        spec = ServerSpec(collection="/c.anki2", port=9003, log_dir=str(tmp_path))
        c = ShrikeClient(spec.url, spec=spec, autostart=False)
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        with (
            patch("shrike.client.daemon.is_server_alive", return_value=False),
            patch("shrike.client.daemon.cleanup_state"),
            patch("shrike.client.subprocess.Popen", return_value=proc),
            patch.object(ShrikeClient, "wait_until_ready", return_value=None),
            pytest.raises(ServerStartError),
        ):
            c.ensure_running(spec)

    def test_wait_until_ready_times_out(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        # Mock the poll sleep: otherwise the loop eats one full 0.2s interval
        # before re-checking the deadline.
        with (
            patch.object(ShrikeClient, "server_status", return_value=None),
            patch("shrike.client.time.sleep", lambda *_: None),
        ):
            assert c.wait_until_ready(timeout=0.01) is None

    def test_call_autostarts_then_retries(self, tmp_path) -> None:
        spec = ServerSpec(collection="/c.anki2", port=9004, log_dir=str(tmp_path))
        c = ShrikeClient(spec.url, spec=spec, autostart=True)
        body = {"result": {"structuredContent": {"ok": 1}}}
        calls = {"n": 0}

        def post(url, json=None, headers=None, timeout=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("down")
            return _resp(200, body)

        with (
            patch("httpx.Client.post", side_effect=post),
            patch.object(ShrikeClient, "ensure_running", return_value=spec.url) as er,
        ):
            assert c._call("t") == {"ok": 1}
        er.assert_called_once()
        assert calls["n"] == 2
