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


def _capture_post(body: dict):
    """A patched httpx.Client.post that records the URL + JSON body it was sent.

    The actions edge (#687) POSTs the action's arguments as the JSON body to
    ``/actions/{name}`` and gets the response-model dict back directly.
    """
    captured: dict = {}

    def _post(self, url, *, json, **kwargs):  # noqa: ANN001, A002
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = kwargs.get("headers")
        return _resp(200, body)

    return _post, captured


class TestSelectorInjection:
    """#68: the client injects its --profile/--collection selector into every
    routed action's arguments; list_profiles (registry-level) is exempt."""

    def test_selector_injected_into_arguments(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False, collection="work")
        post, captured = _capture_post({})
        with patch("httpx.Client.post", post):
            c._action("collection_info", {"include": ["summary"]})
        assert captured["url"].endswith("/actions/collection_info")
        assert captured["json"]["collection"] == "work"
        assert captured["json"]["include"] == ["summary"]

    def test_no_selector_leaves_arguments_untouched(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)  # collection=None
        post, captured = _capture_post({})
        with patch("httpx.Client.post", post):
            c._action("collection_info", {})
        assert "collection" not in captured["json"]

    def test_explicit_call_collection_wins(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False, collection="work")
        post, captured = _capture_post({})
        with patch("httpx.Client.post", post):
            c._action("collection_info", {"collection": "override"})
        assert captured["json"]["collection"] == "override"

    def test_list_profiles_is_exempt(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False, collection="work")
        post, captured = _capture_post({})
        with patch("httpx.Client.post", post):
            c._action("list_profiles", {})
        assert "collection" not in captured["json"]


class TestAction:
    def test_success_returns_body(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with patch("httpx.Client.post", return_value=_resp(200, {"ok": 1})):
            assert c._action("t") == {"ok": 1}

    def test_posts_to_actions_route_with_wire_header(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        from shrike.client import WIRE_VERSION_HEADER
        from shrike.schemas import WIRE_PROTOCOL_VERSION

        post, captured = _capture_post({"ok": 1})
        with patch("httpx.Client.post", post):
            c._action("collection_info", {})
        assert captured["url"] == "http://x:1/actions/collection_info"
        assert captured["headers"][WIRE_VERSION_HEADER] == str(WIRE_PROTOCOL_VERSION)

    def test_input_error_raises_server_error(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        body = {"code": "input_error", "message": "bad note"}
        with (
            patch("httpx.Client.post", return_value=_resp(400, body)),
            pytest.raises(ServerError, match="bad note"),
        ):
            c._action("t")

    def test_unknown_action_raises_server_error(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        body = {"code": "unknown_action", "message": "No action named 'nope'."}
        with (
            patch("httpx.Client.post", return_value=_resp(404, body)),
            pytest.raises(ServerError, match="No action named"),
        ):
            c._action("nope")

    def test_internal_error_raises_server_error(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        body = {"code": "internal_error", "message": "The server failed to process this action."}
        with (
            patch("httpx.Client.post", return_value=_resp(500, body)),
            pytest.raises(ServerError, match="failed to process"),
        ):
            c._action("t")

    def test_undecodable_error_falls_back_to_http_error(self) -> None:
        # A non-2xx without a recognizable ActionError envelope (e.g. a proxy
        # 502 / the guard's 421) maps to the generic typed HTTP error.
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with (
            patch("httpx.Client.post", return_value=_resp(502, {"oops": True})),
            pytest.raises(ServerHTTPError) as ei,
        ):
            c._action("t")
        assert ei.value.status_code == 502

    def test_unreachable(self) -> None:
        c = ShrikeClient("http://x:1/mcp", autostart=False)
        with (
            patch("httpx.Client.post", side_effect=httpx.ConnectError("down")),
            pytest.raises(ServerUnreachableError),
        ):
            c._action("t")


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
        calls = {"n": 0}

        def post(url, json=None, headers=None, timeout=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("down")
            return _resp(200, {"ok": 1})

        with (
            patch("httpx.Client.post", side_effect=post),
            patch.object(ShrikeClient, "ensure_running", return_value=spec.url) as er,
        ):
            assert c._action("t") == {"ok": 1}
        er.assert_called_once()
        assert calls["n"] == 2
