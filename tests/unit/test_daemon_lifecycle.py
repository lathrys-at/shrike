"""Unit coverage for daemon lifecycle helpers: the lock, state files, the
shutdown/kill primitives, and server_status. (stop_server's escalation ladder is
covered separately in test_daemon.py.)"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest

from shrike import daemon
from shrike.daemon import (
    AlreadyRunningError,
    ServerLock,
    _force_kill,
    _request_http_shutdown,
    _signal_term,
    is_server_alive,
    read_pid,
    read_server_meta,
    server_status,
)


class TestServerLock:
    def test_acquire_writes_pid_and_meta(self, tmp_path):
        lock = ServerLock(state_dir_override=tmp_path)
        lock.acquire(meta={"url": "http://x", "pid": 1})
        try:
            assert (tmp_path / "server.pid").read_text() == str(os.getpid())
            assert json.loads((tmp_path / "server.json").read_text())["url"] == "http://x"
        finally:
            lock.release()

    def test_second_acquire_raises_already_running(self, tmp_path):
        held = ServerLock(state_dir_override=tmp_path)
        held.acquire(meta={"pid": 42})
        try:
            with pytest.raises(AlreadyRunningError, match="42"):
                ServerLock(state_dir_override=tmp_path).acquire(meta={})
        finally:
            held.release()

    def test_release_cleans_state_files(self, tmp_path):
        lock = ServerLock(state_dir_override=tmp_path)
        lock.acquire(meta={})
        lock.release()
        assert not (tmp_path / "server.pid").exists()
        assert not (tmp_path / "server.json").exists()

    def test_context_manager_releases(self, tmp_path):
        lock = ServerLock(state_dir_override=tmp_path)
        with lock:
            lock.acquire(meta={})
            assert is_server_alive(tmp_path) is True
        assert is_server_alive(tmp_path) is False

    def test_release_without_acquire_is_noop(self, tmp_path):
        ServerLock(state_dir_override=tmp_path).release()  # no exception


class TestIsServerAlive:
    def test_false_when_unlocked(self, tmp_path):
        assert is_server_alive(tmp_path) is False

    def test_true_when_held(self, tmp_path):
        lock = ServerLock(state_dir_override=tmp_path)
        lock.acquire(meta={})
        try:
            assert is_server_alive(tmp_path) is True
        finally:
            lock.release()


class TestReadStateFiles:
    def test_read_meta_valid(self, tmp_path):
        (tmp_path / "server.json").write_text(json.dumps({"url": "http://x"}))
        assert read_server_meta(tmp_path) == {"url": "http://x"}

    def test_read_meta_missing(self, tmp_path):
        assert read_server_meta(tmp_path) is None

    def test_read_meta_corrupt(self, tmp_path):
        (tmp_path / "server.json").write_text("{not json")
        assert read_server_meta(tmp_path) is None

    def test_read_pid_valid(self, tmp_path):
        (tmp_path / "server.pid").write_text("4242\n")
        assert read_pid(tmp_path) == 4242

    def test_read_pid_missing(self, tmp_path):
        assert read_pid(tmp_path) is None

    def test_read_pid_invalid(self, tmp_path):
        (tmp_path / "server.pid").write_text("not-a-number")
        assert read_pid(tmp_path) is None


class TestRequestHttpShutdown:
    def test_returns_true_on_200(self):
        with patch("httpx.post", return_value=MagicMock(status_code=200)):
            assert _request_http_shutdown("http://127.0.0.1:8372/mcp") is True

    def test_returns_false_on_non_200(self):
        with patch("httpx.post", return_value=MagicMock(status_code=503)):
            assert _request_http_shutdown("http://127.0.0.1:8372/mcp") is False

    def test_returns_false_on_connect_error(self):
        with patch("httpx.post", side_effect=httpx.ConnectError("boom")):
            assert _request_http_shutdown("http://127.0.0.1:8372/mcp") is False


class TestSignalHelpers:
    def test_force_kill_unix_uses_sigkill(self):
        import signal

        with patch.object(daemon.sys, "platform", "linux"), patch("os.kill") as kill:
            _force_kill(123)
        kill.assert_called_once_with(123, signal.SIGKILL)

    def test_force_kill_windows_uses_sigterm(self):
        import signal

        with patch.object(daemon.sys, "platform", "win32"), patch("os.kill") as kill:
            _force_kill(123)
        kill.assert_called_once_with(123, signal.SIGTERM)

    def test_signal_term_unix_success(self):
        with patch.object(daemon.sys, "platform", "linux"), patch("os.kill") as kill:
            assert _signal_term(123) is True
        kill.assert_called_once()

    def test_signal_term_unix_process_gone(self):
        with (
            patch.object(daemon.sys, "platform", "linux"),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            assert _signal_term(123) is False

    def test_signal_term_windows_noop(self):
        with patch.object(daemon.sys, "platform", "win32"), patch("os.kill") as kill:
            assert _signal_term(123) is False
        kill.assert_not_called()


class TestServerStatus:
    def test_not_running_clean(self):
        with (
            patch.object(daemon, "is_server_alive", return_value=False),
            patch.object(daemon, "read_server_meta", return_value=None),
            patch.object(daemon, "read_pid", return_value=None),
            patch.object(daemon, "PID_FILE", MagicMock(exists=lambda: False)),
            patch.object(daemon, "cleanup_state") as cleanup,
        ):
            assert server_status() == {"running": False}
            cleanup.assert_not_called()

    def test_not_running_cleans_stale(self):
        with (
            patch.object(daemon, "is_server_alive", return_value=False),
            patch.object(daemon, "read_server_meta", return_value={"url": "x"}),
            patch.object(daemon, "read_pid", return_value=None),
            patch.object(daemon, "PID_FILE", MagicMock(exists=lambda: True)),
            patch.object(daemon, "cleanup_state") as cleanup,
        ):
            assert server_status() == {"running": False}
            cleanup.assert_called_once()

    def _running_with_started(self, started: str) -> dict:
        with (
            patch.object(daemon, "is_server_alive", return_value=True),
            patch.object(
                daemon,
                "read_server_meta",
                return_value={
                    "url": "http://x",
                    "collection": "/c",
                    "log_level": "info",
                    "log_dir": "/l",
                    "started": started,
                },
            ),
            patch.object(daemon, "read_pid", return_value=4242),
            patch.object(daemon, "PID_FILE", MagicMock(exists=lambda: True)),
        ):
            return server_status()

    def test_running_uptime_hours(self):
        started = (datetime.now(UTC) - timedelta(hours=2, minutes=3)).isoformat()
        result = self._running_with_started(started)
        assert result["running"] is True
        assert result["pid"] == 4242
        assert result["url"] == "http://x"
        assert result["uptime"] == "2h 3m"

    def test_running_uptime_minutes(self):
        started = (datetime.now(UTC) - timedelta(minutes=5, seconds=10)).isoformat()
        assert self._running_with_started(started)["uptime"].endswith("s")
        assert self._running_with_started(started)["uptime"].startswith("5m")

    def test_running_uptime_seconds(self):
        started = (datetime.now(UTC) - timedelta(seconds=20)).isoformat()
        assert self._running_with_started(started)["uptime"].endswith("s")

    def test_running_invalid_started_no_uptime(self):
        result = self._running_with_started("not-a-date")
        assert result["running"] is True
        assert "uptime" not in result

    def test_running_without_meta(self):
        with (
            patch.object(daemon, "is_server_alive", return_value=True),
            patch.object(daemon, "read_server_meta", return_value=None),
            patch.object(daemon, "read_pid", return_value=7),
            patch.object(daemon, "PID_FILE", MagicMock(exists=lambda: True)),
        ):
            result = server_status()
        assert result == {"running": True, "pid": 7}
