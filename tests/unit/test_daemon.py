"""Unit tests for daemon.stop_server's HTTP → SIGTERM → SIGKILL escalation.

These cover the three-tier shutdown ladder (audit §5) by patching the
side-effecting helpers, so no real process is spawned or signalled.
"""

from __future__ import annotations

from typing import Any

import pytest

from shrike import daemon


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the side-effecting helpers; record SIGTERM / force-kill / cleanup.

    Per-test overrides set ``is_server_alive`` and ``_request_http_shutdown``.
    """
    recorded: dict[str, Any] = {"sigterm": [], "force_kill": [], "cleanup": 0}

    monkeypatch.setattr(daemon, "read_pid", lambda *a, **k: 4242)
    monkeypatch.setattr(
        daemon, "read_server_meta", lambda *a, **k: {"url": "http://127.0.0.1:8372/mcp"}
    )
    monkeypatch.setattr(daemon.time, "sleep", lambda *_: None)

    def _cleanup(*a: Any, **k: Any) -> None:
        recorded["cleanup"] += 1

    def _term(pid: int) -> bool:
        recorded["sigterm"].append(pid)
        return True

    def _kill(pid: int) -> None:
        recorded["force_kill"].append(pid)

    monkeypatch.setattr(daemon, "cleanup_state", _cleanup)
    monkeypatch.setattr(daemon, "_signal_term", _term)
    monkeypatch.setattr(daemon, "_force_kill", _kill)
    return recorded


def _alive_sequence(monkeypatch: pytest.MonkeyPatch, values: list[bool]) -> None:
    """Make ``is_server_alive`` yield ``values`` in order across calls."""
    it = iter(values)
    monkeypatch.setattr(daemon, "is_server_alive", lambda *a, **k: next(it))


class TestStopServerEscalation:
    def test_http_shutdown_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]
    ) -> None:
        # Guard sees it alive; after the HTTP shutdown the loop sees it gone.
        _alive_sequence(monkeypatch, [True, False])
        monkeypatch.setattr(daemon, "_request_http_shutdown", lambda url: True)

        result = daemon.stop_server(timeout=0.05)

        assert result == {"stopped": True, "pid": 4242, "forced": False}
        assert calls["sigterm"] == []
        assert calls["force_kill"] == []
        assert calls["cleanup"] == 1

    def test_falls_back_to_sigterm_when_http_fails(
        self, monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]
    ) -> None:
        # HTTP refused → SIGTERM; the server exits during the SIGTERM wait.
        _alive_sequence(monkeypatch, [True, False])
        monkeypatch.setattr(daemon, "_request_http_shutdown", lambda url: False)

        result = daemon.stop_server(timeout=0.05)

        assert result == {"stopped": True, "pid": 4242, "forced": False}
        assert calls["sigterm"] == [4242]
        assert calls["force_kill"] == []
        assert calls["cleanup"] == 1

    def test_escalates_to_force_kill_when_sigterm_ignored(
        self, monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]
    ) -> None:
        # Hung server: stays alive through HTTP + SIGTERM, so we SIGKILL.
        monkeypatch.setattr(daemon, "is_server_alive", lambda *a, **k: True)
        monkeypatch.setattr(daemon, "_request_http_shutdown", lambda url: False)

        result = daemon.stop_server(timeout=0.02)

        assert result == {"stopped": True, "pid": 4242, "forced": True}
        assert calls["sigterm"] == [4242]
        assert calls["force_kill"] == [4242]
        assert calls["cleanup"] == 1

    def test_force_kill_after_http_accepted_but_no_exit(
        self, monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]
    ) -> None:
        # HTTP accepted but the process never dies → SIGTERM → SIGKILL.
        monkeypatch.setattr(daemon, "is_server_alive", lambda *a, **k: True)
        monkeypatch.setattr(daemon, "_request_http_shutdown", lambda url: True)

        result = daemon.stop_server(timeout=0.02)

        assert result["forced"] is True
        assert calls["force_kill"] == [4242]


class TestStopServerNotRunning:
    def test_not_running_clean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, calls: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(daemon, "is_server_alive", lambda *a, **k: False)
        # No stale state files present.
        monkeypatch.setattr(daemon, "META_FILE", tmp_path / "nope.json")
        monkeypatch.setattr(daemon, "PID_FILE", tmp_path / "nope.pid")

        result = daemon.stop_server()

        assert result == {"stopped": False, "reason": "not running"}
        assert calls["force_kill"] == []

    def test_not_running_cleans_stale_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, calls: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(daemon, "is_server_alive", lambda *a, **k: False)
        stale = tmp_path / "server.json"
        stale.write_text("{}")
        monkeypatch.setattr(daemon, "META_FILE", stale)
        monkeypatch.setattr(daemon, "PID_FILE", tmp_path / "nope.pid")

        result = daemon.stop_server()

        assert result["stopped"] is False
        assert "stale state" in result["reason"]
        assert calls["cleanup"] == 1
