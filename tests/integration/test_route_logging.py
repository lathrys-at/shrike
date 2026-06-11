"""Custom-route access logging (#328): every guarded route — /status polls
included — logs method, path, status, and duration at INFO."""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

# e.g. "POST /index/save -> 200 (3ms)"
_ACCESS = re.compile(r"(GET|POST) (/\S*) -> (\d{3}) \((\d+)ms\)")


def _log_text(server) -> str:
    log_file = Path(server.log_dir) / "shrike.log"
    return log_file.read_text() if log_file.exists() else ""


def _wait_for(server, predicate, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        text = _log_text(server)
        if predicate(text):
            return text
        time.sleep(0.1)
    return text


class TestRouteAccessLog:
    def test_post_route_logs_status_and_duration(self, server) -> None:
        base = server.url.rsplit("/", 1)[0]
        assert httpx.post(f"{base}/index/save", timeout=10.0).status_code == 200
        text = _wait_for(server, lambda t: "POST /index/save -> 200" in t)
        match = next(
            (m for m in _ACCESS.finditer(text) if m.group(2) == "/index/save"),
            None,
        )
        assert match is not None, "no access line for POST /index/save"
        assert match.group(1) == "POST"
        assert match.group(3) == "200"
        assert int(match.group(4)) >= 0  # the duration is present and parseable

    def test_media_404_logged_with_status(self, server) -> None:
        base = server.url.rsplit("/", 1)[0]
        assert httpx.get(f"{base}/media/never-there.png", timeout=10.0).status_code == 404
        text = _wait_for(server, lambda t: "/media/never-there.png -> 404" in t)
        assert "GET /media/never-there.png -> 404" in text

    def test_status_poll_logged_at_info(self, server) -> None:
        # Every served route logs at INFO — /status polls included (the
        # operator wants to see what the server did, not a filtered view).
        base = server.url.rsplit("/", 1)[0]
        assert httpx.get(f"{base}/status", timeout=10.0).status_code == 200
        text = _wait_for(server, lambda t: "GET /status -> 200" in t)
        assert "GET /status -> 200" in text
