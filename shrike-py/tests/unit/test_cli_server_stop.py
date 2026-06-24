"""Unit coverage for the `shrike server stop` CLI command.

Driven with mocked daemon helpers so no real server is stopped. Patches target
`shrike.cli.server_cmd.*` (where the names are bound).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli

SC = "shrike.cli.server_cmd"


@pytest.fixture
def run(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str, **kwargs):
        return runner.invoke(cli, ["--config", str(cfg), *args], **kwargs)

    return _run


class TestServerStop:
    def test_not_running(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.META_FILE") as meta_file,
        ):
            meta_file.exists.return_value = False
            result = run("server", "stop")
        assert "not running" in result.output.lower()

    def test_not_running_cleans_stale_state(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.read_server_meta", return_value={"pid": 1}),
            patch(f"{SC}.cleanup_state") as cleanup,
        ):
            result = run("server", "stop")
        cleanup.assert_called_once()
        assert "stale state" in result.output

    def test_stopped(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.stop_server", return_value={"stopped": True}),
        ):
            result = run("server", "stop")
        assert "Server stopped" in result.output

    def test_stopped_forced(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.stop_server", return_value={"stopped": True, "forced": True}),
        ):
            result = run("server", "stop")
        assert "forced kill" in result.output

    def test_stop_failed_reason(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.stop_server", return_value={"stopped": False, "reason": "stuck"}),
        ):
            result = run("server", "stop")
        assert "stuck" in result.output

    def test_json(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.stop_server", return_value={"stopped": True}),
        ):
            result = run("--json", "server", "stop")
        import json

        assert json.loads(result.output)["stopped"] is True

    def test_not_running_json(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.read_server_meta", return_value=None),
        ):
            result = run("--json", "server", "stop")
        import json

        data = json.loads(result.output)
        assert data["stopped"] is False and data["reason"] == "not running"

    def test_stop_failed_no_reason_uses_unknown(self, run):
        with (
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.stop_server", return_value={"stopped": False}),
        ):
            result = run("server", "stop")
        assert "unknown" in result.output
