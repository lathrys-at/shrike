"""Unit coverage for the `shrike server` CLI commands (start/stop/status/logs).

These manage the daemon: spawning, stopping, status reporting, and log viewing.
Driven with mocked daemon helpers / client / subprocess so no real server is
spawned. Patches target `shrike.cli.server_cmd.*` (where the names are bound).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli.server_cmd import _wait_for_server
from shrike.schemas import (
    CoverageCell,
    CoverageMatrix,
    CoverageRow,
    EmbeddingDown,
    EmbeddingRunning,
    IndexBuilding,
    IndexErrored,
    IndexModalityStat,
    IndexProgress,
    IndexReady,
    ServerStatus,
)

SC = "shrike.cli.server_cmd"


@pytest.fixture
def run(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str, **kwargs):
        return runner.invoke(cli, ["--config", str(cfg), *args], **kwargs)

    return _run


def _server(index=None, embedding=None, *, log_dir="/logs", coverage=None, embedding_spaces=None):
    return ServerStatus(
        wire_protocol_version=1,
        pid=4242,
        url="http://127.0.0.1:8372/mcp",
        collection="/c.anki2",
        log_level="info",
        log_dir=log_dir,
        uptime="0:01:00",
        embedding=embedding or EmbeddingRunning(available=True, url="http://e", pid=99, model="/m"),
        embedding_spaces=embedding_spaces or [],
        index=index or IndexReady(state="ready", size=5, ndim=384, col_mod=7, path="/i"),
        coverage=coverage,
    )


class TestServerStatus:
    def test_responsive_renders(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server()
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert result.exit_code == 0
        assert "Server is running" in result.output
        assert "4242" in result.output
        assert "available" in result.output  # embedding
        assert "ready" in result.output  # index

    def test_responsive_json(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server()
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("--json", "server", "status")
        assert result.exit_code == 0
        import json

        assert json.loads(result.output)["pid"] == 4242

    def test_index_building_and_embedding_down(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server(
            index=IndexBuilding(state="building", progress=IndexProgress(indexed=2, total=8)),
            embedding=EmbeddingDown(state="failed"),
        )
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert "building" in result.output
        assert "2 / 8" in result.output
        assert "failed to start" in result.output

    def test_index_error(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server(index=IndexErrored(state="error", error="boom"))
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert "boom" in result.output

    def test_coverage_matrix_not_in_status(self, run):
        # The coverage matrix moved to `shrike search coverage` (#683): even when
        # /status still carries `coverage`, `server status` must not render it.
        coverage = CoverageMatrix(
            text=CoverageRow(
                text=CoverageCell.NATIVE,
                image=CoverageCell.VIA_DERIVED_TEXT,
                audio=CoverageCell.UNAVAILABLE,
            ),
        )
        fake = MagicMock()
        fake.server_status.return_value = _server(coverage=coverage)
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert result.exit_code == 0
        assert "Coverage" not in result.output
        assert "via text" not in result.output

    def test_status_on_own_line_and_section_order(self, run):
        # §B reshape (#684): the section header is the identity (`Index`,
        # `Embedding`), `Status:` moves onto its own line, and the order is
        # Index → Derived text → Recognition → Embedding.
        fake = MagicMock()
        fake.server_status.return_value = _server()
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        # Status moved to its own line (header carries no inline status now).
        assert "Status:" in out
        assert "Index:" not in out
        assert "Embedding:" not in out
        # Section order: Index before Derived text before Recognition before Embedding.
        i_index = out.index("Index")
        i_derived = out.index("Derived text")
        i_recog = out.index("Recognition")
        i_embed = out.index("Embedding")
        assert i_index < i_derived < i_recog < i_embed

    def test_index_per_modality_breakdown(self, run):
        # §C index (#684): each sub-index reports its own size/ndim.
        fake = MagicMock()
        fake.server_status.return_value = _server(
            index=IndexReady(
                state="ready",
                size=87,
                ndim=768,
                modalities=[
                    IndexModalityStat(modality="text", size=50, ndim=768),
                    IndexModalityStat(modality="image", size=37, ndim=512),
                ],
            )
        )
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        assert "Vectors (text)" in out
        assert "Vectors (image)" in out
        assert "768 dims" in out
        assert "512 dims" in out

    def test_embedding_per_space(self, run):
        # §C embedding (#681): one `Embedding [<modalities>]` block per space.
        fake = MagicMock()
        fake.server_status.return_value = _server(
            embedding=EmbeddingRunning(available=True, model="gemma", modalities=["text"]),
            embedding_spaces=[
                EmbeddingRunning(available=True, model="gemma", modalities=["text"]),
                EmbeddingRunning(available=True, model="clip", modalities=["text", "image"]),
            ],
        )
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        assert "Embedding [text]" in out
        assert "Embedding [text, image]" in out
        assert "gemma" in out
        assert "clip" in out

    def test_running_but_unresponsive(self, run):
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={"pid": 7, "url": "http://x"}),
        ):
            result = run("server", "status")
        assert "not responding" in result.output
        assert "http://x" in result.output

    def test_not_running_exits_1(self, run):
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.META_FILE") as meta_file,
            patch(f"{SC}.cleanup_state"),
        ):
            meta_file.exists.return_value = False
            result = run("server", "status")
        assert result.exit_code == 1
        assert "not running" in result.output.lower()

    def test_not_running_json(self, run):
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.META_FILE") as meta_file,
            patch(f"{SC}.cleanup_state"),
        ):
            meta_file.exists.return_value = False
            result = run("--json", "server", "status")
        import json

        assert json.loads(result.output)["running"] is False


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
            patch(f"{SC}.META_FILE") as meta_file,
            patch(f"{SC}.cleanup_state") as cleanup,
        ):
            meta_file.exists.return_value = True
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


_LOG = "2025-05-24T10:30:00 INFO  shrike.tools  list_notes deck=Test\n"


class TestServerLogs:
    def _logfile(self, tmp_path) -> Path:
        p = tmp_path / "shrike.log"
        p.write_text(_LOG * 3)
        return p

    def test_file_plain(self, run, tmp_path):
        with patch(f"{SC}.get_log_file", return_value=self._logfile(tmp_path)):
            result = run("server", "logs")
        assert "list_notes" in result.output

    def test_file_json(self, run, tmp_path):
        with patch(f"{SC}.get_log_file", return_value=self._logfile(tmp_path)):
            result = run("--json", "server", "logs")
        import json

        data = json.loads(result.output)
        assert data["messages"] and data["messages"][0]["logger"] == "shrike.tools"

    def test_not_found(self, run, tmp_path):
        with patch(f"{SC}.get_log_file", return_value=tmp_path / "missing.log"):
            result = run("server", "logs")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_json_and_follow_conflict(self, run):
        result = run("--json", "server", "logs", "--follow")
        assert result.exit_code != 0
        assert "cannot be used together" in result.output

    def test_stdin_plain(self, run):
        result = run("--no-pretty", "server", "logs", "--stdin", input=_LOG)
        assert "list_notes" in result.output

    def test_stdin_json(self, run):
        result = run("--json", "server", "logs", "--stdin", input=_LOG)
        import json

        assert json.loads(result.output)["messages"][0]["level"] == "INFO"

    def test_follow(self, run, tmp_path):
        # _tail_follow prints the tail, then loops on select.select — make the
        # first poll raise KeyboardInterrupt so it exits cleanly.
        logfile = self._logfile(tmp_path)
        with (
            patch(f"{SC}.get_log_file", return_value=logfile),
            patch("select.select", side_effect=KeyboardInterrupt),
        ):
            result = run("server", "logs", "--follow")
        assert "list_notes" in result.output
        assert "following" in result.output


class TestServerStart:
    def _common(self, tmp_path, **overrides):
        """Patches for the daemon path; overridable per test."""
        defaults = dict(
            is_server_alive=MagicMock(return_value=False),
            cleanup_state=MagicMock(),
            get_log_file=MagicMock(return_value=tmp_path / "shrike.log"),
            STATE_DIR=tmp_path / "state",
        )
        defaults.update(overrides)
        return patch.multiple(SC, **defaults)

    def test_no_collection_errors(self, run):
        with patch(f"{SC}.is_server_alive", return_value=False):
            result = run("server", "start")
        assert result.exit_code != 0
        assert "No collection path" in result.output

    def test_already_running_errors(self, run, tmp_path):
        with (
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={"pid": 5, "url": "http://x"}),
        ):
            result = run("server", "start", "--collection", str(tmp_path / "c.anki2"))
        assert result.exit_code != 0
        assert "already running" in result.output

    def test_daemon_success(self, run, tmp_path):
        proc = MagicMock(pid=123)
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=_server()),
        ):
            result = run(
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
            )
        assert result.exit_code == 0
        assert "Server is running" in result.output

    def test_daemon_started_not_responding(self, run, tmp_path):
        proc = MagicMock(pid=123)
        proc.poll.return_value = None  # still alive, just slow
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=None),
        ):
            result = run(
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
            )
        assert result.exit_code == 0
        assert "not yet responding" in result.output

    def test_daemon_process_exited(self, run, tmp_path):
        proc = MagicMock(pid=123, returncode=1)
        proc.poll.return_value = 1  # exited
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=None),
        ):
            result = run(
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
            )
        assert result.exit_code != 0
        assert "exited with code 1" in result.output

    def test_save_config(self, run, tmp_path):
        proc = MagicMock(pid=123)
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=_server()),
            patch(f"{SC}.save_config", return_value=str(tmp_path / "config.yml")) as save,
        ):
            result = run(
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--save-config",
            )
        assert result.exit_code == 0
        save.assert_called_once()
        assert "Config saved" in result.output

    def test_save_config_with_all_options(self, run, tmp_path):
        # Exercise the optional save-config branches (transport, embedding, cache).
        proc = MagicMock(pid=123)
        captured = {}
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=_server()),
            patch(f"{SC}.save_config", side_effect=lambda cfg, _p: captured.update(cfg) or "/cfg"),
        ):
            result = run(
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--allow-remote",
                "--allowed-host",
                "h:*",
                "--allowed-origin",
                "http://o",
                "--no-dns-rebinding-protection",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--index-save-delay",
                "30",
                "--embedding-model",
                str(tmp_path / "m.gguf"),
                "--save-config",
            )
        assert result.exit_code == 0
        assert captured["server"]["allow_remote"] is True
        assert captured["server"]["allowed_hosts"] == ["h:*"]
        assert captured["server"]["no_dns_rebinding_protection"] is True
        assert captured["cache_dir"].endswith("cache")

    def test_daemon_success_json(self, run, tmp_path):
        proc = MagicMock(pid=123)
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=_server()),
        ):
            result = run(
                "--json",
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
            )
        import json

        assert json.loads(result.output)["pid"] == 4242

    def test_daemon_not_responding_json(self, run, tmp_path):
        proc = MagicMock(pid=123)
        proc.poll.return_value = None
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=None),
        ):
            result = run(
                "--json",
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
            )
        import json

        assert json.loads(result.output)["responding"] is False


class TestWaitForServer:
    def test_returns_status_once_responsive(self):
        client = MagicMock()
        client.server_status.side_effect = [None, _server()]
        with (
            patch(f"{SC}.ShrikeClient", return_value=client),
            patch(f"{SC}.time.sleep"),
        ):
            status = _wait_for_server("http://127.0.0.1:8372/mcp", show_spinner=False)
        assert status is not None and status.pid == 4242

    def test_times_out(self):
        client = MagicMock()
        client.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=client),
            patch(f"{SC}.time.sleep"),
            patch(f"{SC}.time.monotonic", side_effect=[0.0, 0.0, 100.0]),
        ):
            status = _wait_for_server("http://x", timeout=1.0, show_spinner=False)
        assert status is None

    def test_foreground(self, run, tmp_path):
        saved_argv = sys.argv[:]
        main = MagicMock()
        try:
            with (
                patch(f"{SC}.is_server_alive", return_value=False),
                patch("shrike.server.server.main", main),
            ):
                result = run(
                    "server",
                    "start",
                    "--collection",
                    str(tmp_path / "c.anki2"),
                    "--log-dir",
                    str(tmp_path / "logs"),
                    "--foreground",
                )
        finally:
            sys.argv = saved_argv
        assert result.exit_code == 0
        main.assert_called_once()
        assert "foreground" in result.output
