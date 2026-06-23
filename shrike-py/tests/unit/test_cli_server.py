"""Unit coverage for the `shrike server` CLI commands (start/stop/status/logs).

These manage the daemon: spawning, stopping, status reporting, and log viewing.
Driven with mocked daemon helpers / client / subprocess so no real server is
spawned. Patches target `shrike.cli.server_cmd.*` (where the names are bound).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Force ``shrike.server.__init__`` to completion at collection time, before any test in
# this module patches ``shrike.server.main``. A worker otherwise imports the package only
# when a test first touches it, and both touchpoints — ``mock.patch("shrike.server.main",
# ...)`` and the foreground path's ``from shrike.server import main`` — resolve the target
# by import side-effect. The flake (#872) was ``main`` observed absent from
# ``shrike.server.__dict__`` under xdist, i.e. the package observed mid-``__init__`` (before
# the eager ``def main`` line runs). Completing the import once here closes that window for
# the whole worker: ``main`` is then a settled, permanently-bound attribute that no later
# resolution can see partial. The heavy serve graph stays deferred to ``main()`` call time.
import click
import pytest
from click.testing import CliRunner

import shrike.server
from shrike.cli import cli
from shrike.cli.server_cmd import _wait_for_server
from shrike.schemas import (
    CollectionStatus,
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
        # The coverage matrix lives in `shrike search coverage`, not `server
        # status`: even when /status still carries `coverage`, `server status`
        # must not render it.
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
        # The section header is the identity (`Index`, `Embedding`), `Status:` is
        # on its own line, and the order is
        # Index → Derived text → Recognition → Embedding.
        fake = MagicMock()
        fake.server_status.return_value = _server()
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        # Status is on its own line (the header carries no inline status).
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
        # Each sub-index reports its own size/ndim.
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
        # One `Embedding [<modalities>]` block per space.
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


class TestRenderStatusBranches:
    def test_log_and_cooperative_locking_released(self, run):
        # The optional `log` line and the cooperative-locking row (released/idle).
        status = _server()
        status.log = "/logs/shrike.log"
        status.locking = "cooperative"
        status.collection_held = False
        fake = MagicMock()
        fake.server_status.return_value = status
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert result.exit_code == 0, result.output
        assert "Log:" in result.output
        assert "cooperative" in result.output
        assert "released (idle)" in result.output

    def test_cooperative_locking_held(self, run):
        status = _server()
        status.locking = "cooperative"
        status.collection_held = True
        fake = MagicMock()
        fake.server_status.return_value = status
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert "collection held" in result.output

    def test_render_status_omits_log_and_uptime_when_absent(self):
        # _render_status is reached with log/uptime already set from the CLI path,
        # so the no-log/no-uptime branches are exercised by calling it directly.
        from shrike.cli.server_cmd import _render_status

        status = _server()
        status.log = None
        status.uptime = None
        buf: list[str] = []
        with (
            patch(f"{SC}.output.kv", side_effect=lambda label, *a, **k: buf.append(label)),
            patch(f"{SC}.status_render"),
        ):
            _render_status(status)
        assert "Log" not in buf
        assert "Uptime" not in buf
        assert "URL" in buf  # the always-present rows still render

    def test_collections_table_rendered(self, run):
        # Multi-collection routing renders the per-collection table with the
        # default marker, idle/released/open states, the (boot) tag, and the
        # footnote about --profile.
        status = _server()
        status.collections = [
            CollectionStatus(
                name="main",
                path="/m.anki2",
                registered=True,
                is_default=True,
                active=True,
                held=True,
                index_state="ready",
            ),
            CollectionStatus(
                name="idle",
                path="/i.anki2",
                registered=True,
                is_default=False,
                active=False,
                held=None,
                index_state=None,
            ),
            CollectionStatus(
                name="released",
                path="/r.anki2",
                registered=True,
                is_default=False,
                active=True,
                held=False,
                index_state="ready",
            ),
            CollectionStatus(
                name="booting",
                path="/b.anki2",
                registered=False,
                is_default=False,
                active=True,
                held=True,
                index_state="building",
            ),
        ]
        fake = MagicMock()
        fake.server_status.return_value = status
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        assert "Collections" in out
        assert "main" in out and "idle" in out and "released" in out and "booting" in out
        assert "not opened" in out  # active=False state
        assert "(boot)" in out  # unregistered tag
        assert "--profile" in out


class TestServerStatusJsonAndMeta:
    def test_unresponsive_json(self, run):
        # Running but unresponsive, --json: emits running=True, responsive=False.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={"pid": 7, "url": "http://x"}),
        ):
            result = run("--json", "server", "status")
        import json

        data = json.loads(result.output)
        assert data["running"] is True and data["responsive"] is False
        assert data["pid"] == 7

    def test_unresponsive_renders_url_and_pid(self, run):
        # The pretty unresponsive branch prints both the URL and the PID rows.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={"pid": 7, "url": "http://x"}),
        ):
            result = run("server", "status")
        assert "URL:" in result.output and "http://x" in result.output
        assert "PID:" in result.output and "7" in result.output

    def test_unresponsive_meta_without_url_or_pid(self, run):
        # Partial meta (neither url nor pid) exercises the negative side of both
        # optional-row branches: the header prints, no URL/PID rows.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={}),
        ):
            result = run("server", "status")
        assert "not responding" in result.output
        assert "URL:" not in result.output
        assert "PID:" not in result.output

    def test_not_running_cleans_stale_meta_file(self, run):
        # META_FILE present but the daemon is dead → cleanup_state is called.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.META_FILE") as meta_file,
            patch(f"{SC}.cleanup_state") as cleanup,
        ):
            meta_file.exists.return_value = True
            result = run("server", "status")
        assert result.exit_code == 1
        cleanup.assert_called_once()


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

    def test_log_dir_resolved_from_server_meta(self, run, tmp_path):
        # With no get_log_file stub, the meta's log_dir drives resolution: a real
        # log file under the meta dir is read.
        log_dir = tmp_path / "metalogs"
        log_dir.mkdir()
        (log_dir / "shrike.log").write_text(_LOG)
        with patch(f"{SC}.read_server_meta", return_value={"log_dir": str(log_dir)}):
            result = run("server", "logs")
        assert "list_notes" in result.output

    def test_stdin_plain_skips_blank_lines(self, run):
        # The plain emit path drops blank/whitespace-only lines (767->exit arc).
        result = run("--no-pretty", "server", "logs", "--stdin", input=_LOG + "   \n\n")
        assert "list_notes" in result.output
        # Exactly one non-blank content line emitted.
        assert result.output.strip().count("list_notes") == 1

    def test_stdin_pretty_skips_blank_lines(self, run):
        # A blank line yields style_log_line() is None → not printed (the
        # styled-None arc); a valid line still renders.
        result = run("server", "logs", "--stdin", input="\n" + _LOG + "\n")
        assert "list_notes" in result.output

    def test_json_skips_unparseable_lines(self, run):
        # parse_log_line() returns None for a malformed line → it is dropped from
        # the messages array (776->774 arc), the valid one is kept.
        import json

        result = run("--json", "server", "logs", "--stdin", input="not a log\n" + _LOG)
        data = json.loads(result.output)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["logger"] == "shrike.tools"


class TestTailFollow:
    def test_open_error_raises_click_exception(self, run, tmp_path):
        from shrike.cli.server_cmd import _tail_follow

        missing = tmp_path / "nope" / "shrike.log"  # parent dir absent → OSError
        with pytest.raises(click.ClickException) as ei:
            _tail_follow(missing, 10, pretty=False)
        assert "Cannot read log file" in str(ei.value)

    def test_stream_without_fileno_uses_plain_sleep(self, run, tmp_path):
        # A stream lacking fileno() can't be select()'d, so the loop falls back to
        # a plain sleep (the no-fileno branch). A custom fake file drives it.
        from shrike.cli import server_cmd

        class _NoFilenoStream:
            def __init__(self) -> None:
                self._reads = [_LOG, ""]  # initial content, then nothing new

            def read(self) -> str:
                return self._reads.pop(0) if self._reads else ""

            def close(self) -> None:
                pass

            # Deliberately no fileno attribute.

        logfile = tmp_path / "shrike.log"
        logfile.write_text(_LOG)
        with (
            patch.object(server_cmd, "open", lambda *a, **k: _NoFilenoStream(), create=True),
            patch(f"{SC}.time.sleep", side_effect=[None, KeyboardInterrupt]),
        ):
            server_cmd._tail_follow(logfile, 1, pretty=False)
        # No exception escaped; the no-fileno sleep path was taken.

    def test_select_error_falls_back_to_sleep_then_emits_new_data(self, run, tmp_path):
        # A ValueError from select.select falls through to time.sleep; the sleep
        # appends a new line, which the subsequent read() emits — then a second
        # sleep raises KeyboardInterrupt to stop the loop.
        from shrike.cli.server_cmd import _tail_follow

        logfile = tmp_path / "shrike.log"
        logfile.write_text(_LOG)
        new_line = "2025-05-24T10:31:00 INFO  shrike.tools  appended_line ok\n"

        def sleeper(_secs):
            # 1st sleep: no append → the next read is empty (the loop-back arc).
            # 2nd sleep: append a line → the next read emits it (the new-data arc).
            # 3rd sleep: stop the follow loop.
            sleeper.calls += 1  # type: ignore[attr-defined]
            if sleeper.calls == 2:  # type: ignore[attr-defined]
                with open(logfile, "a") as fh:
                    fh.write(new_line)
                return None
            if sleeper.calls >= 3:  # type: ignore[attr-defined]
                raise KeyboardInterrupt
            return None

        sleeper.calls = 0  # type: ignore[attr-defined]

        buf: list[str] = []
        with (
            patch("select.select", side_effect=ValueError("bad fd")),
            patch(f"{SC}.time.sleep", side_effect=sleeper),
            patch(f"{SC}.click.echo", side_effect=lambda s="": buf.append(s)),
        ):
            _tail_follow(logfile, 1, pretty=False)
        # The appended line was read after the sleep and emitted (the new-data loop).
        assert any("appended_line" in line for line in buf)


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


class TestServerStartProfileAndV2:
    def _common(self, tmp_path, **overrides):
        defaults = dict(
            is_server_alive=MagicMock(return_value=False),
            cleanup_state=MagicMock(),
            get_log_file=MagicMock(return_value=tmp_path / "shrike.log"),
            STATE_DIR=tmp_path / "state",
        )
        defaults.update(overrides)
        return patch.multiple(SC, **defaults)

    def _run_cfg(self, tmp_path, cfg_text, *args, **kwargs):
        cfg = tmp_path / "config.yml"
        cfg.write_text(cfg_text)
        return CliRunner().invoke(cli, ["--config", str(cfg), *args], **kwargs)

    def test_profile_error_becomes_click_exception(self, tmp_path):
        from shrike.harness.profiles import ProfileError

        with (
            self._common(tmp_path),
            patch(f"{SC}.resolve_embedding_profile", side_effect=ProfileError("bad backend")),
        ):
            result = self._run_cfg(
                tmp_path,
                "",
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
            )
        assert result.exit_code != 0
        assert "bad backend" in result.output

    def test_v2_config_rejects_ocr_backend(self, tmp_path):
        # A v2 config (embedders:) plus --ocr-backend is the forbidden mix.
        cfg_text = "embedders:\n  text:\n    runtime: onnx\n"
        with (
            self._common(tmp_path),
            patch(f"{SC}.resolve_embedding_profile", return_value={}),
        ):
            result = self._run_cfg(
                tmp_path,
                cfg_text,
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--ocr-backend",
                "apple",
            )
        assert result.exit_code != 0
        assert "--ocr-backend is incompatible" in result.output

    def test_v2_config_passes_config_to_daemon(self, tmp_path):
        # A v2 config rides --config to the spawned daemon (the embedding args
        # branch); --no-embedding is appended.
        cfg_text = "embedders:\n  text:\n    runtime: onnx\n"
        proc = MagicMock(pid=123)
        captured_args: list[str] = []

        def fake_popen(args, **_):
            captured_args.extend(args)
            return proc

        with (
            self._common(tmp_path),
            patch(f"{SC}.resolve_embedding_profile", return_value={}),
            patch(f"{SC}.subprocess.Popen", side_effect=fake_popen),
            patch(f"{SC}._wait_for_server", return_value=_server()),
        ):
            result = self._run_cfg(
                tmp_path,
                cfg_text,
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--no-embedding",
            )
        assert result.exit_code == 0, result.output
        assert "--config" in captured_args
        assert "--no-embedding" in captured_args

    def test_v2_save_config_skips_legacy_embedding_section(self, tmp_path):
        # With a v2 config, --save-config must NOT write a legacy embedding:
        # section (the forbidden v2+legacy mix), and the JSON path suppresses the
        # "Config saved" advisory.
        cfg_text = "embedders:\n  text:\n    runtime: onnx\n"
        proc = MagicMock(pid=123)
        captured: dict = {}
        with (
            self._common(tmp_path),
            patch(f"{SC}.resolve_embedding_profile", return_value={"model": "/m"}),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=_server()),
            patch(f"{SC}.save_config", side_effect=lambda cfg, _p: captured.update(cfg) or "/cfg"),
        ):
            result = self._run_cfg(
                tmp_path,
                cfg_text,
                "--json",
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--save-config",
            )
        assert result.exit_code == 0, result.output
        # Under a v2 config, the resolved model is NOT bridged back into a legacy
        # embedding: section (that would create the forbidden v2+legacy mix).
        assert captured.get("embedding", {}).get("model") != "/m"
        # JSON output: no "Config saved" advisory line.
        assert "Config saved" not in result.output

    def test_save_config_persists_cooperative_lock(self, tmp_path):
        # The cooperative-lock + hold-seconds save-config branches.
        proc = MagicMock(pid=123)
        captured: dict = {}
        with (
            self._common(tmp_path),
            patch(f"{SC}.subprocess.Popen", return_value=proc),
            patch(f"{SC}._wait_for_server", return_value=_server()),
            patch(f"{SC}.save_config", side_effect=lambda cfg, _p: captured.update(cfg) or "/cfg"),
        ):
            result = self._run_cfg(
                tmp_path,
                "",
                "server",
                "start",
                "--collection",
                str(tmp_path / "c.anki2"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--cooperative-lock",
                "--lock-hold-seconds",
                "12",
                "--save-config",
            )
        assert result.exit_code == 0, result.output
        assert captured["server"]["cooperative_lock"] is True
        assert captured["server"]["lock_hold_seconds"] == 12


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
                patch("shrike.server.main", main),
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

    def test_server_main_is_eager_and_patchable(self):
        """``shrike.server.main`` is an eager attribute — a thin wrapper that defers the
        ``shrike.server.server`` import to call time — so it is always present on the
        package and ``mock.patch("shrike.server.main", ...)`` round-trips deterministically.

        With no lazy ``__getattr__`` fallback, a patch can only resolve ``main`` from the
        real ``__dict__`` entry, so ``unittest.mock`` records ``local=True`` and restores
        via ``setattr`` — never the ``delattr`` restore path (taken when a name resolves
        only through ``__getattr__``) that would drop ``main`` for the rest of the worker."""
        # ``shrike.server`` is imported at module top, so ``main`` is already bound here.
        # Attribute access, not ``__dict__`` membership: the contract is "``main`` is a
        # present, callable attribute", and access is what ``mock.patch`` exercises.
        assert not hasattr(shrike.server, "__getattr__"), "no lazy fallback to delattr-trap"
        assert callable(shrike.server.main)
        for _ in range(3):  # a patch round-trip must leave ``main`` callable, repeatably
            with patch("shrike.server.main", MagicMock()) as m:
                assert shrike.server.main is m
            assert callable(shrike.server.main)  # restored to the real wrapper


class TestGlobalStateDirWiring:
    def test_state_dir_threads_into_autostart_spec(self, tmp_path: Path) -> None:
        # A global --state-dir must reach the auto-start spec, not just control
        # discovery, so a daemon spawned on connection failure writes server.json
        # where the client then looks (otherwise control routes 404 on a mismatch).
        cfg = tmp_path / "config.yml"
        cfg.write_text("collection: /c.anki2\n")
        custom = tmp_path / "custom-state"
        status_client = MagicMock()
        status_client.server_status.return_value = _server()
        with (
            patch("shrike.client.ShrikeClient", return_value=MagicMock()) as root_ctor,
            patch(f"{SC}.ShrikeClient", return_value=status_client),
        ):
            result = CliRunner().invoke(
                cli, ["--config", str(cfg), "--state-dir", str(custom), "server", "status"]
            )
        assert result.exit_code == 0
        _, kwargs = root_ctor.call_args
        assert kwargs["state_dir"] == custom
        assert kwargs["spec"].state_dir == str(custom)
