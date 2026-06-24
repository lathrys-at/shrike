"""Unit coverage for the `shrike server logs` CLI command (incl. --follow).

Driven with mocked daemon helpers / subprocess so no real server is spawned.
Patches target `shrike.cli.server_cmd.*` (where the names are bound).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click
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
