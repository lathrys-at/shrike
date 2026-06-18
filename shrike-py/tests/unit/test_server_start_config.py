"""`shrike server start` config persistence.

``server start`` is a no-write operation by default, with an explicit
``--save-config`` opt-in. Writing ``config.yml`` on first run but silently
ignoring later flags would leave the on-disk config stale — a different
``--collection`` changes the running daemon while a subsequent no-flag start
reads the *old* collection — so start never writes unless asked.

These tests drive the real ``server start`` command with the daemon spawn mocked
out, so only the config-writing decision is exercised.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from shrike.cli import cli


@pytest.fixture
def mocked_daemon(monkeypatch, tmp_path):
    """Stub the daemon spawn so ``server start`` runs without a real server.

    Redirects the state dir into the temp tree, reports no running server, makes
    ``Popen`` a no-op, and has the readiness probe return None (the "started but
    not yet responding" path) — enough to reach and pass the config-save block.
    """
    monkeypatch.setattr("shrike.cli.server_cmd.STATE_DIR", tmp_path / "state")
    monkeypatch.setattr("shrike.cli.server_cmd.is_server_alive", lambda: False)
    monkeypatch.setattr("shrike.cli.server_cmd._wait_for_server", lambda url: None)

    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 4242
    monkeypatch.setattr("shrike.cli.server_cmd.subprocess.Popen", lambda *a, **k: proc)
    return proc


def _start(runner: CliRunner, config_path: Path, *extra: str) -> object:
    collection = config_path.parent / "collection.anki2"
    log_dir = config_path.parent / "logs"
    return runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "server",
            "start",
            "--collection",
            str(collection),
            "--log-dir",
            str(log_dir),
            *extra,
        ],
    )


def test_no_save_config_flag_writes_nothing(mocked_daemon, tmp_path):
    """A plain `server start` never creates a config file (no hidden first-run write)."""
    config_path = tmp_path / "config.yml"
    result = _start(CliRunner(), config_path)

    assert result.exit_code == 0, result.output
    assert not config_path.exists()


def test_save_config_flag_persists_collection(mocked_daemon, tmp_path):
    """`--save-config` writes the resolved collection to config.yml."""
    config_path = tmp_path / "config.yml"
    result = _start(CliRunner(), config_path, "--save-config")

    assert result.exit_code == 0, result.output
    assert config_path.exists()
    saved = yaml.safe_load(config_path.read_text())
    assert saved["collection"] == str(tmp_path / "collection.anki2")


def test_no_save_config_flag_preserves_existing_config(mocked_daemon, tmp_path):
    """Starting on a new collection without `--save-config` must not rewrite an
    existing config that points at the old one."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"collection": "/old/A.anki2"}))

    result = _start(CliRunner(), config_path)

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(config_path.read_text())
    assert saved["collection"] == "/old/A.anki2"


def test_save_config_flag_overwrites_existing_config(mocked_daemon, tmp_path):
    """`--save-config` brings a stale config back in line with the flags given."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"collection": "/old/A.anki2"}))

    result = _start(CliRunner(), config_path, "--save-config")

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(config_path.read_text())
    assert saved["collection"] == str(tmp_path / "collection.anki2")
