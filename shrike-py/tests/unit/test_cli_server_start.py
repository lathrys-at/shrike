"""Unit coverage for the `shrike server start` CLI command (incl. profiles/v2,
the wait-for-server poll, and global state-dir wiring).

These manage the daemon: spawning and waiting for it to come up. Driven with
mocked daemon helpers / client / subprocess so no real server is spawned. Patches
target `shrike.cli.server_cmd.*` (where the names are bound).
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
import pytest
from click.testing import CliRunner

import shrike.server
from shrike.cli import cli
from shrike.cli.server_cmd import _wait_for_server
from shrike.schemas import (
    EmbeddingRunning,
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
