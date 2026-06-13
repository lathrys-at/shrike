"""CLI behavior for `shrike profile` (#66, slice 1).

These commands are pure config operations — they never reach the server — so
the tests drive the real CLI against an isolated --config file and assert on
the persisted registry.
"""

from __future__ import annotations

import os

from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli.config import load_config
from shrike.registry import Registry


def _run(tmp_path, args, **kwargs):
    config_path = tmp_path / "config.yml"
    result = CliRunner().invoke(cli, ["--config", str(config_path), "profile", *args], **kwargs)
    return result, config_path


def _registry(config_path):
    return Registry.from_config(load_config(config_path))


class TestProfileAdd:
    def test_add_persists_and_normalizes(self, tmp_path):
        result, cfg = _run(tmp_path, ["add", "work", "~/decks/work.anki2"])
        assert result.exit_code == 0, result.output
        reg = _registry(cfg)
        assert reg.names() == ["work"]
        assert reg.get("work").path == os.path.abspath(os.path.expanduser("~/decks/work.anki2"))
        # First profile is the implicit default.
        assert reg.default == "work"

    def test_add_default_flag(self, tmp_path):
        _run(tmp_path, ["add", "work", "/a/work.anki2"])
        result, cfg = _run(tmp_path, ["add", "home", "/a/home.anki2", "--default"])
        assert result.exit_code == 0, result.output
        assert _registry(cfg).default == "home"

    def test_add_duplicate_errors(self, tmp_path):
        _run(tmp_path, ["add", "work", "/a/work.anki2"])
        result, cfg = _run(tmp_path, ["add", "work", "/b/work.anki2"])
        assert result.exit_code != 0
        assert "already registered" in result.output.lower()
        # Original path unchanged.
        assert _registry(cfg).get("work").path == "/a/work.anki2"

    def test_add_json(self, tmp_path):
        result, _ = _run(tmp_path, ["add", "work", "/a/work.anki2", "--json"])
        assert result.exit_code == 0, result.output
        assert '"name": "work"' in result.output


class TestProfileRemove:
    def test_remove_persists(self, tmp_path):
        _run(tmp_path, ["add", "work", "/a/work.anki2"])
        _run(tmp_path, ["add", "home", "/a/home.anki2"])
        result, cfg = _run(tmp_path, ["remove", "work"])
        assert result.exit_code == 0, result.output
        assert _registry(cfg).names() == ["home"]
        # Sole survivor became the default.
        assert _registry(cfg).default == "home"

    def test_remove_unknown_errors(self, tmp_path):
        result, _ = _run(tmp_path, ["remove", "ghost"])
        assert result.exit_code != 0
        assert "not registered" in result.output.lower()


class TestProfileDefault:
    def test_default_switches(self, tmp_path):
        _run(tmp_path, ["add", "work", "/a/work.anki2"])
        _run(tmp_path, ["add", "home", "/a/home.anki2"])
        result, cfg = _run(tmp_path, ["default", "home"])
        assert result.exit_code == 0, result.output
        assert _registry(cfg).default == "home"

    def test_default_unknown_errors(self, tmp_path):
        _run(tmp_path, ["add", "work", "/a/work.anki2"])
        result, _ = _run(tmp_path, ["default", "ghost"])
        assert result.exit_code != 0
        assert "not registered" in result.output.lower()


class TestProfileList:
    def test_list_empty(self, tmp_path):
        result, _ = _run(tmp_path, ["list"])
        assert result.exit_code == 0, result.output
        assert "no profiles registered" in result.output.lower()

    def test_list_marks_default(self, tmp_path):
        _run(tmp_path, ["add", "work", "/a/work.anki2"])
        _run(tmp_path, ["add", "home", "/a/home.anki2", "--default"])
        result, _ = _run(tmp_path, ["list"])
        assert result.exit_code == 0, result.output
        assert "work" in result.output
        assert "home" in result.output
        assert "active default" in result.output.lower()

    def test_list_json(self, tmp_path):
        _run(tmp_path, ["add", "work", "/a/work.anki2"])
        result, _ = _run(tmp_path, ["list", "--json"])
        assert result.exit_code == 0, result.output
        assert '"default": "work"' in result.output
        assert '"name": "work"' in result.output
