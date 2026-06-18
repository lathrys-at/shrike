"""CLI behavior for `shrike profile`.

These commands are pure config operations — they never reach the server — so
the tests drive the real CLI against an isolated --config file and assert on
the persisted registry. `--discover` points ANKI_BASE at a synthesized
prefs21.db so the Anki-base-dir scan is exercised end-to-end.
"""

from __future__ import annotations

import os
import pickle
import sqlite3

from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli.config import load_config
from shrike.harness.registry import Registry


def _run(tmp_path, args, **kwargs):
    config_path = tmp_path / "config.yml"
    result = CliRunner().invoke(cli, ["--config", str(config_path), "profile", *args], **kwargs)
    return result, config_path


def _registry(config_path):
    return Registry.from_config(load_config(config_path))


def _write_anki_base(base, names, *, make_collections=()):
    """Synthesize <base>/prefs21.db (Anki's schema + a _global row)."""
    base.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(base / "prefs21.db")
    try:
        conn.execute(
            "create table if not exists profiles "
            "(name text primary key collate nocase, data blob not null)"
        )
        conn.execute(
            "insert or replace into profiles values ('_global', ?)",
            (pickle.dumps({}, protocol=4),),
        )
        for name in names:
            conn.execute(
                "insert or replace into profiles values (?, ?)",
                (name, pickle.dumps({}, protocol=4)),
            )
        conn.commit()
    finally:
        conn.close()
    for name in make_collections:
        (base / name).mkdir(parents=True, exist_ok=True)
        (base / name / "collection.anki2").write_bytes(b"SQLite format 3\x00")


class TestProfileCreate:
    def test_create_persists_and_normalizes(self, tmp_path):
        result, cfg = _run(tmp_path, ["create", "work", "~/decks/work.anki2"])
        assert result.exit_code == 0, result.output
        reg = _registry(cfg)
        assert reg.names() == ["work"]
        assert reg.get("work").path == os.path.abspath(os.path.expanduser("~/decks/work.anki2"))
        # First profile is the implicit default.
        assert reg.default == "work"

    def test_create_default_flag(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        result, cfg = _run(tmp_path, ["create", "home", "/a/home.anki2", "--default"])
        assert result.exit_code == 0, result.output
        assert _registry(cfg).default == "home"

    def test_create_duplicate_errors(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        result, cfg = _run(tmp_path, ["create", "work", "/b/work.anki2"])
        assert result.exit_code != 0
        assert "already registered" in result.output.lower()
        # Original path unchanged.
        assert _registry(cfg).get("work").path == "/a/work.anki2"

    def test_create_json(self, tmp_path):
        result, _ = _run(tmp_path, ["create", "work", "/a/work.anki2", "--json"])
        assert result.exit_code == 0, result.output
        assert '"name": "work"' in result.output


class TestProfileRename:
    def test_rename_persists_and_preserves(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        _run(tmp_path, ["create", "home", "/a/home.anki2"])
        result, cfg = _run(tmp_path, ["rename", "work", "job"])
        assert result.exit_code == 0, result.output
        reg = _registry(cfg)
        # In-place: list order preserved, path carried across.
        assert reg.names() == ["job", "home"]
        assert reg.get("job").path == "/a/work.anki2"
        # Default (the first-registered "work") follows the rename.
        assert reg.default == "job"

    def test_rename_unknown_errors(self, tmp_path):
        result, _ = _run(tmp_path, ["rename", "ghost", "x"])
        assert result.exit_code != 0
        assert "not registered" in result.output.lower()

    def test_rename_to_taken_errors(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        _run(tmp_path, ["create", "home", "/a/home.anki2"])
        result, cfg = _run(tmp_path, ["rename", "work", "home"])
        assert result.exit_code != 0
        assert "already registered" in result.output.lower()
        # Both unchanged.
        assert _registry(cfg).names() == ["work", "home"]

    def test_rename_json(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        result, _ = _run(tmp_path, ["rename", "work", "job", "--json"])
        assert result.exit_code == 0, result.output
        assert '"renamed": "job"' in result.output


class TestProfileDelete:
    def test_delete_persists(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        _run(tmp_path, ["create", "home", "/a/home.anki2"])
        result, cfg = _run(tmp_path, ["delete", "work"])
        assert result.exit_code == 0, result.output
        assert _registry(cfg).names() == ["home"]
        # Sole survivor became the default.
        assert _registry(cfg).default == "home"

    def test_delete_unknown_errors(self, tmp_path):
        result, _ = _run(tmp_path, ["delete", "ghost"])
        assert result.exit_code != 0
        assert "not registered" in result.output.lower()


class TestProfileDefault:
    def test_default_switches(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        _run(tmp_path, ["create", "home", "/a/home.anki2"])
        result, cfg = _run(tmp_path, ["default", "home"])
        assert result.exit_code == 0, result.output
        assert _registry(cfg).default == "home"

    def test_default_unknown_errors(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        result, _ = _run(tmp_path, ["default", "ghost"])
        assert result.exit_code != 0
        assert "not registered" in result.output.lower()


class TestProfileList:
    def test_list_empty(self, tmp_path):
        result, _ = _run(tmp_path, ["list"])
        assert result.exit_code == 0, result.output
        assert "no profiles registered" in result.output.lower()

    def test_list_marks_default(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        _run(tmp_path, ["create", "home", "/a/home.anki2", "--default"])
        result, _ = _run(tmp_path, ["list"])
        assert result.exit_code == 0, result.output
        assert "work" in result.output
        assert "home" in result.output
        assert "active default" in result.output.lower()

    def test_list_json(self, tmp_path):
        _run(tmp_path, ["create", "work", "/a/work.anki2"])
        result, _ = _run(tmp_path, ["list", "--json"])
        assert result.exit_code == 0, result.output
        assert '"default": "work"' in result.output
        assert '"name": "work"' in result.output


class TestProfileListDiscover:
    def test_discover_lists_anki_profiles(self, tmp_path, monkeypatch):
        anki = tmp_path / "Anki2"
        _write_anki_base(anki, ["User 1", "Work"], make_collections=["User 1"])
        monkeypatch.setenv("ANKI_BASE", str(anki))
        result, _ = _run(tmp_path, ["list", "--discover"])
        assert result.exit_code == 0, result.output
        assert "User 1" in result.output
        assert "Work" in result.output
        # The profile with no collection file on disk is flagged missing.
        assert "missing" in result.output.lower()

    def test_discover_annotates_registered_by_path(self, tmp_path, monkeypatch):
        anki = tmp_path / "Anki2"
        _write_anki_base(anki, ["Work"], make_collections=["Work"])
        monkeypatch.setenv("ANKI_BASE", str(anki))
        # Register the same collection under a *different* friendly name —
        # membership is path-based, so it still reads as registered.
        coll = str(anki / "Work" / "collection.anki2")
        _run(tmp_path, ["create", "myhandle", coll])
        result, _ = _run(tmp_path, ["list", "--discover"])
        assert result.exit_code == 0, result.output
        assert "registered" in result.output.lower()

    def test_discover_empty_when_no_base(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANKI_BASE", str(tmp_path / "absent"))
        result, _ = _run(tmp_path, ["list", "--discover"])
        assert result.exit_code == 0, result.output
        assert "no anki profiles found" in result.output.lower()

    def test_discover_json(self, tmp_path, monkeypatch):
        anki = tmp_path / "Anki2"
        _write_anki_base(anki, ["Work"], make_collections=["Work"])
        monkeypatch.setenv("ANKI_BASE", str(anki))
        result, _ = _run(tmp_path, ["list", "--discover", "--json"])
        assert result.exit_code == 0, result.output
        assert '"name": "Work"' in result.output
        assert '"exists": true' in result.output
        assert '"registered": false' in result.output
        assert '"base_dir"' in result.output
