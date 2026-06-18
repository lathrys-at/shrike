"""Anki-base-dir discovery: base-dir resolution + prefs21.db reading.

The fixtures synthesize a real prefs21.db with Anki's exact schema (the
``profiles(name, data)`` table, a ``_global`` meta row, pickled blobs) so the
reader is exercised against the on-disk shape it will meet in the field, not a
mock.
"""

from __future__ import annotations

import pickle
import sqlite3
import sys

import pytest

from shrike.platform.paths import (
    AnkiProfile,
    anki_base_dir,
    anki_prefs_db,
    discover_anki_profiles,
)


def _write_prefs_db(base, names, *, make_collections=()):
    """Create <base>/prefs21.db with Anki's schema and the given profile names.

    A ``_global`` meta row is always written (the reader must exclude it). For
    each name in ``make_collections`` a real <base>/<name>/collection.anki2 file
    is created so the ``exists`` flag can be exercised both ways.
    """
    base.mkdir(parents=True, exist_ok=True)
    db = base / "prefs21.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "create table if not exists profiles "
            "(name text primary key collate nocase, data blob not null)"
        )
        conn.execute(
            "insert or replace into profiles values ('_global', ?)",
            (pickle.dumps({"firstRun": False}, protocol=4),),
        )
        for name in names:
            conn.execute(
                "insert or replace into profiles values (?, ?)",
                (name, pickle.dumps({"syncKey": None}, protocol=4)),
            )
        conn.commit()
    finally:
        conn.close()
    for name in make_collections:
        folder = base / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "collection.anki2").write_bytes(b"SQLite format 3\x00")
    return db


class TestAnkiBaseDir:
    def test_anki_base_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANKI_BASE", str(tmp_path / "custom"))
        assert anki_base_dir() == tmp_path / "custom"

    def test_anki_base_env_override_expands_user(self, monkeypatch):
        monkeypatch.setenv("ANKI_BASE", "~/my-anki")
        result = anki_base_dir()
        assert "~" not in str(result)
        assert result.name == "my-anki"

    def test_platform_default_no_override(self, monkeypatch):
        monkeypatch.delenv("ANKI_BASE", raising=False)
        result = anki_base_dir()
        # Always ends in Anki2, and matches the running platform's convention.
        assert result.name == "Anki2"
        if sys.platform == "darwin":
            assert "Library/Application Support" in str(result)
        elif sys.platform.startswith("win"):
            assert result.parent.name.lower() != ""  # %APPDATA% or home
        else:
            # Linux: XDG_DATA_HOME or ~/.local/share
            assert ".local/share" in str(result) or "XDG" not in str(result)

    def test_linux_respects_xdg_data_home(self, tmp_path, monkeypatch):
        if sys.platform == "darwin" or sys.platform.startswith("win"):
            pytest.skip("XDG only governs the Linux/other branch")
        monkeypatch.delenv("ANKI_BASE", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        assert anki_base_dir() == tmp_path / "xdg" / "Anki2"

    def test_prefs_db_path(self, tmp_path):
        assert anki_prefs_db(tmp_path) == tmp_path / "prefs21.db"


class TestDiscoverAnkiProfiles:
    def test_missing_base_dir_is_empty(self, tmp_path):
        assert discover_anki_profiles(tmp_path / "nope") == []

    def test_base_dir_without_prefs_is_empty(self, tmp_path):
        (tmp_path / "Anki2").mkdir()
        assert discover_anki_profiles(tmp_path / "Anki2") == []

    def test_reads_profiles_excludes_global(self, tmp_path):
        base = tmp_path / "Anki2"
        _write_prefs_db(base, ["User 1", "Work"], make_collections=["User 1"])
        profiles = discover_anki_profiles(base)
        names = [p.name for p in profiles]
        assert "_global" not in names
        assert names == ["User 1", "Work"]  # ordered by name

    def test_collection_path_is_folder_per_profile(self, tmp_path):
        base = tmp_path / "Anki2"
        _write_prefs_db(base, ["Work"])
        (profile,) = discover_anki_profiles(base)
        assert profile.collection_path == str(base / "Work" / "collection.anki2")

    def test_exists_flag(self, tmp_path):
        base = tmp_path / "Anki2"
        _write_prefs_db(base, ["Here", "Gone"], make_collections=["Here"])
        by_name = {p.name: p for p in discover_anki_profiles(base)}
        assert by_name["Here"].exists is True
        assert by_name["Gone"].exists is False

    def test_uses_default_base_when_none(self, tmp_path, monkeypatch):
        base = tmp_path / "Anki2"
        _write_prefs_db(base, ["Default"])
        monkeypatch.setenv("ANKI_BASE", str(base))
        profiles = discover_anki_profiles()  # no explicit base → ANKI_BASE
        assert [p.name for p in profiles] == ["Default"]

    def test_corrupt_db_degrades_to_empty(self, tmp_path):
        base = tmp_path / "Anki2"
        base.mkdir()
        # A file named prefs21.db that isn't a SQLite DB → sqlite3.Error, caught.
        (base / "prefs21.db").write_text("not a database")
        assert discover_anki_profiles(base) == []

    def test_wrong_schema_degrades_to_empty(self, tmp_path):
        base = tmp_path / "Anki2"
        base.mkdir()
        conn = sqlite3.connect(base / "prefs21.db")
        conn.execute("create table other (x int)")
        conn.commit()
        conn.close()
        # No `profiles` table → the SELECT raises, caught → empty.
        assert discover_anki_profiles(base) == []

    def test_opens_read_only(self, tmp_path):
        """The reader must not create or write the file. A non-existent prefs is
        empty (no file created); an existing one is left byte-identical."""
        base = tmp_path / "Anki2"
        base.mkdir()
        # No prefs file: discovery must not create one (mode=ro).
        discover_anki_profiles(base)
        assert not (base / "prefs21.db").exists()
        # Existing file: unchanged after a read.
        db = _write_prefs_db(base, ["A"])
        before = db.read_bytes()
        discover_anki_profiles(base)
        assert db.read_bytes() == before

    def test_returns_anki_profile_instances(self, tmp_path):
        base = tmp_path / "Anki2"
        _write_prefs_db(base, ["X"])
        (p,) = discover_anki_profiles(base)
        assert isinstance(p, AnkiProfile)
