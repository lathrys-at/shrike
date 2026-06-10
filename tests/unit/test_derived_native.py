"""Native derived-text engine (#281): cross-engine interop + always-available FTS5.

The behavioural parity gate is test_derived.py run with SHRIKE_NATIVE_DERIVED=1
(it passes unmodified — CI's gated native lane). This file pins the rest: a
``shrike.db`` written by either engine opens and searches identically under the
other (plain SQLite file compatibility), and the native path never needs the
FTS5 availability probe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from shrike.derived import DerivedTextStore

requires_shrike_native = pytest.mark.skipif(
    importlib.util.find_spec("shrike_native") is None,
    reason="shrike_native extension not installed (scripts/build-native.sh)",
)

ROWS = [
    (1, "field", "Front", "the mitochondria is the powerhouse of the cell"),
    (1, "field", "Back", "cellular respiration produces ATP"),
    (2, "field", "Front", "die Hauptstadt von Österreich ist Wien"),
]


def _assert_store_serves(store: DerivedTextStore) -> None:
    assert store.available is True
    assert store.size == 3
    hits = store.search_substring("mitochondria")
    assert hits is not None and hits[0].note_id == 1
    fuzzy = store.search_fuzzy("mitochondira")  # transposition typo
    assert any(nid == 1 for nid, _ in fuzzy)


@requires_shrike_native
class TestNativeDerivedEngine:
    def test_native_engine_selected_and_serves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shrike.derived import NativeDerivedEngine

        monkeypatch.setenv("SHRIKE_NATIVE_DERIVED", "1")
        store = DerivedTextStore(path=tmp_path / "shrike.db")
        assert isinstance(store._engine, NativeDerivedEngine)
        store.build(ROWS, col_mod=100)
        _assert_store_serves(store)
        store.close()

    def test_python_written_db_opens_under_native(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = DerivedTextStore(path=tmp_path / "shrike.db")
        store.build(ROWS, col_mod=100)
        store.close()

        monkeypatch.setenv("SHRIKE_NATIVE_DERIVED", "1")
        reopened = DerivedTextStore(path=tmp_path / "shrike.db")
        assert reopened.col_mod == 100  # no rebuild needed across the engine switch
        assert reopened.check_drift(100) is False
        _assert_store_serves(reopened)
        reopened.close()

    def test_native_written_db_opens_under_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SHRIKE_NATIVE_DERIVED", "1")
        store = DerivedTextStore(path=tmp_path / "shrike.db")
        store.build(ROWS, col_mod=42)
        store.close()

        monkeypatch.delenv("SHRIKE_NATIVE_DERIVED")
        from shrike.derived import SqliteDerivedEngine

        reopened = DerivedTextStore(path=tmp_path / "shrike.db")
        assert isinstance(reopened._engine, SqliteDerivedEngine)
        assert reopened.col_mod == 42
        assert reopened.check_drift(42) is False
        _assert_store_serves(reopened)
        reopened.close()

    def test_probe_is_constant_true_on_native_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHRIKE_NATIVE_DERIVED", "1")
        assert DerivedTextStore._probe_fts5() is True

    def test_missing_extension_degrades_to_stdlib_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shrike import derived

        monkeypatch.setenv("SHRIKE_NATIVE_DERIVED", "1")
        monkeypatch.setattr(
            derived.NativeDerivedEngine,
            "__init__",
            lambda self, path: (_ for _ in ()).throw(ImportError("not installed")),
        )
        store = DerivedTextStore(path=tmp_path / "shrike.db")
        assert isinstance(store._engine, derived.SqliteDerivedEngine)
        store.close()
