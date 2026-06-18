"""Native derived-text engine: on-disk round-trip + always-available FTS5.

The behavioural gate is test_derived.py, which runs entirely on the native
engine. This file pins the rest: a ``shrike.db`` written by one instance opens
and searches identically under a fresh one (plain SQLite file compatibility),
and the bundled-SQLite build never fails the FTS5 probe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from shrike.harness.derived import DerivedTextStore

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
    def test_native_engine_selected_and_serves(self, tmp_path: Path) -> None:
        from shrike.harness.derived import NativeDerivedEngine

        store = DerivedTextStore(path=tmp_path / "shrike.db")
        assert isinstance(store._engine, NativeDerivedEngine)
        store.build(ROWS, col_mod=100)
        _assert_store_serves(store)
        store.close()

    def test_on_disk_db_round_trips(self, tmp_path: Path) -> None:
        store = DerivedTextStore(path=tmp_path / "shrike.db")
        store.build(ROWS, col_mod=100)
        store.close()

        reopened = DerivedTextStore(path=tmp_path / "shrike.db")
        assert reopened.col_mod == 100  # no rebuild needed across a reopen
        assert reopened.check_drift(100) is False
        _assert_store_serves(reopened)
        reopened.close()

    def test_probe_passes_on_native_path(self) -> None:
        # Under the default (bundled-SQLite) build the native probe must pass.
        # A platform-linked build probes the host library instead; this dev/CI
        # build is the bundled one.
        import shrike_native

        assert DerivedTextStore._probe_fts5() is True
        assert shrike_native.derived_fts5_probe() is True

    def test_bundled_flag_matches_probe_guarantee(self) -> None:
        # Diagnostics surface: when the build claims bundled SQLite, the probe
        # may never fail; otherwise it merely reports the host library.
        import shrike_native

        if shrike_native.derived_sqlite_bundled():
            assert shrike_native.derived_fts5_probe() is True
