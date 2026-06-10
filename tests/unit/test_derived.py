"""Unit tests for the derived-text store (#98) — the FTS5 trigram sidecar.

These run against real SQLite (FTS5 + the trigram tokenizer ship with CPython's sqlite3 on every
platform we target). They exercise the store directly: build, substring + fuzzy lookups with their
source/ref/snippet provenance, the multi-source seam (#199's future ``ocr``/``asr`` slot), the
incremental ingest/remove path, drift detection + the col_mod watermark, and graceful degradation
when FTS5 is unavailable.
"""

from __future__ import annotations

import pytest

from shrike.derived import DerivedTextStore
from shrike.index import IndexState

# (note_id, source, ref, text) — the build-row shape (CollectionWrapper.derived_field_rows).
ROWS = [
    (1, "field", "Front", "Mitochondria are the powerhouse of the cell"),
    (1, "field", "Back", "They generate ATP"),
    (2, "field", "Front", "Ribosomes synthesize protein"),
    (3, "field", "Front", "The Krebs cycle"),
]


@pytest.fixture()
def store(tmp_path):
    s = DerivedTextStore(path=tmp_path / "shrike.db")
    assert s.available is False  # not ready until a build stamps col_mod
    s.build(ROWS, col_mod=100)
    yield s
    s.close()


class TestBuild:
    def test_available_after_build(self, store):
        assert store.available is True
        assert store.state == IndexState.READY
        assert store.col_mod == 100
        # One row per non-empty (note, source, ref); note 1 has two fields.
        assert store.size == 4

    def test_build_skips_blank_text(self, tmp_path):
        s = DerivedTextStore(path=tmp_path / "shrike.db")
        s.build([(1, "field", "Front", "  "), (1, "field", "Back", "real")], col_mod=1)
        assert s.size == 1  # the blank field is not indexed
        s.close()


class TestSubstring:
    def test_match_carries_source_ref_snippet(self, store):
        hits = store.search_substring("powerhouse")
        assert hits is not None
        assert len(hits) == 1
        m = hits[0]
        assert m.note_id == 1
        assert m.source == "field"
        assert m.ref == "Front"
        assert "powerhouse" in (m.snippet or "")

    def test_case_insensitive(self, store):
        assert [m.note_id for m in store.search_substring("MITOCHONDRIA")] == [1]

    def test_contiguous_only(self, store):
        # A phrase match is a contiguous substring: "powerhouse cell" is not literally present.
        assert store.search_substring("powerhouse cell") == []

    def test_subtrigram_query_returns_none(self, store):
        # < 3 chars: FTS5 trigram can't match it → None tells the caller to use the find_notes
        # fallback (not [] — that would wrongly read as "no hits").
        assert store.search_substring("xy") is None

    def test_unavailable_store_returns_none(self, tmp_path):
        s = DerivedTextStore(path=tmp_path / "shrike.db")  # never built
        assert s.search_substring("anything") is None
        s.close()

    def test_limit_caps_results(self, tmp_path):
        s = DerivedTextStore(path=tmp_path / "shrike.db")
        s.build([(i, "field", "F", f"shared token n{i}") for i in range(10)], col_mod=1)
        assert len(s.search_substring("shared", limit=3)) == 3
        s.close()


class TestFuzzy:
    def test_typo_matches_intended_note(self, store):
        # "mitochndria" (dropped 'o') still shares most trigrams with note 1.
        hits = store.search_fuzzy("mitochndria")
        assert [nid for nid, _ in hits] == [1]
        assert hits[0][1].source == "field"
        assert hits[0][1].ref == "Front"

    def test_typo_picks_the_right_note(self, store):
        # "protien" (transposed) → the protein note, not the mitochondria one.
        assert [nid for nid, _ in store.search_fuzzy("protien")] == [2]

    def test_one_shared_trigram_is_noise_floored(self, store):
        # A query sharing only one trigram with any note must not surface it (FUZZY_MIN_SHARED).
        assert store.search_fuzzy("zzkrx") == []

    def test_dedup_one_row_per_note(self, store):
        # Note 1 has two fields; a fuzzy query matching both yields one (best) row for the note.
        hits = store.search_fuzzy("mitochondria")
        assert [nid for nid, _ in hits].count(1) == 1

    def test_unavailable_returns_empty(self, tmp_path):
        s = DerivedTextStore(path=tmp_path / "shrike.db")  # never built → graceful empty
        assert s.search_fuzzy("mitochondria") == []
        s.close()


class TestSourceSeam:
    """A second source under one note id is searchable + removable independently (the #199 seam)."""

    def test_second_source_searchable_with_its_provenance(self, store):
        store.ingest(1, "ocr", {"diagram.png": "cristae folds visible in the image"})
        hits = store.search_substring("cristae")
        assert hits is not None and len(hits) == 1
        assert hits[0].note_id == 1
        assert hits[0].source == "ocr"
        assert hits[0].ref == "diagram.png"

    def test_remove_one_source_leaves_the_other(self, store):
        store.ingest(1, "ocr", {"diagram.png": "cristae folds"})
        store.remove([1], source="ocr")
        assert store.search_substring("cristae") == []  # ocr source gone
        assert [m.note_id for m in store.search_substring("powerhouse")] == [1]  # field stays

    def test_ingest_replaces_only_its_own_source(self, store):
        # Re-ingesting "field" for note 1 replaces its field rows but never touches an ocr source.
        store.ingest(1, "ocr", {"diagram.png": "cristae"})
        store.ingest(1, "field", {"Front": "Updated mitochondria text"})
        assert [m.note_id for m in store.search_substring("cristae")] == [1]
        assert store.search_substring("powerhouse") == []  # old field text replaced
        assert [m.note_id for m in store.search_substring("Updated")] == [1]


class TestIncremental:
    def test_ingest_adds_a_note(self, store):
        store.ingest(4, "field", {"Front": "Newly authored card"})
        assert [m.note_id for m in store.search_substring("authored")] == [4]

    def test_ingest_is_idempotent_replace(self, store):
        store.ingest(1, "field", {"Front": "first revision"})
        store.ingest(1, "field", {"Front": "second revision"})
        assert store.search_substring("first") == []
        assert [m.note_id for m in store.search_substring("second")] == [1]

    def test_remove_drops_all_sources(self, store):
        store.ingest(1, "ocr", {"x.png": "cristae"})
        store.remove([1])
        assert store.search_substring("powerhouse") == []
        assert store.search_substring("cristae") == []


class TestDriftAndColMod:
    def test_no_drift_at_built_mod(self, store):
        assert store.check_drift(100) is False

    def test_drift_when_col_mod_moves(self, store):
        assert store.check_drift(200) is True

    def test_setter_advances_watermark_without_rebuild(self, store):
        # An incremental edit / metadata bump advances the watermark in place so the next boot
        # sees no drift (and doesn't re-ingest the whole collection).
        store.col_mod = 200
        assert store.col_mod == 200
        assert store.check_drift(200) is False

    def test_never_built_drifts(self, tmp_path):
        s = DerivedTextStore(path=tmp_path / "shrike.db")
        assert s.check_drift(1) is True  # col_mod is None → needs a build
        s.close()


class TestPersistence:
    def test_reopen_loads_built_state(self, tmp_path):
        path = tmp_path / "shrike.db"
        s1 = DerivedTextStore(path=path)
        s1.build(ROWS, col_mod=100)
        s1.close()

        s2 = DerivedTextStore(path=path)
        assert s2.available is True
        assert s2.col_mod == 100
        assert [m.note_id for m in s2.search_substring("powerhouse")] == [1]
        s2.close()


class TestCorruptRecovery:
    """A corrupt/unreadable sidecar is never fatal (review F1) — it's recreated, not crashed on."""

    def test_garbage_file_is_recreated(self, tmp_path):
        path = tmp_path / "shrike.db"
        path.write_bytes(b"this is not a sqlite database at all, just junk bytes" * 50)
        s = DerivedTextStore(path=path)  # must not raise out of __init__
        # Recovered to a clean, usable store (the corrupt file was dropped + recreated).
        assert s.available is False  # fresh: no build has run yet
        s.build(ROWS, col_mod=1)
        assert s.available is True
        assert [m.note_id for m in s.search_substring("powerhouse")] == [1]
        s.close()


class TestBuildFailure:
    def test_failed_build_rolls_back(self, store):
        # Review F8: a build that raises mid-transaction rolls back, so a later size() SELECT on the
        # shared connection sees the intact pre-build data, not a half-cleared index. A non-iterable
        # `rows` raises inside the locked transaction, *after* the two DELETEs.
        with pytest.raises(TypeError):
            store.build(42, col_mod=200)  # type: ignore[arg-type]
        assert store.state == IndexState.ERROR
        # size() reads on the same connection: without the rollback it would see the DELETEd 0 rows;
        # with it, the original 4 are intact.
        assert store.size == 4


class TestFts5Unavailable:
    def test_degrades_when_probe_fails(self, tmp_path, monkeypatch):
        # Simulate a SQLite build without FTS5/trigram: the store is inert and every lookup signals
        # the caller to fall back (substring → None, fuzzy → []), never raising.
        monkeypatch.setattr(DerivedTextStore, "_probe_fts5", staticmethod(lambda: False))
        s = DerivedTextStore(path=tmp_path / "shrike.db")
        assert s.available is False
        assert s.status()["fts5"] is False
        assert s.search_substring("mitochondria") is None
        assert s.search_fuzzy("mitochondria") == []
        s.build(ROWS, col_mod=1)  # a no-op, not a crash
        assert s.size == 0
        s.close()


class TestStatus:
    def test_status_shape(self, store):
        st = store.status()
        assert st["state"] == "ready"
        assert st["available"] is True
        assert st["fts5"] is True
        assert st["size"] == 4
        assert st["col_mod"] == 100
        assert "path" in st
