"""Tests for the IndexEngine seam (#267) — the frozen future-FFI surface.

The behavioural contract is pinned by the existing ``test_index.py`` suite through
``VectorIndex`` (which delegates storage to the engine). These tests pin the *seam*
itself: protocol conformance and the engine quirks that are part of the frozen
contract (phantom hit, multi-key dedup, remove counts) exercised directly, so a
later native engine (#273) can run through them verbatim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shrike.embedding_base import IMAGE, TEXT, IndexEngine
from shrike.index_engine import UsearchIndexEngine, modality_file_paths

NDIM = 8


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=NDIM)
    return [float(x) for x in v / np.linalg.norm(v)]


class TestProtocolConformance:
    def test_usearch_engine_satisfies_protocol(self) -> None:
        assert isinstance(UsearchIndexEngine(), IndexEngine)

    def test_instance_per_space_no_shared_state(self) -> None:
        a, b = UsearchIndexEngine(), UsearchIndexEngine()
        a.add(TEXT, [1], [_vec(1)])
        assert a.size == 1
        assert b.size == 0


class TestEngineQuirks:
    """The engine quirks #267 freezes as part of the contract."""

    def test_empty_index_search_yields_no_phantom_hit(self) -> None:
        # An empty USearch index can return a phantom (key 0, distance 0) match; the engine
        # filters it, so an empty (created-but-unpopulated) sub-index searches to [].
        engine = UsearchIndexEngine()
        engine.ensure(TEXT, NDIM)
        rankings = engine.search_by_modality(np.array([_vec(0)], dtype=np.float32), 5)
        assert rankings == [{}]

    def test_multi_key_dedup_is_min_distance_per_note(self) -> None:
        # Two image vectors under one key: the note appears once, at its best distance.
        engine = UsearchIndexEngine()
        engine.add(TEXT, [1], [_vec(1)])
        engine.add(IMAGE, [1, 1], [_vec(1), _vec(2)])
        rankings = engine.search_by_modality(np.array([_vec(1)], dtype=np.float32), 5)
        image_hits = rankings[0][IMAGE]
        assert [h["note_id"] for h in image_hits] == [1]
        assert image_hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)

    def test_remove_returns_text_index_count(self) -> None:
        engine = UsearchIndexEngine()
        engine.add(TEXT, [1, 2], [_vec(1), _vec(2)])
        engine.add(IMAGE, [1, 1], [_vec(3), _vec(4)])
        # 2 notes removed from text; the 2 image vectors go too but aren't counted.
        assert engine.remove([1, 2, 999]) == 2
        assert engine.size == 0

    def test_modalities_filter_narrows_search(self) -> None:
        engine = UsearchIndexEngine()
        engine.add(TEXT, [1], [_vec(1)])
        engine.add(IMAGE, [1], [_vec(2)])
        rankings = engine.search_by_modality(
            np.array([_vec(1)], dtype=np.float32), 5, modalities=(TEXT,)
        )
        assert set(rankings[0]) == {TEXT}


class TestPersistence:
    def test_save_restore_round_trip(self, tmp_path: Path) -> None:
        engine = UsearchIndexEngine()
        engine.add(TEXT, [1, 2], [_vec(1), _vec(2)])
        engine.add(IMAGE, [1], [_vec(3)])
        engine.save(str(tmp_path))

        fresh = UsearchIndexEngine()
        assert fresh.restore(str(tmp_path)) is True
        assert fresh.size == 3
        assert fresh.ndim == NDIM
        assert fresh.contains(1) and fresh.contains(2)
        assert fresh.modality_sizes() == {TEXT: 2, IMAGE: 1}

    def test_restore_missing_dir_is_empty_success(self, tmp_path: Path) -> None:
        engine = UsearchIndexEngine()
        assert engine.restore(str(tmp_path / "nope")) is True
        assert engine.size == 0

    def test_restore_corrupt_file_clears_and_fails(self, tmp_path: Path) -> None:
        (tmp_path / "index.usearch").write_bytes(b"not a usearch file")
        engine = UsearchIndexEngine()
        assert engine.restore(str(tmp_path)) is False
        assert engine.size == 0
        assert engine.ndim is None

    def test_save_deletes_stale_modality_file(self, tmp_path: Path) -> None:
        engine = UsearchIndexEngine()
        engine.add(TEXT, [1], [_vec(1)])
        engine.add(IMAGE, [1], [_vec(2)])
        engine.save(str(tmp_path))
        image_file = tmp_path / "index.image.usearch"
        assert image_file.exists()

        engine.clear()
        engine.add(TEXT, [1], [_vec(1)])
        engine.save(str(tmp_path))
        assert not image_file.exists()

    def test_modality_file_paths_covers_text_file(self, tmp_path: Path) -> None:
        names = [p.name for p in modality_file_paths(tmp_path)]
        assert "index.usearch" in names
        assert "index.image.usearch" in names


class TestVectorAccess:
    def test_keys_and_get(self) -> None:
        engine = UsearchIndexEngine()
        engine.add(TEXT, [3, 1], [_vec(3), _vec(1)])
        assert engine.keys() == [1, 3]
        got = np.atleast_2d(np.asarray(engine.get(1), dtype=np.float32))
        assert got.shape == (1, NDIM)
        np.testing.assert_allclose(got[0], np.array(_vec(1), dtype=np.float32), atol=1e-6)
        assert engine.get(42) is None or len(np.atleast_1d(engine.get(42))) == 0
