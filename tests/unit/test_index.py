"""Tests for the shrike.index module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from shrike.index import VectorIndex

NDIM = 8


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings based on text hash."""
    rng = np.random.default_rng(42)
    vecs = []
    for text in texts:
        seed = hash(text) % (2**31)
        local_rng = np.random.default_rng(seed)
        vec = local_rng.standard_normal(NDIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vecs.append(vec.tolist())
    return vecs


@pytest.fixture()
def embedding_service() -> MagicMock:
    svc = MagicMock()
    svc.embed = MagicMock(side_effect=_fake_embed)
    return svc


@pytest.fixture()
def index(tmp_path: Path, embedding_service: MagicMock) -> VectorIndex:
    return VectorIndex(tmp_path / "index", embedding_service=embedding_service)


class TestInit:
    def test_empty_index(self, index: VectorIndex) -> None:
        assert index.size == 0
        assert index.ndim is None
        assert index.available is False

    def test_available_after_add(self, index: VectorIndex) -> None:
        index.add([1], ["hello"])
        assert index.available is True

    def test_not_available_without_embedding(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        assert idx.available is False

    def test_creates_dir_on_add(self, index: VectorIndex) -> None:
        index.add([1], ["hello"])
        assert index._dir.exists()


class TestAdd:
    def test_add_single(self, index: VectorIndex) -> None:
        count = index.add([100], ["test text"])
        assert count == 1
        assert index.size == 1
        assert index.contains(100)

    def test_add_multiple(self, index: VectorIndex) -> None:
        ids = [1, 2, 3]
        texts = ["alpha", "beta", "gamma"]
        count = index.add(ids, texts)
        assert count == 3
        assert index.size == 3
        for nid in ids:
            assert index.contains(nid)

    def test_add_replaces_existing(self, index: VectorIndex) -> None:
        index.add([1], ["original"])
        index.add([1], ["updated"])
        assert index.size == 1
        assert index.contains(1)

    def test_add_empty(self, index: VectorIndex) -> None:
        count = index.add([], [])
        assert count == 0
        assert index.size == 0

    def test_add_mismatched_lengths(self, index: VectorIndex) -> None:
        with pytest.raises(ValueError, match="same length"):
            index.add([1, 2], ["only one text"])

    def test_add_without_embedding_service(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        with pytest.raises(RuntimeError, match="No embedding service"):
            idx.add([1], ["text"])

    def test_sets_ndim_on_first_add(self, index: VectorIndex) -> None:
        assert index.ndim is None
        index.add([1], ["hello"])
        assert index.ndim == NDIM

    def test_calls_embed_service(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.add([1, 2], ["hello", "world"])
        embedding_service.embed.assert_called_once_with(["hello", "world"])


class TestRemove:
    def test_remove_existing(self, index: VectorIndex) -> None:
        index.add([1, 2, 3], ["a", "b", "c"])
        removed = index.remove([2])
        assert removed == 1
        assert index.size == 2
        assert not index.contains(2)

    def test_remove_nonexistent(self, index: VectorIndex) -> None:
        index.add([1], ["a"])
        removed = index.remove([999])
        assert removed == 0
        assert index.size == 1

    def test_remove_from_empty(self, index: VectorIndex) -> None:
        removed = index.remove([1])
        assert removed == 0

    def test_remove_multiple(self, index: VectorIndex) -> None:
        index.add([1, 2, 3, 4], ["a", "b", "c", "d"])
        removed = index.remove([2, 4])
        assert removed == 2
        assert index.size == 2
        assert index.contains(1)
        assert not index.contains(2)
        assert index.contains(3)
        assert not index.contains(4)


class TestSearch:
    def test_search_returns_results(self, index: VectorIndex) -> None:
        index.add([1, 2, 3], ["cat", "dog", "fish"])
        results = index.search(["cat"], top_k=3)
        assert len(results) == 1
        assert len(results[0]) <= 3
        assert all("note_id" in r and "distance" in r for r in results[0])

    def test_search_nearest_is_self(self, index: VectorIndex) -> None:
        index.add([1, 2, 3], ["cat", "dog", "fish"])
        results = index.search(["cat"], top_k=1)
        assert results[0][0]["note_id"] == 1
        assert results[0][0]["distance"] < 0.01

    def test_search_multiple_queries(self, index: VectorIndex) -> None:
        index.add([1, 2], ["hello", "world"])
        results = index.search(["hello", "world"], top_k=2)
        assert len(results) == 2
        assert results[0][0]["note_id"] == 1
        assert results[1][0]["note_id"] == 2

    def test_search_respects_top_k(self, index: VectorIndex) -> None:
        index.add([1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"])
        results = index.search(["a"], top_k=2)
        assert len(results[0]) == 2

    def test_search_empty_index(self, index: VectorIndex) -> None:
        results = index.search(["hello"])
        assert results == [[]]

    def test_search_not_available(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        results = idx.search(["hello"])
        assert results == [[]]


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, embedding_service=embedding_service)
        idx1.add([10, 20, 30], ["alpha", "beta", "gamma"])
        idx1.save()

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        assert idx2.size == 3
        assert idx2.ndim == NDIM
        assert idx2.contains(10)
        assert idx2.contains(20)
        assert idx2.contains(30)

    def test_load_nonexistent_is_empty(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "no-such-dir")
        assert idx.size == 0
        assert idx.ndim is None

    def test_clear_removes_files(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        path = tmp_path / "index"
        idx = VectorIndex(path, embedding_service=embedding_service)
        idx.add([1], ["hello"])
        idx.save()
        assert (path / "index.usearch").exists()
        assert (path / "index.meta.json").exists()

        idx.clear()
        assert idx.size == 0
        assert not (path / "index.usearch").exists()
        assert not (path / "index.meta.json").exists()

    def test_search_after_load(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, embedding_service=embedding_service)
        idx1.add([1, 2], ["hello", "world"])
        idx1.save()

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        results = idx2.search(["hello"], top_k=1)
        assert results[0][0]["note_id"] == 1


class TestContains:
    def test_contains_existing(self, index: VectorIndex) -> None:
        index.add([42], ["text"])
        assert index.contains(42) is True

    def test_contains_missing(self, index: VectorIndex) -> None:
        assert index.contains(999) is False

    def test_contains_after_remove(self, index: VectorIndex) -> None:
        index.add([42], ["text"])
        index.remove([42])
        assert index.contains(42) is False


class TestStatus:
    def test_status_empty(self, index: VectorIndex) -> None:
        s = index.status()
        assert s["available"] is False
        assert s["size"] == 0
        assert s["ndim"] is None

    def test_status_after_add(self, index: VectorIndex) -> None:
        index.add([1, 2], ["a", "b"])
        s = index.status()
        assert s["size"] == 2
        assert s["ndim"] == NDIM

    def test_status_not_available(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        assert idx.status()["available"] is False
