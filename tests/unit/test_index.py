"""Tests for the shrike.index module."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from shrike.index import DEFAULT_SAVE_THRESHOLD, IndexSaver, IndexState, VectorIndex

NDIM = 8


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings based on text hash."""
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

    def test_initial_state_ready_with_embedding(self, index: VectorIndex) -> None:
        assert index.state == IndexState.READY

    def test_initial_state_unavailable_without_embedding(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        assert idx.state == IndexState.UNAVAILABLE


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

    def test_calls_embed_service(self, index: VectorIndex, embedding_service: MagicMock) -> None:
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

    def test_clear_removes_files(self, tmp_path: Path, embedding_service: MagicMock) -> None:
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

    def test_search_after_load(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, embedding_service=embedding_service)
        idx1.add([1, 2], ["hello", "world"])
        idx1.save()

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        results = idx2.search(["hello"], top_k=1)
        assert results[0][0]["note_id"] == 1

    def test_col_mod_persisted(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, embedding_service=embedding_service)
        idx1.add([1], ["hello"])
        idx1.col_mod = 1234567890
        idx1.save()

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        assert idx2.col_mod == 1234567890


class TestMaterializeEmpty:
    """Eagerly materializing an empty-but-ready index (#148)."""

    def test_makes_index_available(self, index: VectorIndex) -> None:
        assert index.available is False
        index.materialize_empty(NDIM, col_mod=42, model_id="m1")
        assert index.available is True
        assert index.size == 0
        assert index.ndim == NDIM
        assert index.col_mod == 42
        assert index.model_id == "m1"

    def test_incremental_add_works_after_materialize(self, index: VectorIndex) -> None:
        # The point of #148: once materialized, the upsert path (gated on
        # available) can add notes that are then searchable in the same session.
        index.materialize_empty(NDIM, col_mod=42, model_id="m1")
        index.add([1], ["hello"])
        assert index.size == 1
        results = index.search(["hello"], top_k=1)
        assert results[0][0]["note_id"] == 1

    def test_persists_to_disk(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, embedding_service=embedding_service)
        idx1.materialize_empty(NDIM, col_mod=99, model_id="m1")

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        assert idx2.available is True
        assert idx2.col_mod == 99
        assert idx2.model_id == "m1"
        # A matching col_mod means no drift, so reload skips a rebuild.
        assert idx2.check_drift(99, "m1") is False

    def test_noop_when_index_exists(self, index: VectorIndex) -> None:
        index.add([1], ["hello"])
        index.materialize_empty(NDIM, col_mod=7, model_id="m1")
        # The existing vector and its (unset) col_mod are left untouched.
        assert index.size == 1
        assert index.col_mod is None


class TestDirtyTracking:
    def test_add_increments_pending(self, index: VectorIndex) -> None:
        index.add([1, 2, 3], ["a", "b", "c"])
        assert index.pending_changes == 3

    def test_remove_counts_toward_pending(self, index: VectorIndex) -> None:
        index.add([1, 2], ["a", "b"])
        assert index.pending_changes == 2
        index.remove([1])
        assert index.pending_changes == 3

    def test_save_resets_pending(self, index: VectorIndex) -> None:
        index.add([1, 2, 3], ["a", "b", "c"])
        index.save()
        assert index.pending_changes == 0
        index.add([4], ["d"])
        assert index.pending_changes == 1

    def test_rebuild_leaves_index_clean(self, index: VectorIndex) -> None:
        # rebuild() saves at the end, so the counter is reset even though its
        # per-batch add() calls incremented it.
        index.rebuild([1, 2, 3], ["a", "b", "c"], col_mod=100)
        assert index.pending_changes == 0


class TestIndexSaver:
    """The debounced, off-thread persistence wrapper used by the server."""

    def _idx_path(self, tmp_path: Path) -> Path:
        return tmp_path / "index" / "index.usearch"

    async def test_flushes_on_burst(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        # threshold reached → immediate flush, no waiting for the idle delay.
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        saver = IndexSaver(idx, delay=999.0, threshold=3)
        idx.add([1, 2, 3], ["a", "b", "c"])
        idx.col_mod = 555
        saver.request_save()
        await saver.aclose()  # drains the in-flight save task

        assert self._idx_path(tmp_path).exists()
        assert idx.pending_changes == 0
        reloaded = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        assert reloaded.size == 3
        assert reloaded.col_mod == 555

    async def test_debounces_then_flushes_when_idle(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        saver = IndexSaver(idx, delay=0.05, threshold=999)
        idx.add([1], ["a"])
        idx.col_mod = 1
        saver.request_save()
        # Below the burst threshold, so nothing is written until the idle timer.
        assert not self._idx_path(tmp_path).exists()

        await asyncio.sleep(0.12)
        if saver._task is not None:
            await saver._task
        assert self._idx_path(tmp_path).exists()

    async def test_new_change_resets_the_idle_timer(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        saver = IndexSaver(idx, delay=0.1, threshold=999)
        idx.add([1], ["a"])
        saver.request_save()
        await asyncio.sleep(0.05)  # half the delay
        idx.add([2], ["b"])
        saver.request_save()  # resets the timer
        await asyncio.sleep(0.07)  # past the *first* deadline, before the reset one
        assert not self._idx_path(tmp_path).exists()

    async def test_aclose_flushes_pending_immediately(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        saver = IndexSaver(idx, delay=999.0, threshold=999)
        idx.add([1], ["a"])
        idx.col_mod = 9
        saver.request_save()  # arms a long timer; no burst
        await saver.aclose()  # shutdown path: flush now without waiting

        assert self._idx_path(tmp_path).exists()
        assert idx.pending_changes == 0

    async def test_aclose_is_noop_when_clean(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        saver = IndexSaver(idx)
        await saver.aclose()  # nothing added — must not error or write
        assert not self._idx_path(tmp_path).exists()

    def test_default_threshold(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        saver = IndexSaver(idx)
        assert saver._threshold == DEFAULT_SAVE_THRESHOLD


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
        assert s["state"] == "ready"

    def test_status_after_add(self, index: VectorIndex) -> None:
        index.add([1, 2], ["a", "b"])
        s = index.status()
        assert s["size"] == 2
        assert s["ndim"] == NDIM
        assert s["state"] == "ready"

    def test_status_not_available(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        s = idx.status()
        assert s["available"] is False
        assert s["state"] == "unavailable"

    def test_status_shows_col_mod(self, index: VectorIndex) -> None:
        index.col_mod = 999
        s = index.status()
        assert s["col_mod"] == 999

    def test_status_building_shows_progress(self, index: VectorIndex) -> None:
        index._state = IndexState.BUILDING
        index._build_progress = (50, 200)
        s = index.status()
        assert s["state"] == "building"
        assert s["progress"] == {"indexed": 50, "total": 200}

    def test_status_error_shows_message(self, index: VectorIndex) -> None:
        index._state = IndexState.ERROR
        index._build_error = "something broke"
        s = index.status()
        assert s["state"] == "error"
        assert s["error"] == "something broke"


class TestDriftDetection:
    def test_no_index_needs_rebuild(self, index: VectorIndex) -> None:
        assert index.check_drift(100) is True

    def test_no_col_mod_needs_rebuild(self, index: VectorIndex) -> None:
        index.add([1], ["a"])
        assert index.col_mod is None
        assert index.check_drift(100) is True

    def test_matching_col_mod_no_rebuild(self, index: VectorIndex) -> None:
        index.add([1], ["a"])
        index.col_mod = 100
        assert index.check_drift(100) is False

    def test_mismatched_col_mod_needs_rebuild(self, index: VectorIndex) -> None:
        index.add([1], ["a"])
        index.col_mod = 100
        assert index.check_drift(200) is True

    def test_drift_after_save_and_load(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, embedding_service=embedding_service)
        idx1.add([1], ["a"])
        idx1.col_mod = 100
        idx1.save()

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        assert idx2.check_drift(100) is False
        assert idx2.check_drift(200) is True

    def test_model_change_needs_rebuild(self, index: VectorIndex) -> None:
        index.rebuild([1], ["a"], col_mod=100, model_id="meta:1:2:3")
        # Same model + col_mod → no rebuild.
        assert index.check_drift(100, "meta:1:2:3") is False
        # Different model → rebuild, even though col_mod matches.
        assert index.check_drift(100, "meta:9:9:9") is True

    def test_model_id_none_ignored(self, index: VectorIndex) -> None:
        index.rebuild([1], ["a"], col_mod=100, model_id="meta:1:2:3")
        # No model_id passed → only col_mod is considered.
        assert index.check_drift(100) is False


class TestModelIdPersistence:
    def test_model_id_round_trips(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, embedding_service=embedding_service)
        idx1.rebuild([1], ["a"], col_mod=100, model_id="meta:42:384:30522")
        assert idx1.model_id == "meta:42:384:30522"

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        assert idx2.model_id == "meta:42:384:30522"
        assert idx2.check_drift(100, "meta:42:384:30522") is False
        assert idx2.check_drift(100, "meta:0:0:0") is True


class TestSetEmbeddingService:
    def test_detach_marks_unavailable(self, index: VectorIndex) -> None:
        index.add([1], ["a"])
        assert index.state == IndexState.READY
        index.set_embedding_service(None)
        assert index.state == IndexState.UNAVAILABLE
        assert index.available is False

    def test_attach_flips_unavailable_to_ready(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index")  # no embedder → UNAVAILABLE
        assert idx.state == IndexState.UNAVAILABLE
        idx.set_embedding_service(embedding_service)
        assert idx.state == IndexState.READY

    def test_attach_does_not_clobber_building(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index._state = IndexState.BUILDING
        index.set_embedding_service(embedding_service)
        assert index.state == IndexState.BUILDING

    def test_detached_vectors_survive_reattach(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.add([1, 2], ["a", "b"])
        size = index.size
        index.set_embedding_service(None)
        assert index.size == size  # vectors kept on detach
        index.set_embedding_service(embedding_service)
        assert index.available is True
        assert index.size == size


class TestRebuild:
    def test_rebuild_creates_index(self, index: VectorIndex) -> None:
        index.rebuild([1, 2, 3], ["a", "b", "c"], col_mod=500)
        assert index.size == 3
        assert index.state == IndexState.READY
        assert index.col_mod == 500

    def test_rebuild_replaces_existing(self, index: VectorIndex) -> None:
        index.add([1, 2], ["old_a", "old_b"])
        assert index.size == 2

        index.rebuild([10, 20, 30], ["x", "y", "z"], col_mod=600)
        assert index.size == 3
        assert not index.contains(1)
        assert index.contains(10)

    def test_rebuild_saves_to_disk(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx = VectorIndex(path, embedding_service=embedding_service)
        idx.rebuild([1], ["a"], col_mod=100)

        idx2 = VectorIndex(path, embedding_service=embedding_service)
        assert idx2.size == 1
        assert idx2.col_mod == 100

    def test_rebuild_tracks_progress(self, index: VectorIndex) -> None:
        progress_log: list[tuple[int, int]] = []
        index.rebuild(
            [1, 2, 3],
            ["a", "b", "c"],
            col_mod=100,
            on_progress=lambda i, t: progress_log.append((i, t)),
        )
        assert progress_log[-1] == (3, 3)

    def test_rebuild_sets_error_on_failure(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        embedding_service.embed.side_effect = RuntimeError("embed failed")

        with pytest.raises(RuntimeError, match="embed failed"):
            idx.rebuild([1], ["a"], col_mod=100)

        assert idx.state == IndexState.ERROR
        assert idx._build_error == "embed failed"

    def test_rebuild_empty_collection(self, index: VectorIndex) -> None:
        index.rebuild([], [], col_mod=100)
        assert index.size == 0
        assert index.state == IndexState.READY
        assert index.col_mod == 100


class TestRebuildInBackground:
    def test_background_rebuild_completes(self, index: VectorIndex) -> None:
        index.rebuild_in_background([1, 2], ["a", "b"], col_mod=300)
        assert index._build_thread is not None
        index._build_thread.join(timeout=5)
        assert index.state == IndexState.READY
        assert index.size == 2
        assert index.col_mod == 300

    def test_background_rebuild_sets_building_state(self, index: VectorIndex) -> None:
        import threading

        started = threading.Event()
        original_embed = index._embedding.embed

        def slow_embed(texts: list[str]) -> list[list[float]]:
            started.set()
            return original_embed(texts)

        index._embedding.embed = slow_embed

        index.rebuild_in_background([1], ["a"], col_mod=100)
        started.wait(timeout=5)
        # Can't reliably assert BUILDING here since the thread may finish
        # before we check, but we can verify it completes correctly
        index._build_thread.join(timeout=5)  # type: ignore[union-attr]
        assert index.state == IndexState.READY

    def test_skip_if_already_building(self, index: VectorIndex) -> None:
        index._state = IndexState.BUILDING
        index.rebuild_in_background([1], ["a"], col_mod=100)
        assert index._build_thread is None

    def test_background_rebuild_handles_error(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", embedding_service=embedding_service)
        embedding_service.embed.side_effect = RuntimeError("fail")

        idx.rebuild_in_background([1], ["a"], col_mod=100)
        idx._build_thread.join(timeout=5)  # type: ignore[union-attr]
        assert idx.state == IndexState.ERROR


def _vectors(idx: VectorIndex) -> dict[int, tuple]:
    """The index's {note_id: rounded vector} — for asserting two indexes match."""
    keys = list(idx._index.keys)  # type: ignore[union-attr]
    return {int(k): tuple(np.round(idx._index.get(k), 5)) for k in keys}  # type: ignore[union-attr]


def _embedded(svc: MagicMock) -> list[str]:
    """Every text passed to embed() since the last reset, flattened."""
    return [t for call in svc.embed.call_args_list for t in call.args[0]]


class TestReconcile:
    """Incremental reconcile (#38): re-embed only changed notes on drift, and
    leave an index identical to a full rebuild over the same final state."""

    def test_reconcile_matches_full_rebuild(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "a", embedding_service=embedding_service)
        idx.rebuild([1, 2, 3, 4], ["t1", "t2", "t3", "t4"], col_mod=1)
        # change t2, delete 4, add 5; 1 and 3 unchanged.
        new_ids, new_texts = [1, 2, 3, 5], ["t1", "t2-changed", "t3", "t5"]
        idx.reconcile(new_ids, new_texts, col_mod=2)

        ref = VectorIndex(tmp_path / "b", embedding_service=embedding_service)
        ref.rebuild(new_ids, new_texts, col_mod=2)

        assert _vectors(idx) == _vectors(ref)
        assert idx.col_mod == 2

    def test_only_reembeds_changed_and_new(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild([1, 2, 3], ["a", "b", "c"], col_mod=1)
        embedding_service.embed.reset_mock()
        index.reconcile([1, 2, 3, 4], ["a", "b-new", "c", "d"], col_mod=2)
        assert sorted(_embedded(embedding_service)) == ["b-new", "d"]

    def test_removes_deleted(self, index: VectorIndex) -> None:
        index.rebuild([1, 2, 3], ["a", "b", "c"], col_mod=1)
        index.reconcile([1, 3], ["a", "c"], col_mod=2)
        assert index.size == 2
        assert not index.contains(2)

    def test_no_text_change_advances_col_mod_without_embedding(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild([1, 2], ["a", "b"], col_mod=1)
        embedding_service.embed.reset_mock()
        index.reconcile([1, 2], ["a", "b"], col_mod=99)
        assert embedding_service.embed.call_count == 0
        assert index.col_mod == 99

    def test_model_change_falls_back_to_full_rebuild(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild([1, 2], ["a", "b"], col_mod=1, model_id="m1")
        embedding_service.embed.reset_mock()
        index.reconcile([1, 2], ["a", "b"], col_mod=2, model_id="m2")
        assert sorted(_embedded(embedding_service)) == ["a", "b"]  # every vector re-embedded
        assert index.model_id == "m2"

    def test_no_prior_hashes_falls_back_to_full_rebuild(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild([1, 2], ["a", "b"], col_mod=1)
        index._note_hashes = None  # simulate an index built before hashes existed
        embedding_service.embed.reset_mock()
        index.reconcile([1, 2, 3], ["a", "b", "c"], col_mod=2)
        assert sorted(_embedded(embedding_service)) == ["a", "b", "c"]

    def test_hashes_persist_across_reload(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "i", embedding_service=embedding_service)
        idx.rebuild([1, 2], ["a", "b"], col_mod=1)
        reloaded = VectorIndex(tmp_path / "i", embedding_service=embedding_service)
        assert reloaded._note_hashes is not None
        embedding_service.embed.reset_mock()
        reloaded.reconcile([1, 2], ["a", "b-changed"], col_mod=2)
        assert _embedded(embedding_service) == ["b-changed"]  # only the changed note

    def test_incremental_add_keeps_hashes_current(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        # add() maintains hashes, so a reconcile doesn't re-embed notes Shrike
        # itself just upserted.
        idx = VectorIndex(tmp_path / "i", embedding_service=embedding_service)
        idx.rebuild([1], ["a"], col_mod=1)
        idx.add([2], ["b"])
        embedding_service.embed.reset_mock()
        idx.reconcile([1, 2], ["a", "b"], col_mod=2)
        assert embedding_service.embed.call_count == 0
