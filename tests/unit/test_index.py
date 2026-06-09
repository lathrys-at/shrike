"""Tests for the shrike.index module."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from shrike.embedding_base import IMAGE, TEXT
from shrike.index import (
    DEFAULT_SAVE_THRESHOLD,
    INDEX_SCHEMA_VERSION,
    IndexSaver,
    IndexState,
    NoteEmbedInput,
    VectorIndex,
)

NDIM = 8


def _inp(ids: list[int], texts: list[str]) -> list[NoteEmbedInput]:
    """Adapt the legacy (ids, texts) test shape to NoteEmbedInputs (text-only)."""
    return [NoteEmbedInput(int(i), t) for i, t in zip(ids, texts, strict=True)]


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
    svc.embed_texts = MagicMock(side_effect=_fake_embed)
    svc.modalities = frozenset({TEXT})
    return svc


@pytest.fixture()
def index(tmp_path: Path, embedding_service: MagicMock) -> VectorIndex:
    return VectorIndex(tmp_path / "index", backend=embedding_service)


def _img_vec(data: bytes) -> list[float]:
    """Deterministic fake image embedding from the bytes (distinct from text vectors)."""
    seed = int.from_bytes(hashlib.blake2b(data, digest_size=4).digest(), "big")
    vec = np.random.default_rng(seed).standard_normal(NDIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture()
def image_backend() -> MagicMock:
    """A text+image (CLIP-like) backend: embed_images maps bytes to a vector per content."""
    svc = MagicMock()
    svc.embed_texts = MagicMock(side_effect=_fake_embed)
    svc.embed_images = MagicMock(side_effect=lambda imgs: [_img_vec(b) for b in imgs])
    svc.modalities = frozenset({TEXT, IMAGE})
    return svc


@pytest.fixture()
def image_index(tmp_path: Path, image_backend: MagicMock) -> VectorIndex:
    return VectorIndex(tmp_path / "index", backend=image_backend)


class TestInit:
    def test_empty_index(self, index: VectorIndex) -> None:
        assert index.size == 0
        assert index.ndim is None
        assert index.available is False

    def test_available_after_add(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["hello"]))
        assert index.available is True

    def test_not_available_without_embedding(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        assert idx.available is False

    def test_creates_dir_on_add(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["hello"]))
        assert index._dir.exists()

    def test_initial_state_ready_with_embedding(self, index: VectorIndex) -> None:
        assert index.state == IndexState.READY

    def test_initial_state_unavailable_without_embedding(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        assert idx.state == IndexState.UNAVAILABLE


class TestAdd:
    def test_add_single(self, index: VectorIndex) -> None:
        count = index.add(_inp([100], ["test text"]))
        assert count == 1
        assert index.size == 1
        assert index.contains(100)

    def test_add_multiple(self, index: VectorIndex) -> None:
        ids = [1, 2, 3]
        texts = ["alpha", "beta", "gamma"]
        count = index.add(_inp(ids, texts))
        assert count == 3
        assert index.size == 3
        for nid in ids:
            assert index.contains(nid)

    def test_add_replaces_existing(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["original"]))
        index.add(_inp([1], ["updated"]))
        assert index.size == 1
        assert index.contains(1)

    def test_add_empty(self, index: VectorIndex) -> None:
        count = index.add(_inp([], []))
        assert count == 0
        assert index.size == 0

    def test_add_without_embedding_service(self, tmp_path: Path) -> None:
        idx = VectorIndex(tmp_path / "index")
        with pytest.raises(RuntimeError, match="No embedding service"):
            idx.add(_inp([1], ["text"]))

    def test_sets_ndim_on_first_add(self, index: VectorIndex) -> None:
        assert index.ndim is None
        index.add(_inp([1], ["hello"]))
        assert index.ndim == NDIM

    def test_calls_embed_service(self, index: VectorIndex, embedding_service: MagicMock) -> None:
        index.add(_inp([1, 2], ["hello", "world"]))
        embedding_service.embed_texts.assert_called_once_with(["hello", "world"])


class TestRemove:
    def test_remove_existing(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2, 3], ["a", "b", "c"]))
        removed = index.remove([2])
        assert removed == 1
        assert index.size == 2
        assert not index.contains(2)

    def test_remove_nonexistent(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["a"]))
        removed = index.remove([999])
        assert removed == 0
        assert index.size == 1

    def test_remove_from_empty(self, index: VectorIndex) -> None:
        removed = index.remove([1])
        assert removed == 0

    def test_remove_multiple(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2, 3, 4], ["a", "b", "c", "d"]))
        removed = index.remove([2, 4])
        assert removed == 2
        assert index.size == 2
        assert index.contains(1)
        assert not index.contains(2)
        assert index.contains(3)
        assert not index.contains(4)


class TestSearch:
    def test_search_returns_results(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2, 3], ["cat", "dog", "fish"]))
        results = index.search(["cat"], top_k=3)
        assert len(results) == 1
        assert len(results[0]) <= 3
        assert all("note_id" in r and "distance" in r for r in results[0])

    def test_search_nearest_is_self(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2, 3], ["cat", "dog", "fish"]))
        results = index.search(["cat"], top_k=1)
        assert results[0][0]["note_id"] == 1
        assert results[0][0]["distance"] < 0.01

    def test_search_multiple_queries(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2], ["hello", "world"]))
        results = index.search(["hello", "world"], top_k=2)
        assert len(results) == 2
        assert results[0][0]["note_id"] == 1
        assert results[1][0]["note_id"] == 2

    def test_search_respects_top_k(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"]))
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
        idx1 = VectorIndex(path, backend=embedding_service)
        idx1.add(_inp([10, 20, 30], ["alpha", "beta", "gamma"]))
        idx1.save()

        idx2 = VectorIndex(path, backend=embedding_service)
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
        idx = VectorIndex(path, backend=embedding_service)
        idx.add(_inp([1], ["hello"]))
        idx.save()
        assert (path / "index.usearch").exists()
        assert (path / "index.meta.json").exists()

        idx.clear()
        assert idx.size == 0
        assert not (path / "index.usearch").exists()
        assert not (path / "index.meta.json").exists()

    def test_search_after_load(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, backend=embedding_service)
        idx1.add(_inp([1, 2], ["hello", "world"]))
        idx1.save()

        idx2 = VectorIndex(path, backend=embedding_service)
        results = idx2.search(["hello"], top_k=1)
        assert results[0][0]["note_id"] == 1

    def test_col_mod_persisted(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, backend=embedding_service)
        idx1.add(_inp([1], ["hello"]))
        idx1.col_mod = 1234567890
        idx1.save()

        idx2 = VectorIndex(path, backend=embedding_service)
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
        index.add(_inp([1], ["hello"]))
        assert index.size == 1
        results = index.search(["hello"], top_k=1)
        assert results[0][0]["note_id"] == 1

    def test_persists_to_disk(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, backend=embedding_service)
        idx1.materialize_empty(NDIM, col_mod=99, model_id="m1")

        idx2 = VectorIndex(path, backend=embedding_service)
        assert idx2.available is True
        assert idx2.col_mod == 99
        assert idx2.model_id == "m1"
        # A matching col_mod means no drift, so reload skips a rebuild.
        assert idx2.check_drift(99, "m1") is False

    def test_noop_when_index_exists(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["hello"]))
        index.materialize_empty(NDIM, col_mod=7, model_id="m1")
        # The existing vector and its (unset) col_mod are left untouched.
        assert index.size == 1
        assert index.col_mod is None


class TestDirtyTracking:
    def test_add_increments_pending(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2, 3], ["a", "b", "c"]))
        assert index.pending_changes == 3

    def test_remove_counts_toward_pending(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2], ["a", "b"]))
        assert index.pending_changes == 2
        index.remove([1])
        assert index.pending_changes == 3

    def test_save_resets_pending(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2, 3], ["a", "b", "c"]))
        index.save()
        assert index.pending_changes == 0
        index.add(_inp([4], ["d"]))
        assert index.pending_changes == 1

    def test_rebuild_leaves_index_clean(self, index: VectorIndex) -> None:
        # rebuild() saves at the end, so the counter is reset even though its
        # per-batch add() calls incremented it.
        index.rebuild(_inp([1, 2, 3], ["a", "b", "c"]), col_mod=100)
        assert index.pending_changes == 0


class TestIndexSaver:
    """The debounced, off-thread persistence wrapper used by the server."""

    def _idx_path(self, tmp_path: Path) -> Path:
        return tmp_path / "index" / "index.usearch"

    async def test_flushes_on_burst(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        # threshold reached → immediate flush, no waiting for the idle delay.
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        saver = IndexSaver(idx, delay=999.0, threshold=3)
        idx.add(_inp([1, 2, 3], ["a", "b", "c"]))
        idx.col_mod = 555
        saver.request_save()
        await saver.aclose()  # drains the in-flight save task

        assert self._idx_path(tmp_path).exists()
        assert idx.pending_changes == 0
        reloaded = VectorIndex(tmp_path / "index", backend=embedding_service)
        assert reloaded.size == 3
        assert reloaded.col_mod == 555

    async def test_debounces_then_flushes_when_idle(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        saver = IndexSaver(idx, delay=0.05, threshold=999)
        idx.add(_inp([1], ["a"]))
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
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        saver = IndexSaver(idx, delay=0.1, threshold=999)
        idx.add(_inp([1], ["a"]))
        saver.request_save()
        await asyncio.sleep(0.05)  # half the delay
        idx.add(_inp([2], ["b"]))
        saver.request_save()  # resets the timer
        await asyncio.sleep(0.07)  # past the *first* deadline, before the reset one
        assert not self._idx_path(tmp_path).exists()

    async def test_aclose_flushes_pending_immediately(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        saver = IndexSaver(idx, delay=999.0, threshold=999)
        idx.add(_inp([1], ["a"]))
        idx.col_mod = 9
        saver.request_save()  # arms a long timer; no burst
        await saver.aclose()  # shutdown path: flush now without waiting

        assert self._idx_path(tmp_path).exists()
        assert idx.pending_changes == 0

    async def test_aclose_is_noop_when_clean(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        saver = IndexSaver(idx)
        await saver.aclose()  # nothing added — must not error or write
        assert not self._idx_path(tmp_path).exists()

    def test_default_threshold(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        saver = IndexSaver(idx)
        assert saver._threshold == DEFAULT_SAVE_THRESHOLD


class TestContains:
    def test_contains_existing(self, index: VectorIndex) -> None:
        index.add(_inp([42], ["text"]))
        assert index.contains(42) is True

    def test_contains_missing(self, index: VectorIndex) -> None:
        assert index.contains(999) is False

    def test_contains_after_remove(self, index: VectorIndex) -> None:
        index.add(_inp([42], ["text"]))
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
        index.add(_inp([1, 2], ["a", "b"]))
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
        index.add(_inp([1], ["a"]))
        assert index.col_mod is None
        assert index.check_drift(100) is True

    def test_matching_col_mod_no_rebuild(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["a"]))
        index.col_mod = 100
        assert index.check_drift(100) is False

    def test_mismatched_col_mod_needs_rebuild(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["a"]))
        index.col_mod = 100
        assert index.check_drift(200) is True

    def test_drift_after_save_and_load(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, backend=embedding_service)
        idx1.add(_inp([1], ["a"]))
        idx1.col_mod = 100
        idx1.save()

        idx2 = VectorIndex(path, backend=embedding_service)
        assert idx2.check_drift(100) is False
        assert idx2.check_drift(200) is True

    def test_model_change_needs_rebuild(self, index: VectorIndex) -> None:
        index.rebuild(_inp([1], ["a"]), col_mod=100, model_id="meta:1:2:3")
        # Same model + col_mod → no rebuild.
        assert index.check_drift(100, "meta:1:2:3") is False
        # Different model → rebuild, even though col_mod matches.
        assert index.check_drift(100, "meta:9:9:9") is True

    def test_model_id_none_ignored(self, index: VectorIndex) -> None:
        index.rebuild(_inp([1], ["a"]), col_mod=100, model_id="meta:1:2:3")
        # No model_id passed → only col_mod is considered.
        assert index.check_drift(100) is False


class TestModelIdPersistence:
    def test_model_id_round_trips(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx1 = VectorIndex(path, backend=embedding_service)
        idx1.rebuild(_inp([1], ["a"]), col_mod=100, model_id="meta:42:384:30522")
        assert idx1.model_id == "meta:42:384:30522"

        idx2 = VectorIndex(path, backend=embedding_service)
        assert idx2.model_id == "meta:42:384:30522"
        assert idx2.check_drift(100, "meta:42:384:30522") is False
        assert idx2.check_drift(100, "meta:0:0:0") is True


class TestSetBackend:
    def test_detach_marks_unavailable(self, index: VectorIndex) -> None:
        index.add(_inp([1], ["a"]))
        assert index.state == IndexState.READY
        index.set_backend(None)
        assert index.state == IndexState.UNAVAILABLE
        assert index.available is False

    def test_attach_flips_unavailable_to_ready(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index")  # no embedder → UNAVAILABLE
        assert idx.state == IndexState.UNAVAILABLE
        idx.set_backend(embedding_service)
        assert idx.state == IndexState.READY

    def test_attach_does_not_clobber_building(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index._state = IndexState.BUILDING
        index.set_backend(embedding_service)
        assert index.state == IndexState.BUILDING

    def test_detached_vectors_survive_reattach(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.add(_inp([1, 2], ["a", "b"]))
        size = index.size
        index.set_backend(None)
        assert index.size == size  # vectors kept on detach
        index.set_backend(embedding_service)
        assert index.available is True
        assert index.size == size


class TestRebuild:
    def test_rebuild_creates_index(self, index: VectorIndex) -> None:
        index.rebuild(_inp([1, 2, 3], ["a", "b", "c"]), col_mod=500)
        assert index.size == 3
        assert index.state == IndexState.READY
        assert index.col_mod == 500

    def test_rebuild_replaces_existing(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2], ["old_a", "old_b"]))
        assert index.size == 2

        index.rebuild(_inp([10, 20, 30], ["x", "y", "z"]), col_mod=600)
        assert index.size == 3
        assert not index.contains(1)
        assert index.contains(10)

    def test_rebuild_saves_to_disk(self, tmp_path: Path, embedding_service: MagicMock) -> None:
        path = tmp_path / "index"
        idx = VectorIndex(path, backend=embedding_service)
        idx.rebuild(_inp([1], ["a"]), col_mod=100)

        idx2 = VectorIndex(path, backend=embedding_service)
        assert idx2.size == 1
        assert idx2.col_mod == 100

    def test_rebuild_tracks_progress(self, index: VectorIndex) -> None:
        progress_log: list[tuple[int, int]] = []
        index.rebuild(
            _inp([1, 2, 3], ["a", "b", "c"]),
            col_mod=100,
            on_progress=lambda i, t: progress_log.append((i, t)),
        )
        assert progress_log[-1] == (3, 3)

    def test_rebuild_sets_error_on_failure(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        embedding_service.embed_texts.side_effect = RuntimeError("embed failed")

        with pytest.raises(RuntimeError, match="embed failed"):
            idx.rebuild(_inp([1], ["a"]), col_mod=100)

        assert idx.state == IndexState.ERROR
        assert idx._build_error == "embed failed"

    def test_rebuild_empty_collection(self, index: VectorIndex) -> None:
        index.rebuild(_inp([], []), col_mod=100)
        assert index.size == 0
        assert index.state == IndexState.READY
        assert index.col_mod == 100


class TestRebuildInBackground:
    def test_background_rebuild_completes(self, index: VectorIndex) -> None:
        index.rebuild_in_background(_inp([1, 2], ["a", "b"]), col_mod=300)
        assert index._build_thread is not None
        index._build_thread.join(timeout=5)
        assert index.state == IndexState.READY
        assert index.size == 2
        assert index.col_mod == 300

    def test_background_rebuild_sets_building_state(self, index: VectorIndex) -> None:
        import threading

        started = threading.Event()
        original_embed = index._embedding.embed_texts

        def slow_embed(texts: list[str]) -> list[list[float]]:
            started.set()
            return original_embed(texts)

        index._embedding.embed_texts = slow_embed

        index.rebuild_in_background(_inp([1], ["a"]), col_mod=100)
        started.wait(timeout=5)
        # Can't reliably assert BUILDING here since the thread may finish
        # before we check, but we can verify it completes correctly
        index._build_thread.join(timeout=5)  # type: ignore[union-attr]
        assert index.state == IndexState.READY

    def test_skip_if_already_building(self, index: VectorIndex) -> None:
        index._state = IndexState.BUILDING
        index.rebuild_in_background(_inp([1], ["a"]), col_mod=100)
        assert index._build_thread is None

    def test_background_rebuild_handles_error(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        embedding_service.embed_texts.side_effect = RuntimeError("fail")

        idx.rebuild_in_background(_inp([1], ["a"]), col_mod=100)
        idx._build_thread.join(timeout=5)  # type: ignore[union-attr]
        assert idx.state == IndexState.ERROR


def _vectors(idx: VectorIndex) -> dict[int, frozenset]:
    """The index's {note_id: set of rounded vectors across every modality sub-index} — for
    asserting two indexes match. Each sub-index is multi=True so keys repeat (once per vector) and
    ``get(key)`` returns all of a note's vectors as a 2D array; collapse each note's vectors (text +
    images, across sub-indexes) into one order-independent set.
    """
    out: dict[int, set] = {}
    for sub in idx._indexes.values():
        for k in {int(key) for key in sub.keys}:
            vecs = np.atleast_2d(np.asarray(sub.get(k)))
            out.setdefault(k, set()).update(tuple(np.round(row, 5)) for row in vecs)
    return {k: frozenset(v) for k, v in out.items()}


def _embedded(svc: MagicMock) -> list[str]:
    """Every text passed to embed() since the last reset, flattened."""
    return [t for call in svc.embed_texts.call_args_list for t in call.args[0]]


class TestReconcile:
    """Incremental reconcile (#38): re-embed only changed notes on drift, and
    leave an index identical to a full rebuild over the same final state."""

    def test_reconcile_matches_full_rebuild(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "a", backend=embedding_service)
        idx.rebuild(_inp([1, 2, 3, 4], ["t1", "t2", "t3", "t4"]), col_mod=1)
        # change t2, delete 4, add 5; 1 and 3 unchanged.
        new_ids, new_texts = [1, 2, 3, 5], ["t1", "t2-changed", "t3", "t5"]
        idx.reconcile(_inp(new_ids, new_texts), col_mod=2)

        ref = VectorIndex(tmp_path / "b", backend=embedding_service)
        ref.rebuild(_inp(new_ids, new_texts), col_mod=2)

        assert _vectors(idx) == _vectors(ref)
        assert idx.col_mod == 2

    def test_only_reembeds_changed_and_new(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild(_inp([1, 2, 3], ["a", "b", "c"]), col_mod=1)
        embedding_service.embed_texts.reset_mock()
        index.reconcile(_inp([1, 2, 3, 4], ["a", "b-new", "c", "d"]), col_mod=2)
        assert sorted(_embedded(embedding_service)) == ["b-new", "d"]

    def test_removes_deleted(self, index: VectorIndex) -> None:
        index.rebuild(_inp([1, 2, 3], ["a", "b", "c"]), col_mod=1)
        index.reconcile(_inp([1, 3], ["a", "c"]), col_mod=2)
        assert index.size == 2
        assert not index.contains(2)

    def test_no_text_change_advances_col_mod_without_embedding(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild(_inp([1, 2], ["a", "b"]), col_mod=1)
        embedding_service.embed_texts.reset_mock()
        index.reconcile(_inp([1, 2], ["a", "b"]), col_mod=99)
        assert embedding_service.embed_texts.call_count == 0
        assert index.col_mod == 99

    def test_model_change_falls_back_to_full_rebuild(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild(_inp([1, 2], ["a", "b"]), col_mod=1, model_id="m1")
        embedding_service.embed_texts.reset_mock()
        index.reconcile(_inp([1, 2], ["a", "b"]), col_mod=2, model_id="m2")
        assert sorted(_embedded(embedding_service)) == ["a", "b"]  # every vector re-embedded
        assert index.model_id == "m2"

    def test_no_prior_hashes_falls_back_to_full_rebuild(
        self, index: VectorIndex, embedding_service: MagicMock
    ) -> None:
        index.rebuild(_inp([1, 2], ["a", "b"]), col_mod=1)
        index._note_hashes = None  # simulate an index built before hashes existed
        embedding_service.embed_texts.reset_mock()
        index.reconcile(_inp([1, 2, 3], ["a", "b", "c"]), col_mod=2)
        assert sorted(_embedded(embedding_service)) == ["a", "b", "c"]

    def test_hashes_persist_across_reload(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "i", backend=embedding_service)
        idx.rebuild(_inp([1, 2], ["a", "b"]), col_mod=1)
        reloaded = VectorIndex(tmp_path / "i", backend=embedding_service)
        assert reloaded._note_hashes is not None
        embedding_service.embed_texts.reset_mock()
        reloaded.reconcile(_inp([1, 2], ["a", "b-changed"]), col_mod=2)
        assert _embedded(embedding_service) == ["b-changed"]  # only the changed note

    def test_incremental_add_keeps_hashes_current(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        # add() maintains hashes, so a reconcile doesn't re-embed notes Shrike
        # itself just upserted.
        idx = VectorIndex(tmp_path / "i", backend=embedding_service)
        idx.rebuild(_inp([1], ["a"]), col_mod=1)
        idx.add(_inp([2], ["b"]))
        embedding_service.embed_texts.reset_mock()
        idx.reconcile(_inp([1, 2], ["a", "b"]), col_mod=2)
        assert embedding_service.embed_texts.call_count == 0


def _resolver(store: dict[str, bytes]):
    """An image resolver backed by an in-memory {filename: bytes} map (None when missing)."""
    return lambda name: store.get(name)


class TestMultiVector:
    """Multi-vector index: a note maps to its text vector + one vector per image (#162)."""

    def test_add_stores_text_and_image_vectors(self, image_index: VectorIndex) -> None:
        image_index.set_image_resolver(_resolver({"a.png": b"AAA", "b.png": b"BBB"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png", "b.png"])])
        assert image_index.size == 3  # 1 text + 2 image vectors under key 1
        assert image_index.contains(1)

    def test_missing_image_skipped(self, image_index: VectorIndex) -> None:
        image_index.set_image_resolver(_resolver({}))  # nothing resolves
        image_index.add([NoteEmbedInput(1, "cat", ["missing.png"])])
        assert image_index.size == 1  # text only, no crash

    def test_no_resolver_embeds_text_only(self, image_index: VectorIndex) -> None:
        # An image-capable backend with no resolver attached still embeds only text.
        image_index.add([NoteEmbedInput(1, "cat", ["a.png"])])
        assert image_index.size == 1

    def test_search_dedups_to_one_result_per_note(self, image_index: VectorIndex) -> None:
        image_index.set_image_resolver(_resolver({"a.png": b"AAA"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png"])])  # 2 vectors under key 1
        results = image_index.search(["cat"], top_k=5)[0]
        nids = [r["note_id"] for r in results]
        assert nids.count(1) == 1  # one result for the note, not one per vector

    def test_remove_drops_all_note_vectors(self, image_index: VectorIndex) -> None:
        image_index.set_image_resolver(_resolver({"a.png": b"AAA"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png"])])
        assert image_index.size == 2
        image_index.remove([1])
        assert image_index.size == 0  # remove-by-key drops text + image vectors

    def test_media_hash_changes_when_image_added(self, image_index: VectorIndex) -> None:
        # For an image backend, a note's hash folds in its image names → an image change re-embeds.
        assert image_index._note_hash("cat", []) != image_index._note_hash("cat", ["a.png"])
        assert image_index._note_hash("cat", ["a.png"]) != image_index._note_hash("cat", ["b.png"])
        # Order-independent (sorted): same set of names → same hash.
        assert image_index._note_hash("cat", ["a.png", "b.png"]) == image_index._note_hash(
            "cat", ["b.png", "a.png"]
        )

    def test_media_hash_ignored_for_text_backend(self, index: VectorIndex) -> None:
        # A text-only backend never folds image names in — no spurious re-embed on upgrade.
        assert index._note_hash("cat", []) == index._note_hash("cat", ["a.png"])

    def test_reconcile_reembeds_note_on_image_change(self, image_index: VectorIndex) -> None:
        image_index.set_image_resolver(_resolver({"a.png": b"AAA"}))
        image_index.rebuild([NoteEmbedInput(1, "cat", [])], col_mod=1, model_id="m")
        assert image_index.size == 1  # text only
        image_index.reconcile([NoteEmbedInput(1, "cat", ["a.png"])], col_mod=2, model_id="m")
        assert image_index.size == 2  # text + image after the image was added

    def test_check_drift_rebuilds_pre_restructure_index_for_image_backend(
        self, tmp_path: Path, image_backend: MagicMock
    ) -> None:
        from usearch.index import Index

        idx = VectorIndex(tmp_path / "index", backend=image_backend)
        idx._indexes = {TEXT: Index(ndim=NDIM, metric="cos", dtype="f32", multi=True)}
        idx._col_mod, idx._model_id, idx._schema = 5, "m", 1  # pre-#201a (v1) single-index layout
        # Same col_mod + model, but a v1 index mixed text+image under one key → rebuild to split.
        assert idx.check_drift(5, "m") is True

    def test_check_drift_no_rebuild_for_text_backend_pre_restructure(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        from usearch.index import Index

        idx = VectorIndex(tmp_path / "index", backend=embedding_service)
        idx._indexes = {TEXT: Index(ndim=NDIM, metric="cos", dtype="f32", multi=True)}
        idx._col_mod, idx._model_id, idx._schema = 5, "m", 1  # pre-#201a (v1) layout
        # A text-only backend never triggers a per-modality restructure → no upgrade rebuild.
        assert idx.check_drift(5, "m") is False

    def test_late_arriving_image_reembeds_on_reconcile(self, image_index: VectorIndex) -> None:
        # F1: a note authored *before* its image is stored. The name-only hash must NOT claim the
        # image (it didn't embed), so a later reconcile (once the image lands) re-embeds it —
        # keeping reconcile == full rebuild. A read-based resolver doubles as the presence check.
        store: dict[str, bytes] = {}
        image_index.set_image_resolver(lambda n: store.get(n))
        image_index.rebuild([NoteEmbedInput(1, "cat", ["late.png"])], col_mod=1, model_id="m")
        assert image_index.size == 1  # image missing at embed time → text vector only
        store["late.png"] = b"LATE"  # the media is stored later (store_media)
        image_index.reconcile([NoteEmbedInput(1, "cat", ["late.png"])], col_mod=2, model_id="m")
        assert image_index.size == 2  # reconcile noticed missing->present and embedded the image

    def test_permanently_missing_image_does_not_loop(self, image_index: VectorIndex) -> None:
        # The flip side of F1: a referenced-but-never-stored image must NOT re-embed every drift
        # (its name isn't folded into the hash, so the hash is stable).
        image_index.set_image_resolver(lambda _n: None)  # nothing ever resolves
        image_index.rebuild([NoteEmbedInput(1, "cat", ["gone.png"])], col_mod=1, model_id="m")
        assert image_index.size == 1
        image_index._embedding.embed_texts.reset_mock()
        image_index.reconcile([NoteEmbedInput(1, "cat", ["gone.png"])], col_mod=2, model_id="m")
        assert image_index.size == 1
        image_index._embedding.embed_texts.assert_not_called()  # no spurious re-embed


class TestPerModality:
    """Per-modality sub-indexes + search_by_modality (#201a): text and image vectors live in
    separate USearch sub-indexes, so a query can be ranked independently per modality."""

    def test_add_routes_text_and_image_to_separate_subindexes(
        self, image_index: VectorIndex
    ) -> None:
        image_index.set_image_resolver(_resolver({"a.png": b"AAA", "b.png": b"BBB"}))
        image_index.add(
            [NoteEmbedInput(1, "cat", ["a.png", "b.png"]), NoteEmbedInput(2, "dog", [])]
        )
        assert len(image_index._indexes[TEXT]) == 2  # one text vector per note
        assert len(image_index._indexes[IMAGE]) == 2  # two image vectors, both under note 1
        assert 1 in image_index._indexes[IMAGE]
        assert 2 not in image_index._indexes[IMAGE]  # note 2 has no images

    def test_text_only_backend_makes_no_image_subindex(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2], ["cat", "dog"]))
        assert set(index._indexes) == {TEXT}

    def test_search_by_modality_ranks_each_modality(self, image_index: VectorIndex) -> None:
        image_index.set_image_resolver(_resolver({"a.png": b"AAA"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png"]), NoteEmbedInput(2, "dog", [])])
        rankings = image_index.search_by_modality(["cat"], top_k=5)
        assert len(rankings) == 1
        r = rankings[0]
        # The text query matches its own note's text vector best → note 1 leads the text ranking;
        # both text notes appear.
        assert r["text"][0]["note_id"] == 1
        assert {h["note_id"] for h in r["text"]} == {1, 2}
        # Only note 1 has an image, so the image ranking contains just note 1.
        assert [h["note_id"] for h in r["image"]] == [1]

    def test_image_ranking_dedups_to_best_per_note(self, image_index: VectorIndex) -> None:
        # Max-sim-over-items: a note with several images appears once, at its best (min) distance.
        image_index.set_image_resolver(_resolver({"a.png": b"AAA", "b.png": b"BBB"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png", "b.png"])])
        img = image_index.search_by_modality(["cat"], top_k=5)[0]["image"]
        assert [h["note_id"] for h in img] == [1]

    def test_search_by_modality_omits_empty_modalities(self, index: VectorIndex) -> None:
        index.add(_inp([1, 2], ["cat", "dog"]))
        rankings = index.search_by_modality(["cat"], top_k=5)
        assert set(rankings[0]) == {"text"}  # no image sub-index → no image key

    def test_search_uses_text_vectors_not_image(self, image_index: VectorIndex) -> None:
        # The neighbour search() path is text-backed: querying "cat" self-matches note 1's *text*
        # vector at distance ~0, not its (unrelated, nonzero-distance) image vector.
        image_index.set_image_resolver(_resolver({"a.png": b"AAA"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png"])])
        results = image_index.search(["cat"], top_k=5)[0]
        assert results[0]["note_id"] == 1
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-4)

    def test_remove_clears_every_subindex(self, image_index: VectorIndex) -> None:
        image_index.set_image_resolver(_resolver({"a.png": b"AAA"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png"])])
        assert len(image_index._indexes[IMAGE]) == 1
        image_index.remove([1])
        assert len(image_index._indexes[TEXT]) == 0
        assert len(image_index._indexes[IMAGE]) == 0

    def test_readd_with_fewer_images_drops_stale_image_vectors(
        self, image_index: VectorIndex
    ) -> None:
        # Re-adding a note that lost its images must not leave orphaned image vectors behind.
        image_index.set_image_resolver(_resolver({"a.png": b"AAA"}))
        image_index.add([NoteEmbedInput(1, "cat", ["a.png"])])
        assert len(image_index._indexes[IMAGE]) == 1
        image_index.add([NoteEmbedInput(1, "cat", [])])  # same note, no images now
        assert len(image_index._indexes[IMAGE]) == 0
        assert len(image_index._indexes[TEXT]) == 1


class TestPerModalityMigration:
    """v1 (pre-#201a) → v2 (per-modality) index migration and on-disk layout (#201a)."""

    def test_image_index_persists_per_modality_files(
        self, tmp_path: Path, image_backend: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "i", backend=image_backend)
        idx.set_image_resolver(_resolver({"a.png": b"AAA"}))
        idx.rebuild([NoteEmbedInput(1, "cat", ["a.png"])], col_mod=1, model_id="m")
        assert (tmp_path / "i" / "index.usearch").exists()  # text sub-index keeps the original name
        assert (tmp_path / "i" / "index.image.usearch").exists()  # image sub-index is a new file
        reloaded = VectorIndex(tmp_path / "i", backend=image_backend)
        assert reloaded.size == 2
        assert reloaded._schema == INDEX_SCHEMA_VERSION
        assert "image" in reloaded.search_by_modality(["cat"], top_k=5)[0]

    def test_text_only_v1_index_loads_as_text_without_rebuild(
        self, tmp_path: Path, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "i", backend=embedding_service)
        idx.rebuild(_inp([1, 2], ["a", "b"]), col_mod=1, model_id="m")
        # Strip the schema marker to simulate a pre-#201a (v1) index on disk.
        meta_path = tmp_path / "i" / "index.meta.json"
        meta = json.loads(meta_path.read_text())
        del meta["schema"]
        meta_path.write_text(json.dumps(meta))

        reloaded = VectorIndex(tmp_path / "i", backend=embedding_service)
        assert reloaded._schema == 1  # recognised as v1
        assert reloaded.size == 2  # the v1 index.usearch loaded straight into the text sub-index
        assert set(reloaded._indexes) == {TEXT}
        assert reloaded.check_drift(1, "m") is False  # text-only → no restructure rebuild

    def test_save_removes_stale_image_file_after_backend_switch(
        self, tmp_path: Path, image_backend: MagicMock, embedding_service: MagicMock
    ) -> None:
        idx = VectorIndex(tmp_path / "i", backend=image_backend)
        idx.set_image_resolver(_resolver({"a.png": b"AAA"}))
        idx.rebuild([NoteEmbedInput(1, "cat", ["a.png"])], col_mod=1, model_id="m")
        assert (tmp_path / "i" / "index.image.usearch").exists()
        # Switch to a text-only backend and rebuild: no image sub-index, and the stale file is gone
        # so it isn't reloaded as a phantom image index on the next startup.
        idx.set_backend(embedding_service)
        idx.rebuild(_inp([1], ["cat"]), col_mod=2, model_id="m2")
        assert set(idx._indexes) == {TEXT}
        assert not (tmp_path / "i" / "index.image.usearch").exists()
