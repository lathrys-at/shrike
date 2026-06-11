"""The kernel index orchestrator binding (#332, S3c-4).

``KernelIndex`` re-homes the ``VectorIndex`` orchestration (drift, per-note
fingerprints, sidecars, state machine, reconcile/rebuild) into the kernel,
sharing ONE engine with the harness's ``NativeIndexEngine`` search handle.
The async ops run on the asyncio loop via the bridge, embedding through the
harness backend by ``PyEmbedder`` — these tests prove the whole inversion,
plus persistence round-trips against the Python orchestrator's file formats.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

shrike_native = pytest.importorskip("shrike_native")


def _run(coro):
    return asyncio.run(coro)


class _Backend:
    """Deterministic 4-dim unit vectors keyed off a text hash."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            b = hashlib.blake2b(text.encode(), digest_size=1).digest()[0] / 255.0
            n = (b * b + 1.0) ** 0.5
            out.append([b / n, 1.0 / n, 0.0, 0.0])
        return out


def _make(tmp_path):
    engine = shrike_native.NativeIndexEngine(["text", "image"])
    index = shrike_native.KernelIndex(str(tmp_path), engine)
    return engine, index


class TestKernelIndex:
    def test_rebuild_drives_the_shared_engine(self, tmp_path) -> None:
        async def flow():
            engine, index = _make(tmp_path)
            backend = _Backend()
            embedder = shrike_native.PyEmbedder.capture(backend)
            assert index.state() == "unavailable"
            await index.rebuild([(1, "alpha", []), (2, "beta", [])], 10, "m", embedder)
            return engine, index, backend

        engine, index, backend = _run(flow())
        assert index.state() == "ready"
        assert index.col_mod() == 10
        assert index.model_id() == "m"
        assert index.has_note_hashes()
        # The SAME engine the harness searches now holds the vectors.
        assert engine.size() == 2
        assert sorted(engine.keys()) == [1, 2]
        assert backend.calls, "the embed went through the harness backend"
        assert not index.check_drift(10, "m")
        assert index.check_drift(11, "m")
        assert index.check_drift(10, "other")

    def test_reconcile_is_incremental_and_watermark_aware(self, tmp_path) -> None:
        async def flow():
            engine, index = _make(tmp_path)
            backend = _Backend()
            embedder = shrike_native.PyEmbedder.capture(backend)
            inputs = [(1, "one", []), (2, "two", [])]
            await index.rebuild(inputs, 1, "m", embedder)
            rebuild_calls = len(backend.calls)

            # Watermark-only drift: same content, new col_mod → no embeds.
            await index.reconcile(inputs, 2, "m", embedder)
            assert len(backend.calls) == rebuild_calls
            assert index.col_mod() == 2

            # A real edit re-embeds only the changed note; a vanished one drops.
            await index.reconcile([(1, "one EDITED", []), (3, "three", [])], 3, "m", embedder)
            return engine, index, backend

        engine, index, backend = _run(flow())
        assert index.col_mod() == 3
        assert sorted(engine.keys()) == [1, 3]
        assert backend.calls[-1] == ["one EDITED", "three"]

    def test_add_and_remove_maintain_fingerprints(self, tmp_path) -> None:
        async def flow():
            engine, index = _make(tmp_path)
            embedder = shrike_native.PyEmbedder.capture(_Backend())
            await index.rebuild([(1, "seed", [])], 1, "m", embedder)
            added = await index.add([(2, "later", [])], embedder)
            assert added == 1
            # Re-adding note 1 with new text replaces, not duplicates.
            await index.add([(1, "seed v2", [])], embedder)
            return engine, index

        engine, index = _run(flow())
        assert sorted(engine.keys()) == [1, 2]
        assert index.remove([1]) == 1
        assert engine.keys() == [2]

    def test_persistence_round_trips_for_a_fresh_handle(self, tmp_path) -> None:
        async def flow():
            _, index = _make(tmp_path)
            embedder = shrike_native.PyEmbedder.capture(_Backend())
            await index.rebuild([(1, "alpha", []), (2, "beta", [])], 7, "m", embedder)
            index.save()

        _run(flow())
        engine2 = shrike_native.NativeIndexEngine(["text", "image"])
        reopened = shrike_native.KernelIndex(str(tmp_path), engine2)
        assert engine2.size() == 2
        assert reopened.col_mod() == 7
        assert reopened.has_note_hashes()
        assert not reopened.check_drift(7, "m")
        status = json.loads(reopened.status_json())
        assert status["size"] == 2
        assert status["col_mod"] == 7

    def test_materialize_empty_is_ready(self, tmp_path) -> None:
        _, index = _make(tmp_path)
        index.materialize_empty(4, 3, "m")
        assert index.state() == "ready"
        assert not index.check_drift(3, "m")

    def test_embed_failure_surfaces_and_marks_error(self, tmp_path) -> None:
        class _Broken:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("backend down")

        async def flow():
            _, index = _make(tmp_path)
            embedder = shrike_native.PyEmbedder.capture(_Broken())
            with pytest.raises(shrike_native.NativeUnavailableError):
                await index.rebuild([(1, "alpha", [])], 1, "m", embedder)
            return index

        index = _run(flow())
        assert index.state() == "error"


class TestKernelIndexSaver:
    def test_debounce_flushes_after_the_delay(self, tmp_path) -> None:
        async def flow():
            engine, index = _make(tmp_path)
            embedder = shrike_native.PyEmbedder.capture(_Backend())
            await index.rebuild([(1, "alpha", [])], 1, "m", embedder)
            await index.add([(2, "beta", [])], embedder)
            host = shrike_native.LoopTimerHost.capture()
            saver = shrike_native.KernelIndexSaver(index, host, 0.05, 100)
            saver.request_save()
            assert saver.pending_changes() == 1
            await asyncio.sleep(0.3)
            assert saver.pending_changes() == 0

        _run(flow())
        # The debounced flush persisted the post-rebuild add.
        engine2 = shrike_native.NativeIndexEngine(["text", "image"])
        shrike_native.KernelIndex(str(tmp_path), engine2)
        assert engine2.size() == 2

    def test_burst_cap_flushes_immediately(self, tmp_path) -> None:
        async def flow():
            _, index = _make(tmp_path)
            embedder = shrike_native.PyEmbedder.capture(_Backend())
            await index.rebuild([(1, "alpha", [])], 1, "m", embedder)
            host = shrike_native.LoopTimerHost.capture()
            saver = shrike_native.KernelIndexSaver(index, host, 60.0, 2)
            saver.request_save()
            saver.request_save()  # hits the cap → immediate flush
            assert saver.pending_changes() == 0

        _run(flow())
