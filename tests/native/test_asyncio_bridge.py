"""The runtime-less asyncio bridge (#332, S3a).

Kernel futures awaited natively on the asyncio loop: each poll is a plain loop
callback, wakes re-enter via ``call_soon_threadsafe``, no tokio and no threads
owned by the binding. These tests prove the full chain — open → serialized
collection ops → close — plus cancellation and concurrent awaits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "async_collection_open"),
    reason="anki-core build required (scripts/build-native.sh)",
)


def _run(coro):
    return asyncio.run(coro)


class TestAsyncioBridge:
    def test_open_op_close_round_trip(self, tmp_path) -> None:
        async def flow() -> int:
            col = await shrike_native.async_collection_open(str(tmp_path / "c.anki2"))
            mod = await col.col_mod()
            assert isinstance(mod, int)
            assert await col.find_notes("deck:*") == []
            await col.close()
            return mod

        assert _run(flow()) >= 0

    def test_ops_serialize_under_concurrent_awaits(self, tmp_path) -> None:
        # gather() drives many bridged futures on one loop; the injected
        # executor serializes the underlying collection jobs.
        async def flow() -> list[list[int]]:
            col = await shrike_native.async_collection_open(str(tmp_path / "c.anki2"))
            results = await asyncio.gather(*(col.find_notes("deck:*") for _ in range(8)))
            await col.close()
            return results

        assert _run(flow()) == [[] for _ in range(8)]

    def test_native_errors_surface_as_exceptions(self, tmp_path) -> None:
        async def flow() -> None:
            col = await shrike_native.async_collection_open(str(tmp_path / "c.anki2"))
            with pytest.raises(shrike_native.NativeInputError):
                await col.find_notes("prop:bogus(((")
            await col.close()

        _run(flow())

    def test_cancellation_is_cooperative(self, tmp_path) -> None:
        # Cancelling the asyncio.Future must not crash the bridge; later ops
        # on the same collection keep working.
        async def flow() -> None:
            col = await shrike_native.async_collection_open(str(tmp_path / "c.anki2"))
            fut = col.col_mod()
            fut.cancel()
            with pytest.raises(asyncio.CancelledError):
                await fut
            assert isinstance(await col.col_mod(), int)
            await col.close()

        _run(flow())

    def test_interoperates_with_the_sync_core(self, tmp_path) -> None:
        # The async surface reads what the sync core wrote (separate opens —
        # sequential, never concurrent on one file).
        path = str(tmp_path / "c.anki2")
        core = shrike_native.CollectionCore(path)
        out = json.loads(
            core.upsert_notes(
                json.dumps(
                    [{"note_type": "Basic", "deck": "D", "fields": {"Front": "Q", "Back": "A"}}]
                ),
                "allow",
                False,
            )
        )
        assert out[0]["status"] == "created"
        core.close()

        async def flow() -> list[int]:
            col = await shrike_native.async_collection_open(path)
            ids = await col.find_notes("deck:D")
            await col.close()
            return ids

        assert len(_run(flow())) == 1


class TestWorkerExecutor:
    """S3b: the harness-OWNED worker thread drives collection jobs off the loop."""

    def test_jobs_run_on_the_harness_thread(self, tmp_path) -> None:
        import threading

        ex = shrike_native.WorkerExecutor()
        worker = threading.Thread(target=ex.worker_loop, daemon=True)
        worker.start()

        async def flow() -> list[list[int]]:
            col = await shrike_native.async_collection_open(str(tmp_path / "c.anki2"), executor=ex)
            results = await asyncio.gather(*(col.find_notes("deck:*") for _ in range(4)))
            await col.close()
            return results

        assert _run(flow()) == [[] for _ in range(4)]
        ex.shutdown()
        worker.join(timeout=10)
        assert not worker.is_alive()

    def test_worker_loop_is_single_claim(self) -> None:
        ex = shrike_native.WorkerExecutor()
        ex.shutdown()  # close the queue so the claimed loop returns immediately
        ex.worker_loop()
        with pytest.raises(shrike_native.NativeInternalError):
            ex.worker_loop()  # the receiver is already claimed


class TestLoopTimerHost:
    """S3c-1: the harness's asyncio timers, injected into the kernel's TimerHost port."""

    def test_timer_fires_through_the_loop(self) -> None:
        async def flow() -> bool:
            host = shrike_native.LoopTimerHost.capture()
            return await shrike_native.timer_probe(host, 0.01)

        assert _run(flow()) is True

    def test_cancel_suppresses_the_job(self) -> None:
        async def flow() -> bool:
            host = shrike_native.LoopTimerHost.capture()
            # Cancel (at 10ms) far before the 5s job would fire.
            return await shrike_native.timer_probe(host, 5.0, cancel_after=0.01)

        assert _run(flow()) is False


class TestPyEmbedder:
    """S3c-2b: the kernel's Embedder seam driven by a harness backend."""

    class _Backend:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.calls: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            if self.fail:
                raise RuntimeError("backend down")
            self.calls.append(list(texts))
            return [[float(len(t)), 1.0] for t in texts]

    def test_kernel_trait_drives_the_python_backend(self) -> None:
        backend = self._Backend()

        async def flow() -> list[list[float]]:
            emb = shrike_native.PyEmbedder.capture(backend)
            return await shrike_native.embedder_probe(emb, ["ab", "cdef"])

        assert _run(flow()) == [[2.0, 1.0], [4.0, 1.0]]
        assert backend.calls == [["ab", "cdef"]]

    def test_backend_failure_maps_to_unavailable(self) -> None:
        async def flow() -> None:
            emb = shrike_native.PyEmbedder.capture(self._Backend(fail=True))
            with pytest.raises(shrike_native.NativeUnavailableError, match="backend down"):
                await shrike_native.embedder_probe(emb, ["x"])

        _run(flow())
