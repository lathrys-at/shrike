"""The runtime-less asyncio bridge (#332, S3a).

Kernel futures awaited natively on the asyncio loop: each poll is a plain loop
callback, wakes re-enter via ``call_soon_threadsafe``, no tokio and no threads
owned by the binding. These tests prove the full chain — open → serialized
collection ops → close — plus cancellation and concurrent awaits.
"""

from __future__ import annotations

import asyncio
import gc
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


def _settled_live_callbacks() -> int:
    """The live-callback count after collecting any residue.

    Earlier tests in the same process can leave callback↔future cycles
    (e.g. a cancellation whose done callbacks never ran before its loop
    closed) that OUR ``gc.collect()`` would otherwise sweep mid-assertion —
    so both the baseline and the final count are read post-collection.
    Cycles can chain, so collect a few passes.
    """
    for _ in range(3):
        gc.collect()
    return int(shrike_native.bridge_live_poll_callbacks())


class TestBridgeLifecycle:
    """#387: the bridge releases its state however an op's observation ends.

    ``bridge_parked_forever`` is a never-resolving future that *retains its
    waker* — the in-flight-op shape (a oneshot receiver parks its waker the
    same way) whose stored waker once formed a Rust-side reference cycle
    (future → waker → callback → future slot) invisible to Python's GC.
    ``bridge_live_poll_callbacks`` counts live callbacks Rust-side, so these
    assertions don't depend on the GC even seeing the objects.
    """

    def test_abandoned_loop_releases_bridge_state(self) -> None:
        baseline = _settled_live_callbacks()
        holder: list = []

        async def start() -> None:
            holder.append(shrike_native.bridge_parked_forever())
            await asyncio.sleep(0)  # let the first poll park the waker

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(start())
            assert shrike_native.bridge_live_poll_callbacks() > baseline
        finally:
            loop.close()
        # The loop died with the op still pending; dropping the asyncio
        # future must release the callback — the waker's weak reference and
        # the GC-visible callback↔future cycle are exactly what #387 fixed.
        del holder[:], loop
        assert _settled_live_callbacks() == baseline

    def test_cancellation_releases_bridge_state_promptly(self) -> None:
        baseline = _settled_live_callbacks()

        async def flow() -> None:
            fut = shrike_native.bridge_parked_forever()
            await asyncio.sleep(0)
            fut.cancel()
            with pytest.raises(asyncio.CancelledError):
                await fut
            # The done callback fired on the cancellation itself — the Rust
            # future is dropped without waiting for a wake that never comes.
            for _ in range(3):
                await asyncio.sleep(0)

        asyncio.run(flow())
        assert _settled_live_callbacks() == baseline

    def test_completion_releases_bridge_state(self, tmp_path) -> None:
        baseline = _settled_live_callbacks()

        async def flow() -> None:
            col = await shrike_native.async_collection_open(str(tmp_path / "c.anki2"))
            assert isinstance(await col.col_mod(), int)
            await col.close()

        asyncio.run(flow())
        assert _settled_live_callbacks() == baseline


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
