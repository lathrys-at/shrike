"""Regression: two concurrent settle markers must still open the readiness gate.

Each cooperative re-acquire (or a /reload mid-boot) begins a new readiness
generation and spawns a `_settle_and_mark_ready` marker task. A marker's
`settle_background()` must gather only the maintenance it waits on, never a
sibling marker — otherwise marker-1 awaits marker-2 and marker-2 awaits
marker-1 (cyclic await), neither sets `_ready`, and the data plane is gated
forever. `settle_background` excludes every marker task, breaking the cycle so
the latest generation's marker opens the gate.
"""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness  # noqa: E402


async def _assemble(tmp_path) -> Harness:
    runtime = EmbeddingRuntime(model=None)
    derived = DerivedTextStore(
        path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
    )
    return await Harness.assemble(
        collection_path=str(tmp_path / "collection.anki2"),
        cache_dir=str(tmp_path / "cache"),
        runtime=runtime,
        derived=derived,
        cooperative=True,
        hold_seconds=5.0,
        media_read=None,
        media_exists=None,
    )


def test_two_concurrent_settle_markers_still_open_the_gate(tmp_path) -> None:
    # Two concurrent readiness generations each begin a generation and spawn a
    # `_settle_and_mark_ready` marker. The markers run `settle_background`, which
    # must NOT gather a sibling marker — otherwise marker-1 awaits marker-2 and
    # marker-2 awaits marker-1 (cyclic await), neither sets `_ready`, and the
    # data plane is gated forever. The maintenance each marker actually waits on
    # is a trivial controllable task here, so the test exercises the
    # marker-exclusion logic deterministically without a real-kernel re-acquire
    # (whose rebuild churn is orthogonal to this asyncio-level cycle).
    async def flow() -> None:
        harness = await _assemble(tmp_path)
        await harness.boot(start_embedding=False)
        assert harness.is_ready

        gate = asyncio.Event()

        async def maintenance() -> None:
            await gate.wait()

        # Two overlapping generations: each clears `_ready`, queues a maintenance
        # task, and spawns its marker (the exact shape of two back-to-back
        # re-acquires, minus the kernel ops).
        for _ in range(2):
            harness._begin_generation()
            harness._spawn_bg(maintenance())
            harness._spawn_marker(harness._generation)

        assert not harness.is_ready, "the overlapping generations cleared the gate"
        # Let the markers reach their settle_background await, then release the
        # maintenance both wait on. With the cycle present, the markers would be
        # awaiting EACH OTHER, so releasing the maintenance would not free them.
        await asyncio.sleep(0)
        gate.set()

        try:
            await asyncio.wait_for(harness.await_ready(), timeout=30)
            gated = False
        except TimeoutError:
            gated = True

        await harness.settle_background()
        await asyncio.sleep(0)
        settled = all(t.done() for t in harness._bg_tasks)

        await harness.close()
        assert not gated, "data plane gated forever after two concurrent markers"
        assert settled, "marker tasks left hung (settle-marker cycle)"

    asyncio.run(flow())


def test_readiness_does_not_block_on_a_live_recognition_sweep(tmp_path) -> None:
    # Readiness is "the index/derived maintenance has settled". A recognition
    # sweep (OCR/ASR, many seconds) is NOT that maintenance, so settle_background
    # — which a re-acquire/reload readiness marker awaits — must NOT block on it.
    # Otherwise a re-acquire overlapping a live sweep parks the data plane behind
    # the whole sweep (or the 30s gate fail-safe). The sweep's own writes still
    # serialize through the kernel; close() drains it directly.
    async def flow() -> None:
        harness = await _assemble(tmp_path)
        await harness.boot(start_embedding=False)

        sweeping = asyncio.Event()

        async def long_sweep() -> None:
            # Stand in for a multi-second OCR/ASR sweep that never finishes
            # within the test.
            await sweeping.wait()

        # Install it exactly as the harness tracks the recognition sweep.
        harness._recognition_task = harness._spawn_bg(long_sweep())

        # settle_background must return PROMPTLY, not wait for the sweep.
        await asyncio.wait_for(harness.settle_background(), timeout=5)
        assert not harness._recognition_task.done(), "the sweep is still running"

        sweeping.set()
        await harness.close()

    asyncio.run(flow())
