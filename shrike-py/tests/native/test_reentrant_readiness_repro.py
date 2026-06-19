"""Regression: two concurrent re-acquires must still open the readiness gate.

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


def test_two_concurrent_reacquires_still_open_the_gate(tmp_path) -> None:
    async def flow() -> None:
        harness = await _assemble(tmp_path)
        await harness.boot(start_embedding=False)
        assert harness.is_ready

        col_mod = await harness.wrapper.col_mod()
        harness._spawn_reacquire_tasks(col_mod)
        harness._spawn_reacquire_tasks(col_mod)

        for _ in range(1000):
            await asyncio.sleep(0)
            if harness.is_ready and all(t.done() for t in harness._bg_tasks):
                break

        # The gate must be open and every marker/maintenance task settled — the
        # pre-fix cycle left two markers hung forever (is_ready False, tasks
        # never done). close() drains anything still in flight either way.
        gated = not harness.is_ready
        settled = all(t.done() for t in harness._bg_tasks)
        await harness.close()
        assert not gated, "data plane gated forever after two re-acquires"
        assert settled, "re-acquire tasks left hung (settle-marker cycle)"

    asyncio.run(flow())
