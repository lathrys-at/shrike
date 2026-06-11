"""The kernel-mode server core (#332 S3d-2d): Harness over a real AsyncKernel.

Embedding-free assembly: the kernel opens on the loop with the harness
thread driving its executor, the wrapper rides run_job, the derived store
builds on drift, and the operational verbs return the wire shapes the
routes serve — all without a model, mirroring a no-embedding boot.
"""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.embedding import EmbeddingRuntime  # noqa: E402
from shrike.harness import Harness, KernelConfigError  # noqa: E402


async def _assemble(tmp_path, *, cooperative: bool = False) -> Harness:
    runtime = EmbeddingRuntime(model=None)
    derived = DerivedTextStore(
        path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
    )
    return await Harness.assemble(
        collection_path=str(tmp_path / "collection.anki2"),
        cache_dir=str(tmp_path / "cache"),
        runtime=runtime,
        derived=derived,
        cooperative=cooperative,
        hold_seconds=5.0,
        media_read=None,
        media_exists=None,
    )


class TestHarness:
    def test_boot_status_and_verbs_without_embedding(self, tmp_path) -> None:
        async def flow():
            harness = await _assemble(tmp_path)
            await harness.boot(start_embedding=False)

            # Ops flow through the wrapper → run_job → shared core.
            notes = await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "harness boot", "Back": "b"},
                    }
                ]
            )
            assert notes[0]["status"] == "created"

            status = await harness.status()
            assert status["embedding"]["state"] == "not_configured"
            assert status["index"]["state"] == "unavailable"
            assert status["locking"] == "permanent"

            # Index verbs degrade correctly without a backend.
            with pytest.raises(KernelConfigError):
                await harness.rebuild_index()
            assert (await harness.save_index())["status"] == "empty"
            assert (await harness.stop_embedding())["status"] == "not_running"

            # Reload re-opens and reports; no embedder → no rebuild.
            reloaded = await harness.reload()
            assert reloaded["status"] == "reloaded"
            assert reloaded["rebuilding"] is False

            await harness.close()

        asyncio.run(flow())

    def test_derived_store_builds_on_boot_drift(self, tmp_path) -> None:
        async def flow():
            harness = await _assemble(tmp_path)
            await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "the mitochondria", "Back": "powerhouse"},
                    }
                ]
            )
            await harness.boot(start_embedding=False)
            # The boot saw drift and built; wait for the background build.
            for _ in range(100):
                if harness.derived.status().get("state") == "ready":
                    break
                await asyncio.sleep(0.05)
            assert harness.derived.status()["state"] == "ready"
            await harness.close()

        asyncio.run(flow())
