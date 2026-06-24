"""The kernel-mode server core: Harness over a real AsyncKernel.

Embedding-free assembly: the kernel opens on the loop with the harness
thread driving its executor, the wrapper rides run_job, the derived store
builds on drift, and the operational verbs return the wire shapes the
routes serve — all without a model, mirroring a no-embedding boot.
"""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness, KernelConfigError  # noqa: E402

from .conftest import _assemble  # noqa: E402


class TestDerivedNamespaceParity:
    """The host DerivedTextStore (the /status read surface) and the
    kernel's DerivedEngine must open the SAME per-collection derived namespace.

    The namespace canonicalizes the collection path, and that canonicalization
    differs by whether the file EXISTS at computation time (existing → realpath,
    which folds a symlinked prefix; absent → a lexical abspath that does NOT).
    Building the host store BEFORE the kernel creates a fresh collection's file
    would hash it under the abspath namespace while the kernel uses the realpath
    one — host /status reads an EMPTY store while the kernel's search store holds
    the rows. Harness.assemble builds the host store AFTER open, so the file
    exists for both and they realpath identically.
    """

    def test_host_store_and_kernel_resolve_same_path_for_fresh_collection(self, tmp_path) -> None:
        # A FRESH collection (file absent at assemble time) reached via a
        # SYMLINKED prefix (so abspath != realpath, like macOS
        # /var/folders -> /private/var/...): post-open, the host store realpaths
        # to the kernel's namespace.
        from shrike.harness import cache_layout

        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        # Pass the SYMLINKED spelling — abspath keeps the link, realpath folds it.
        cache_dir = str(link / "cache")
        collection = str(link / "collection.anki2")  # does NOT exist yet

        async def flow():
            runtime = EmbeddingRuntime(model=None)
            # No `derived` injected: assemble resolves it post-open.
            harness = await Harness.assemble(
                collection_path=collection,
                cache_dir=cache_dir,
                runtime=runtime,
                cooperative=False,
                hold_seconds=5.0,
                media_read=None,
                media_exists=None,
            )
            try:
                # The kernel's own path (Rust) and the host store's path agree —
                # and equal the host recomputation now that the file exists.
                kernel_path = shrike_native.derived_db_path(cache_dir, collection)
                assert str(harness.derived._path) == kernel_path
                assert cache_layout.derived_db_path(cache_dir, collection) == kernel_path
            finally:
                await harness.close()

        asyncio.run(flow())

    def test_host_status_store_sees_rows_through_boot(self, tmp_path) -> None:
        # End-to-end via the production boot path (no injected derived store):
        # upsert a note, boot (which builds the derived store kernel-side and
        # settles the host read surface), and assert the host /status store —
        # built by assemble at the kernel's namespace — reports ready AND a
        # substring query finds the row. The symlinked prefix + fresh collection
        # is the condition under which a stale abspath namespace would read empty.
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        cache_dir = str(link / "cache")
        collection = str(link / "collection.anki2")

        async def flow():
            runtime = EmbeddingRuntime(model=None)
            harness = await Harness.assemble(
                collection_path=collection,
                cache_dir=cache_dir,
                runtime=runtime,
                cooperative=False,
                hold_seconds=5.0,
                media_read=None,
                media_exists=None,
            )
            try:
                await harness.wrapper.upsert_notes(
                    [
                        {
                            "note_type": "Basic",
                            "deck": "Default",
                            "fields": {"Front": "the krebs cycle", "Back": "citric acid"},
                        }
                    ]
                )
                await harness.boot(start_embedding=False)
                # boot() drives the maintenance to quiescence and opens the data
                # plane — ready on return, no status poll + sleep.
                assert harness.is_ready
                assert harness.derived.status()["state"] == "ready"
                hits = harness.derived.search_substring("krebs", 10)
                assert hits, "host store must see the rows on the shared shrike.db"
            finally:
                await harness.close()

        asyncio.run(flow())


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
            # Recognition is off until a backend is configured: the
            # keyed-by-source map is empty (distinct from attached-but-errored).
            assert status["recognition"] == {}
            # The cross-modal coverage matrix: shape-stable, every
            # (query, target) cell `unavailable` with embedding down — nothing
            # is reachable natively or via derived text.
            assert status["coverage"] == {
                "text": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
                "image": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
                "audio": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
            }

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
            # boot() awaits the drift build to quiescence — ready, no poll.
            assert harness.is_ready
            assert harness.derived.status()["state"] == "ready"
            await harness.close()

        asyncio.run(flow())


class _FakeRouterManager:
    """A stand-in for shrike_native.LlamaServerManager.router(...) — records
    start/stop calls and reports running across them, so the harness's
    spawn-once / owner-only-stop lifecycle is provable without a real
    llama-server."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self._running = False

    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.starts += 1
        self._running = True

    def stop(self) -> None:
        self.stops += 1
        self._running = False


class TestSharedRouterLifecycle:
    """The shared llama.cpp router manager is spawned ONCE and stopped
    only by the OWNER — never N spawns, never killed out from under a routed
    (non-owning) harness."""

    async def _assemble_with_router(self, tmp_path, mgr, *, owns_runtime: bool) -> Harness:
        runtime = EmbeddingRuntime(model=None)
        derived = DerivedTextStore(
            path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
        )
        return await Harness.assemble(
            collection_path=str(tmp_path / "collection.anki2"),
            cache_dir=str(tmp_path / "cache"),
            runtime=runtime,
            derived=derived,
            cooperative=False,
            hold_seconds=5.0,
            media_read=None,
            media_exists=None,
            owns_runtime=owns_runtime,
            shared_llama_manager=mgr,
        )

    def test_ensure_router_spawns_once_then_owner_stops_it(self, tmp_path) -> None:
        async def flow():
            mgr = _FakeRouterManager()
            harness = await self._assemble_with_router(tmp_path, mgr, owns_runtime=True)
            # First ensure spawns it; a second ensure is a no-op (already up) —
            # this is what prevents N spawns when N spaces each trigger a start.
            await harness._ensure_shared_router()
            await harness._ensure_shared_router()
            assert mgr.starts == 1
            assert mgr.running()
            # The owner stops it exactly once on close.
            await harness.close()
            assert mgr.stops == 1
            assert not mgr.running()

        asyncio.run(flow())

    def test_non_owner_close_never_stops_the_shared_router(self, tmp_path) -> None:
        async def flow():
            mgr = _FakeRouterManager()
            harness = await self._assemble_with_router(tmp_path, mgr, owns_runtime=False)
            await harness._ensure_shared_router()
            assert mgr.starts == 1
            # A routed harness does not own the runtime, so its close must
            # leave the shared router running for the owner + siblings.
            await harness.close()
            assert mgr.stops == 0
            assert mgr.running()

        asyncio.run(flow())

    def test_ensure_router_respawns_after_a_stop(self, tmp_path) -> None:
        # The stop→start cycle (embedding stop then start): once the router is
        # stopped, a later _ensure_shared_router must respawn it (the guard keys
        # on running(), so a stopped manager starts again).
        async def flow():
            mgr = _FakeRouterManager()
            harness = await self._assemble_with_router(tmp_path, mgr, owns_runtime=True)
            await harness._ensure_shared_router()
            assert mgr.starts == 1 and mgr.running()
            # Simulate `embedding stop` freeing the router.
            mgr.stop()
            assert not mgr.running()
            # The next start cycle respawns it (not a no-op against a dead one).
            await harness._ensure_shared_router()
            assert mgr.starts == 2 and mgr.running()
            await harness.close()
            assert not mgr.running()

        asyncio.run(flow())
