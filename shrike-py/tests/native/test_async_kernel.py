"""The full kernel binding.

``AsyncKernel`` is the assembled kernel driven from asyncio: one open
collection + kernel-internal index orchestration + the derived store, every
op an awaitable. The harness supplies its parts — a worker executor, a
``PyEmbedder`` over its backend, the loop's timers — and shares the kernel's
engine/core handles for its own read/search surfaces.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "async_kernel_open"),
    reason="anki-core build required (scripts/build-native.sh)",
)

from .conftest import _Backend, _open  # noqa: E402


class TestRebuildDerived:
    """The FTS5 rebuild runs kernel-side — rows never cross the FFI; the op
    returns (row_count, the build's col_mod snapshot)."""

    def test_rebuild_derived_builds_and_returns_snapshot(self, tmp_path) -> None:
        from shrike.harness import cache_layout

        async def flow():
            collection_path = str(tmp_path / "collection.anki2")
            kernel = await shrike_native.async_kernel_open(collection_path, str(tmp_path / "cache"))
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes(
                [
                    (basic, 1, ["the krebs cycle", "citric acid"], []),
                    (basic, 1, ["unrelated front", "unrelated back"], []),
                ],
                "allow",
            )
            rows, dmod = await kernel.rebuild_derived()
            assert rows == 4  # 2 notes x 2 non-empty fields
            assert dmod == core.col_mod()
            await kernel.close()
            # The build landed in the sidecar: a fresh engine on the same
            # shrike.db sees the rows (and the stamped watermark). The store is
            # namespaced per collection, so resolve the path the kernel
            # wrote, not the flat cache root.
            db_path = cache_layout.derived_db_path(str(tmp_path / "cache"), collection_path)
            from shrike.harness.derived import SCHEMA_VERSION

            engine = shrike_native.DerivedTextEngine(db_path, SCHEMA_VERSION)
            try:
                assert engine.get_col_mod() == dmod
                hits = engine.search_substring("krebs", 10)
                assert hits, "the rebuilt FTS5 store must match the seeded text"
            finally:
                engine.close()

        asyncio.run(flow())


class TestSaverTuning:
    """The --index-save-* tuning reaches the kernel's saver."""

    def test_save_threshold_flushes_immediately(self, tmp_path) -> None:
        # threshold=1: the first indexed change forces a flush, so the index
        # lands on disk without an explicit save_index() or close().
        from shrike.harness import cache_layout

        async def flow():
            backend = _Backend()
            collection_path = str(tmp_path / "collection.anki2")
            kernel = await shrike_native.async_kernel_open(
                collection_path,
                str(tmp_path / "cache"),
                save_threshold=1,
            )
            kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
            assert await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes([(basic, 1, ["flush me", "now"], [])], "allow")
            # The index lands in the per-collection namespace, not the
            # flat cache root — resolve the same dir the kernel writes.
            index_dir = cache_layout.collection_index_dir(str(tmp_path / "cache"), collection_path)
            index_file = Path(index_dir) / "index.usearch"
            # The threshold flush is async (a spawned save) — poll briefly.
            for _ in range(100):
                if index_file.exists():
                    break
                await asyncio.sleep(0.05)
            assert index_file.exists(), "threshold=1 must flush without an explicit save"
            await kernel.close()

        asyncio.run(flow())


class TestAsyncKernel:
    def test_upsert_search_delete_flow(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            assert await kernel.reindex_if_needed()  # empty → materialize
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            results = await kernel.upsert_notes(
                [
                    (basic, 1, ["the mitochondria powerhouse", "energy"], ["smoke"]),
                    (basic, 1, ["newton laws of motion", "mechanics"], []),
                    (basic, 1, ["the mitochondria powerhouse", "dupe"], []),
                    (basic, 1, ["", "empty first field"], []),
                ],
                "skip",
            )
            assert [r[0] for r in results] == ["created", "created", "skipped", "error"]
            assert results[0][1] is not None
            assert "first field" in results[3][2]
            await kernel.settle()

            # The kernel maintained the index: its shared engine handle sees
            # exactly the created notes' vectors.
            engine = kernel.engine_handle()
            created = [r[1] for r in results if r[0] == "created"]
            assert sorted(engine.keys()) == sorted(created)

            # Fused search finds the note with semantic + lexical signals.
            hits = await kernel.search("mitochondria powerhouse", 5)
            assert hits[0][0] == results[0][1]
            signals = [s for s, _ in hits[0][2]]
            assert "text" in signals
            assert "exact" in signals or "fuzzy" in signals

            # Watermarks advanced: no drift after the kernel's own writes.
            assert not await kernel.reindex_if_needed()

            # Delete propagates to vectors too. The maintained op returns
            # {deleted, not_found} JSON in its single write job.
            deleted = json.loads(await kernel.delete_notes([results[0][1]]))
            assert deleted == {"deleted": [results[0][1]], "not_found": []}
            await kernel.settle()
            assert sorted(engine.keys()) == sorted(created[1:])
            assert not await kernel.reindex_if_needed()

            status = json.loads(kernel.index_status_json())
            assert status["state"] == "ready"
            assert status["model_id"] == "test-backend:v1"
            await kernel.close()
            return backend

        backend = asyncio.run(flow())
        assert backend.calls, "embeds went through the harness backend"

    def test_search_lexical_single_matches_fused_lexical(self, tmp_path) -> None:
        """The dedicated single-query lexical routine returns the SAME wire as the
        general fused path with semantic disabled — it trims only orchestration (no
        embed, no compute-pool chunk fan-out), never behaviour. Parity holds across
        substring, fuzzy (typo), tag scope, and exclude."""

        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            assert await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [
                    (basic, 1, ["the mitochondria powerhouse", "energy"], ["smoke"]),
                    (basic, 1, ["mitochondrial membrane", "biology"], ["smoke"]),
                    (basic, 1, ["newton laws of motion", "mechanics"], []),
                ],
                "skip",
            )
            await kernel.settle()
            first = results[0][1]

            cases: list[tuple[str, int, dict[str, object]]] = [
                ("mitochondria", 10, {}),  # exact substring + fuzzy
                ("mitochondrai", 10, {}),  # transposed typo -> fuzzy
                ("mito", 10, {}),  # short substring
                ("newton", 10, {}),  # the other note
                ("zzzznomatch", 10, {}),  # no matches -> empty groups, both paths
                ("mitochondria", 10, {"tags": ["smoke"]}),  # tag-scoped
                ("mitochondria", 10, {"exclude": [first]}),  # exclude the top hit
                ("mitochondria", 10, {"deck": "Default"}),  # deck-scoped
                ("mitochondria", 0, {}),  # limit=0 -> the return-all sentinel branch
            ]
            exact_matches = 0
            for q, lim, kw in cases:
                single = json.loads(await kernel.search_lexical_single(q, lim, **kw))
                fused = json.loads(
                    await kernel.search_fused([(q, q, True)], lim, 0.5, semantic=False, **kw)
                )
                assert single == fused, f"lexical-single != fused-lexical for {q!r} lim={lim} {kw}"
                if q == "mitochondria" and lim == 10 and not kw:
                    exact_matches = sum(len(g["matches"]) for g in single["groups"])
            # Teeth: the plain exact-substring case must actually return hits, so the
            # parity assertions above are not vacuously comparing two empty results.
            assert exact_matches >= 1, "exact-substring 'mitochondria' must return matches"
            await kernel.close()

        asyncio.run(flow())

    def test_is_settled_tracks_the_ingest_drain(self, tmp_path) -> None:
        """The non-blocking freshness probe behind the stale-read advisory: False
        while a write's embed is still draining, True once the queue drains."""
        import threading

        class _GatedBackend(_Backend):
            def __init__(self) -> None:
                super().__init__()
                self.release = threading.Event()

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                self.release.wait(timeout=30.0)
                return super().embed_texts(texts)

        async def flow() -> None:
            backend = _GatedBackend()
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
            await kernel.settle()
            assert kernel.is_settled() is True, "an idle kernel is settled"

            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            # Commits immediately; the embed maintenance parks in the gated backend.
            await kernel.upsert_notes([(basic, 1, ["paris france", "geo"], [])], "allow")
            assert kernel.is_settled() is False, "a draining write is not settled"

            backend.release.set()
            await kernel.settle()
            assert kernel.is_settled() is True, "the drained queue is settled again"
            await kernel.close()

        asyncio.run(flow())

    def test_action_search_notes_brackets_freshness_at_read_time(self, tmp_path) -> None:
        """`stale` is computed INSIDE the search read (the col_mod + settled
        bracket), so it describes the snapshot actually read — not a pre-sample.
        A settled search is fresh; a search while a write drains is stale."""
        import threading

        class _GatedBackend(_Backend):
            def __init__(self) -> None:
                super().__init__()
                self.release = threading.Event()
                self.release.set()  # open by default; a test closes it to park

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                self.release.wait(timeout=30.0)
                return super().embed_texts(texts)

        async def flow() -> None:
            backend = _GatedBackend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes([(basic, 1, ["paris france", "geo"], [])], "allow")
            await kernel.settle()

            vectors = backend.embed_texts(["france"])

            def search(c):
                return shrike_native.action_search_notes(
                    c,
                    kernel.engine_handle(),
                    None,
                    [("france", "france", True)],
                    vectors,
                    10,
                    0.0,
                    kernel=kernel,
                    semantic=True,
                )

            # Settled: the read saw a stable, drained snapshot → not stale.
            settled_raw = await kernel.run_job(lambda: search(core))
            assert json.loads(settled_raw)["stale"] is False

            # Park the embed and enqueue a write: the bracket's settled probe reads
            # false while the read runs → stale, even though the result is served.
            backend.release.clear()
            await kernel.upsert_notes([(basic, 1, ["lyon france", "geo"], [])], "allow")
            assert kernel.is_settled() is False
            draining_raw = await kernel.run_job(lambda: search(core))
            assert json.loads(draining_raw)["stale"] is True

            backend.release.set()
            await kernel.settle()
            await kernel.close()

        asyncio.run(flow())

    def test_batch_is_one_embed_call(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            backend.calls.clear()
            results = await kernel.upsert_notes(
                [(basic, 1, [f"note number {i}", "back"], []) for i in range(10)],
                "error",
            )
            assert all(r[0] == "created" for r in results)
            await kernel.close()
            return backend

        backend = asyncio.run(flow())
        # One batched embed for the whole creation set (10 << the 64 chunk).
        assert len(backend.calls) == 1
        assert len(backend.calls[0]) == 10

    def test_restart_reconciles_unflushed_index(self, tmp_path) -> None:
        async def first():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes([(basic, 1, ["paris is in france", "geo"], [])], "error")
            # close() flushes; drift on restart is the *collection* moving on…
            await kernel.close()

        async def second():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            # Flushed at close → current on reopen.
            assert not await kernel.reindex_if_needed()
            hits = await kernel.search("paris france", 5)
            assert hits, "the persisted index serves after restart"
            await kernel.close()

        asyncio.run(first())
        asyncio.run(second())

    def test_release_reopen_cycle(self, tmp_path) -> None:
        async def flow():
            kernel = await _open(tmp_path, _Backend())
            await kernel.release()
            await kernel.reopen()
            assert isinstance(await kernel.col_mod(), int)
            await kernel.close()

        asyncio.run(flow())


class TestRunJob:
    def test_run_job_serializes_and_rethrows(self, tmp_path) -> None:
        async def flow():
            kernel = await _open(tmp_path, _Backend())
            core = kernel.core_handle()

            # The callable runs on the kernel executor over the shared core.
            basic = await kernel.run_job(lambda: core.notetype_id("Basic"))
            assert isinstance(basic, int)

            # A Python exception rethrows as-is through the awaitable.
            def boom() -> None:
                raise ValueError("job exploded")

            with pytest.raises(ValueError, match="job exploded"):
                await kernel.run_job(boom)
            await kernel.close()

        asyncio.run(flow())


class TestRebuildIndex:
    def test_explicit_rebuild_is_full(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes(
                [(basic, 1, [f"note {i}", "b"], []) for i in range(3)], "error"
            )
            await kernel.settle()
            backend.calls.clear()
            total = await kernel.rebuild_index()
            assert total == 3
            # FULL: every note re-embedded even though nothing drifted.
            assert sum(len(c) for c in backend.calls) == 3
            assert json.loads(kernel.index_status_json())["state"] == "ready"
            await kernel.close()

        asyncio.run(flow())

    def test_rebuild_without_embedder_is_unavailable(self, tmp_path) -> None:
        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            with pytest.raises(shrike_native.NativeUnavailableError):
                await kernel.rebuild_index()
            await kernel.close()

        asyncio.run(flow())


class TestNamedUpsert:
    def test_wire_shape_create_update_and_maintenance(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            engine = kernel.engine_handle()

            created = json.loads(
                await kernel.upsert_notes_json(
                    json.dumps(
                        [
                            {
                                "note_type": "Basic",
                                "deck": "Default",
                                "fields": {"Front": "the krebs cycle", "Back": "atp"},
                            }
                        ]
                    ),
                    "error",
                    False,
                )
            )
            assert created[0]["status"] == "created"
            nid = created[0]["id"]
            await kernel.settle()
            assert engine.contains(nid), "create maintained the index"
            vec_before = engine.get(nid)

            # The UPDATE half: same id, new text → re-embedded (replace).
            updated = json.loads(
                await kernel.upsert_notes_json(
                    json.dumps([{"id": nid, "fields": {"Front": "the calvin cycle"}}]),
                    "error",
                    False,
                )
            )
            assert updated[0]["status"] == "updated"
            await kernel.settle()
            assert engine.get(nid) != vec_before, "update re-embedded the note"
            assert not await kernel.reindex_if_needed(), "watermarks current"

            # dry_run writes nothing and maintains nothing.
            dry = json.loads(
                await kernel.upsert_notes_json(
                    json.dumps(
                        [
                            {
                                "note_type": "Basic",
                                "deck": "Default",
                                "fields": {"Front": "never written", "Back": "x"},
                            }
                        ]
                    ),
                    "error",
                    True,
                )
            )
            assert dry[0]["status"] == "ok"
            assert engine.size() == 1
            await kernel.close()

        asyncio.run(flow())


class TestCloseDrainsTheActor:
    """AsyncKernel.close routes through Kernel::close, which drains the
    collection actor — after it resolves, nothing is in flight and the actor
    is GONE (a subsequent collection op fails actor-gone rather than queueing
    into a leaked task).
    """

    def test_ops_after_close_hit_a_drained_actor(self, tmp_path) -> None:
        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [(basic, 1, ["drained actor", "guard"], [])], "error"
            )
            assert all(r[0] == "created" for r in results)

            await kernel.close()

            # The actor is drained: the op can't be queued into a leaked
            # task — it fails loud instead.
            with pytest.raises(shrike_native.NativeInternalError, match="actor is gone"):
                await kernel.col_mod()

            # And a second close is idempotent (the drain already happened).
            await kernel.close()

        asyncio.run(flow())
