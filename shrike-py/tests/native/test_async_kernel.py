"""The full kernel binding (#332, S3d-1b).

``AsyncKernel`` is the assembled kernel driven from asyncio: one open
collection + kernel-internal index orchestration + the derived store, every
op an awaitable. The harness supplies its parts — a worker executor, a
``PyEmbedder`` over its backend, the loop's timers — and shares the kernel's
engine/core handles for its own read/search surfaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "async_kernel_open"),
    reason="anki-core build required (scripts/build-native.sh)",
)


class _Backend:
    """Deterministic unit vectors + the EmbedderBackend metadata surface."""

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

    def model_fingerprint(self) -> str:
        return "test-backend:v1"

    def embedding_dim(self) -> int:
        return 4


async def _open(tmp_path, backend):
    kernel = await shrike_native.async_kernel_open(
        str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
    )
    kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
    return kernel


class TestRebuildDerived:
    """#445 checkpoint 3: the FTS5 rebuild runs kernel-side — rows never
    cross the FFI; the op returns (row_count, the build's col_mod snapshot)."""

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
            # namespaced per collection (#547), so resolve the path the kernel
            # wrote, not the flat cache root.
            db_path = cache_layout.derived_db_path(str(tmp_path / "cache"), collection_path)
            engine = shrike_native.DerivedTextEngine(db_path, 2)
            try:
                assert engine.get_col_mod() == dmod
                hits = engine.search_substring("krebs", 10)
                assert hits, "the rebuilt FTS5 store must match the seeded text"
            finally:
                engine.close()

        asyncio.run(flow())


class TestSaverTuning:
    """#355 item 2: the --index-save-* tuning reaches the kernel's saver."""

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
            # The index lands in the per-collection namespace (#67), not the
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

            # Delete propagates to vectors too. The maintained op (#604) returns
            # {deleted, not_found} JSON in its single write job.
            deleted = json.loads(await kernel.delete_notes([results[0][1]]))
            assert deleted == {"deleted": [results[0][1]], "not_found": []}
            assert sorted(engine.keys()) == sorted(created[1:])
            assert not await kernel.reindex_if_needed()

            status = json.loads(kernel.index_status_json())
            assert status["state"] == "ready"
            assert status["model_id"] == "test-backend:v1"
            await kernel.close()
            return backend

        backend = asyncio.run(flow())
        assert backend.calls, "embeds went through the harness backend"

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


class TestCrossSpaceWiring:
    """#234: the PRODUCTION search path (`action_search_notes` + the host
    `cross_space=` param) threads injected secondary-space results into the
    fusion + runs the relative gate — proving the host plumbing at the binding
    level, WITHOUT depending on the secondary-space WRITE fan-out (the
    immediately-following #232 write PR). The cross_space JSON here is exactly
    the shape `build_cross_space_json` produces."""

    class _Planted:
        """Plants EXACT vectors per text keyword so cosines are controlled: the
        text-target is at cos 0.6 from the query (below the injected vision 0.95
        → the gate opens), and the image note is ORTHOGONAL (cos 0 → never
        surfaces via the primary text signal)."""

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            out = []
            for t in texts:
                if "qqquery" in t:
                    out.append([1.0, 0.0, 0.0])  # the query axis
                elif "tttarget" in t:
                    out.append([0.6, 0.8, 0.0])  # cos 0.6 with the query
                else:  # the image note's text — orthogonal to the query
                    out.append([0.0, 0.0, 1.0])
            return out

        def model_fingerprint(self) -> str:
            return "planted:v1"

        def embedding_dim(self) -> int:
            return 3

    def test_action_search_notes_threads_cross_space_and_runs_the_gate(self, tmp_path) -> None:
        async def flow():
            backend = self._Planted()
            kernel = await _open(tmp_path, backend)
            assert await kernel.reindex_if_needed()  # materialize empty
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            # A text-target note (planted cos 0.6 from the query, below the
            # injected vision 0.95 → the gate opens) + an "image-bearing" note
            # whose text is ORTHOGONAL to the query (so the primary text signal
            # never surfaces it — only the injected vision space does, the
            # cross-space payoff).
            results = await kernel.upsert_notes(
                [
                    (basic, 1, ["tttarget photosynthesis adjacent card", "biology"], []),
                    (basic, 1, ["zzz an orthogonal picture-only card zzz", "img"], []),
                ],
                "error",
            )
            image_note = results[1][1]

            # The host embeds the PRIMARY query (#331) — one vector per source.
            query = "qqquery photosynthesis overview"
            vectors = backend.embed_texts([query])

            # Build the cross_space JSON exactly as `build_cross_space_json`
            # would: a "clip" vision space surfacing the image note at a STRONG
            # image cosine (dist 0.05 → cos 0.95) whose best clears the primary
            # text best for this off-topic-for-text query, so the relative gate
            # OPENS and the image note joins via `image#clip`.
            cross_space = json.dumps(
                [
                    {
                        "space_key": "clip",
                        "per_source": [
                            {
                                "modality_hits": {"image": [[image_note], [0.05]]},
                                "best_query_cosine": 0.95,
                            }
                        ],
                    }
                ]
            )

            def call(c):
                return shrike_native.action_search_notes(
                    c,
                    kernel.engine_handle(),
                    None,
                    [(query, query, True)],
                    vectors,
                    10,
                    0.3,  # threshold (excludes the orthogonal image note from the text signal)
                    kernel=kernel,
                    semantic=True,
                    index_size=kernel.engine_handle().size(),
                    cross_space=cross_space,
                )

            raw = await kernel.run_job(lambda: call(core))
            groups = json.loads(raw)
            matches = groups[0]["matches"]
            # SearchMatch flattens the note, so the id is at the top level.
            ids = [m["id"] for m in matches]
            # The injected vision space surfaced the image note through the host
            # param + the gate (it would be ABSENT in a text-only search — the
            # primary never matched its text).
            assert image_note in ids, "cross_space threaded through and the gate fired"
            img_match = next(m for m in matches if m["id"] == image_note)
            signals = [c["signal"] for c in img_match["provenance"]]
            assert "image#clip" in signals, "per-space provenance carried end-to-end"

            # N=1 byte-identical control: the SAME search with cross_space absent
            # (the single-space case) never surfaces the image note — the host
            # param defaults to None and the path is exactly today's.
            def call_n1(c):
                return shrike_native.action_search_notes(
                    c,
                    kernel.engine_handle(),
                    None,
                    [(query, query, True)],
                    vectors,
                    10,
                    0.3,
                    kernel=kernel,
                    semantic=True,
                    index_size=kernel.engine_handle().size(),
                )

            raw_n1 = await kernel.run_job(lambda: call_n1(core))
            ids_n1 = [m["id"] for m in json.loads(raw_n1)[0]["matches"]]
            assert image_note not in ids_n1, "without cross_space the image note is absent (N=1)"

            await kernel.close()

        asyncio.run(flow())


class _ImageBackend(_Backend):
    """A dual-encoder stand-in: advertises the image modality and embeds
    image bytes into the same 4-dim space."""

    modalities = frozenset({"text", "image"})

    def __init__(self) -> None:
        super().__init__()
        self.image_calls: list[int] = []

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        self.image_calls.append(len(images))
        out = []
        for data in images:
            b = hashlib.blake2b(data, digest_size=1).digest()[0] / 255.0
            n = (b * b + 1.0) ** 0.5
            out.append([1.0 / n, b / n, 0.0, 0.0])
        return out


class TestAsyncKernelImages:
    def test_image_seam_embeds_resolvable_images(self, tmp_path) -> None:
        media = tmp_path / "media"
        media.mkdir()
        (media / "diagram.png").write_bytes(b"png-bytes-here")

        def read(name: str) -> bytes | None:
            p = media / name
            return p.read_bytes() if p.exists() else None

        def exists(name: str) -> bool:
            return (media / name).exists()

        async def flow():
            backend = _ImageBackend()
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend), read, exists)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [
                    (basic, 1, ['has a picture <img src="diagram.png">', "back"], []),
                    (basic, 1, ['missing <img src="nope.png">', "back"], []),
                ],
                "error",
            )
            assert all(r[0] == "created" for r in results)
            engine = kernel.engine_handle()
            pictured, plain = results[0][1], results[1][1]
            # The resolvable image embedded under the note's key; the missing
            # one quietly contributed nothing (graceful degradation).
            assert engine.modality_keys("image") == [pictured]
            assert engine.modality_contains("text", plain)
            # No drift after the kernel's own multimodal writes.
            assert not await kernel.reindex_if_needed()
            await kernel.close()
            return backend

        backend = asyncio.run(flow())
        assert backend.image_calls == [1], "one image embed for the one resolvable file"


class TestLiveTwoSpaceEndToEnd:
    """#232: the full live multi-space loop a dedicated text + separate CLIP
    config enables. Upsert a note with an image → its image lands in the CLIP
    space's ImageOnly index → the PRODUCTION search (build_cross_space_json +
    action_search_notes + the PR-C gate) surfaces the image-bearing note. This
    is the e2e PR-C couldn't write (no write path then)."""

    class _TextPrimary:
        """A dedicated text embedder (3-dim planted): the query and a distractor
        text note share an axis; the image note's text is orthogonal."""

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            out = []
            for t in texts:
                if "qqquery" in t:
                    out.append([1.0, 0.0, 0.0])
                elif "tttarget" in t:
                    out.append([0.6, 0.8, 0.0])  # cos 0.6 with the query
                else:
                    out.append([0.0, 0.0, 1.0])  # orthogonal (the image note)
            return out

        def model_fingerprint(self) -> str:
            return "text-primary:v1"

        def embedding_dim(self) -> int:
            return 3

    class _Clip:
        """A separate CLIP: its TEXT tower embeds the query, its IMAGE tower
        embeds the picture, planted so the query→image cosine is high (gate
        opens) for the diagram and the note's own text is irrelevant to it."""

        modalities = frozenset({"text", "image"})

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            # The CLIP text tower: the query lands on the same axis as the
            # diagram image (cos high → the gate opens).
            return [[1.0, 0.0] for _ in texts]

        def embed_images(self, images: list[bytes]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in images]  # aligned with the query

        def model_fingerprint(self) -> str:
            return "clip:v1"

        def embedding_dim(self) -> int:
            return 2

    def test_image_note_surfaces_via_clip_space_through_production_search(self, tmp_path) -> None:
        media = tmp_path / "media"
        media.mkdir()
        (media / "diagram.png").write_bytes(b"diagram-bytes")

        def read(name: str) -> bytes | None:
            p = media / name
            return p.read_bytes() if p.exists() else None

        def exists(name: str) -> bool:
            return (media / name).exists()

        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            # Text-primary FIRST, then the separate CLIP image space (its own key
            # + the media resolver). Two spaces, dedicated-text + separate-CLIP.
            text_backend = self._TextPrimary()
            clip_backend = self._Clip()
            kernel.attach_embedder(shrike_native.PyEmbedder.capture(text_backend))
            kernel.attach_embedder(
                shrike_native.PyEmbedder.capture(clip_backend),
                read,
                exists,
                space_key="clip:v1",
            )
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            # The image-bearing note (orthogonal text) + a distractor text note.
            results = await kernel.upsert_notes(
                [
                    (basic, 1, ['zzz picture only <img src="diagram.png">', "b"], []),
                    (basic, 1, ["tttarget some text", "b"], []),
                ],
                "error",
            )
            image_note = results[0][1]

            # The image landed in the CLIP space's ImageOnly index.
            assert kernel.embed_space_count() == 2
            # Run the PRODUCTION search path: build cross_space (embeds the query
            # into the CLIP space + searches it) + action_search_notes.
            query = "qqquery find the diagram"
            vectors = text_backend.embed_texts([query])
            cross_space = await kernel.build_cross_space_json([query], 10)
            # The CLIP space contributed (non-empty), and the gate will open
            # (query→image cos high).
            assert json.loads(cross_space), "the CLIP space produced cross-space hits"

            def call(c):
                return shrike_native.action_search_notes(
                    c,
                    kernel.engine_handle(),
                    None,
                    [(query, query, True)],
                    vectors,
                    10,
                    0.3,
                    kernel=kernel,
                    semantic=True,
                    index_size=kernel.engine_handle().size(),
                    cross_space=cross_space,
                )

            raw = await kernel.run_job(lambda: call(core))
            ids = [m["id"] for m in json.loads(raw)[0]["matches"]]
            assert image_note in ids, (
                "the image-bearing note surfaced via the CLIP space + the gate — "
                "the full live multi-space loop"
            )
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


class TestEmbedderSlot:
    def test_detach_degrades_and_reattach_recovers(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            kernel.detach_embedder()
            assert json.loads(kernel.index_status_json())["state"] == "unavailable"
            # Creates still work and lexical search still serves.
            results = await kernel.upsert_notes(
                [(basic, 1, ["paris is the capital of france", "geo"], [])], "error"
            )
            assert results[0][0] == "created"
            hits = await kernel.search("capital of france", 5)
            assert hits[0][0] == results[0][1]
            assert all(s != "text" for s, _ in hits[0][2])

            # Re-attach (a fresh capture, like an embedding restart): the index
            # watermark stayed put, so reindex embeds the note created while
            # detached.
            kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
            assert json.loads(kernel.index_status_json())["state"] == "ready"
            assert await kernel.reindex_if_needed()
            hits = await kernel.search("capital of france", 5)
            assert any(s == "text" for s, _ in hits[0][2])
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


class TestWrapperOverKernel:
    def test_wrapper_ops_serialize_through_the_kernel(self, tmp_path) -> None:
        from shrike.harness.collection import CollectionWrapper

        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            wrapper = CollectionWrapper.over_kernel(kernel, str(tmp_path / "collection.anki2"))
            # The wrapper's async surface rides run_job over the shared core.
            assert await wrapper.col_mod() >= 0
            notes = await wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "via wrapper", "Back": "b"},
                    }
                ]
            )
            assert notes[0]["status"] == "created"
            # The kernel sees the same collection (one shared core).
            assert await kernel.col_mod() >= 0

            # Loop-free phases don't exist in kernel mode.
            with pytest.raises(RuntimeError, match="kernel mode"):
                wrapper.run_sync(lambda c: c.col_mod())
            with pytest.raises(RuntimeError, match="kernel mode"):
                wrapper.release_now()

            wrapper.close()  # must NOT close the kernel's core
            assert await kernel.col_mod() >= 0
            await kernel.close()

        asyncio.run(flow())


class TestCooperativeReopen:
    def test_kernel_writes_self_heal_after_release(self, tmp_path) -> None:
        # The review-found regression: an idle release closed the collection
        # and kernel write ops errored CollectionNotOpen (the reopen-on-demand
        # lived only in the Python wrapper). Kernel-side ensure_open fixes it.
        async def flow():
            kernel = await _open(tmp_path, _Backend())
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            await kernel.release()
            results = await kernel.upsert_notes(
                [(basic, 1, ["written while released", "b"], [])], "error"
            )
            assert results[0][0] == "created"

            await kernel.release()
            wire = json.loads(
                await kernel.upsert_notes_json(
                    json.dumps(
                        [
                            {
                                "note_type": "Basic",
                                "deck": "Default",
                                "fields": {"Front": "wire after release", "Back": "b"},
                            }
                        ]
                    ),
                    "error",
                    False,
                )
            )
            assert wire[0]["status"] == "created"
            await kernel.close()

        asyncio.run(flow())


class _StubRecognizer:
    """The RecognizerBackend wire contract: blocking recognize() returning
    (text, confidence, segments_json) tuples; one result per item."""

    def __init__(self, fingerprint: str = "stub:v1") -> None:
        self._fingerprint = fingerprint
        self.calls: list[int] = []

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        self.calls.append(len(items))
        out = []
        for data in items:
            text = data.decode("utf-8", errors="replace")
            segments = json.dumps(
                [{"text": text, "confidence": 0.92, "bbox": [0.0, 0.0, 1.0, 0.2]}]
            )
            out.append((text, 0.92, segments))
        return out

    def model_fingerprint(self) -> str:
        return self._fingerprint


class TestRecognition:
    """The #228 pipeline through the binding: the asyncio dispatch (loop →
    thread pool → oneshot), the bounded sweep, and both consumers."""

    def test_sweep_feeds_lexical_and_vector_consumers(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")

            await kernel.upsert_notes(
                [(basic, 1, ['See <img src="cycle.png">', "back"], [])], "error"
            )

            media = {"cycle.png": b"the citric acid cycle spins in the matrix"}
            recognizer = _StubRecognizer()
            kernel.attach_recognizer(
                shrike_native.Recognizer.capture(recognizer),
                media.get,
                lambda name: name in media,
            )

            report = json.loads(await kernel.recognize_pending(10))
            assert report["status"] == "ran"
            assert report["stored"] == 1
            assert recognizer.calls == [1]

            # Lexical: the phrase lives only inside the image.
            hits = await kernel.search("citric acid", 5)
            assert hits, "OCR text is lexically searchable"
            # Idempotent: nothing pending on the second sweep.
            again = json.loads(await kernel.recognize_pending(10))
            assert again["status"] == "idle"
            assert recognizer.calls == [1], "no re-recognition"

            await kernel.close()

        asyncio.run(flow())

    def test_raising_resolver_degrades_gracefully(self, tmp_path) -> None:
        # #386: a buggy/misconfigured Python resolver raises — the binding
        # logs and degrades (read → None, exists → False) instead of
        # crashing the sweep or recognizing empty bytes.
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes([(basic, 1, ['Img <img src="boom.png">', "b"], [])], "error")

            def raising(name: str):
                raise RuntimeError(f"resolver exploded on {name}")

            recognizer = _StubRecognizer()

            # exists raises → treated absent → nothing pending at all.
            kernel.attach_recognizer(shrike_native.Recognizer.capture(recognizer), raising, raising)
            report = json.loads(await kernel.recognize_pending(10))
            assert report["status"] == "idle"
            assert recognizer.calls == []

            # exists succeeds but read raises → the item is skipped (never
            # recognized over empty bytes), stored nothing, stays pending.
            kernel.attach_recognizer(
                shrike_native.Recognizer.capture(recognizer), raising, lambda n: True
            )
            report = json.loads(await kernel.recognize_pending(10))
            assert report["status"] == "ran"
            assert report["recognized"] == 0
            assert report["stored"] == 0
            assert recognizer.calls == []

            # A healed resolver picks the item up on the next sweep.
            media = {"boom.png": b"now readable after all"}
            kernel.attach_recognizer(
                shrike_native.Recognizer.capture(recognizer),
                media.get,
                lambda n: n in media,
            )
            report = json.loads(await kernel.recognize_pending(10))
            assert report["status"] == "ran"
            assert report["stored"] == 1
            assert recognizer.calls == [1]

            await kernel.close()

        asyncio.run(flow())

    def test_fingerprint_change_invalidates(self, tmp_path) -> None:
        async def flow():
            backend = _Backend()
            kernel = await _open(tmp_path, backend)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            await kernel.upsert_notes([(basic, 1, ['Img <img src="a.png">', "b"], [])], "error")
            media = {"a.png": b"recognized by engine version one"}

            first = _StubRecognizer("engine:v1")
            kernel.attach_recognizer(
                shrike_native.Recognizer.capture(first), media.get, lambda n: n in media
            )
            assert json.loads(await kernel.recognize_pending(10))["stored"] == 1

            # Same engine re-attached: nothing to do.
            kernel.attach_recognizer(
                shrike_native.Recognizer.capture(first), media.get, lambda n: n in media
            )
            assert json.loads(await kernel.recognize_pending(10))["status"] == "idle"

            # A NEW engine fingerprint invalidates and re-recognizes.
            second = _StubRecognizer("engine:v2")
            kernel.attach_recognizer(
                shrike_native.Recognizer.capture(second), media.get, lambda n: n in media
            )
            rerun = json.loads(await kernel.recognize_pending(10))
            assert rerun["status"] == "ran"
            assert rerun["stored"] == 1
            assert second.calls == [1]

            await kernel.close()

        asyncio.run(flow())


class TestCloseDrainsTheActor:
    """#374 design 7 (the review-HIGH regression guard): AsyncKernel.close
    routes through Kernel::close, which drains the collection actor — after
    it resolves, nothing is in flight and the actor is GONE (a subsequent
    collection op fails actor-gone rather than queueing into a leaked task).
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
