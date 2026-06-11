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

            # Delete propagates to vectors too.
            assert await kernel.delete_notes([results[0][1]]) == 1
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
        from shrike.collection import CollectionWrapper

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
