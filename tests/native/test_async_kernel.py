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
    embedder = shrike_native.PyEmbedder.capture(backend)
    return await shrike_native.async_kernel_open(
        str(tmp_path / "collection.anki2"), str(tmp_path / "cache"), embedder
    )


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
            embedder = shrike_native.PyEmbedder.capture(backend)
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"),
                str(tmp_path / "cache"),
                embedder,
                None,
                read,
                exists,
            )
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
