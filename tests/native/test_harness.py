"""The kernel-mode server core (#332 S3d-2d): Harness over a real AsyncKernel.

Embedding-free assembly: the kernel opens on the loop with the harness
thread driving its executor, the wrapper rides run_job, the derived store
builds on drift, and the operational verbs return the wire shapes the
routes serve — all without a model, mirroring a no-embedding boot.
"""

from __future__ import annotations

import asyncio
import sys

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
            # Recognition is off until a backend is configured (#228).
            assert status["recognition"] == {"state": "unavailable", "backend": None}

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


class TestEmbedQueryCache:
    def test_repeat_queries_reuse_the_vector(self, tmp_path) -> None:
        from types import SimpleNamespace

        from shrike.harness import KernelIndexView

        class _Counting:
            def __init__(self) -> None:
                self.calls = 0

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                self.calls += 1
                return [[1.0, 0.0] for _ in texts]

        async def flow():
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "c.anki2"), str(tmp_path / "cache")
            )
            backend = _Counting()
            runtime = SimpleNamespace(backend=backend)
            view = KernelIndexView(kernel, runtime)  # type: ignore[arg-type]

            first = view.embed_queries(["krebs cycle"])
            again = view.embed_queries(["krebs cycle"])
            assert first == again
            assert backend.calls == 1, "the repeat came from the cache"

            # A new backend identity (model swap) never reuses entries.
            runtime.backend = _Counting()
            view.embed_queries(["krebs cycle"])
            assert runtime.backend.calls == 1
            await kernel.close()

        asyncio.run(flow())


class _StubOcr:
    """RecognizerBackend wire contract over a canned mapping."""

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        return [(data.decode(), 0.9, "") for data in items]

    def model_fingerprint(self) -> str:
        return "stub-ocr:v1"


class TestRecognition:
    def test_sweep_without_embedding_feeds_lexical_search(self, tmp_path) -> None:
        # Recognition is independent of the embed slot: with embedding off,
        # the sweep still lands OCR rows in the derived store (vectors mint
        # later, when an embedder attaches and reindexes).
        async def flow():
            media = {"krebs.png": b"oxaloacetate condenses with acetyl coa"}
            runtime = EmbeddingRuntime(model=None)
            derived = DerivedTextStore(
                path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
            )
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                derived=derived,
                cooperative=False,
                hold_seconds=5.0,
                media_read=media.get,
                media_exists=lambda name: name in media,
            )
            await harness.boot(start_embedding=False)
            await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": 'See <img src="krebs.png">', "Back": "b"},
                    }
                ]
            )

            harness.attach_recognizer(_StubOcr())
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 1

            # The lexical consumer sees it through the SAME store file.
            rows = harness.derived.search_substring("oxaloacetate", limit=5)
            assert rows, "OCR text reached the lexical store"

            harness.detach_recognizer()
            await harness.close()

        asyncio.run(flow())

    def test_sweep_stops_on_unreadable_prefix_instead_of_spinning(self, tmp_path) -> None:
        # #386 livelock: with more pending than one batch and a permanently
        # unreadable PREFIX of the pending order, the kernel re-takes the
        # same window every call (skipped items stay pending). The sweep
        # driver must stop on the no-progress batch (recognized == 0) and
        # return — the next sweep trigger (boot, /reload, cooperative
        # re-acquire) retries when the read may have healed.
        async def flow():
            unreadable = {"u1.png", "u2.png"}
            media = {
                "u1.png": b"unreadable prefix one",
                "u2.png": b"unreadable prefix two",
                "ok.png": b"readable tail oxaloacetate",
            }
            runtime = EmbeddingRuntime(model=None)
            derived = DerivedTextStore(
                path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
            )
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                derived=derived,
                cooperative=False,
                hold_seconds=5.0,
                media_read=lambda name: None if name in unreadable else media.get(name),
                media_exists=lambda name: name in media,
            )
            await harness.boot(start_embedding=False)
            await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {
                            "Front": '<img src="u1.png"> <img src="u2.png"> <img src="ok.png">',
                            "Back": "b",
                        },
                    }
                ]
            )

            harness.attach_recognizer(_StubOcr())
            # Bounded wait: a livelocked loop awaits between batches, so a
            # regression fails by timeout here rather than hanging the suite.
            report = await asyncio.wait_for(harness.recognition_sweep(batch_size=2), timeout=30)
            assert report["status"] == "ran"
            assert report["recognized"] == 0
            assert report["remaining"] == 1
            assert report["total_stored"] == 0

            # Healed reads drain to completion on the next sweep.
            unreadable.clear()
            report = await harness.recognition_sweep(batch_size=2)
            assert report["total_stored"] == 3
            assert report["remaining"] == 0
            rows = harness.derived.search_substring("oxaloacetate", limit=5)
            assert rows, "the tail item landed once the prefix healed"

            harness.detach_recognizer()
            await harness.close()

        asyncio.run(flow())

    @pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision is macOS-only")
    def test_native_vision_sweep_end_to_end(self, tmp_path) -> None:
        # #342 P3: the native recognizer rides the kernel sweep with no Python
        # on the recognition path — harness attach takes the native pyclass
        # straight through (AnyRecognizer::Native → Blocking → the
        # runtime's blocking pool → Vision), and the recognized text lands as derived rows the
        # lexical consumer reads back.
        PIL = pytest.importorskip("PIL")  # noqa: F841 — fixture rendering only
        import io

        from PIL import Image, ImageDraw

        from shrike.recognition import make_recognizer

        img = Image.new("RGB", (640, 120), "white")
        ImageDraw.Draw(img).text((20, 40), "oxaloacetate condenses", fill="black", font_size=28)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        async def flow():
            media = {"krebs.png": buf.getvalue()}
            runtime = EmbeddingRuntime(model=None)
            derived = DerivedTextStore(
                path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
            )
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                derived=derived,
                cooperative=False,
                hold_seconds=5.0,
                media_read=media.get,
                media_exists=lambda name: name in media,
            )
            await harness.boot(start_embedding=False)
            await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": 'See <img src="krebs.png">', "Back": "b"},
                    }
                ]
            )

            backend = make_recognizer("apple")
            assert backend.model_fingerprint().startswith("apple-vision-swift:")
            harness.attach_recognizer(backend)
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 1

            rows = harness.derived.search_substring("oxaloacetate", limit=5)
            assert rows, "native OCR text reached the lexical store"

            harness.detach_recognizer()
            await harness.close()

        asyncio.run(flow())

    def test_attach_without_media_access_is_a_config_error(self, tmp_path) -> None:
        async def flow():
            harness = await _assemble(tmp_path)
            await harness.boot(start_embedding=False)
            with pytest.raises(KernelConfigError):
                harness.attach_recognizer(_StubOcr())
            await harness.close()

        asyncio.run(flow())

    def test_start_recognition_unknown_backend_degrades_to_error(self, tmp_path) -> None:
        # The runtime surface: an unknown/unavailable backend marks the
        # recognition state 'error' without disturbing the rest of boot.
        async def flow():
            media = {"x.png": b"text"}
            runtime = EmbeddingRuntime(model=None)
            derived = DerivedTextStore(
                path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
            )
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                derived=derived,
                cooperative=False,
                hold_seconds=5.0,
                media_read=media.get,
                media_exists=lambda name: name in media,
            )
            await harness.boot(start_embedding=False)

            harness.start_recognition("nope")
            status = await harness.status()
            assert status["recognition"]["state"] == "error"
            await harness.close()

        asyncio.run(flow())


class TestDedupOverOcr:
    def test_dedup_search_covers_ocr_vectors_max_over_items(self, tmp_path) -> None:
        # #205: the dedup neighbor path (KernelIndexView.search — the exact
        # call _attach_neighbors makes) matches a draft against ALL of a
        # note's text-modality vectors, max-over-items: a card whose content
        # lives ONLY inside an image surfaces as a near-dupe through its OCR
        # vector, while the card's own field text shares nothing with the
        # draft. Text-to-text — no modality gap, no activation gate.
        import hashlib
        from types import SimpleNamespace

        from shrike.harness import KernelIndexView

        class _TokenHash:
            """Token-overlap cosine: shared words → similarity."""

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                out = []
                for t in texts:
                    v = [0.0] * 32
                    for tok in t.lower().split():
                        h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16)
                        v[h % 32] += 1.0
                    n = sum(x * x for x in v) ** 0.5 or 1.0
                    out.append([x / n for x in v])
                return out

            def model_fingerprint(self) -> str:
                return "tok:v1"

            def embedding_dim(self) -> int:
                return 32

        async def flow():
            media = {"cycle.png": b"oxaloacetate condenses with acetyl coa"}
            runtime = EmbeddingRuntime(model=None)
            derived = DerivedTextStore(
                path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
            )
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                derived=derived,
                cooperative=False,
                hold_seconds=5.0,
                media_read=media.get,
                media_exists=lambda name: name in media,
            )
            await harness.boot(start_embedding=False)
            backend = _TokenHash()
            harness.kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
            await harness.kernel.reindex_if_needed()

            notes = await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": 'See diagram <img src="cycle.png">', "Back": "b"},
                    },
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "qq unrelated filler card qq", "Back": "b"},
                    },
                ]
            )
            diagram_id = notes[0]["id"]

            harness.attach_recognizer(_StubOcr())
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 1

            # The dedup path's exact call, with the same backend embedding the
            # draft query (a fresh view over the kernel's engine, like the
            # server wires it).
            view = KernelIndexView(harness.kernel, SimpleNamespace(backend=backend))  # type: ignore[arg-type]
            draft = "oxaloacetate condenses with acetyl coa today"
            hits = view.search([draft], top_k=5)[0]
            scores = {h["note_id"]: 1.0 - h["distance"] for h in hits}
            assert diagram_id in scores, "the image-only content surfaced as a near-dupe"
            assert scores[diagram_id] >= 0.6, (
                f"clears the dedup threshold via the OCR vector: {scores[diagram_id]:.3f}"
            )

            await harness.close()

        asyncio.run(flow())
