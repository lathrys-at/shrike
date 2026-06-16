"""The kernel-mode server core (#332 S3d-2d): Harness over a real AsyncKernel.

Embedding-free assembly: the kernel opens on the loop with the harness
thread driving its executor, the wrapper rides run_job, the derived store
builds on drift, and the operational verbs return the wire shapes the
routes serve — all without a model, mirroring a no-embedding boot.
"""

from __future__ import annotations

import asyncio
import json
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


class TestDerivedNamespaceParity:
    """#562: the host DerivedTextStore (the /status read surface) and the
    kernel's DerivedEngine must open the SAME per-collection derived namespace.

    The namespace canonicalizes the collection path, and that canonicalization
    differs by whether the file EXISTS at computation time (existing → realpath,
    which folds a symlinked prefix; absent → a lexical abspath that does NOT).
    The old caller order built the host store BEFORE the kernel created a fresh
    collection's file, so the host hashed under the abspath namespace while the
    kernel used the realpath one — host /status read an EMPTY store while the
    kernel's search store held the rows. Harness.assemble now builds the host
    store AFTER open, so the file exists for both and they realpath identically.
    """

    def test_host_store_and_kernel_resolve_same_path_for_fresh_collection(self, tmp_path) -> None:
        # The repro condition: a FRESH collection (file absent at assemble time)
        # reached via a SYMLINKED prefix (so abspath != realpath, like macOS
        # /var/folders -> /private/var/...). With the pre-open build, the host
        # store landed in the abspath namespace; post-open it realpaths to the
        # kernel's namespace.
        from shrike import cache_layout

        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        # Pass the SYMLINKED spelling — abspath keeps the link, realpath folds it.
        cache_dir = str(link / "cache")
        collection = str(link / "collection.anki2")  # does NOT exist yet

        async def flow():
            runtime = EmbeddingRuntime(model=None)
            # No `derived` injected: assemble resolves it post-open (the fix).
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
        # substring query finds the row. Under the old caller order this host
        # store sat in a stale abspath namespace and read empty (#562). The
        # symlinked prefix + fresh collection is the exact bug condition.
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
                for _ in range(100):
                    if harness.derived.status().get("state") == "ready":
                        break
                    await asyncio.sleep(0.05)
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
            # Recognition is off until a backend is configured (#228/#485): the
            # keyed-by-source map is empty (distinct from attached-but-errored).
            assert status["recognition"] == {}
            # The cross-modal coverage matrix (#498/#235): shape-stable, every
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
            # The boot saw drift and built; wait for the background build.
            for _ in range(100):
                if harness.derived.status().get("state") == "ready":
                    break
                await asyncio.sleep(0.05)
            assert harness.derived.status()["state"] == "ready"
            await harness.close()

        asyncio.run(flow())


class _FakeRouterManager:
    """A stand-in for shrike_native.LlamaServerManager.router(...) — records
    start/stop calls and reports running across them, so the harness's
    spawn-once / owner-only-stop lifecycle (#567) is provable without a real
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
    """#567: the shared llama.cpp router manager is spawned ONCE and stopped
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
            # A routed (#68) harness does not own the runtime, so its close must
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


class _StubAsr:
    """A captured ASR recognizer (#485): transcribes the audio bytes and
    carries a single time-`Span` segment (the audio locator, vs OCR's bbox).
    The RecognizerBackend wire contract — captured behind PyRecognizer, like a
    custom OCR backend — proving the audio path end-to-end without the
    platform AppleSpeechTranscriber (mobile-only, never the server build)."""

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        # segments JSON carries a "span" locator (time range), serialized like
        # the Rust Segment with Locator::Span — the kernel stores it opaquely.
        out = []
        for data in items:
            text = data.decode()
            segments = json.dumps([{"text": text, "confidence": 0.9, "span": [0.0, 2.5]}])
            out.append((text, 0.9, segments))
        return out

    def model_fingerprint(self) -> str:
        return "stub-asr:v1"


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
            # Deterministic (#471): the boot-drift derived rebuild commits
            # off the actor; settle it before racing ingests against it.
            await harness.settle_background()
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

    def test_asr_sweep_over_sound_media_through_the_binding(self, tmp_path) -> None:
        # #485 PR2: the AUDIO path through the harness/binding. A note with a
        # [sound:] ref is enumerated (note_sound_refs), a captured ASR stub for
        # the `asr` purpose transcribes it (LexicalAndVector), and the
        # transcript is both lexically searchable AND vector-minting through the
        # binding search — proving audio end-to-end without the platform engine.
        import hashlib
        from types import SimpleNamespace

        from shrike.harness import KernelIndexView

        class _TokenHash:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                out = []
                for t in texts:
                    v = [0.0] * 64
                    for tok in t.lower().split():
                        h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16)
                        v[h % 64] += 1.0
                    n = sum(x * x for x in v) ** 0.5 or 1.0
                    out.append([x / n for x in v])
                return out

            def model_fingerprint(self) -> str:
                return "tok:v1"

            def embedding_dim(self) -> int:
                return 64

        async def flow():
            media = {"lecture.mp3": b"mitochondria are the powerhouse of the cell"}
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
                        "fields": {"Front": "Listen [sound:lecture.mp3]", "Back": "b"},
                    }
                ]
            )
            audio_id = notes[0]["id"]

            # Captured ASR stub for the audio purpose (source "asr").
            harness.attach_recognizer(_StubAsr(), "asr")
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 1, report

            # LexicalAndVector: the transcript reaches the lexical store ...
            assert harness.derived.search_substring("powerhouse of the cell", limit=5), (
                "asr transcript reached the lexical store"
            )
            # ... and mints a vector reachable through the binding search.
            view = KernelIndexView(harness.kernel, SimpleNamespace(backend=backend))  # type: ignore[arg-type]
            hits = view.search(["cell powerhouse mitochondria"], top_k=5)[0]
            assert any(h["note_id"] == audio_id for h in hits), (
                "asr transcript reachable via the vector"
            )

            harness.detach_recognizer("asr")
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
            # The no-progress STOP is bounded by kernel-call COUNT, not wall
            # clock (#525): the driver must return after exactly one no-progress
            # batch. `max_batches` is a generous livelock ceiling so a #386
            # regression (re-taking the same window forever) returns instead of
            # hanging the suite — but a correct driver stops at batch 1, well
            # under it. Asserting `batches == 1` (not a 30s timeout) makes the
            # test deterministic under load: pass/fail keys on the driver's
            # logic, never on how fast a contended machine runs one batch.
            report = await harness.recognition_sweep(batch_size=2, max_batches=50)
            assert report["batches"] == 1, (
                "the sweep must STOP on the first no-progress batch, not re-take "
                "the unreadable window (a #386 livelock regression would loop to "
                f"the max_batches ceiling): batches={report['batches']}"
            )
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
    @pytest.mark.skipif(
        not hasattr(shrike_native, "AppleVisionRecognizer"),
        reason="engine-apple not compiled into this build (mobile-only since #496; "
        "test re-homing is #514)",
    )
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
            # An unknown OCR kind lands an 'error' row under the ocr source
            # (#485): the engine is attached-but-errored, not absent.
            assert status["recognition"]["ocr"]["state"] == "error"
            await harness.close()

        asyncio.run(flow())

    def test_start_recognition_describe_missing_key_env_is_an_error_row(self, tmp_path) -> None:
        # #485 PR1: a missing api_key_env raises a RuntimeError the boot path
        # catches → an 'error' row keyed by the vlm source, with NO network
        # probe and NO attach. The per-engine /status map populates without
        # disturbing boot, and OCR is untouched (the vlm error adds no ocr row).
        async def flow():
            media = {"x.png": b"text"}
            harness = await self._describe_harness(tmp_path, media)

            harness.start_recognition_describe(
                "http://127.0.0.1:9", api_key_env="SHRIKE_TEST_DESCRIBE_KEY_UNSET_XYZ"
            )
            status = await harness.status()
            assert status["recognition"]["vlm"]["state"] == "error"
            assert status["recognition"]["vlm"]["backend"] == "describe-remote"
            assert "ocr" not in status["recognition"]
            await harness.close()

        asyncio.run(flow())

    def test_start_recognition_describe_unreachable_endpoint_reports_error(self, tmp_path) -> None:
        # #485 PR1 (follow-up B): an endpoint that answers NEITHER /health nor
        # /v1/models (a closed port) is reported as an 'error' engine row — a
        # degraded engine must be visible, not silently 'ready'. The engine
        # still ATTACHES (the row carries its degenerate fingerprint), so the
        # backlog stays pending per the chunk-Err-aborts contract until a sweep
        # reaches the endpoint.
        async def flow():
            media = {"x.png": b"text"}
            harness = await self._describe_harness(tmp_path, media)

            # Port 9 (discard) is closed → health_ok() False and model_info()
            # empty → reachable False → an 'error' row that nonetheless carries
            # a (degenerate) fingerprint, proving the engine attached.
            harness.start_recognition_describe("http://127.0.0.1:9")
            status = await harness.status()
            assert status["recognition"]["vlm"]["state"] == "error"
            assert status["recognition"]["vlm"]["backend"] == "describe-remote"
            assert status["recognition"]["vlm"]["fingerprint"], (
                "the engine attached with a (degenerate) fingerprint"
            )
            assert status["recognition"]["vlm"]["fingerprint"].endswith(":prompt=1")
            await harness.close()

        asyncio.run(flow())

    @staticmethod
    async def _describe_harness(tmp_path, media):
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
        return harness


class TestDedupOverOcr:
    def test_dedup_search_covers_ocr_vectors_max_over_items(self, tmp_path) -> None:
        # #205: the text-space semantic ranking (KernelIndexView.search, the
        # max-over-items dedup the neighbor path relies on) matches a draft
        # against ALL of a note's text-modality vectors: a card whose content
        # lives ONLY inside an image surfaces as a near-dupe through its OCR
        # vector, while the card's own field text shares nothing with the
        # draft. Text-to-text — no modality gap, no activation gate. (Neighbors
        # now route through the fused search action, #531, which ranks over the
        # same text space; this pins the underlying max-over-items property.)
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


class _StubDescribe:
    """A captured describe recognizer (#485, the PyRecognizer wire contract):
    canned generated prose under the kernel's ``vlm`` purpose. Distinct
    fingerprint from OCR so the two engines key their meta independently."""

    def __init__(self, prose: str) -> None:
        self._prose = prose

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        # One description per image; confidence 1.0 (substance gates it), no
        # segments (a description locates nothing).
        return [(self._prose, 1.0, "") for _ in items]

    def model_fingerprint(self) -> str:
        return "describe:stub:v1"


class TestDescribeAttach:
    def test_describe_routes_vector_only_through_the_binding_search(self, tmp_path) -> None:
        # #485 PR1: a describe recognizer attached for the ``vlm`` purpose mints
        # a TEXT-space vector for its generated prose (semantically searchable)
        # but its prose is NEVER reachable via the LEXICAL surfaces (exact /
        # fuzzy) — the VectorOnly destination, proven through the SAME pyo3
        # search path the server serves (`harness.kernel.search`, the fused
        # exact/fuzzy/text ranking), not just at the kernel layer #538 covered.
        # OCR alongside it stays lexically searchable (byte-identical routing).
        import hashlib
        from types import SimpleNamespace

        from shrike.harness import KernelIndexView

        class _TokenHash:
            """Token-overlap cosine: shared words → similarity (no model)."""

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                out = []
                for t in texts:
                    v = [0.0] * 64
                    for tok in t.lower().split():
                        h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16)
                        v[h % 64] += 1.0
                    n = sum(x * x for x in v) ** 0.5 or 1.0
                    out.append([x / n for x in v])
                return out

            def model_fingerprint(self) -> str:
                return "tok:v1"

            def embedding_dim(self) -> int:
                return 64

        # The describe prose: a literal phrase ("sunlit mountain valley") that
        # lives ONLY here — never in the note's field text or the OCR text — so
        # a lexical hit on it could ONLY come from the describe row. The token
        # bag also overlaps a non-literal semantic query.
        describe_prose = "a photograph of a sunlit mountain valley with grazing cattle at dawn"

        async def flow():
            # The OCR text is distinct visible text; the field text shares
            # nothing with either, so each signal's provenance is unambiguous.
            media = {"photo.png": b"figure 7 chlorophyll absorption spectrum"}
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
                        "fields": {"Front": 'zzz qqq <img src="photo.png">', "Back": "b"},
                    }
                ]
            )
            photo_id = notes[0]["id"]

            # Both engines over the one image: OCR (lexical + vector) and
            # describe (the captured stub, vector-only) — attached for their
            # own purposes through the binding's purpose-aware attach.
            harness.attach_recognizer(_StubOcr2("figure 7 chlorophyll absorption spectrum"), "ocr")
            harness.attach_recognizer(_StubDescribe(describe_prose), "vlm")
            # NB: attach_recognizer (the low-level seam these stubs use) does
            # not touch the /status engine map — that's owned by
            # start_recognition[_describe], exercised in the profiles/boot
            # tests. Here the kernel-level attach is what we assert routing on.

            report = await harness.recognition_sweep(batch_size=4)
            # Two recognized (OCR + describe over the one image), both stored.
            assert report["total_stored"] == 2, report

            # (1) The describe prose mints a TEXT vector: a non-literal token-bag
            # query (no shared literal substring with anything stored) surfaces
            # the note through the `text` signal in the FUSED binding search.
            sem = await harness.kernel.search("mountain valley cattle grazing dawn", 5)
            sem_hit = next((h for h in sem if h[0] == photo_id), None)
            assert sem_hit is not None, "describe vector ranks in the fused search"
            signals = {s for s, _ in sem_hit[2]}
            assert "text" in signals, f"the describe vector mints a text signal: {signals}"

            # (2) The describe prose is HIDDEN from the lexical surfaces: a
            # literal phrase that lives ONLY in the describe prose must not hit
            # on `exact` or `fuzzy` through the binding search (VectorOnly).
            lex = await harness.kernel.search("sunlit mountain valley", 5)
            lex_hit = next((h for h in lex if h[0] == photo_id), None)
            if lex_hit is not None:
                lex_signals = {s for s, _ in lex_hit[2]}
                assert "exact" not in lex_signals and "fuzzy" not in lex_signals, (
                    f"describe prose leaked into the lexical surfaces: {lex_signals}"
                )

            # (3) Contrast — OCR is byte-identical routing: its visible text IS
            # lexically reachable (the `exact` signal) through the same search.
            ocr = await harness.kernel.search("chlorophyll absorption spectrum", 5)
            ocr_hit = next((h for h in ocr if h[0] == photo_id), None)
            assert ocr_hit is not None, "OCR text searchable"
            assert "exact" in {s for s, _ in ocr_hit[2]}, "OCR stays lexical (unchanged)"

            # The describe row IS stored in the derived store (for provenance +
            # reconcile — VectorOnly hides it from SEARCH, it isn't dropped):
            # the low-level store facade (unfiltered by design — it is not a
            # search entry point) sees both rows. The exclusion lives at the
            # search path (asserted above via kernel.search), not at storage.
            assert harness.derived.search_substring("chlorophyll absorption", limit=5)
            assert harness.derived.search_substring("sunlit mountain valley", limit=5), (
                "the describe row is stored (provenance + reconcile) even though search hides it"
            )

            # The dedup view (the other binding search seam) also surfaces the
            # describe vector max-over-items — a query overlapping the prose
            # finds the note even though its field text shares nothing.
            view = KernelIndexView(harness.kernel, SimpleNamespace(backend=backend))  # type: ignore[arg-type]
            hits = view.search(["sunlit mountain valley with grazing cattle"], top_k=5)[0]
            assert any(h["note_id"] == photo_id for h in hits), (
                "describe vector reachable through the dedup search path"
            )

            harness.detach_recognizer("vlm")
            harness.detach_recognizer("ocr")
            await harness.close()

        asyncio.run(flow())

    def test_describe_vector_only_through_the_search_notes_action(self, tmp_path) -> None:
        # #485 PR1 (follow-up A): the VectorOnly invariant proven at the actual
        # `search_notes` MCP ACTION path real clients hit — not just the kernel
        # search. A literal phrase living ONLY in the describe prose returns the
        # note (if at all) WITHOUT a `substring` annotation or an `exact`/`fuzzy`
        # provenance signal; the OCR visible text DOES carry the `substring`
        # annotation + `exact` signal. This closes the surface-level gap: the
        # action threads `hidden_lexical_sources` into the native search, so a
        # describe-source row can never surface a lexical hit to a client.
        import hashlib
        from types import SimpleNamespace

        from mcp.server.fastmcp import FastMCP

        from shrike.harness import KernelIndexView
        from shrike.tools import register_tools

        class _TokenHash:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                out = []
                for t in texts:
                    v = [0.0] * 64
                    for tok in t.lower().split():
                        h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16)
                        v[h % 64] += 1.0
                    n = sum(x * x for x in v) ** 0.5 or 1.0
                    out.append([x / n for x in v])
                return out

            def model_fingerprint(self) -> str:
                return "tok:v1"

            def embedding_dim(self) -> int:
                return 64

        describe_prose = "a photograph of a sunlit mountain valley with grazing cattle at dawn"

        async def flow():
            media = {"photo.png": b"figure 7 chlorophyll absorption spectrum"}
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
                        "fields": {"Front": 'zzz qqq <img src="photo.png">', "Back": "b"},
                    }
                ]
            )
            photo_id = notes[0]["id"]

            harness.attach_recognizer(_StubOcr2("figure 7 chlorophyll absorption spectrum"), "ocr")
            harness.attach_recognizer(_StubDescribe(describe_prose), "vlm")
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 2, report

            # Register the REAL MCP action registry against the harness's
            # surfaces (the search-facing KernelIndexView embeds queries with the
            # same backend, like the server wires it), then drive search_notes —
            # the exact path a client's tools/call reaches.
            view = KernelIndexView(harness.kernel, SimpleNamespace(backend=backend))  # type: ignore[arg-type]
            mcp = FastMCP("test")
            register_tools(
                mcp, harness.wrapper, index=view, kernel=harness.kernel, derived=harness.derived
            )

            async def search(q: str) -> list[dict]:
                _, structured = await mcp.call_tool("search_notes", {"queries": [q]})
                groups = structured["results"]
                # One query → at most one group; flatten its matches.
                return groups[0]["matches"] if groups else []

            # (1) A query that is a VERBATIM literal substring of the describe
            # prose AND a strong token overlap — so it surfaces the note via the
            # describe `text` vector (a non-vacuous guard: the note IS present),
            # yet must carry NO `substring` annotation and NO `exact`/`fuzzy`
            # provenance signal through the action. If hidden_lexical_sources
            # weren't threaded into the native search, this exact-substring query
            # would mint an `exact` signal off the describe row — it must not.
            desc_matches = await search("sunlit mountain valley with grazing cattle at dawn")
            desc_hit = next((m for m in desc_matches if m["id"] == photo_id), None)
            assert desc_hit is not None, "the describe vector surfaces the note via text"
            desc_signals = {p["signal"] for p in desc_hit.get("provenance", [])}
            assert "text" in desc_signals, f"reachable only via the vector: {desc_signals}"
            assert desc_hit.get("substring") is None, (
                f"describe prose leaked a substring annotation: {desc_hit.get('substring')}"
            )
            assert not (desc_signals & {"exact", "fuzzy"}), (
                f"describe prose leaked a lexical signal through search_notes: {desc_signals}"
            )

            # (2) The OCR visible text: the note IS returned with a substring
            # annotation + an exact provenance signal (byte-identical routing).
            ocr_matches = await search("chlorophyll absorption spectrum")
            ocr_hit = next((m for m in ocr_matches if m["id"] == photo_id), None)
            assert ocr_hit is not None, "OCR text searchable through search_notes"
            assert ocr_hit.get("substring") is not None, "OCR carries the substring annotation"
            assert "exact" in {p["signal"] for p in ocr_hit.get("provenance", [])}, (
                "OCR carries the exact provenance signal"
            )

            harness.detach_recognizer("vlm")
            harness.detach_recognizer("ocr")
            await harness.close()

        asyncio.run(flow())


class _StubOcr2:
    """A captured OCR recognizer over canned visible text (the wire contract).
    Separate from the bytes-echoing _StubOcr so the OCR text is independent of
    the (opaque) image bytes in the describe test."""

    def __init__(self, text: str) -> None:
        self._text = text

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        return [(self._text, 0.9, "") for _ in items]

    def model_fingerprint(self) -> str:
        return "stub-ocr:v2"


# ── Multi-space embedding set (#233) ─────────────────────────────────────────


class _FakeEmbedBackend:
    """A captured embedder backend over the wire contract (`embed_texts` /
    `model_fingerprint` / `embedding_dim`) — distinct fingerprints make two
    distinct kernel embed spaces (#233)."""

    def __init__(self, fingerprint: str, dim: int = 16) -> None:
        self._fp = fingerprint
        self._dim = dim
        self.running = True

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self._dim - 1) for _ in texts]

    def model_fingerprint(self) -> str:
        return self._fp

    def embedding_dim(self) -> int:
        return self._dim

    @property
    def modalities(self):
        return frozenset({"text"})

    def stop(self) -> None:
        self.running = False


class _FakeRuntime:
    """A minimal stand-in for EmbeddingRuntime exercising the harness fan-out
    contract (`running` / `start` / `backend` / `backend_kind` / `stop`). A
    `fail=True` runtime raises on start — proving a failing space degrades only
    itself."""

    def __init__(self, fingerprint: str, *, fail: bool = False) -> None:
        self._fp = fingerprint
        self._fail = fail
        self.backend_kind = "onnx"
        self.backend = None

    @property
    def running(self) -> bool:
        return self.backend is not None and self.backend.running

    def start(self, **_overrides):
        if self._fail:
            raise RuntimeError(f"space {self._fp} cannot start")
        self.backend = _FakeEmbedBackend(self._fp)
        return self.backend

    def stop(self) -> None:
        if self.backend is not None:
            self.backend.stop()
            self.backend = None


class TestMultiSpaceFanOut:
    def test_two_spaces_attach_and_a_failing_one_leaves_the_other_live(self, tmp_path) -> None:
        # The harness fan-out (#233): the primary plus one secondary space both
        # attach → the kernel holds TWO embed spaces. A SECOND secondary that
        # fails to start degrades only its own space — the others stay live.
        async def flow():
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
                media_read=None,
                media_exists=None,
                secondary_runtimes=[
                    _FakeRuntime("space:b"),
                    _FakeRuntime("space:c", fail=True),  # degrades only itself
                ],
            )
            await harness.boot(start_embedding=False)
            assert harness.kernel.embed_space_count() == 0

            # Attach the primary (real onnx-free PyEmbedder) + fan out.
            primary = _FakeEmbedBackend("space:a")
            harness._attach(primary)  # primary attaches unkeyed (N=1 path)
            assert harness.kernel.embed_space_count() == 1
            await harness._attach_secondaries({})

            # space:a (primary) + space:b (secondary) are live; space:c failed.
            assert harness.kernel.embed_space_count() == 2
            assert harness.secondary_runtimes[0].running  # b started
            assert not harness.secondary_runtimes[1].running  # c failed

            # A second fan-out is idempotent for an already-running secondary.
            await harness._attach_secondaries({})
            assert harness.kernel.embed_space_count() == 2

            # stop_embedding clears every space and stops the secondaries.
            harness.kernel.detach_embedder()
            for rt in harness.secondary_runtimes:
                rt.stop()
            assert harness.kernel.embed_space_count() == 0

            await harness.close()

        asyncio.run(flow())


class _RecalibKernel:
    """A minimal kernel stand-in counting secondary-floor recalibrations and
    forcing a reconcile (reindex_if_needed -> True)."""

    def __init__(self) -> None:
        self.recalibrated = 0

    async def reindex_if_needed(self) -> bool:
        return True

    async def calibrate_secondary_floors(self, margin: float) -> list[tuple[str, float | None]]:
        self.recalibrated += 1
        return []


class _RecalibWrapper:
    cooperative = False

    async def reopen(self) -> None:
        return None

    async def col_mod(self) -> int:
        return 123


class _RecalibRuntime:
    backend = object()


def _make_recalib_harness() -> Harness:
    h = Harness.__new__(Harness)
    h.kernel = _RecalibKernel()  # type: ignore[assignment]
    h.wrapper = _RecalibWrapper()  # type: ignore[assignment]
    h.runtime = _RecalibRuntime()  # type: ignore[assignment]
    h.secondary_runtimes = []
    h.cross_space_floor_margin = 1.0
    h._bg_tasks = set()

    async def _noop_build() -> None:
        return None

    h._maybe_build_derived = _noop_build  # type: ignore[assignment]
    return h


class TestReloadRecalibratesSecondaryFloors:
    """#596: reload() must recalibrate the secondary cross-space image floor
    after a reconcile, like every other reindex path
    (_drive_reindex/_rebuild_then_calibrate/_drive_boot_reindex). Otherwise the
    #576/#580 floor stays computed against pre-reload vectors and mis-gates the
    image space until the next boot/rebuild/cooperative-reacquire. N>=2 only."""

    def test_drive_boot_reindex_recalibrates(self) -> None:  # control
        h = _make_recalib_harness()
        asyncio.run(h._drive_boot_reindex())
        assert h.kernel.recalibrated == 1

    def test_reload_recalibrates_after_reindex(self) -> None:
        h = _make_recalib_harness()
        out = asyncio.run(h.reload())
        assert out["rebuilding"] is True
        assert h.kernel.recalibrated == 1, (
            "reload() reconciled drift but never recalibrated the secondary "
            "cross-space image floor (stale floor mis-gates the image space)"
        )


class TestFacadeReadinessBootWindow:
    """Repro for #650 — the recognition-vector family flake's CONFIRMED root
    cause (NOT #628's stale "absent OCR vector / 0.154" framing).

    Team-debug @ b6c574e found the family flake (`test_asr_sweep_*`,
    `test_sweep_stops_*`) is a facade-readiness race, not a missing vector:
    `boot()` spawns the host ``DerivedTextStore`` rebuild fire-and-forget
    (``_maybe_build_derived`` -> ``_spawn_bg(_rebuild_derived())`` — never
    awaited), so the facade sits at ``state=BUILDING`` until that un-awaited
    task's ``await kernel.rebuild_derived()`` continuation runs
    ``settle_external_build`` -> ``READY``. While ``BUILDING``,
    ``DerivedTextStore.search_substring`` short-circuits to ``None`` (gated on
    ``available == _state==READY``) **even though the recognition rows it would
    return are already present in shrike.db**, written kernel-side by
    ``recognition_sweep``. Under ``bazel test //tests/native:native
    --local_test_jobs=10+`` (~10x process oversubscription) the un-awaited
    settle is descheduled past a test's ``search_substring`` read -> the read
    returns ``None`` -> the family asserts fail (~0.5%). The note's text vector
    is always present (``modality_get`` count == 2); no ``SQLITE_BUSY``.

    This pins the PRODUCT invariant the fix must restore: the boot-rebuild
    ``BUILDING`` window MUST NOT silently drop already-present derived rows. The
    same gap means a real ``search_notes`` substring/fuzzy query during the boot
    build-window transiently misses OCR/ASR rows (a silent field-fall-back).

    Deterministic: the ``BUILDING`` window is entered with the real
    ``claim_external_build()`` primitive ``_rebuild_derived`` itself uses,
    instead of racing for it under load.
    """

    def test_search_substring_serves_present_rows_during_building(self, tmp_path) -> None:
        import hashlib

        class _TokenHash:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                out = []
                for t in texts:
                    v = [0.0] * 64
                    for tok in t.lower().split():
                        h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16)
                        v[h % 64] += 1.0
                    n = sum(x * x for x in v) ** 0.5 or 1.0
                    out.append([x / n for x in v])
                return out

            def model_fingerprint(self) -> str:
                return "tok:v1"

            def embedding_dim(self) -> int:
                return 64

        async def flow():
            media = {"lecture.mp3": b"mitochondria are the powerhouse of the cell"}
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
            harness.kernel.attach_embedder(shrike_native.PyEmbedder.capture(_TokenHash()))
            await harness.kernel.reindex_if_needed()

            await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "Listen [sound:lecture.mp3]", "Back": "b"},
                    }
                ]
            )
            harness.attach_recognizer(_StubAsr(), "asr")
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 1, report

            # Settle the boot rebuild so we ISOLATE the BUILDING gate (not a boot
            # race) and establish the precondition: when READY, the kernel-written
            # ASR row is genuinely present and findable through the facade.
            await harness.settle_background()
            assert harness.derived.available, "precondition: facade READY after settle"
            assert harness.derived.search_substring("powerhouse of the cell", limit=5), (
                "precondition: the ASR row is present and findable when the facade is READY"
            )

            # Re-enter the exact state the un-awaited boot rebuild produces under
            # load: facade BUILDING, with the row already in shrike.db.
            assert harness.derived.claim_external_build(), "entered the BUILDING window"
            try:
                rows = harness.derived.search_substring("powerhouse of the cell", limit=5)
                # THE DEFECT (#650): present rows are silently dropped while BUILDING.
                assert rows, (
                    "facade must serve already-present recognition rows during the boot "
                    "BUILDING window, not silently field-fall-back to None (#650)"
                )
            finally:
                harness.derived.settle_external_build(harness.derived.col_mod or 0)
                await harness.close()

        asyncio.run(flow())
