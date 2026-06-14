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

# ── #650 angle-A instrumentation (TEMPORARY) ────────────────────────────────
# Capture the FULL engine + derived state at a family assert so a failure under
# the bazel xdist load repro classifies ABSENT vs UNRECALLED definitively.
import logging as _logging  # noqa: E402
import os  # noqa: E402

# Discriminator (#650, lead request #2): record whether the kernel's swallow
# WARNING ("reading recognized texts failed; embedding without them",
# lib.rs:1276) fires — it forwards through pyo3-log to the `shrike_kernel`
# python logger. Presence => H-D1 (un-retried SQLITE_BUSY on the vector-minting
# derived read). We capture EVERY warning+ record so we also see the Err variant.
_SWALLOW_PHRASE = "reading recognized texts failed"
_LOG_RECORDS: list[str] = []


class _RecordingHandler(_logging.Handler):
    def emit(self, record: _logging.LogRecord) -> None:
        try:
            _LOG_RECORDS.append(f"{record.name}:{record.levelname}:{record.getMessage()}")
        except Exception:
            pass


def _install_log_capture() -> None:
    """Attach a recording handler to the root + shrike_kernel loggers (once),
    forcing levels low enough that pyo3-log WARNINGs are delivered."""
    root = _logging.getLogger()
    if not any(isinstance(h, _RecordingHandler) for h in root.handlers):
        h = _RecordingHandler()
        h.setLevel(_logging.DEBUG)
        root.addHandler(h)
    root.setLevel(_logging.DEBUG)
    _logging.getLogger("shrike_kernel").setLevel(_logging.DEBUG)
    # pyo3-log caches the level filter on first log; the native side reads the
    # effective python level. DEBUG here lets the WARNING through regardless.


# Install at import so the recording handler is live before ANY sweep runs.
_install_log_capture()


def _capture_650(tag, harness, note_id, *, draft=None, hits=None):
    """Snapshot the recognition-vector state and return a string for the assert.

    Also appends a JSON line to ``$TEST_UNDECLARED_OUTPUTS_DIR/_650_capture.jsonl``
    (or ``/tmp`` when not under bazel) so the evidence survives the test process.
    """
    import json as _json
    import time as _time
    import traceback as _tb

    snap = {"tag": tag, "note_id": int(note_id), "ts": _time.time(), "pid": os.getpid()}
    # 1) Vector counts in the TEXT modality (the decisive datum).
    try:
        eng = harness.kernel.engine_handle()
        vecs = eng.modality_get("text", int(note_id))
        snap["text_vec_count"] = None if vecs is None else len(vecs)
        snap["text_keys"] = sorted(int(k) for k in eng.modality_keys("text"))
        snap["text_distinct_keys"] = sorted(set(snap["text_keys"]))
        snap["engine_keys"] = sorted(int(k) for k in eng.keys())
    except Exception as e:  # pragma: no cover - diagnostic only
        snap["engine_error"] = f"{type(e).__name__}: {e}\n{_tb.format_exc()}"
    # 2) Derived-store availability/state/size + the background-build state.
    try:
        snap["derived_available"] = bool(harness.derived.available)
        snap["derived_status"] = harness.derived.status()
        snap["bg_tasks_pending"] = len(getattr(harness, "_bg_tasks", ()))
    except Exception as e:  # pragma: no cover
        snap["derived_error"] = f"{type(e).__name__}: {e}"
    # 3) The search hits, if provided (UNRECALLED needs the present-but-unranked).
    if hits is not None:
        snap["draft"] = draft
        snap["hits"] = [
            {"note_id": int(h["note_id"]), "distance": float(h["distance"])} for h in hits
        ]
    # 4) THE DISCRIMINATOR (#650 lead #2): did the kernel swallow-WARNING fire?
    swallow = [r for r in _LOG_RECORDS if _SWALLOW_PHRASE in r]
    snap["swallow_warning_fired"] = bool(swallow)
    snap["swallow_records"] = swallow[:5]
    # Any kernel warning/error at all (the Err variant lives here, e.g. a
    # SQLITE_BUSY vs another db_err).
    snap["kernel_warns"] = [
        r for r in _LOG_RECORDS if ":WARNING:" in r or ":ERROR:" in r
    ][:10]
    line = _json.dumps(snap, sort_keys=True)
    out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or "/tmp"
    try:
        with open(os.path.join(out_dir, "_650_capture.jsonl"), "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    return line


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
            sub = harness.derived.search_substring("powerhouse of the cell", limit=5)
            cap_sub = _capture_650("asr-substring", harness, audio_id)
            assert sub, f"asr transcript reached the lexical store | sub={sub!r} | {cap_sub}"
            # ... and mints a vector reachable through the binding search.
            view = KernelIndexView(harness.kernel, SimpleNamespace(backend=backend))  # type: ignore[arg-type]
            hits = view.search(["cell powerhouse mitochondria"], top_k=5)[0]
            cap_vec = _capture_650(
                "asr-vector", harness, audio_id, draft="cell powerhouse mitochondria", hits=hits
            )
            assert any(h["note_id"] == audio_id for h in hits), (
                f"asr transcript reachable via the vector | {cap_vec}"
            )

            harness.detach_recognizer("asr")
            await harness.close()

        asyncio.run(flow())

    def test_asr_sweep_TOGGLE_settled(self, tmp_path) -> None:
        # #650 angle-A TOGGLE (TEMPORARY): byte-identical to
        # test_asr_sweep_over_sound_media_through_the_binding EXCEPT it awaits
        # `settle_background()` before reading the Python derived facade. If the
        # original flakes (facade still BUILDING) and THIS never does, the cause
        # is the un-awaited background derived-store build, not an absent vector.
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

            harness.attach_recognizer(_StubAsr(), "asr")
            report = await harness.recognition_sweep(batch_size=4)
            assert report["total_stored"] == 1, report

            # THE TOGGLE: drain the background derived-store build before reading
            # the Python facade. (The original test omits this.)
            await harness.settle_background()

            sub = harness.derived.search_substring("powerhouse of the cell", limit=5)
            cap_sub = _capture_650("asr-toggle-substring", harness, audio_id)
            assert sub, f"[TOGGLE] asr transcript reached the lexical store | {cap_sub}"

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
            cap = _capture_650("dedup", harness, diagram_id, draft=draft, hits=hits)
            assert diagram_id in scores, f"the image-only content surfaced as a near-dupe | {cap}"
            assert scores[diagram_id] >= 0.6, (
                f"clears the dedup threshold via the OCR vector: {scores[diagram_id]:.3f} | {cap}"
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
            cap_sem = _capture_650("describe-sem", harness, photo_id)
            assert sem_hit is not None, f"describe vector ranks in the fused search | {cap_sem}"
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
            cap_ocr = _capture_650("describe-ocr", harness, photo_id)
            assert ocr_hit is not None, f"OCR text searchable | {cap_ocr}"
            assert "exact" in {s for s, _ in ocr_hit[2]}, (
                f"OCR stays lexical (unchanged) | {cap_ocr}"
            )

            # The describe row IS stored in the derived store (for provenance +
            # reconcile — VectorOnly hides it from SEARCH, it isn't dropped):
            # the low-level store facade (unfiltered by design — it is not a
            # search entry point) sees both rows. The exclusion lives at the
            # search path (asserted above via kernel.search), not at storage.
            _ds1 = harness.derived.search_substring("chlorophyll absorption", limit=5)
            cap_ds = _capture_650("describe-store", harness, photo_id)
            assert _ds1, f"derived store chlorophyll | _ds1={_ds1!r} | {cap_ds}"
            assert harness.derived.search_substring("sunlit mountain valley", limit=5), (
                f"the describe row is stored (provenance + reconcile) | {cap_ds}"
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
