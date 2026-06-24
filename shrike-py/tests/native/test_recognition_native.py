"""Native recognition: the harness sweep and the kernel-binding pipeline.

The harness-level recognition surface (OCR/ASR/describe attach, the bounded
sweep, dedup over OCR vectors) and the lower kernel-binding recognition pipeline
(asyncio dispatch, the bounded sweep, both consumers) live together here. The
kernel-binding classes additionally require an anki-core build.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from typing import Any

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness, KernelConfigError  # noqa: E402

from .conftest import _assemble, _Backend, _open, _StubAsr  # noqa: E402

_requires_async_kernel = pytest.mark.skipif(
    not hasattr(shrike_native, "async_kernel_open"),
    reason="anki-core build required (scripts/build-native.sh)",
)


def _text_neighbors(harness: Harness, backend: Any, query: str, top_k: int = 5) -> list[dict]:
    """Nearest text-modality neighbours of ``query`` over the kernel's engine.

    Embeds the query with ``backend`` (unit-normalized — the engine's
    inner-product metric assumes unit vectors), then searches the kernel's
    Arc-shared engine handle in the ``text`` modality. Returns one
    ``{note_id, distance}`` dict per hit. This is the dedup/neighbour path's
    underlying text-to-text ranking, exercised directly against the engine.
    """
    raw = backend.embed_texts([query])[0]
    norm = math.sqrt(sum(x * x for x in raw))
    vector = [x / norm for x in raw] if norm > 0 else raw
    rankings = harness.kernel.engine_handle().search_by_modality([vector], top_k, ["text"])
    ids, distances = rankings[0].get("text", ([], []))
    return [
        {"note_id": int(nid), "distance": float(dist)}
        for nid, dist in zip(ids, distances, strict=True)
    ]


class _StubOcr:
    """RecognizerBackend wire contract over a canned mapping."""

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        return [(data.decode(), 0.9, "") for data in items]

    def model_fingerprint(self) -> str:
        return "stub-ocr:v1"


class TestHarnessRecognition:
    def test_sweep_without_embedding_feeds_lexical_search(self, tmp_path) -> None:
        # Recognition is independent of the embed slot: with embedding off,
        # the sweep still lands OCR rows in the derived store (vectors mint
        # later, when an embedder attaches and reindexes).
        async def flow():
            media = {"krebs.png": b"oxaloacetate condenses with acetyl coa"}
            runtime = EmbeddingRuntime(model=None)
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                cooperative=False,
                hold_seconds=5.0,
                media_read=media.get,
                media_exists=lambda name: name in media,
            )
            await harness.boot(start_embedding=False)
            # Deterministic: the boot-drift derived rebuild commits
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
            rows = harness.derived.search_fuzzy("oxaloacetate", top_k=5)
            assert rows, "OCR text reached the lexical store"

            harness.detach_recognizer()
            await harness.close()

        asyncio.run(flow())

    def test_asr_sweep_over_sound_media_through_the_binding(self, tmp_path) -> None:
        # The AUDIO path through the harness/binding. A note with a
        # [sound:] ref is enumerated (note_sound_refs), a captured ASR stub for
        # the `asr` purpose transcribes it (LexicalAndVector), and the
        # transcript is both lexically searchable AND vector-minting through the
        # binding search — proving audio end-to-end without the platform engine.
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
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
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
            assert harness.derived.search_fuzzy("powerhouse of the cell", top_k=5), (
                "asr transcript reached the lexical store"
            )
            # ... and mints a vector reachable through the text-modality search.
            hits = _text_neighbors(harness, backend, "cell powerhouse mitochondria")
            assert any(h["note_id"] == audio_id for h in hits), (
                "asr transcript reachable via the vector"
            )

            harness.detach_recognizer("asr")
            await harness.close()

        asyncio.run(flow())

    def test_sweep_stops_on_unreadable_prefix_instead_of_spinning(self, tmp_path) -> None:
        # The livelock case: with more pending than one batch and a permanently
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
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
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
            # clock: the driver must return after exactly one no-progress
            # batch. `max_batches` is a generous livelock ceiling so a
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
            rows = harness.derived.search_fuzzy("oxaloacetate", top_k=5)
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
        # The native recognizer rides the kernel sweep with no Python
        # on the recognition path — harness attach takes the native pyclass
        # straight through (AnyRecognizer::Native → Blocking → the
        # runtime's blocking pool → Vision), and the recognized text lands as derived rows the
        # lexical consumer reads back.
        PIL = pytest.importorskip("PIL")  # noqa: F841 — fixture rendering only
        import io

        from PIL import Image, ImageDraw

        from shrike.harness.engines.recognition import make_recognizer

        img = Image.new("RGB", (640, 120), "white")
        ImageDraw.Draw(img).text((20, 40), "oxaloacetate condenses", fill="black", font_size=28)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        async def flow():
            media = {"krebs.png": buf.getvalue()}
            runtime = EmbeddingRuntime(model=None)
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
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

            rows = harness.derived.search_fuzzy("oxaloacetate", top_k=5)
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
            # An unknown OCR kind lands an 'error' row under the ocr source:
            # the engine is attached-but-errored, not absent.
            assert status["recognition"]["ocr"]["state"] == "error"
            await harness.close()

        asyncio.run(flow())

    def test_start_recognition_describe_missing_key_env_is_an_error_row(self, tmp_path) -> None:
        # A missing api_key_env raises a RuntimeError the boot path
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
        # An endpoint that answers NEITHER /health nor
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
        # The text-space semantic ranking (the max-over-items dedup the neighbor
        # path relies on) matches a draft against ALL of a note's text-modality
        # vectors: a card whose content lives ONLY inside an image surfaces as a
        # near-dupe through its OCR vector, while the card's own field text shares
        # nothing with the draft. Text-to-text — no modality gap, no activation
        # gate. (Neighbors route through the fused search action, which ranks over
        # the same text space; this pins the underlying max-over-items property.)
        import hashlib

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

            # The dedup path's exact ranking, with the same backend embedding the
            # draft query against the kernel's engine, like the server wires it.
            draft = "oxaloacetate condenses with acetyl coa today"
            hits = _text_neighbors(harness, backend, draft)
            scores = {h["note_id"]: 1.0 - h["distance"] for h in hits}
            assert diagram_id in scores, "the image-only content surfaced as a near-dupe"
            assert scores[diagram_id] >= 0.6, (
                f"clears the dedup threshold via the OCR vector: {scores[diagram_id]:.3f}"
            )

            await harness.close()

        asyncio.run(flow())


class _StubDescribe:
    """A captured describe recognizer (the PyRecognizer wire contract):
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
        # A describe recognizer attached for the ``vlm`` purpose mints
        # a TEXT-space vector for its generated prose (semantically searchable)
        # but its prose is NEVER reachable via the LEXICAL surfaces (exact /
        # fuzzy) — the VectorOnly destination, proven through the SAME pyo3
        # search path the server serves (`harness.kernel.search`, the fused
        # exact/fuzzy/text ranking).
        # OCR alongside it stays lexically searchable (byte-identical routing).
        import hashlib

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
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
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
            assert harness.derived.search_fuzzy("chlorophyll absorption", top_k=5)
            assert harness.derived.search_fuzzy("sunlit mountain valley", top_k=5), (
                "the describe row is stored (provenance + reconcile) even though search hides it"
            )

            # The dedup ranking (the other binding search seam) also surfaces the
            # describe vector max-over-items — a query overlapping the prose
            # finds the note even though its field text shares nothing.
            hits = _text_neighbors(harness, backend, "sunlit mountain valley with grazing cattle")
            assert any(h["note_id"] == photo_id for h in hits), (
                "describe vector reachable through the dedup search path"
            )

            harness.detach_recognizer("vlm")
            harness.detach_recognizer("ocr")
            await harness.close()

        asyncio.run(flow())

    def test_describe_vector_only_through_the_search_notes_action(self, tmp_path) -> None:
        # The VectorOnly invariant proven at the actual
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

        from shrike.api.tools import register_tools
        from shrike.harness.harness import KernelIndexView

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
            # surfaces (the search-facing KernelIndexView over the kernel, like
            # the server wires it), then drive search_notes — the exact path a
            # client's tools/call reaches.
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


@_requires_async_kernel
class TestKernelRecognition:
    """The recognition pipeline through the binding: the asyncio dispatch (loop
    → thread pool → oneshot), the bounded sweep, and both consumers."""

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
        # A buggy/misconfigured Python resolver raises — the binding
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
