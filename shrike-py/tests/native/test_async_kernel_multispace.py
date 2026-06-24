"""Cross-space wiring and live multi-space search over the AsyncKernel."""

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

from .conftest import _Backend, _open  # noqa: E402


class TestCrossSpaceWiring:
    """The binding-level fused-search path (`action_search_notes` + an injected
    `cross_space=` param) threads secondary-space results into the fusion + runs
    the relative gate — proving the cross-space plumbing in ISOLATION from the
    secondary-space WRITE fan-out (which `kernel.search_fused` couples to, since
    it builds cross-space in-core). The cross_space JSON here is exactly the
    shape `build_cross_space_json` produces."""

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

            # The host embeds the PRIMARY query — one vector per source.
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
                    cross_space=cross_space,
                )

            raw = await kernel.run_job(lambda: call(core))
            # action_search_notes wraps the fused groups with the read-time `stale`
            # verdict: {"groups": [...], "stale": bool}.
            groups = json.loads(raw)["groups"]
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
                )

            raw_n1 = await kernel.run_job(lambda: call_n1(core))
            ids_n1 = [m["id"] for m in json.loads(raw_n1)["groups"][0]["matches"]]
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
            await kernel.settle()
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
    """The full live multi-space loop a dedicated text + separate CLIP
    config enables. Upsert a note with an image → its image lands in the CLIP
    space's ImageOnly index → the PRODUCTION search (build_cross_space_json +
    action_search_notes + the relative gate) surfaces the image-bearing note."""

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
            await kernel.settle()

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
                    cross_space=cross_space,
                )

            raw = await kernel.run_job(lambda: call(core))
            ids = [m["id"] for m in json.loads(raw)["groups"][0]["matches"]]
            assert image_note in ids, (
                "the image-bearing note surfaced via the CLIP space + the gate — "
                "the full live multi-space loop"
            )
            await kernel.close()

        asyncio.run(flow())
