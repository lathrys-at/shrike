"""The CLIP backend against a real small CLIP export (image<->text shared space).

Mocked mechanics live in ``tests/unit/test_embedding_clip.py``; here we run ``ClipBackend``
against the actual ``Xenova/clip-vit-base-patch32`` (q4) ONNX graphs so the preprocessing, I/O, and
the shared-space property are exercised for real. The semantic assertion uses solid-colour
images (deterministic, no network beyond the cached model): a colour image lands nearer its own
colour word than unrelated concepts, proving a text query retrieves by image content.
"""

from __future__ import annotations

import asyncio
import colorsys
import io
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from shrike.harness.engines.embedding.clip import ClipBackend
from shrike.harness.index import ACTIVATION_MARGIN, CALIB_MIN, activation_floor
from tests.integration.conftest import requires_clip, requires_shrike_native

pytestmark = [pytest.mark.integration, pytest.mark.embedding]

_CLIP_DIM = 512
# Unrelated query texts: a solid-colour image must land nearer *its own colour word* than any of
# these. (Deterministic, no network. NOTE: the comparison is colour-vs-*unrelated-concept*, not
# colour-vs-other-colour — the latter gap is ~0.05 and flips across int8 onnxruntime builds, the
# former is ~0.09 and robust.)
_UNRELATED = ["a photograph of a cat", "a circuit diagram schematic", "a page of printed text"]


# ONE started backend for the whole module: every consumer exercises the same
# default quantized graphs read-only, and the target runs serially (xdist=None in
# BUILD.bazel).
@pytest.fixture(scope="module")
def be(clip_model: Path) -> Iterator[ClipBackend]:
    backend = ClipBackend(model=str(clip_model))  # auto-discovers the quantized graphs
    backend.start()
    yield backend
    backend.stop()


# 224x224 == the CLIP preprocess crop size: the native preprocess runs a
# CatmullRom resize UNCONDITIONALLY, and that resize — not decode or inference —
# dominates per-image cost (53ms from 256², 37ms from 96², 16.7ms at crop size).
# Solid-colour vectors are bit-identical across source sizes, so the contracts are
# unaffected; the one deliberately non-224 image below (300x200) stays as the
# resize+crop path canary.
def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (224, 224)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@requires_clip
@requires_shrike_native
class TestClipModel:
    def test_shared_space_retrieves_by_image(self, be: ClipBackend) -> None:
        from PIL import Image

        for color, name in [((220, 30, 30), "red"), ((30, 30, 220), "blue")]:
            iv = np.array(be.embed_images([Image.new("RGB", (224, 224), color)])[0])
            match = float(iv @ np.array(be.embed_texts([f"a solid {name} colour image"])[0]))
            others = [float(iv @ np.array(be.embed_texts([t])[0])) for t in _UNRELATED]
            # A text query lands nearer the matching image than unrelated ones — image-by-text.
            assert match > max(others) + 0.03, f"{name}: match={match:.3f} others={others}"

    def test_dims_normalized_and_distinct(self, be: ClipBackend) -> None:
        from PIL import Image

        assert be.embedding_dim() == _CLIP_DIM
        tvecs = be.embed_texts(["a cat", "a dog"])
        # Deliberately non-224: the resize+crop path canary (see _png_bytes note).
        ivec = np.array(be.embed_images([Image.new("RGB", (300, 200), (10, 200, 10))])[0])
        # Text + image both land in the same 512-dim space, L2-normalized.
        assert all(len(v) == _CLIP_DIM for v in tvecs) and len(ivec) == _CLIP_DIM
        assert np.isclose(np.linalg.norm(tvecs[0]), 1.0) and np.isclose(np.linalg.norm(ivec), 1.0)
        # The encoders actually run (distinct inputs → distinct vectors).
        assert not np.allclose(tvecs[0], tvecs[1])

    def test_quantized_clip_is_serial(self, be: ClipBackend) -> None:
        # The quantized graphs are batch-variant (dynamic quantization), so the probe forces serial.
        assert be._safe_batch == 1
        assert be.health()["batch"] == "serial"

    def test_health(self, be: ClipBackend) -> None:
        h = be.health()
        assert h["available"] is True
        assert h["backend"] == "clip"
        assert h["modalities"] == ["image", "text"]
        assert h["provider"] == "CPUExecutionProvider"


@requires_clip
@requires_shrike_native
class TestClipImageIndex:
    """End-to-end per-modality index over the kernel: a text query retrieves a note by its
    image, and the per-modality image ranker surfaces it at rank-1 across the gap. Uses the
    shared module backend; each test opens its own (cheap) kernel + collection against it."""

    @staticmethod
    async def _open_kernel(tmp_path: Path, be: ClipBackend):
        """A real AsyncKernel over a fresh collection, with the CLIP backend attached the way
        the server does it (native_embedder + the media-dir resolver pair)."""
        import shrike_native

        from shrike.server.server import _make_image_resolver

        collection = str(tmp_path / "c.anki2")
        media_dir = collection[: -len(".anki2")] + ".media"
        os.makedirs(media_dir, exist_ok=True)
        kernel = await shrike_native.async_kernel_open(collection, str(tmp_path / "cache"))
        read, exists = _make_image_resolver(media_dir)
        kernel.attach_embedder(be.native_embedder(), read, exists)
        await kernel.reindex_if_needed()  # materialize the (empty) index
        return kernel, media_dir

    @staticmethod
    async def _seed(kernel, notes: list[dict]) -> list[int]:
        import json

        results = json.loads(await kernel.upsert_notes_json(json.dumps(notes), "allow", False))
        assert all(r["status"] in ("created", "updated") for r in results), results
        # Writes are write-only: the embed + index add drain off the ingest
        # queue, so settle before a caller inspects the engine.
        await kernel.settle()
        return [r["id"] for r in results]

    @staticmethod
    def _note(front: str) -> dict:
        return {"deck": "Test", "note_type": "Basic", "fields": {"Front": front, "Back": "."}}

    @staticmethod
    def _best_image(engine, be: ClipBackend, query: str, k: int = 2):
        """(ids, distances) of the image-modality ranking for a text query."""
        ranking = engine.search_by_modality(be.embed_texts([query]), k)[0]
        return ranking["image"]

    def test_image_note_rank_one_via_image_ranker(self, be: ClipBackend, tmp_path: Path) -> None:
        from PIL import Image

        async def flow():
            kernel, media_dir = await self._open_kernel(tmp_path, be)
            Image.new("RGB", (224, 224), (220, 30, 30)).save(os.path.join(media_dir, "red.png"))
            # The note's TEXT never names a colour; its meaning lives in the image.
            red, other = await self._seed(
                kernel,
                [self._note('study card <img src="red.png">'), self._note("ancient rome")],
            )
            engine = kernel.engine_handle()
            # red: text + image = 2 vectors (image in its own sub-index); other: text = 1.
            assert engine.size() == 3
            assert engine.modality_keys("image") == [red]

            # Per-modality retrieval: the image ranking is a separate signal, so the
            # image-bearing note surfaces at rank-1 *in that ranking* regardless of CLIP's
            # modality gap (text-text cos ~0.72 vs text-image ~0.32) — which a single deduped
            # cosine ranking could not deliver (the red note's own TEXT names no colour).
            ids, matching = self._best_image(engine, be, "a solid red colour image")
            assert ids[0] == red

            # And it retrieves by image *content*: the red note's image vector is nearer the
            # matching colour query than an unrelated-concept query (the robust
            # colour-vs-unrelated regime — colour-vs-colour is ~0.05 and flips across int8
            # builds, so it's avoided).
            _, unrelated = self._best_image(engine, be, "a circuit diagram schematic")
            assert matching[0] < unrelated[0]
            await kernel.close()

        asyncio.run(flow())

    def test_update_drops_removed_image_vector(self, be: ClipBackend, tmp_path: Path) -> None:
        from PIL import Image

        async def flow():
            kernel, media_dir = await self._open_kernel(tmp_path, be)
            Image.new("RGB", (224, 224), (220, 30, 30)).save(os.path.join(media_dir, "red.png"))
            red, other = await self._seed(
                kernel,
                [self._note('study card <img src="red.png">'), self._note("ancient rome")],
            )
            engine = kernel.engine_handle()
            assert engine.size() == 3
            # The red note loses its image (its embedding fingerprint changes) → the kernel's
            # maintenance drops the image vector for exactly that note; the other is untouched.
            import json

            await kernel.upsert_notes_json(
                json.dumps([{"id": red, "fields": {"Front": "study card"}}]), "allow", False
            )
            await kernel.settle()
            assert engine.size() == 2  # red: text only now ; rome: text
            assert engine.modality_keys("image") == []
            await kernel.close()

        asyncio.run(flow())

    def test_activation_gate_calibrates_and_passes_genuine_match(
        self, be: ClipBackend, tmp_path: Path
    ) -> None:
        import json

        from PIL import Image

        async def flow():
            kernel, media_dir = await self._open_kernel(tmp_path, be)
            # A ≥CALIB_MIN colour collection so calibration produces image stats on the real
            # CLIP model: each card a distinct solid-colour image with colour-neutral text
            # (card 0 is red).
            n = CALIB_MIN + 2
            notes = []
            for i in range(n):
                r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 0.9)  # hue 0 (card 0) is red
                rgb = (int(r * 255), int(g * 255), int(b * 255))
                fn = f"c{i}.png"
                Image.new("RGB", (224, 224), rgb).save(os.path.join(media_dir, fn))
                notes.append(self._note(f'study card number {i} <img src="{fn}">'))
            await self._seed(kernel, notes)
            await kernel.rebuild_index()  # full rebuild calibrates the gate

            # Offline calibration ran on the real model and produced image stats.
            raw = json.loads(kernel.index_status_json())
            stats = raw["activation"]
            assert stats["image"]["n"] >= CALIB_MIN
            assert stats["image"]["std"] > 0.0
            # Per-modality breakdown: the two-space CLIP index reports a text AND
            # an image sub-index, each with its own size/ndim.
            mods = {m["modality"]: m for m in raw["modalities"]}
            assert {"text", "image"} <= set(mods)
            assert mods["text"]["size"] == n and mods["image"]["size"] == n
            assert mods["text"]["ndim"] and mods["image"]["ndim"]
            floor = activation_floor(stats["image"], ACTIVATION_MARGIN)
            assert floor is not None

            engine = kernel.engine_handle()

            def best_sim(query: str) -> float:
                _, dists = self._best_image(engine, be, query, k=1)
                return 1.0 - dists[0]

            # A genuine colour-content query clears the floor → its image card passes the gate
            # (the gate must not suppress real matches), and it's distinctly stronger than an
            # off-topic query — which falls toward/below the floor and is gated out (the
            # deterministic drop is unit-tested in test_tools_search.py; colour-vs-unrelated is
            # the robust ~0.09 regime).
            red_best = best_sim("a solid red colour image")
            noise_best = best_sim("a circuit diagram schematic")
            assert red_best > floor
            assert red_best > noise_best
            await kernel.close()

        asyncio.run(flow())

    def test_missing_media_file_skipped(self, be: ClipBackend, tmp_path: Path) -> None:
        async def flow():
            kernel, _media_dir = await self._open_kernel(tmp_path, be)
            # Reference a file that isn't in the media dir → skipped, text still indexed.
            (red,) = await self._seed(
                kernel, [self._note('study card <img src="does-not-exist.png">')]
            )
            engine = kernel.engine_handle()
            assert engine.size() == 1
            assert engine.contains(red)
            await kernel.close()

        asyncio.run(flow())


@requires_clip
@requires_shrike_native
class TestClipNativeSeam:
    """The native CLIP engine's binding seams. ClipBackend IS the native engine."""

    def test_image_input_forms_agree(self, be, tmp_path: Path) -> None:
        # bytes / path / PIL image all land on the same vector (the facade
        # converts losslessly to bytes for the FFI).
        from PIL import Image

        data = _png_bytes((10, 200, 10))
        path = tmp_path / "green.png"
        path.write_bytes(data)
        pil = Image.open(io.BytesIO(data))

        v_bytes = np.array(be.embed_images([data])[0], dtype=np.float32)
        v_path = np.array(be.embed_images([str(path)])[0], dtype=np.float32)
        v_pil = np.array(be.embed_images([pil])[0], dtype=np.float32)
        np.testing.assert_array_equal(v_bytes, v_path)
        np.testing.assert_allclose(v_bytes, v_pil, atol=1e-5)

    def test_kernel_native_attach_embeds_images(self, be, tmp_path: Path) -> None:
        """The CLIP composition attaches native — ONE adapted engine serves both
        kernel halves, and neither text nor image embeds re-enter the facade (the
        counter pin)."""
        import shrike_native

        media = {"red.png": _png_bytes((200, 30, 30))}
        calls = {"embed_texts": 0, "embed_images": 0}
        originals = {name: getattr(be, name) for name in calls}
        for name, orig in originals.items():

            def counting(items, _orig=orig, _name=name):  # type: ignore[no-untyped-def]
                calls[_name] += 1
                return _orig(items)

            setattr(be, name, counting)

        async def flow() -> bool:
            kernel = await shrike_native.async_kernel_open(
                str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
            )
            handle = be.native_embedder()
            baseline = dict(calls)  # construction may probe; the hot path may not
            kernel.attach_embedder(handle, media.get, lambda n: n in media)
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [(basic, 1, ['a colour swatch <img src="red.png">', "back"], [])],
                "error",
            )
            assert all(r[0] == "created" for r in results)
            nid = results[0][1]
            await kernel.settle()
            engine = kernel.engine_handle()
            has_image = engine.modality_contains("image", nid)
            await kernel.close()
            assert calls == baseline, f"kernel embeds re-entered the facade: {calls}"
            return has_image

        try:
            assert asyncio.run(flow())  # the image vector landed, natively
        finally:
            for name, orig in originals.items():
                setattr(be, name, orig)
