"""The CLIP backend against a real small CLIP export (image<->text shared space).

Mocked mechanics live in ``tests/unit/test_embedding_clip.py``; here we run ``ClipBackend``
against the actual ``Xenova/clip-vit-base-patch32`` ONNX graphs so the preprocessing, I/O, and
the shared-space property are exercised for real. The semantic assertion uses solid-colour
images (deterministic, no network beyond the cached model): a colour image lands nearer its own
colour word than unrelated concepts, proving a text query retrieves by image content. (Richer
image-by-text quality was measured in the Phase-3a eval, #193.)
"""

from __future__ import annotations

import colorsys
import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from shrike.embedding_clip import ClipBackend
from shrike.index import CALIB_MIN, activation_floor
from shrike.tools import ACTIVATION_MARGIN
from tests.integration.conftest import requires_clip, requires_shrike_native

pytestmark = [pytest.mark.integration, pytest.mark.embedding]

_CLIP_DIM = 512
# Unrelated query texts: a solid-colour image must land nearer *its own colour word* than any of
# these. (Deterministic, no network. NOTE: the comparison is colour-vs-*unrelated-concept*, not
# colour-vs-other-colour — the latter gap is ~0.05 and flips across int8 onnxruntime builds, the
# former is ~0.09 and robust. Richer real-image quality was measured in the Phase-3a eval, #193.)
_UNRELATED = ["a photograph of a cat", "a circuit diagram schematic", "a page of printed text"]


def _make(w, notes):
    import json

    return w.run_sync(lambda c: json.loads(c.upsert_notes(json.dumps(notes), "allow", False)))


# ONE started backend for the whole module (#441 — this file previously loaded
# the ~147 MB text+vision model once per class, and test_clip_native.py loaded a
# third identical copy): every consumer exercises the same default quantized
# graphs read-only, and the target runs serially (xdist=None in BUILD.bazel).
@pytest.fixture(scope="module")
def be(clip_model: Path) -> Iterator[ClipBackend]:
    backend = ClipBackend(model=str(clip_model))  # auto-discovers the quantized graphs
    backend.start()
    yield backend
    backend.stop()


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (256, 256)) -> bytes:
    import io

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
            iv = np.array(be.embed_images([Image.new("RGB", (256, 256), color)])[0])
            match = float(iv @ np.array(be.embed_texts([f"a solid {name} colour image"])[0]))
            others = [float(iv @ np.array(be.embed_texts([t])[0])) for t in _UNRELATED]
            # A text query lands nearer the matching image than unrelated ones — image-by-text.
            assert match > max(others) + 0.03, f"{name}: match={match:.3f} others={others}"

    def test_dims_normalized_and_distinct(self, be: ClipBackend) -> None:
        from PIL import Image

        assert be.embedding_dim() == _CLIP_DIM
        tvecs = be.embed_texts(["a cat", "a dog"])
        ivec = np.array(be.embed_images([Image.new("RGB", (300, 200), (10, 200, 10))])[0])
        # Text + image both land in the same 512-dim space, L2-normalized.
        assert all(len(v) == _CLIP_DIM for v in tvecs) and len(ivec) == _CLIP_DIM
        assert np.isclose(np.linalg.norm(tvecs[0]), 1.0) and np.isclose(np.linalg.norm(ivec), 1.0)
        # The encoders actually run (distinct inputs → distinct vectors).
        assert not np.allclose(tvecs[0], tvecs[1])

    def test_int8_clip_is_serial(self, be: ClipBackend) -> None:
        # The quantized graphs are batch-variant (dynamic int8), so the probe forces serial.
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
    """End-to-end per-modality index (#162 Phase 3c → search #201a): a text query retrieves a note
    by its image, and the per-modality image ranker surfaces it at rank-1 across the gap.
    Uses the shared module backend; each test builds its own (cheap) collection + index."""

    @staticmethod
    def _collection(tmp_path: Path):
        import os

        from PIL import Image

        from shrike.collection import CollectionWrapper

        w = CollectionWrapper(str(tmp_path / "c.anki2"))
        os.makedirs(w.media_dir, exist_ok=True)
        Image.new("RGB", (128, 128), (220, 30, 30)).save(os.path.join(w.media_dir, "red.png"))
        # The note's TEXT never names a colour; its meaning lives in the image.
        red = _make(
            w,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": 'study card <img src="red.png">', "Back": "."},
                }
            ],
        )[0]["id"]
        other = _make(
            w,
            [
                {
                    "deck": "Test",
                    "note_type": "Basic",
                    "fields": {"Front": "ancient rome", "Back": "."},
                }
            ],
        )[0]["id"]
        return w, red, other

    def _index(self, be: ClipBackend, tmp_path: Path, w):
        from shrike.index import VectorIndex
        from shrike.server import _make_image_resolver

        idx = VectorIndex(tmp_path / "index", backend=be)
        idx.set_image_resolver(*_make_image_resolver(w.media_dir))
        return idx

    def test_image_note_rank_one_via_image_ranker(self, be: ClipBackend, tmp_path: Path) -> None:

        w, red, other = self._collection(tmp_path)
        try:
            idx = self._index(be, tmp_path, w)
            from shrike.embedding_base import NoteEmbedInput

            inputs = w.run_sync(
                lambda c: [
                    NoteEmbedInput(note_id=n, text=t, image_names=imgs)
                    for n, t, imgs in c.note_embed_inputs([red, other])
                ]
            )
            idx.rebuild(inputs, col_mod=1, model_id=be.model_fingerprint())
            # red: text + image = 2 vectors (image in its own sub-index); other: text = 1.
            assert idx.size == 3
            assert len(idx._indexes["image"]) == 1  # exactly the red note's image vector

            # Per-modality retrieval (#201a): the image ranking is a separate signal, so the
            # image-bearing note surfaces at rank-1 *in that ranking* regardless of CLIP's modality
            # gap (text-text cos ~0.72 vs text-image ~0.32) — which a single deduped cosine ranking
            # could not deliver (the red note's own TEXT is "study card", naming no colour).
            matching = idx.search_by_modality(["a solid red colour image"], top_k=2)[0]
            assert matching["image"][0]["note_id"] == red

            # And it retrieves by image *content*: the red note's image vector is nearer the
            # matching colour query than an unrelated-concept query (the robust colour-vs-unrelated
            # regime — colour-vs-colour is ~0.05 and flips across int8 builds, so it's avoided).
            unrelated = idx.search_by_modality(["a circuit diagram schematic"], top_k=2)[0]
            assert matching["image"][0]["distance"] < unrelated["image"][0]["distance"]
        finally:
            w.close()

    # NOTE (#441): reconcile-on-image-removal and missing-media-skip are pinned
    # with mocks in tests/unit/test_index.py (the logic lives in VectorIndex and
    # never reaches the model for the interesting branch) — not re-proven here.

    @staticmethod
    def _colour_collection(tmp_path: Path, n: int):
        """A collection of ``n`` cards, each a distinct solid-colour image with colour-neutral text
        (card 0 is red). Enough cards to calibrate the activation gate on the real model."""
        import os

        from PIL import Image

        from shrike.collection import CollectionWrapper

        w = CollectionWrapper(str(tmp_path / "c.anki2"))
        os.makedirs(w.media_dir, exist_ok=True)
        ids: list[int] = []
        for i in range(n):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 0.9)  # hue 0 (card 0) is red
            rgb = (int(r * 255), int(g * 255), int(b * 255))
            fn = f"c{i}.png"
            Image.new("RGB", (96, 96), rgb).save(os.path.join(w.media_dir, fn))
            note = {
                "deck": "Test",
                "note_type": "Basic",
                "fields": {"Front": f'study card number {i} <img src="{fn}">', "Back": "."},
            }
            nid = _make(w, [note])[0]["id"]
            ids.append(nid)
        return w, ids

    def test_activation_gate_calibrates_and_passes_genuine_match(
        self, be: ClipBackend, tmp_path: Path
    ) -> None:

        # A ≥CALIB_MIN colour collection so calibration produces image stats on the real CLIP model.
        w, ids = self._colour_collection(tmp_path, CALIB_MIN + 2)
        try:
            idx = self._index(be, tmp_path, w)
            from shrike.embedding_base import NoteEmbedInput

            inputs = w.run_sync(
                lambda c: [
                    NoteEmbedInput(note_id=n, text=t, image_names=imgs)
                    for n, t, imgs in c.note_embed_inputs(ids)
                ]
            )
            idx.rebuild(inputs, col_mod=1, model_id=be.model_fingerprint())

            # Offline calibration (#201b) ran on the real model and produced image-modality stats.
            stats = idx.activation_stats
            assert stats["image"]["n"] >= CALIB_MIN
            assert stats["image"]["std"] > 0.0
            floor = activation_floor(stats["image"], ACTIVATION_MARGIN)
            assert floor is not None

            def _best_image_sim(query: str) -> float:
                ranking = idx.search_by_modality([query], top_k=1)[0]["image"]
                return 1.0 - ranking[0]["distance"]

            # A genuine colour-content query clears the floor → its image card passes the gate (the
            # gate must not suppress real matches), and it's distinctly stronger than an off-topic
            # query — which falls toward/below the floor and is gated out (the deterministic drop is
            # unit-tested in test_tools_search.py; colour-vs-unrelated is the robust ~0.09 regime).
            red_best = _best_image_sim("a solid red colour image")
            noise_best = _best_image_sim("a circuit diagram schematic")
            assert red_best > floor
            assert red_best > noise_best
        finally:
            w.close()


@requires_clip
@requires_shrike_native
class TestClipNativeSeam:
    """The native CLIP engine's binding seams (absorbed from test_clip_native.py,
    #441 — since the #278 cutover ClipBackend IS the native engine, so the
    file split documented a distinction that no longer exists; its fingerprint
    and retrieval tests duplicated conformance / TestClipModel)."""

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
        """#342 P2: the CLIP composition attaches native — ONE adapted engine
        serves both kernel halves, and neither text nor image embeds re-enter
        the facade (the counter pin)."""
        import asyncio

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
