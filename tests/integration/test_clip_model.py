"""The CLIP backend against a real small CLIP export (image<->text shared space).

Mocked mechanics live in ``tests/unit/test_embedding_clip.py``; here we run ``ClipBackend``
against the actual ``Xenova/clip-vit-base-patch32`` ONNX graphs so the preprocessing, I/O, and
the shared-space property are exercised for real. The semantic assertion uses solid-colour
images (deterministic, no network beyond the cached model): a colour image lands nearer its own
colour word than unrelated concepts, proving a text query retrieves by image content. (Richer
image-by-text quality was measured in the Phase-3a eval, #193.)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from shrike.embedding_clip import ClipBackend
from tests.integration.conftest import requires_clip

pytestmark = [pytest.mark.integration, pytest.mark.embedding]

_CLIP_DIM = 512
# Unrelated query texts: a solid-colour image must land nearer *its own colour word* than any of
# these. (Deterministic, no network. NOTE: the comparison is colour-vs-*unrelated-concept*, not
# colour-vs-other-colour — the latter gap is ~0.05 and flips across int8 onnxruntime builds, the
# former is ~0.09 and robust. Richer real-image quality was measured in the Phase-3a eval, #193.)
_UNRELATED = ["a photograph of a cat", "a circuit diagram schematic", "a page of printed text"]


@requires_clip
class TestClipModel:
    # One started backend for the whole class: every test here exercises the *same*
    # default quantized graphs read-only (embed/health/_safe_batch), so a per-test
    # ClipBackend.start() would reload the ~147 MB text+vision model 4× for no reason
    # — the dominant cost in this lane. Class-scoped, torn down once.
    @pytest.fixture(scope="class")
    def be(self, clip_model: Path) -> Iterator[ClipBackend]:
        backend = ClipBackend(model=str(clip_model))  # auto-discovers the quantized graphs
        backend.start()
        yield backend
        backend.stop()

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
class TestClipImageIndex:
    """End-to-end multi-vector index (#162 Phase 3c): a text query retrieves a note by its image."""

    # Class-scoped started backend (same rationale as TestClipModel, #215): load the ~147 MB CLIP
    # once for the class. Each test builds its own (cheap) collection + index against it.
    @pytest.fixture(scope="class")
    def be(self, clip_model: Path) -> Iterator[ClipBackend]:
        backend = ClipBackend(model=str(clip_model))
        backend.start()
        yield backend
        backend.stop()

    @staticmethod
    def _collection(tmp_path: Path):
        import os

        from PIL import Image

        from shrike.collection import CollectionWrapper

        w = CollectionWrapper(str(tmp_path / "c.anki2"))
        os.makedirs(w.media_dir, exist_ok=True)
        Image.new("RGB", (128, 128), (220, 30, 30)).save(os.path.join(w.media_dir, "red.png"))
        # The note's TEXT never names a colour; its meaning lives in the image.
        red = w.run_sync(
            lambda _c: w._upsert_notes(
                [
                    {
                        "deck": "Test",
                        "note_type": "Basic",
                        "fields": {"Front": 'study card <img src="red.png">', "Back": "."},
                    }
                ]
            )
        )[0]["id"]
        other = w.run_sync(
            lambda _c: w._upsert_notes(
                [
                    {
                        "deck": "Test",
                        "note_type": "Basic",
                        "fields": {"Front": "ancient rome", "Back": "."},
                    }
                ]
            )
        )[0]["id"]
        return w, red, other

    def _index(self, be: ClipBackend, tmp_path: Path, w):
        from shrike.index import VectorIndex
        from shrike.server import _make_image_resolver

        idx = VectorIndex(tmp_path / "index", backend=be)
        idx.set_image_resolver(_make_image_resolver(w.media_dir))
        return idx

    def test_note_image_is_indexed_and_retrievable(self, be: ClipBackend, tmp_path: Path) -> None:
        from shrike.collection import CollectionWrapper

        w, red, other = self._collection(tmp_path)
        try:
            idx = self._index(be, tmp_path, w)
            inputs = w.run_sync(lambda c: CollectionWrapper._note_embed_inputs(c, [red, other]))
            idx.rebuild(inputs, col_mod=1, model_id=be.model_fingerprint())
            # The image vector is indexed (red: text + image = 2 vectors; other: text = 1).
            assert idx.size == 3
            # The image-bearing note is retrievable. NB: *ranking* image hits above competing text
            # across CLIP's modality gap (text-text cos ~0.72 vs text-image ~0.32) is rank fusion —
            # the Search epic (#180) / Phase 3d. 3c stores the image vectors so fusion can use them;
            # here we assert the data layer (indexed + retrievable; differentiation tested below).
            nids = [r["note_id"] for r in idx.search(["a solid red colour image"], top_k=2)[0]]
            assert red in nids
        finally:
            w.close()

    def test_reconcile_reembeds_when_image_removed(self, be: ClipBackend, tmp_path: Path) -> None:
        from shrike.collection import CollectionWrapper
        from shrike.index import NoteEmbedInput

        w, red, other = self._collection(tmp_path)
        try:
            idx = self._index(be, tmp_path, w)
            mid = be.model_fingerprint()
            inputs = w.run_sync(lambda c: CollectionWrapper._note_embed_inputs(c, [red, other]))
            idx.rebuild(inputs, col_mod=1, model_id=mid)
            assert idx.size == 3
            # The red note loses its image (its embedding fingerprint changes) → reconcile drops
            # the image vector for exactly that note; the unrelated note is untouched.
            idx.reconcile(
                [NoteEmbedInput(red, "study card", []), NoteEmbedInput(other, "ancient rome", [])],
                col_mod=2,
                model_id=mid,
            )
            assert idx.size == 2  # red: text only now ; rome: text
        finally:
            w.close()

    def test_missing_media_file_skipped(self, be: ClipBackend, tmp_path: Path) -> None:
        from shrike.index import NoteEmbedInput

        w, red, other = self._collection(tmp_path)
        try:
            idx = self._index(be, tmp_path, w)
            # Reference a file that isn't in the media dir → skipped, text still indexed, no crash.
            idx.rebuild(
                [NoteEmbedInput(red, "study card", ["does-not-exist.png"])],
                col_mod=1,
                model_id=be.model_fingerprint(),
            )
            assert idx.size == 1
        finally:
            w.close()
