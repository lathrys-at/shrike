"""The CLIP backend against a real small CLIP export (image<->text shared space).

Mocked mechanics live in ``tests/unit/test_embedding_clip.py``; here we run ``ClipBackend``
against the actual ``Xenova/clip-vit-base-patch32`` ONNX graphs so the preprocessing, I/O, and
the shared-space property are exercised for real. The semantic assertion uses solid-colour
images (deterministic, no network beyond the cached model) — CLIP reliably places a red image
nearer "a solid red image" than "a solid blue image", proving a text query retrieves by image
content. (Richer image-by-text quality was measured in the Phase-3a eval, #193.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shrike.embedding_clip import ClipBackend
from tests.integration.conftest import requires_clip

pytestmark = [pytest.mark.integration, pytest.mark.embedding]

_CLIP_DIM = 512


def _backend(clip_model: Path) -> ClipBackend:
    be = ClipBackend(model=str(clip_model), variant="quantized")
    be.start()
    return be


@requires_clip
class TestClipModel:
    def test_shared_space_retrieves_by_image(self, clip_model: Path) -> None:
        from PIL import Image

        be = _backend(clip_model)
        red = Image.new("RGB", (256, 256), (220, 30, 30))
        blue = Image.new("RGB", (256, 256), (30, 30, 220))
        ri = np.array(be.embed_images([red])[0])
        bi = np.array(be.embed_images([blue])[0])
        tr = np.array(be.embed_texts(["a solid red image"])[0])
        tb = np.array(be.embed_texts(["a solid blue image"])[0])
        # A text query lands nearer the matching image than the mismatching one — image-by-text.
        assert float(ri @ tr) > float(ri @ tb)
        assert float(bi @ tb) > float(bi @ tr)

    def test_dims_normalized_and_distinct(self, clip_model: Path) -> None:
        from PIL import Image

        be = _backend(clip_model)
        assert be.embedding_dim() == _CLIP_DIM
        tvecs = be.embed_texts(["a cat", "a dog"])
        ivec = np.array(be.embed_images([Image.new("RGB", (300, 200), (10, 200, 10))])[0])
        # Text + image both land in the same 512-dim space, L2-normalized.
        assert all(len(v) == _CLIP_DIM for v in tvecs) and len(ivec) == _CLIP_DIM
        assert np.isclose(np.linalg.norm(tvecs[0]), 1.0) and np.isclose(np.linalg.norm(ivec), 1.0)
        # The encoders actually run (distinct inputs → distinct vectors).
        assert not np.allclose(tvecs[0], tvecs[1])

    def test_int8_clip_is_serial(self, clip_model: Path) -> None:
        # The quantized graphs are batch-variant (dynamic int8), so the probe forces serial.
        be = _backend(clip_model)
        assert be._safe_batch == 1
        assert be.health()["batch"] == "serial"

    def test_health(self, clip_model: Path) -> None:
        be = _backend(clip_model)
        h = be.health()
        assert h["available"] is True
        assert h["backend"] == "clip"
        assert h["modalities"] == ["image", "text"]
        assert h["provider"] == "CPUExecutionProvider"
