# NOTE (#278 cutover): the Python-engine parity cases (text float-noise
# agreement, image semantic-equivalence vs PIL) retired with the Python engine;
# the retrieval-quality and input-form gates below are what remain meaningful.
"""Vision-path tests for the native CLIP engine (#271).

The conformance harness (test_backend_conformance.py, case ``clip-vit-b32``)
covers the protocol surface and text-path determinism. This file pins what is
specific to the vision path: the fingerprint namespace (``clip-rs:``, the
engine's frozen vector-space identity), cross-modal retrieval quality, and that
every accepted image input form (bytes / path / PIL) lands on the same vector.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests.integration.conftest import requires_clip, requires_shrike_native

pytestmark = [pytest.mark.integration, pytest.mark.embedding]


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (256, 256)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@requires_clip
@requires_shrike_native
class TestClipVisionPath:
    @pytest.fixture(scope="class")
    def be(self, clip_model: Path) -> Iterator:
        from shrike.embedding_clip import ClipBackend

        backend = ClipBackend(model=str(clip_model))
        backend.start()
        yield backend
        backend.stop()

    def test_fingerprint_namespace(self, be) -> None:
        # clip-rs: is the native engine's vector-space identity, kept verbatim
        # from the dual-engine bake so indexes built then load without a rebuild.
        assert be.model_fingerprint().startswith("clip-rs:")

    def test_cross_modal_retrieval(self, be) -> None:
        # The retrieval-quality gate: a red image is closer to the red-text
        # query than to an unrelated one.
        img = _png_bytes((200, 30, 30))
        ivec = np.array(be.embed_images([img])[0], dtype=np.float32)
        red = np.array(be.embed_texts(["a solid red colour image"])[0], dtype=np.float32)
        other = np.array(be.embed_texts(["ancient rome"])[0], dtype=np.float32)
        assert _cos(ivec, red) > _cos(ivec, other)

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
