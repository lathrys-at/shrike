"""Parity tests for the native CLIP engine (#271).

The conformance harness (test_backend_conformance.py, case ``clip-rs-vit-b32``)
covers the protocol surface and text-path parity. This file pins what is
specific to the vision path: the native preprocessing pipeline (image crate)
is pixel-different from PIL's, so image vectors are asserted as **semantically
equivalent** (high cosine agreement + identical retrieval behaviour), never
byte-equal — which is exactly why the kind namespaces its fingerprint
``clip-rs:`` (decision recorded on the issue).
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
class TestClipNativeParity:
    @pytest.fixture(scope="class")
    def backends(self, clip_model: Path) -> Iterator[tuple]:
        from shrike.embedding_clip import ClipBackend

        py = ClipBackend(model=str(clip_model))
        rs = ClipBackend(model=str(clip_model), native=True)
        py.start()
        rs.start()
        yield py, rs
        py.stop()
        rs.stop()

    def test_fingerprints_namespace_separately(self, backends: tuple) -> None:
        py, rs = backends
        assert py.model_fingerprint().startswith("clip:")
        assert rs.model_fingerprint().startswith("clip-rs:")

    def test_text_vectors_agree(self, backends: tuple) -> None:
        # Same tokenizer ids, same graph, same dylib — text vectors agree at
        # float-noise tier (normalization summation order only).
        py, rs = backends
        texts = ["a solid red colour image", "ancient rome"]
        pv = np.array(py.embed_texts(texts), dtype=np.float32)
        rv = np.array(rs.embed_texts(texts), dtype=np.float32)
        np.testing.assert_allclose(pv, rv, atol=1e-4)

    def test_image_vectors_semantically_equivalent(self, backends: tuple) -> None:
        # PIL bicubic vs image-crate Catmull-Rom: pixels differ, vectors must
        # still agree strongly (same content, same graph).
        py, rs = backends
        img = _png_bytes((200, 30, 30))
        pv = np.array(py.embed_images([img])[0], dtype=np.float32)
        rv = np.array(rs.embed_images([img])[0], dtype=np.float32)
        assert _cos(pv, rv) > 0.98

    def test_cross_modal_retrieval_matches_python(self, backends: tuple) -> None:
        # The retrieval-quality equivalence gate: under both engines, a red
        # image is closer to the red-text query than to an unrelated one.
        py, rs = backends
        img = _png_bytes((200, 30, 30))
        for be in (py, rs):
            ivec = np.array(be.embed_images([img])[0], dtype=np.float32)
            red = np.array(be.embed_texts(["a solid red colour image"])[0], dtype=np.float32)
            other = np.array(be.embed_texts(["ancient rome"])[0], dtype=np.float32)
            assert _cos(ivec, red) > _cos(ivec, other)

    def test_image_input_forms_agree(self, backends: tuple, tmp_path: Path) -> None:
        # bytes / path / PIL image all land on the same vector under the native
        # engine (the facade converts losslessly to bytes for the FFI).
        from PIL import Image

        _, rs = backends
        data = _png_bytes((10, 200, 10))
        path = tmp_path / "green.png"
        path.write_bytes(data)
        pil = Image.open(io.BytesIO(data))

        v_bytes = np.array(rs.embed_images([data])[0], dtype=np.float32)
        v_path = np.array(rs.embed_images([str(path)])[0], dtype=np.float32)
        v_pil = np.array(rs.embed_images([pil])[0], dtype=np.float32)
        np.testing.assert_array_equal(v_bytes, v_path)
        np.testing.assert_allclose(v_bytes, v_pil, atol=1e-5)
