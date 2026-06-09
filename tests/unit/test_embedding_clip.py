"""Tests for the CLIP backend (shrike.embedding_clip), with onnxruntime/tokenizers/PIL faked.

The shared-space *quality* (a text query near the matching image) needs a real model and lives
in the integration suite; here we fake the heavy deps so the mechanics — the text/image feeds,
L2 normalization, the image-preprocessing math, provider resolution, fingerprint, health, and
the batch-safety wiring — are covered without the `clip` extra (the coverage lane installs only
`.[dev]`).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shrike.embedding_base import IMAGE, TEXT
from shrike.embedding_clip import ClipBackend
from shrike.embedding_onnx_common import resolve_execution_providers

_DIM = 8


# -- fakes -------------------------------------------------------------------


class _FakeInput:
    def __init__(self, name: str, type: str = "tensor(int64)") -> None:
        self.name = name
        self.type = type


class _FakeOutput:
    def __init__(self, shape: list) -> None:
        self.shape = shape


class _FakeTextSession:
    def __init__(self, providers: list[str] | None = None) -> None:
        self._providers = list(providers or ["CPUExecutionProvider"])
        self.last_feed: dict | None = None
        self.run_calls: list[int] = []

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput("input_ids")]  # CLIP text declares only input_ids

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput(["batch", _DIM])]  # text_embeds [batch, dim], static dim

    def get_providers(self) -> list[str]:
        return self._providers

    def run(self, _outputs: object, feed: dict) -> list[np.ndarray]:
        self.last_feed = feed
        n = feed["input_ids"].shape[0]
        self.run_calls.append(n)
        # Deterministic, content-independent → batch-safe (all-ones rows, distinct magnitude).
        return [np.ones((n, _DIM), dtype=np.float32) * 2.0]


class _FakeVisionSession(_FakeTextSession):
    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput("pixel_values", "tensor(float)")]

    def run(self, _outputs: object, feed: dict) -> list[np.ndarray]:
        self.last_feed = feed
        n = feed["pixel_values"].shape[0]
        self.run_calls.append(n)
        return [np.full((n, _DIM), 3.0, dtype=np.float32)]


class _FakeVariantTextSession(_FakeTextSession):
    """Output direction depends on batch size → the probe forces serial."""

    def run(self, _outputs: object, feed: dict) -> list[np.ndarray]:
        self.last_feed = feed
        n = feed["input_ids"].shape[0]
        self.run_calls.append(n)
        arr = np.ones((n, _DIM), dtype=np.float32)
        arr[:, 0] = float(n)
        return [arr]


class _FakeMaskedTextSession(_FakeTextSession):
    """Declares both input_ids (int32) and attention_mask (int64) — exercises the F3 feed path."""

    def get_inputs(self) -> list[_FakeInput]:
        return [
            _FakeInput("input_ids", "tensor(int32)"),
            _FakeInput("attention_mask", "tensor(int64)"),
        ]


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _FakeTokenizer:
    def enable_truncation(self, **_kw: object) -> None:
        pass

    def enable_padding(self, *, length: int = 77, **_kw: object) -> None:
        self._length = length

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoding]:
        return [_FakeEncoding([1] * 77) for _ in texts]


class _FakeImage:
    """A minimal PIL.Image stand-in: tracks size, returns a constant-value array."""

    def __init__(self, size: tuple[int, int] = (300, 200), value: int = 128) -> None:
        self.size = size
        self._value = value

    def convert(self, _mode: str) -> _FakeImage:
        return self

    def resize(self, size: tuple[int, int], _resample: object = None) -> _FakeImage:
        self.size = size
        return self

    def crop(self, box: tuple[int, int, int, int]) -> _FakeImage:
        left, top, right, bottom = box
        self.size = (right - left, bottom - top)
        return self

    def __array__(self, dtype: object = None) -> np.ndarray:
        w, h = self.size
        arr = np.full((h, w, 3), self._value, dtype=np.uint8)
        return arr.astype(dtype) if dtype else arr


def _model_dir(tmp_path: Path, *, variant: str = "") -> Path:
    """A CLIP model dir: onnx/text_model*.onnx + vision + tokenizer + preprocessor_config."""
    (tmp_path / "onnx").mkdir(parents=True, exist_ok=True)
    suffix = f"_{variant}" if variant else ""
    (tmp_path / "onnx" / f"text_model{suffix}.onnx").write_bytes(b"text-graph")
    (tmp_path / "onnx" / f"vision_model{suffix}.onnx").write_bytes(b"vision-graph-bytes")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                # size (resize) deliberately != crop_size, so a test can tell they're read apart.
                "size": {"shortest_edge": 256},
                "crop_size": {"height": 224, "width": 224},
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
            }
        )
    )
    return tmp_path


def _start(
    be: ClipBackend,
    *,
    text_session: type = _FakeTextSession,
    available: list[str] | None = None,
) -> None:
    """Inject fake onnxruntime/tokenizers/PIL so start() runs without the clip extra."""
    fake_ort = types.ModuleType("onnxruntime")
    avail = available or ["CPUExecutionProvider"]

    def _make_session(path: str, providers: list[str] | None = None) -> object:
        cls = text_session if "text_model" in path else _FakeVisionSession
        return cls(providers)

    fake_ort.InferenceSession = _make_session  # type: ignore[attr-defined]
    fake_ort.get_available_providers = lambda: list(avail)  # type: ignore[attr-defined]

    fake_tok = types.ModuleType("tokenizers")
    fake_tok.Tokenizer = types.SimpleNamespace(from_file=lambda _p: _FakeTokenizer())  # type: ignore[attr-defined]

    fake_pil = types.ModuleType("PIL")
    fake_image = types.ModuleType("PIL.Image")
    fake_image.open = lambda _fp: _FakeImage()  # type: ignore[attr-defined]
    fake_image.Resampling = types.SimpleNamespace(BICUBIC=3)  # type: ignore[attr-defined]
    fake_pil.Image = fake_image  # type: ignore[attr-defined]

    with patch.dict(
        sys.modules,
        {"onnxruntime": fake_ort, "tokenizers": fake_tok, "PIL": fake_pil, "PIL.Image": fake_image},
    ):
        be.start()


# -- tests -------------------------------------------------------------------


class TestProviderResolution:
    def test_unavailable_dropped_cpu_appended(self) -> None:
        resolved, dropped = resolve_execution_providers(
            ["CPUExecutionProvider"], ["CUDAExecutionProvider"]
        )
        assert resolved == ["CPUExecutionProvider"]
        assert dropped == ["CUDAExecutionProvider"]

    def test_available_kept_first_cpu_fallback(self) -> None:
        resolved, dropped = resolve_execution_providers(
            ["CoreMLExecutionProvider", "CPUExecutionProvider"], ["CoreMLExecutionProvider"]
        )
        assert resolved == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert dropped == []


class TestClipBackend:
    def test_modalities(self) -> None:
        assert ClipBackend(model="x").modalities == frozenset({TEXT, IMAGE})

    def test_missing_dep_raises_with_clip_hint(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        with (
            patch.dict(sys.modules, {"onnxruntime": None}),
            pytest.raises(ImportError, match="clip"),
        ):
            be.start()

    def test_embed_texts_feeds_input_ids(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        vecs = be.embed_texts(["a cat", "a dog"])
        assert len(vecs) == 2
        assert len(vecs[0]) == _DIM
        assert set(be._text_sess.last_feed) == {"input_ids"}
        assert be._text_sess.last_feed["input_ids"].shape == (2, 77)
        # L2-normalized.
        assert np.isclose(np.linalg.norm(vecs[0]), 1.0)

    def test_embed_images_preprocesses_to_pixel_values(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        vecs = be.embed_images([b"img-bytes", b"img-bytes"])
        assert len(vecs) == 2
        assert len(vecs[0]) == _DIM
        # pixel_values are [B, 3, 224, 224].
        assert be._vis_sess.last_feed["pixel_values"].shape == (2, 3, 224, 224)
        assert np.isclose(np.linalg.norm(vecs[0]), 1.0)

    def test_preprocess_normalizes_and_transposes(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        chw = be._preprocess(_FakeImage(value=128))
        assert chw.shape == (3, 224, 224)
        # (128/255 - mean) / std, per channel.
        expected = (128 / 255.0 - be._mean) / be._std
        assert np.allclose(chw[:, 0, 0], expected)

    def test_dim_and_fingerprint(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        assert be.embedding_dim() == _DIM
        fp = be.model_fingerprint()
        assert fp.startswith("clip:text_model.onnx:")
        assert "vision_model.onnx:" in fp and "imgprep=" in fp and "textprep=" in fp

    def test_auto_discovers_quantized_only(self, tmp_path: Path) -> None:
        # A quantized-only export (the CI fixture's shape) loads without any variant param (F1).
        be = ClipBackend(model=str(_model_dir(tmp_path, variant="quantized")))
        _start(be)
        assert be._text_path is not None and be._text_path.name == "text_model_quantized.onnx"
        assert be._vis_path is not None and be._vis_path.name == "vision_model_quantized.onnx"

    def test_prefers_full_precision_over_quantized(self, tmp_path: Path) -> None:
        # A dir with both → the full-precision pair wins (better quality, and it batches).
        d = _model_dir(tmp_path)  # plain text_model.onnx + vision_model.onnx
        (d / "onnx" / "text_model_quantized.onnx").write_bytes(b"q")
        (d / "onnx" / "vision_model_quantized.onnx").write_bytes(b"q")
        be = ClipBackend(model=str(d))
        _start(be)
        assert be._text_path is not None and be._text_path.name == "text_model.onnx"

    def test_reads_size_and_crop_independently(self, tmp_path: Path) -> None:
        # F2: resize target is the `size`/`shortest_edge` field, crop is `crop_size` — not the same.
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        assert be._resize == 256 and be._crop == 224

    def test_scalar_size_and_crop(self, tmp_path: Path) -> None:
        # F5: the bare-scalar `crop_size`/`size` form (some exports) must not crash start().
        d = _model_dir(tmp_path)
        (d / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "size": 248,
                    "crop_size": 224,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                }
            )
        )
        be = ClipBackend(model=str(d))
        _start(be)
        assert be._resize == 248 and be._crop == 224

    def test_feeds_declared_attention_mask_with_dtype(self, tmp_path: Path) -> None:
        # F3: a text graph declaring input_ids(int32)+attention_mask gets both, each its dtype.
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be, text_session=_FakeMaskedTextSession)
        be.embed_texts(["a cat"])
        feed = be._text_sess.last_feed
        assert set(feed) == {"input_ids", "attention_mask"}
        assert feed["input_ids"].dtype == np.int32  # declared int32 → cast down
        assert feed["attention_mask"].dtype == np.int64

    def test_health(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        h = be.health()
        assert h["available"] is True
        assert h["backend"] == "clip"
        assert h["modalities"] == ["image", "text"]
        assert h["provider"] == "CPUExecutionProvider"

    def test_batch_safe_model_batches(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)  # default fake is content-independent → safe
        assert be._safe_batch >= 2
        be._text_sess.run_calls.clear()
        be.embed_texts(["a", "b", "c"])
        assert be._text_sess.run_calls == [3]

    def test_variant_model_serial(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be, text_session=_FakeVariantTextSession)
        assert be._safe_batch == 1
        be._text_sess.run_calls.clear()
        be.embed_texts(["a", "b", "c"])
        assert be._text_sess.run_calls == [1, 1, 1]

    def test_not_running_raises(self) -> None:
        be = ClipBackend(model="x")
        with pytest.raises(RuntimeError, match="not running"):
            be.embed_texts(["a"])
        with pytest.raises(RuntimeError, match="not running"):
            be.embed_images([b"x"])
