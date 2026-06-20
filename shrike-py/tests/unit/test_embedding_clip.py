"""Tests for the CLIP backend facade (shrike.embedding_clip), with a fake native engine.

The shared-space *quality* (a text query near the matching image) needs a real model and lives
in the integration suite; here we fake the onnxruntime carrier and the native engine so the
mechanics — graph/variant discovery, preprocessor-config parsing, provider resolution,
fingerprint, health, and the batch-safety wiring — are covered without the `clip` extra.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from shrike.harness.engines.embedding.base import IMAGE, TEXT
from shrike.harness.engines.embedding.batching import BATCH_PROBE_TEXTS
from shrike.harness.engines.embedding.clip import ClipBackend
from shrike.harness.engines.embedding.onnx_common import resolve_execution_providers

_DIM = 8


# -- fakes -------------------------------------------------------------------


class _FakeEngine:
    """Content-independent (batch-safe) engine; records chunk sizes + init kwargs."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.text_calls: list[int] = []
        self.image_calls: list[int] = []
        self._providers = list(kwargs.get("providers") or ["CPUExecutionProvider"])

    def active_providers(self) -> list[str]:
        return self._providers

    def dim(self) -> int | None:
        return _DIM

    def embed_text_chunk(self, texts: list[str]) -> list[list[float]]:
        self.text_calls.append(len(texts))
        return [[1.0 / (_DIM**0.5)] * _DIM for _ in texts]

    def embed_image_chunk(self, images: list[bytes]) -> list[list[float]]:
        self.image_calls.append(len(images))
        return [[1.0 / (_DIM**0.5)] * _DIM for _ in images]


class _FakeVariantEngine(_FakeEngine):
    """Output direction depends on batch size → the probe forces serial."""

    def embed_text_chunk(self, texts: list[str]) -> list[list[float]]:
        self.text_calls.append(len(texts))
        return [[float(len(texts)), *([0.5] * (_DIM - 1))] for _ in texts]


class _FakeVisionVariantEngine(_FakeEngine):
    """Text batches safely, but the VISION graph is batch-variant (the mixed-precision
    case: fp text + int8 vision). The min(text, vision) probe must force the whole engine
    serial — the text probe alone would wrongly clear it."""

    def embed_image_chunk(self, images: list[bytes]) -> list[list[float]]:
        self.image_calls.append(len(images))
        # Output depends on the batch size → batched != serial → the vision probe trips.
        return [[float(len(images)), *([0.5] * (_DIM - 1))] for _ in images]


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
    engine_cls: type = _FakeEngine,
    available: list[str] | None = None,
) -> None:
    """Inject a fake onnxruntime carrier + native engine so start() runs without the extra."""
    fake_ort = types.ModuleType("onnxruntime")
    avail = available or ["CPUExecutionProvider"]
    fake_ort.get_available_providers = lambda: list(avail)  # type: ignore[attr-defined]

    fake_native = types.ModuleType("shrike_native")
    fake_native.init_onnx_runtime = lambda _path: None  # type: ignore[attr-defined]
    fake_native.ClipEmbedder = engine_cls  # type: ignore[attr-defined]

    with (
        patch.dict(sys.modules, {"onnxruntime": fake_ort, "shrike_native": fake_native}),
        patch(
            "shrike.harness.engines.embedding.onnx.locate_ort_dylib",
            lambda: Path("/fake/libort.so"),
        ),
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

    def test_missing_dep_raises_with_install_hint(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        with (
            patch.dict(sys.modules, {"onnxruntime": None}),
            pytest.raises(ImportError, match="pip install onnxruntime"),
        ):
            be.start()

    def test_embed_texts_shape(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        vecs = be.embed_texts(["a cat", "a dog"])
        assert len(vecs) == 2
        assert len(vecs[0]) == _DIM

    def test_embed_images_marshals_bytes(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        vecs = be.embed_images([b"img-bytes", b"img-bytes"])
        assert len(vecs) == 2
        assert len(vecs[0]) == _DIM
        assert be._native_engine.image_calls  # bytes reached the engine

    def test_engine_receives_preprocessor_scalars(self, tmp_path: Path) -> None:
        # Only scalars cross the FFI: mean/std/resize/crop parsed here, passed in.
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        kw = be._native_engine.init_kwargs
        assert kw["resize"] == 256 and kw["crop"] == 224
        assert pytest.approx(kw["image_mean"][0]) == 0.48145466
        assert kw["context"] == 77

    def test_dim_and_fingerprint(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        assert be.embedding_dim() == _DIM
        fp = be.model_fingerprint()
        # clip-rs: is the native engine's vector-space identity — indexes built
        # then load unchanged.
        assert fp.startswith("clip-rs:text_model.onnx:")
        assert "vision_model.onnx:" in fp and "imgprep=rs" in fp and "textprep=" in fp

    def test_auto_discovers_quantized_only(self, tmp_path: Path) -> None:
        # A quantized-only export (the CI fixture's shape) loads without any variant param.
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
        # Resize target is the `size`/`shortest_edge` field, crop is `crop_size` — not the same.
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)
        assert be._resize == 256 and be._crop == 224

    def test_scalar_size_and_crop(self, tmp_path: Path) -> None:
        # The bare-scalar `crop_size`/`size` form (some exports) must not crash start().
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

    def test_missing_image_mean_std_defaults_to_openai_clip(self, tmp_path: Path) -> None:
        # Some exports (e.g. MobileCLIP) omit image_mean/image_std; a CLIP image processor
        # falls back to the OpenAI CLIP constants, so start() must not KeyError and must
        # pass those defaults to the engine.
        d = _model_dir(tmp_path)
        (d / "preprocessor_config.json").write_text(
            json.dumps({"size": {"shortest_edge": 256}, "crop_size": {"height": 256}})
        )
        be = ClipBackend(model=str(d))
        _start(be)
        kw = be._native_engine.init_kwargs
        assert pytest.approx(kw["image_mean"][0]) == 0.48145466
        assert pytest.approx(kw["image_std"][0]) == 0.26862954

    def test_do_normalize_false_uses_identity(self, tmp_path: Path) -> None:
        # An explicit do_normalize:false means no normalization → identity mean/std,
        # even if stray image_mean/std values are present in the config.
        d = _model_dir(tmp_path)
        (d / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "size": 224,
                    "crop_size": 224,
                    "do_normalize": False,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                }
            )
        )
        be = ClipBackend(model=str(d))
        _start(be)
        kw = be._native_engine.init_kwargs
        assert kw["image_mean"] == [0.0, 0.0, 0.0]
        assert kw["image_std"] == [1.0, 1.0, 1.0]

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
        be._native_engine.text_calls.clear()
        be.embed_texts(["a", "b", "c"])
        assert be._native_engine.text_calls == [3]

    def test_variant_model_serial(self, tmp_path: Path) -> None:
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be, engine_cls=_FakeVariantEngine)
        assert be._safe_batch == 1
        be._native_engine.text_calls.clear()
        be.embed_texts(["a", "b", "c"])
        assert be._native_engine.text_calls == [1, 1, 1]

    def test_variant_vision_forces_serial(self, tmp_path: Path) -> None:
        # The mixed-precision guard: a pair (safe text + batch-variant vision) must take
        # min(text, vision) and embed BOTH paths serially — the text probe alone would clear it.
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be, engine_cls=_FakeVisionVariantEngine)
        assert be._safe_batch == 1
        be._native_engine.image_calls.clear()
        be.embed_images([b"x", b"y", b"z"])
        assert be._native_engine.image_calls == [1, 1, 1]
        # And the safe text path is dragged serial too (one safe_batch governs both halves).
        be._native_engine.text_calls.clear()
        be.embed_texts(["a", "b", "c"])
        assert be._native_engine.text_calls == [1, 1, 1]

    def test_safe_pair_keeps_full_batch(self, tmp_path: Path) -> None:
        # A uniform-safe pair must batch to the full set on BOTH paths — min(text, vision)
        # must not lower a safe pair's ceiling (the image set is sized to match the text set).
        be = ClipBackend(model=str(_model_dir(tmp_path)))
        _start(be)  # default fake is content-independent on both halves → safe
        assert be._safe_batch == len(BATCH_PROBE_TEXTS)
        be._native_engine.image_calls.clear()
        be.embed_images([b"a", b"b", b"c"])
        assert be._native_engine.image_calls == [3]

    def test_not_running_raises(self) -> None:
        be = ClipBackend(model="x")
        with pytest.raises(RuntimeError, match="not running"):
            be.embed_texts(["a"])
        with pytest.raises(RuntimeError, match="not running"):
            be.embed_images([b"x"])
