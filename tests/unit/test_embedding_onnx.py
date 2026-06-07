"""Tests for the shrike.embedding_onnx OnnxBackend (mocked onnxruntime/tokenizers).

Pooling math, fingerprint behaviour, file resolution, and the embed plumbing are
covered without a real ONNX model — onnxruntime/tokenizers are faked. A real
end-to-end embed against an actual model lives in the integration suite, run
against both backends.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import TEXT
from shrike.embedding_onnx import OnnxBackend

# -- Fakes for onnxruntime / tokenizers --------------------------------------


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOutput:
    def __init__(self, shape: list) -> None:
        self.shape = shape


class _FakeSession:
    """Returns an all-ones [B, S, H=4] last_hidden_state regardless of input."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._inputs = [
            _FakeInput("input_ids"),
            _FakeInput("attention_mask"),
            _FakeInput("token_type_ids"),
        ]

    def get_inputs(self) -> list[_FakeInput]:
        return self._inputs

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput([None, None, 4])]

    def run(self, _outputs: object, feed: dict) -> list[np.ndarray]:
        batch, seq = feed["input_ids"].shape
        return [np.ones((batch, seq, 4), dtype=np.float32)]


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


class _FakeTokenizer:
    def token_to_id(self, _tok: str) -> int:
        return 0

    def enable_padding(self, **_kwargs: object) -> None:
        pass

    def enable_truncation(self, **_kwargs: object) -> None:
        pass

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoding]:
        return [_FakeEncoding([1, 2, 3]) for _ in texts]


class _FakeTokenizerClass:
    @staticmethod
    def from_file(_path: str) -> _FakeTokenizer:
        return _FakeTokenizer()


def _model_dir(tmp_path: Path, *, content: bytes = b"onnx-bytes") -> Path:
    """A minimal HF-style ONNX model dir (model.onnx + tokenizer.json)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model.onnx").write_bytes(content)
    (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


# -- Pooling ------------------------------------------------------------------


class TestPooling:
    def test_mean_ignores_padding(self) -> None:
        be = OnnxBackend(model="x", pooling="mean")
        token_emb = np.array([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])
        mask = np.array([[1, 1, 0]])  # third token is padding
        assert be._pool(token_emb, mask).tolist() == [[2.0, 2.0]]

    def test_cls_takes_first_token(self) -> None:
        be = OnnxBackend(model="x", pooling="cls")
        token_emb = np.array([[[5.0, 6.0], [1.0, 1.0]]])
        mask = np.array([[1, 1]])
        assert be._pool(token_emb, mask).tolist() == [[5.0, 6.0]]

    def test_last_takes_last_real_token(self) -> None:
        be = OnnxBackend(model="x", pooling="last")
        token_emb = np.array([[[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]]])
        mask = np.array([[1, 1, 0]])  # last real token is index 1
        assert be._pool(token_emb, mask).tolist() == [[2.0, 2.0]]


# -- Construction -------------------------------------------------------------


class TestConstruction:
    def test_default_pooling_is_mean(self) -> None:
        assert OnnxBackend(model="x")._pooling == "mean"

    def test_rejects_none_pooling(self) -> None:
        with pytest.raises(ValueError, match="pooling"):
            OnnxBackend(model="x", pooling="none")

    def test_modalities_text_only(self) -> None:
        assert OnnxBackend(model="x").modalities == frozenset({TEXT})

    def test_default_provider_is_cpu(self) -> None:
        assert OnnxBackend(model="x")._providers == ["CPUExecutionProvider"]


# -- File resolution ----------------------------------------------------------


class TestResolveFiles:
    def test_dir_with_root_files(self, tmp_path: Path) -> None:
        _model_dir(tmp_path)
        onnx_path, tok_path = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.name == "model.onnx"
        assert tok_path.name == "tokenizer.json"

    def test_dir_with_onnx_subdir(self, tmp_path: Path) -> None:
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx" / "model.onnx").write_bytes(b"x")
        (tmp_path / "onnx" / "tokenizer.json").write_text("{}")
        onnx_path, tok_path = OnnxBackend(model=str(tmp_path))._resolve_files()
        assert onnx_path.parent.name == "onnx"
        assert tok_path.parent.name == "onnx"

    def test_direct_onnx_file(self, tmp_path: Path) -> None:
        _model_dir(tmp_path)
        be = OnnxBackend(model=str(tmp_path / "model.onnx"))
        onnx_path, tok_path = be._resolve_files()
        assert onnx_path.name == "model.onnx"
        assert tok_path.name == "tokenizer.json"

    def test_missing_tokenizer_raises(self, tmp_path: Path) -> None:
        (tmp_path / "model.onnx").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            OnnxBackend(model=str(tmp_path))._resolve_files()

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="model.onnx"):
            OnnxBackend(model=str(tmp_path))._resolve_files()

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            OnnxBackend(model=str(tmp_path / "nope"))._resolve_files()


# -- Fingerprint --------------------------------------------------------------


class TestFingerprint:
    def test_format_and_namespace(self, tmp_path: Path) -> None:
        _model_dir(tmp_path, content=b"hello")  # 5 bytes
        fp = OnnxBackend(model=str(tmp_path), pooling="mean").model_fingerprint()
        assert fp == f"onnx:model.onnx:5:pool=mean:textprep={EMBED_TEXT_VERSION}"
        # Namespaced distinctly from a llama fingerprint (meta:/file:).
        assert fp.split(":")[0] == "onnx"

    def test_pooling_changes_fingerprint(self, tmp_path: Path) -> None:
        _model_dir(tmp_path)
        mean_fp = OnnxBackend(model=str(tmp_path), pooling="mean").model_fingerprint()
        cls_fp = OnnxBackend(model=str(tmp_path), pooling="cls").model_fingerprint()
        assert mean_fp != cls_fp

    def test_content_change_changes_fingerprint(self, tmp_path: Path) -> None:
        d1 = _model_dir(tmp_path / "a", content=b"aa")
        d2 = _model_dir(tmp_path / "b", content=b"aaaaaa")
        assert (
            OnnxBackend(model=str(d1)).model_fingerprint()
            != OnnxBackend(model=str(d2)).model_fingerprint()
        )

    def test_normalize_not_in_fingerprint(self, tmp_path: Path) -> None:
        # Normalization is scale-only (cos is scale-invariant), so it must NOT
        # change the fingerprint — same reasoning as llama's --embd-normalize.
        _model_dir(tmp_path)
        on = OnnxBackend(model=str(tmp_path), normalize=True).model_fingerprint()
        off = OnnxBackend(model=str(tmp_path), normalize=False).model_fingerprint()
        assert on == off


# -- Lifecycle + embed (mocked session/tokenizer) -----------------------------


class TestLifecycleAndEmbed:
    def _start(self, be: OnnxBackend) -> None:
        # Inject fake onnxruntime/tokenizers modules into sys.modules so these
        # tests run WITHOUT the optional 'onnx' extra installed — the coverage
        # lane installs only `.[dev]`. `OnnxBackend.start()` imports both lazily,
        # so the fakes satisfy `import onnxruntime` / `from tokenizers import
        # Tokenizer` regardless of whether the real packages are present.
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = _FakeSession  # type: ignore[attr-defined]
        fake_tok = types.ModuleType("tokenizers")
        fake_tok.Tokenizer = _FakeTokenizerClass  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"onnxruntime": fake_ort, "tokenizers": fake_tok}):
            be.start()

    def test_running_toggles(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        assert be.running is False
        self._start(be)
        assert be.running is True
        be.stop()
        assert be.running is False

    def test_embed_shape_and_normalized(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)), pooling="mean", normalize=True)
        self._start(be)
        vecs = be.embed_texts(["a", "b"])
        assert len(vecs) == 2
        assert all(len(v) == 4 for v in vecs)
        # mean of all-ones rows = all-ones; L2-normalized over H=4 → 0.5 each.
        assert all(abs(x - 0.5) < 1e-6 for x in vecs[0])

    def test_embed_unnormalized(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)), normalize=False)
        self._start(be)
        vecs = be.embed_texts(["a"])
        assert all(abs(x - 1.0) < 1e-6 for x in vecs[0])  # raw mean of all-ones

    def test_embed_empty_returns_empty(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        assert be.embed_texts([]) == []

    def test_embed_raises_when_not_running(self) -> None:
        with pytest.raises(RuntimeError, match="not running"):
            OnnxBackend(model="x").embed_texts(["a"])

    def test_embedding_dim_from_static_shape(self, tmp_path: Path) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        assert be.embedding_dim() == 4

    def test_embedding_dim_none_when_stopped(self) -> None:
        assert OnnxBackend(model="x").embedding_dim() is None

    def test_start_missing_dependency_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import → ImportError
        with pytest.raises(ImportError, match="onnx"):
            be.start()


# -- Health -------------------------------------------------------------------


class TestHealth:
    def test_health_when_stopped(self) -> None:
        h = OnnxBackend(model="x").health()
        assert h["available"] is False
        assert h["backend"] == "onnx"
        assert h["providers"] == ["CPUExecutionProvider"]
