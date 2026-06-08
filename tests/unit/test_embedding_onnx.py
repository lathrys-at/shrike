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
    def __init__(self, name: str, type_: str = "tensor(int64)") -> None:
        self.name = name
        self.type = type_


class _FakeOutput:
    def __init__(self, shape: list) -> None:
        self.shape = shape


class _FakeSession:
    """All-ones session: 3D [B, S, H=4] last_hidden_state, int64 inputs.

    Subclasses tweak ``_input_type`` (declared input dtype) and ``_output_ndim``
    (2 = pre-pooled sentence embedding, 3 = token-level). ``last_feed`` records the
    most recent feed so tests can assert the fed dtype.
    """

    _input_type = "tensor(int64)"
    _output_ndim = 3
    _input_names = ("input_ids", "attention_mask", "token_type_ids")

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._inputs = [_FakeInput(n, self._input_type) for n in self._input_names]
        self.last_feed: dict | None = None

    def get_inputs(self) -> list[_FakeInput]:
        return self._inputs

    def get_outputs(self) -> list[_FakeOutput]:
        shape = [None, None, 4] if self._output_ndim == 3 else [None, 4]
        return [_FakeOutput(shape)]

    def run(self, _outputs: object, feed: dict) -> list[np.ndarray]:
        self.last_feed = feed
        batch, seq = feed["input_ids"].shape
        if self._output_ndim == 2:
            return [np.ones((batch, 4), dtype=np.float32)]
        return [np.ones((batch, seq, 4), dtype=np.float32)]


class _FakeSession2D(_FakeSession):
    """Emits an already-pooled [B, H] sentence embedding (no token axis)."""

    _output_ndim = 2


class _FakeSessionInt32(_FakeSession):
    """Declares int32 inputs (onnxruntime won't auto-cast int64 → int32)."""

    _input_type = "tensor(int32)"


class _FakeSession2Input(_FakeSession):
    """Declares only input_ids + attention_mask (no token_type_ids), like DistilBERT."""

    _input_names = ("input_ids", "attention_mask")


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


class _FakeTokenizer:
    """BERT/WordPiece-style: has a "[PAD]" token (id 5, deliberately non-zero so the
    fallback id 0 is distinguishable). Records the pad_id passed to enable_padding."""

    padding_pad_id: int | None = None

    def token_to_id(self, tok: str) -> int | None:
        return 5 if tok == "[PAD]" else 0

    def enable_padding(self, *, pad_id: int = 0, **_kwargs: object) -> None:
        self.padding_pad_id = pad_id

    def enable_truncation(self, **_kwargs: object) -> None:
        pass

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoding]:
        return [_FakeEncoding([1, 2, 3]) for _ in texts]


class _FakeTokenizerRoberta(_FakeTokenizer):
    """RoBERTa/BPE-style: no "[PAD]" but a "<pad>" (id 1)."""

    def token_to_id(self, tok: str) -> int | None:
        return 1 if tok == "<pad>" else None


class _FakeTokenizerNoPadToken(_FakeTokenizer):
    """Neither "[PAD]" nor "<pad>" — forces the id-0 final fallback."""

    def token_to_id(self, _tok: str) -> int | None:
        return None


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

    def test_max_length_default_and_override(self) -> None:
        # None (e.g. no --embedding-context-size) → the 256 default; else honored.
        assert OnnxBackend(model="x")._max_length == 256
        assert OnnxBackend(model="x", max_length=None)._max_length == 256
        assert OnnxBackend(model="x", max_length=512)._max_length == 512


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
    def _start(
        self,
        be: OnnxBackend,
        session_cls: type = _FakeSession,
        tokenizer_cls: type = _FakeTokenizer,
    ) -> None:
        # Inject fake onnxruntime/tokenizers modules into sys.modules so these
        # tests run WITHOUT the optional 'onnx' extra installed — the coverage
        # lane installs only `.[dev]`. `OnnxBackend.start()` imports both lazily,
        # so the fakes satisfy `import onnxruntime` / `from tokenizers import
        # Tokenizer` regardless of whether the real packages are present.
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = session_cls  # type: ignore[attr-defined]
        fake_tok = types.ModuleType("tokenizers")

        class _TokFactory:
            @staticmethod
            def from_file(_path: str) -> object:
                return tokenizer_cls()

        fake_tok.Tokenizer = _TokFactory  # type: ignore[attr-defined]
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

    def test_pre_pooled_2d_output_used_directly(self, tmp_path: Path) -> None:
        # A model whose first output is already a [B, H] sentence embedding is
        # used as-is, not pooled — must not crash _pool with an IndexError/ValueError.
        be = OnnxBackend(model=str(_model_dir(tmp_path)), normalize=False)
        self._start(be, _FakeSession2D)
        vecs = be.embed_texts(["a", "b"])
        assert len(vecs) == 2
        assert all(len(v) == 4 for v in vecs)
        assert all(abs(x - 1.0) < 1e-6 for x in vecs[0])  # raw [1,1,1,1], no pooling

    def test_inputs_cast_to_declared_int32(self, tmp_path: Path) -> None:
        # onnxruntime won't auto-cast; an int32-input graph must be fed int32.
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be, _FakeSessionInt32)
        be.embed_texts(["a"])
        assert be._session.last_feed["input_ids"].dtype == np.int32

    def test_feed_filter_drops_undeclared_inputs(self, tmp_path: Path) -> None:
        # A model declaring only input_ids+attention_mask (DistilBERT/RoBERTa):
        # token_type_ids must be filtered out of the feed, since onnxruntime rejects
        # an input the graph doesn't declare.
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be, _FakeSession2Input)
        be.embed_texts(["a"])
        assert set(be._session.last_feed) == {"input_ids", "attention_mask"}

    def test_pad_token_used_when_present(self, tmp_path: Path) -> None:
        # A WordPiece tokenizer with a real "[PAD]" id uses it (the fake's "[PAD]" is
        # id 5, distinct from both the "<pad>" path and the 0 fallback).
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be)
        assert be._tokenizer.padding_pad_id == 5

    def test_pad_token_resolves_roberta_pad(self, tmp_path: Path) -> None:
        # No "[PAD]" but a "<pad>" (RoBERTa/BPE): resolve it rather than fall to 0 —
        # RoBERTa's position ids depend on the real pad id. (The real DistilRoBERTa
        # test exercises this end to end.)
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be, tokenizer_cls=_FakeTokenizerRoberta)
        assert be._tokenizer.padding_pad_id == 1
        assert len(be.embed_texts(["a", "b"])[0]) == 4  # still embeds with padding

    def test_pad_token_fallback_when_none(self, tmp_path: Path) -> None:
        # Neither "[PAD]" nor "<pad>" in the vocab: final fallback to id 0. (Padded
        # positions are masked out in _pool regardless — see TestPooling.)
        be = OnnxBackend(model=str(_model_dir(tmp_path)))
        self._start(be, tokenizer_cls=_FakeTokenizerNoPadToken)
        assert be._tokenizer.padding_pad_id == 0

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
