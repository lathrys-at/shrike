"""Real ONNX models exercised directly through OnnxBackend (#172 review).

This is the *anchor* for the mocked unit tests in `tests/unit/test_embedding_onnx.py`:
it runs OnnxBackend against actual ONNX exports, so the mocks' assumed
`get_inputs().type` strings, output ranks, and tokenizer behaviour stay falsifiable
rather than drifting from reality. No server here — just the backend + a real model.

Three model lineages, by design:

- **MiniLM int8** (BERT/WordPiece, 384-dim, has `[PAD]`) for pooling, truncation, and
  the int8 batch-variant → serial determinism lock.
- **DistilRoBERTa int8** (BPE, 768-dim, **no** `[PAD]`) for the RoBERTa-only deltas: its
  own dimensionality, the `<pad>` resolution firing for real, and the same serial lock.
- **MiniLM fp32** (non-quantized) for the opposite case: the probe finds it batch-**safe**,
  so it batches, and batched is bit-exact with serial.

The int8 and fp32 / RoBERTa models do **not** share a vector space, so there's no
cross-model comparison. All need onnxruntime/tokenizers (the `onnx` extra), so the module
is `embedding`-marked and each class carries `requires_onnxruntime`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shrike.embed_batching import BATCH_DRIFT_TOL, max_probe_drift
from shrike.embedding_onnx import OnnxBackend
from tests.integration.conftest import requires_onnxruntime

pytestmark = [pytest.mark.integration, pytest.mark.embedding]

_MINILM_NDIM = 384
_ROBERTA_NDIM = 768


@requires_onnxruntime
class TestOnnxRealMiniLM:
    """Pooling + truncation against the real pinned MiniLM int8 export."""

    def test_poolings_differ(self, onnx_model: Path) -> None:
        text = ["the derivative is the instantaneous rate of change of a function"]

        def vec(pooling: str) -> np.ndarray:
            be = OnnxBackend(model=str(onnx_model), pooling=pooling)
            be.start()
            return np.array(be.embed_texts(text)[0])

        mean, cls, last = vec("mean"), vec("cls"), vec("last")
        assert mean.shape == (_MINILM_NDIM,)
        # The three strategies pool the same token embeddings differently, so a real
        # graph must produce genuinely different vectors (not just exercise the branch).
        assert not np.allclose(mean, cls)
        assert not np.allclose(mean, last)
        assert not np.allclose(cls, last)

    def test_context_size_truncates(self, onnx_model: Path) -> None:
        # Trap 2: both caps are far below the model's positional limit (256/512); we
        # test that a *lower* cap truncates, never that the cap can exceed the model.
        long_text = (
            "the derivative is the instantaneous rate of change of a function "
            "and the integral is the accumulation of a quantity over an interval"
        )  # well over 8 tokens

        be8 = OnnxBackend(model=str(onnx_model), pooling="mean", normalize=False, max_length=8)
        be8.start()
        be64 = OnnxBackend(model=str(onnx_model), pooling="mean", normalize=False, max_length=64)
        be64.start()

        cap8 = np.array(be8.embed_texts([long_text])[0])
        cap64 = np.array(be64.embed_texts([long_text])[0])
        assert cap8.shape == cap64.shape == (_MINILM_NDIM,)
        # Truncating to 8 tokens drops most of the note, so the pooled vector differs.
        assert not np.allclose(cap8, cap64)

    def test_embedding_is_batch_independent(self, onnx_model: Path) -> None:
        # A note's vector is EXACTLY independent of its batch-mates. int8 dynamic
        # quantization computes activation scales over the whole batch tensor, so a
        # *batched* embed would make a note's vector depend on its batch-mates'
        # content — the same note would embed differently in a rebuild (batch of 64)
        # than in an incremental upsert (batch of 1), breaking the index's contract
        # that a reconcile's end state is identical to a full rebuild. The startup
        # probe detects int8 is batch-variant and embeds it serially; this is the lock.
        be = OnnxBackend(model=str(onnx_model))
        be.start()
        assert be._safe_batch == 1  # probe found the int8 model batch-variant → serial
        text = "mitochondria are the powerhouse of the cell"
        alone = np.array(be.embed_texts([text])[0])
        with_others = np.array(
            be.embed_texts([text, "an unrelated and deliberately long sentence about taxes"])[0]
        )
        assert np.array_equal(alone, with_others)

    def test_probe_set_is_sensitive_to_int8_variance(self, onnx_model: Path) -> None:
        # The spiked probe set must trip the int8 batch-variance with comfortable margin, so
        # the model is classified serial — a future bland set that drifted under tol would
        # silently mark it safe and reintroduce the non-determinism. ~10x is the guard.
        be = OnnxBackend(model=str(onnx_model))
        be.start()
        assert max_probe_drift(be._embed_chunk) > 10 * BATCH_DRIFT_TOL


@requires_onnxruntime
class TestOnnxRealDistilRoberta:
    """The RoBERTa-only deltas: 768-dim, no-`[PAD]` fallback, masking."""

    def test_dimension_is_768(self, distilroberta_model: Path) -> None:
        be = OnnxBackend(model=str(distilroberta_model))
        be.start()
        vecs = be.embed_texts(["a sentence about cells", "a sentence about momentum"])
        assert all(len(v) == _ROBERTA_NDIM for v in vecs)
        assert be.embedding_dim() == _ROBERTA_NDIM

    def test_no_pad_token_resolves_and_embeds(self, distilroberta_model: Path) -> None:
        # Precondition: a genuinely no-`[PAD]` tokenizer (the mock only approximates
        # it) — token_to_id("[PAD]") is None and "<pad>" exists, so start() resolves
        # "<pad>". OnnxBackend then starts and embeds without error.
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(distilroberta_model / "tokenizer.json"))
        assert tok.token_to_id("[PAD]") is None
        assert tok.token_to_id("<pad>") is not None

        be = OnnxBackend(model=str(distilroberta_model))
        be.start()
        vecs = be.embed_texts(["short", "a noticeably longer sentence"])
        assert len(vecs) == 2
        assert all(len(v) == _ROBERTA_NDIM for v in vecs)

    def test_embedding_is_batch_independent(self, distilroberta_model: Path) -> None:
        # The same determinism lock as MiniLM, on a second model lineage: the probe
        # finds this int8 model batch-variant and embeds serially, so a note's vector
        # is EXACTLY independent of its batch-mates.
        be = OnnxBackend(model=str(distilroberta_model))
        be.start()
        assert be._safe_batch == 1
        text = "mitochondria are the powerhouse of the cell"
        alone = np.array(be.embed_texts([text])[0])
        with_others = np.array(
            be.embed_texts([text, "an unrelated and deliberately long sentence about taxes"])[0]
        )
        assert np.array_equal(alone, with_others)

    def test_probe_set_is_sensitive_to_int8_variance(self, distilroberta_model: Path) -> None:
        be = OnnxBackend(model=str(distilroberta_model))
        be.start()
        assert max_probe_drift(be._embed_chunk) > 10 * BATCH_DRIFT_TOL


@requires_onnxruntime
class TestOnnxRealFp32:
    """The fp32 (non-quantized) MiniLM: the probe finds it batch-safe, so it batches —
    and batched is bit-exact with serial (no dynamic activation quantization)."""

    def test_probe_finds_batch_safe_and_batched_equals_serial(self, onnx_fp32_model: Path) -> None:
        be = OnnxBackend(model=str(onnx_fp32_model))
        be.start()
        assert be._safe_batch >= 2  # fp32 batches deterministically → probe says safe

        text = "mitochondria are the powerhouse of the cell"
        serial = np.array(be.embed_texts([text])[0])  # chunk of 1
        # Embedded in a real batch (chunk of 2), the same note must be bit-identical.
        batched = np.array(
            be.embed_texts([text, "an unrelated and deliberately long sentence about taxes"])[0]
        )
        assert serial.shape == (_MINILM_NDIM,)
        assert np.array_equal(serial, batched)

    def test_batch_size_cap_chunks(self, onnx_fp32_model: Path) -> None:
        # The cap limits chunk size even on a batch-safe model; results are unchanged.
        capped = OnnxBackend(model=str(onnx_fp32_model), batch_size=2)
        capped.start()
        uncapped = OnnxBackend(model=str(onnx_fp32_model))
        uncapped.start()
        texts = ["one", "two", "three", "four", "five"]
        assert np.array_equal(
            np.array(capped.embed_texts(texts)), np.array(uncapped.embed_texts(texts))
        )
