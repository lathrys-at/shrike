"""In-process ONNX embedding backend (onnxruntime + tokenizers).

An alternative to the llama-server backend for deployments where a pinned
llama.cpp binary is the wrong fit. It runs entirely in-process — no subprocess,
no port, no health-wait, no orphan-reaping: ``start()`` loads an
``onnxruntime.InferenceSession`` and a HuggingFace ``tokenizers.Tokenizer``;
``stop()`` drops them. Because of that, none of llama-server's process machinery
lives here.

Pooling (mean/cls/last) and L2 normalization — which llama-server does internally
— are done here in numpy over the model's token embeddings. Pooling changes a
vector's *direction*, so it's vector-affecting and folded into
``model_fingerprint`` (a change forces an index rebuild, exactly as
``--embedding-pooling`` does for llama). Normalization only changes a vector's
*magnitude*; USearch's ``cos`` metric is scale-invariant (see ``index.py``), so it
never changes ranking and is deliberately *not* in the fingerprint — the same
reasoning that makes llama's ``--embd-normalize`` moot.

The heavy deps (``onnxruntime``, ``tokenizers``) ship as the optional
``shrike[onnx]`` extra and are imported lazily, so importing this module is cheap;
``start()`` raises ``ImportError`` with an install hint when they're absent.

Model layout: ``model`` points either at a directory holding ``model.onnx`` (or
``onnx/model.onnx``) plus ``tokenizer.json``, or directly at a ``.onnx`` file with
``tokenizer.json`` beside it — the standard sentence-transformers ONNX export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from shrike.embed_batching import probe_max_safe_batch
from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import TEXT

logger = logging.getLogger("shrike.embedding")

DEFAULT_MAX_LENGTH = 256
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
# Pooling strategies this backend implements (llama also offers "none", which is
# meaningless for a single per-note vector and is rejected here).
_POOLINGS = frozenset({"mean", "cls", "last"})
# onnxruntime declares input types as strings like "tensor(int64)" and does NOT
# auto-cast a fed array, so we match each input's declared integer dtype (some
# mobile/quantized exports use int32 rather than int64).
_ORT_INT_DTYPES = {"tensor(int64)": np.int64, "tensor(int32)": np.int32}


class OnnxBackend:
    """In-process onnxruntime text-embedding backend (text-only).

    Implements the :class:`~shrike.embedding_base.EmbedderBackend` protocol.
    """

    modalities = frozenset({TEXT})

    def __init__(
        self,
        *,
        model: str,
        pooling: str | None = None,
        normalize: bool = True,
        providers: list[str] | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        log_dir: str | Path | None = None,  # accepted for runtime parity; unused
    ) -> None:
        self._model = model
        pool = pooling or "mean"
        if pool not in _POOLINGS:
            raise ValueError(
                f"ONNX backend pooling must be one of {sorted(_POOLINGS)} (got {pool!r}); "
                "'none' is not supported (a note needs a single pooled vector)."
            )
        self._pooling = pool
        self._normalize = normalize
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
        # Token truncation length. Plumbed from --embedding-context-size; None
        # means the default (fine for all-MiniLM's 256-token limit).
        self._max_length = max_length or DEFAULT_MAX_LENGTH
        # Optional cap on the chunk size (memory/VRAM); None = batch as large as safe.
        self._batch_cap = batch_size
        # Largest batch size proven safe by the startup probe; 1 == embed serially.
        self._safe_batch = 1
        self._session: Any = None
        self._tokenizer: Any = None
        self._dim: int | None = None

    @property
    def running(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def _resolve_files(self) -> tuple[Path, Path]:
        """Locate the ``.onnx`` graph and its ``tokenizer.json``."""
        p = Path(self._model)
        if p.is_dir():
            onnx_path = next(
                (p / rel for rel in ("model.onnx", "onnx/model.onnx") if (p / rel).is_file()),
                None,
            )
            if onnx_path is None:
                raise FileNotFoundError(f"No model.onnx found under {p}")
            tok_path = next(
                (
                    p / rel
                    for rel in ("tokenizer.json", "onnx/tokenizer.json")
                    if (p / rel).is_file()
                ),
                None,
            )
            if tok_path is None:
                raise FileNotFoundError(f"No tokenizer.json found under {p}")
            return onnx_path, tok_path
        if p.suffix == ".onnx" and p.is_file():
            tok_path = p.parent / "tokenizer.json"
            if not tok_path.is_file():
                raise FileNotFoundError(f"No tokenizer.json beside {p}")
            return p, tok_path
        raise FileNotFoundError(f"ONNX model path not found: {p}")

    def start(self) -> None:
        """Load the inference session and tokenizer (in-process)."""
        if self.running:
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "The ONNX embedding backend needs the 'onnx' optional dependency. "
                "Install it with: pip install 'shrike[onnx]'"
            ) from e

        onnx_path, tok_path = self._resolve_files()
        logger.info(
            "Loading ONNX embedding model: %s (providers=%s, pooling=%s)",
            onnx_path,
            ",".join(self._providers),
            self._pooling,
        )
        self._session = ort.InferenceSession(str(onnx_path), providers=self._providers)
        tokenizer = Tokenizer.from_file(str(tok_path))
        # Resolve the pad token across tokenizer conventions: BERT/WordPiece names it
        # "[PAD]", RoBERTa/BART BPE uses "<pad>". The *real* pad id matters because
        # some architectures (RoBERTa) derive position ids from which tokens != the
        # pad id, so padding a batch with the wrong id shifts the real tokens' positions
        # and corrupts their embeddings; fall back to id 0 only if neither name exists.
        # (Padding applies whenever a chunk of >1 is embedded — i.e. for a batch-safe
        # model. A real DistilRoBERTa run surfaced the convention; a BERT-tokenizer mock
        # never reaches the "<pad>" branch.)
        pad_id, pad_token = tokenizer.token_to_id("[PAD]"), "[PAD]"
        if pad_id is None:
            pad_id, pad_token = tokenizer.token_to_id("<pad>"), "<pad>"
        if pad_id is None:
            pad_id, pad_token = 0, "[PAD]"
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer = tokenizer

        # Probe batch-safety once: int8 dynamic-quant exports are batch-variant (a
        # note's vector depends on its batch-mates), fp16/fp32 are not. Batch only as
        # large as is provably safe; serial otherwise. A probe failure (after its own
        # retries) falls back to serial rather than failing boot.
        try:
            self._safe_batch = probe_max_safe_batch(self._embed_chunk)
            if self._safe_batch == 1 and self._batch_cap and self._batch_cap > 1:
                logger.warning(
                    "Embedding model is batch-variant; embedding serially (batch size 1) for "
                    "determinism — use a different model/backend combination for batched "
                    "throughput."
                )
            elif self._batch_cap and self._batch_cap > self._safe_batch:
                logger.info(
                    "--embedding-batch-size %d exceeds the probe-verified ceiling %d; "
                    "capping there.",
                    self._batch_cap,
                    self._safe_batch,
                )
        except Exception as e:  # noqa: BLE001 — never fail boot on a probe hiccup
            logger.warning("Batch-safety probe failed (%s); embedding serially.", e)
            self._safe_batch = 1
        logger.info(
            "ONNX embedding model ready (%s, %s)",
            onnx_path.name,
            "serial" if self._safe_batch == 1 else "batched",
        )

    def stop(self) -> None:
        """Drop the session and tokenizer."""
        self._session = None
        self._tokenizer = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors (one per input), chunked by the safe batch size.

        A startup probe decides whether this model batches deterministically. If it does
        (fp16/fp32 ONNX), texts are embedded in chunks as large as the input (capped by
        ``--embedding-batch-size``). If it doesn't (int8 dynamic-quant — a note's vector
        would depend on its batch-mates, breaking the index's reconcile==rebuild
        invariant), each text is embedded **serially** (batch size 1). So a note's vector
        is always a pure function of its text, regardless of how the index batched it.
        """
        if not self.running:
            raise RuntimeError("ONNX embedding backend is not running")
        if not texts:
            return []
        bs = self._effective_batch(len(texts))
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            out.extend(self._embed_chunk(texts[i : i + bs]))
        return out

    def _effective_batch(self, n: int) -> int:
        """Chunk size to embed with: 1 if variant/capped-to-serial, else the smaller of the
        proven-safe batch and the operator's cap, never exceeding what the probe verified."""
        if self._safe_batch <= 1:
            return 1
        limit = min(self._batch_cap, self._safe_batch) if self._batch_cap else self._safe_batch
        return max(1, min(limit, n))

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts as a single batch (one vector per input).

        A 1-element chunk pads nothing; a multi-element chunk pads to the longest and the
        attention mask excludes the padding. This is the unit the batch-safety probe and
        ``embed_texts`` build on.
        """
        encodings = self._tokenizer.encode_batch(texts)
        available = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        attention_mask = available["attention_mask"]
        # Feed only the inputs the graph declares, each cast to that input's declared
        # integer dtype (onnxruntime won't auto-cast).
        feed = {
            inp.name: available[inp.name].astype(_ORT_INT_DTYPES.get(inp.type, np.int64))
            for inp in self._session.get_inputs()
            if inp.name in available
        }
        out = np.asarray(self._session.run(None, feed)[0], dtype=np.float32)
        # A token-level last_hidden_state [batch, seq, hidden] needs pooling; some exports
        # emit an already-pooled sentence embedding [batch, hidden], used directly.
        if out.ndim == 3:
            vectors = self._pool(out, attention_mask)
        elif out.ndim == 2:
            vectors = out
        else:
            raise RuntimeError(
                f"ONNX model first output has rank {out.ndim}; expected 2 (pooled) or 3 (tokens)"
            )
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        self._dim = int(vectors.shape[1])
        result: list[list[float]] = vectors.astype(np.float32).tolist()
        return result

    def _pool(self, token_emb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Reduce token embeddings [B,S,H] to sentence vectors [B,H]."""
        pooled: np.ndarray
        if self._pooling == "cls":
            pooled = token_emb[:, 0, :]
        elif self._pooling == "last":
            # The last non-pad token of each row (mask is 1 for real tokens).
            lengths = mask.sum(axis=1)
            idx = np.clip(lengths - 1, 0, token_emb.shape[1] - 1)
            pooled = token_emb[np.arange(token_emb.shape[0]), idx, :]
        else:
            # mean: average over real tokens only (mask out padding).
            m = mask[:, :, None].astype(np.float32)
            summed = (token_emb * m).sum(axis=1)
            counts = np.clip(m.sum(axis=1), 1e-9, None)
            pooled = summed / counts
        return pooled

    def embedding_dim(self) -> int | None:
        """The model's embedding width, or ``None`` if it can't be determined.

        Tries the session's static output shape, then probes with a tiny embed
        (mirroring the llama backend), caching the result.
        """
        if self._dim is not None:
            return self._dim
        if self._session is not None:
            shape = self._session.get_outputs()[0].shape
            if shape and isinstance(shape[-1], int):
                self._dim = int(shape[-1])
                return self._dim
        if not self.running:
            return None
        try:
            vectors = self.embed_texts([" "])
        except Exception:
            return None
        return len(vectors[0]) if vectors and vectors[0] else None

    def model_fingerprint(self) -> str:
        """Stable identity for the loaded model — the index ``model_id``.

        Prefixed ``onnx:`` so it never collides with a llama-server fingerprint
        (``meta:``/``file:``) for the "same" model under a different runtime.
        Folds in pooling (vector-affecting) and the note-text normalization
        version; normalization is omitted on purpose (scale-invariant under the
        ``cos`` metric, see module docstring).
        """
        path = Path(self._model)
        # Use the resolved .onnx file's size when we can, else the path's own.
        try:
            onnx_path, _ = self._resolve_files()
            name, size = onnx_path.name, onnx_path.stat().st_size
        except OSError:
            name, size = path.name, -1
        return f"onnx:{name}:{size}:pool={self._pooling}:textprep={EMBED_TEXT_VERSION}"

    def health(self) -> dict[str, Any]:
        """Status dict for ``/status`` (carries at least ``available``)."""
        return {
            "available": self.running,
            "backend": "onnx",
            "model": self._model,
            "providers": self._providers,
            # batch_safe is the model's probed capability; batch is the *effective*
            # behaviour, which a --embedding-batch-size cap of 1 can force back to serial.
            "batch_safe": self._safe_batch >= 2,
            "batch": "batched" if self._effective_batch(2) >= 2 else "serial",
        }
