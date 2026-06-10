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

from shrike.embed_batching import ProbeError, probe_max_safe_batch
from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import TEXT
from shrike.embedding_onnx_common import ORT_INT_DTYPES, resolve_execution_providers

logger = logging.getLogger("shrike.embedding")

DEFAULT_MAX_LENGTH = 256
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
# Pooling strategies this backend implements (llama also offers "none", which is
# meaningless for a single per-note vector and is rejected here).
_POOLINGS = frozenset({"mean", "cls", "last"})
# The model inputs this backend supplies (the standard sentence-transformers set). A model
# with a *required* input outside this set — most commonly `position_ids` — can't be driven,
# and we'd rather fail loud at start() (below) than silently break embedding. Single source
# of truth for both the feed in `_embed_chunk` and the start()-time diagnostic.
_SUPPLIED_INPUTS = frozenset({"input_ids", "attention_mask", "token_type_ids"})
# Match each input's declared integer dtype (some quantized exports use int32). Shared with the
# CLIP backend; aliased so the feed in `_embed_chunk` reads unchanged.
_ORT_INT_DTYPES = ORT_INT_DTYPES


def locate_ort_dylib() -> Path:
    """The onnxruntime shared library inside the installed onnxruntime wheel.

    The native (Rust ``ort``) engine dlopens this exact library — the pinned
    runtime the Python backend already uses — so the two engines always run the
    same onnxruntime build (#269's linkage decision; no duplicated runtime).
    """
    import onnxruntime

    capi = Path(onnxruntime.__file__).parent / "capi"
    for p in sorted(capi.iterdir()):
        name = p.name
        if "providers" in name:
            continue
        if name.startswith("libonnxruntime.") and (".so" in name or name.endswith(".dylib")):
            return p
        if name == "onnxruntime.dll":
            return p
    raise FileNotFoundError(f"no onnxruntime shared library found under {capi}")


class OnnxBackend:
    """In-process onnxruntime text-embedding backend (text-only).

    Implements the :class:`~shrike.embedding_base.EmbedderBackend` protocol.

    Two engines under one facade: the default Python engine (onnxruntime-python
    + tokenizers + numpy pooling) and, with ``native=True`` (the ``onnx-rs``
    backend kind, #270), the Rust engine in ``shrike_native`` (ort + the same
    tokenizers crate + ndarray pooling) — same model layout, same provider
    resolution, same batch-safety probe, driven through the same onnxruntime
    shared library. Measured parity: the native engine's vectors differ from
    the Python engine's only at float-noise level (~3e-08, pooling summation
    order), which by epic #265 convention 7 is **not** bit-exact — so the
    native fingerprint is namespaced ``onnx-rs:`` and switching kinds rebuilds
    the index once rather than ever silently mixing the two spaces.
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
        native: bool = False,
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
        # The execution providers onnxruntime actually loaded (set at start()).
        self._active_providers: list[str] = []
        # Token truncation length. Plumbed from --embedding-context-size; None
        # means the default (fine for all-MiniLM's 256-token limit).
        self._max_length = max_length or DEFAULT_MAX_LENGTH
        # Optional cap on the chunk size (memory/VRAM); None = batch as large as safe.
        self._batch_cap = batch_size
        # Largest batch size proven safe by the startup probe; 1 == embed serially.
        self._safe_batch = 1
        # Engine selection (#270): False = onnxruntime-python, True = shrike_native.
        self._native = native
        self._native_engine: Any = None
        self._session: Any = None
        self._tokenizer: Any = None
        self._dim: int | None = None

    @property
    def running(self) -> bool:
        if self._native:
            return self._native_engine is not None
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
        """Load the inference engine and tokenizer (in-process)."""
        if self.running:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "The ONNX embedding backend needs the 'onnx' optional dependency. "
                "Install it with: pip install 'shrike[onnx]'"
            ) from e

        onnx_path, tok_path = self._resolve_files()
        # Resolve providers against what onnxruntime actually has: drop any requested
        # provider that isn't available — with a clear warning, rather than onnxruntime's
        # silent CPU fallback — and always keep CPU as the final fallback so an unavailable
        # accelerator degrades instead of hard-erroring. The same resolution governs both
        # engines (the native one receives the already-resolved list).
        available = list(ort.get_available_providers())
        resolved, dropped = resolve_execution_providers(available, self._providers)
        if dropped:
            logger.warning(
                "Requested ONNX execution provider(s) %s not available (have %s); using %s.",
                dropped,
                available,
                resolved,
            )
        logger.info(
            "Loading ONNX embedding model: %s (providers=%s, pooling=%s, engine=%s)",
            onnx_path,
            ",".join(resolved),
            self._pooling,
            "native" if self._native else "python",
        )

        if self._native:
            self._start_native(onnx_path, tok_path, resolved)
            self._finish_start(onnx_path)
            return

        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "The ONNX embedding backend needs the 'onnx' optional dependency. "
                "Install it with: pip install 'shrike[onnx]'"
            ) from e
        self._session = ort.InferenceSession(str(onnx_path), providers=resolved)
        # What actually loaded (a provider available-but-failed-to-init won't be here).
        self._active_providers = list(self._session.get_providers())
        unloaded = [
            p for p in resolved if p != "CPUExecutionProvider" and p not in self._active_providers
        ]
        if unloaded:
            logger.warning(
                "ONNX execution provider(s) %s did not load; running on %s.",
                unloaded,
                self._active_providers,
            )
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
        self._finish_start(onnx_path)

    def _start_native(self, onnx_path: Path, tok_path: Path, resolved: list[str]) -> None:
        """Bring up the Rust engine (#270): same model, same resolved providers,
        the same onnxruntime shared library (dlopened from the installed wheel),
        with tokenization/pooling/normalization done crate-side."""
        try:
            import shrike_native
        except ImportError as e:
            raise ImportError(
                "The onnx-rs embedding backend needs the shrike-native extension. "
                "Install the shrike-native wheel, or build it from a checkout with "
                "scripts/build-native.sh."
            ) from e

        shrike_native.init_onnx_runtime(str(locate_ort_dylib()))
        self._native_engine = shrike_native.OnnxTextEmbedder(
            str(onnx_path),
            str(tok_path),
            providers=resolved,
            pooling=self._pooling,
            normalize=self._normalize,
            max_length=self._max_length,
        )
        self._active_providers = list(self._native_engine.active_providers())

    def _unsupported_inputs(self) -> list[str]:
        """Graph inputs outside the supplied sentence-transformers set (diagnostics)."""
        if self._native:
            if self._native_engine is None:
                return []
            return sorted(self._native_engine.unsupported_inputs())
        if self._session is None:
            return []
        return sorted({i.name for i in self._session.get_inputs()} - _SUPPLIED_INPUTS)

    def _finish_start(self, onnx_path: Path) -> None:
        """The engine-agnostic tail of start(): the batch-safety probe + logging.

        Probe batch-safety once: int8 dynamic-quant exports are batch-variant (a
        note's vector depends on its batch-mates), fp16/fp32 are not. Batch only as
        large as is provably safe; serial otherwise. Unlike llama's (transient, HTTP)
        probe, an ONNX serial-embed failure is deterministic — serial won't help, so we
        fail start() loud rather than masquerade as serial. The most common cause is a
        model with a required input we don't supply (e.g. position_ids), named below.
        """
        try:
            self._safe_batch = probe_max_safe_batch(self._embed_chunk)
        except ProbeError as e:
            unsupported = self._unsupported_inputs()
            if unsupported:
                raise RuntimeError(
                    f"ONNX model requires input(s) {unsupported} this backend does not supply "
                    f"(it provides {sorted(_SUPPLIED_INPUTS)}); use a standard "
                    f"sentence-transformers ONNX export."
                ) from e
            raise RuntimeError(f"ONNX embedding model could not be driven: {e}") from e
        if self._safe_batch == 1 and self._batch_cap and self._batch_cap > 1:
            logger.warning(
                "Embedding model is batch-variant; embedding serially (batch size 1) for "
                "determinism — use a different model/backend combination for batched throughput."
            )
        elif self._batch_cap and self._batch_cap > self._safe_batch:
            logger.info(
                "--embedding-batch-size %d exceeds the probe-verified ceiling %d; capping there.",
                self._batch_cap,
                self._safe_batch,
            )
        logger.info(
            "ONNX embedding model ready (%s, %s)",
            onnx_path.name,
            "serial" if self._safe_batch == 1 else "batched",
        )

    def stop(self) -> None:
        """Drop the engine (session/tokenizer, or the native embedder)."""
        self._session = None
        self._tokenizer = None
        self._native_engine = None

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
        if self._native:
            vectors_native: list[list[float]] = self._native_engine.embed_chunk(texts)
            if vectors_native:
                self._dim = len(vectors_native[0])
            return vectors_native
        encodings = self._tokenizer.encode_batch(texts)
        available = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        attention_mask = available["attention_mask"]
        # Feed only the inputs we supply *and* the graph declares, each cast to that input's
        # declared integer dtype (onnxruntime won't auto-cast). A required input outside
        # `_SUPPLIED_INPUTS` is detected at start(); here it's simply absent from the feed.
        feed = {
            inp.name: available[inp.name].astype(_ORT_INT_DTYPES.get(inp.type, np.int64))
            for inp in self._session.get_inputs()
            if inp.name in _SUPPLIED_INPUTS
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
        if self._native and self._native_engine is not None:
            dim = self._native_engine.dim()
            if dim is not None:
                self._dim = int(dim)
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
        # The native engine's vectors are float-noise-different from the Python
        # engine's (pooling summation order) — not bit-exact, so it gets its own
        # namespace (epic #265 convention 7): switching engines rebuilds once,
        # never mixes spaces.
        family = "onnx-rs" if self._native else "onnx"
        return f"{family}:{name}:{size}:pool={self._pooling}:textprep={EMBED_TEXT_VERSION}"

    def health(self) -> dict[str, Any]:
        """Status dict for ``/status`` (carries at least ``available``)."""
        return {
            "available": self.running,
            "backend": "onnx-rs" if self._native else "onnx",
            "model": self._model,
            "requested_providers": self._providers,
            "active_providers": self._active_providers,
            # The effective execution provider (the highest-priority one that loaded), so
            # status can show "running on CPU" when a requested accelerator silently fell back.
            "provider": self._active_providers[0] if self._active_providers else None,
            # batch_safe is the model's probed capability; batch is the *effective*
            # behaviour, which a --embedding-batch-size cap of 1 can force back to serial.
            "batch_safe": self._safe_batch >= 2,
            "batch": "batched" if self._effective_batch(2) >= 2 else "serial",
        }
