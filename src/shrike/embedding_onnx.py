"""In-process ONNX embedding backend (the Rust ``ort`` engine in shrike_native).

An alternative to the llama-server backend for deployments where a pinned
llama.cpp binary is the wrong fit. It runs entirely in-process — no subprocess,
no port, no health-wait, no orphan-reaping: ``start()`` loads the model into the
native engine (the ``ort`` crate + the Rust ``tokenizers`` crate, #270);
``stop()`` drops it. Tokenization, pooling, and L2 normalization all happen
crate-side. Native-only since the #278 cutover (the Python onnxruntime+numpy
engine retired with it).

The onnxruntime *wheel* (the optional ``shrike[onnx]`` extra) remains the
runtime carrier: the native engine dlopens the wheel's shared library — the
pinned, provider-complete onnxruntime build — so there is exactly one runtime
on disk (#269's linkage decision), and the wheel's Python API is still used for
provider discovery (``get_available_providers``).

Pooling (mean/cls/last) changes a vector's *direction*, so it's vector-affecting
and folded into ``model_fingerprint`` (a change forces an index rebuild, exactly
as ``--embedding-pooling`` does for llama). Normalization only changes a vector's
*magnitude*; USearch's ``cos`` metric is scale-invariant (see ``index.py``), so it
never changes ranking and is deliberately *not* in the fingerprint — the same
reasoning that makes llama's ``--embd-normalize`` moot.

Model layout: ``model`` points either at a directory holding ``model.onnx`` (or
``onnx/model.onnx``) plus ``tokenizer.json``, or directly at a ``.onnx`` file with
``tokenizer.json`` beside it — the standard sentence-transformers ONNX export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shrike.embed_batching import ProbeError, probe_max_safe_batch
from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import TEXT
from shrike.embedding_onnx_common import resolve_execution_providers

logger = logging.getLogger("shrike.embedding")

DEFAULT_MAX_LENGTH = 256
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
# Pooling strategies this backend implements (llama also offers "none", which is
# meaningless for a single per-note vector and is rejected here).
_POOLINGS = frozenset({"mean", "cls", "last"})
# The model inputs this backend supplies (the standard sentence-transformers set). A model
# with a *required* input outside this set — most commonly `position_ids` — can't be driven,
# and we'd rather fail loud at start() (below) than silently break embedding.
_SUPPLIED_INPUTS = frozenset({"input_ids", "attention_mask", "token_type_ids"})


def locate_ort_dylib() -> Path:
    """The onnxruntime shared library inside the installed onnxruntime wheel.

    The native (Rust ``ort``) engine dlopens this exact library — the pinned,
    provider-complete onnxruntime build the wheel ships — so the engine and the
    wheel always run the same onnxruntime (#269's linkage decision; no
    duplicated runtime).
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
    """In-process ONNX text-embedding backend (text-only, native engine).

    Implements the :class:`~shrike.embedding_base.EmbedderBackend` protocol.
    The engine is the Rust ``shrike_native.OnnxTextEmbedder`` (#270): ort +
    the tokenizers crate + ndarray pooling, driven through the onnxruntime
    shared library from the installed wheel.
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
        # The execution providers the engine actually loaded (set at start()).
        self._active_providers: list[str] = []
        # Token truncation length. Plumbed from --embedding-context-size; None
        # means the default (fine for all-MiniLM's 256-token limit).
        self._max_length = max_length or DEFAULT_MAX_LENGTH
        # Optional cap on the chunk size (memory/VRAM); None = batch as large as safe.
        self._batch_cap = batch_size
        # Largest batch size proven safe by the startup probe; 1 == embed serially.
        self._safe_batch = 1
        self._native_engine: Any = None
        self._dim: int | None = None

    @property
    def running(self) -> bool:
        return self._native_engine is not None

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
        """Load the native inference engine (in-process)."""
        if self.running:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "The ONNX embedding backend needs the 'onnx' optional dependency "
                "(the onnxruntime wheel carries the shared runtime the native "
                "engine loads). Install it with: pip install 'shrike[onnx]'"
            ) from e

        onnx_path, tok_path = self._resolve_files()
        # Resolve providers against what onnxruntime actually has: drop any requested
        # provider that isn't available — with a clear warning, rather than onnxruntime's
        # silent CPU fallback — and always keep CPU as the final fallback so an unavailable
        # accelerator degrades instead of hard-erroring.
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
            "Loading ONNX embedding model: %s (providers=%s, pooling=%s)",
            onnx_path,
            ",".join(resolved),
            self._pooling,
        )

        import shrike_native

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
        # What actually loaded (a provider available-but-failed-to-init won't be here).
        unloaded = [
            p for p in resolved if p != "CPUExecutionProvider" and p not in self._active_providers
        ]
        if unloaded:
            logger.warning(
                "ONNX execution provider(s) %s did not load; running on %s.",
                unloaded,
                self._active_providers,
            )
        self._finish_start(onnx_path)

    def _unsupported_inputs(self) -> list[str]:
        """Graph inputs outside the supplied sentence-transformers set (diagnostics)."""
        if self._native_engine is None:
            return []
        return sorted(self._native_engine.unsupported_inputs())

    def _finish_start(self, onnx_path: Path) -> None:
        """The tail of start(): the batch-safety probe + logging.

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
        """Drop the native engine."""
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

        This is the unit the batch-safety probe and ``embed_texts`` build on.
        """
        vectors: list[list[float]] = self._native_engine.embed_chunk(texts)
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def embedding_dim(self) -> int | None:
        """The model's embedding width, or ``None`` if it can't be determined.

        Asks the engine's static output shape, then probes with a tiny embed
        (mirroring the llama backend), caching the result.
        """
        if self._dim is not None:
            return self._dim
        if not self.running:
            return None
        dim = self._native_engine.dim()
        if dim is not None:
            self._dim = int(dim)
            return self._dim
        try:
            vectors = self.embed_texts([" "])
        except Exception:
            return None
        return len(vectors[0]) if vectors and vectors[0] else None

    def model_fingerprint(self) -> str:
        """Stable identity for the loaded model — the index ``model_id``.

        The ``onnx-rs:`` namespace is the native engine's vector-space identity,
        kept verbatim from the dual-engine bake (#270) so indexes built then load
        without a rebuild. It never collides with a llama-server fingerprint
        (``meta:``/``file:``) or the retired Python engine's ``onnx:`` (whose
        vectors were float-noise-different — epic #265 convention 7: a non-bit-
        exact engine change rebuilds once, never silently mixes spaces). Folds in
        pooling (vector-affecting) and the note-text normalization version;
        normalization is omitted on purpose (scale-invariant under the ``cos``
        metric, see module docstring).
        """
        path = Path(self._model)
        # Use the resolved .onnx file's size when we can, else the path's own.
        try:
            onnx_path, _ = self._resolve_files()
            name, size = onnx_path.name, onnx_path.stat().st_size
        except OSError:
            name, size = path.name, -1
        return f"onnx-rs:{name}:{size}:pool={self._pooling}:textprep={EMBED_TEXT_VERSION}"

    def health(self) -> dict[str, Any]:
        """Status dict for ``/status`` (carries at least ``available``)."""
        return {
            "available": self.running,
            "backend": "onnx",
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
