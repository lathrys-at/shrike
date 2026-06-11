"""llama-server embedding backend + the runtime that owns its lifecycle.

``LlamaServerBackend`` is a thin facade (#342 P4) composing the two native
pieces: ``shrike_native.LlamaServerManager`` (subprocess lifecycle — spawn,
health-wait, orphan reaping, escalating stop) and
``shrike_native.RemoteEmbedder`` (the generic OpenAI-compatible embeddings
client). One implementation of the
:class:`~shrike.embedding_base.EmbedderBackend` protocol (``OnnxBackend`` /
``ClipBackend`` are the others). The Shrike server owns the llama-server
process as a direct child; the child is terminated on shutdown.

The backend exposes a simple sync interface:
    be = LlamaServerBackend(model="/path/to/model.gguf", log_dir="/path/to/logs")
    be.start()                 # spawns llama-server, waits for health
    vecs = be.embed_texts(["hello", "world"])  # list[list[float]]
    be.stop()                  # SIGTERM → SIGKILL fallback

``EmbeddingService`` is kept as a backward-compatible alias of
``LlamaServerBackend``. ``EmbeddingRuntime`` selects a backend by *kind*
(``llama``/``onnx``) and manages start/stop plus the binding to the index.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import shrike_native

from shrike.embed_batching import probe_max_safe_batch
from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import TEXT, EmbedderBackend

if TYPE_CHECKING:
    from shrike.index import VectorIndex

# Embedding backend kinds the runtime can construct (see EmbeddingRuntime).
# The onnx/clip backends run the native (Rust) engines, unconditional since the
# #278 cutover; "onnx-rs"/"clip-rs" — the kinds that selected those engines
# during the dual-engine parity bake (#270/#271) — remain accepted as aliases so
# existing configs keep working.
SUPPORTED_BACKENDS = ("llama", "onnx", "clip")
BACKEND_ALIASES = {"onnx-rs": "onnx", "clip-rs": "clip"}
DEFAULT_BACKEND = "llama"

logger = logging.getLogger("shrike.embedding")

DEFAULT_PORT = 8373
DEFAULT_HOST = "127.0.0.1"
HEALTH_TIMEOUT = 30.0  # mirrored by the native manager; kept for messages/tests


class LlamaServerBackend:
    """A llama-server subprocess backend for computing text embeddings.

    Implements the :class:`~shrike.embedding_base.EmbedderBackend` protocol. The
    GGUF/MLX models it serves are text-only, so it advertises ``{TEXT}``.
    """

    # llama-server here serves text-embedding models only.
    modalities = frozenset({TEXT})

    def __init__(
        self,
        *,
        model: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        log_dir: str | Path | None = None,
        context_size: int | None = None,
        threads: int | None = None,
        gpu_layers: int | None = None,
        pooling: str | None = None,
        extra_args: Sequence[str] | None = None,
        llama_server: str | None = None,
        pid_file: str | Path | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._port = port
        # Optional cap on the per-request batch (None = whole input); _safe_batch is
        # set by the startup batch-safety probe (1 == serial). llama is normally safe.
        self._batch_cap = batch_size
        self._safe_batch = 1
        self._base_url = f"http://{host}:{port}"
        self._model_name: str | None = None
        # The native lifecycle manager (#342 P4b): spawn + health-wait +
        # PID-file orphan reaping + SIGTERM→SIGKILL stop, all crate-side
        # (including the reserved-flag guard on the extra_args passthrough).
        self._manager = shrike_native.LlamaServerManager(
            model,
            host=host,
            port=port,
            binary=llama_server,
            log_dir=str(log_dir) if log_dir else None,
            context_size=context_size,
            threads=threads,
            gpu_layers=gpu_layers,
            pooling=pooling,
            extra_args=list(extra_args) if extra_args else [],
            pid_file=str(pid_file) if pid_file else None,
        )
        self._pooling = pooling
        # The native HTTP client (#342 P4a): one unpinned client for health and
        # model metadata; a model-pinned twin is built once the name is known.
        self._client = shrike_native.RemoteEmbedder(self._base_url)
        self._remote: Any = None

    @property
    def running(self) -> bool:
        return bool(self._manager.running())

    @property
    def url(self) -> str:
        return self._base_url

    def start(self) -> None:
        """Start llama-server (native manager: reap → spawn → health-wait)."""
        if self.running:
            logger.warning("Embedding service already running (PID %s)", self._manager.pid())
            return

        started = time.perf_counter()
        self._manager.start()

        # Cache the model's reported name/alias and build the model-pinned
        # embed client (a multi-model endpoint resolves the right one; a
        # single-model llama-server ignores the pin).
        self._model_name = self.model_info().get("id") or Path(self._model).name
        self._remote = shrike_native.RemoteEmbedder(self._base_url, model=self._model_name)

        # Batch-safety probe (universal across backends): confirm a note's vector is
        # independent of its batch-mates before batching requests. llama computes in fp,
        # so it is normally safe; the check guards against a model/config that isn't. The
        # probe retries internally; a persistent failure falls back to serial rather than
        # failing boot — real usage will surface any deeper problem.
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
            "Embedding service ready (PID %s, %s, %.1fs)",
            self._manager.pid(),
            "serial" if self._safe_batch == 1 else "batched",
            time.perf_counter() - started,
        )

    def stop(self) -> None:
        """Stop the llama-server subprocess (SIGTERM → SIGKILL, native)."""
        self._manager.stop()
        self._remote = None

    def health(self) -> dict[str, Any]:
        """Return health status suitable for inclusion in /status responses."""
        if not self.running:
            return {"available": False}

        return {
            "available": self._client.health_ok(),
            "pid": self._manager.pid(),
            "url": self._base_url,
            "model": self._model,
            # batch_safe is the model's probed capability; batch is the *effective*
            # behaviour (a --embedding-batch-size cap of 1 forces serial).
            "batch_safe": self._safe_batch >= 2,
            "batch": "batched" if self._effective_batch(2) >= 2 else "serial",
        }

    def model_info(self) -> dict[str, Any]:
        """Metadata for the loaded model, from llama-server's ``/v1/models``.

        Returns a dict with ``id`` (the model name/alias) and ``meta`` (numeric
        descriptors such as ``n_params``, ``n_embd``, ``n_vocab``, ``size``).
        Returns ``{}`` if the service is down or the endpoint/shape is missing
        (e.g. an older llama.cpp).
        """
        if not self.running:
            return {}
        ident, meta_json = self._client.model_info()
        meta = json.loads(meta_json)
        if ident is None and not meta:
            return {}
        return {"id": ident, "meta": meta}

    def embedding_dim(self) -> int | None:
        """The loaded model's embedding dimension (``n_embd``), or ``None``.

        Read from llama-server's ``/v1/models`` metadata (the same block the
        fingerprint uses). Falls back to probing — a tiny embed call whose vector
        length is the dimension — when the metadata omits it (an older llama.cpp),
        so an empty-at-boot index can still be materialized at the right width
        (#148). Returns ``None`` only if both routes fail.
        """
        meta = self.model_info().get("meta") or {}
        n_embd = meta.get("n_embd")
        if n_embd:
            return int(n_embd)
        try:
            vectors = self.embed_texts([" "])
        except Exception:
            return None
        return len(vectors[0]) if vectors and vectors[0] else None

    def model_fingerprint(self) -> str:
        """A stable identity for the loaded embedding model.

        Built from llama-server's reported metadata (parameter count, embedding
        dim, vocab size, training context, tensor byte size) — fast, and it
        describes the model actually producing vectors. Falls back to the model
        filename plus on-disk size when llama-server doesn't expose metadata.

        The model *name* is deliberately excluded: it's the weakest signal
        (renames would force needless rebuilds; same-name re-quantizations would
        slip through — which the numeric fields catch).

        An explicit pooling type is folded in: it isn't in the model metadata,
        but changing it changes every vector, so it must invalidate the index.
        Left out when ``pooling`` is unset, so an index built before this setting
        existed still matches (the GGUF's own pooling is unchanged).

        The generic ``--embedding-arg`` passthrough is also folded in, under a
        conservative policy: *any* change to it forces a rebuild. Shrike can't
        tell a vector-affecting flag from a perf-only one in an opaque token bag,
        so it trades the occasional needless re-embed for never silently mixing
        vector spaces. (Vector-affecting flags should be typed settings — like
        ``--embedding-pooling`` — not buried in the passthrough.) Reserved flags
        are excluded because they never reach llama-server.

        Finally the note-text normalization version (``EMBED_TEXT_VERSION``) is
        appended unconditionally: the text we feed the model is as much a part of
        the vector space as the model itself, so changing how notes are cleaned
        must invalidate the index. (Unlike pooling/passthrough this is never
        omitted — an index built under the old raw-text scheme *should* rebuild.)
        """
        meta = self.model_info().get("meta") or {}
        fields = ("n_params", "n_embd", "n_vocab", "n_ctx_train", "size")
        if any(meta.get(f) is not None for f in fields):
            base = "meta:" + ":".join(str(meta.get(f, "")) for f in fields)
        else:
            path = Path(self._model)
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            base = f"file:{path.name}:{size}"

        if self._pooling:
            base = f"{base}:pool={self._pooling}"
        passthrough = list(self._manager.passthrough_tokens())
        if passthrough:
            base = f"{base}:args={' '.join(passthrough)}"
        return f"{base}:textprep={EMBED_TEXT_VERSION}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of texts (one vector per input).

        Chunked by the batch size proven safe at startup: when safe (llama computes in
        fp, so normally the whole input is one request — unchanged behaviour), capped by
        ``--embedding-batch-size``; when variant, one text per request. Raises
        RuntimeError if the service is not running.
        """
        if not self.running:
            raise RuntimeError("Embedding service is not running")
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
        """One ``/v1/embeddings`` request via the native client (which pins
        the model and orders vectors by the response's own ``index``)."""
        client = self._remote if self._remote is not None else self._client
        vectors: list[list[float]] = client.embed_chunk(texts)
        return vectors

    def native_embedder(self) -> Any:
        """The kernel-slot handle (#342 P4): the model-pinned remote client
        composed behind the engine contract, so kernel embeds run native
        end-to-end (lane → pool thread → HTTP) and never re-enter this
        facade — the same handover the onnx/clip facades make. The facade
        keeps lifecycle (spawn, health-wait, the probe, orphan reaping),
        identity assembly, and ``health()``. Must be called from a coroutine
        context (it captures the running loop).
        """
        if not self.running or self._remote is None:
            raise RuntimeError("Embedding service is not running")
        return shrike_native.NativeEmbedder.from_remote(
            self._remote,
            fingerprint=self.model_fingerprint(),
            dim=self.embedding_dim(),
            safe_batch=self._effective_batch(self._safe_batch),
        )


# Backward-compatible alias: the llama-server backend was the original (and only)
# embedding service. Existing imports of ``EmbeddingService`` keep working.
EmbeddingService = LlamaServerBackend


class EmbeddingRuntime:
    """Owns the embedding backend lifecycle and its binding to the vector index.

    Backend-agnostic: it selects a backend by *kind* (``llama``/``onnx``), holds
    the parameters needed to (re)start it, the current backend (or ``None`` when
    stopped), and a reference to the index it attaches/detaches. A lock serializes
    start/stop so concurrent requests can't spawn two backends.

    Both backends share most params (model, pooling); the rest are backend-scoped
    and simply ignored by the one they don't apply to (``host``/``port``/
    ``gpu_layers``/``extra_args``/``llama_server`` are llama-only; ``providers``/
    ``normalize`` are ONNX-only). ``_make_backend`` builds the right one.

    Rebuild orchestration is intentionally *not* here — that needs the collection
    wrapper and lives in the server's request/boot path.
    """

    def __init__(
        self,
        *,
        index: VectorIndex | None = None,
        backend: str = DEFAULT_BACKEND,
        model: str | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        log_dir: str | Path | None = None,
        context_size: int | None = None,
        threads: int | None = None,
        gpu_layers: int | None = None,
        pooling: str | None = None,
        extra_args: Sequence[str] | None = None,
        llama_server: str | None = None,
        pid_file: str | Path | None = None,
        onnx_providers: Sequence[str] | None = None,
        normalize: bool = True,
        batch_size: int | None = None,
    ) -> None:
        self._index = index
        self._backend_kind = BACKEND_ALIASES.get(backend, backend)
        self._model = model
        self._host = host
        self._port = port
        self._log_dir = Path(log_dir) if log_dir else None
        self._context_size = context_size
        self._threads = threads
        self._gpu_layers = gpu_layers
        self._pooling = pooling
        self._extra_args = list(extra_args) if extra_args else []
        self._llama_server = llama_server
        self._pid_file = Path(pid_file) if pid_file else None
        self._onnx_providers = list(onnx_providers) if onnx_providers else None
        self._normalize = normalize
        self._batch_size = batch_size
        self._backend: EmbedderBackend | None = None
        self._lock = threading.Lock()
        # Tracks why the backend isn't running, so status can distinguish a
        # deliberate stop from a failed start or a missing model.
        self._last_start_failed = False

    @property
    def backend(self) -> EmbedderBackend | None:
        return self._backend

    @property
    def backend_kind(self) -> str:
        return self._backend_kind

    # Backward-compatible alias for the current backend (was ``service`` when the
    # only backend was llama-server). Returns the active EmbedderBackend or None.
    @property
    def service(self) -> EmbedderBackend | None:
        return self._backend

    @property
    def running(self) -> bool:
        return self._backend is not None and self._backend.running

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def state(self) -> str:
        """One of ``running``/``failed``/``not_configured``/``stopped``."""
        if self.running:
            return "running"
        if self._last_start_failed:
            return "failed"
        if not self._model:
            return "not_configured"
        return "stopped"

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = (
            {"available": False} if self._backend is None else self._backend.health()
        )
        info["state"] = self.state
        return info

    def start(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        port: int | None = None,
        context_size: int | None = None,
        threads: int | None = None,
        gpu_layers: int | None = None,
        pooling: str | None = None,
        extra_args: Sequence[str] | None = None,
        llama_server: str | None = None,
        onnx_providers: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> EmbedderBackend:
        """Start the embedding backend and attach it to the index.

        Non-``None`` overrides update the stored params (so a later restart
        reuses them). If a backend is already running, returns it unchanged.
        Raises ``ValueError`` if no model is configured or the backend kind is
        unknown, ``FileNotFoundError`` / ``RuntimeError`` if it won't start, or
        ``ImportError`` if the ONNX optional dependency isn't installed.
        """
        with self._lock:
            if self._backend is not None and self._backend.running:
                return self._backend

            if backend is not None:
                self._backend_kind = backend
            if model is not None:
                self._model = model
            if port is not None:
                self._port = port
            if context_size is not None:
                self._context_size = context_size
            if threads is not None:
                self._threads = threads
            if gpu_layers is not None:
                self._gpu_layers = gpu_layers
            if pooling is not None:
                self._pooling = pooling
            if extra_args is not None:
                self._extra_args = list(extra_args)
            if llama_server is not None:
                self._llama_server = llama_server
            if onnx_providers is not None:
                self._onnx_providers = list(onnx_providers) or None
            if batch_size is not None:
                self._batch_size = batch_size

            if not self._model:
                raise ValueError("No embedding model configured")

            try:
                # Construction inside the try too, so a bad backend kind or an
                # OnnxBackend pooling/ImportError marks the runtime failed (state
                # reports "failed", not "stopped").
                be = self._make_backend()
                be.start()
            except Exception:
                self._last_start_failed = True
                raise
            self._last_start_failed = False
            self._backend = be
            # Legacy facade coupling; the kernel-mode server attaches the
            # backend itself (PyEmbedder.capture must run on the loop).
            if self._index is not None:
                self._index.set_backend(be)
            return be

    def _make_backend(self) -> EmbedderBackend:
        """Construct (but don't start) the backend for the configured kind.

        The onnx/clip backends import their heavy deps lazily, so ``shrike[onnx]`` /
        ``shrike[clip]`` stay optional — a missing dependency surfaces as ``ImportError``
        only when that backend is actually selected.
        """
        assert self._model is not None  # callers check before constructing
        if self._backend_kind == "onnx":
            from shrike.embedding_onnx import OnnxBackend

            return OnnxBackend(
                model=self._model,
                pooling=self._pooling,
                normalize=self._normalize,
                providers=self._onnx_providers,
                # --embedding-context-size doubles as ONNX's token-truncation
                # length (None → the backend's 256 default).
                max_length=self._context_size,
                batch_size=self._batch_size,
                log_dir=self._log_dir,
            )
        if self._backend_kind == "clip":
            from shrike.embedding_clip import ClipBackend

            return ClipBackend(
                model=self._model,
                providers=self._onnx_providers,
                batch_size=self._batch_size,
                log_dir=self._log_dir,
            )
        if self._backend_kind == "llama":
            return LlamaServerBackend(
                model=self._model,
                host=self._host,
                port=self._port,
                log_dir=self._log_dir,
                context_size=self._context_size,
                threads=self._threads,
                gpu_layers=self._gpu_layers,
                pooling=self._pooling,
                extra_args=self._extra_args,
                llama_server=self._llama_server,
                pid_file=self._pid_file,
                batch_size=self._batch_size,
            )
        raise ValueError(
            f"Unknown embedding backend {self._backend_kind!r} "
            f"(expected one of {', '.join(SUPPORTED_BACKENDS)})"
        )

    def stop(self) -> bool:
        """Detach from the index and stop the backend. Returns False if not running."""
        with self._lock:
            if self._backend is None:
                return False
            if self._index is not None:
                self._index.set_backend(None)
            self._backend.stop()
            self._backend = None
            self._last_start_failed = False
            return True
