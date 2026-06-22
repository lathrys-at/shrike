"""llama-server embedding backend + the runtime that owns its lifecycle.

``LlamaServerBackend`` is a thin facade composing the two native pieces:
``shrike_native.LlamaServerManager`` (subprocess lifecycle — spawn,
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
(``llama``/``onnx``) and manages start/stop; the server harness attaches the
started backend to the kernel's embed slot itself.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import shrike_native

from shrike.harness.engines.embedding.base import IMAGE, TEXT, EmbedderBackend
from shrike.harness.engines.embedding.batching import probe_max_safe_batch
from shrike.harness.engines.embedding.text import EMBED_TEXT_VERSION

# Embedding backend kinds the runtime can construct (see EmbeddingRuntime).
# The onnx/clip backends run the native (Rust) engines; "onnx-rs"/"clip-rs"
# remain accepted as aliases so existing configs keep working.
SUPPORTED_BACKENDS = ("llama", "onnx", "clip", "synthetic")
BACKEND_ALIASES = {"onnx-rs": "onnx", "clip-rs": "clip"}
DEFAULT_BACKEND = "llama"

logger = logging.getLogger("shrike.embedding")

DEFAULT_PORT = 8373
DEFAULT_HOST = "127.0.0.1"


class LlamaServerBackend:
    """A llama-server subprocess backend for computing text embeddings.

    Implements the :class:`~shrike.embedding_base.EmbedderBackend` protocol. The
    GGUF/MLX models it serves are text-only, so it advertises ``{TEXT}``.
    """

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
        modalities: frozenset[str] = frozenset({TEXT}),
        mmprojs: Sequence[str] | None = None,
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
        self._modalities = modalities
        # Per-modality projectors loaded with the server for a multimodal omni
        # entry; empty = a text-only embeddings server. Vector-affecting, so
        # they fold into the fingerprint.
        self._mmprojs = list(mmprojs) if mmprojs else []
        # The native lifecycle manager: spawn + health-wait + PID-file orphan
        # reaping + SIGTERM→SIGKILL stop, all crate-side (including the
        # reserved-flag guard on the extra_args passthrough).
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
            mmprojs=self._mmprojs,
        )
        self._pooling = pooling
        # The native HTTP client: one unpinned client for health and model
        # metadata; a model-pinned twin is built once the name is known.
        self._client = shrike_native.RemoteEmbedder(self._base_url)
        self._remote: Any = None

    @property
    def modalities(self) -> frozenset[str]:
        return self._modalities

    @property
    def running(self) -> bool:
        return bool(self._manager.running())

    @property
    def assume_normalized(self) -> bool:
        # llama-server output is not guaranteed unit (model- and pooling-
        # dependent), so the kernel's boundary normalize stays on.
        return False

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

        # An image entry must have actually loaded a vision mmproj — fail fast
        # at boot, not at the first image embed in a sweep. Unlike the probe
        # below (which degrades in place), this re-raises, so it must
        # stop the spawned child first — a degraded boot has no future start()
        # to reap the orphan, and it would hold its port + VRAM for the
        # daemon's life otherwise.
        if IMAGE in self._modalities and not self._remote.vision_capable():
            self._manager.stop()
            self._remote = None
            raise RuntimeError(
                "embedder declares image modality but the managed llama-server did not load "
                "a vision projector — set managed.llama_server.mmprojs to the model's vision "
                "mmproj(s)"
            )

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
        so an empty-at-boot index can still be materialized at the right width.
        Returns ``None`` only if both routes fail.
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
        Left out when ``pooling`` is unset, so an unset-pooling index keeps the
        same fingerprint (the GGUF's own pooling is then in force).

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
        # The mmproj set is vector-affecting: a different projector produces
        # different image vectors, so changing it must rebuild. The
        # text model's `/v1/models` meta says nothing about the projector, so
        # this is the only thing distinguishing two omni configs on the same
        # text model. Folded as name:size (sorted) — size disambiguates two
        # different projectors sharing a basename, matching the `file:`
        # fallback's convention. Omitted when none, so a text-only fingerprint
        # carries no mmproj segment.
        if self._mmprojs:
            parts = []
            for p in self._mmprojs:
                try:
                    size = Path(p).stat().st_size
                except OSError:
                    size = -1
                parts.append(f"{Path(p).name}:{size}")
            base = f"{base}:mmproj={' '.join(sorted(parts))}"
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

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        """Embed images via the native multimodal dialect — direct path for
        callers/tests; the kernel rides ``native_embedder``. Only when the
        managed server loaded a vision projector."""
        if IMAGE not in self._modalities:
            raise RuntimeError("this embedder does not serve images (no image modality)")
        if not self.running or self._remote is None:
            raise RuntimeError("Embedding service is not running")
        vectors: list[list[float]] = self._remote.embed_image_chunk(images)
        return vectors

    def native_embedder(self) -> Any:
        """The kernel-slot handle: the model-pinned remote client composed
        behind the engine contract, so kernel embeds run native end-to-end
        (lane → pool thread → HTTP) and never re-enter this facade — the same
        handover the onnx/clip facades make. The facade keeps lifecycle
        (spawn, health-wait, the probe, orphan reaping), identity assembly,
        and ``health()``. Composes the image half too for a multimodal entry.
        Must be called from a coroutine context (it captures the running loop).
        """
        if not self.running or self._remote is None:
            raise RuntimeError("Embedding service is not running")
        return shrike_native.NativeEmbedder.from_remote(
            self._remote,
            fingerprint=self.model_fingerprint(),
            dim=self.embedding_dim(),
            safe_batch=self._effective_batch(self._safe_batch),
            images=IMAGE in self._modalities,
        )


# Backward-compatible alias so existing imports of ``EmbeddingService`` keep working.
EmbeddingService = LlamaServerBackend


class RemoteBackend:
    """An embeddings endpoint Shrike does not manage: a v2 ``embedders:``
    entry with ``runtime: remote`` and an explicit ``endpoint`` (cloud, tailnet,
    any OpenAI-compatible server), or ``managed.llama_server.manage: attach``
    (an existing local llama-server someone else owns — never spawned, reaped,
    or stopped by Shrike).

    Implements :class:`~shrike.embedding_base.EmbedderBackend`. The lifecycle
    half of :class:`LlamaServerBackend` is exactly what this class doesn't
    have: ``start()`` reads the API key from the entry's ``api_key_env``
    (referenced, never inline), proves connectivity/auth with one embed call,
    and runs the batch-safety probe; ``stop()`` just drops the client.
    Config-only — there is deliberately no flag spelling for this backend
    (structured entries ride ``--config``).

    A ``modalities: [text, image]`` entry against a llama.cpp multimodal
    endpoint also serves images over the native dialect: the same remote
    engine embeds both, so the kernel composition exposes both halves
    and ``start`` fails fast if the endpoint can't actually serve vision.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str | None = None,
        api_key_env: str | None = None,
        batch_size: int | None = None,
        modalities: frozenset[str] = frozenset({TEXT}),
        router_managed: bool = False,
        pooling: str | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._batch_cap = batch_size
        self._modalities = modalities
        # Router-managed: this remote talks to a SHARED llama.cpp router
        # serving many models, so the endpoint's `/v1/models[0]` is NOT this
        # space's model. The pinned `model` is the authoritative identity, and
        # the embedding dim must come from an actual embed of THIS model (the
        # returned vector length), never `model_info().meta.n_embd`. A
        # router-managed remote therefore REQUIRES an explicit model (the
        # routing key), which profiles.py guarantees.
        self._router_managed = router_managed
        # The router-wide pooling: vector-affecting, so it folds into the
        # router fingerprint (`remote:{model}:pool={pooling}`) — a pooling change
        # rebuilds every router space. Only meaningful for a router-managed
        # remote (the router launches with --pooling); None for any other remote
        # (an endpoint/attached server owns its own pooling, rejected upstream).
        self._pooling = pooling
        self._safe_batch = 1
        self._model_name: str | None = None
        self._remote: Any = None

    @property
    def modalities(self) -> frozenset[str]:
        return self._modalities

    @property
    def running(self) -> bool:
        # No process to poll: "running" is "start() validated the endpoint".
        # A later outage surfaces as embed errors (and a failed restart).
        return self._remote is not None

    @property
    def assume_normalized(self) -> bool:
        # A remote service makes no unit-output guarantee — this is the canonical
        # non-normalizing backend the kernel's boundary normalize exists to cover.
        return False

    @property
    def url(self) -> str:
        return self._endpoint

    def start(self) -> None:
        """Validate the endpoint and build the model-pinned client.

        Raises ``RuntimeError`` when the referenced API-key env var is unset,
        or whatever the first embed call surfaces (bad endpoint, bad key) —
        the runtime marks the start failed; nothing degrades silently.
        """
        if self.running:
            return
        started = time.perf_counter()
        api_key = None
        if self._api_key_env:
            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"api_key_env names {self._api_key_env}, which is not set in the "
                    "server's environment (secrets are referenced, never inline)"
                )

        if self._router_managed:
            # The shared router lists MANY models at /v1/models; data[0] is not
            # ours (and our model may not be loaded yet — lazy load). The pinned
            # `model` IS the identity, so skip the unpinned model_info probe and
            # pin directly. profiles.py guarantees a router consumer has a model.
            self._model_name = self._model
        else:
            probe_client = shrike_native.RemoteEmbedder(self._endpoint, api_key=api_key)
            ident, _meta = probe_client.model_info()
            self._model_name = self._model or ident
        remote = shrike_native.RemoteEmbedder(
            self._endpoint, api_key=api_key, model=self._model_name
        )
        # Connectivity/auth proof: one tiny embed. Cloud endpoints have no
        # /health route, so an embed is the one universal liveness signal —
        # and it surfaces auth errors at start instead of first use.
        remote.embed_chunk([" "])
        # An image entry must hit an endpoint that actually serves vision
        # (the native dialect's mmproj). Fail fast at start — the same
        # boundary every capability mismatch hits, not a first-image surprise.
        if IMAGE in self._modalities and not remote.vision_capable():
            raise RuntimeError(
                f"embedder declares image modality but the endpoint at {self._endpoint} "
                "does not serve image embeddings — its model needs a vision mmproj loaded "
                "(managed.llama_server.mmprojs, or an attached server started with --mmproj)"
            )
        self._remote = remote

        try:
            self._safe_batch = probe_max_safe_batch(self._embed_chunk)
        except Exception as e:  # noqa: BLE001 — never fail start on a probe hiccup
            logger.warning("Batch-safety probe failed (%s); embedding serially.", e)
            self._safe_batch = 1
        logger.info(
            "Remote embedding endpoint ready (%s, model %s, %s, %.1fs)",
            self._endpoint,
            self._model_name or "endpoint default",
            "serial" if self._effective_batch(2) == 1 else "batched",
            time.perf_counter() - started,
        )

    def stop(self) -> None:
        """Forget the endpoint client. The remote service is not ours to stop."""
        self._remote = None

    def health(self) -> dict[str, Any]:
        if not self.running:
            return {"available": False}
        return {
            "available": True,
            "url": self._endpoint,
            "model": self._model_name or self._model,
            "batch_safe": self._safe_batch >= 2,
            "batch": "batched" if self._effective_batch(2) >= 2 else "serial",
        }

    def model_info(self) -> dict[str, Any]:
        """``/v1/models`` metadata, as :meth:`LlamaServerBackend.model_info`."""
        if not self.running:
            return {}
        ident, meta_json = self._remote.model_info()
        meta = json.loads(meta_json)
        if ident is None and not meta:
            return {}
        return {"id": ident, "meta": meta}

    def embedding_dim(self) -> int | None:
        if self._router_managed:
            # The shared router's /v1/models[0] is NOT this space's model, so
            # meta.n_embd would be the WRONG dimension whenever two router
            # models differ in width. The pinned `_remote` client embeds THIS
            # model, so the returned vector length is authoritative — probe it,
            # never read the shared metadata.
            try:
                vectors = self.embed_texts([" "])
            except Exception:
                return None
            return len(vectors[0]) if vectors and vectors[0] else None
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
        """The vector-space identity for an unmanaged endpoint.

        The ``meta:`` recipe when the endpoint serves llama-style numeric
        metadata (an attached llama-server); otherwise the *model name* — for
        a cloud endpoint the name IS the identity (``text-embedding-3-small``
        names one embedding space), and there is no file on disk to fall back
        to. The endpoint URL is deliberately excluded: two endpoints serving
        the same model share a vector space. ``textprep`` appended as always.

        A router-managed remote is pinned to ``remote:{model_name}``
        unconditionally: the shared router's ``/v1/models[0]`` lists MANY models
        and is not this space's, so the ``meta:`` recipe would be IDENTICAL
        across every space sharing the router — collapsing their distinct vector
        spaces. The pinned model name is the per-space discriminator, and the
        router-wide pooling (vector-affecting) folds in as ``:pool={pooling}`` so
        a pooling change rebuilds every router space (mirroring the llama
        facade's pooling fold).
        """
        if not self._router_managed:
            meta = self.model_info().get("meta") or {}
            fields = ("n_params", "n_embd", "n_vocab", "n_ctx_train", "size")
            if any(meta.get(f) is not None for f in fields):
                base = "meta:" + ":".join(str(meta.get(f, "")) for f in fields)
                return f"{base}:textprep={EMBED_TEXT_VERSION}"
        base = f"remote:{self._model_name or self._model or 'default'}"
        if self._router_managed and self._pooling:
            base = f"{base}:pool={self._pooling}"
        return f"{base}:textprep={EMBED_TEXT_VERSION}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
        if self._safe_batch <= 1:
            return 1
        limit = min(self._batch_cap, self._safe_batch) if self._batch_cap else self._safe_batch
        return max(1, min(limit, n))

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = self._remote.embed_chunk(texts)
        return vectors

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        """Embed images via the native multimodal dialect — the direct path
        for callers/tests; the kernel rides ``native_embedder``. Available
        only when the entry declares image modality."""
        if IMAGE not in self._modalities:
            raise RuntimeError("this embedder does not serve images (no image modality)")
        if not self.running:
            raise RuntimeError("Embedding service is not running")
        vectors: list[list[float]] = self._remote.embed_image_chunk(images)
        return vectors

    def native_embedder(self) -> Any:
        """The kernel-slot handle — the same composition as the llama facade.
        Composes the image half too when the entry declares image modality;
        the one remote engine serves both."""
        if not self.running:
            raise RuntimeError("Embedding service is not running")
        return shrike_native.NativeEmbedder.from_remote(
            self._remote,
            fingerprint=self.model_fingerprint(),
            dim=self.embedding_dim(),
            safe_batch=self._effective_batch(self._safe_batch),
            images=IMAGE in self._modalities,
        )


class EmbeddingRuntime:
    """Owns the embedding backend lifecycle.

    Backend-agnostic: it selects a backend by *kind* (``llama``/``onnx``), holds
    the parameters needed to (re)start it, and the current backend (or ``None``
    when stopped). A lock serializes start/stop so concurrent requests can't
    spawn two backends. Attaching the started backend to the kernel's embed
    slot is the harness's job (``Harness._attach``).

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
        endpoint: str | None = None,
        api_key_env: str | None = None,
        modalities: frozenset[str] = frozenset({TEXT}),
        mmprojs: Sequence[str] | None = None,
        router_managed: bool = False,
    ) -> None:
        self._backend_kind = BACKEND_ALIASES.get(backend, backend)
        self._endpoint = endpoint
        self._api_key_env = api_key_env
        # Router-managed remote: the `remote` backend talks to a SHARED
        # llama.cpp router (spawned once, owned by the harness), so its
        # fingerprint/dim derive from the pinned model, not the shared endpoint.
        self._router_managed = router_managed
        self._modalities = modalities
        # Per-modality multimodal projectors for a managed omni embeddings
        # server; empty for text-only or the in-process backends.
        self._mmprojs = list(mmprojs) if mmprojs else []
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

    # Backward-compatible alias for the current backend. Returns the active
    # EmbedderBackend or None.
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
        if not self._configured:
            return "not_configured"
        return "stopped"

    @property
    def _configured(self) -> bool:
        # A remote backend is configured by its endpoint alone (the endpoint's
        # default model is a valid choice); the synthetic backend loads no model
        # at all; every other kind needs a model.
        return (
            bool(self._model)
            or (self._backend_kind == "remote" and bool(self._endpoint))
            or self._backend_kind == "synthetic"
        )

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = (
            {"available": False} if self._backend is None else self._backend.health()
        )
        if self._backend is not None and self._backend.running:
            # The space's modalities — what /status reports per space, and what
            # the coverage matrix is computed from.
            info.setdefault("modalities", sorted(self._backend.modalities))
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
        endpoint: str | None = None,
        api_key_env: str | None = None,
    ) -> EmbedderBackend:
        """Start the embedding backend.

        Non-``None`` overrides update the stored params (so a later restart
        reuses them). If a backend is already running, returns it unchanged.
        Raises ``ValueError`` if no model is configured or the backend kind is
        unknown, ``FileNotFoundError`` / ``RuntimeError`` if it won't start, or
        ``ImportError`` if the ONNX optional dependency isn't installed.

        **SSRF defense-in-depth:** ``endpoint`` / ``api_key_env`` are
        config-only — they reach the runtime via the constructor
        (``profiles.py`` → ``plan_to_runtime_params``), never as a start-time
        override. They are rejected here with ``ValueError`` so that even a
        future careless ``POST /embedding/start`` route body that forwards them
        cannot point the embedding traffic at an attacker-chosen endpoint. The
        remote endpoints stay operator-configured; the remote engines now pin
        the configured host's IP and re-vet redirects (cross-host refused), but
        the historical lack of any classifier on those paths is why the
        endpoint must not be settable over HTTP. (Both kwargs stay on the
        signature so the rejection is explicit, not a silent ``**kwargs`` drop.)
        """
        if endpoint is not None or api_key_env is not None:
            raise ValueError(
                "endpoint/api_key_env are config-only and cannot be set via a "
                "start() override (they reach the runtime through the embedders: "
                "config entry, not an HTTP /embedding/start body)"
            )
        with self._lock:
            if self._backend is not None and self._backend.running:
                return self._backend

            if backend is not None:
                # Normalize the same way __init__ does (BACKEND_ALIASES), so a
                # documented alias ("onnx-rs"/"clip-rs") on the start() override
                # path behaves like the ctor — and a bad override can't poison
                # _backend_kind for subsequent no-override starts.
                self._backend_kind = BACKEND_ALIASES.get(backend, backend)
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
            # endpoint/api_key_env are rejected above (config-only) — no
            # override assignment here by design.

            if not self._configured:
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
            return be

    def _make_backend(self) -> EmbedderBackend:
        """Construct (but don't start) the backend for the configured kind.

        The onnx/clip backends import onnxruntime lazily — it's a hard dependency
        of the published wheel, but an environment missing it still surfaces
        a clean ``ImportError`` only when that backend is actually selected.
        """
        if self._backend_kind == "remote":
            # Config-only: an unmanaged endpoint entry (or manage: attach).
            # No flag spells this kind — it arrives via --config.
            assert self._endpoint is not None  # _configured checked by callers
            return RemoteBackend(
                endpoint=self._endpoint,
                model=self._model,
                api_key_env=self._api_key_env,
                batch_size=self._batch_size,
                modalities=self._modalities,
                router_managed=self._router_managed,
                # The router-wide pooling folds into a router space's
                # fingerprint; harmless (and None) for any non-router remote.
                pooling=self._pooling if self._router_managed else None,
            )
        if self._backend_kind == "synthetic":
            # No model, no external deps: a deterministic in-process embedder for
            # benchmarks and fast tests. Gated to non-release builds — the config
            # layer has already refused it where `engine-synthetic` is absent.
            from shrike.harness.engines.embedding.synthetic import SyntheticBackend

            return SyntheticBackend(modalities=self._modalities)
        assert self._model is not None  # callers check before constructing
        if self._backend_kind == "onnx":
            from shrike.harness.engines.embedding.onnx import OnnxBackend

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
            from shrike.harness.engines.embedding.clip import ClipBackend

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
                modalities=self._modalities,
                mmprojs=self._mmprojs,
            )
        raise ValueError(
            f"Unknown embedding backend {self._backend_kind!r} "
            f"(expected one of {', '.join(SUPPORTED_BACKENDS)})"
        )

    def stop(self) -> bool:
        """Stop the backend. Returns False if not running."""
        with self._lock:
            if self._backend is None:
                return False
            self._backend.stop()
            self._backend = None
            self._last_start_failed = False
            return True
