"""In-process CLIP backend — image and text into one shared vector space (native engine).

A CLIP export is a **dual encoder**: a text graph (``text_model.onnx``: ``input_ids`` →
``text_embeds``) and a vision graph (``vision_model.onnx``: ``pixel_values`` → ``image_embeds``),
both projecting into the *same* space (cosine-comparable after L2 normalization). So a text query
can retrieve a card by the content of its image — the capability the Phase-3a eval (#193) proved.
This backend advertises ``modalities = {text, image}``; text-only collections keep using the
small text backends (the ``modalities`` seam).

The engine is the Rust ``shrike_native.ClipEmbedder`` (#271), native-only since the #278 cutover
(the Python onnxruntime + PIL engine retired with it): both graphs, tokenization, and the image
preprocessing pipeline (resize → center-crop → rescale → normalize, read from the model's
``preprocessor_config.json``) run crate-side, dlopening the onnxruntime shared library from the
installed wheel — the same single-runtime linkage as the text backend. Unlike the text-only
``OnnxBackend`` there's no pooling: both graphs emit a pre-pooled, projected vector. Provider
resolution and the batch-safety probe are shared with the text-backend work (#175/#176): the text
and vision graphs share the model's quantization, so one probe on the text path governs both (an
int8 CLIP is batch-variant → serial; an fp CLIP batches).
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any

from shrike.embed_batching import probe_image_max_safe_batch, probe_max_safe_batch
from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import IMAGE, TEXT
from shrike.embedding_onnx_common import resolve_execution_providers

logger = logging.getLogger("shrike.embedding")

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
# CLIP text models use a fixed 77-token context (positional embeddings); pad/truncate to it.
CLIP_CONTEXT = 77
# CLIP exports ship a graph at several precisions; auto-discovery prefers full precision (best
# quality, and it batches) and falls back to a quantized one, picking the first precision for
# which *both* graphs exist (so text and vision stay the same precision — no mixed-precision pair).
# Explicit operator selection of a precision is tracked in #210.
_VARIANT_SUFFIXES = ("", "_fp16", "_quantized", "_int8", "_uint8", "_q4", "_bnb4")

# An image accepted by embed_images: a PIL image, a filesystem path, or raw bytes.
ImageInput = Any


def _read_edge(value: Any, keys: tuple[str, ...], default: int) -> int:
    """Read an image dimension from a preprocessor_config field that may be a dict
    (``{"shortest_edge": 224}`` / ``{"height": 224, "width": 224}``) or a bare scalar (``224``).
    """
    if isinstance(value, dict):
        for k in keys:
            if value.get(k):
                return int(value[k])
        return default
    if isinstance(value, (int, float)):
        return int(value)
    return default


class ClipBackend:
    """In-process CLIP backend (text + image, native engine).

    Implements the :class:`~shrike.embedding_base.EmbedderBackend` protocol, plus an
    ``embed_images`` method available because ``IMAGE`` is in ``modalities``.
    """

    modalities = frozenset({TEXT, IMAGE})

    def __init__(
        self,
        *,
        model: str,
        providers: list[str] | None = None,
        batch_size: int | None = None,
        log_dir: str | Path | None = None,  # accepted for runtime parity; unused
    ) -> None:
        self._model = model
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
        self._active_providers: list[str] = []
        self._batch_cap = batch_size
        self._safe_batch = 1
        self._native_engine: Any = None
        self._mean: list[float] | None = None
        self._std: list[float] | None = None
        self._resize = 224  # shortest-edge resize target (preprocessor "size")
        self._crop = 224  # center-crop size (preprocessor "crop_size")
        self._dim: int | None = None
        self._text_path: Path | None = None
        self._vis_path: Path | None = None

    @property
    def running(self) -> bool:
        return self._native_engine is not None

    def _resolve_files(self) -> tuple[Path, Path, Path, Path]:
        """Locate the text graph, vision graph, tokenizer.json, preprocessor_config.json.

        The graphs are auto-discovered across precisions (``_VARIANT_SUFFIXES``): the first
        precision for which *both* ``text_model`` and ``vision_model`` exist wins, so a full export
        loads its full-precision pair and a quantized-only export (the CI fixture) loads that —
        without a mixed-precision pair (which would break the one-probe-governs-both assumption).
        """
        root = Path(self._model)
        if not root.is_dir():
            raise FileNotFoundError(f"CLIP model path is not a directory: {root}")

        def _find(name: str) -> Path | None:
            return next((root / r for r in (f"onnx/{name}", name) if (root / r).is_file()), None)

        text_path = vis_path = None
        for suffix in _VARIANT_SUFFIXES:
            text_path = _find(f"text_model{suffix}.onnx")
            vis_path = _find(f"vision_model{suffix}.onnx")
            if text_path and vis_path:
                if suffix:
                    logger.info("CLIP: loading the '%s' graph variant", suffix.lstrip("_"))
                break
        if not (text_path and vis_path):
            raise FileNotFoundError(
                f"No matching text_model*.onnx + vision_model*.onnx under {root}"
            )

        tok = _find("tokenizer.json")
        if tok is None:
            raise FileNotFoundError(f"No tokenizer.json under {root}")
        pp = _find("preprocessor_config.json")
        if pp is None:
            raise FileNotFoundError(f"No preprocessor_config.json under {root}")
        return text_path, vis_path, tok, pp

    def start(self) -> None:
        """Load the native engine: both graphs, tokenizer, and image preprocessing."""
        if self.running:
            return
        started = time.perf_counter()
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "The CLIP embedding backend needs the onnxruntime wheel (it carries "
                "the shared runtime the native engine loads; a hard dependency of "
                "shrike-mcp). Install it with: pip install onnxruntime"
            ) from e

        self._text_path, self._vis_path, tok_path, pp_path = self._resolve_files()

        available = list(ort.get_available_providers())
        resolved, dropped = resolve_execution_providers(available, self._providers)
        if dropped:
            logger.warning(
                "Requested ONNX execution provider(s) %s not available (have %s); using %s.",
                dropped,
                available,
                resolved,
            )

        # The preprocessor config is parsed here: only scalars cross the FFI.
        pp = json.loads(pp_path.read_text())
        self._mean = [float(v) for v in pp["image_mean"]]
        self._std = [float(v) for v in pp["image_std"]]
        # CLIP preprocessing has two independent knobs: resize the shortest edge to `size`, then
        # center-crop to `crop_size`. Each may be a dict or a bare scalar across exports.
        self._resize = _read_edge(pp.get("size"), ("shortest_edge", "height", "width"), 224)
        self._crop = _read_edge(pp.get("crop_size"), ("height", "width"), self._resize)

        import shrike_native

        from shrike.embedding_onnx import locate_ort_dylib

        shrike_native.init_onnx_runtime(str(locate_ort_dylib()))
        self._native_engine = shrike_native.ClipEmbedder(
            str(self._text_path),
            str(self._vis_path),
            str(tok_path),
            providers=resolved,
            image_mean=self._mean,
            image_std=self._std,
            resize=self._resize,
            crop=self._crop,
            context=CLIP_CONTEXT,
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
        self._finish_start(started)

    def _finish_start(self, started: float) -> None:
        """The tail of start(): the batch-safety probe + logging."""
        # _resolve_files auto-discovers both graphs at the *same* precision, so a uniform
        # export's text probe already predicts the vision path. We still probe BOTH and take
        # the min (#211) to harden against a hand-assembled mixed-precision pair (fp text +
        # int8 vision a user dropped on disk), where the vision graph batches
        # non-deterministically and the text probe alone would wrongly clear it — breaking the
        # reconcile==rebuild invariant for image vectors.
        try:
            text_safe = probe_max_safe_batch(self._embed_text_chunk)
            vision_safe = probe_image_max_safe_batch(self._embed_image_bytes_chunk)
            self._safe_batch = min(text_safe, vision_safe)
            if self._safe_batch == 1 and self._batch_cap and self._batch_cap > 1:
                culprit = "vision graph" if vision_safe < text_safe else "model"
                logger.warning(
                    "CLIP %s is batch-variant; embedding serially (batch size 1) for "
                    "determinism — use a uniform fp CLIP export for batched throughput.",
                    culprit,
                )
        except Exception as e:  # noqa: BLE001 — never fail boot on a probe hiccup
            logger.warning("Batch-safety probe failed (%s); embedding serially.", e)
            self._safe_batch = 1

        assert self._text_path is not None and self._vis_path is not None
        logger.info(
            "CLIP model ready (%s + %s, %s, %.1fs)",
            self._text_path.name,
            self._vis_path.name,
            "serial" if self._safe_batch == 1 else "batched",
            time.perf_counter() - started,
        )

    def stop(self) -> None:
        self._native_engine = None

    def _effective_batch(self, n: int) -> int:
        if self._safe_batch <= 1:
            return 1
        limit = min(self._batch_cap, self._safe_batch) if self._batch_cap else self._safe_batch
        return max(1, min(limit, n))

    # -- text ----------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed query/note text into the shared space (chunked by the safe batch size)."""
        if not self.running:
            raise RuntimeError("CLIP embedding backend is not running")
        if not texts:
            return []
        bs = self._effective_batch(len(texts))
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            out.extend(self._embed_text_chunk(texts[i : i + bs]))
        return out

    def _embed_text_chunk(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = self._native_engine.embed_text_chunk(texts)
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    # -- image ---------------------------------------------------------------

    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        """Embed images (PIL / path / bytes) into the shared space (chunked by the safe batch)."""
        if not self.running:
            raise RuntimeError("CLIP embedding backend is not running")
        if not images:
            return []
        bs = self._effective_batch(len(images))
        out: list[list[float]] = []
        for i in range(0, len(images), bs):
            out.extend(self._embed_image_chunk(images[i : i + bs]))
        return out

    def _embed_image_chunk(self, images: list[ImageInput]) -> list[list[float]]:
        # Marshaling rule: only bytes cross the FFI — the engine decodes and
        # preprocesses crate-side.
        return self._embed_image_bytes_chunk([self._image_bytes(im) for im in images])

    def _embed_image_bytes_chunk(self, images: list[bytes]) -> list[list[float]]:
        """Embed a chunk of already-encoded image bytes (the vision-probe path)."""
        vectors: list[list[float]] = self._native_engine.embed_image_chunk(images)
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def _image_bytes(self, image: ImageInput) -> bytes:
        """Encoded bytes for the native engine (which decodes crate-side)."""
        if isinstance(image, bytes):
            return image
        if isinstance(image, (str, Path)):
            return Path(image).read_bytes()
        # A PIL image (test convenience) — round-tripped through lossless PNG.
        # Duck-typed so this module never imports PIL itself.
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")  # lossless — pixels unchanged
        return buf.getvalue()

    # -- shared --------------------------------------------------------------

    def embedding_dim(self) -> int | None:
        if self._dim is not None:
            return self._dim
        if not self.running:
            return None
        dim = self._native_engine.dim()
        if dim is not None:
            self._dim = int(dim)
            return self._dim
        return len(self.embed_texts([" "])[0])

    def model_fingerprint(self) -> str:
        """Stable identity for the index ``model_id`` — both graphs + image-prep + text-prep.

        The ``clip-rs:`` namespace is the native engine's vector-space identity, kept
        verbatim from the dual-engine bake (#271) so indexes built then load without a
        rebuild; it never collides with a text-only ``onnx-rs:``/``meta:`` fingerprint
        (or the retired Python engine's ``clip:``, whose PIL-bicubic image vectors were
        pixel-different — epic #265 convention 7). A change to either graph (or the
        crate-side preprocessing, via its version counter) forces a re-embed.
        """

        def _sz(p: Path | None) -> int:
            try:
                return p.stat().st_size if p else -1
            except OSError:
                return -1

        import shrike_native

        t, v = self._text_path, self._vis_path
        tn = t.name if t else "text_model"
        vn = v.name if v else "vision_model"
        prep = f"imgprep=rs{shrike_native.IMAGE_PREP_VERSION_RS}:textprep={EMBED_TEXT_VERSION}"
        return f"clip-rs:{tn}:{_sz(t)}:{vn}:{_sz(v)}:{prep}"

    def native_embedder(self) -> Any:
        """The kernel-slot handle (#342 P2): the dual encoder composed behind
        the engine contract — ONE adapted instance serving both the text and
        image halves, so kernel embeds (text, OCR re-embeds, image vectors)
        never re-enter this facade. Must be called from a coroutine context
        (it captures the running loop).
        """
        if not self.running:
            raise RuntimeError("CLIP embedding backend is not running")
        import shrike_native

        return shrike_native.NativeEmbedder.from_clip(
            self._native_engine,
            fingerprint=self.model_fingerprint(),
            dim=self.embedding_dim(),
            safe_batch=self._effective_batch(self._safe_batch),
        )

    def health(self) -> dict[str, Any]:
        return {
            "available": self.running,
            "backend": "clip",
            "model": self._model,
            "modalities": sorted(self.modalities),
            "requested_providers": self._providers,
            "active_providers": self._active_providers,
            "provider": self._active_providers[0] if self._active_providers else None,
            "batch_safe": self._safe_batch >= 2,
            "batch": "batched" if self._effective_batch(2) >= 2 else "serial",
        }
