"""In-process CLIP backend — image and text into one shared vector space (onnxruntime).

A CLIP export is a **dual encoder**: a text graph (``text_model.onnx``: ``input_ids`` →
``text_embeds``) and a vision graph (``vision_model.onnx``: ``pixel_values`` → ``image_embeds``),
both projecting into the *same* space (cosine-comparable after L2 normalization). So a text query
can retrieve a card by the content of its image — the capability the Phase-3a eval (#193) proved.
This backend advertises ``modalities = {text, image}``; text-only collections keep using the
small text backends (the ``modalities`` seam).

Unlike the text-only ``OnnxBackend`` there's no pooling: both graphs emit a pre-pooled, projected
vector. Image preprocessing (resize → center-crop → rescale → normalize) is read from the model's
``preprocessor_config.json`` and done in PIL + numpy (no torch/torchvision). Provider resolution
and the batch-safety probe are reused from the text-backend work (#175/#176): the text and vision
graphs share the model's quantization, so one probe on the text path governs both (an int8 CLIP is
batch-variant → serial; an fp CLIP batches).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np

from shrike.embed_batching import probe_max_safe_batch
from shrike.embed_text import EMBED_TEXT_VERSION
from shrike.embedding_base import IMAGE, TEXT
from shrike.embedding_onnx_common import ORT_INT_DTYPES, resolve_execution_providers

logger = logging.getLogger("shrike.embedding")

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
# CLIP text models use a fixed 77-token context (positional embeddings); pad/truncate to it.
CLIP_CONTEXT = 77
# Bump if the image-preprocessing pipeline changes (folds into model_fingerprint, like
# EMBED_TEXT_VERSION does for text — a change must invalidate the image vectors).
IMAGE_PREP_VERSION = 1
# CLIP exports ship a graph at several precisions; auto-discovery prefers full precision (best
# quality, and it batches) and falls back to a quantized one, picking the first precision for
# which *both* graphs exist (so text and vision stay the same precision — no mixed-precision pair).
# Explicit operator selection of a precision is tracked in #210.
_VARIANT_SUFFIXES = ("", "_fp16", "_quantized", "_int8", "_uint8", "_q4", "_bnb4")

# The text inputs this backend can supply; whichever the graph declares are fed (CLIP text usually
# declares only input_ids, but some exports also take attention_mask). Mirrors OnnxBackend.
_SUPPLIED_TEXT_INPUTS = frozenset({"input_ids", "attention_mask"})

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
    """In-process onnxruntime CLIP backend (text + image).

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
        native: bool = False,
    ) -> None:
        self._model = model
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
        self._active_providers: list[str] = []
        self._batch_cap = batch_size
        self._safe_batch = 1
        # Engine selection (#271): False = onnxruntime-python + PIL, True = shrike_native.
        self._native = native
        self._native_engine: Any = None
        self._text_sess: Any = None
        self._vis_sess: Any = None
        self._tokenizer: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._resize = 224  # shortest-edge resize target (preprocessor "size")
        self._crop = 224  # center-crop size (preprocessor "crop_size")
        self._dim: int | None = None
        self._text_path: Path | None = None
        self._vis_path: Path | None = None
        self._Image: Any = None  # the PIL.Image module, captured at start()

    @property
    def running(self) -> bool:
        if self._native:
            return self._native_engine is not None
        return self._text_sess is not None and self._vis_sess is not None

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
        """Load both inference sessions, the tokenizer, and the image preprocessing config."""
        if self.running:
            return
        try:
            import json

            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "The CLIP embedding backend needs the 'clip' optional dependency. "
                "Install it with: pip install 'shrike[clip]'"
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

        # The preprocessor config is parsed here for both engines: only scalars
        # cross the FFI into the native engine.
        pp = json.loads(pp_path.read_text())
        self._mean = np.array(pp["image_mean"], dtype=np.float32)
        self._std = np.array(pp["image_std"], dtype=np.float32)
        # CLIP preprocessing has two independent knobs: resize the shortest edge to `size`, then
        # center-crop to `crop_size`. Each may be a dict or a bare scalar across exports.
        self._resize = _read_edge(pp.get("size"), ("shortest_edge", "height", "width"), 224)
        self._crop = _read_edge(pp.get("crop_size"), ("height", "width"), self._resize)

        if self._native:
            self._start_native(tok_path, resolved)
            self._finish_start()
            return

        try:
            from PIL import Image
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "The CLIP embedding backend needs the 'clip' optional dependency. "
                "Install it with: pip install 'shrike[clip]'"
            ) from e

        self._Image = Image  # captured so _preprocess doesn't re-import per image
        self._text_sess = ort.InferenceSession(str(self._text_path), providers=resolved)
        self._vis_sess = ort.InferenceSession(str(self._vis_path), providers=resolved)
        self._active_providers = list(self._text_sess.get_providers())
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
        tokenizer.enable_truncation(max_length=CLIP_CONTEXT)
        tokenizer.enable_padding(length=CLIP_CONTEXT)
        self._tokenizer = tokenizer
        self._finish_start()

    def _start_native(self, tok_path: Path, resolved: list[str]) -> None:
        """Bring up the Rust CLIP engine (#271): both graphs + in-crate tokenization
        and image preprocessing, dlopening the same onnxruntime shared library."""
        try:
            import shrike_native
        except ImportError as e:
            raise ImportError(
                "The clip-rs embedding backend needs the shrike-native extension. "
                "Install the shrike-native wheel, or build it from a checkout with "
                "scripts/build-native.sh."
            ) from e

        from shrike.embedding_onnx import locate_ort_dylib

        assert self._text_path is not None and self._vis_path is not None
        assert self._mean is not None and self._std is not None
        shrike_native.init_onnx_runtime(str(locate_ort_dylib()))
        self._native_engine = shrike_native.ClipEmbedder(
            str(self._text_path),
            str(self._vis_path),
            str(tok_path),
            providers=resolved,
            image_mean=[float(v) for v in self._mean],
            image_std=[float(v) for v in self._std],
            resize=self._resize,
            crop=self._crop,
            context=CLIP_CONTEXT,
        )
        self._active_providers = list(self._native_engine.active_providers())

    def _finish_start(self) -> None:
        """The engine-agnostic tail of start(): the batch-safety probe + logging."""
        # Both graphs are auto-discovered at the *same* precision (_resolve_files), so one probe on
        # the text path governs the vision path too. (A hand-assembled mixed-precision pair — fp
        # text + int8 vision — would batch the vision graph non-deterministically; that's
        # unsupported, which is why _resolve_files refuses to mix precisions. Probing the vision
        # path directly is tracked in #211.)
        try:
            self._safe_batch = probe_max_safe_batch(self._embed_text_chunk)
            if self._safe_batch == 1 and self._batch_cap and self._batch_cap > 1:
                logger.warning(
                    "CLIP model is batch-variant; embedding serially (batch size 1) for "
                    "determinism — use an fp CLIP export for batched throughput."
                )
        except Exception as e:  # noqa: BLE001 — never fail boot on a probe hiccup
            logger.warning("Batch-safety probe failed (%s); embedding serially.", e)
            self._safe_batch = 1

        assert self._text_path is not None and self._vis_path is not None
        logger.info(
            "CLIP model ready (%s + %s, %s)",
            self._text_path.name,
            self._vis_path.name,
            "serial" if self._safe_batch == 1 else "batched",
        )

    def stop(self) -> None:
        self._text_sess = None
        self._vis_sess = None
        self._tokenizer = None
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
        if self._native:
            vectors_native: list[list[float]] = self._native_engine.embed_text_chunk(texts)
            if vectors_native:
                self._dim = len(vectors_native[0])
            return vectors_native
        encodings = self._tokenizer.encode_batch(texts)
        # Feed whichever of input_ids/attention_mask the graph declares, each cast to its declared
        # integer dtype (onnxruntime won't auto-cast). CLIP text usually declares only input_ids,
        # but some exports also take attention_mask — mirrors OnnxBackend's feed-what's-declared.
        supplied = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        feed = {
            inp.name: supplied[inp.name].astype(ORT_INT_DTYPES.get(inp.type, np.int64))
            for inp in self._text_sess.get_inputs()
            if inp.name in _SUPPLIED_TEXT_INPUTS
        }
        out = np.asarray(self._text_sess.run(None, feed)[0], dtype=np.float32)
        return self._normalize(out)

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
        if self._native:
            # Marshaling rule: only bytes cross the FFI. Paths are read here; a
            # PIL image (test convenience) is round-tripped through lossless PNG.
            vectors_native: list[list[float]] = self._native_engine.embed_image_chunk(
                [self._image_bytes(im) for im in images]
            )
            if vectors_native:
                self._dim = len(vectors_native[0])
            return vectors_native
        pixels = np.stack([self._preprocess(im) for im in images]).astype(np.float32)
        feed = {self._vis_sess.get_inputs()[0].name: pixels}
        out = np.asarray(self._vis_sess.run(None, feed)[0], dtype=np.float32)
        return self._normalize(out)

    def _image_bytes(self, image: ImageInput) -> bytes:
        """Encoded bytes for the native engine (which decodes crate-side)."""
        if isinstance(image, bytes):
            return image
        if isinstance(image, (str, Path)):
            return Path(image).read_bytes()
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")  # lossless — pixels unchanged
        return buf.getvalue()

    def _preprocess(self, image: ImageInput) -> np.ndarray:
        """CLIP preprocessing → CHW float32 (resize shortest-edge, center-crop, rescale, normalize)."""  # noqa: E501

        assert self._mean is not None and self._std is not None  # set in start()
        img: Any
        if isinstance(image, bytes):
            img = self._Image.open(io.BytesIO(image))
        elif isinstance(image, (str, Path)):
            img = self._Image.open(image)
        else:
            img = image  # assume a PIL image
        img = img.convert("RGB")
        # Resize the shortest edge to `_resize`, then center-crop to `_crop` (independent knobs).
        w, h = img.size
        scale = self._resize / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), self._Image.Resampling.BICUBIC)
        w, h = img.size
        c = self._crop
        left, top = (w - c) // 2, (h - c) // 2
        img = img.crop((left, top, left + c, top + c))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - self._mean) / self._std
        chw: np.ndarray = arr.transpose(2, 0, 1)  # HWC → CHW
        return chw

    # -- shared --------------------------------------------------------------

    def _normalize(self, vecs: np.ndarray) -> list[list[float]]:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)
        self._dim = int(vecs.shape[1])
        result: list[list[float]] = vecs.astype(np.float32).tolist()
        return result

    def embedding_dim(self) -> int | None:
        if self._dim is not None:
            return self._dim
        if not self.running:
            return None
        if self._native:
            dim = self._native_engine.dim()
            if dim is not None:
                self._dim = int(dim)
                return self._dim
            return len(self.embed_texts([" "])[0])
        # Prefer the graph's static output dim (text_embeds is [batch, dim]); embed only if dynamic.
        shape = self._text_sess.get_outputs()[0].shape
        if shape and isinstance(shape[-1], int):
            self._dim = int(shape[-1])
            return self._dim
        return len(self.embed_texts([" "])[0])

    def model_fingerprint(self) -> str:
        """Stable identity for the index ``model_id`` — both graphs + image-prep + text-prep.

        Prefixed ``clip:`` so it never collides with a text-only ``onnx:``/``meta:`` fingerprint;
        a change to either graph (or the preprocessing) forces a re-embed.
        """

        def _sz(p: Path | None) -> int:
            try:
                return p.stat().st_size if p else -1
            except OSError:
                return -1

        t, v = self._text_path, self._vis_path
        tn = t.name if t else "text_model"
        vn = v.name if v else "vision_model"
        # The native engine's image preprocessing (the image crate's Catmull-Rom
        # bicubic) is pixel-different from PIL's bicubic, so its vectors live in
        # a (slightly) different space — namespaced accordingly (epic #265
        # convention 7), with its own image-prep version counter.
        if self._native:
            import shrike_native

            prep = f"imgprep=rs{shrike_native.IMAGE_PREP_VERSION_RS}:textprep={EMBED_TEXT_VERSION}"
            return f"clip-rs:{tn}:{_sz(t)}:{vn}:{_sz(v)}:{prep}"
        prep = f"imgprep={IMAGE_PREP_VERSION}:textprep={EMBED_TEXT_VERSION}"
        return f"clip:{tn}:{_sz(t)}:{vn}:{_sz(v)}:{prep}"

    def health(self) -> dict[str, Any]:
        return {
            "available": self.running,
            "backend": "clip-rs" if self._native else "clip",
            "model": self._model,
            "modalities": sorted(self.modalities),
            "requested_providers": self._providers,
            "active_providers": self._active_providers,
            "provider": self._active_providers[0] if self._active_providers else None,
            "batch_safe": self._safe_batch >= 2,
            "batch": "batched" if self._effective_batch(2) >= 2 else "serial",
        }
