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

logger = logging.getLogger("shrike.embedding")

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
# CLIP text models use a fixed 77-token context (positional embeddings); pad/truncate to it.
CLIP_CONTEXT = 77
# Bump if the image-preprocessing pipeline changes (folds into model_fingerprint, like
# EMBED_TEXT_VERSION does for text — a change must invalidate the image vectors).
IMAGE_PREP_VERSION = 1

# An image accepted by embed_images: a PIL image, a filesystem path, or raw bytes.
ImageInput = Any


def resolve_execution_providers(
    available: list[str], requested: list[str]
) -> tuple[list[str], list[str]]:
    """Intersect requested providers with what onnxruntime has, append CPU as the final
    fallback, dedup. Returns (resolved, dropped). Mirrors OnnxBackend's resolution so an
    unavailable accelerator degrades with a warning rather than onnxruntime's silent CPU fallback.
    """
    resolved: list[str] = []
    for p in [*requested, "CPUExecutionProvider"]:
        if p in available and p not in resolved:
            resolved.append(p)
    dropped = [p for p in requested if p not in available]
    return resolved, dropped


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
        variant: str | None = None,
        log_dir: str | Path | None = None,  # accepted for runtime parity; unused
    ) -> None:
        self._model = model
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
        self._active_providers: list[str] = []
        self._batch_cap = batch_size
        # A graph-file suffix, e.g. "quantized" → text_model_quantized.onnx (None = plain).
        self._variant = variant
        self._safe_batch = 1
        self._text_sess: Any = None
        self._vis_sess: Any = None
        self._tokenizer: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._crop = 224
        self._dim: int | None = None
        self._text_path: Path | None = None
        self._vis_path: Path | None = None
        self._Image: Any = None  # the PIL.Image module, captured at start()

    @property
    def running(self) -> bool:
        return self._text_sess is not None and self._vis_sess is not None

    def _resolve_files(self) -> tuple[Path, Path, Path, Path]:
        """Locate text graph, vision graph, tokenizer.json, preprocessor_config.json."""
        root = Path(self._model)
        if not root.is_dir():
            raise FileNotFoundError(f"CLIP model path is not a directory: {root}")
        suffix = f"_{self._variant}" if self._variant else ""

        def _graph(stem: str) -> Path:
            for rel in (f"onnx/{stem}{suffix}.onnx", f"{stem}{suffix}.onnx"):
                if (root / rel).is_file():
                    return root / rel
            raise FileNotFoundError(f"No {stem}{suffix}.onnx under {root}")

        tok = next(
            (root / r for r in ("tokenizer.json", "onnx/tokenizer.json") if (root / r).is_file()),
            None,
        )
        if tok is None:
            raise FileNotFoundError(f"No tokenizer.json under {root}")
        pp = next(
            (
                root / r
                for r in ("preprocessor_config.json", "onnx/preprocessor_config.json")
                if (root / r).is_file()
            ),
            None,
        )
        if pp is None:
            raise FileNotFoundError(f"No preprocessor_config.json under {root}")
        return _graph("text_model"), _graph("vision_model"), tok, pp

    def start(self) -> None:
        """Load both inference sessions, the tokenizer, and the image preprocessing config."""
        if self.running:
            return
        try:
            import json

            import onnxruntime as ort
            from PIL import Image
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "The CLIP embedding backend needs the 'clip' optional dependency. "
                "Install it with: pip install 'shrike[clip]'"
            ) from e

        self._Image = Image  # captured so _preprocess doesn't re-import per image
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

        pp = json.loads(pp_path.read_text())
        self._mean = np.array(pp["image_mean"], dtype=np.float32)
        self._std = np.array(pp["image_std"], dtype=np.float32)
        crop = pp.get("crop_size", {})
        self._crop = int(crop.get("height") or crop.get("width") or 224)

        # Both graphs share the model's quantization, so one probe on the text path governs both.
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
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        # CLIP text graph declares only input_ids (it finds the EOS position internally).
        feed = {self._text_sess.get_inputs()[0].name: input_ids}
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
        pixels = np.stack([self._preprocess(im) for im in images]).astype(np.float32)
        feed = {self._vis_sess.get_inputs()[0].name: pixels}
        out = np.asarray(self._vis_sess.run(None, feed)[0], dtype=np.float32)
        return self._normalize(out)

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
        s = self._crop
        w, h = img.size
        scale = s / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), self._Image.Resampling.BICUBIC)
        w, h = img.size
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
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
        prep = f"imgprep={IMAGE_PREP_VERSION}:textprep={EMBED_TEXT_VERSION}"
        return f"clip:{tn}:{_sz(t)}:{vn}:{_sz(v)}:{prep}"

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
