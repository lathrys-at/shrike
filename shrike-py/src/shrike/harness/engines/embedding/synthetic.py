"""In-process synthetic backend — a deterministic, dependency-free embedder.

The native ``shrike_native.SyntheticEmbedder`` (Rust, behind the non-default
``engine-synthetic`` feature) maps each input's bytes to a fixed-dimensionality
unit vector: same input, same vector; distinct inputs spread across the sphere —
with no model to load and negligible per-call cost. It advertises
``{text, image}`` so a single space serves both modalities.

It exists for benchmarking and fast deterministic tests: a profile attaches it
(``runtime: synthetic``) when the goal is to measure the kernel/IO/orchestration
cost a workflow pays, not model-inference time. The vectors carry NO semantics —
neighbour relationships are meaningless, so it is never a search-quality
backend. The capability is gated to non-release builds; a config naming it on a
release build is refused at profile resolution (``engine-synthetic`` is absent
from ``build_features()`` there).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shrike.harness.engines.embedding.base import IMAGE, TEXT

logger = logging.getLogger("shrike.embedding")

# A synthetic embedder loads no model, so its dimension is a free choice; 384
# matches the common small text model (all-MiniLM-L6-v2), keeping index sizing
# comparable to a real-backend run.
DEFAULT_SYNTHETIC_DIM = 384
# Each vector is a pure function of its own input — no batch variance — so the
# chunk size is purely a throughput knob, not a correctness one.
_SAFE_BATCH = 512


class SyntheticBackend:
    """In-process deterministic backend (text + image, native engine).

    Implements the :class:`~shrike.harness.engines.embedding.base.EmbedderBackend`
    protocol, plus ``embed_images`` (``IMAGE`` is in ``modalities``).
    """

    modalities = frozenset({TEXT, IMAGE})

    def __init__(
        self,
        *,
        dim: int = DEFAULT_SYNTHETIC_DIM,
        modalities: frozenset[str] | None = None,
    ) -> None:
        self._dim = dim
        self._engine: Any = None
        # The full text+image space by default; a text-only profile entry
        # narrows it, and image vectors then never index (the modalities seam).
        if modalities is not None:
            self.modalities = frozenset(modalities)

    @property
    def running(self) -> bool:
        return self._engine is not None

    def start(self) -> None:
        if self._engine is not None:
            return
        import shrike_native

        self._engine = shrike_native.SyntheticEmbedder(dim=self._dim)

    def stop(self) -> None:
        self._engine = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.running:
            raise RuntimeError("synthetic embedding backend is not running")
        if not texts:
            return []
        result: list[list[float]] = self._engine.embed_chunk(texts)
        return result

    def embed_images(self, images: list[bytes | str | Path]) -> list[list[float]]:
        """Embed images (raw bytes or a filesystem path) into the shared space."""
        if not self.running:
            raise RuntimeError("synthetic embedding backend is not running")
        if not images:
            return []
        # Only bytes cross the FFI; the synthetic engine hashes them directly.
        encoded = [im if isinstance(im, bytes) else Path(im).read_bytes() for im in images]
        result: list[list[float]] = self._engine.embed_image_chunk(encoded)
        return result

    def embedding_dim(self) -> int | None:
        return self._dim

    def model_fingerprint(self) -> str:
        """Stable identity for the index ``model_id``. The ``synthetic:`` family
        token keeps its vectors distinct from any real backend's, so a stub-mode
        index never loads under a real run (and vice versa)."""
        return f"synthetic:v1:dim={self._dim}"

    def native_embedder(self) -> Any:
        """The kernel-slot handle: the native engine composed behind the engine
        contract (one adapted instance serving both halves), so kernel embeds
        run native and never re-enter this facade."""
        if not self.running:
            raise RuntimeError("synthetic embedding backend is not running")
        import shrike_native

        return shrike_native.NativeEmbedder.from_synthetic(
            self._engine,
            fingerprint=self.model_fingerprint(),
            dim=self._dim,
            safe_batch=_SAFE_BATCH,
        )

    def health(self) -> dict[str, Any]:
        return {
            "available": self.running,
            "backend": "synthetic",
            "dim": self._dim,
            "modalities": sorted(self.modalities),
        }
