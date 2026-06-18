"""Embedding backend protocol — the seam every embedder implements.

Shrike's server boot path (`server.py`) and the kernel attach point depend only
on this minimal surface, never on a specific runtime. The implementations are
``LlamaServerBackend`` (a llama-server subprocess, `embedding.py`) and
``OnnxBackend`` (in-process onnxruntime, `embedding_onnx.py`); a future multimodal
embedder is just another implementation that advertises more ``modalities``.

Keeping the surface this small is what lets the backend be swapped without
touching the index (kernel-owned since #332/#353): drift detection, the
per-note fingerprints, and persistence are all backend-agnostic — they only
ever call ``embed_texts`` and read ``model_fingerprint``/``embedding_dim``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Modality identifiers a backend may advertise in ``modalities``. Phase 1 backends
# are all text-only; IMAGE/AUDIO/VIDEO/PDF are the multimodal seam (#162) and are
# named here so a later slice extends this set rather than inventing strings.
TEXT = "text"
IMAGE = "image"
AUDIO = "audio"
VIDEO = "video"
PDF = "pdf"

# An image resolver maps a media filename to its bytes (None = missing/unreadable → skipped).
ImageResolver = Callable[[str], "bytes | None"]
# A cheap presence check for a media filename (a stat, not a byte read) — folded into the hash.
ImageExists = Callable[[str], bool]


@dataclass(frozen=True)
class NoteEmbedInput:
    """One note's embedding inputs: its normalized text plus any image filenames.

    Produced by ``CollectionWrapper.note_embed_inputs`` (a cheap DB + regex pass, no file reads);
    the index turns it into a text vector and — for an image-capable backend — one vector per
    resolvable image, all stored under the ``note_id`` key.

    This boundary *type* lives here with the protocol seam (not in ``index.py``):
    it's what the collection produces and any index/embedder consumes, so homing it
    on the leaf protocol module keeps the module graph acyclic (#266).
    """

    note_id: int
    text: str
    image_names: list[str] = field(default_factory=list)


@runtime_checkable
class EmbedderBackend(Protocol):
    """The minimal surface the index and server need from an embedder.

    Each implementation owns its own runtime specifics — a subprocess + port and
    orphan-reaping for llama-server, an in-process inference session for ONNX —
    and none of that leaks through this protocol.
    """

    @property
    def modalities(self) -> frozenset[str]:
        """The input modalities this backend's model can embed (always ⊇ {TEXT}).

        This is the graceful-degradation switch. A text query only retrieves
        media when media vectors live in the *same* space in the index, which
        requires a backend whose ``modalities`` covers that media type. A
        text-only backend therefore simply never embeds media: search over
        media-by-content quietly returns nothing rather than erroring, and every
        other search path is unchanged. Text-only is a first-class, permanent
        capability — not a thing that breaks once a multimodal backend exists.
        """
        ...

    @property
    def running(self) -> bool:
        """True once the backend is started/loaded and can embed."""
        ...

    def start(self) -> None:
        """Bring the backend up (spawn the server / load the model + tokenizer)."""
        ...

    def stop(self) -> None:
        """Tear the backend down, releasing its resources."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text inputs into vectors — one per input, in a shared space."""
        ...

    def embedding_dim(self) -> int | None:
        """The model's embedding width, or ``None`` if it can't be determined."""
        ...

    def model_fingerprint(self) -> str:
        """A stable identity for the loaded model — the index ``model_id``.

        It must change whenever the produced vectors would change (model, runtime,
        pooling, text normalization), so a model/backend swap forces a rebuild
        rather than silently mixing two vector spaces. By convention each backend
        prefixes its fingerprint with a token that names its own family
        (``meta:``/``file:`` for llama-server, ``onnx:`` for ONNX), so vectors
        from different backends never collide even for the "same" model — and an
        existing index needs no change to gain that distinctness.
        """
        ...

    def health(self) -> dict[str, Any]:
        """Status dict for the ``/status`` endpoint (carries at least ``available``)."""
        ...
