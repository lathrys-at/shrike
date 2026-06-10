"""Embedding backend protocol — the seam every embedder implements.

Shrike's vector index (`index.py`) and server boot path (`server.py`) depend only
on this minimal surface, never on a specific runtime. The implementations are
``LlamaServerBackend`` (a llama-server subprocess, `embedding.py`) and
``OnnxBackend`` (in-process onnxruntime, `embedding_onnx.py`); a future multimodal
embedder is just another implementation that advertises more ``modalities``.

Keeping the surface this small is what lets the backend be swapped without
touching the index: drift detection, the per-note hash sidecar, and persistence
are all backend-agnostic — they only ever call ``embed_texts`` and read
``model_fingerprint``/``embedding_dim``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


@runtime_checkable
class IndexEngine(Protocol):
    """The storage engine under the ``VectorIndex`` orchestrator (#267).

    **Frozen as the future FFI surface** (#273): the native ``shrike-index``
    crate implements this verbatim, so the calls are coarse and batched,
    trafficking only in i64 key arrays, f32 vector arrays (numpy or nested
    sequences), and small JSON-able dicts — never live Python objects. The
    orchestrator keeps everything that is *policy*: the state machine, drift
    detection, the reconcile hash-diff and its fallback ladder, background
    threads, and metadata persistence. Implementations are instance-per-space
    with no global state (#232's multi-space manager is "make N engines").

    Engine quirks are part of the contract (pinned by the unit suite): the
    phantom ``(0, 0)`` hit on an empty index is filtered inside
    ``search_by_modality``; multi-key dedup is min-distance-per-note (== max-sim
    over a note's vectors); ``remove`` returns the count removed from the *text*
    sub-index.
    """

    @property
    def size(self) -> int:
        """Total vectors across every modality sub-index."""
        ...

    @property
    def ndim(self) -> int | None:
        """The shared vector dimension, or None before the first add/restore."""
        ...

    def modality_sizes(self) -> dict[str, int]:
        """Vector count per loaded modality (a created-but-empty sub-index counts, at 0)."""
        ...

    def ensure(self, modality: str, ndim: int) -> None:
        """Create the (empty) sub-index for a modality if it doesn't exist yet."""
        ...

    def clear(self) -> None:
        """Drop every in-memory sub-index (file deletion is the orchestrator's)."""
        ...

    def restore(self, path: str, candidate_keys: Sequence[int] | None = None) -> bool:
        """Load sub-index files under ``path``; False (and empty) on a corrupt present file.

        ``candidate_keys`` are the note ids that may be indexed (the
        orchestrator's hashes sidecar). The Python engine ignores them (its
        binding enumerates keys natively); the Rust engine reconstructs its
        per-key map from them, returning False — the standard drift rebuild —
        when they can't account for every stored vector (#273).
        """
        ...

    def save(self, path: str) -> None:
        """Persist every loaded sub-index under ``path``; delete stale modality files."""
        ...

    def add(self, modality: str, keys: Any, vectors: Any) -> None:
        """Add f32 vectors under i64 keys to one modality (pure add — no replace)."""
        ...

    def remove(self, keys: Any) -> int:
        """Remove the keys' vectors from every sub-index; returns the text-index count."""
        ...

    def search_by_modality(
        self,
        query_vectors: Any,
        k: int,
        *,
        modalities: Sequence[str] | None = None,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Per-query ``{modality: [{note_id, distance}, ...]}`` rankings (max-sim per note)."""
        ...

    def contains(self, key: int) -> bool:
        """Whether a note is indexed (every indexed note has a text vector)."""
        ...

    def keys(self) -> list[int]:
        """The distinct note ids in the text sub-index."""
        ...

    def get(self, key: int) -> Any:
        """A note's stored text vector(s), or None if absent."""
        ...

    def calibrate_activation(
        self, sample_size: int, k: int, min_count: int
    ) -> dict[str, dict[str, float]]:
        """Per-(non-text-)modality best-match ``{n, mean, std}`` stats (#201b), ``{}`` if N/A."""
        ...
