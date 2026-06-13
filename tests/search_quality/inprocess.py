"""In-process search-quality harness for the deterministic CI classes (#559).

The deterministic classes need exactly-controlled cosines (to pin the RRF
fused order, the exact-override tier, and the activation gate), with NO model
download and NO spawned server — so they run in normal CI. This module is the
``tests/native/test_harness.py`` pattern distilled to a reusable builder:

  - :class:`StubEmbedder` — a deterministic text+image embedder. Text vectors
    are keyed by a ``@@P:key@@`` marker (robust to the ``FieldName: …`` wrapping
    + HTML/cloze normalization the kernel applies before embedding) or by the
    exact query string; image vectors by ``sha256`` of the image bytes; anything
    un-planted falls back to a stable token/byte hash. Advertises
    ``{text, image}`` and is CAPTURED via ``PyEmbedder.capture`` (no
    ``native_embedder``), so the kernel drives it exactly as a custom backend —
    the full fusion pipeline runs against the planted vectors.

  - :func:`build_harness` — assembles a real :class:`~shrike.harness.Harness`
    over a throwaway collection, attaches the stub (with the image resolver, so
    image vectors index and the activation gate can calibrate), and returns the
    harness + a ``call`` coroutine that drives the REAL ``search_notes`` MCP
    action through ``FastMCP.call_tool`` — the exact path a client's
    ``tools/call`` reaches.

Why in-process and not a spawned server: the spawned-server path opens TWO
derived-store handles (the kernel's and the host facade's) that can resolve to
different per-collection namespaces, so a lexical hit a query should make can
read empty. In-process there is ONE store and one harness — the fusion is
exercised faithfully and deterministically.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import shrike_native

from shrike.derived import DerivedTextStore, NativeDerivedEngine
from shrike.embedding import EmbeddingRuntime
from shrike.embedding_base import IMAGE, TEXT
from shrike.harness import Harness, KernelIndexView

_MARKER_RE = re.compile(r"@@P:([A-Za-z0-9_-]+)@@")


def l2(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def onehot(dim: int, axis: int, secondary: float = 0.0, *, offset: int = 1) -> list[float]:
    """A unit vector along ``axis`` with optional bleed into the next axis — a
    cheap controllable-cosine generator (``cos = 1/√(1+s²)`` to the pure axis)."""
    v = [0.0] * dim
    v[axis] = 1.0
    if secondary:
        v[(axis + offset) % dim] = secondary
    return l2(v)


def _hash_vec(payload: bytes, dim: int) -> list[float]:
    """Deterministic token/byte-hash fallback for an un-planted input."""
    v = [0.0] * dim
    for token in payload.split():
        v[int(hashlib.blake2b(token, digest_size=2).hexdigest(), 16) % dim] += 1.0
    if not any(v):
        v[int(hashlib.blake2b(payload, digest_size=2).hexdigest(), 16) % dim] = 1.0
    return l2(v)


@dataclass
class StubEmbedder:
    """A deterministic text+image embedder driven by planted-vector maps.

    ``texts`` maps a card marker key (``@@P:key@@`` in a field) OR an exact
    query string to a vector; ``images`` maps ``sha256(bytes).hexdigest()`` to a
    vector. Un-planted inputs hash. All vectors are L2-normalized on use, so the
    planted cosines are literally dot products.
    """

    dim: int = 16
    texts: dict[str, list[float]] = field(default_factory=dict)
    images: dict[str, list[float]] = field(default_factory=dict)
    fingerprint: str = "stub:v1"

    # -- authoring helpers ---------------------------------------------------
    def plant_text(self, key: str, vec: Sequence[float]) -> None:
        self.texts[key] = l2(vec)

    def plant_query(self, query: str, vec: Sequence[float]) -> None:
        self.texts[query] = l2(vec)

    def plant_image(self, raw: bytes, vec: Sequence[float]) -> None:
        self.images[hashlib.sha256(raw).hexdigest()] = l2(vec)

    # -- EmbedderBackend surface (captured: no native_embedder) ---------------
    @property
    def modalities(self) -> frozenset[str]:
        return frozenset({TEXT, IMAGE})

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            marker = _MARKER_RE.search(text)
            planted: list[float] | None = None
            if marker is not None:
                planted = self.texts.get(marker.group(1))
            if planted is None:
                planted = self.texts.get(text)
            out.append(
                planted if planted is not None else _hash_vec(text.encode("utf-8"), self.dim)
            )
        return out

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        out: list[list[float]] = []
        for raw in images:
            planted = self.images.get(hashlib.sha256(raw).hexdigest())
            out.append(planted if planted is not None else _hash_vec(raw, self.dim))
        return out

    def model_fingerprint(self) -> str:
        return self.fingerprint

    def embedding_dim(self) -> int:
        return self.dim


# A coroutine that runs one search_notes call and returns the structured result.
SearchCall = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class InProcessSearch:
    """The assembled harness + the real-action search driver."""

    harness: Harness
    backend: StubEmbedder | None
    mcp: Any
    media: dict[str, bytes]

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Drive the REAL ``search_notes`` action via ``call_tool``.

        Returns the full structured response (``results``/``message``/
        ``completeness``) so callers can read provenance AND degradation."""
        args: dict[str, Any] = {"queries": [query], **kwargs}
        _, structured = await self.mcp.call_tool("search_notes", args)
        return structured

    async def matches(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        resp = await self.search(query, **kwargs)
        groups = resp.get("results", [])
        return groups[0]["matches"] if groups else []

    async def finalize(self) -> None:
        """Index + derived-ingest everything upserted so far.

        In this minimal assembly the incremental upsert tail does not populate
        the engine, so a test upserts its whole corpus and then calls this once:
        a full index rebuild (vectors + the activation calibration when ≥30
        images are present) and a derived rebuild (FTS5 substring/fuzzy rows).
        After it, the REAL ``search_notes`` action sees a complete index +
        store — exactly what a booted server serves. With no embedder attached
        (the embedding-down class) only the derived store is built — there is no
        vector index to rebuild."""
        if self.backend is not None:
            await self.harness.kernel.rebuild_index()
        await self.harness.kernel.rebuild_derived()

    def index_status(self) -> dict[str, Any]:
        import json

        return json.loads(self.harness.kernel.index_status_json())


def to_returned_cards(matches: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Adapt search_notes match dicts into the metric engine's ``ReturnedCard``s."""
    from tests.search_quality.metrics import ReturnedCard

    cards = []
    for rank, m in enumerate(matches, start=1):
        signals = frozenset(p["signal"] for p in m.get("provenance", []))
        cards.append(
            ReturnedCard(
                note_id=int(m["id"]),
                rank=rank,
                signals=signals,
                score=m.get("score"),
                has_substring=m.get("substring") is not None,
                has_fuzzy=m.get("fuzzy") is not None,
            )
        )
    return cards


def to_ranked_cards(matches: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Adapt match dicts into ``RankedCard``s (per-signal ranks from provenance)
    for the pure RRF golden-order recompute."""
    from tests.search_quality.metrics import RankedCard

    out = []
    for m in matches:
        ranks = {p["signal"]: int(p["rank"]) for p in m.get("provenance", [])}
        out.append(RankedCard(note_id=int(m["id"]), signal_ranks=ranks))
    return out


async def build_harness(
    tmp_path: Path,
    backend: StubEmbedder | None,
    *,
    media: Mapping[str, bytes] | None = None,
    attach_media: bool = True,
) -> InProcessSearch:
    """Assemble a harness over a throwaway collection with the stub attached.

    ``attach_media`` wires the image resolver into ``attach_embedder`` so image
    vectors index (required for the activation-gate class); leave it off for a
    text-only scenario. Pass ``backend=None`` to leave the embed slot EMPTY (the
    embedding-down degradation class: lexical search still works, semantic
    announces unavailable). Registers the real MCP tool registry against the
    harness's surfaces and returns the driver."""
    from types import SimpleNamespace

    from mcp.server.fastmcp import FastMCP

    from shrike.tools import register_tools

    media_map: dict[str, bytes] = dict(media or {})
    runtime = EmbeddingRuntime(model=None)
    derived = DerivedTextStore(
        path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
    )
    harness = await Harness.assemble(
        collection_path=str(tmp_path / "collection.anki2"),
        cache_dir=str(tmp_path / "cache"),
        runtime=runtime,
        derived=derived,
        cooperative=False,
        hold_seconds=5.0,
        media_read=media_map.get,
        media_exists=lambda name: name in media_map,
    )
    await harness.boot(start_embedding=False)
    if backend is not None:
        captured = shrike_native.PyEmbedder.capture(backend)
        if attach_media:
            harness.kernel.attach_embedder(captured, media_map.get, lambda name: name in media_map)
        else:
            harness.kernel.attach_embedder(captured)
        await harness.kernel.reindex_if_needed()

    # The search-facing view embeds queries with the SAME backend (as the
    # server wires it); a None backend leaves the semantic tier off.
    view = KernelIndexView(harness.kernel, SimpleNamespace(backend=backend))  # type: ignore[arg-type]
    mcp = FastMCP("search-quality")
    register_tools(mcp, harness.wrapper, index=view, kernel=harness.kernel, derived=harness.derived)
    return InProcessSearch(harness=harness, backend=backend, mcp=mcp, media=media_map)
