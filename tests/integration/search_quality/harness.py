"""The search-quality run harness (#559).

``run_search_quality(mcp, manifest)`` builds the adversarial collection through
the REAL tools — ``store_media`` (so a collision-rename is transparent) then
``upsert_notes`` with the returned filenames substituted into ``$IMG:handle`` —
waits the index ready, runs each manifest query through the REAL
``search_notes`` JSON-RPC action, and feeds the returned-with-provenance + the
graded gold to the pure metric engine. One engine, two consumers: the pytest
suite asserts the resulting :class:`SuiteReport`; ``scripts/eval_search_quality``
renders it to ``eval/search_quality/RESULTS.md``. No mock — fusion / per-modality
ranking / the activation gate / the exact override all run as a client hits them.

PR1 builds only ``source: generated`` media (inline ``bytes_b64`` or a
synthesized PNG) — no network. PR2 adds the ``source: commons`` resolve→pin→cache
path behind the same loader.
"""

from __future__ import annotations

import base64
import re
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass

from tests.integration.search_quality.manifest import CardSpec, Manifest, MediaSpec
from tests.integration.search_quality.metrics import (
    QueryReport,
    ReturnedCard,
    SuiteReport,
    evaluate_query,
)

# A `store_media`-style callable the harness drives, and a `search_notes`-style
# one. In the suite these are the conftest ``MCPClient`` (a ``tools/call`` over
# HTTP); typed structurally so the harness has no conftest import cycle.
ToolCall = Callable[[str, dict], dict]

_IMG_TOKEN = re.compile(r"\$IMG:([A-Za-z0-9_-]+)")


def _png_bytes(seed: bytes, *, size: int = 8) -> bytes:
    """A tiny deterministic solid-colour PNG keyed by ``seed``.

    The planted backend keys image vectors by ``sha256(bytes)``, so what matters
    is that distinct handles yield distinct bytes — a 1-pixel-repeated solid
    colour derived from the seed is enough and needs no Pillow. (Used when a
    generated media row carries no inline ``bytes_b64``.)"""
    h = zlib.adler32(seed)
    r, g, b = (h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF)
    raw = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def _media_bytes(spec: MediaSpec) -> bytes:
    """Resolve a generated media row to bytes (PR1: no network)."""
    if spec.source != "generated":
        raise NotImplementedError(
            f"media source {spec.source!r} is a PR2 path (Commons resolve→pin→cache); "
            "PR1 manifests use source='generated'"
        )
    b64 = spec.spec.get("bytes_b64")
    if isinstance(b64, str):
        return base64.b64decode(b64)
    return _png_bytes(spec.handle.encode("utf-8"))


@dataclass
class BuiltCard:
    """A planted card with its manifest id mapped to the live Anki note id."""

    manifest_id: int
    note_id: int


def build_collection(mcp: ToolCall, manifest: Manifest) -> dict[int, int]:
    """Plant the manifest's cards through ``store_media`` + ``upsert_notes``.

    Returns ``{manifest_id: anki_note_id}`` so gold (keyed by manifest id) maps
    onto returned cards (keyed by anki id). Each card's ``$IMG:handle`` tokens
    are replaced with the actual ``store_media`` filename (collision-safe).
    """
    id_map: dict[int, int] = {}
    for card in manifest.cards:
        filenames = _store_card_media(mcp, card)
        fields = {k: _substitute(v, filenames) for k, v in card.fields.items()}
        resp = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": card.deck,
                        "note_type": card.note_type,
                        "fields": fields,
                        "tags": list(card.tags),
                    }
                ],
                "on_duplicate": "allow",
            },
        )
        result = resp["results"][0]
        if result.get("status") == "error":
            raise RuntimeError(f"card {card.id} failed to upsert: {result}")
        id_map[card.id] = int(result["id"])
    return id_map


def _store_card_media(mcp: ToolCall, card: CardSpec) -> dict[str, str]:
    """Store each of a card's images, returning ``{handle: stored_filename}``."""
    out: dict[str, str] = {}
    for media in card.media:
        raw = _media_bytes(media)
        resp = mcp(
            "store_media",
            {
                "media": [
                    {
                        "filename": f"{media.handle}.png",
                        "data": base64.b64encode(raw).decode("ascii"),
                    }
                ]
            },
        )
        result = resp["results"][0]
        if result.get("status") != "stored":
            raise RuntimeError(f"store_media failed for {media.handle}: {result}")
        out[media.handle] = result["filename"]
    return out


def _substitute(text: str, filenames: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        handle = m.group(1)
        if handle not in filenames:
            raise KeyError(f"$IMG:{handle} referenced but no such media handle on the card")
        return filenames[handle]

    return _IMG_TOKEN.sub(repl, text)


def _returned_cards(group: dict) -> list[ReturnedCard]:
    """Parse a search_notes result group into the metric engine's input."""
    cards: list[ReturnedCard] = []
    for rank, match in enumerate(group.get("matches", []), start=1):
        signals = frozenset(p["signal"] for p in match.get("provenance", []))
        cards.append(
            ReturnedCard(
                note_id=int(match["id"]),
                rank=rank,
                signals=signals,
                score=match.get("score"),
                has_substring=match.get("substring") is not None,
                has_fuzzy=match.get("fuzzy") is not None,
            )
        )
    return cards


def _remap_gold(gold_grades: dict[int, int], id_map: dict[int, int]) -> dict[int, int]:
    """Translate manifest-id-keyed grades to live-anki-id-keyed grades.

    A gold/hard-negative id with no planted card (a manifest authoring slip) is
    dropped — it can never be returned, so it would only ever read as a recall
    miss against a card that doesn't exist.
    """
    return {id_map[mid]: g for mid, g in gold_grades.items() if mid in id_map}


def run_search_quality(
    mcp: ToolCall,
    manifest: Manifest,
    *,
    id_map: dict[int, int] | None = None,
) -> SuiteReport:
    """Build the collection (unless ``id_map`` is supplied) and grade every query.

    ``id_map`` lets a caller build once and re-grade (the #234 sweep re-runs
    queries over the same planted collection with different SearchArgs).
    """
    if id_map is None:
        id_map = build_collection(mcp, manifest)

    reports: list[QueryReport] = []
    for q in manifest.queries:
        args: dict = {"queries": [q.q], "top_k": q.top_k, "tier": "full"}
        if q.threshold is not None:
            args["threshold"] = q.threshold
        resp = mcp("search_notes", args)
        groups = resp.get("results", [])
        group = groups[0] if groups else {"matches": []}
        returned = _returned_cards(group)

        # The response announces degradation via any of: a `message`, a
        # `completeness != "full"`, or every returned score being None (the
        # embedding tier didn't run). The metric engine consumes the boolean.
        announced = (
            bool(resp.get("message"))
            or resp.get("completeness") not in (None, "full")
            or (bool(returned) and all(c.score is None for c in returned))
        )

        from dataclasses import replace

        gold = replace(q.gold, grades=_remap_gold(dict(q.gold.grades), id_map))
        reports.append(
            evaluate_query(
                q.q,
                q.adversarial_class,
                returned,
                gold,
                response_announced_degradation=announced,
                expects_degradation=q.expects_degradation,
            )
        )
    return SuiteReport(queries=tuple(reports))
