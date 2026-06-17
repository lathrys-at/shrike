"""The search-quality manifest loader (#559).

Parses the reconciled manifest schema into typed objects the harness builds a
collection from and the metric engine grades against. The schema::

    { "closed_world": true,              // ungraded-returned ⇒ grade-0
      "cards": [
        { "id": 2, "kind": "image_only", "deck": "AdversarialEval::Image",
          "note_type": "Basic", "tags": ["adv-image"],
          "fields": {"Front": "Review card 2", "Back": "<img src=\"$IMG:heart\">"},
          "media": [{"handle": "heart", "source": "commons|generated",
                     "spec": {"search_term": "..."} | {"bytes_b64": "..."}}] } ],
      "queries": [
        { "q": "...", "adversarial_class": "modality_gap",
          "expected_signal": "image", "modality": "image|text",
          "gold": [{"id": 2, "grade": 3}],
          "hard_negatives": [{"id": 301, "grade": 0}],
          "top_k": 10, "threshold": 0.5 } ] }

``media[].handle`` is LOGICAL: the harness substitutes the
``store_media``-returned filename into ``$IMG:handle`` at build time, so a
collision-rename is transparent (#559). ``source: generated`` is the PR1
network-free path (image bytes inline as ``bytes_b64``, or synthesized);
``source: commons`` is the PR2 resolve→pin→cache path (named here, not built).
``grade`` defaults to 3 when omitted on a gold row, 0 on a hard-negative row.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tests.search_quality.metrics import GradedGold


@dataclass(frozen=True)
class MediaSpec:
    """One image attached to a card. ``handle`` is the logical name referenced
    as ``$IMG:handle`` in a field; ``source`` selects how bytes are obtained."""

    handle: str
    source: str  # "generated" (PR1) | "commons" (PR2)
    spec: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CardSpec:
    """One note to plant in the adversarial collection."""

    id: int
    kind: str  # text_only | image_only | both | distractor
    deck: str
    note_type: str
    fields: Mapping[str, str]
    tags: tuple[str, ...] = ()
    media: tuple[MediaSpec, ...] = ()


@dataclass(frozen=True)
class QuerySpec:
    """One query to run, its graded gold, and its adversarial intent."""

    q: str
    adversarial_class: str
    gold: GradedGold
    expected_signal: str | None
    modality: str
    top_k: int
    threshold: float | None
    # PR1 degradation classes set these so the metric engine can require the
    # response to *announce* the degradation (message / completeness / score).
    expects_degradation: bool = False


@dataclass(frozen=True)
class Manifest:
    closed_world: bool
    cards: tuple[CardSpec, ...]
    queries: tuple[QuerySpec, ...]
    # An optional planted-vector plan filename (relative to the manifest) for the
    # deterministic classes — the harness points the planted backend at it.
    plan: str | None = None
    name: str = "manifest"

    @property
    def card_by_id(self) -> dict[int, CardSpec]:
        return {c.id: c for c in self.cards}


def _media(rows: list[dict]) -> tuple[MediaSpec, ...]:
    return tuple(
        MediaSpec(
            handle=str(r["handle"]),
            source=str(r.get("source", "generated")),
            spec=dict(r.get("spec", {})),
        )
        for r in rows
    )


def _gold(rows: list[dict], *, default_grade: int) -> dict[int, int]:
    return {int(r["id"]): int(r.get("grade", default_grade)) for r in rows}


def load_manifest(path: str | Path) -> Manifest:
    """Parse a manifest JSON into typed objects.

    Gold and hard-negatives merge into one ``{id: grade}`` map per query
    (gold rows default grade 3, hard-negatives default grade 0). ``expected_signal``
    ``null``/absent means a null-gold/precision query (no winning signal).
    """
    path = Path(path)
    data = json.loads(path.read_text())
    closed_world = bool(data.get("closed_world", True))

    cards = tuple(
        CardSpec(
            id=int(c["id"]),
            kind=str(c.get("kind", "text_only")),
            deck=str(c.get("deck", "AdversarialEval")),
            note_type=str(c.get("note_type", "Basic")),
            fields=dict(c["fields"]),
            tags=tuple(c.get("tags", [])),
            media=_media(c.get("media", [])),
        )
        for c in data.get("cards", [])
    )

    queries: list[QuerySpec] = []
    for q in data.get("queries", []):
        grades = _gold(q.get("gold", []), default_grade=3)
        grades.update(_gold(q.get("hard_negatives", []), default_grade=0))
        top_k = int(q.get("top_k", 10))
        expected = q.get("expected_signal")
        queries.append(
            QuerySpec(
                q=str(q["q"]),
                adversarial_class=str(q.get("adversarial_class", "unclassified")),
                gold=GradedGold(
                    grades=grades,
                    expected_signal=None if expected in (None, "null") else str(expected),
                    closed_world=closed_world,
                    top_k=top_k,
                ),
                expected_signal=None if expected in (None, "null") else str(expected),
                modality=str(q.get("modality", "text")),
                top_k=top_k,
                threshold=q.get("threshold"),
                expects_degradation=bool(q.get("expects_degradation", False)),
            )
        )

    return Manifest(
        closed_world=closed_world,
        cards=cards,
        queries=tuple(queries),
        plan=data.get("plan"),
        name=str(data.get("name", path.stem)),
    )
