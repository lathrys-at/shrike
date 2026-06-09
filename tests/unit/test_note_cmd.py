"""Unit tests for `note_cmd` CLI helpers — the search-match badge rendering (#182)."""

from __future__ import annotations

from typing import Any

from shrike.cli.note_cmd import _search_match_badges
from shrike.schemas import SearchMatch, SignalContribution, SubstringInfo


def _match(**kw: Any) -> SearchMatch:
    base = {"id": 1, "note_type": "Basic", "deck": "D", "modified": "2024-01-01T00:00:00"}
    return SearchMatch(**{**base, **kw})


class TestSearchMatchBadges:
    """The `note search` pretty badge shows a non-text provenance facet, but not `text`/`exact`
    which the score / `match:` badges already imply (#182 review)."""

    def test_image_facet_renders(self):
        # The keep-branch: a non-text modality is otherwise invisible from a bare score, so it
        # surfaces on its own.
        m = _match(score=0.30, provenance=[SignalContribution(signal="image", rank=1)])
        assert _search_match_badges(m) == "image · 0.30"

    def test_text_only_has_no_signal_prefix(self):
        # `text` is already implied by the score badge → not repeated.
        m = _match(score=0.85, provenance=[SignalContribution(signal="text", rank=1)])
        assert _search_match_badges(m) == "0.85"

    def test_exact_only_has_no_signal_prefix(self):
        # `exact` is already implied by the `match:` field badge → not repeated.
        m = _match(
            substring=SubstringInfo(matched_fields=["Front"]),
            provenance=[SignalContribution(signal="exact", rank=1)],
        )
        assert _search_match_badges(m) == "match: Front"

    def test_text_and_exact_not_doubled(self):
        m = _match(
            score=0.85,
            substring=SubstringInfo(matched_fields=["Front"]),
            provenance=[
                SignalContribution(signal="exact", rank=1),
                SignalContribution(signal="text", rank=2),
            ],
        )
        assert _search_match_badges(m) == "0.85 · match: Front"

    def test_modality_facet_alongside_exact(self):
        # The facet shows; the redundant `exact` does not.
        m = _match(
            score=0.30,
            substring=SubstringInfo(matched_fields=["Front"]),
            provenance=[
                SignalContribution(signal="image", rank=1),
                SignalContribution(signal="exact", rank=1),
            ],
        )
        assert _search_match_badges(m) == "image · 0.30 · match: Front"
