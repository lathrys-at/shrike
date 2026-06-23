"""Unit coverage for `shrike collection info` render branches.

Covers the bare-summary path, the detail-flag include assembly, and each
detail render (note types, decks, tags, stats) plus the JSON variant and the
empty/None negative branches. Driven through Click's CliRunner with a mocked
ShrikeClient — no server.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli import client as cli_client
from shrike.schemas import (
    CollectionInfo,
    DeckInfo,
    DeckStat,
    NoteTypeDetail,
    NoteTypeInfo,
    Stats,
    Summary,
    TemplateInfo,
)


@pytest.fixture
def fake() -> MagicMock:
    return MagicMock(spec=cli_client.ShrikeClient)


@pytest.fixture
def run(tmp_path, fake):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str, **kwargs):
        with patch("shrike.client.ShrikeClient", return_value=fake):
            return runner.invoke(cli, ["--config", str(cfg), *args], **kwargs)

    return _run


def _summary() -> Summary:
    return Summary(
        path="/c.anki2",
        created="2026-01-01",
        modified="2026-01-02",
        notes=10,
        cards=20,
        decks=2,
        note_types=3,
        tags=4,
        due_today=5,
    )


class TestInfoSummary:
    def test_bare_summary_renders_key_values(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(summary=_summary())
        result = run("collection", "info")
        assert result.exit_code == 0, result.output
        assert "Collection:" in result.output
        assert "/c.anki2" in result.output
        assert "Due today:" in result.output
        # No detail flags → only the summary include is requested.
        assert fake.collection_info.call_args.kwargs["include"] == ["summary"]

    def test_bare_no_summary_is_silent(self, run, fake) -> None:
        # No summary in the response and no flags → nothing rendered, no crash.
        fake.collection_info.return_value = CollectionInfo(summary=None)
        result = run("collection", "info")
        assert result.exit_code == 0, result.output
        assert "Collection:" not in result.output

    def test_json_emits_full_response(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(summary=_summary())
        result = run("--json", "collection", "info")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["summary"]["notes"] == 10


class TestInfoDetailIncludes:
    def test_all_flags_assemble_include_list(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(
            summary=_summary(),
            note_types=[NoteTypeInfo(name="Basic", id=1, type="standard", fields=["F"])],
            decks=[DeckInfo(name="D", id=1, note_count=3)],
            tags=["t"],
            stats=Stats(total_notes=10, total_cards=20),
        )
        result = run("collection", "info", "--types", "--decks", "--tags", "--stats")
        assert result.exit_code == 0, result.output
        include = fake.collection_info.call_args.kwargs["include"]
        assert include == ["summary", "note_types", "decks", "tags", "stats"]

    def test_type_details_implies_note_types_include(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(
            note_types=[NoteTypeInfo(name="Basic", id=1, type="standard", fields=["F"])],
        )
        result = run("collection", "info", "--type-details", "Basic")
        assert result.exit_code == 0, result.output
        kwargs = fake.collection_info.call_args.kwargs
        assert "note_types" in kwargs["include"]
        assert kwargs["note_type_details"] == ["Basic"]


class TestInfoDetailRender:
    def test_note_types_with_detail_renders_panel(self, run, fake) -> None:
        nt = NoteTypeInfo(
            name="Basic",
            id=1,
            type="standard",
            fields=["Front", "Back"],
            detail=NoteTypeDetail(
                templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
                css=".card {}",
            ),
        )
        fake.collection_info.return_value = CollectionInfo(note_types=[nt])
        result = run("collection", "info", "--types")
        assert result.exit_code == 0, result.output
        assert "Basic" in result.output
        assert "Card 1" in result.output  # the detail panel rendered

    def test_decks_render_with_counts(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(
            decks=[DeckInfo(name="Japanese", id=1, note_count=42)]
        )
        result = run("collection", "info", "--decks")
        assert result.exit_code == 0, result.output
        assert "Showing 1 decks" in result.output
        assert "Japanese" in result.output and "42" in result.output

    def test_tags_render_sorted(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(tags=["zebra", "alpha"])
        result = run("collection", "info", "--tags")
        assert result.exit_code == 0, result.output
        assert "Showing 2 tags" in result.output
        # Sorted ascending: alpha precedes zebra.
        assert result.output.index("alpha") < result.output.index("zebra")

    def test_stats_render_with_deck_breakdown(self, run, fake) -> None:
        fake.collection_info.return_value = CollectionInfo(
            stats=Stats(
                total_notes=10,
                total_cards=20,
                cards_due_today=3,
                new_cards=4,
                decks_summary={"Bio": DeckStat(notes=5, due=2)},
            )
        )
        result = run("collection", "info", "--stats")
        assert result.exit_code == 0, result.output
        assert "Showing statistics" in result.output
        assert "Bio" in result.output

    def test_stats_without_deck_breakdown_omits_table(self, run, fake) -> None:
        # Empty decks_summary → the per-deck table is skipped (negative branch).
        fake.collection_info.return_value = CollectionInfo(
            stats=Stats(total_notes=1, total_cards=1, decks_summary={})
        )
        result = run("collection", "info", "--stats")
        assert result.exit_code == 0, result.output
        assert "Showing statistics" in result.output
        assert "Deck" not in result.output  # the breakdown header

    def test_multiple_sections_separated_by_blank_lines(self, run, fake) -> None:
        # Decks + tags + stats all present exercises the `if printed` separators.
        fake.collection_info.return_value = CollectionInfo(
            note_types=[NoteTypeInfo(name="Basic", id=1, type="standard", fields=["F"])],
            decks=[DeckInfo(name="D", id=1, note_count=1)],
            tags=["t"],
            stats=Stats(total_notes=1, total_cards=1),
        )
        result = run("collection", "info", "--types", "--decks", "--tags", "--stats")
        assert result.exit_code == 0, result.output
        # Sections appear in order: note types, decks, tags, stats.
        out = result.output
        assert out.index("Showing 1 decks") < out.index("Showing 1 tags")
        assert out.index("Showing 1 tags") < out.index("Showing statistics")
