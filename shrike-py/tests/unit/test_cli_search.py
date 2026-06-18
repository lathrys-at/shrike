"""Unit coverage for the `shrike search` group.

The retrieval group exposes `search <query>` (a default-command group),
`search query`, and `search coverage` (the cross-modal coverage matrix, not in
`server status`). These drive the group with a mocked client via Click's
CliRunner — no server.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.cli.search_cmd import _search_match_badges
from shrike.schemas import (
    CoverageCell,
    CoverageMatrix,
    CoverageRow,
    EmbeddingRunning,
    FuzzyMatch,
    IndexReady,
    ListNotesResponse,
    SearchMatch,
    SearchResponse,
    ServerStatus,
    SignalContribution,
    SubstringInfo,
)


@pytest.fixture
def run(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args, client=None):
        fake = client or MagicMock()
        with patch("shrike.client.ShrikeClient", return_value=fake):
            res = runner.invoke(cli, ["--config", str(cfg), *args])
        return res, fake

    return _run


def _help_commands(run, *group):
    res, _ = run(*group, "--help")
    out, started = [], False
    for line in res.output.splitlines():
        if line.strip() == "Commands:":
            started = True
            continue
        if started and (m := re.match(r"^\s{2}(\S+)\s{2,}", line)):
            out.append(m.group(1))
    return out


class TestSearchGroupShape:
    def test_help_lists_only_query_and_coverage(self, run):
        # The default `search` command is hidden; only the named subcommands show.
        assert _help_commands(run, "search") == ["query", "coverage"]

    def test_search_is_top_level(self, run):
        # `search` is a top-level group, listed at the root and invokable.
        assert "search" in _help_commands(run)
        res, _ = run("search", "--help")
        assert res.exit_code == 0

    def test_bare_search_errors(self, run):
        # No query and no subcommand → usage error (the default command needs input).
        res, _ = run("search")
        assert res.exit_code != 0


class TestDefaultSearchCommand:
    def test_free_text_query_dispatches_to_search_notes(self, run):
        fake = MagicMock()
        fake.search_notes.return_value = SearchResponse(results=[])
        res, _ = run("--json", "search", "electron transport", client=fake)
        assert res.exit_code == 0, res.output
        assert fake.search_notes.called
        assert fake.search_notes.call_args.kwargs["queries"] == ["electron transport"]

    def test_leading_option_routes_to_default_command(self, run):
        # `search --similar-to N` is the default command's option, not the group's.
        fake = MagicMock()
        fake.search_notes.return_value = SearchResponse(results=[])
        res, _ = run("--json", "search", "--similar-to", "123", client=fake)
        assert res.exit_code == 0, res.output
        assert fake.search_notes.call_args.kwargs["ids"] == [123]

    def test_top_k_and_threshold_forwarded(self, run):
        fake = MagicMock()
        fake.search_notes.return_value = SearchResponse(results=[])
        res, _ = run("--json", "search", "x", "--limit", "5", "--threshold", "0.7", client=fake)
        assert res.exit_code == 0, res.output
        assert fake.search_notes.call_args.kwargs["limit"] == 5
        assert fake.search_notes.call_args.kwargs["threshold"] == 0.7


class TestSearchQuery:
    def test_query_subcommand_dispatches_to_query(self, run):
        fake = MagicMock()
        fake.query.return_value = ListNotesResponse(notes=[], total=0, limit=50)
        res, _ = run("--json", "search", "query", "is:due", client=fake)
        assert res.exit_code == 0, res.output
        assert fake.query.called
        assert not fake.search_notes.called
        assert fake.query.call_args.args[0] == "is:due"

    def test_query_requires_expression(self, run):
        fake = MagicMock()
        res, _ = run("search", "query", client=fake)
        assert res.exit_code != 0
        fake.query.assert_not_called()


def _status_with_coverage() -> ServerStatus:
    return ServerStatus(
        wire_protocol_version=1,
        pid=1,
        url="http://127.0.0.1:8372/mcp",
        collection="/c.anki2",
        log_level="info",
        log_dir="/logs",
        embedding=EmbeddingRunning(available=True),
        index=IndexReady(state="ready", size=1, ndim=2),
        coverage=CoverageMatrix(
            text=CoverageRow(
                text=CoverageCell.NATIVE,
                image=CoverageCell.VIA_DERIVED_TEXT,
                audio=CoverageCell.UNAVAILABLE,
            ),
        ),
    )


class TestSearchCoverage:
    def test_renders_matrix(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _status_with_coverage()
        res, _ = run("search", "coverage", client=fake)
        assert res.exit_code == 0, res.output
        assert "Coverage" in res.output
        assert "native" in res.output
        assert "via text" in res.output

    def test_json_emits_coverage(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _status_with_coverage()
        res, _ = run("--json", "search", "coverage", client=fake)
        assert res.exit_code == 0, res.output
        assert "native" in res.output

    def test_server_not_running_errors(self, run):
        fake = MagicMock()
        fake.server_status.return_value = None
        res, _ = run("search", "coverage", client=fake)
        assert res.exit_code != 0
        assert "not running" in res.output.lower() or "not responding" in res.output.lower()


def _match(**kw: Any) -> SearchMatch:
    base = {"id": 1, "note_type": "Basic", "deck": "D", "modified": "2024-01-01T00:00:00"}
    return SearchMatch(**{**base, **kw})


class TestSearchMatchBadges:
    """The `search` pretty badge shows a non-text provenance facet, but not `text`/`exact`
    which the score / `match:` badges already imply."""

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

    def test_fuzzy_facet_renders(self):
        # A fuzzy-only near-miss is otherwise invisible (no score, no `match:`), so the
        # `fuzzy` facet surfaces on its own — like `image`, it's a non-{text,exact} signal.
        m = _match(
            fuzzy=FuzzyMatch(source="field", ref="Front", snippet="…protein…"),
            provenance=[SignalContribution(signal="fuzzy", rank=1)],
        )
        assert _search_match_badges(m) == "fuzzy"
