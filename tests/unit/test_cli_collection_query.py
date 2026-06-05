"""CLI handling for `shrike collection query` (#97), with the client stubbed."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import ListNotesResponse

_RESULT = ListNotesResponse(
    notes=[
        {
            "id": 1,
            "note_type": "Basic",
            "deck": "D",
            "tags": ["q"],
            "modified": "2026-06-04T00:00:00Z",
            "content": {"Front": "hello", "Back": "world"},
        }
    ],
    total=1,
    limit=50,
)


def _run(args, **kwargs):
    with patch("shrike.client.ShrikeClient.query", return_value=_RESULT) as m:
        result = CliRunner().invoke(cli, ["collection", "query", *args], **kwargs)
    return result, m


class TestCollectionQueryCLI:
    def test_renders_results(self):
        result, m = _run(["tag:q"])
        assert result.exit_code == 0
        assert m.call_args.args[0] == "tag:q"
        assert "hello" in result.output

    def test_brief_requests_meta(self):
        _, m = _run(["tag:q", "--brief"])
        assert m.call_args.kwargs["fields"] == "meta"

    def test_default_requests_full(self):
        _, m = _run(["tag:q"])
        assert m.call_args.kwargs["fields"] == "full"

    def test_limit_forwarded(self):
        _, m = _run(["deck:*", "--limit", "100"])
        assert m.call_args.kwargs["limit"] == 100

    def test_json_emits_response(self):
        result, _ = _run(["--json", "tag:q"])
        assert result.exit_code == 0
        assert '"total": 1' in result.output or '"total":1' in result.output

    def test_requires_expression(self):
        result, m = _run([])
        assert result.exit_code != 0
        m.assert_not_called()
