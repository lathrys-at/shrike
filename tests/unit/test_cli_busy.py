"""CLI renders a busy collection as a clean, actionable message (#65)."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.client import CollectionBusyError


def test_busy_shows_actionable_message_no_traceback():
    msg = "The collection is in use by another process (is Anki open?). Close it and try again."
    with patch("shrike.client.ShrikeClient.collection_info", side_effect=CollectionBusyError(msg)):
        result = CliRunner().invoke(cli, ["info"])
    assert result.exit_code != 0
    assert "in use by another process" in result.output
    assert "Traceback" not in result.output
