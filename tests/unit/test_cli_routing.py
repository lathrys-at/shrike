"""CLI collection routing — the global --profile/--collection selector (#68 S2).

The root option is stored on the client as its per-call selector; these pin
that wiring (the client itself injects it into each call — see
test_client.TestSelectorInjection).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import CollectionInfo, Summary


def _run(tmp_path, args, **kwargs):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    fake = MagicMock()
    fake.collection_info.return_value = CollectionInfo(
        summary=Summary(
            path="/x.anki2",
            created="2024",
            modified="2024",
            notes=0,
            cards=0,
            decks=0,
            note_types=0,
            tags=0,
            due_today=0,
        )
    )
    with patch("shrike.client.ShrikeClient", return_value=fake) as ctor:
        result = CliRunner().invoke(cli, ["--config", str(cfg), *args], **kwargs)
    return result, ctor


class TestProfileSelector:
    def test_profile_passed_as_collection_to_client(self, tmp_path):
        result, ctor = _run(tmp_path, ["--profile", "work", "collection", "info"])
        assert result.exit_code == 0, result.output
        # The client is constructed with the selector as its `collection`.
        _, kwargs = ctor.call_args
        assert kwargs["collection"] == "work"

    def test_collection_alias_passed_as_collection(self, tmp_path):
        # --collection is an alias of --profile at the root (routes by name).
        result, ctor = _run(tmp_path, ["--collection", "home", "collection", "info"])
        assert result.exit_code == 0, result.output
        _, kwargs = ctor.call_args
        assert kwargs["collection"] == "home"

    def test_no_selector_is_none(self, tmp_path):
        result, ctor = _run(tmp_path, ["collection", "info"])
        assert result.exit_code == 0, result.output
        _, kwargs = ctor.call_args
        assert kwargs["collection"] is None
