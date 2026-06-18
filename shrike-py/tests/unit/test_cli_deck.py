"""CLI argument handling for `shrike deck`, with the client stubbed."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import (
    CollectionInfo,
    DeckInfo,
    DeleteDecksResponse,
    UpsertDecksResponse,
)

_UPSERT_OK = UpsertDecksResponse(results=[{"status": "created", "id": 1, "name": "X"}])


def _invoke(args, *, upsert=None, delete=None, decks=None, **kwargs):
    info = CollectionInfo(decks=decks if decks is not None else [])
    with (
        patch("shrike.client.ShrikeClient.upsert_decks", return_value=upsert or _UPSERT_OK) as up,
        patch(
            "shrike.client.ShrikeClient.delete_decks",
            return_value=delete or DeleteDecksResponse(deleted=[]),
        ) as dl,
        patch("shrike.client.ShrikeClient.collection_info", return_value=info),
    ):
        result = CliRunner().invoke(cli, ["deck", *args], **kwargs)
    return result, up, dl


class TestDeckCreate:
    def test_create_sends_name(self):
        result, up, _ = _invoke(["create", "Japanese::Vocab"])
        assert result.exit_code == 0, result.output
        up.assert_called_once_with([{"name": "Japanese::Vocab"}])


class TestDeckRename:
    def test_rename_resolves_old_to_id(self):
        decks = [DeckInfo(name="Old", id=42)]
        result, up, _ = _invoke(["rename", "Old", "New"], decks=decks)
        assert result.exit_code == 0, result.output
        up.assert_called_once_with([{"id": 42, "name": "New"}])

    def test_rename_unknown_old_errors(self):
        result, up, _ = _invoke(["rename", "Ghost", "New"], decks=[DeckInfo(name="Other", id=1)])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
        up.assert_not_called()

    def test_rename_by_numeric_id(self):
        decks = [DeckInfo(name="Old", id=42)]
        result, up, _ = _invoke(["rename", "42", "New"], decks=decks)
        assert result.exit_code == 0, result.output
        up.assert_called_once_with([{"id": 42, "name": "New"}])

    def test_rename_by_hash_id(self):
        decks = [DeckInfo(name="Old", id=42)]
        result, up, _ = _invoke(["rename", "#42", "New"], decks=decks)
        assert result.exit_code == 0, result.output
        up.assert_called_once_with([{"id": 42, "name": "New"}])


class TestMatchDeck:
    """The deck-ref matcher used by `deck rename` (mirrors the server rule)."""

    def _decks(self):
        return [DeckInfo(name="Alpha", id=10), DeckInfo(name="999", id=20)]

    def test_name_match(self):
        from shrike.cli.deck_cmd import _match_deck

        assert _match_deck(self._decks(), "Alpha").id == 10

    def test_hash_id_match(self):
        from shrike.cli.deck_cmd import _match_deck

        assert _match_deck(self._decks(), "#10").id == 10

    def test_numeric_prefers_id_over_name(self):
        from shrike.cli.deck_cmd import _match_deck

        # "999" is both deck #20's name and not an id present → matches by name;
        # but a bare number equal to an existing id matches that id first.
        assert _match_deck(self._decks(), "20").id == 20  # id 20 wins
        assert _match_deck(self._decks(), "999").id == 20  # no id 999 → name "999"

    def test_no_match(self):
        from shrike.cli.deck_cmd import _match_deck

        assert _match_deck(self._decks(), "Nope") is None


class TestDeckDelete:
    def test_delete_yes_skips_prompt(self):
        result, _, dl = _invoke(
            ["delete", "A", "B", "--yes"], delete=DeleteDecksResponse(deleted=["A", "B"])
        )
        assert result.exit_code == 0, result.output
        dl.assert_called_once_with(["A", "B"])

    def test_delete_prompt_cancel(self):
        result, _, dl = _invoke(["delete", "A"], input="n\n")
        assert "Cancelled" in result.output
        dl.assert_not_called()

    def test_delete_not_empty_exits_nonzero(self):
        result, _, _ = _invoke(
            ["delete", "A", "--yes"], delete=DeleteDecksResponse(not_empty=["A"])
        )
        assert result.exit_code != 0
        assert "not empty" in result.output.lower()
