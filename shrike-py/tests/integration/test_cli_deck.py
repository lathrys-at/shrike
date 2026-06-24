"""CLI integration tests for the `shrike deck` command group."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestDeckGroup:
    """Deck lifecycle via CLI."""

    def _deck_names(self, runner):
        return {d["name"] for d in runner.json(["collection", "info", "--decks"])["decks"]}

    def test_create(self, runner):
        data = runner.json(["deck", "create", "CLIDeck::Sub"])
        assert data["results"][0]["status"] == "created"
        assert "CLIDeck::Sub" in self._deck_names(runner)

    def test_rename(self, runner):
        runner.json(["deck", "create", "RenmeFrom"])
        data = runner.json(["deck", "rename", "RenmeFrom", "RenmeTo"])
        assert data["results"][0]["status"] == "updated"
        names = self._deck_names(runner)
        assert "RenmeTo" in names and "RenmeFrom" not in names

    def test_rename_unknown_errors(self, runner):
        result = runner.invoke(["deck", "rename", "NoSuchDeck", "Whatever"])
        assert result.exit_code != 0

    def test_delete_empty(self, runner):
        runner.json(["deck", "create", "DeleteMe"])
        data = runner.json(["deck", "delete", "DeleteMe", "--yes"])
        assert data["deleted"] == ["DeleteMe"]
        assert "DeleteMe" not in self._deck_names(runner)

    def test_delete_non_empty_refused(self, runner):
        runner.json(
            [
                "note",
                "create",
                "--deck",
                "FullDeck",
                "--type",
                "Basic",
                "-f",
                "Front=Q",
                "-f",
                "Back=A",
            ]
        )
        result = runner.invoke(["deck", "delete", "FullDeck", "--yes"])
        assert result.exit_code != 0
        assert "not empty" in result.output.lower()
        assert "FullDeck" in self._deck_names(runner)

    def _deck_id(self, runner, name):
        return str(
            next(
                d["id"]
                for d in runner.json(["collection", "info", "--decks"])["decks"]
                if d["name"] == name
            )
        )

    def test_rename_and_delete_by_id(self, runner):
        created = runner.json(["deck", "create", "IDDeck"])
        did = str(created["results"][0]["id"])
        # rename by #id (deck id is unchanged by a rename)
        data = runner.json(["deck", "rename", f"#{did}", "IDDeck2"])
        assert data["results"][0]["status"] == "updated"
        assert "IDDeck2" in self._deck_names(runner)
        # delete the (empty) deck by bare numeric id
        out = runner.json(["deck", "delete", did, "--yes"])
        assert out["deleted"] == [did]

    def test_note_list_by_deck_id(self, runner):
        runner.json(
            ["note", "create", "--deck", "NLD", "--type", "Basic", "-f", "Front=q", "-f", "Back=a"]
        )
        did = self._deck_id(runner, "NLD")
        assert runner.json(["note", "list", "--deck", f"#{did}"])["total"] == 1
