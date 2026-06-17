"""Shared Click group classes: a canonical-order group and the search default-command group.

``OrderedGroup`` replaces Click's default alphabetical ``list_commands`` with a
single **canonical order** (epic #682 §G), applied uniformly to the root group
and every subgroup. Click derives both ``--help`` listings *and* shell
completions from ``list_commands``, so ordering it once fixes both surfaces.

``SearchGroup`` is the ``shrike search`` group's class: ``search <query>`` runs a
default search command, while ``search coverage`` / ``search query`` dispatch to
named subcommands (epic #682 §A).
"""

from __future__ import annotations

import click

# Canonical subcommand order per group (epic #682 §G), keyed by group name. Each
# value lists the group's subcommand names in display order; any command not
# listed here falls to the end, alphabetically (so a new command is visible but
# never silently reorders the curated set). The verb classes behind the order:
# create → update|rename → delete → list → show → status|info|check →
# search|query|coverage → domain ops → subgroups last; service start/stop lead;
# `collection media` keeps its natural I/O flow (store, fetch, list, delete).
CANONICAL_ORDER: dict[str, list[str]] = {
    # Root: collection · search · server · note · deck · type · profile · completion
    "shrike": [
        "collection",
        "search",
        "server",
        "note",
        "deck",
        "type",
        "profile",
        "completion",
    ],
    "note": ["create", "update", "delete", "list", "show", "tag", "replace", "migrate-type"],
    "type": ["create", "update", "delete", "list", "show"],
    "deck": ["create", "rename", "delete"],
    "profile": ["add", "remove", "default", "list"],
    "search": ["query", "coverage"],
    "collection": ["info", "check", "export", "import", "prune", "reload", "tag", "media"],
    "media": ["store", "fetch", "list", "delete"],
    "server": ["start", "stop", "status", "logs", "embedding", "index"],
    "embedding": ["start", "stop", "status"],
    "index": ["rebuild", "save", "status"],
}


def _ordered(names: list[str], order: list[str]) -> list[str]:
    """Order ``names`` by ``order``; anything not in ``order`` trails, alphabetical."""
    rank = {name: i for i, name in enumerate(order)}
    return sorted(names, key=lambda n: (rank.get(n, len(order)), n))


class OrderedGroup(click.Group):
    """A ``click.Group`` whose subcommands list in the canonical order (#682 §G).

    The order is looked up by the group's own name in ``CANONICAL_ORDER``; a
    group with no entry falls back to alphabetical (Click's default), so adding a
    group without curating its order is safe.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        order = CANONICAL_ORDER.get(self.name or "")
        names = list(super().list_commands(ctx))
        return _ordered(names, order) if order else sorted(names)


class SearchGroup(OrderedGroup):
    """The ``shrike search`` group: a default search command + named subcommands.

    ``shrike search <query>`` runs the default ``search`` command (semantic +
    substring retrieval); ``shrike search coverage`` / ``shrike search query``
    dispatch to those subcommands. The injection fires only when the first token
    isn't a known subcommand, ``--help``, or empty, so ``search --help`` and the
    subcommands keep working — and a query like ``search "electron"`` or an
    option-led ``search --similar-to 1`` both route to the default command (its
    options, not the group's, which is why we inject before the group parses).
    """

    #: The command invoked when the first token isn't a known subcommand.
    default_command = "search"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # Inject the default command name *before* the group parses its args, so a
        # leading option (`--similar-to`) or a free-text query (`"electron"`) is
        # parsed by the default command rather than rejected as a group option /
        # unknown subcommand. `--help` (no args, or an explicit `--help`) and a
        # real subcommand are left to the group so help and dispatch still work.
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)
