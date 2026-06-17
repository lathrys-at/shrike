"""Canonical subcommand ordering + the command-surface rehome (#683 / #682 §A,§G).

`OrderedGroup.list_commands` drives both `--help` and shell completions, so these
pin the §G order for the root group and every subgroup, that the rehomed paths
resolve, and that the old top-level commands are a clean break (removed).
"""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from shrike.cli import cli

# The canonical per-group order from epic #682 §G (post-rehome). `profile` keeps
# its current add/remove/default/list verbs — the #686 rename to
# create/rename/delete updates this list when it lands.
EXPECTED_ORDER = {
    (): ["collection", "search", "server", "note", "deck", "type", "profile", "completion"],
    ("note",): ["create", "update", "delete", "list", "show", "tag", "replace", "migrate-type"],
    ("type",): ["create", "update", "delete", "list", "show"],
    ("deck",): ["create", "rename", "delete"],
    ("profile",): ["add", "remove", "default", "list"],
    ("search",): ["query", "coverage"],
    ("collection",): ["info", "check", "export", "import", "prune", "reload", "tag", "media"],
    ("collection", "media"): ["store", "fetch", "list", "delete"],
    ("server",): ["start", "stop", "status", "logs", "embedding", "index"],
    ("server", "embedding"): ["start", "stop", "status"],
    ("server", "index"): ["rebuild", "save", "status"],
}


def _help_commands(*group: str) -> list[str]:
    res = CliRunner().invoke(cli, [*group, "--help"])
    assert res.exit_code == 0, res.output
    out, started = [], False
    for line in res.output.splitlines():
        if line.strip() == "Commands:":
            started = True
            continue
        if started and (m := re.match(r"^\s{2}(\S+)\s{2,}", line)):
            out.append(m.group(1))
    return out


@pytest.mark.parametrize("group,expected", list(EXPECTED_ORDER.items()), ids=lambda v: str(v))
def test_canonical_order(group, expected):
    assert _help_commands(*group) == expected


# The old top-level commands are a clean break — removed, must error (#683 §A).
REMOVED_TOP_LEVEL = [
    ["note", "search"],
    ["info"],
    ["export"],
    ["import"],
    ["tag"],
    ["media"],
    ["index"],
    ["embedding"],
    ["collection", "query"],
]


@pytest.mark.parametrize("path", REMOVED_TOP_LEVEL, ids=lambda p: " ".join(p))
def test_removed_command_errors(path):
    res = CliRunner().invoke(cli, [*path, "--help"])
    assert res.exit_code == 2, res.output  # Click "No such command"


# The rehomed paths resolve (their --help renders).
REHOMED_PATHS = [
    ["collection", "info"],
    ["collection", "export"],
    ["collection", "import"],
    ["collection", "tag", "rename"],
    ["collection", "media", "store"],
    ["server", "embedding", "status"],
    ["server", "index", "status"],
    ["search", "query"],
    ["search", "coverage"],
]


@pytest.mark.parametrize("path", REHOMED_PATHS, ids=lambda p: " ".join(p))
def test_rehomed_path_resolves(path):
    res = CliRunner().invoke(cli, [*path, "--help"])
    assert res.exit_code == 0, res.output
