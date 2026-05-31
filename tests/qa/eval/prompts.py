"""Canonical agent prompts for the eval runs.

Two configs:
- ``with_skill``: the agent is told to read and follow SKILL.md (the skill under
  test).
- ``baseline``: the same task and CLI, but NO skill — what the model does by
  default. The with-skill − baseline delta is the skill's measured lift.

Keeping the wording here (not improvised per run) is what makes runs comparable
and the eval reproducible.
"""

from __future__ import annotations

SKILL_PATH = "/Users/lupine/Development/shrike/skills/anki-cards/SKILL.md"
SHRIKE_BIN = "/Users/lupine/Development/shrike/.venv/bin/shrike"

CLI_REFERENCE = "/Users/lupine/Development/shrike/docs/cli-reference.md"

_CLI_BLURB = f"""\
You interact with the user's Anki collection through the Shrike CLI. A Shrike
server is already running locally. Use exactly this binary (the project venv):

  {SHRIKE_BIN}

e.g. `{SHRIKE_BIN} info --decks --types --tags`,
`{SHRIKE_BIN} note search "concept" --json`. Pass --json for structured output.
Do NOT start or stop the server."""


def with_skill_prompt(user_request: str) -> str:
    return f"""\
You are an assistant that has the "anki-cards" skill available. Before doing
anything else, read the skill instructions at {SKILL_PATH} and the reference
files it points to (references/examples.md and references/shrike-cli.md), and
follow them carefully for the task below.

{_CLI_BLURB}

Now handle this user request, exactly as the skill directs:

<user_request>
{user_request}
</user_request>

When finished, give a concise final report: which deck(s) you used, which note
type(s) and why, the tags you applied, the front of each card you created, and
any judgment calls."""


def baseline_prompt(user_request: str) -> str:
    return f"""\
You are a helpful assistant. The user wants to turn the material below into Anki
flashcards in their collection.

{_CLI_BLURB}

A CLI reference (commands, flags, JSON shapes) is at {CLI_REFERENCE} — consult
it as needed rather than probing --help repeatedly.

Handle this user request:

<user_request>
{user_request}
</user_request>

When finished, give a concise final report: which deck(s) you used, which note
type(s), the tags you applied, and the front of each card you created."""
