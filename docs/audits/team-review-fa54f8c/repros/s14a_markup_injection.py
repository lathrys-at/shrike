"""S14a-1 repro (preserved by lead; rev-S14a worktree reaped).
Rich-markup injection from untrusted note content: output.py interpolates note
field/tag/deck/snippet into Rich markup f-strings with NO rich.markup.escape(),
then console.print(markup=True). (A) well-formed tags → terminal spoofing;
(B) malformed tag → uncaught MarkupError crashes the whole command (content DoS).
RED at fa54f8c. Every display mode affected (detail, brief table, search snippet).
--json safe; --no-pretty avoids the spoof but still crashes.
Run: SHRIKE_SKIP_NATIVE_STALE_CHECK=1 .venv/bin/python -m pytest <this> -q -p no:cacheprovider
Fix: rich.markup.escape() untrusted content (or pass as Text/markup=False).
"""
from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.errors import MarkupError

from shrike.cli import output
from shrike.schemas import Note


def _render(note: Note) -> str:
    buf = io.StringIO()
    cap = Console(file=buf, force_terminal=True, color_system="standard", highlight=False)
    orig = output.console
    output.console = cap
    try:
        output.note_detail(note)
    finally:
        output.console = orig
    return buf.getvalue()


def test_field_value_markup_is_interpreted_not_escaped() -> None:
    note = Note(id=1, note_type="Basic", deck="Default", tags=[],
                modified="2026-01-01T00:00:00",
                content={"Front": "Capital of France?", "Back": "[blink]gotcha[/blink]"})
    rendered = _render(note)
    # PREDICTED-CORRECT: tags shown literally (escaped). Today: interpreted (RED).
    assert "[blink]" in rendered, f"markup INTERPRETED, not escaped — injection.\n{rendered!r}"


def test_malformed_field_value_does_not_crash_the_command() -> None:
    note = Note(id=2, note_type="Basic", deck="Default", tags=[],
                modified="2026-01-01T00:00:00",
                content={"Front": "Q", "Back": "see [/cyan] here"})  # stray closing tag
    try:
        _render(note)
    except MarkupError as err:
        pytest.fail(f"untrusted content raised unhandled MarkupError (content DoS): {err}")


def test_brief_table_tag_markup_is_interpreted() -> None:
    note = Note(id=8, note_type="Basic", deck="Default", tags=["[blink]flashy[/blink]"],
                modified="2026-01-01T00:00:00", content=None)
    row = output.note_summary_row(note)
    buf = io.StringIO()
    cap = Console(file=buf, force_terminal=True, color_system="standard", highlight=False)
    orig = output.console
    output.console = cap
    try:
        output.table(["ID", "Type", "Deck", "Tags", "Modified"], [row])
    finally:
        output.console = orig
    assert "[blink]" in buf.getvalue(), "tag markup interpreted in brief table — injection"

# S14a-2 (Low): --json --pretty mutual-exclusion is order-dependent.
# `info --pretty --json` does NOT raise the documented UsageError (only the
# --json-first order does), because _merge_pretty only errors when json is
# already set. See report; fix: check both flags after parsing, order-independent.
