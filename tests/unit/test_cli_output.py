"""Untrusted note/media content must render literally, never as Rich markup (#593).

Note field/tag/deck/snippet text, note-type/field/template names, and media
filenames are authored by anyone who can write the collection (Anki sync, a
shared/imported ``.apkg``, an MCP upsert). They reach ``console.print`` in
``cli/output.py`` (and the media/export leaf renders) which parses ``[..]`` as
Rich markup. Two impacts the escape fix closes:

  (A) terminal spoofing — a well-formed ``[blink]``/``[/cyan]`` restyles or
      conceals output;
  (B) a content-driven CLI crash (DoS) — a malformed ``[/tag]`` raises an
      uncaught ``rich.errors.MarkupError`` and kills the whole command.

Each test renders a bracket-bearing value and asserts it appears *literally* in
the output and that no ``MarkupError`` escapes. These are RED before the
``rich.markup.escape()`` fix and GREEN after.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console
from rich.errors import MarkupError

from shrike import client as shrike_client
from shrike.cli import cli, output
from shrike.schemas import (
    CollectionCheckResponse,
    CollectionInfo,
    DeckInfo,
    FieldDetail,
    ListMediaResponse,
    MediaFileInfo,
    Note,
    NoteTypeDetail,
    NoteTypeInfo,
    SearchMatch,
    SubstringInfo,
    TemplateInfo,
    UpsertNoteError,
)

# Tag bodies chosen to exercise both failure modes: a well-formed style tag
# (spoof) and a stray closing tag (the MarkupError crash on benign content).
SPOOF = "[blink]gotcha[/blink]"
MALFORMED = "see [/cyan] here"


def _capture(render) -> str:
    """Run ``render`` against a fresh terminal-emulating console, return its text.

    Surfaces a ``MarkupError`` as a test failure (the content-driven DoS) rather
    than letting it propagate as an error.
    """
    buf = io.StringIO()
    cap = Console(file=buf, force_terminal=True, color_system="standard", highlight=False)
    orig = output.console
    output.console = cap
    try:
        render()
    except MarkupError as err:  # pragma: no cover - asserted as a failure
        pytest.fail(f"untrusted content raised an uncaught MarkupError (content DoS): {err}")
    finally:
        output.console = orig
    return buf.getvalue()


def _note(**kw) -> Note:
    base = {
        "id": 1,
        "note_type": "Basic",
        "deck": "Default",
        "tags": [],
        "modified": "2026-01-01T00:00:00",
        "content": None,
    }
    base.update(kw)
    return Note(**base)


def test_note_detail_field_value_rendered_literally() -> None:
    out = _capture(lambda: output.note_detail(_note(content={"Front": "Q", "Back": SPOOF})))
    assert "[blink]" in out and "[/blink]" in out


def test_note_detail_malformed_field_does_not_crash() -> None:
    # The crash repro: a stray closing tag in benign synced content.
    out = _capture(lambda: output.note_detail(_note(content={"Front": "Q", "Back": MALFORMED})))
    assert "[/cyan]" in out


def test_note_detail_tag_rendered_literally() -> None:
    out = _capture(lambda: output.note_detail(_note(tags=[SPOOF])))
    assert "[blink]" in out


def test_note_detail_field_name_rendered_literally() -> None:
    out = _capture(lambda: output.note_detail(_note(content={SPOOF: "value"})))
    assert "[blink]" in out


def test_note_detail_deck_and_type_rendered_literally() -> None:
    out = _capture(lambda: output.note_detail(_note(deck=SPOOF, note_type=MALFORMED)))
    assert "[blink]" in out and "[/cyan]" in out


def test_summary_row_tag_rendered_literally_in_table() -> None:
    note = _note(tags=[SPOOF], deck=MALFORMED)
    row = output.note_summary_row(note)
    out = _capture(lambda: output.table(["ID", "Type", "Deck", "Tags", "Modified"], [row]))
    assert "[blink]" in out and "[/cyan]" in out


def test_note_type_detail_template_and_field_rendered_literally() -> None:
    nt = NoteTypeInfo(
        name=SPOOF,
        id=1,
        fields=[MALFORMED],
        type="standard",
        detail=NoteTypeDetail(
            templates=[TemplateInfo(name=SPOOF, front="Front " + MALFORMED, back=SPOOF)],
            css="/* " + MALFORMED + " */",
            fields=[FieldDetail(name=SPOOF, font="Arial", size=20, description=MALFORMED)],
        ),
    )
    out = _capture(lambda: output.note_type_detail(nt))
    # Both the spoof markup and the malformed closing tag survive literally.
    assert "[blink]" in out and "[/cyan]" in out


def test_result_status_error_rendered_literally() -> None:
    res = UpsertNoteError(status="error", index=0, error=MALFORMED)
    out = _capture(lambda: output.result_status([res]))
    assert "[/cyan]" in out


def test_error_helper_escapes_message() -> None:
    buf = io.StringIO()
    cap = Console(file=buf, force_terminal=True, color_system="standard", highlight=False)
    orig = output.err_console
    output.err_console = cap
    try:
        output.error(MALFORMED)
    except MarkupError as err:  # pragma: no cover - asserted as a failure
        pytest.fail(f"error() raised an uncaught MarkupError on data: {err}")
    finally:
        output.err_console = orig
    assert "[/cyan]" in buf.getvalue()


def test_success_helper_escapes_message() -> None:
    out = _capture(lambda: output.success(SPOOF))
    assert "[blink]" in out


def test_search_match_detail_does_not_crash() -> None:
    # The search render uses note_detail for the full (non-brief) view; a snippet
    # and field content with a stray tag must not crash it.
    match = SearchMatch(
        id=5,
        note_type="Basic",
        deck=MALFORMED,
        tags=[SPOOF],
        modified="2026-01-01T00:00:00",
        content={"Front": "Q", "Back": MALFORMED},
        substring=SubstringInfo(matched_fields=["Back"], snippet=MALFORMED),
    )
    out = _capture(lambda: output.note_detail(match, subtitle="[exact]"))
    assert "[blink]" in out and "[/cyan]" in out


def test_media_list_filename_rendered_literally_in_table() -> None:
    # A bracket-bearing media filename reproduces the same crash (broadened scope).
    resp = ListMediaResponse(
        media_dir="/m",
        count=1,
        files=[MediaFileInfo(filename=MALFORMED + ".png", mime="image/png", size_bytes=10)],
    )
    # media_list builds rows then calls output.table; mirror that here.
    rows = [[output.esc(f.filename), output.esc(f.mime or ""), "10 B"] for f in resp.files]
    out = _capture(lambda: output.table(["Name", "Type", "Size"], rows))
    assert "[/cyan]" in out


def test_esc_neutralizes_markup() -> None:
    # The centralized helper turns brackets into literal text.
    assert output.esc(SPOOF) == r"\[blink]gotcha\[/blink]"
    assert "[" not in output.esc(MALFORMED).replace("\\[", "")


# --- End-to-end CLI adversarial cases (the joint-review must-fix) -------------
#
# Drive the real CLI commands (`info --decks/--tags`, `collection check`) with a
# stubbed client returning bracket-bearing collection content. These cover the
# sibling render files (info_cmd/collection_cmd) the first sweep missed. Each
# runs with `catch_exceptions=False` so an uncaught MarkupError fails the test,
# and asserts the bracketed content survives literally (no restyle).


@pytest.fixture
def cli_run(tmp_path):
    """Invoke the real CLI with a MagicMock ShrikeClient and an empty config."""
    fake = MagicMock(spec=shrike_client.ShrikeClient)
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str):
        with patch("shrike.client.ShrikeClient", return_value=fake):
            return runner.invoke(cli, ["--config", str(cfg), *args], catch_exceptions=False)

    _run.fake = fake  # type: ignore[attr-defined]
    return _run


def test_info_decks_renders_bracketed_deck_name_literally(cli_run) -> None:
    # Case 1: a deck named "[red]PWNED[/red]" must render literally (no restyle).
    cli_run.fake.collection_info.return_value = CollectionInfo(
        decks=[DeckInfo(name="[red]PWNED[/red]", id=1, note_count=3)]
    )
    result = cli_run("info", "--decks")
    assert result.exit_code == 0, result.output
    assert "[red]PWNED[/red]" in result.output


def test_info_decks_and_tags_malformed_markup_does_not_crash(cli_run) -> None:
    # Case 2: a deck "Deck [/notreal]" and a tag "[/x]" must not raise MarkupError.
    cli_run.fake.collection_info.return_value = CollectionInfo(
        decks=[DeckInfo(name="Deck [/notreal]", id=1, note_count=0)],
        tags=["[/x]"],
    )
    decks = cli_run("info", "--decks")
    assert decks.exit_code == 0, decks.output
    assert "[/notreal]" in decks.output

    tags = cli_run("info", "--tags")
    assert tags.exit_code == 0, tags.output
    assert "[/x]" in tags.output


def test_collection_check_renders_bracketed_filename_literally(cli_run) -> None:
    # Case 3: an unused media file "a[/z].png" must render literally, no crash.
    cli_run.fake.collection_check.return_value = CollectionCheckResponse(
        media_dir="/media",
        unused=["a[/z].png"],
    )
    result = cli_run("collection", "check")
    assert result.exit_code == 0, result.output
    assert "a[/z].png" in result.output
