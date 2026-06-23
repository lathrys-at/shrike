"""Untrusted note/media content must render literally, never as Rich markup.

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

import click
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
    UpsertNoteOk,
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


# --- End-to-end CLI adversarial cases -----------------------------------------
#
# Drive the real CLI commands (`info --decks/--tags`, `collection check`) with a
# stubbed client returning bracket-bearing collection content. These cover the
# sibling render files (info_cmd/collection_cmd). Each runs with
# `catch_exceptions=False` so an uncaught MarkupError fails the test, and asserts
# the bracketed content survives literally (no restyle).


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
    result = cli_run("collection", "info", "--decks")
    assert result.exit_code == 0, result.output
    assert "[red]PWNED[/red]" in result.output


def test_info_decks_and_tags_malformed_markup_does_not_crash(cli_run) -> None:
    # Case 2: a deck "Deck [/notreal]" and a tag "[/x]" must not raise MarkupError.
    cli_run.fake.collection_info.return_value = CollectionInfo(
        decks=[DeckInfo(name="Deck [/notreal]", id=1, note_count=0)],
        tags=["[/x]"],
    )
    decks = cli_run("collection", "info", "--decks")
    assert decks.exit_code == 0, decks.output
    assert "[/notreal]" in decks.output

    tags = cli_run("collection", "info", "--tags")
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


# --- --json/--pretty mutual exclusion is order-independent --------------------
#
# The mutual-exclusion guard must reject `--json --pretty` regardless of which
# flag comes first. The eager callbacks fire in token order, so the guard must
# check the other flag from both, or `--pretty --json` silently lets --json win.

_MUTEX = "--pretty and --json are mutually exclusive"


@pytest.mark.parametrize(
    "flags",
    [
        ["--json", "--pretty"],
        ["--pretty", "--json"],  # was silently accepted before the fix
    ],
)
def test_json_pretty_conflict_errors_regardless_of_order(cli_run, flags) -> None:
    # A valid response is stubbed so the command would otherwise succeed — the
    # only thing that can make it fail is the mutual-exclusion guard itself.
    cli_run.fake.collection_info.return_value = CollectionInfo()
    result = cli_run("collection", "info", *flags)
    assert result.exit_code != 0, result.output
    assert _MUTEX in result.output


@pytest.mark.parametrize(
    "flags",
    [
        ["--json", "--no-pretty"],
        ["--no-pretty", "--json"],
        ["--json"],
    ],
)
def test_json_with_no_pretty_is_allowed(cli_run, flags) -> None:
    # --json + --no-pretty both mean "no styling": compatible, never an error.
    cli_run.fake.collection_info.return_value = CollectionInfo()
    result = cli_run("collection", "info", *flags)
    assert result.exit_code == 0, result.output
    assert _MUTEX not in result.output


# --- NoteIDType identifier parsing --------------------------------------------
#
# The `#id`/name-or-id edge: the param type accepts a bare int, a "#"-prefixed
# string, and rejects non-numeric input with a Click failure.


def test_note_id_passes_int_through() -> None:
    assert output.NOTE_ID.convert(42, None, None) == 42


def test_note_id_strips_hash_prefix() -> None:
    assert output.NOTE_ID.convert("#170000123", None, None) == 170000123


def test_note_id_rejects_non_numeric() -> None:
    with pytest.raises(click.exceptions.BadParameter):
        output.NOTE_ID.convert("notanid", None, None)


def test_note_id_coerces_non_str_non_int_via_int() -> None:
    # A value that is neither int nor str falls through to int() — a float with
    # an integral value coerces (the defensive non-int/non-str branch).
    assert output.NOTE_ID.convert(7.0, None, None) == 7


# --- table / template / result_status render branches -------------------------


def test_table_empty_rows_renders_none_placeholder() -> None:
    out = _capture(lambda: output.table(["A", "B"], []))
    assert "(none)" in out


def test_append_template_field_multiline_indents_each_line() -> None:
    # A multiline template value takes the indented-block branch (not inline).
    lines: list[str] = []
    output._append_template_field(lines, "Front", "line one\nline two")
    # The label is on its own line; each content line is indented separately.
    assert "Front:" in lines[0]
    assert "line one" not in lines[0]  # value is NOT inline on the label line
    assert any("line one" in line for line in lines[1:])
    assert any("line two" in line for line in lines[1:])


def test_append_template_field_single_line_inlines_value() -> None:
    # The single-line branch keeps the value on the label line.
    lines: list[str] = []
    output._append_template_field(lines, "Back", "just one")
    assert len(lines) == 1
    assert "Back:" in lines[0] and "just one" in lines[0]


def test_note_type_table_renders_header_and_rows() -> None:
    nts = [
        NoteTypeInfo(name="Basic", id=10, type="standard", fields=["Front", "Back"]),
        NoteTypeInfo(name="Cloze", id=20, type="cloze", fields=["Text"]),
    ]
    out = _capture(lambda: output.note_type_table(nts, "/c.anki2"))
    assert "Showing 2 note type(s)" in out
    assert "Basic" in out and "Cloze" in out
    assert "#10" in out and "#20" in out


def test_note_type_detail_without_detail_renders_only_summary() -> None:
    # detail is None → the templates/CSS block is skipped (the negative branch).
    nt = NoteTypeInfo(name="Basic", id=10, type="standard", fields=["Front", "Back"])
    out = _capture(lambda: output.note_type_detail(nt))
    assert "Basic" in out
    assert "Templates" not in out
    assert "CSS" not in out


def test_note_type_detail_with_single_line_template_inlines_it() -> None:
    # detail present, no field metadata, single-line template fronts/backs:
    # exercises the no-fields branch and the inline (single-line) template path.
    nt = NoteTypeInfo(
        name="Basic",
        id=10,
        type="standard",
        fields=["Front", "Back"],
        detail=NoteTypeDetail(
            fields=[],
            templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
            css=".card { color: red; }",
        ),
    )
    out = _capture(lambda: output.note_type_detail(nt))
    assert "Templates" in out
    assert "Card 1" in out
    assert "CSS" in out
    # No per-field metadata header when detail.fields is empty.
    assert "Fields\n" not in out or "Front" in out


def test_result_status_created_updated_and_unknown() -> None:
    out = _capture(
        lambda: output.result_status(
            [
                UpsertNoteOk(status="created", id=1),
                UpsertNoteOk(status="updated", id=2),
            ]
        )
    )
    assert "Created note" in out and "#1" in out
    assert "Updated note" in out and "#2" in out


def test_result_status_unknown_variant_str_fallback() -> None:
    # A result whose status is none of created/updated/error hits the `else`
    # branch and is rendered via str(); UpsertNoteSkipped is exactly that case.
    from shrike.schemas import UpsertNoteSkipped

    skipped = UpsertNoteSkipped(status="skipped", index=0, reason="duplicate")
    out = _capture(lambda: output.result_status([skipped]))
    assert "skipped" in out or "duplicate" in out


# --- small render helpers + JSON conversion -----------------------------------


def test_parse_comma_separated_splits_and_strips() -> None:
    # `--tags a, b ,,c` + a repeat → a flat, stripped, empty-dropped tuple.
    out = output.parse_comma_separated(None, None, ("a, b ,,c", "d"))
    assert out == ("a", "b", "c", "d")


def test_to_jsonable_recurses_lists_and_passes_plain_data() -> None:
    note = _note(id=3)
    out = output._to_jsonable([note, {"plain": 1}])
    assert out[0]["id"] == 3
    assert out[1] == {"plain": 1}


def test_section_prints_title() -> None:
    out = _capture(lambda: output.section("My Section"))
    assert "My Section" in out


def test_kv_prints_label_and_value_with_indent() -> None:
    out = _capture(lambda: output.kv("Label", "value", indent=2))
    assert "Label:" in out and "value" in out


def test_note_summary_row_keeps_date_without_t_separator() -> None:
    # A modified value with no "T" is rendered verbatim (the no-split branch).
    row = output.note_summary_row(_note(modified="2026-01-02"))
    assert row[-1] == "2026-01-02"


def test_note_type_detail_field_without_description_omits_quote() -> None:
    # A field-detail with no description skips the trailing ` · "…"` (the
    # negative description branch).
    nt = NoteTypeInfo(
        name="Basic",
        id=10,
        type="standard",
        fields=["Front"],
        detail=NoteTypeDetail(
            fields=[FieldDetail(name="Front", font="Arial", size=20, description="")],
            templates=[TemplateInfo(name="Card 1", front="{{Front}}", back="{{Back}}")],
            css="",
        ),
    )
    out = _capture(lambda: output.note_type_detail(nt))
    assert "Front" in out
    assert '"' not in out  # no description quote emitted
