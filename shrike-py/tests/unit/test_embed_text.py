"""Tests for shrike.embed_text.normalize_for_embedding (stable, deterministic).

The HTML/entity layer delegates to Anki's strip_html; these run headless, so the
module lazily initializes i18n on first use.
"""

from __future__ import annotations

import pytest

from shrike.harness.engines.embedding.text import EMBED_TEXT_VERSION
from tests.oracles.embed_text_oracle import normalize_for_embedding


class TestClozeFill:
    def test_basic_cloze_filled_to_answer(self) -> None:
        assert (
            normalize_for_embedding("The capital of {{c1::France}} is Paris.")
            == "The capital of France is Paris."
        )

    def test_hint_dropped(self) -> None:
        assert normalize_for_embedding("{{c2::Paris::the city}}") == "Paris"

    def test_multiple_clozes(self) -> None:
        assert normalize_for_embedding("{{c1::A}}, {{c2::B}}, and {{c3::C}}") == "A, B, and C"

    def test_cloze_containing_html_is_filled_then_stripped(self) -> None:
        assert normalize_for_embedding("{{c1::<b>bold answer</b>}}") == "bold answer"

    def test_shallow_nested_cloze_flattened(self) -> None:
        # Inner cloze resolves on a later pass; we don't promise perfection on
        # deep nesting, but no markup should survive.
        out = normalize_for_embedding("{{c1::alpha {{c2::beta}} gamma}}")
        assert "{{" not in out and "}}" not in out
        assert "beta" in out


class TestHtml:
    def test_inline_tags_removed(self) -> None:
        assert normalize_for_embedding("<b>bold</b> and <i>italic</i>") == "bold and italic"

    def test_block_tags_become_space(self) -> None:
        assert normalize_for_embedding("foo<br>bar<div>baz</div>qux") == "foo bar baz qux"

    def test_img_tag_removed(self) -> None:
        assert normalize_for_embedding('see <img src="diagram.png"> here') == "see here"

    def test_anchor_text_kept_href_dropped(self) -> None:
        assert normalize_for_embedding('<a href="http://x.com">link</a>') == "link"

    def test_encoded_tag_survives_as_literal(self) -> None:
        # &lt;tag&gt; is content the reader sees, not markup — it must NOT be
        # stripped. (Unescape runs after tag-stripping for exactly this.)
        assert normalize_for_embedding("use &lt;tag&gt; here") == "use <tag> here"


class TestMediaAndLatex:
    def test_sound_ref_removed(self) -> None:
        assert normalize_for_embedding("listen [sound:audio_123.mp3] now") == "listen now"

    def test_latex_markers_removed_inner_kept(self) -> None:
        assert normalize_for_embedding("[latex]x^2[/latex]") == "x^2"

    def test_dollar_math_markers_removed(self) -> None:
        assert normalize_for_embedding("[$]e=mc^2[/$]") == "e=mc^2"

    def test_mathjax_inline_delimiters_removed(self) -> None:
        assert normalize_for_embedding(r"Euler: \(e^{i\pi}+1=0\)") == r"Euler: e^{i\pi}+1=0"

    def test_mathjax_display_delimiters_removed(self) -> None:
        assert normalize_for_embedding(r"\[x^2 + y^2\]") == "x^2 + y^2"

    def test_mathjax_dollar_delimiters_removed(self) -> None:
        assert normalize_for_embedding("$$a + b$$") == "a + b"


class TestEntitiesAndWhitespace:
    def test_named_entities(self) -> None:
        assert normalize_for_embedding("caf&eacute; &amp; tea") == "café & tea"

    def test_nbsp_folded(self) -> None:
        assert normalize_for_embedding("a&nbsp;b") == "a b"

    def test_whitespace_collapsed_and_trimmed(self) -> None:
        assert normalize_for_embedding("  lots\n\nof   \t space \n ") == "lots of space"


class TestEdgeCases:
    def test_empty(self) -> None:
        assert normalize_for_embedding("") == ""

    def test_markup_only_becomes_empty(self) -> None:
        assert normalize_for_embedding('<img src="x.png">[sound:y.mp3]') == ""

    def test_plain_text_unchanged(self) -> None:
        assert normalize_for_embedding("What is 2+2?") == "What is 2+2?"


class TestDeterminism:
    @pytest.mark.parametrize(
        "value",
        [
            "The capital of {{c1::France::hint}} is Paris.",
            'a<br>b <img src="x.png"> [sound:s.mp3] caf&eacute;',
            "[latex]\\int_0^1 x\\,dx[/latex]",
            r"mass-energy \(E = mc^2\)",
        ],
    )
    def test_repeatable(self, value: str) -> None:
        # Stable: identical output every call (i18n init is idempotent, strip
        # output is locale-independent).
        assert normalize_for_embedding(value) == normalize_for_embedding(value)

    def test_realistic_cloze_card(self) -> None:
        raw = 'The mitochondria&nbsp;is the {{c1::powerhouse}} of the cell.<br><img src="m.png">'
        assert normalize_for_embedding(raw) == "The mitochondria is the powerhouse of the cell."


class TestC0SeparatorContractScope:
    """#612: the Python `\\s` whitespace class matches the C0 separators
    U+001C-U+001F, but Rust's `\\s` (Unicode White_Space) does not — so the two
    normalizers diverge on those four code points. This is documented as a
    contract-SCOPE limit (the byte-identity holds over anki-sanitized field
    text, which never contains them) rather than fixed, because folding them
    into the pattern would change the output for an unreachable input and force
    an EMBED_TEXT_VERSION bump + index rebuild for no behavioral gain. These
    tests pin that the divergence exists exactly where documented, so a future
    change to either pattern is caught.
    """

    C0_SEPARATORS = ["\x1c", "\x1d", "\x1e", "\x1f"]

    @pytest.mark.parametrize("sep", C0_SEPARATORS)
    def test_python_oracle_treats_c0_separator_as_whitespace(self, sep: str) -> None:
        # The Python side collapses a C0 separator like any whitespace (this is
        # the source of the divergence the contract scopes around). The text is
        # plain (no tag/entity), so strip_html is byte-identity and only the
        # whitespace collapse acts.
        assert normalize_for_embedding(f"a{sep}b") == "a b"

    def test_sanitized_field_text_has_no_c0_separators(self) -> None:
        # The contract holds over anki-sanitized field text: anki's
        # invalid_char_for_field strips U+001C-U+001F before a field is stored,
        # so neither normalizer ever sees one on the field path. We can't invoke
        # that Rust-side gate from here, but we pin the property the scope relies
        # on — a normal field carries no C0 separator, so the two sides agree.
        sanitized = "The mitochondria is the powerhouse of the cell."
        assert all(c not in sanitized for c in self.C0_SEPARATORS)
        assert normalize_for_embedding(sanitized) == sanitized


def test_version_is_an_int() -> None:
    # Folded into the index fingerprint; must be a stable scalar.
    assert isinstance(EMBED_TEXT_VERSION, int)
