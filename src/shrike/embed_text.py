"""Stable note-text normalization for embedding.

The text fed to the embedder must be a function of a note's stored field values
that is stable over time — identical whether a note was just upserted or re-read
during a full rebuild, and independent of which card a cloze note happens to
generate. Anki field values are messy: HTML, media refs, cloze markup, math.
Embedding them raw makes ``The capital of {{c1::France}} is Paris`` land far from
its rendered meaning, and garbage here propagates through every search and
neighbor lookup downstream.

We normalize each field value to plain rendered text. The HTML→text + entity
step delegates to **Anki's own (Rust-backed) ``strip_html``** — far more robust
on malformed/exotic markup than a regex, and exactly what Anki itself uses (it
even leaves an *encoded* tag like ``&lt;tag&gt;`` as the literal ``<tag>`` the
reader sees). We handle the parts Anki's stripper doesn't, *around* it:

* **before** — reveal cloze (``{{c1::ans::hint}}`` → ``ans``; Anki's stripper
  leaves cloze markup alone, since filling it is a template-render concern);
  drop MathJax/LaTeX wrappers keeping the inner source; drop ``[sound:…]``; and
  convert block tags (``<br>``, ``<div>``, …) to spaces — Anki's stripper
  *glues* text across them (``a<br>b`` → ``ab``), which is wrong for embedding;
* Anki's ``strip_html`` then removes the remaining (inline) tags and ``<img>``
  and unescapes entities;
* **after** — fold non-breaking spaces and collapse whitespace.

Operating on field *values* (not rendered cards) is deliberate: a note is the
embedding unit, a cloze note generates N cards, and card templates add
presentational scaffolding (``{{FrontSide}}``, ``<hr id=answer>``, hidden
``[...]`` on the cloze question side) that is noise for semantic search.

The output is a function of the field value plus the pinned Anki version's
stripper. ``EMBED_TEXT_VERSION`` is folded into the index fingerprint
(``model_id``), so any change to this output — ours *or* an Anki-version bump
that alters stripping — should bump the version and force a one-time index
rebuild rather than silently mixing old- and new-style vectors.
"""

from __future__ import annotations

import re

# Bump on any change to normalize_for_embedding's output (incl. an Anki upgrade
# whose stripper changes). v1 is the first normalized scheme — it replaces the
# original raw field concatenation, which had no version token, so existing
# indexes mismatch and rebuild once on upgrade.
EMBED_TEXT_VERSION = 1

# {{c1::answer}} / {{c1::answer::hint}}. Content excludes "{{"/"}}", so an
# innermost cloze matches first and an outer one only once its inner ones are
# resolved; iterating flattens nesting. The hint (after the first "::") is dropped.
_CLOZE_RE = re.compile(r"\{\{c\d+::((?:(?!\{\{|\}\}).)*)\}\}", re.DOTALL)
# [sound:file.mp3]
_SOUND_RE = re.compile(r"\[sound:[^\]]*\]", re.IGNORECASE)
# Legacy image-LaTeX wrappers: [latex] [/latex] [$] [/$] [$$] [/$$] — markers
# only, inner source kept.
_LATEX_RE = re.compile(r"\[/?(?:latex|\$\$?)\]", re.IGNORECASE)
# MathJax delimiters: \( \) \[ \] and $$ — drop the delimiters, keep the inner
# LaTeX source as weak signal.
_MATHJAX_RE = re.compile(r"\\[()\[\]]|\$\$")
# Block-level tags and <br> → whitespace, so "a<br>b" doesn't become "ab" once
# Anki's stripper (which glues across block tags) runs.
_BLOCK_TAG_RE = re.compile(r"(?i)<\s*/?\s*(?:br|div|p|li|ul|ol|tr|td|h[1-6]|blockquote)\b[^>]*>")
_WS_RE = re.compile(r"\s+")


def _fill_clozes(text: str) -> str:
    """Replace cloze deletions with their answer text, dropping wrapper + hint.

    Iterates to flatten shallow nesting; bounded so a pathological input can't
    spin. Deeply nested clozes are vanishingly rare and only affect embedding
    quality, never correctness.
    """
    for _ in range(10):
        new = _CLOZE_RE.sub(lambda m: m.group(1).split("::", 1)[0], text)
        if new == text:
            return new
        text = new
    return text


def _strip_html(text: str) -> str:
    """Anki's Rust-backed HTML→text (NORMAL: drops tags + <img>, unescapes entities)."""
    import anki.lang
    from anki.utils import strip_html

    if anki.lang.current_i18n is None:
        # Initialize i18n once so the stripper works headless. "en" only selects
        # message catalogs; it does not affect HTML-stripping output. Guard so we
        # never clobber a language a host app already configured.
        anki.lang.set_lang("en")
    return strip_html(text)


def normalize_for_embedding(value: str) -> str:
    """Turn one raw Anki field value into stable plain text for embedding.

    Returns ``""`` for empty input or a value that was nothing but markup/media.
    """
    if not value:
        return ""
    text = _fill_clozes(value)
    text = _LATEX_RE.sub(" ", text)
    text = _MATHJAX_RE.sub(" ", text)
    text = _SOUND_RE.sub(" ", text)
    text = _BLOCK_TAG_RE.sub(" ", text)
    text = _strip_html(text)
    text = text.replace("\xa0", " ")
    return _WS_RE.sub(" ", text).strip()
