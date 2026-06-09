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

import contextlib
import re
from html.parser import HTMLParser

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


# Media references that make a field non-empty even with no text: <img>, <audio>,
# <video>, <object>, <embed>, <source>, and [sound:…]. Used by collection_prune's
# empty-note rule (#89), not by embedding (which strips media out entirely).
_MEDIA_RE = re.compile(r"(?i)<\s*(?:img|audio|video|object|embed|source)\b|\[sound:")


class _ImgSrcParser(HTMLParser):
    """Collect ``<img>`` ``src`` attribute values. A real parser (not a regex) so an earlier
    ``data-src=`` or a ``src=`` *inside another attribute's value* can't be mistaken for the tag's
    own ``src`` — the failure modes a regex over ``<img …>`` hits on lazy-load / web-pasted markup.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            for name, val in attrs:
                if name == "src" and val:
                    self.srcs.append(val)


def extract_image_refs(value: str) -> list[str]:
    """Image filenames referenced by a field's ``<img src>`` tags — in order, de-duplicated.

    Returns each src's basename (the flat form ``store_media`` returns and the media dir keys on);
    remote srcs (``scheme://…``) are skipped (not local media). Only ``<img>`` — the embeddable
    image modality; ``[sound:]``/``<audio>``/``<video>`` are other modalities for a later slice.
    The collection resolves these names to bytes via the media dir before handing them to a
    CLIP-style backend; ``normalize_for_embedding`` still strips ``<img>`` out of the *text*.
    """
    if not value or "<img" not in value.lower():
        return []
    parser = _ImgSrcParser()
    with contextlib.suppress(Exception):  # malformed markup must not break extraction
        parser.feed(value)
    names: list[str] = []
    seen: set[str] = set()
    for raw in parser.srcs:
        src = raw.strip()  # HTMLParser already unescapes entities in attribute values
        if not src or "://" in src:  # empty or a remote URL → not local media
            continue
        name = src.rsplit("/", 1)[-1]  # basename; Anki's media dir is flat
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def field_is_blank(value: str) -> bool:
    """True if a field value carries no content — no text and no media.

    A field is blank when it strips to nothing (HTML removed, ``&nbsp;`` and
    whitespace folded) **and** references no media. An image- or audio-only field
    is therefore *not* blank, so a note made only of media is never treated as
    empty. This is the per-field rule behind removing empty notes; it is stricter
    than ``normalize_for_embedding`` (which deliberately drops media to ``""``).
    """
    if not value:
        return True
    if _MEDIA_RE.search(value):
        return False
    return not _strip_html(value).replace("\xa0", " ").strip()
