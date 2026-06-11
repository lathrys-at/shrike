"""Apple Vision recognition tests (#221, native since #342 P3) — macOS gated.

These render text with Pillow and OCR it through the real Vision framework
via the native engine (`shrike_native.AppleVisionRecognizer`), so they're
skipped off macOS; CI exercises them on the macOS cross-platform lane. The
kernel-side seam and the gating policy are covered backend-free in the Rust +
native suites; the Rust crate carries its own fixture-driven live tests.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision is macOS-only")

PIL = pytest.importorskip("PIL")

import shrike_native  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from shrike.recognition import make_recognizer  # noqa: E402


def _render(text: str, size: tuple[int, int] = (640, 120)) -> bytes:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 40), text, fill="black", font_size=28)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_reads_rendered_text_with_box_and_confidence():
    backend = make_recognizer("apple")
    text, confidence, segments_json = backend.recognize([_render("electron transport chain")])[0]

    assert "electron transport chain" in text.lower()
    assert 0.0 < confidence <= 1.0
    segments = json.loads(segments_json)
    assert segments, "per-line segments are retained (the one-pass contract)"
    box = segments[0]["bbox"]
    assert len(box) == 4
    # Normalized, top-left origin: every coordinate in [0, 1].
    assert all(0.0 <= v <= 1.0 for v in box)


def test_empty_and_blank_images_are_zero_confidence():
    backend = make_recognizer("apple")
    # Empty bytes → empty recognition, never an exception.
    assert backend.recognize([b""])[0] == ("", 0.0, "")
    # A blank canvas → no text, zero confidence, no segments.
    blank = io.BytesIO()
    Image.new("RGB", (200, 80), "white").save(blank, format="PNG")
    text, confidence, segments = backend.recognize([blank.getvalue()])[0]
    assert text == "" and confidence == 0.0 and segments == ""


def test_batch_preserves_order():
    backend = make_recognizer("apple")
    results = backend.recognize([_render("alpha first line"), _render("beta second line")])
    assert "alpha" in results[0][0].lower()
    assert "beta" in results[1][0].lower()


def test_fingerprint_is_stable_and_versioned():
    backend = make_recognizer("apple")
    fp = backend.model_fingerprint()
    assert fp is not None
    assert fp.startswith("apple-vision:")
    assert fp == backend.model_fingerprint()  # stable


def test_make_recognizer_selects_apple():
    assert isinstance(make_recognizer("apple"), shrike_native.AppleVisionRecognizer)
    with pytest.raises(ValueError, match="unknown OCR backend"):
        make_recognizer("nope")
