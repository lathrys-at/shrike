"""Apple Vision recognition tests (#221, native since #342 P3).

The Vision tests render text with Pillow and OCR it through the real Vision
framework via the native engine (`shrike_native.AppleVisionRecognizer`), so
they're skipped off macOS — and, since the #496 boundary enforcement, on the
server build entirely (platform engines are mobile-only; the engine is absent
from default builds — re-homing this coverage onto an engine-apple test build
is #514). The kernel-side seam and the gating policy are covered backend-free
in the Rust + native suites; the Rust crate carries its own fixture-driven
live tests. Two tests DO run on the server build: the unknown-kind error and
the clean degrade when the engine isn't compiled in.
"""

from __future__ import annotations

import io
import json
import sys

import pytest
import shrike_native

from shrike.recognition import make_recognizer

_HAS_ENGINE = hasattr(shrike_native, "AppleVisionRecognizer")

# The Vision-engine tests need macOS AND a build carrying the engine.
requires_vision = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision is macOS-only"),
    pytest.mark.skipif(
        not _HAS_ENGINE,
        reason="engine-apple not compiled into this build (mobile-only since #496; "
        "test re-homing is #514)",
    ),
]

PIL = pytest.importorskip("PIL")

from PIL import Image, ImageDraw  # noqa: E402


def _render(text: str, size: tuple[int, int] = (640, 120)) -> bytes:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 40), text, fill="black", font_size=28)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestVisionEngine:
    """The live-engine half — engine-apple builds on macOS only."""

    pytestmark = requires_vision

    def test_reads_rendered_text_with_box_and_confidence(self):
        backend = make_recognizer("apple")
        text, confidence, segments_json = backend.recognize([_render("electron transport chain")])[
            0
        ]

        assert "electron transport chain" in text.lower()
        assert 0.0 < confidence <= 1.0
        segments = json.loads(segments_json)
        assert segments, "per-line segments are retained (the one-pass contract)"
        box = segments[0]["bbox"]
        assert len(box) == 4
        # Normalized, top-left origin: every coordinate in [0, 1].
        assert all(0.0 <= v <= 1.0 for v in box)

    def test_empty_and_blank_images_are_zero_confidence(self):
        backend = make_recognizer("apple")
        # Empty bytes → empty recognition, never an exception.
        assert backend.recognize([b""])[0] == ("", 0.0, "")
        # A blank canvas → no text, zero confidence, no segments.
        blank = io.BytesIO()
        Image.new("RGB", (200, 80), "white").save(blank, format="PNG")
        text, confidence, segments = backend.recognize([blank.getvalue()])[0]
        assert text == "" and confidence == 0.0 and segments == ""

    def test_batch_preserves_order(self):
        backend = make_recognizer("apple")
        results = backend.recognize([_render("alpha first line"), _render("beta second line")])
        assert "alpha" in results[0][0].lower()
        assert "beta" in results[1][0].lower()

    def test_fingerprint_is_stable_and_versioned(self):
        backend = make_recognizer("apple")
        fp = backend.model_fingerprint()
        assert fp is not None
        assert fp.startswith("apple-vision-swift:")
        assert fp == backend.model_fingerprint()  # stable

    def test_make_recognizer_selects_apple(self):
        assert isinstance(make_recognizer("apple"), shrike_native.AppleVisionRecognizer)


def test_make_recognizer_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown OCR backend"):
        make_recognizer("nope")


@pytest.mark.skipif(_HAS_ENGINE, reason="needs a build WITHOUT engine-apple (the server build)")
def test_make_recognizer_without_engine_errors_cleanly():
    # The #496 boundary: the server build doesn't compile platform engines, so
    # selecting `apple` degrades exactly like a missing optional dependency —
    # an ImportError the boot path catches (recognition state `error`, boot
    # undisturbed) — with a message pointing at the replacement path.
    with pytest.raises(ImportError, match="#502"):
        make_recognizer("apple")
