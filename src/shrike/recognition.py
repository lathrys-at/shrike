"""Recognition backends (#228/#221): the harness side of the kernel's
``Recognizer`` seam.

A backend satisfies the ``RecognizerBackend`` protocol — a *blocking*
``recognize(items)`` (the kernel dispatches it to asyncio's thread pool, never
the collection executor) returning one ``(text, confidence, segments_json)``
tuple per item, plus a ``model_fingerprint()`` whose change invalidates all
derived text on the next sweep. The kernel never knows which engine this is;
``Harness.attach_recognizer`` captures any conforming object.

``AppleVisionBackend`` is the first engine (#221): Apple's Vision framework
via pyobjc (the ``shrike[vision]`` extra, macOS only) — local, deterministic,
no model download, and it emits per-line confidence + normalized boxes, which
is exactly the one-pass text+positions contract (#230 reads the boxes back).
"""

from __future__ import annotations

import json
import logging
import platform
from typing import Any, Protocol

logger = logging.getLogger("shrike.recognition")

# Vision's request revision is folded into the fingerprint: an OS upgrade that
# changes recognition output should re-derive text, exactly like an embedding
# model change rebuilds vectors.
OCR_BACKENDS = ("apple",)


class RecognizerBackend(Protocol):
    """The wire contract `PyRecognizer.capture` expects (see py_recognizer.rs)."""

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]: ...

    def model_fingerprint(self) -> str | None: ...


def make_recognizer(kind: str) -> RecognizerBackend:
    """Construct a recognition backend by kind (the `_make_backend` pattern):
    imports are lazy so the extras stay optional — a missing dependency
    surfaces as ImportError only when that backend is selected."""
    if kind == "apple":
        return AppleVisionBackend()
    raise ValueError(f"unknown OCR backend {kind!r} (choices: {', '.join(OCR_BACKENDS)})")


class AppleVisionBackend:
    """OCR via Apple's Vision framework (macOS, `shrike[vision]`).

    Each image runs one ``VNRecognizeTextRequest`` (accurate level, language
    correction on). Per-observation candidates carry text + confidence +
    a normalized bounding box, which Vision reports in a bottom-left origin —
    converted here to the top-left ``[x, y, w, h]`` the segments contract
    uses. The flattened text joins lines in Vision's reading order; the
    overall confidence is the mean of line confidences (0 for no text).
    """

    def __init__(self) -> None:
        import Vision  # noqa: F401 — fail at construction, not first use

        self._vision = Vision

    def model_fingerprint(self) -> str | None:
        revision = self._vision.VNRecognizeTextRequestRevision3
        return f"apple-vision:rev{revision}:macos{platform.mac_ver()[0]}"

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        return [self._recognize_one(data) for data in items]

    def _recognize_one(self, data: bytes) -> tuple[str, float, str]:
        vision = self._vision
        import Foundation

        if not data:
            return ("", 0.0, "")
        ns_data = Foundation.NSData.dataWithBytes_length_(data, len(data))
        handler = vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        request = vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            logger.warning("Vision request failed: %s", error)
            return ("", 0.0, "")

        lines: list[str] = []
        segments: list[dict[str, Any]] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = str(candidate.string())
            confidence = float(candidate.confidence())
            box = observation.boundingBox()
            # Vision: normalized, origin bottom-left → top-left [x, y, w, h].
            x = float(box.origin.x)
            w = float(box.size.width)
            h = float(box.size.height)
            y = 1.0 - float(box.origin.y) - h
            lines.append(text)
            segments.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
                }
            )
        if not lines:
            return ("", 0.0, "")
        overall = sum(s["confidence"] for s in segments) / len(segments)
        return ("\n".join(lines), overall, json.dumps(segments))
