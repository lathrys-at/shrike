"""Recognition backends (#228/#221): config→construct for the kernel's
``Recognizer`` slot.

Since the engine-plugin migration (#342 P3) the Apple Vision engine is native
(`shrike-recognize-apple`; since #398 its platform glue is Swift behind Rust,
driving Apple's Swift-only ``RecognizeTextRequest`` API — nothing extra to
install at runtime; Vision and the Swift runtime ship with macOS).
``make_recognizer`` constructs the native backend object;
``Harness.attach_recognizer`` hands it to the kernel, where recognition runs
native end-to-end, Python never on the sweep path.

A *custom* backend remains first-class: any object satisfying
``RecognizerBackend`` — a blocking ``recognize(items)`` returning one
``(text, confidence, segments_json)`` tuple per item plus a
``model_fingerprint()`` — attaches through the ``PyRecognizer`` capture seam
instead (the kernel dispatches it to asyncio's thread pool).
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger("shrike.recognition")

OCR_BACKENDS = ("apple",)


class RecognizerBackend(Protocol):
    """The wire contract `PyRecognizer.capture` expects (see py_recognizer.rs)."""

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]: ...

    def model_fingerprint(self) -> str | None: ...


def make_recognizer(kind: str) -> RecognizerBackend:
    """Construct a recognition backend by kind (the `_make_backend` pattern).
    Unavailability — the engine not compiled into this build (platform
    engines are mobile-only since the #496 boundary), or the native engine
    off macOS on a build that has it — surfaces as ImportError so the boot
    path degrades exactly as a missing optional dependency did."""
    if kind == "apple":
        import shrike_native

        cls = getattr(shrike_native, "AppleVisionRecognizer", None)
        if cls is None:
            raise ImportError(
                "the Apple Vision OCR engine is not compiled into this build — "
                "platform engines are mobile-only (docs/distribution.md); the "
                "server-profile replacement is the remote recognizer rows (#502)"
            )
        try:
            # Typed through the protocol: the lint lane runs without the
            # native package installed, where the constructor types as Any.
            backend: RecognizerBackend = cls()
        except shrike_native.NativeUnavailableError as e:
            raise ImportError(str(e)) from e
        return backend
    raise ValueError(f"unknown OCR backend {kind!r} (choices: {', '.join(OCR_BACKENDS)})")
