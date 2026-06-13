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
import os
from typing import Any, Protocol

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


def make_describe_recognizer(
    endpoint: str,
    *,
    model: str | None = None,
    api_key_env: str | None = None,
    mmproj: str | None = None,
) -> Any:
    """Construct the remote VLM describe engine (#433/#485) for the kernel's
    ``describe`` recognition purpose — image→descriptive prose into the text
    embedding space (vector-only).

    Proves connectivity (``/health`` or, failing that, ``/v1/models``) and
    composes the describe fingerprint host-side from the endpoint's model
    identity (``RemoteDescriber.compose_fingerprint`` — the crate's recipe),
    folding ``mmproj`` ONLY for a host-launched local managed server (a cloud
    endpoint passes ``None``, byte-identical to "no mmproj suffix"; the
    ``prompt=N`` suffix is unconditional). Returns a constructed
    ``RemoteDescriber`` carrying that fingerprint, ready for the native attach.

    Raises ``ImportError`` when the engine isn't compiled into this build
    (so boot degrades like a missing optional dependency), ``RuntimeError``
    on a missing key env / dead endpoint.
    """
    import shrike_native

    cls = getattr(shrike_native, "RemoteDescriber", None)
    if cls is None:
        raise ImportError(
            "the remote VLM describe engine is not compiled into this build "
            "(needs the engine-remote feature)"
        )
    api_key = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"api_key_env names {api_key_env}, which is not set in the server's "
                "environment (secrets are referenced, never inline)"
            )
    # A probe client (no fingerprint yet) reads the endpoint's identity, then
    # composes the fingerprint the attached engine will carry.
    probe = cls(endpoint, api_key=api_key, model=model)
    if not probe.health_ok():
        # No /health (a cloud endpoint may not serve it) → fall back to
        # model_info as the liveness signal; an unreachable endpoint yields
        # an empty ModelInfo, which composes to describe:<model|unknown> and
        # still attaches (the sweep's chunk-Err-aborts contract leaves the
        # backlog pending until the endpoint comes up).
        logger.info("describe endpoint %s has no /health; probing /v1/models", endpoint)
    model_id, meta_json = probe.model_info()
    fingerprint = cls.compose_fingerprint(model_id, meta_json, model, mmproj)
    logger.info("describe recognizer fingerprint: %s", fingerprint)
    return cls(endpoint, api_key=api_key, model=model, fingerprint=fingerprint)
