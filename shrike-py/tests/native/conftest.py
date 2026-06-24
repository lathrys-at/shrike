"""Fixtures for the native collection-core parity harness.

These tests require an `anki-core` build of the extension
(`scripts/build-native.sh --anki-core`); on a default build every test here
skips. The hard safety rule is enforced structurally: each test opens its own
fresh temp collection through the native core ONLY — the pip `anki` package is
never used on the same file (cross-core parity cases run the pip side in a
subprocess on a separate collection).

The kernel runs a harness-driven ``current_thread`` runtime with no lazy
fallback, so any test that drives an ``AsyncKernel`` op over the asyncio bridge
needs the committed driver threads parked. The session-scoped ``_driven_runtime``
fixture (autouse) installs the runtime and parks them once for the whole native
test process — reusing the production :class:`DrivenRuntime`, so the test path is
the real assembly path. Pure-sync ``CollectionCore`` tests don't need it but are
unaffected (a parked, idle runtime costs nothing).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

import pytest


def _native_collection_core() -> type | None:
    try:
        import shrike_native
    except ImportError:
        return None
    return getattr(shrike_native, "CollectionCore", None)


CORE = _native_collection_core()

requires_anki_core = pytest.mark.skipif(
    CORE is None,
    reason="needs an anki-core build of shrike_native (scripts/build-native.sh default)",
)


@pytest.fixture(scope="session", autouse=True)
def _driven_runtime() -> Iterator[None]:
    """Install + park the kernel's committed driver threads for the session.

    The kernel runtime is harness-driven (no lazy default), so an ``AsyncKernel``
    op only makes progress while a driver thread drives it. Install once (the seam
    is set-once and the threads outlive any kernel, exactly as in production) and
    tear down at session end. A no-op on a build without the kernel bridge (the
    compute-only extension), which those tests skip anyway.

    Process-global guard: when the unit and native suites share one pytest process
    (``pytest tests/unit tests/native``), both trees' autouse fixtures fire — but
    the kernel runtime is set-once, so only the FIRST may park the driver threads
    (a second ``drive_collection`` would hit "already claimed"). A marker on the
    ``shrike_native`` module (the one object both conftests share) elects the
    single owner."""
    try:
        import shrike_native

        from shrike.platform.driven_runtime import DrivenRuntime
    except ImportError:
        yield
        return
    if not hasattr(shrike_native, "init_driven_runtime"):
        # A build without the driven-runtime bridge (compute-only); the
        # kernel-driving tests skip on the missing CollectionCore/AsyncKernel.
        yield
        return
    if getattr(shrike_native, "_shrike_test_driven", False):
        # Another suite's fixture already owns the driven runtime this process.
        yield
        return

    shrike_native._shrike_test_driven = True
    runtime = DrivenRuntime()
    runtime.install()
    runtime.start()
    try:
        yield
    finally:
        runtime.shutdown()
        shrike_native._shrike_test_driven = False


@pytest.fixture
def native_core(tmp_path):
    """A native CollectionCore on a fresh temp collection, closed after."""
    assert CORE is not None
    core = CORE(str(tmp_path / "collection.anki2"))
    yield core
    core.close()


# ── Shared native test helpers ───────────────────────────────────────────────
#
# Stub backends/recognizers and the open/assemble helpers used by more than one
# split native test module. Imports of shrike_native and the harness packages are
# done at call time so this conftest stays importable on builds lacking them (the
# test modules themselves ``pytest.importorskip("shrike_native")`` before reaching
# these).


class _Backend:
    """Deterministic unit vectors + the EmbedderBackend metadata surface."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            b = hashlib.blake2b(text.encode(), digest_size=1).digest()[0] / 255.0
            n = (b * b + 1.0) ** 0.5
            out.append([b / n, 1.0 / n, 0.0, 0.0])
        return out

    def model_fingerprint(self) -> str:
        return "test-backend:v1"

    def embedding_dim(self) -> int:
        return 4


async def _open(tmp_path, backend):
    import shrike_native

    kernel = await shrike_native.async_kernel_open(
        str(tmp_path / "collection.anki2"), str(tmp_path / "cache")
    )
    kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
    return kernel


async def _assemble(tmp_path, *, cooperative: bool = False):
    from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine
    from shrike.harness.engines.embedding.runtime import EmbeddingRuntime
    from shrike.harness.harness import Harness

    runtime = EmbeddingRuntime(model=None)
    derived = DerivedTextStore(
        path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
    )
    return await Harness.assemble(
        collection_path=str(tmp_path / "collection.anki2"),
        cache_dir=str(tmp_path / "cache"),
        runtime=runtime,
        derived=derived,
        cooperative=cooperative,
        hold_seconds=5.0,
        media_read=None,
        media_exists=None,
    )


class _StubAsr:
    """A captured ASR recognizer: transcribes the audio bytes and
    carries a single time-`Span` segment (the audio locator, vs OCR's bbox).
    The RecognizerBackend wire contract — captured behind PyRecognizer, like a
    custom OCR backend — proving the audio path end-to-end without the
    platform AppleSpeechTranscriber (mobile-only, never the server build)."""

    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        # segments JSON carries a "span" locator (time range), serialized like
        # the Rust Segment with Locator::Span — the kernel stores it opaquely.
        out = []
        for data in items:
            text = data.decode()
            segments = json.dumps([{"text": text, "confidence": 0.9, "span": [0.0, 2.5]}])
            out.append((text, 0.9, segments))
        return out

    def model_fingerprint(self) -> str:
        return "stub-asr:v1"
