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
    compute-only extension), which those tests skip anyway."""
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

    runtime = DrivenRuntime()
    runtime.install()
    runtime.start()
    try:
        yield
    finally:
        runtime.shutdown()


@pytest.fixture
def native_core(tmp_path):
    """A native CollectionCore on a fresh temp collection, closed after."""
    assert CORE is not None
    core = CORE(str(tmp_path / "collection.anki2"))
    yield core
    core.close()
