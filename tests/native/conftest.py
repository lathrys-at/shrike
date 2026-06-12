"""Fixtures for the native collection-core parity harness (#278 series, step 1).

These tests require an `anki-core` build of the extension
(`scripts/build-native.sh --anki-core`); on a default build every test here
skips. The hard safety rule is enforced structurally: each test opens its own
fresh temp collection through the native core ONLY — the pip `anki` package is
never used on the same file (cross-core parity cases run the pip side in a
subprocess on a separate collection).
"""

from __future__ import annotations

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


@pytest.fixture
def native_core(tmp_path):
    """A native CollectionCore on a fresh temp collection, closed after."""
    assert CORE is not None
    core = CORE(str(tmp_path / "collection.anki2"))
    yield core
    core.close()
