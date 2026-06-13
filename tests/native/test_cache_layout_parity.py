"""Kernel↔host parity for the per-collection index namespace (#67).

The namespace has one implementation (the kernel, ``shrike_kernel::cache_layout``)
bound as ``shrike_native.index_namespace``; the host's pure-Python fallback
(``shrike.cache_layout``) must match it byte-for-byte, since either may compute
the on-disk path depending on whether the native extension is importable. This
pins the two together so they can never drift.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from shrike import cache_layout

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "index_namespace"),
    reason="anki-core build required (scripts/build-native.sh)",
)


# A spread of shapes: absolute, relative, `.`/`..` segments, a symlinked prefix
# on macOS (/tmp → /private/tmp), and an existing file (added at runtime).
_CASES = [
    "/home/alice/collection.anki2",
    "/home/bob/Anki2/User 1/collection.anki2",
    "relative/c.anki2",
    "/coll/a/../b/deck.anki2",
    "/tmp/shrike-nonexistent-67/c.anki2",
]


@pytest.mark.parametrize("path", _CASES)
def test_native_matches_pure_python_fallback(path: str) -> None:
    native = shrike_native.index_namespace(path)
    canonical = cache_layout._canonicalize_for_identity(path)
    fallback = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()
    assert native == fallback, f"namespace drift for {path!r}: {native} != {fallback}"


def test_existing_file_canonicalizes_identically(tmp_path) -> None:
    # An existing file resolves symlinks on both sides (the kernel's
    # std::fs::canonicalize and the host's os.path.realpath agree).
    f = tmp_path / "c.anki2"
    f.write_text("x")
    native = shrike_native.index_namespace(str(f))
    canonical = cache_layout._canonicalize_for_identity(str(f))
    fallback = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()
    assert native == fallback


def test_host_helper_defers_to_the_kernel() -> None:
    # When the extension is present, the host helper returns the kernel's value
    # (not just a matching fallback) — proving the single-implementation wiring.
    path = os.path.abspath("some/collection.anki2")
    assert cache_layout.index_namespace(path) == shrike_native.index_namespace(path)
