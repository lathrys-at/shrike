"""Kernel↔host parity for the per-collection cache layout.

The namespace + derived-store path each have one implementation (the kernel,
``shrike_kernel::cache_layout``) bound as ``shrike_native.index_namespace`` /
``shrike_native.derived_db_path``; the host (``shrike.cache_layout``) must match
them byte-for-byte, since either side may compute the on-disk path depending on
whether the native extension is importable, and the kernel's ``DerivedEngine``
and host's ``DerivedTextStore`` open the SAME ``shrike.db``. This pins them so
they can never drift.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from shrike.harness import cache_layout

shrike_native = pytest.importorskip("shrike_native")

pytestmark = pytest.mark.skipif(
    not hasattr(shrike_native, "index_namespace"),
    reason="anki-core build required (scripts/build-native.sh)",
)


# A spread of shapes: absolute, relative, `.`/`..` segments, a symlinked prefix
# on macOS (/tmp → /private/tmp), an existing file (added at runtime), and a raw
# `~`-path — neither side expands `~` (the contract is callers pre-expand), so a
# `~`-spelling must hash identically on both. This pins the divergence that a
# re-introduced expanduser() on either side would cause.
_CASES = [
    "/home/alice/collection.anki2",
    "/home/bob/Anki2/User 1/collection.anki2",
    "relative/c.anki2",
    "/coll/a/../b/deck.anki2",
    "/tmp/shrike-nonexistent-67/c.anki2",
    "~/Anki2/collection.anki2",
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


# -- derived-store path parity ------------------------------------------------

_CACHE = "/var/cache/shrike"


@pytest.mark.parametrize("path", _CASES)
def test_derived_db_path_native_matches_host(path: str) -> None:
    # The kernel's DerivedEngine and the host's DerivedTextStore must open the
    # exact same shrike.db, so the two derived_db_path implementations must
    # agree byte-for-byte across every path shape (incl. `~`, symlinks, absent).
    native = shrike_native.derived_db_path(_CACHE, path)
    host = cache_layout.derived_db_path(_CACHE, path)
    assert native == host, f"derived path drift for {path!r}: {native} != {host}"


def test_derived_db_path_existing_file_agrees(tmp_path) -> None:
    f = tmp_path / "c.anki2"
    f.write_text("x")
    assert shrike_native.derived_db_path(_CACHE, str(f)) == cache_layout.derived_db_path(
        _CACHE, str(f)
    )


def test_derived_db_path_shares_index_namespace_distinct_subtree() -> None:
    # The derived path uses the SAME namespace as the index but a parallel
    # `derived/` subtree (the deliberate index-vs-derived separation).
    path = "/coll/sep.anki2"
    ns = cache_layout.index_namespace(path)
    derived = cache_layout.derived_db_path(_CACHE, path)
    index_dir = cache_layout.collection_index_dir(_CACHE, path)
    assert derived == os.path.join(_CACHE, "derived", ns, "shrike.db")
    assert os.path.dirname(derived) != index_dir
    assert ns in derived and ns in index_dir
