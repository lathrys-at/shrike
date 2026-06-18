"""Per-collection index namespacing — the host-side helper.

The kernel owns the identity derivation and the on-disk layout (those are
pinned by Rust tests); this exercises the Python mirror the harness/CLI use to
resolve the same ``<cache_dir>/index/<namespace>/`` the kernel writes, plus the
pure-Python fallback (the determinism the native parity test in tests/native
pins byte-for-byte against the kernel).
"""

from __future__ import annotations

import hashlib
import os

from shrike.harness import cache_layout


def test_namespace_is_stable() -> None:
    # The load-bearing property: the same collection path always yields the
    # same namespace, so an index built one run is found the next.
    a = cache_layout.index_namespace("/coll/deck.anki2")
    b = cache_layout.index_namespace("/coll/deck.anki2")
    assert a == b
    # Hex digest of a 16-byte blake2b → 32 hex chars.
    assert len(a) == 32
    assert all(c in "0123456789abcdef" for c in a)


def test_distinct_collections_get_distinct_namespaces() -> None:
    # The isolation property: two collections (even same basename) never
    # collide.
    a = cache_layout.index_namespace("/home/alice/collection.anki2")
    b = cache_layout.index_namespace("/home/bob/collection.anki2")
    assert a != b


def test_index_dir_nests_under_the_index_subdir() -> None:
    cache = "/var/cache/shrike"
    d = cache_layout.collection_index_dir(cache, "/coll/deck.anki2")
    ns = cache_layout.index_namespace("/coll/deck.anki2")
    assert d == os.path.join(cache, cache_layout.INDEX_SUBDIR, ns)
    # Two collections sharing one cache dir resolve to different index dirs.
    other = cache_layout.collection_index_dir(cache, "/coll/other.anki2")
    assert other != d
    assert d.startswith(os.path.join(cache, cache_layout.INDEX_SUBDIR))


def test_relative_and_absolute_spellings_collapse(tmp_path) -> None:
    # An existing file: a `.`-laden spelling resolves to the same identity as
    # the plain absolute path (canonicalization folds them).
    f = tmp_path / "c.anki2"
    f.write_text("x")
    plain = cache_layout.index_namespace(str(f))
    dotted = cache_layout.index_namespace(str(tmp_path / "." / "c.anki2"))
    assert plain == dotted


def test_pure_python_fallback_matches_the_documented_scheme() -> None:
    # The fallback (used when the native extension isn't importable) is a
    # blake2b-16 of the canonicalized path. Pin the scheme so a refactor can't
    # silently change the on-disk namespace for plain-client environments.
    path = "/coll/deck.anki2"
    canonical = cache_layout._canonicalize_for_identity(path)
    expected = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()
    assert cache_layout.index_namespace(path) == expected


def test_absent_file_does_not_resolve_symlinked_prefixes() -> None:
    # A fresh (not-yet-created) collection canonicalizes LEXICALLY (no symlink
    # resolution), matching the kernel's canonicalize→lexical-absolute fallback
    # — so the host and kernel agree on the namespace before the file exists.
    # `abspath` collapses `..` without resolving symlinks.
    assert cache_layout._canonicalize_for_identity("/a/b/../c.anki2") == "/a/c.anki2"
