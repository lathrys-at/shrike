"""Per-collection cache layout (#67): where a collection's vector index lives.

The vector index (``index.usearch`` + ``index.meta.json``) is namespaced per
collection under the shared cache dir, so one daemon serving several collections
never collides their indexes. The load-bearing boundary (#69): **index identity
keys on a stable function of the collection FILE PATH, never the profile name**
— every collection has a path; not every collection is registered. The path is
the only identity always available, which is what lets #67 land independently of
the registry (#66) and how the routing capstone (#68) wires them (a selector
resolves name → path via the registry, and the path determines the namespace).

The kernel owns the identity derivation and writes the files; this module is the
host-side mirror so the harness/CLI can resolve the same
``<cache_dir>/index/<namespace>/`` the kernel writes (status reporting, the #68
routing, tests). The namespace itself comes from the kernel
(``shrike_native.index_namespace``) — one implementation, no parity drift — with
a pure-Python fallback used only when the native extension isn't importable (a
plain client environment), pinned byte-for-byte against the kernel by a test.
"""

from __future__ import annotations

import hashlib
import os

# The subdirectory under the cache dir that holds the per-collection index
# namespaces — kept in sync with ``shrike_kernel::cache_layout::INDEX_SUBDIR``.
INDEX_SUBDIR = "index"


def _canonicalize_for_identity(collection_path: str) -> str:
    """Resolve a collection path to the stable string the identity hashes.

    Mirrors the kernel's ``canonicalize_for_identity`` (cache_layout.rs)
    **byte-for-byte**: an **existing** file is fully canonicalized (``realpath``
    — folds ``..``, symlinks, and a relative-vs-absolute spelling of the same
    file to one identity, like Rust's ``std::fs::canonicalize``); an **absent**
    file (a fresh collection) falls back to a lexical absolutize (``abspath`` —
    collapses ``.``/``..`` WITHOUT resolving symlinks, matching the kernel's
    lexical-absolute fallback) so the key is still stable run-to-run. The split
    matters where a path prefix is a symlink (e.g. macOS ``/tmp`` →
    ``/private/tmp``): ``realpath`` would resolve it for an absent file but the
    kernel's fallback would not.

    Deliberately does **no** ``~`` expansion — the kernel's canonicalizer does
    not either, so doing it here would diverge the two namespaces for a raw
    ``~``-path. The contract is that callers pass already-expanded paths: the
    registry stores ``abspath(expanduser(...))`` and ``config.resolve_collection``
    expanduser's before the path ever reaches here.
    """
    if os.path.exists(collection_path):
        return os.path.realpath(collection_path)
    return os.path.abspath(collection_path)


def index_namespace(collection_path: str) -> str:
    """The stable, path-derived identity for a collection's vector index.

    Defers to the kernel (``shrike_native.index_namespace``) so there is one
    implementation; the pure-Python fallback (blake2b of the canonicalized path,
    16-byte digest, hex) is used only when the native extension isn't available
    and is pinned byte-for-byte against the kernel by a parity test.
    """
    try:
        import shrike_native

        return str(shrike_native.index_namespace(collection_path))
    except (ImportError, AttributeError):
        canonical = _canonicalize_for_identity(collection_path)
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def collection_index_dir(cache_dir: str, collection_path: str) -> str:
    """The per-collection index directory: ``<cache_dir>/index/<namespace>/``.

    This is where the kernel writes ``index.usearch`` / ``index.meta.json`` for
    ``collection_path``; the host resolves the same location for diagnostics and
    (in #68) per-collection routing. ``cache_dir`` is the base cache dir the
    ``config.resolve_cache_dir`` cascade yields.
    """
    return os.path.join(cache_dir, INDEX_SUBDIR, index_namespace(collection_path))
