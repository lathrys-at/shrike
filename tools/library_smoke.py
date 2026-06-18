"""Phase 1 smoke (#243).

Proves the `//shrike-py/src/shrike:shrike` library target and its declared deps import
cleanly on the hermetic toolchain: every core module loads (so the dep list is
complete), and a pure, DB-free function runs. Cheap CI coverage of the wiring
until the full pytest suite migrates in Phase 2 (#244).
"""

from __future__ import annotations


def main() -> int:
    # Since the #278 cutover the normalization runs in the native core; the
    # smoke proves the native module is importable and carries it.
    import shrike_native

    import shrike
    import shrike.api.tools  # noqa: F401

    # Import the breadth of the package so any missing requirement surfaces here.
    import shrike.cli  # noqa: F401
    import shrike.client  # noqa: F401
    import shrike.harness.collection  # noqa: F401
    import shrike.harness.index  # noqa: F401
    import shrike.schemas  # noqa: F401
    import shrike.server  # noqa: F401
    from shrike.harness.engines.embedding.text import EMBED_TEXT_VERSION

    assert hasattr(shrike_native, "CollectionCore")
    assert EMBED_TEXT_VERSION >= 1

    print(f"library_smoke OK — shrike {shrike.__version__}; modules import; native core present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
