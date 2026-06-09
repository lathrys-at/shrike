"""Phase 1 smoke (#243).

Proves the `//src/shrike:shrike` library target and its declared deps import
cleanly on the hermetic toolchain: every core module loads (so the dep list is
complete), and a pure, DB-free function runs. Cheap CI coverage of the wiring
until the full pytest suite migrates in Phase 2 (#244).
"""

from __future__ import annotations


def main() -> int:
    import shrike

    # Import the breadth of the package so any missing requirement surfaces here.
    import shrike.cli  # noqa: F401
    import shrike.client  # noqa: F401
    import shrike.collection  # noqa: F401
    import shrike.index  # noqa: F401
    import shrike.schemas  # noqa: F401
    import shrike.server  # noqa: F401
    import shrike.tools  # noqa: F401
    from shrike.embed_text import normalize_for_embedding

    # Pure function — HTML strip + cloze reveal, no collection or network.
    out = normalize_for_embedding("<b>Bonjour</b> {{c1::le monde}}")
    assert "Bonjour" in out and "le monde" in out, f"unexpected: {out!r}"

    print(f"library_smoke OK — shrike {shrike.__version__}; modules import; embed_text works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
