"""Phase 5 smoke (#247): import the PyO3 extension and call into it.

Proves the polyglot seam works end to end — a Rust+PyO3 cdylib, built by
rules_rust against the hermetic CPython, is importable and callable from a Bazel
py_test. If this passes on linux + macOS, the compiled-extension mechanism the
native epics (#219/#224) need is validated.
"""

from __future__ import annotations


def main() -> int:
    import _demo

    assert _demo.add(2, 3) == 5, _demo.add(2, 3)
    assert _demo.add(-1, 1) == 0

    info = _demo.backend_info()
    assert isinstance(info, str) and "pyo3" in info, info

    # It must be the native extension, not some Python stand-in.
    assert _demo.__file__.endswith(".so"), _demo.__file__

    print(f"demo_smoke OK — {info}; module={_demo.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
