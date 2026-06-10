"""Crate-layering gate (#269, epic #265 convention 5).

`pyo3` is allowed ONLY in the Python binding crates: `shrike-py` (the real
binding module) and the `_demo` polyglot proof. Kernel/compute crates must stay
pure Rust — that's what makes the stretch end-state (a no-CPython kernel)
structural rather than aspirational. This test reads every crate manifest in
the workspace and fails if any other crate declares a pyo3 dependency.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

# Crates allowed to depend on pyo3 (by Cargo package name).
PYO3_ALLOWED = {"shrike-py", "shrike-native-demo"}


def manifest_paths() -> list[Path]:
    """Every crate manifest shipped as test data (runfiles cwd is the repo root)."""
    root = Path("native")
    found = sorted(root.glob("*/Cargo.toml"))
    if not found:
        raise SystemExit("layering_check found no crate manifests under native/")
    return found


def main() -> int:
    failures: list[str] = []
    for path in manifest_paths():
        manifest = tomllib.loads(path.read_text())
        package = manifest.get("package", {}).get("name")
        if package is None:
            continue  # the workspace root manifest
        deps: set[str] = set()
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            deps.update(manifest.get(section, {}))
        if "pyo3" in deps and package not in PYO3_ALLOWED:
            failures.append(
                f"{path}: crate '{package}' depends on pyo3 — only {sorted(PYO3_ALLOWED)} may"
                " (epic #265 convention 5: compute/kernel crates stay pure Rust)"
            )
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
