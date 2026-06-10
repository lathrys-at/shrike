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
    """Every workspace member's manifest, derived from the root Cargo.toml.

    Driving the scan from the *members list* (not a directory glob) makes a
    coverage gap loud: under Bazel only data-declared files exist in runfiles,
    so a new crate whose manifest wasn't added to this test's `data` would
    silently escape a glob — here it fails the run instead.
    """
    root = Path("native/Cargo.toml")
    workspace = tomllib.loads(root.read_text())
    members = workspace.get("workspace", {}).get("members", [])
    if not members:
        raise SystemExit("layering_check: no workspace members in native/Cargo.toml")
    paths: list[Path] = []
    missing: list[str] = []
    for member in members:
        manifest = Path("native") / member / "Cargo.toml"
        if manifest.is_file():
            paths.append(manifest)
        else:
            missing.append(str(manifest))
    if missing:
        raise SystemExit(
            "layering_check: member manifest(s) not in runfiles — add them to the "
            f"test's data in native/BUILD.bazel: {missing}"
        )
    return paths


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
