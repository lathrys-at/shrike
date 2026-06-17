"""Crate-layering gate (#269, epic #265 convention 5; engine purity #342; layer floor #704).

Structural rules over every workspace manifest:

1. `pyo3` is allowed ONLY in the Python binding crate `shrike-py` (the real
   binding module). Kernel/compute crates must stay pure Rust — that's what
   makes the no-CPython kernel structural rather than aspirational.
2. `shrike-kernel` names NO engine crate (#342): it consumes the
   shrike-engine-api traits and composes whatever engines the host attaches —
   a dependency on a concrete engine (ort embedding, a platform recognizer, a
   remote client, subprocess management) is an architecture regression. The
   check is over the kernel's TRANSITIVE closure across workspace members
   (#380): naming an engine through an intermediary (the since-dissolved
   shrike-compute crate's shrike-embed leak) still links the whole engine
   stack into the kernel, so the direct-deps-only check was a hole, not a
   gate.
3. The layer FLOOR (#704): every low/utility/contract-floor crate's transitive
   closure contains NEITHER `shrike-kernel` NOR any engine crate. The kernel
   and engines sit ABOVE these crates, so a floor crate that (even
   transitively) reaches up into the kernel or an engine has inverted the
   layer graph. Rules 2 and 3 are the same primitive applied in both
   directions: rule 2 checks the kernel's closure for engines; rule 3 checks
   each floor crate's closure for the kernel-and-above set.

Adding a crate to the floor as the #703 reorg lands new low-utility crates
(`shrike-error`, `shrike-network`, `shrike-process`, `shrike-media`,
`shrike-cache`, `shrike-store`, …) is a ONE-LINE addition to `LAYER_FLOOR`
below — the closure machinery handles the rest.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

# Crates allowed to depend on pyo3 (by Cargo package name).
PYO3_ALLOWED = {"shrike-py"}

# Engine crates the kernel must NEVER name (#342). Grown as engine crates are
# added; the kernel's only engine-shaped dep is shrike-engine-api (the traits).
# (The shared LOW utility crates — shrike-network (the SSRF primitives), plus
# the coming shrike-process — are intentionally NOT here: they
# sit BELOW both the kernel and the engine crates, so BOTH may depend on them
# without inverting the layer graph. They live in LAYER_FLOOR instead.)
ENGINE_CRATES = {
    "shrike-embed",
    "shrike-recognize-apple",
    "shrike-embed-remote",
    "shrike-describe-remote",
    "shrike-llama-server",
}

# The layer FLOOR (#704): low/utility/contract crates that sit below both the
# kernel and the engines. Their transitive closure must reach NEITHER the
# kernel NOR any engine — depending UP into those layers inverts the graph.
# The assertion is on each floor crate's OUTGOING edges, so the legitimate
# downward edges INTO these crates (kernel -> shrike-engine-api, kernel ->
# shrike-network, an engine -> shrike-network, …) are fine — only edges OUT of a
# floor crate are constrained.
#
# As the #703 reorg adds further low-utility crates (the coming
# shrike-process / shrike-media / shrike-cache), extend this set — one line per
# crate.
LAYER_FLOOR = {
    "shrike-error",
    "shrike-network",
    "shrike-schemas",
    "shrike-engine-api",  # the kernel<->ort firewall — a thin contract, stays floor
    "shrike-store",
}

# The set a floor crate's closure must avoid: the kernel and everything at or
# above it (the engines). Naming this once keeps rule 3 a single reusable
# closure check.
KERNEL_AND_ABOVE = ENGINE_CRATES | {"shrike-kernel"}


def manifest_paths() -> list[Path]:
    """Every workspace member's manifest, derived from the root Cargo.toml.

    Driving the scan from the *members list* (not a directory glob) makes a
    coverage gap loud: under Bazel only data-declared files exist in runfiles,
    so a new crate whose manifest wasn't added to this test's `data` would
    silently escape a glob — here it fails the run instead.
    """
    root = Path("shrike-core/Cargo.toml")
    workspace = tomllib.loads(root.read_text())
    members = workspace.get("workspace", {}).get("members", [])
    if not members:
        raise SystemExit("layering_check: no workspace members in shrike-core/Cargo.toml")
    paths: list[Path] = []
    missing: list[str] = []
    for member in members:
        manifest = Path("shrike-core") / member / "Cargo.toml"
        if manifest.is_file():
            paths.append(manifest)
        else:
            missing.append(str(manifest))
    if missing:
        raise SystemExit(
            "layering_check: member manifest(s) not in runfiles — add them to the "
            f"test's data in shrike-core/BUILD.bazel: {missing}"
        )
    return paths


def transitive_closure(
    start: str, runtime_deps: dict[str, set[str]], direct: set[str]
) -> dict[str, list[str]]:
    """`start`'s transitive workspace-member closure, with witness paths.

    The reusable "does crate X's closure contain any crate in set Y" primitive:
    intersect the returned keys with Y. Walks `[dependencies]` edges only beyond
    the first hop (dev/build deps of an intermediary don't link into `start`),
    starting from `start`'s own `direct` deps (pass ALL sections there — they DO
    link into `start`'s own lib/tests). Returns `{member: dep chain from start}`
    so a violation names the leak path, not just the leaked crate.
    """
    members = set(runtime_deps)
    chains: dict[str, list[str]] = {}
    frontier: list[str] = []
    for dep in sorted(direct & members):
        chains[dep] = [start, dep]
        frontier.append(dep)
    while frontier:
        crate = frontier.pop()
        for dep in sorted(runtime_deps.get(crate, set()) & members):
            if dep not in chains:
                chains[dep] = [*chains[crate], dep]
                frontier.append(dep)
    return chains


def closure_violations(
    start: str,
    forbidden: set[str],
    runtime_deps: dict[str, set[str]],
    all_deps: dict[str, set[str]],
    paths_by_package: dict[str, Path],
) -> dict[str, list[str]]:
    """Witness chains for any `forbidden` crate in `start`'s transitive closure.

    `{leaked crate: chain from start}` for each forbidden crate reachable from
    `start` — empty when the layer rule holds. `start` must be a workspace
    member (its direct deps come from `all_deps`).
    """
    chains = transitive_closure(start, runtime_deps, all_deps[start])
    return {crate: chains[crate] for crate in sorted(set(chains) & forbidden)}


def main() -> int:
    failures: list[str] = []
    # Per-member dep sets: every section for the direct checks; the
    # `[dependencies]` section alone for the transitive (link-graph) walk.
    all_deps: dict[str, set[str]] = {}
    runtime_deps: dict[str, set[str]] = {}
    paths_by_package: dict[str, Path] = {}
    for path in manifest_paths():
        manifest = tomllib.loads(path.read_text())
        package = manifest.get("package", {}).get("name")
        if package is None:
            continue  # the workspace root manifest
        deps: set[str] = set()
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            deps.update(manifest.get(section, {}))
        all_deps[package] = deps
        runtime_deps[package] = set(manifest.get("dependencies", {}))
        paths_by_package[package] = path
        if "pyo3" in deps and package not in PYO3_ALLOWED:
            failures.append(
                f"{path}: crate '{package}' depends on pyo3 — only {sorted(PYO3_ALLOWED)} may"
                " (epic #265 convention 5: compute/kernel crates stay pure Rust)"
            )

    # The transitive gate (#380): no engine crate anywhere in the kernel's
    # workspace-member closure — a leak through an intermediary links the
    # engine stack into the kernel just as surely as naming it directly.
    if "shrike-kernel" not in all_deps:
        raise SystemExit("layering_check: shrike-kernel not among workspace members")
    kernel_path = paths_by_package["shrike-kernel"]
    for crate, chain in closure_violations(
        "shrike-kernel", ENGINE_CRATES, runtime_deps, all_deps, paths_by_package
    ).items():
        failures.append(
            f"{kernel_path}: shrike-kernel transitively links engine crate '{crate}'"
            f" via {' -> '.join(chain)} — the kernel composes engines it is"
            " GIVEN (#342/#380); break the intermediary dependency instead"
        )

    # The layer-floor gate (#704): the same closure check applied to each floor
    # crate, forbidding the kernel-and-above set. A floor crate reaching up into
    # the kernel or an engine — directly or through an intermediary — has
    # inverted the layer graph.
    for floor in sorted(LAYER_FLOOR):
        if floor not in all_deps:
            raise SystemExit(
                f"layering_check: layer-floor crate '{floor}' not among workspace members"
                " — drop it from LAYER_FLOOR (it was renamed/removed) or restore the crate"
            )
        floor_path = paths_by_package[floor]
        for crate, chain in closure_violations(
            floor, KERNEL_AND_ABOVE, runtime_deps, all_deps, paths_by_package
        ).items():
            kind = "the kernel" if crate == "shrike-kernel" else f"engine crate '{crate}'"
            failures.append(
                f"{floor_path}: layer-floor crate '{floor}' transitively reaches {kind}"
                f" via {' -> '.join(chain)} — floor/utility crates sit BELOW the kernel and"
                " engines (#704); a dependency UP into them inverts the layer graph"
            )

    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
