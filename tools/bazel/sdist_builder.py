"""Build the Python sdist for the //:sdist Bazel rule (#245).

Invoked as a build action: stages the declared source files into a writable tree
(Bazel inputs are read-only, and hatch-vcs's build hook writes _version.py into
src/shrike), pins the version from STABLE_VERSION via SETUPTOOLS_SCM_PRETEND_VERSION
(so the sandbox needs no git), and runs hatchling through the `build` API with no
isolation (build/hatchling/hatch-vcs come from this binary's runfiles, not a pip
install). The produced shrike_mcp-<version>.tar.gz is copied to the rule's output.

The explicit sdist file selection is injected into the *staged* pyproject.toml (not
the repo's), so the repo's `python -m build` keeps its git-based sdist unchanged while
//:sdist ships exactly the staged tree (everything the rule's srcs put here).

Args (read from a Bazel param file via @file): --version-file <stable-status.txt>,
--out <output path>, then the source file paths (positional).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile


def _stable_version(version_file: str) -> str:
    with open(version_file) as f:
        for line in f:
            if line.startswith("STABLE_VERSION "):
                return line.split(" ", 1)[1].strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(fromfile_prefix_chars="@")
    ap.add_argument("--version-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("srcs", nargs="*")
    args = ap.parse_args()

    version = _stable_version(args.version_file) or "0.0.0"
    out = os.path.abspath(args.out)

    with tempfile.TemporaryDirectory() as tmp:
        stage = os.path.join(tmp, "src")
        # Sources arrive as execroot-relative paths; replicate that layout under a
        # writable root so hatchling reads pyproject.toml + the include set from it.
        for src in args.srcs:
            dst = os.path.join(stage, src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        # Inject an explicit, git-independent sdist file selection into the staged
        # pyproject (not the repo's): ship everything staged here — the rule's srcs
        # already exclude the BUILD files and bytecode. Keeps `python -m build` in the
        # repo (git-based) unchanged.
        with open(os.path.join(stage, "pyproject.toml"), "a") as f:
            f.write('\n[tool.hatch.build.targets.sdist]\ninclude = ["**/*"]\n')

        outdir = os.path.join(tmp, "out")
        os.makedirs(outdir)
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = version

        from build import ProjectBuilder  # from this binary's runfiles

        ProjectBuilder(stage).build("sdist", outdir)

        produced = glob.glob(os.path.join(outdir, "*.tar.gz"))
        if not produced:
            sys.exit("sdist: no tarball produced")
        shutil.copy2(produced[0], out)


if __name__ == "__main__":
    main()
