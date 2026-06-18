"""Build the Python sdist for the //shrike-py:sdist Bazel rule (#245).

Invoked as a build action: stages the declared source files into a writable tree
(Bazel inputs are read-only, and hatch-vcs's build hook writes _version.py into
src/shrike), pins the version from STABLE_VERSION via SETUPTOOLS_SCM_PRETEND_VERSION
(so the sandbox needs no git), and runs hatchling through the `build` API with no
isolation (build/hatchling/hatch-vcs come from this binary's runfiles, not a pip
install). The produced shrike_py-<version>.tar.gz is copied to the rule's output.

The explicit sdist file selection is injected into the *staged* pyproject.toml (not
the repo's), so the repo's `python -m build` keeps its git-based sdist unchanged while
//shrike-py:sdist ships exactly the staged tree (everything the rule's srcs put here).

--project-subdir names the directory within the stage that holds pyproject.toml — the
harness now lives in shrike-py/ (#731), so pyproject + the package sit under
shrike-py/ while README/LICENSE/CHANGELOG stay at the repo root (one project-wide
set). The repo-root files are staged outside the project subdir, so they're COPIED
into it (and the staged pyproject's `readme = "../README.md"` is rewritten to the
local copy) — otherwise hatchling's `include` glob, rooted at the project dir, would
ship neither the metadata long-description source nor the licence in the tarball.

Args (read from a Bazel param file via @file): --version-file <stable-status.txt>,
--out <output path>, --project-subdir <dir within stage>, then the source file paths
(positional).
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
    # The dir within the stage that carries pyproject.toml (the project root for the
    # build). Empty = the stage root (the pre-#731 layout).
    ap.add_argument("--project-subdir", default="")
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

        # The project root for the build (where pyproject.toml lives).
        project_root = os.path.join(stage, args.project_subdir) if args.project_subdir else stage

        # Repo-root files staged outside the project subdir (README/LICENSE/CHANGELOG)
        # must live INSIDE the project root or hatchling's include glob (rooted there)
        # won't package them. Copy each stage-root file that isn't already under the
        # project subdir into the project root.
        if args.project_subdir:
            for name in os.listdir(stage):
                p = os.path.join(stage, name)
                if name == args.project_subdir:
                    continue
                if os.path.isfile(p):
                    shutil.copy2(p, os.path.join(project_root, name))

        # The shipped pyproject reads the README from the repo root (`../README.md`,
        # #731); inside the staged project root the copy sits beside pyproject, so
        # rewrite it to the local name for this build only (the repo's pyproject is
        # untouched).
        pyproject = os.path.join(project_root, "pyproject.toml")
        with open(pyproject) as f:
            text = f.read()
        text = text.replace('readme = "../README.md"', 'readme = "README.md"')
        # Inject an explicit, git-independent sdist file selection: ship everything
        # staged here — the rule's srcs already exclude the BUILD files and bytecode.
        # Keeps `python -m build` in the repo (git-based) unchanged.
        text += '\n[tool.hatch.build.targets.sdist]\ninclude = ["**/*"]\n'
        with open(pyproject, "w") as f:
            f.write(text)

        outdir = os.path.join(tmp, "out")
        os.makedirs(outdir)
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = version

        from build import ProjectBuilder  # from this binary's runfiles

        ProjectBuilder(project_root).build("sdist", outdir)

        produced = glob.glob(os.path.join(outdir, "*.tar.gz"))
        if not produced:
            sys.exit("sdist: no tarball produced")
        shutil.copy2(produced[0], out)


if __name__ == "__main__":
    main()
