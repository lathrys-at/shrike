#!/usr/bin/env python3
"""Package a Shrike skill folder into a distributable ``.skill`` (or ``.zip``).

A ``.skill`` file is just a zip whose root contains the skill folder
(``create-cards/SKILL.md``, ``create-cards/references/…``), which is what Claude
Desktop / claude.ai expect when you upload a skill. This mirrors the layout the
skill-creator uses, so the output installs the same way.

Usage:
    scripts/package-skill.py                                # packages shrike-skills/create-cards → dist/
    scripts/package-skill.py shrike-skills/create-cards     # explicit folder
    scripts/package-skill.py shrike-skills/create-cards -o /tmp   # custom output dir
    scripts/package-skill.py shrike-skills/create-cards --zip     # name it .zip instead

Build artifacts (__pycache__, *.pyc, .DS_Store, node_modules) and a top-level
``evals/`` directory are excluded.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILL = REPO_ROOT / "shrike-skills" / "create-cards"

EXCLUDE_DIRS = {"__pycache__", "node_modules", ".git"}
EXCLUDE_GLOBS = {"*.pyc", "*.pyo"}
EXCLUDE_FILES = {".DS_Store"}
# Excluded only at the skill root (test scaffolding, not shipped content).
ROOT_EXCLUDE_DIRS = {"evals"}


def _should_exclude(arcname: Path) -> bool:
    parts = arcname.parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return True
    # parts[0] is the skill folder name; parts[1] (if any) is its first subdir.
    if len(parts) > 1 and parts[1] in ROOT_EXCLUDE_DIRS:
        return True
    if arcname.name in EXCLUDE_FILES:
        return True
    return any(fnmatch.fnmatch(arcname.name, pat) for pat in EXCLUDE_GLOBS)


def _validate(skill_path: Path) -> str | None:
    """Lightweight checks: a folder with a SKILL.md that has name + description
    frontmatter. Returns an error string, or None if valid."""
    if not skill_path.is_dir():
        return f"not a directory: {skill_path}"
    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        return f"SKILL.md not found in {skill_path}"

    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return "SKILL.md has no YAML frontmatter (must start with '---')"
    end = text.find("\n---", 3)
    if end == -1:
        return "SKILL.md frontmatter is not closed with '---'"
    front = text[3:end]
    for key in ("name", "description"):
        if not any(line.lstrip().startswith(f"{key}:") for line in front.splitlines()):
            return f"SKILL.md frontmatter is missing '{key}:'"
    return None


def package_skill(skill_path: Path, output_dir: Path, *, extension: str = "skill") -> Path | None:
    skill_path = skill_path.resolve()

    err = _validate(skill_path)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{skill_path.name}.{extension}"

    added = 0
    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(skill_path.rglob("*")):
            if not path.is_file():
                continue
            arcname = path.relative_to(skill_path.parent)
            if _should_exclude(arcname):
                print(f"  skip  {arcname}")
                continue
            zf.write(path, arcname)
            print(f"  add   {arcname}")
            added += 1

    size_kb = out_file.stat().st_size / 1024
    print(f"\nPackaged {added} file(s) → {out_file}  ({size_kb:.1f} KiB)")
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "skill",
        nargs="?",
        type=Path,
        default=DEFAULT_SKILL,
        help="Path to the skill folder (default: shrike-skills/create-cards).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "dist",
        help="Output directory for the package (default: dist/).",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Use a .zip extension instead of .skill (identical contents).",
    )
    args = parser.parse_args()

    result = package_skill(args.skill, args.output, extension="zip" if args.zip else "skill")
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
