#!/usr/bin/env python3
"""Tripwire: tools/llama-server.lock and MODULE.bazel must agree (#566).

The pinned llama.cpp tag + the four per-platform SHA256s live in
``tools/llama-server.lock`` (shell-sourceable, consumed by the CI model-cache
key) AND, duplicated, in MODULE.bazel's four ``llama_server_*`` http_archives
(Bazel can't read the lock at module-resolution time). They can silently drift
— a bump applied to one but not the other ships a tag/sha mismatch.

This check parses both and asserts every value matches: the tag (in each
archive's ``strip_prefix`` and ``urls``) and the per-platform sha. It runs as a
pytest unit test (pip lane) AND as a Bazel ``py_test`` (``__main__`` exits
non-zero on any mismatch), so neither lane can merge a drifted bump.

The lock's underscore platform keys map to MODULE.bazel's archive names and URL
suffixes:

    SHA256_macos_arm64  -> llama_server_macos_arm64  / -bin-macos-arm64.tar.gz
    SHA256_macos_x64    -> llama_server_macos_amd64  / -bin-macos-x64.tar.gz
    SHA256_ubuntu_x64   -> llama_server_linux_amd64  / -bin-ubuntu-x64.tar.gz
    SHA256_ubuntu_arm64 -> llama_server_linux_arm64  / -bin-ubuntu-arm64.tar.gz
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# lock platform key -> (MODULE.bazel http_archive name, release tarball platform suffix)
_PLATFORMS = {
    "macos_arm64": ("llama_server_macos_arm64", "macos-arm64"),
    "macos_x64": ("llama_server_macos_amd64", "macos-x64"),
    "ubuntu_x64": ("llama_server_linux_amd64", "ubuntu-x64"),
    "ubuntu_arm64": ("llama_server_linux_arm64", "ubuntu-arm64"),
}


def _repo_root() -> Path:
    """Locate the checkout root containing both files.

    Walks up from this file (works in the source tree and under a Bazel
    ``py_test`` runfiles tree, where the sources sit at the workspace root).
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "tools" / "llama-server.lock").is_file() and (
            parent / "MODULE.bazel"
        ).is_file():
            return parent
    raise FileNotFoundError(
        f"could not locate tools/llama-server.lock + MODULE.bazel from {here}"
    )


def parse_lock(text: str) -> dict[str, str]:
    """Parse the shell-sourceable ``KEY=VALUE`` lock into a flat dict."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def parse_module_archives(text: str) -> dict[str, dict[str, str]]:
    """Extract each ``llama_server_*`` http_archive's name/sha256/strip_prefix/url.

    Returns ``{archive_name: {"sha256", "strip_prefix", "url"}}`` for every
    archive whose ``name`` starts with ``llama_server_``.
    """
    archives: dict[str, dict[str, str]] = {}
    for block in re.findall(r"http_archive\(\s*(.*?)\)", text, flags=re.DOTALL):
        name_m = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_m or not name_m.group(1).startswith("llama_server_"):
            continue
        sha_m = re.search(r'sha256\s*=\s*"([^"]+)"', block)
        strip_m = re.search(r'strip_prefix\s*=\s*"([^"]+)"', block)
        url_m = re.search(r'urls\s*=\s*\[\s*"([^"]+)"', block)
        archives[name_m.group(1)] = {
            "sha256": sha_m.group(1) if sha_m else "",
            "strip_prefix": strip_m.group(1) if strip_m else "",
            "url": url_m.group(1) if url_m else "",
        }
    return archives


def check(lock_text: str, module_text: str) -> list[str]:
    """Return a list of human-readable mismatches (empty == in sync)."""
    lock = parse_lock(lock_text)
    archives = parse_module_archives(module_text)
    problems: list[str] = []

    tag = lock.get("LLAMA_TAG")
    if not tag:
        return ["tools/llama-server.lock has no LLAMA_TAG"]

    expected_names = {name for name, _ in _PLATFORMS.values()}
    missing = expected_names - set(archives)
    if missing:
        problems.append(f"MODULE.bazel is missing llama_server http_archive(s): {sorted(missing)}")

    for lock_key, (archive_name, suffix) in _PLATFORMS.items():
        sha_key = f"SHA256_{lock_key}"
        lock_sha = lock.get(sha_key)
        if not lock_sha:
            problems.append(f"tools/llama-server.lock has no {sha_key}")
            continue
        arch = archives.get(archive_name)
        if arch is None:
            continue  # already reported as missing
        if arch["sha256"] != lock_sha:
            problems.append(
                f"{archive_name}: MODULE.bazel sha256 {arch['sha256']!r} != "
                f"lock {sha_key} {lock_sha!r}"
            )
        expected_strip = f"llama-{tag}"
        if arch["strip_prefix"] != expected_strip:
            problems.append(
                f"{archive_name}: strip_prefix {arch['strip_prefix']!r} != "
                f"expected {expected_strip!r} (LLAMA_TAG={tag})"
            )
        expected_url = (
            f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/"
            f"llama-{tag}-bin-{suffix}.tar.gz"
        )
        if arch["url"] != expected_url:
            problems.append(f"{archive_name}: url {arch['url']!r} != expected {expected_url!r}")

    return problems


def _load() -> list[str]:
    root = _repo_root()
    lock_text = (root / "tools" / "llama-server.lock").read_text()
    module_text = (root / "MODULE.bazel").read_text()
    return check(lock_text, module_text)


def test_llama_lock_matches_module_bazel() -> None:
    """pytest entry: the lock and MODULE.bazel pin the same tag + shas."""
    problems = _load()
    assert not problems, "tools/llama-server.lock and MODULE.bazel drifted:\n" + "\n".join(
        f"  - {p}" for p in problems
    )


if __name__ == "__main__":
    issues = _load()
    if issues:
        print("tools/llama-server.lock and MODULE.bazel drifted:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        print(
            "\nBump both via tools/update-llama-lock.sh <TAG>, then mirror the "
            "values into MODULE.bazel's llama_server_* http_archives.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("OK: tools/llama-server.lock and MODULE.bazel are in sync.")
