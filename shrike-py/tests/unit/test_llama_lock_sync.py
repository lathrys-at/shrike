"""tools/llama-server.lock and MODULE.bazel must pin the same llama.cpp (#566).

The de-dup tripwire (tools/check_llama_lock.py) also runs as a Bazel py_test
(//tools:llama_lock_in_sync_test); this is its pip-lane twin so a drifted bump fails
`pytest tests/unit` too. The parser logic lives in the tool; this just drives it
against the real checked-in files and exercises the parsers on fixtures.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    """The repository root, where tools/ + MODULE.bazel live.

    The harness moved into shrike-py/ (#731), so parents[2] is now shrike-py/, not
    the repo root — tools/check_llama_lock.py + MODULE.bazel + the lock stayed at the
    root. Resolve via git (path-independent), falling back to walking up to the
    ancestor that carries MODULE.bazel.
    """
    here = Path(__file__).resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=here.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        for parent in here.parents:
            if (parent / "MODULE.bazel").exists():
                return parent
        return here.parents[2]


_REPO_ROOT = _repo_root()
_CHECKER = _REPO_ROOT / "tools" / "check_llama_lock.py"

_spec = importlib.util.spec_from_file_location("check_llama_lock", _CHECKER)
assert _spec and _spec.loader
check_llama_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_llama_lock)


def test_lock_and_module_bazel_in_sync() -> None:
    """The real lock + MODULE.bazel agree on tag and every per-platform sha."""
    lock_text = (_REPO_ROOT / "tools" / "llama-server.lock").read_text()
    module_text = (_REPO_ROOT / "MODULE.bazel").read_text()
    problems = check_llama_lock.check(lock_text, module_text)
    assert not problems, "lock/MODULE.bazel drifted:\n" + "\n".join(f"  - {p}" for p in problems)


def test_check_flags_tag_sha_and_missing_drift() -> None:
    """The checker catches a stale tag, a wrong sha, and a missing archive."""
    lock = (
        "LLAMA_TAG=b9637\n"
        "SHA256_macos_arm64=aaa\nSHA256_macos_x64=bbb\n"
        "SHA256_ubuntu_x64=ccc\nSHA256_ubuntu_arm64=ddd\n"
    )
    module = """
http_archive(
    name = "llama_server_macos_arm64",
    sha256 = "aaa",
    strip_prefix = "llama-b9637",
    urls = ["https://github.com/ggml-org/llama.cpp/releases/download/b9637/llama-b9637-bin-macos-arm64.tar.gz"],
)
http_archive(
    name = "llama_server_macos_amd64",
    sha256 = "WRONG",
    strip_prefix = "llama-b9637",
    urls = ["https://github.com/ggml-org/llama.cpp/releases/download/b9637/llama-b9637-bin-macos-x64.tar.gz"],
)
http_archive(
    name = "llama_server_linux_amd64",
    sha256 = "ccc",
    strip_prefix = "llama-b9415",
    urls = ["https://github.com/ggml-org/llama.cpp/releases/download/b9415/llama-b9415-bin-ubuntu-x64.tar.gz"],
)
"""
    problems = check_llama_lock.check(lock, module)
    joined = "\n".join(problems)
    assert any("WRONG" in p for p in problems), joined  # sha drift
    assert any("b9415" in p for p in problems), joined  # stale tag/url
    assert any("llama_server_linux_arm64" in p for p in problems), joined  # missing archive


def test_parse_lock_ignores_comments_and_blanks() -> None:
    parsed = check_llama_lock.parse_lock("# header\n\nLLAMA_TAG=bX\n  SHA256_macos_x64=zz \n")
    assert parsed == {"LLAMA_TAG": "bX", "SHA256_macos_x64": "zz"}
