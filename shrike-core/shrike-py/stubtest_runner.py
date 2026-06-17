"""Run mypy.stubtest against the shrike_native package (#269's typing gate).

A drifted stub (a Rust signature no longer matching its .pyi) fails this test.
Runs from a neutral temp cwd so the repo's own mypy configuration (tuned for
src/shrike) can't interfere with stub resolution, and hands the child process
the runfiles import path via PYTHONPATH.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["MYPY_CACHE_DIR"] = tempfile.mkdtemp(prefix="stubtest-cache-")
    # Feature-gated symbols (anki-core, #278) are stub-declared but absent
    # from default builds; the allowlist covers them either way.
    allowlist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubtest_allowlist.txt")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "mypy.stubtest",
            "shrike_native",
            "--allowlist",
            allowlist,
            "--ignore-unused-allowlist",
        ],
        env=env,
        cwd=tempfile.mkdtemp(prefix="stubtest-cwd-"),
    )


if __name__ == "__main__":
    sys.exit(main())
