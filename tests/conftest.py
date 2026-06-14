"""Top-level pytest hooks shared by every suite.

The native-staleness backstop (#573): `pip install -e ".[dev]"` builds only the
Python harness, never the Rust `shrike_native` extension (a separate cargo step
in scripts/build-native.sh). After a pull that touches native/, an unguarded
pytest run would silently import a stale `_native.so` and fail with confusing
ABI/import errors. This hook runs the same git-content staleness check the
.envrc auto-rebuild uses and, when the extension is stale or unbuilt, aborts the
session *before* collection imports the extension — failing loud and actionable
instead of crashing deep in an import.

Bypass with SHRIKE_SKIP_NATIVE_STALE_CHECK=1 (Bazel sets it — that lane builds
the extension hermetically and has no venv stamp to read). The check is also a
no-op on a source checkout that lacks the script (sdist/release tree).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STALE_SCRIPT = _REPO_ROOT / "scripts" / "native-stale.sh"


def pytest_configure(config: pytest.Config) -> None:
    """Abort before collection if the native extension is stale or unbuilt."""
    if os.environ.get("SHRIKE_SKIP_NATIVE_STALE_CHECK"):
        return
    if not _STALE_SCRIPT.exists():
        # Source/sdist checkout without the dev scripts — nothing to check.
        return

    # We ARE the authoritative interpreter — hand the script the venv root
    # (sys.prefix) so it resolves the venv even under .venv/bin/pytest / an IDE
    # runner / `uv run`, where VIRTUAL_ENV is unset. The activated path
    # (VIRTUAL_ENV set) is unchanged.
    env = {**os.environ, "SHRIKE_NATIVE_VENV": sys.prefix}
    result = subprocess.run(
        ["bash", str(_STALE_SCRIPT)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise pytest.UsageError(
            "shrike_native is stale or unbuilt — run `scripts/build-native.sh` "
            "(pip install does NOT rebuild it). "
            "Set SHRIKE_SKIP_NATIVE_STALE_CHECK=1 to bypass."
        )
