"""Version consistency checks.

See issue #44: ``shrike.__version__`` (read by ``pyproject.toml`` via
``[tool.hatch.version]``) is a hand-maintained constant that has drifted behind
the latest release tag. This test pins the desired behaviour — the package
version should match the most recent ``vX.Y.Z`` tag — and is marked
``xfail(strict=True)`` until the drift is fixed (durable fix: tag-derived
versioning, #42). When the fix lands the unexpected pass will fail the suite,
forcing this marker's removal (or the test's replacement once the version is
derived from git rather than asserted against it).
"""

from __future__ import annotations

import subprocess

import pytest

import shrike


def _latest_release_tag() -> str | None:
    """Most recent ``vX.Y.Z`` tag by creation date, without the ``v``; None if
    git/tags are unavailable (e.g. a shallow CI checkout)."""
    try:
        out = subprocess.run(
            ["git", "tag", "--sort=-creatordate", "--list", "v[0-9]*"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    tags = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not tags:
        return None
    return tags[0].lstrip("v")


@pytest.mark.xfail(
    strict=True,
    reason="#44: __version__ lags the latest release tag; #42 is the durable fix",
)
def test_version_matches_latest_release_tag() -> None:
    latest = _latest_release_tag()
    if latest is None:
        pytest.skip("no git tags available in this checkout")
    assert shrike.__version__ == latest
