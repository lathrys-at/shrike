"""Version consistency checks.

Issue #44 was version drift: ``__version__`` was a hand-maintained constant that
lagged the release tag. The fix (#42) makes the version *derived* from the git
tag via hatch-vcs, so drift is structurally impossible. These tests verify the
wiring: the version is generated (not the import fallback), well-formed, and
consistent with the latest tag.
"""

from __future__ import annotations

import re
import subprocess

import pytest

import shrike


def _latest_release_tag() -> str | None:
    """Most recent ``vX.Y.Z`` tag by creation date, without the ``v``; None if
    git/tags are unavailable (e.g. a shallow checkout with no tags)."""
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
    return tags[0].lstrip("v") if tags else None


def test_version_is_derived() -> None:
    """hatch-vcs wrote a real version into _version.py — not the import fallback."""
    assert shrike.__version__ != "0.0.0+unknown", (
        "shrike._version was not generated — the package wasn't built "
        "(run `pip install -e .`) or hatch-vcs is misconfigured"
    )
    # PEP 440-ish: at least N.N, optionally with a dev/local suffix.
    assert re.match(r"^\d+\.\d+", shrike.__version__), shrike.__version__


def test_version_at_least_latest_tag() -> None:
    """The derived version is >= the latest release tag — it can never lag it
    (the original #44 drift)."""
    latest = _latest_release_tag()
    if latest is None:
        pytest.skip("no git tags available in this checkout")
    version_cls = pytest.importorskip("packaging.version").Version
    assert version_cls(shrike.__version__) >= version_cls(latest)
