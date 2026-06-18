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
    """The build hook generated _version.py — asserted *directly*.

    ``shrike.__version__`` now falls back to installed-distribution metadata when
    ``_version.py`` is absent (#243), so a bare ``!= "0.0.0+unknown"`` check would
    pass on an *unbuilt* checkout that merely has ``shrike-py`` installed
    elsewhere — masking the #44 drift this guard exists to catch. So check
    generation at the source: import the generated module (its absence means the
    build hook didn't run) and confirm the package re-exports exactly it.
    """
    try:
        from shrike._version import __version__ as generated
    except ImportError:
        pytest.fail(
            "shrike._version was not generated — the package wasn't built "
            "(run `pip install -e .`) or hatch-vcs is misconfigured"
        )
    assert shrike.__version__ == generated, "package didn't re-export the generated version"
    assert shrike.__version__ != "0.0.0+unknown"
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


def test_cli_version_flag_works() -> None:
    """`shrike --version` must not depend on the distribution being named
    `shrike`. It broke once when the PyPI name became `shrike-mcp` but Click's
    version_option still looked up metadata for `shrike` (#61). Feeding the
    version directly from __version__ avoids the lookup."""
    from click.testing import CliRunner

    from shrike.cli import cli

    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert shrike.__version__ in result.output
