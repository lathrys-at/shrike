try:
    # Written at build time by hatch-vcs from the git tag (see pyproject.toml).
    from shrike._version import __version__
except ImportError:  # not built yet — e.g. running from a source tree without an install
    __version__ = "0.0.0+unknown"
