try:
    # Written at build time by hatch-vcs from the git tag (see pyproject.toml) on
    # the pip path, or by the Bazel version stamp on the Bazel path.
    from shrike._version import __version__
except ImportError:
    # No generated _version.py — running from a source tree, or an installed wheel
    # that carries the version in its distribution metadata rather than a file.
    # Fall back to the installed metadata, then a dev sentinel. This keeps
    # __version__ independent of which build system produced the package.
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("shrike-py")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
