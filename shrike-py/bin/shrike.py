"""Launcher for the Shrike CLI (`bazel run //shrike-py/bin:shrike`).

Thin entry point kept out of the `shrike` package. Mirrors the packaged console
script `shrike = shrike.cli:cli` (pyproject).
"""

from shrike.cli import cli

if __name__ == "__main__":
    cli()
