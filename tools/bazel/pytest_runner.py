"""pytest entry point for Bazel `py_test` targets (#244).

rules_python has no pytest rule, so each pytest target runs this launcher with its
test file(s) passed as args (see //tools/bazel:defs.bzl `pytest_test`). Config is
supplied explicitly rather than discovered from pyproject.toml, because under
Bazel the rootdir is the runfiles tree, not the source checkout:

- ``asyncio_mode=auto`` mirrors the pyproject setting the suite relies on.
- ``--basetemp`` is pinned under TEST_TMPDIR so pytest's ``tmp_path`` writes stay
  inside Bazel's sandbox (its default basetemp is outside and would be denied).
- conftest.py is discovered normally: the macro puts it in runfiles next to the
  tests, so pytest's per-directory conftest loading finds it.
"""

from __future__ import annotations

import os
import sys

import pytest


def main() -> int:
    args = ["-p", "no:cacheprovider", "-o", "asyncio_mode=auto"]

    test_tmpdir = os.environ.get("TEST_TMPDIR")
    if test_tmpdir:
        args += ["--basetemp", os.path.join(test_tmpdir, "pytest")]

    # Honor Bazel's test filter (`--test_filter=expr` -> -k expr).
    test_filter = os.environ.get("TESTBRIDGE_TEST_ONLY")
    if test_filter:
        args += ["-k", test_filter]

    # The macro passes the test file path(s) as args.
    args += sys.argv[1:]

    return int(pytest.main(args))


if __name__ == "__main__":
    sys.exit(main())
