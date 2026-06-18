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

Test sharding (`shard_count` on a target): Bazel splits one target into N parallel
test actions, handing each ``TEST_TOTAL_SHARDS``/``TEST_SHARD_INDEX`` and a
``TEST_SHARD_STATUS_FILE`` it must touch to advertise support. We honour the
protocol with a collection hook that keeps a deterministic round-robin 1/N slice
of the collected items (item-level, so a target's lopsided files still split
evenly), so the union across shards is the full set with no overlap.
"""

from __future__ import annotations

import os
import sys

import pytest


def _partition_for_shard(items: list, shard_index: int, total_shards: int) -> tuple[list, list]:
    """Round-robin partition: return (kept, deselected) for this shard.

    Item-level (not file-level) round-robin so a target whose test files are
    lopsided (e.g. one file with 90 tests, another with 5) still splits into
    balanced shards. Deterministic on pytest's stable collection order, so every
    shard — and, under xdist, every worker within a shard — computes the same
    slice; the union across shards is exactly the full set, disjoint.
    """
    kept = items[shard_index::total_shards]
    keep_ids = {id(it) for it in kept}
    deselected = [it for it in items if id(it) not in keep_ids]
    return kept, deselected


class _BazelShardPlugin:
    """Deselect every collected item not belonging to this Bazel shard."""

    def __init__(self, shard_index: int, total_shards: int) -> None:
        self._shard_index = shard_index
        self._total_shards = total_shards

    def pytest_collection_modifyitems(self, config: pytest.Config, items: list) -> None:
        kept, deselected = _partition_for_shard(items, self._shard_index, self._total_shards)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def _sharding() -> tuple[int, int] | None:
    """Bazel's (shard_index, total_shards) when sharding is in effect, else None.

    Bazel only sets these (and TEST_SHARD_STATUS_FILE) when `shard_count` >= 2, so
    an unsharded target sees nothing here and runs the full set unchanged.
    """
    total = os.environ.get("TEST_TOTAL_SHARDS")
    index = os.environ.get("TEST_SHARD_INDEX")
    if not total or not index:
        return None
    total_shards = int(total)
    if total_shards <= 1:
        return None
    return int(index), total_shards


def main() -> int:
    args = ["-p", "no:cacheprovider", "-o", "asyncio_mode=auto"]

    test_tmpdir = os.environ.get("TEST_TMPDIR")
    if test_tmpdir:
        args += ["--basetemp", os.path.join(test_tmpdir, "pytest")]

    # Honor Bazel's test filter (`--test_filter=expr` -> pytest `-k expr`). Note:
    # Bazel's --test_filter is conventionally a regex, but pytest -k is a
    # substring / boolean-expression grammar — plain test names work; a regex
    # won't mean what you'd expect.
    test_filter = os.environ.get("TESTBRIDGE_TEST_ONLY")
    if test_filter:
        args += ["-k", test_filter]

    # The macro passes the test file path(s) as args.
    args += sys.argv[1:]

    # Under `bazel coverage` (#262), run serially: xdist workers are execnet
    # subprocesses the bootstrap's in-process tracer can't see, so a `-n auto`
    # coverage run silently measures only the controller (~10% instead of ~90%).
    # Coverage runs are off the per-PR hot path, so the serial wall-time cost
    # lands only on the dedicated coverage lane.
    if os.environ.get("COVERAGE_DIR"):
        while "-n" in args:
            i = args.index("-n")
            del args[i : i + 2]

    # Sharding protocol (`shard_count` on the target). Touch the status file FIRST
    # — its mere existence is how the runner advertises sharding support; without
    # it Bazel warns and runs the full suite in every shard (no actual split).
    plugins: list = []
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
        with open(shard_status_file, "w"):
            pass
    sharding = _sharding()
    if sharding is not None:
        shard_index, total_shards = sharding
        plugins.append(_BazelShardPlugin(shard_index, total_shards))

    ret = pytest.main(args, plugins=plugins)
    # A target/shard with no matching test returns NO_TESTS_COLLECTED (5). That is
    # success, not failure, in two cases: Bazel applies --test_filter to every
    # target in a `//pkg/...` pattern (a file with no match is fine), and a shard
    # may legitimately draw an empty slice when shard_count exceeds the test count.
    if ret == pytest.ExitCode.NO_TESTS_COLLECTED and (test_filter or sharding is not None):
        return 0
    return int(ret)


if __name__ == "__main__":
    sys.exit(main())
