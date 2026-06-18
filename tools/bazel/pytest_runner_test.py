"""Tests for the Bazel pytest launcher's shard partitioning."""

from __future__ import annotations

import os
import sys

# The launcher lives beside this test in tools/bazel/; ensure its dir is importable
# regardless of how the runfiles tree puts the package on the path.
sys.path.insert(0, os.path.dirname(__file__))

import pytest

import pytest_runner


@pytest.mark.parametrize("total_shards", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("n_items", [0, 1, 5, 50, 234])
def test_partition_is_complete_and_disjoint(n_items: int, total_shards: int) -> None:
    """Every item lands in exactly one shard; the union is the full set."""
    items = list(range(n_items))
    seen: list[int] = []
    for shard_index in range(total_shards):
        kept, deselected = pytest_runner._partition_for_shard(items, shard_index, total_shards)
        # kept and deselected partition the input for this shard.
        assert sorted(kept + deselected) == items
        assert not (set(kept) & set(deselected))
        seen.extend(kept)
    # Across all shards: each item kept exactly once, nothing dropped or duplicated.
    assert sorted(seen) == items


@pytest.mark.parametrize("total_shards", [2, 3, 4])
def test_partition_is_balanced(total_shards: int) -> None:
    """Round-robin keeps shard sizes within one item of each other."""
    items = list(range(234))
    sizes = [
        len(pytest_runner._partition_for_shard(items, i, total_shards)[0])
        for i in range(total_shards)
    ]
    assert max(sizes) - min(sizes) <= 1


def test_partition_uses_identity_not_equality() -> None:
    """Duplicate-valued items (pytest item objects compare oddly) split by identity."""

    class _Item:
        # Intentionally equal under ==, distinct under id() — like collected items
        # whose __eq__ we must not rely on for set membership.
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return 0

    items = [_Item() for _ in range(6)]
    seen_ids: list[int] = []
    for shard_index in range(3):
        kept, _ = pytest_runner._partition_for_shard(items, shard_index, 3)
        seen_ids.extend(id(it) for it in kept)
    assert sorted(seen_ids) == sorted(id(it) for it in items)


def test_sharding_reads_bazel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_TOTAL_SHARDS", raising=False)
    monkeypatch.delenv("TEST_SHARD_INDEX", raising=False)
    assert pytest_runner._sharding() is None

    # shard_count == 1 is "not sharding" — Bazel may still export the vars.
    monkeypatch.setenv("TEST_TOTAL_SHARDS", "1")
    monkeypatch.setenv("TEST_SHARD_INDEX", "0")
    assert pytest_runner._sharding() is None

    monkeypatch.setenv("TEST_TOTAL_SHARDS", "4")
    monkeypatch.setenv("TEST_SHARD_INDEX", "2")
    assert pytest_runner._sharding() == (2, 4)
