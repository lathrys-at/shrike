"""Unit tests for server.py helpers (#79): _maybe_rebuild's rebuild signal."""

from __future__ import annotations

from unittest.mock import MagicMock

from shrike.index import VectorIndex
from shrike.server import _maybe_rebuild


def _index(*, drift: bool) -> MagicMock:
    idx = MagicMock(spec=VectorIndex)
    idx.check_drift.return_value = drift
    return idx


class TestMaybeRebuild:
    def test_starts_rebuild_on_drift(self):
        idx = _index(drift=True)
        assert _maybe_rebuild(idx, "model", 123, [1, 2], ["a", "b"]) is True
        idx.rebuild_in_background.assert_called_once()

    def test_no_rebuild_without_drift(self):
        idx = _index(drift=False)
        assert _maybe_rebuild(idx, "model", 123, [1], ["a"]) is False
        idx.rebuild_in_background.assert_not_called()

    def test_no_rebuild_on_empty_collection_even_if_drifted(self):
        idx = _index(drift=True)
        assert _maybe_rebuild(idx, "model", 123, [], []) is False
        idx.rebuild_in_background.assert_not_called()
