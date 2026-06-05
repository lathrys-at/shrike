"""Unit tests for server.py helpers (#79): _maybe_rebuild's rebuild signal."""

from __future__ import annotations

from unittest.mock import MagicMock

from shrike.embedding import EmbeddingService
from shrike.index import VectorIndex
from shrike.server import _maybe_rebuild


def _index(*, drift: bool) -> MagicMock:
    idx = MagicMock(spec=VectorIndex)
    idx.check_drift.return_value = drift
    return idx


def _embedding(*, dim: int | None = 8) -> MagicMock:
    svc = MagicMock(spec=EmbeddingService)
    svc.embedding_dim.return_value = dim
    return svc


class TestMaybeRebuild:
    def test_starts_reconcile_on_drift(self):
        # Drift reconciles incrementally (reconcile falls back to a full rebuild
        # internally when the model changed or there's no prior per-note state).
        idx = _index(drift=True)
        assert _maybe_rebuild(idx, "model", 123, [1, 2], ["a", "b"], _embedding()) is True
        idx.reconcile_in_background.assert_called_once()
        idx.rebuild_in_background.assert_not_called()
        idx.materialize_empty.assert_not_called()

    def test_no_work_without_drift(self):
        idx = _index(drift=False)
        assert _maybe_rebuild(idx, "model", 123, [1], ["a"], _embedding()) is False
        idx.reconcile_in_background.assert_not_called()
        idx.materialize_empty.assert_not_called()

    def test_empty_collection_materializes_index(self):
        # An empty collection's index is trivially complete: materialize an empty,
        # ready index so later upserts index incrementally (#148). No background
        # work, so the return is still False.
        idx = _index(drift=True)
        svc = _embedding(dim=8)
        assert _maybe_rebuild(idx, "model", 123, [], [], svc) is False
        idx.reconcile_in_background.assert_not_called()
        idx.materialize_empty.assert_called_once_with(8, 123, "model")

    def test_empty_collection_skips_when_dim_unknown(self):
        # If the embedding dimension can't be determined, fall back to the old
        # skip behaviour rather than guessing a width.
        idx = _index(drift=True)
        svc = _embedding(dim=None)
        assert _maybe_rebuild(idx, "model", 123, [], [], svc) is False
        idx.materialize_empty.assert_not_called()
