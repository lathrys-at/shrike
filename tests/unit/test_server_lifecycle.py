"""Unit coverage for server.py module-level helpers and the non-loopback bind
guard. (The async route handlers and full startup are covered by the integration
suite.)"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shrike.index import NoteEmbedInput
from shrike.server import _collect_for_rebuild, _maybe_rebuild, main


class TestCollectForRebuild:
    def test_gathers_ids_mod_and_texts(self, wrapper, basic_note):
        inputs, col_mod = wrapper.run_sync(_collect_for_rebuild)
        assert [i.note_id for i in inputs] == [basic_note]
        assert isinstance(col_mod, int)
        assert len(inputs) == 1
        assert "2+2" in inputs[0].text

    def test_empty_collection(self, wrapper):
        inputs, _col_mod = wrapper.run_sync(_collect_for_rebuild)
        assert inputs == []


class TestMaybeRebuild:
    @staticmethod
    def _embedding(dim: int | None = 8) -> MagicMock:
        svc = MagicMock()
        svc.embedding_dim.return_value = dim
        return svc

    def test_reconciles_on_drift_with_notes(self):
        index = MagicMock()
        index.check_drift.return_value = True
        _maybe_rebuild(
            index,
            "model-1",
            99,
            [NoteEmbedInput(1, "a"), NoteEmbedInput(2, "b")],
            self._embedding(),
        )
        index.reconcile_in_background.assert_called_once_with(
            [NoteEmbedInput(1, "a"), NoteEmbedInput(2, "b")], 99, model_id="model-1"
        )
        index.materialize_empty.assert_not_called()

    def test_no_work_when_no_drift(self):
        index = MagicMock()
        index.check_drift.return_value = False
        _maybe_rebuild(
            index,
            "model-1",
            99,
            [NoteEmbedInput(1, "a"), NoteEmbedInput(2, "b")],
            self._embedding(),
        )
        index.reconcile_in_background.assert_not_called()

    def test_drift_but_empty_collection_materializes(self):
        # An empty collection materializes an empty, ready index (#148) rather
        # than reconciling (nothing to embed) or skipping entirely.
        index = MagicMock()
        index.check_drift.return_value = True
        _maybe_rebuild(index, "model-1", 99, [], self._embedding(8))
        index.reconcile_in_background.assert_not_called()
        index.materialize_empty.assert_called_once_with(8, 99, "model-1")


class TestNonLoopbackGuard:
    def test_refuses_non_loopback_without_allow_remote(self, tmp_path):
        argv = [
            "shrike-server",
            "--collection",
            str(tmp_path / "c.anki2"),
            "--host",
            "0.0.0.0",
        ]
        with (
            patch("sys.argv", argv),
            patch("shrike.server.configure_logging", return_value=tmp_path),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1
