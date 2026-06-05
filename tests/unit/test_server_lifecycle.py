"""Unit coverage for server.py module-level helpers and the non-loopback bind
guard. (The async route handlers and full startup are covered by the integration
suite.)"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shrike.server import _collect_for_rebuild, _maybe_rebuild, main


class TestCollectForRebuild:
    def test_gathers_ids_mod_and_texts(self, wrapper, basic_note):
        note_ids, col_mod, texts = wrapper.run_sync(_collect_for_rebuild)
        assert note_ids == [basic_note]
        assert isinstance(col_mod, int)
        assert len(texts) == 1
        assert "2+2" in texts[0]

    def test_empty_collection(self, wrapper):
        note_ids, _col_mod, texts = wrapper.run_sync(_collect_for_rebuild)
        assert note_ids == []
        assert texts == []


class TestMaybeRebuild:
    def test_rebuilds_on_drift_with_notes(self):
        index = MagicMock()
        index.check_drift.return_value = True
        _maybe_rebuild(index, "model-1", 99, [1, 2], ["a", "b"])
        index.rebuild_in_background.assert_called_once_with(
            [1, 2], ["a", "b"], 99, model_id="model-1"
        )

    def test_no_rebuild_when_no_drift(self):
        index = MagicMock()
        index.check_drift.return_value = False
        _maybe_rebuild(index, "model-1", 99, [1, 2], ["a", "b"])
        index.rebuild_in_background.assert_not_called()

    def test_drift_but_empty_collection_skips(self):
        index = MagicMock()
        index.check_drift.return_value = True
        _maybe_rebuild(index, "model-1", 99, [], [])
        index.rebuild_in_background.assert_not_called()


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
