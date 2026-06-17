"""Unit coverage for server.py module-level helpers and the non-loopback bind
guard. (The async route handlers and full startup are covered by the integration
suite.)"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from shrike.harness.collection import collect_embed_inputs
from shrike.server import main


class TestCollectForRebuild:
    def test_gathers_ids_mod_and_texts(self, wrapper, basic_note):
        inputs, col_mod = wrapper.run_sync(collect_embed_inputs)
        assert [i.note_id for i in inputs] == [basic_note]
        assert isinstance(col_mod, int)
        assert len(inputs) == 1
        assert "2+2" in inputs[0].text

    def test_empty_collection(self, wrapper):
        inputs, _col_mod = wrapper.run_sync(collect_embed_inputs)
        assert inputs == []


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
            patch("shrike.server.server.configure_logging", return_value=tmp_path),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1
