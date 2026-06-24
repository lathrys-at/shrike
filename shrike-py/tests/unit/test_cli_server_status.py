"""Unit coverage for the `shrike server status` CLI command.

Driven with mocked daemon helpers / client / subprocess so no real server is
spawned. Patches target `shrike.cli.server_cmd.*` (where the names are bound).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.schemas import (
    CollectionStatus,
    CoverageCell,
    CoverageMatrix,
    CoverageRow,
    EmbeddingDown,
    EmbeddingRunning,
    IndexBuilding,
    IndexErrored,
    IndexModalityStat,
    IndexProgress,
    IndexReady,
    ServerStatus,
)

SC = "shrike.cli.server_cmd"


@pytest.fixture
def run(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str, **kwargs):
        return runner.invoke(cli, ["--config", str(cfg), *args], **kwargs)

    return _run


def _server(index=None, embedding=None, *, log_dir="/logs", coverage=None, embedding_spaces=None):
    return ServerStatus(
        wire_protocol_version=1,
        pid=4242,
        url="http://127.0.0.1:8372/mcp",
        collection="/c.anki2",
        log_level="info",
        log_dir=log_dir,
        uptime="0:01:00",
        embedding=embedding or EmbeddingRunning(available=True, url="http://e", pid=99, model="/m"),
        embedding_spaces=embedding_spaces or [],
        index=index or IndexReady(state="ready", size=5, ndim=384, col_mod=7, path="/i"),
        coverage=coverage,
    )


class TestServerStatus:
    def test_responsive_renders(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server()
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert result.exit_code == 0
        assert "Server is running" in result.output
        assert "4242" in result.output
        assert "available" in result.output  # embedding
        assert "ready" in result.output  # index

    def test_responsive_json(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server()
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("--json", "server", "status")
        assert result.exit_code == 0
        import json

        assert json.loads(result.output)["pid"] == 4242

    def test_index_building_and_embedding_down(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server(
            index=IndexBuilding(state="building", progress=IndexProgress(indexed=2, total=8)),
            embedding=EmbeddingDown(state="failed"),
        )
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert "building" in result.output
        assert "2 / 8" in result.output
        assert "failed to start" in result.output

    def test_index_error(self, run):
        fake = MagicMock()
        fake.server_status.return_value = _server(index=IndexErrored(state="error", error="boom"))
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert "boom" in result.output

    def test_coverage_matrix_not_in_status(self, run):
        # The coverage matrix lives in `shrike search coverage`, not `server
        # status`: even when /status still carries `coverage`, `server status`
        # must not render it.
        coverage = CoverageMatrix(
            text=CoverageRow(
                text=CoverageCell.NATIVE,
                image=CoverageCell.VIA_DERIVED_TEXT,
                audio=CoverageCell.UNAVAILABLE,
            ),
        )
        fake = MagicMock()
        fake.server_status.return_value = _server(coverage=coverage)
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert result.exit_code == 0
        assert "Coverage" not in result.output
        assert "via text" not in result.output

    def test_status_on_own_line_and_section_order(self, run):
        # The section header is the identity (`Index`, `Embedding`), `Status:` is
        # on its own line, and the order is
        # Index → Derived text → Recognition → Embedding.
        fake = MagicMock()
        fake.server_status.return_value = _server()
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        # Status is on its own line (the header carries no inline status).
        assert "Status:" in out
        assert "Index:" not in out
        assert "Embedding:" not in out
        # Section order: Index before Derived text before Recognition before Embedding.
        i_index = out.index("Index")
        i_derived = out.index("Derived text")
        i_recog = out.index("Recognition")
        i_embed = out.index("Embedding")
        assert i_index < i_derived < i_recog < i_embed

    def test_index_per_modality_breakdown(self, run):
        # Each sub-index reports its own size/ndim.
        fake = MagicMock()
        fake.server_status.return_value = _server(
            index=IndexReady(
                state="ready",
                size=87,
                ndim=768,
                modalities=[
                    IndexModalityStat(modality="text", size=50, ndim=768),
                    IndexModalityStat(modality="image", size=37, ndim=512),
                ],
            )
        )
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        assert "Vectors (text)" in out
        assert "Vectors (image)" in out
        assert "768 dims" in out
        assert "512 dims" in out

    def test_embedding_per_space(self, run):
        # One `Embedding [<modalities>]` block per space.
        fake = MagicMock()
        fake.server_status.return_value = _server(
            embedding=EmbeddingRunning(available=True, model="gemma", modalities=["text"]),
            embedding_spaces=[
                EmbeddingRunning(available=True, model="gemma", modalities=["text"]),
                EmbeddingRunning(available=True, model="clip", modalities=["text", "image"]),
            ],
        )
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        assert "Embedding [text]" in out
        assert "Embedding [text, image]" in out
        assert "gemma" in out
        assert "clip" in out

    def test_running_but_unresponsive(self, run):
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={"pid": 7, "url": "http://x"}),
        ):
            result = run("server", "status")
        assert "not responding" in result.output
        assert "http://x" in result.output

    def test_not_running_exits_1(self, run):
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.META_FILE") as meta_file,
            patch(f"{SC}.cleanup_state"),
        ):
            meta_file.exists.return_value = False
            result = run("server", "status")
        assert result.exit_code == 1
        assert "not running" in result.output.lower()

    def test_not_running_json(self, run):
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.META_FILE") as meta_file,
            patch(f"{SC}.cleanup_state"),
        ):
            meta_file.exists.return_value = False
            result = run("--json", "server", "status")
        import json

        assert json.loads(result.output)["running"] is False


class TestRenderStatusBranches:
    def test_log_and_cooperative_locking_released(self, run):
        # The optional `log` line and the cooperative-locking row (released/idle).
        status = _server()
        status.log = "/logs/shrike.log"
        status.locking = "cooperative"
        status.collection_held = False
        fake = MagicMock()
        fake.server_status.return_value = status
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert result.exit_code == 0, result.output
        assert "Log:" in result.output
        assert "cooperative" in result.output
        assert "released (idle)" in result.output

    def test_cooperative_locking_held(self, run):
        status = _server()
        status.locking = "cooperative"
        status.collection_held = True
        fake = MagicMock()
        fake.server_status.return_value = status
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        assert "collection held" in result.output

    def test_render_status_omits_log_and_uptime_when_absent(self):
        # _render_status is reached with log/uptime already set from the CLI path,
        # so the no-log/no-uptime branches are exercised by calling it directly.
        from shrike.cli.server_cmd import _render_status

        status = _server()
        status.log = None
        status.uptime = None
        buf: list[str] = []
        with (
            patch(f"{SC}.output.kv", side_effect=lambda label, *a, **k: buf.append(label)),
            patch(f"{SC}.status_render"),
        ):
            _render_status(status)
        assert "Log" not in buf
        assert "Uptime" not in buf
        assert "URL" in buf  # the always-present rows still render

    def test_collections_table_rendered(self, run):
        # Multi-collection routing renders the per-collection table with the
        # default marker, idle/released/open states, the (boot) tag, and the
        # footnote about --profile.
        status = _server()
        status.collections = [
            CollectionStatus(
                name="main",
                path="/m.anki2",
                registered=True,
                is_default=True,
                active=True,
                held=True,
                index_state="ready",
            ),
            CollectionStatus(
                name="idle",
                path="/i.anki2",
                registered=True,
                is_default=False,
                active=False,
                held=None,
                index_state=None,
            ),
            CollectionStatus(
                name="released",
                path="/r.anki2",
                registered=True,
                is_default=False,
                active=True,
                held=False,
                index_state="ready",
            ),
            CollectionStatus(
                name="booting",
                path="/b.anki2",
                registered=False,
                is_default=False,
                active=True,
                held=True,
                index_state="building",
            ),
        ]
        fake = MagicMock()
        fake.server_status.return_value = status
        with patch(f"{SC}.ShrikeClient", return_value=fake):
            result = run("server", "status")
        out = result.output
        assert "Collections" in out
        assert "main" in out and "idle" in out and "released" in out and "booting" in out
        assert "not opened" in out  # active=False state
        assert "(boot)" in out  # unregistered tag
        assert "--profile" in out


class TestServerStatusJsonAndMeta:
    def test_unresponsive_json(self, run):
        # Running but unresponsive, --json: emits running=True, responsive=False.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={"pid": 7, "url": "http://x"}),
        ):
            result = run("--json", "server", "status")
        import json

        data = json.loads(result.output)
        assert data["running"] is True and data["responsive"] is False
        assert data["pid"] == 7

    def test_unresponsive_renders_url_and_pid(self, run):
        # The pretty unresponsive branch prints both the URL and the PID rows.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={"pid": 7, "url": "http://x"}),
        ):
            result = run("server", "status")
        assert "URL:" in result.output and "http://x" in result.output
        assert "PID:" in result.output and "7" in result.output

    def test_unresponsive_meta_without_url_or_pid(self, run):
        # Partial meta (neither url nor pid) exercises the negative side of both
        # optional-row branches: the header prints, no URL/PID rows.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=True),
            patch(f"{SC}.read_server_meta", return_value={}),
        ):
            result = run("server", "status")
        assert "not responding" in result.output
        assert "URL:" not in result.output
        assert "PID:" not in result.output

    def test_not_running_cleans_stale_meta_file(self, run):
        # META_FILE present but the daemon is dead → cleanup_state is called.
        fake = MagicMock()
        fake.server_status.return_value = None
        with (
            patch(f"{SC}.ShrikeClient", return_value=fake),
            patch(f"{SC}.is_server_alive", return_value=False),
            patch(f"{SC}.META_FILE") as meta_file,
            patch(f"{SC}.cleanup_state") as cleanup,
        ):
            meta_file.exists.return_value = True
            result = run("server", "status")
        assert result.exit_code == 1
        cleanup.assert_called_once()
