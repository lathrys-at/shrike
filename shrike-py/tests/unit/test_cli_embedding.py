"""Unit coverage for the `shrike server embedding` CLI commands.

These commands are thin clients: they call `ShrikeClient` methods and render the
typed responses. In CI they're otherwise only reached by the embedding-gated
lane (which doesn't count toward the coverage gate), so here we drive them with a
mocked client — no real server, no llama-server — via Click's CliRunner.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.client import ShrikeClient
from shrike.schemas import (
    EmbeddingAlreadyRunning,
    EmbeddingDown,
    EmbeddingNotRunning,
    EmbeddingRunning,
    EmbeddingStarted,
    EmbeddingStopped,
    IndexBuilding,
    IndexProgress,
    IndexReady,
    IndexUnavailable,
    ServerStatus,
)


@pytest.fixture
def fake() -> MagicMock:
    return MagicMock(spec=ShrikeClient)


@pytest.fixture
def run(tmp_path, fake):
    """Invoke the CLI with the client patched to `fake` and an empty config."""
    cfg = tmp_path / "config.yml"
    cfg.write_text("")
    runner = CliRunner()

    def _run(*args: str):
        # Patch at the source: the CLI now imports ShrikeClient lazily inside the
        # root callback (so no-server commands don't pull httpx), so it's no
        # longer a `shrike.cli` attribute.
        with patch("shrike.client.ShrikeClient", return_value=fake):
            return runner.invoke(cli, ["--config", str(cfg), *args], catch_exceptions=False)

    return _run


def _server(index, embedding=None, embedding_spaces=None):
    return ServerStatus(
        wire_protocol_version=1,
        pid=1,
        url="http://127.0.0.1:8372/mcp",
        collection="/c.anki2",
        log_level="info",
        log_dir="/logs",
        embedding=embedding or EmbeddingRunning(available=True),
        embedding_spaces=embedding_spaces or [],
        index=index,
    )


class TestEmbeddingStatus:
    # `embedding status` reads the full /status, so it mocks `server_status`
    # and the embedding shows up under `embedding_spaces`.
    def test_available(self, run, fake):
        fake.server_status.return_value = _server(
            IndexReady(state="ready", size=5, ndim=384),
            embedding=EmbeddingRunning(
                available=True, pid=123, url="http://127.0.0.1:8373", model="/m.gguf"
            ),
        )
        result = run("server", "embedding", "status")
        assert "available" in result.output
        assert "123" in result.output
        assert "/m.gguf" in result.output

    def test_per_space(self, run, fake):
        # A two-space profile: each space is its own `Embedding […]`
        # block keyed by modalities — both must appear.
        fake.server_status.return_value = _server(
            IndexReady(state="ready", size=87, ndim=768),
            embedding=EmbeddingRunning(available=True, model="gemma", modalities=["text"]),
            embedding_spaces=[
                EmbeddingRunning(available=True, model="gemma", modalities=["text"]),
                EmbeddingRunning(available=True, model="clip", modalities=["text", "image"]),
            ],
        )
        result = run("server", "embedding", "status")
        assert "gemma" in result.output
        assert "clip" in result.output
        assert "[text]" in result.output
        assert "[text, image]" in result.output

    @pytest.mark.parametrize(
        "state,needle",
        [("failed", "failed"), ("stopped", "stopped"), ("not_configured", "not configured")],
    )
    def test_down_states(self, run, fake, state, needle):
        fake.server_status.return_value = _server(
            IndexUnavailable(), embedding=EmbeddingDown(state=state)
        )
        result = run("server", "embedding", "status")
        assert needle in result.output

    def test_json(self, run, fake):
        fake.server_status.return_value = _server(
            IndexReady(state="ready", size=1, ndim=8),
            embedding=EmbeddingRunning(available=True, pid=1),
        )
        result = run("--json", "server", "embedding", "status")
        # The full per-space list; a single-space server emits a one-element list.
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["state"] == "running"


class TestEmbeddingStart:
    def test_already_running(self, run, fake):
        fake.embedding_start.return_value = EmbeddingAlreadyRunning(
            status="already_running", embedding=EmbeddingRunning(available=True)
        )
        result = run("server", "embedding", "start")
        assert "already running" in result.output.lower()

    def test_started_not_building(self, run, fake):
        fake.embedding_start.return_value = EmbeddingStarted(
            status="started",
            embedding=EmbeddingRunning(available=True, model="/m.gguf"),
            index=IndexUnavailable(),
        )
        result = run("server", "embedding", "start")
        assert "started" in result.output.lower()
        assert "/m.gguf" in result.output

    def test_started_building_message(self, run, fake):
        fake.embedding_start.return_value = EmbeddingStarted(
            status="started",
            embedding=EmbeddingRunning(available=True),
            index=IndexBuilding(state="building", progress=IndexProgress(indexed=0, total=5)),
        )
        result = run("server", "embedding", "start", "--background")
        assert "rebuild started in the background" in result.output.lower()

    def test_started_building_polls(self, run, fake):
        fake.embedding_start.return_value = EmbeddingStarted(
            status="started",
            embedding=EmbeddingRunning(available=True),
            index=IndexBuilding(state="building", progress=IndexProgress(indexed=0, total=2)),
        )
        fake.server_status.return_value = _server(IndexReady(state="ready", size=2, ndim=8))
        fake.embedding_status.return_value = EmbeddingRunning(available=True, model="/m.gguf")
        result = run("server", "embedding", "start")
        assert result.exit_code == 0
        assert "Index ready" in result.output

    def test_json_already_running(self, run, fake):
        fake.embedding_start.return_value = EmbeddingAlreadyRunning(
            status="already_running", embedding=EmbeddingRunning(available=True)
        )
        result = run("--json", "server", "embedding", "start")
        assert json.loads(result.output)["status"] == "already_running"


class TestEmbeddingStop:
    def test_stopped(self, run, fake):
        fake.embedding_stop.return_value = EmbeddingStopped(
            status="stopped", index=IndexUnavailable()
        )
        result = run("server", "embedding", "stop")
        assert "stopped" in result.output.lower()

    def test_not_running(self, run, fake):
        fake.embedding_stop.return_value = EmbeddingNotRunning(status="not_running")
        result = run("server", "embedding", "stop")
        assert "not running" in result.output.lower()

    def test_json(self, run, fake):
        fake.embedding_stop.return_value = EmbeddingStopped(
            status="stopped", index=IndexUnavailable()
        )
        result = run("--json", "server", "embedding", "stop")
        assert json.loads(result.output)["status"] == "stopped"
