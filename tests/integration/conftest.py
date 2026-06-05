"""Integration test fixtures.

Provides a ``server_factory`` that spins up isolated Shrike MCP servers
on demand — each with its own port, temp collection, and log directory.
Test classes request their own server via class-scoped fixtures so no
test state leaks between classes.

Embedding tests (marked ``@pytest.mark.embedding``) additionally require
``llama-server`` on PATH and a small GGUF model. The model is downloaded
once per CI run and cached in the pytest tmp directory.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.client import ShrikeClient
from tests.integration.model_cache import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_URL,
    cached_model_path,
    download_with_retry,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0.0.0"},
                    },
                },
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=2.0,
            )
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


class ServerInfo:
    """Connection details for a running test server."""

    def __init__(
        self,
        url: str,
        port: int,
        collection_path: str,
        log_dir: str,
        embedding_port: int | None = None,
    ) -> None:
        self.url = url
        self.port = port
        self.collection_path = collection_path
        self.log_dir = log_dir
        self.embedding_port = embedding_port
        self.embedding_url = f"http://127.0.0.1:{embedding_port}" if embedding_port else None


class MCPClient:
    """Thin wrapper that calls MCP tools over HTTP and returns structured results."""

    def __init__(self, url: str) -> None:
        self._url = url

    def __call__(self, tool_name: str, arguments: dict | None = None) -> dict:
        arguments = dict(arguments or {})
        # Each test class shares one collection, and many reuse first-field
        # values across tests as incidental setup. The server defaults
        # on_duplicate="error", which would reject those repeats — so default
        # setup upserts to "allow". Tests that exercise the duplicate policy
        # pass an explicit on_duplicate and are unaffected.
        if tool_name == "upsert_notes" and "on_duplicate" not in arguments:
            arguments["on_duplicate"] = "allow"
        resp = httpx.post(
            self._url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=10.0,
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"JSON-RPC error: {body['error']}")
        result = body["result"]
        if result.get("isError"):
            # Input-validation / execution errors carry text content, not
            # structuredContent — surface the message so tests can assert on it.
            content = result.get("content") or []
            text = next(
                (c.get("text") for c in content if isinstance(c, dict) and c.get("text")),
                "tool error",
            )
            raise RuntimeError(text)
        structured: dict = result["structuredContent"]
        return structured


class CLIRunner:
    """Click test runner pre-configured to target a specific test server."""

    def __init__(self, url: str, config_path: str) -> None:
        self._runner = CliRunner()
        self._url = url
        self._config = config_path

    def invoke(self, args: list[str], **kwargs: Any) -> Any:
        return self._runner.invoke(
            cli,
            ["--config", self._config, "--url", self._url, *args],
            catch_exceptions=False,
            **kwargs,
        )

    def json(self, args: list[str], **kwargs: Any) -> dict:
        result = self.invoke(["--json", *args], **kwargs)
        assert result.exit_code == 0, result.output
        data: dict = json.loads(result.output)
        return data


@pytest.fixture(scope="session")
def server_factory(tmp_path_factory: pytest.TempPathFactory):
    """Factory that creates isolated server instances.

    Each call spins up a new server with its own collection, log dir,
    and random port. All servers are torn down at session end.

    Pass ``embedding_model`` to start with ``--embedding-model``.
    """
    processes: list[subprocess.Popen] = []

    def create(
        name: str = "server",
        *,
        embedding_model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> ServerInfo:
        root = tmp_path_factory.mktemp(name)
        log_dir = root / "logs"
        log_dir.mkdir()
        state_dir = root / "state"
        state_dir.mkdir()
        cache_dir = root / "cache"
        cache_dir.mkdir()
        collection_path = str(root / "collection.anki2")

        port = _free_port()
        url = f"http://127.0.0.1:{port}/mcp"

        cmd = [
            sys.executable,
            "-m",
            "shrike.server",
            "--collection",
            collection_path,
            "--port",
            str(port),
            "--log-dir",
            str(log_dir),
            "--state-dir",
            str(state_dir),
            "--cache-dir",
            str(cache_dir),
            # Short debounce so any test that waits on a live index flush (rather
            # than shutdown persistence) doesn't sit on the 60s production default.
            "--index-save-delay",
            "5",
        ]

        embedding_port: int | None = None
        if embedding_model:
            embedding_port = _free_port()
            cmd.extend(
                [
                    "--embedding-model",
                    embedding_model,
                    "--embedding-port",
                    str(embedding_port),
                ]
            )

        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append(proc)

        timeout = 30.0 if embedding_model else 10.0
        try:
            _wait_for_server(url, timeout=timeout)
        except TimeoutError:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
            raise RuntimeError(
                f"Server '{name}' failed to start.\n"
                f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            ) from None

        return ServerInfo(url, port, collection_path, str(log_dir), embedding_port)

    yield create

    # Signal every server first, then wait — so the per-server shutdowns (each
    # stopping its own llama-server child) overlap instead of running serially.
    for proc in processes:
        proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# -- Shared server + per-test collection reset --------------------------------
#
# Spawning a `python -m shrike.server` subprocess per test class dominated the
# integration suite (each boots anki under coverage). Instead all non-embedding
# tests share ONE server per xdist worker, and the collection is reset to its
# pristine baseline after every test (`_reset_shared_collection`, autouse) so a
# test always starts clean — which keeps even collection-wide assertions
# (`total_notes == 0`) valid regardless of run order. Tests that genuinely need
# their own exclusive collection opt into `isolated_server`.

Baseline = tuple[frozenset[str], frozenset[str]]  # (deck names, note-type names)


def _snapshot_baseline(url: str) -> Baseline:
    client = ShrikeClient(url, autostart=False)
    ci = client.collection_info(include=["decks", "note_types"])
    return (
        frozenset(d.name for d in (ci.decks or [])),
        frozenset(t.name for t in (ci.note_types or [])),
    )


def _reset_to_baseline(url: str, baseline: Baseline) -> None:
    """Return a collection to *baseline*: delete every note, then any deck or
    note type not in the baseline. Notes go first so the decks/(unused) note
    types become deletable. Best-effort and idempotent."""
    baseline_decks, baseline_types = baseline
    client = ShrikeClient(url, autostart=False)

    # `modified_since` an old date matches all notes; paginate since list_notes
    # caps at 200 and a test may have created more. (delete_notes batches >100.)
    while True:
        notes = client.list_notes(modified_since="2000-01-01T00:00:00Z", limit=200).notes
        if not notes:
            break
        client.delete_notes([n.id for n in notes])

    ci = client.collection_info(include=["decks", "note_types"])
    extra_decks = [d.name for d in (ci.decks or []) if d.name not in baseline_decks]
    if extra_decks:
        client.delete_decks(extra_decks)
    extra_types = [t.id for t in (ci.note_types or []) if t.name not in baseline_types]
    if extra_types:
        client.delete_note_types(extra_types)


@contextmanager
def scoped_collection(url: str) -> Iterator[ShrikeClient]:
    """Explicit scope: snapshot the collection on enter, undo any notes / non-
    baseline decks / note types created in the block on exit. Shared-server tests
    get this automatically via the autouse reset; use this to sub-scope within a
    test, or on an `isolated_server`. Yields a `ShrikeClient` for convenience."""
    baseline = _snapshot_baseline(url)
    try:
        yield ShrikeClient(url, autostart=False)
    finally:
        _reset_to_baseline(url, baseline)


@pytest.fixture(scope="session")
def server(server_factory) -> ServerInfo:
    """Session-scoped server shared by all non-embedding tests (one boot per
    xdist worker). The autouse `_reset_shared_collection` resets its collection
    between tests; use `isolated_server` for an exclusive one."""
    return server_factory("session")


@pytest.fixture(scope="session")
def _baseline(server: ServerInfo) -> Baseline:
    """Pristine decks/note-types, snapshotted once before any test mutates."""
    return _snapshot_baseline(server.url)


@pytest.fixture(autouse=True)
def _reset_shared_collection(request: pytest.FixtureRequest) -> Iterator[None]:
    """Reset the shared collection to baseline after each test that used it.
    Skipped for tests that don't touch the shared server (embedding tests on
    `embedding_server`, or tests using `isolated_server`)."""
    if not {"server", "mcp", "runner", "cli_config"} & set(request.fixturenames):
        yield
        return
    # Capture the pristine baseline before the test body runs (session-scoped,
    # so this snapshots once, on the first shared-server test's setup).
    request.getfixturevalue("_baseline")
    server: ServerInfo = request.getfixturevalue("server")
    yield
    _reset_to_baseline(server.url, request.getfixturevalue("_baseline"))


@pytest.fixture(scope="session")
def mcp(server: ServerInfo) -> MCPClient:
    """MCP tool caller bound to the shared server."""
    return MCPClient(server.url)


def _write_cli_config(server: ServerInfo, tmp_path_factory: pytest.TempPathFactory) -> Path:
    config_dir = tmp_path_factory.mktemp("cli-config")
    config_path = config_dir / "config.yml"
    config_path.write_text(
        f"server:\n"
        f"  host: 127.0.0.1\n"
        f"  port: {server.port}\n"
        f"collection: {server.collection_path}\n"
        f"logging:\n"
        f"  dir: {server.log_dir}\n"
    )
    return config_path


@pytest.fixture(scope="session")
def cli_config(server: ServerInfo, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Config file pointing at the shared server."""
    return _write_cli_config(server, tmp_path_factory)


@pytest.fixture(scope="session")
def runner(server: ServerInfo, cli_config: Path) -> CLIRunner:
    """CLI test runner bound to the shared server."""
    return CLIRunner(server.url, str(cli_config))


# -- Opt-in isolation: a dedicated, exclusive collection for one test ----------


@pytest.fixture
def isolated_server(server_factory) -> ServerInfo:
    """A fresh server/collection for a single test — for the rare cases needing a
    pristine, exclusive collection that the autouse reset must not touch. Spawns
    a server, so prefer the shared `server` unless isolation is genuinely needed."""
    return server_factory("isolated")


@pytest.fixture
def isolated_mcp(isolated_server: ServerInfo) -> MCPClient:
    """MCP caller bound to a dedicated `isolated_server`."""
    return MCPClient(isolated_server.url)


@pytest.fixture
def isolated_runner(
    isolated_server: ServerInfo, tmp_path_factory: pytest.TempPathFactory
) -> CLIRunner:
    """CLI runner bound to a dedicated `isolated_server`."""
    return CLIRunner(isolated_server.url, str(_write_cli_config(isolated_server, tmp_path_factory)))


# -- Embedding fixtures --


def _has_llama_server() -> bool:
    return shutil.which("llama-server") is not None


requires_llama_server = pytest.mark.skipif(
    not _has_llama_server(),
    reason="llama-server not found on PATH",
)


@pytest.fixture(scope="session")
def embedding_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide a small embedding model for tests.

    Reuses an already-downloaded copy (a stable, CI-cached dir via
    ``$SHRIKE_TEST_MODEL_DIR``, else a per-session temp dir) and downloads with
    retry/backoff so a transient HuggingFace 429 doesn't fail the lane (#83).
    """
    model_path = cached_model_path(EMBEDDING_MODEL_NAME, tmp_path_factory.mktemp("models"))
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    return download_with_retry(EMBEDDING_MODEL_URL, model_path)


@pytest.fixture(scope="session")
def embedding_server(server_factory, embedding_model: Path) -> ServerInfo:
    """Server with embedding service enabled.

    Session-scoped: its consumers (`TestEmbeddingHealth`, `TestEmbeddings`,
    `TestEmbeddingServiceViaShrike`) are read-only against a stateless embedding
    endpoint, so one llama-server boot serves them all — avoids a model load (and
    a teardown) per class.

    Verifies the embedding service is actually available after server start.
    Fails with diagnostics if it isn't (e.g. shared lib issues, model problems).
    """
    srv = server_factory("embedding", embedding_model=str(embedding_model))

    status_url = srv.url.rsplit("/", 1)[0] + "/status"
    resp = httpx.get(status_url, timeout=5.0)
    status = resp.json()
    emb = status.get("embedding", {})
    if not emb.get("available"):
        log_dir = Path(srv.log_dir)
        stderr_log = log_dir / "llama-server-stderr.log"
        stderr_content = stderr_log.read_text() if stderr_log.exists() else "(no stderr log)"
        server_log = log_dir / "shrike.log"
        server_content = server_log.read_text() if server_log.exists() else "(no server log)"
        raise RuntimeError(
            f"Embedding service not available after server start.\n"
            f"Status: {status}\n"
            f"--- llama-server stderr ---\n{stderr_content}\n"
            f"--- shrike server log ---\n{server_content}"
        )

    return srv
