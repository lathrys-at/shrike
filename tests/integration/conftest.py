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

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from shrike.cli import cli
from shrike.client import ShrikeClient
from tests.integration.model_cache import (
    CLIP_MODEL_DIR_NAME,
    DISTILROBERTA_MODEL_DIR_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_URL,
    ONNX_FP32_MODEL_DIR_NAME,
    ONNX_MODEL_DIR_NAME,
    cached_clip_model_dir,
    cached_distilroberta_model_dir,
    cached_model_path,
    cached_onnx_fp32_model_dir,
    cached_onnx_model_dir,
    download_with_retry,
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers pyproject declares, so they're known under Bazel too —
    pyproject's [tool.pytest.ini_options] isn't read when the rootdir is the
    runfiles tree. A harmless duplicate registration on the pip path."""
    config.addinivalue_line("markers", "integration: spawns a server over HTTP")
    config.addinivalue_line("markers", "embedding: requires llama-server and a model")


# -- Bazel: assemble the pinned model externals into a SHRIKE_TEST_MODEL_DIR tree -
#
# Under `bazel test`, the embedding models are pinned http_file externals (see
# MODULE.bazel) rather than HuggingFace downloads. This copies whichever are in
# the test's runfiles into the <dir_name>/<file> layout model_cache expects and
# points SHRIKE_TEST_MODEL_DIR at it — so model_cache finds them present and never
# downloads. No-op off Bazel, when SHRIKE_TEST_MODEL_DIR is already set (the pip
# lane / CI), or for a target with no model externals (the non-embedding suite),
# so the pip path is untouched.
# Dest dir-names come from model_cache's *_DIR_NAME constants (not re-spelled here),
# so a model rename there can't silently drift this map into the wrong layout — which
# would make model_cache miss the assembled file and fall back to a HuggingFace
# download at test time (the #83/#93 flake). Key = the http_file external's runfiles
# path (MODULE.bazel); value = the model_cache <dir>/<file> layout it assembles into.
_BAZEL_MODELS: dict[str, list[str]] = {
    "model_minilm_int8_onnx/file/model.onnx": [f"{ONNX_MODEL_DIR_NAME}/model.onnx"],
    "model_minilm_tokenizer/file/tokenizer.json": [
        f"{ONNX_MODEL_DIR_NAME}/tokenizer.json",
        f"{ONNX_FP32_MODEL_DIR_NAME}/tokenizer.json",
    ],
    "model_minilm_fp32_onnx/file/model.onnx": [f"{ONNX_FP32_MODEL_DIR_NAME}/model.onnx"],
    "model_roberta_int8_onnx/file/model.onnx": [f"{DISTILROBERTA_MODEL_DIR_NAME}/model.onnx"],
    "model_roberta_tokenizer/file/tokenizer.json": [
        f"{DISTILROBERTA_MODEL_DIR_NAME}/tokenizer.json"
    ],
    "model_clip_text_onnx/file/text_model_quantized.onnx": [
        f"{CLIP_MODEL_DIR_NAME}/text_model_quantized.onnx",
    ],
    "model_clip_vision_onnx/file/vision_model_quantized.onnx": [
        f"{CLIP_MODEL_DIR_NAME}/vision_model_quantized.onnx",
    ],
    "model_clip_tokenizer/file/tokenizer.json": [f"{CLIP_MODEL_DIR_NAME}/tokenizer.json"],
    "model_clip_preprocessor/file/preprocessor_config.json": [
        f"{CLIP_MODEL_DIR_NAME}/preprocessor_config.json",
    ],
    f"model_gguf_minilm/file/{EMBEDDING_MODEL_NAME}": [EMBEDDING_MODEL_NAME],
}

# llama-server binary externals (per-platform; only the host's is in a given test's
# runfiles). The conftest puts it on PATH (so the requires_llama_server gate and the
# server's _find_llama_server both see it) + the lib path (the .dylib/.so sit beside
# the binary). No-op off Bazel / when llama-server is already available (pip lane).
_BAZEL_LLAMA_REPOS = (
    "llama_server_macos_arm64",
    "llama_server_macos_amd64",
    "llama_server_linux_amd64",
    "llama_server_linux_arm64",
)


def _setup_bazel_llama_server() -> None:
    if os.environ.get("LLAMA_SERVER_PATH") or shutil.which("llama-server"):
        return
    if not (os.environ.get("RUNFILES_DIR") or os.environ.get("RUNFILES_MANIFEST_FILE")):
        return
    try:
        from python.runfiles import runfiles
    except ImportError:
        return
    r = runfiles.Create()
    if r is None:
        return
    for repo in _BAZEL_LLAMA_REPOS:
        try:
            loc = r.Rlocation(f"{repo}/llama-server")
        except Exception:  # noqa: BLE001 - repo not in this target's mapping
            continue
        if not loc or not os.path.exists(loc):
            continue
        bindir = os.path.dirname(loc)
        os.environ["LLAMA_SERVER_PATH"] = loc
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        # The shared libs are flat beside the binary.
        for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            os.environ[var] = bindir + os.pathsep + os.environ.get(var, "")
        return


_setup_bazel_llama_server()


def _populate_bazel_model_dir() -> None:
    if os.environ.get("SHRIKE_TEST_MODEL_DIR"):
        return  # already provided (pip lane / CI) — respect it
    if not (os.environ.get("RUNFILES_DIR") or os.environ.get("RUNFILES_MANIFEST_FILE")):
        return  # not under Bazel
    try:
        from python.runfiles import runfiles
    except ImportError:
        return
    r = runfiles.Create()
    if r is None:
        return
    # RUNFILES_* are set under `bazel run` too, where TEST_TMPDIR is absent — guard
    # it like everything else in this function rather than KeyError.
    test_tmp = os.environ.get("TEST_TMPDIR")
    if not test_tmp:
        return
    model_root = Path(test_tmp) / "shrike-models"
    found = False
    for src, dests in _BAZEL_MODELS.items():
        loc = r.Rlocation(src)
        if not loc or not os.path.exists(loc):
            continue
        for dest in dests:
            target = model_root / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                # Copy-then-rename so concurrent xdist workers (this module-level
                # code runs once per worker process) can't observe or produce a
                # half-written model file; os.replace is atomic within TEST_TMPDIR.
                tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
                shutil.copy(loc, tmp)
                os.replace(tmp, target)
        found = True
    if found:
        os.environ["SHRIKE_TEST_MODEL_DIR"] = str(model_root)


_populate_bazel_model_dir()

# -- Per-test mutation tracking (drives the cheap collection reset) -----------
#
# The shared-collection reset runs after every test. Enumerating the collection
# (list all notes + list decks/types) on every test is most of its cost. Instead
# the test clients record what a test *mutated*: a read-only test resets to
# nothing, and created notes are deleted by tracked id rather than re-listed.
# The reset still does one `collection_info` to catch auto-created decks and any
# untracked note (e.g. a pretty-mode CLI create whose id we can't parse) — so a
# tracking gap can never leak state, only cost an extra enumeration.

_MCP_READ_TOOLS = frozenset(
    {
        "collection_info",
        "list_notes",
        "search_notes",
        "fetch_media",
        "list_media",
        "collection_check",
    }
)
_CLI_READ_VERBS = frozenset({"list", "show", "search", "status", "logs", "fetch", "check"})


def _cli_noun_verb(args: list[str]) -> tuple[str | None, str | None]:
    """The (noun, verb) of a CLI invocation, ignoring leading output flags."""
    a = list(args)
    while a and a[0] in ("--json", "--no-pretty", "--pretty"):
        a.pop(0)
    return (a[0] if a else None, a[1] if len(a) > 1 else None)


class _ResetTracker:
    """Records the current test's mutations so the reset can skip enumeration."""

    def __init__(self) -> None:
        self.dirty = False
        self.note_ids: set[int] = set()

    def clear(self) -> None:
        self.dirty = False
        self.note_ids.clear()

    def note_results(self, data: object) -> None:
        results = data.get("results", []) if isinstance(data, dict) else []
        for r in results:
            if isinstance(r, dict) and r.get("status") in ("created", "updated") and "id" in r:
                self.note_ids.add(r["id"])


# Module-level: xdist runs each worker in its own process and tests sequentially
# within it, so one shared tracker (cleared per test) is correct.
_reset_tracker = _ResetTracker()


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


def wait_for_index_ready(server: ServerInfo, timeout: float = 60.0) -> dict:
    """Poll /status until the index is ready and non-empty (shared by the
    embedding suites — every test that triggers a rebuild must wait it out
    before returning, or the running rebuild leaks into later tests, #441)."""
    base = server.url.rsplit("/", 1)[0]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        idx = httpx.get(f"{base}/status", timeout=5.0).json().get("index", {})
        if idx.get("state") == "ready" and idx.get("size", 0) > 0:
            return idx
        time.sleep(0.05)
    raise TimeoutError("Index did not become ready")


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
        # Reuse one keep-alive connection: a fresh httpx.post per call pays a TCP
        # connect (~2.4ms) every time, and a session client makes ~700 calls.
        # 30s: an upsert call embeds its whole batch synchronously, and on a
        # cold CI runner the first llama embed also pays the one-time model
        # preset build — 10s was a working-set assumption, not a contract
        # (#441; the seeding upsert hit it).
        self._client = httpx.Client(timeout=30.0)

    def __call__(self, tool_name: str, arguments: dict | None = None) -> dict:
        arguments = dict(arguments or {})
        # Each test class shares one collection, and many reuse first-field
        # values across tests as incidental setup. The server defaults
        # on_duplicate="error", which would reject those repeats — so default
        # setup upserts to "allow". Tests that exercise the duplicate policy
        # pass an explicit on_duplicate and are unaffected.
        if tool_name == "upsert_notes" and "on_duplicate" not in arguments:
            arguments["on_duplicate"] = "allow"
        resp = self._client.post(
            self._url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json"},
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
        if tool_name not in _MCP_READ_TOOLS:
            _reset_tracker.dirty = True
        if tool_name == "upsert_notes":
            _reset_tracker.note_results(structured)
        return structured

    def __del__(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            with suppress(Exception):
                client.close()


class CLIRunner:
    """Click test runner pre-configured to target a specific test server."""

    def __init__(self, url: str, config_path: str) -> None:
        self._runner = CliRunner()
        self._url = url
        self._config = config_path

    def invoke(self, args: list[str], **kwargs: Any) -> Any:
        noun, verb = _cli_noun_verb(args)
        # Anything that isn't a known read command is treated as a mutation
        # (default-dirty is safe — at worst the reset enumerates needlessly).
        if not (noun == "info" or verb in _CLI_READ_VERBS):
            _reset_tracker.dirty = True
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
        if _cli_noun_verb(args)[0] == "note":
            _reset_tracker.note_results(data)  # track ids from `note create/update`
        return data


def _server_launch_cmd() -> list[str]:
    """Base argv to launch a Shrike server, abstracted over the runtime.

    Under Bazel, `-m shrike.server` against the sandbox's import layout isn't
    reliable, so resolve the //bin:server py_binary through runfiles and run it
    directly (it's a data dep of this conftest's target). Under plain pytest, use
    the current interpreter's `-m shrike.server` — unchanged behaviour, so the
    pip path is untouched (coexistence).
    """
    if os.environ.get("RUNFILES_DIR") or os.environ.get("RUNFILES_MANIFEST_FILE"):
        try:
            from python.runfiles import runfiles
        except ImportError:
            pass
        else:
            r = runfiles.Create()
            if r is not None:
                # Prefer the backend-equipped server (onnx/clip in-process) when a
                # test data-deps it; else the lean default (handles llama, which is
                # a subprocess).
                for name in ("_main/bin/server_embedding", "_main/bin/server"):
                    server = r.Rlocation(name)
                    if server and os.path.exists(server):
                        return [server]
    return [sys.executable, "-m", "shrike.server"]


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
        boot_timeout: float | None = None,
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
            *_server_launch_cmd(),
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

        # Under `bazel coverage` the spawned server self-instruments: its
        # rules_python bootstrap sees the inherited COVERAGE_DIR and writes an
        # lcov on exit — but to a FIXED name (pylcov.dat), so the test process
        # and every server would overwrite each other (last writer wins, and
        # the test exits last — which is exactly how server-side lines used to
        # read as uncovered, #262). Bazel's LcovMerger scans COVERAGE_DIR
        # recursively for *.dat, so giving each server its own subdirectory
        # keeps every report and the merge picks them all up. No-op outside
        # coverage runs (COVERAGE_DIR unset).
        env = None
        if os.environ.get("COVERAGE_DIR"):
            subprocess_cov = Path(os.environ["COVERAGE_DIR"]) / f"sub-{name}-{port}"
            subprocess_cov.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "COVERAGE_DIR": str(subprocess_cov)}
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        processes.append(proc)

        # An embedding server's boot includes the llama-server spawn + model
        # load, which on a cold/slow CI runner has blown a 30s ceiling twice
        # in one day (PRs #459/#472, with warm caches) — give it the same
        # generous deadline the collection_server availability poll uses
        # (#444). Costs nothing when boots are fast; the no-embedding case
        # stays tight. ``boot_timeout`` overrides for boots the heuristic
        # can't see (an attach/remote --config boot embeds over HTTP before
        # serving, #498).
        timeout = boot_timeout if boot_timeout is not None else (120.0 if embedding_model else 10.0)
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


def _delete_all_notes(client: ShrikeClient) -> None:
    # `modified_since` an old date matches all notes; paginate since list_notes
    # caps at 200 and a test may have created more. (delete_notes batches >100.)
    while True:
        notes = client.list_notes(modified_since="2000-01-01T00:00:00Z", limit=200).notes
        if not notes:
            break
        client.delete_notes([n.id for n in notes])


def _reset_to_baseline(url: str, baseline: Baseline) -> None:
    """Undo what the current test mutated, returning the collection to *baseline*.

    Read-only tests do nothing. Otherwise: delete the notes we tracked by id (no
    re-listing), then one `collection_info` gives the deck/type lists *and* a note
    count — if any untracked note slipped through (e.g. a pretty-mode CLI create),
    fall back to listing-and-deleting. Finally drop any non-baseline deck / note
    type. Notes go first so decks/(unused) types become deletable."""
    if not _reset_tracker.dirty:
        return  # the test mutated nothing

    baseline_decks, baseline_types = baseline
    client = ShrikeClient(url, autostart=False)

    if _reset_tracker.note_ids:
        client.delete_notes(list(_reset_tracker.note_ids))

    ci = client.collection_info(include=["summary", "decks", "note_types"])
    if ci.summary and ci.summary.notes:
        _delete_all_notes(client)  # safety net for untracked notes

    extra_decks = [d.name for d in (ci.decks or []) if d.name not in baseline_decks]
    if extra_decks:
        client.delete_decks(extra_decks)
    extra_types = [t.id for t in (ci.note_types or []) if t.name not in baseline_types]
    if extra_types:
        client.delete_note_types(extra_types)


def _enumerate_reset(url: str, baseline: Baseline) -> None:
    """Reset by enumeration (independent of the per-test tracker): delete every
    note, then any non-baseline deck / note type. Used by `scoped_collection`."""
    baseline_decks, baseline_types = baseline
    client = ShrikeClient(url, autostart=False)
    _delete_all_notes(client)
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
    test, or on an `isolated_server`. Yields a `ShrikeClient` for convenience.
    Enumeration-based, so it works regardless of how the block mutated state."""
    baseline = _snapshot_baseline(url)
    try:
        yield ShrikeClient(url, autostart=False)
    finally:
        _enumerate_reset(url, baseline)


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
    `collection_server`, or tests using `isolated_server`)."""
    if not {"server", "mcp", "runner", "cli_config"} & set(request.fixturenames):
        yield
        return
    # Capture the pristine baseline before the test body runs (session-scoped,
    # so this snapshots once, on the first shared-server test's setup).
    request.getfixturevalue("_baseline")
    server: ServerInfo = request.getfixturevalue("server")
    _reset_tracker.clear()  # start the test with a clean mutation record
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


def _has_onnxruntime() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


requires_onnxruntime = pytest.mark.skipif(
    not _has_onnxruntime(),
    reason="onnxruntime not installed (pip install onnxruntime)",
)


def _has_clip() -> bool:
    # PIL is a *test* dep here (fixture image authoring) — the backend itself
    # decodes/preprocesses crate-side and needs only the onnxruntime carrier.
    return _has_onnxruntime() and importlib.util.find_spec("PIL") is not None


requires_clip = pytest.mark.skipif(
    not _has_clip(),
    reason="onnxruntime/pillow not installed (pip install onnxruntime pillow)",
)


def _has_shrike_native() -> bool:
    return importlib.util.find_spec("shrike_native") is not None


requires_shrike_native = pytest.mark.skipif(
    not _has_shrike_native(),
    reason="shrike_native extension not installed (scripts/build-native.sh)",
)


@pytest.fixture(scope="session")
def onnx_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small ONNX text-embedding model dir (model.onnx + tokenizer.json).

    Same all-MiniLM-L6-v2 as the GGUF fixture, in ONNX form (384-dim), so the two
    backends share the same vector space and semantic assertions. Downloaded with
    retry/backoff and reused from the CI-cached model dir (``$SHRIKE_TEST_MODEL_DIR``).
    """
    return cached_onnx_model_dir(tmp_path_factory.mktemp("onnx-model"))


@pytest.fixture(scope="session")
def distilroberta_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second, architecturally-different ONNX model: DistilRoBERTa (768-dim, BPE).

    Its own vector space (not comparable to MiniLM) and no ``[PAD]`` token, so it
    exercises the RoBERTa-only paths the MiniLM can't (#172). Same retry/cache path.
    """
    return cached_distilroberta_model_dir(tmp_path_factory.mktemp("distilroberta-model"))


@pytest.fixture(scope="session")
def onnx_fp32_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fp32 (non-quantized) MiniLM ONNX model — batches bit-exact, so it proves the
    batch-safety probe lets a safe model batch (#174). Same retry/cache path."""
    return cached_onnx_fp32_model_dir(tmp_path_factory.mktemp("onnx-fp32-model"))


@pytest.fixture(scope="session")
def clip_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small CLIP (int8 text+vision graphs) for image<->text tests (#162 Phase 3b).

    Use with ``ClipBackend(model=str(clip_model), variant="quantized")``. Same retry/cache path.
    """
    return cached_clip_model_dir(tmp_path_factory.mktemp("clip-model"))


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


# Seed corpus for the shared collection_server: 10 concept clusters x 5 cards.
CONCEPTS: list[dict[str, Any]] = [
    {
        "deck": "Biology",
        "tag": "cell-biology",
        "cards": [
            ("What is a mitochondrion?", "An organelle that produces ATP"),
            ("What is the inner mitochondrial membrane?", "Site of electron transport"),
            ("What is ATP synthase?", "Enzyme that synthesizes ATP using proton gradient"),
            ("What is the citric acid cycle?", "A metabolic pathway in the matrix"),
            ("What is oxidative phosphorylation?", "ATP production via electron transport"),
        ],
    },
    {
        "deck": "Biology",
        "tag": "genetics",
        "cards": [
            ("What is DNA?", "A double-stranded molecule encoding genetic information"),
            ("What is RNA polymerase?", "The enzyme that transcribes DNA into RNA"),
            ("What is a codon?", "A three-nucleotide sequence coding for an amino acid"),
            ("What is mRNA?", "Messenger RNA carries genetic code from DNA to ribosomes"),
            ("What is translation?", "The process of synthesizing protein from mRNA"),
        ],
    },
    {
        "deck": "Biology",
        "tag": "evolution",
        "cards": [
            ("What is natural selection?", "Differential survival and reproduction of organisms"),
            ("What is genetic drift?", "Random changes in allele frequency in a population"),
            ("What is speciation?", "The formation of new and distinct species"),
            ("What is fitness?", "An organism's ability to survive and reproduce"),
            ("What is adaptation?", "A trait that increases fitness in a given environment"),
        ],
    },
    {
        "deck": "Chemistry",
        "tag": "organic",
        "cards": [
            ("What is a covalent bond?", "A chemical bond formed by sharing electron pairs"),
            ("What is an alkane?", "A saturated hydrocarbon with single bonds only"),
            ("What is a functional group?", "An atom or group giving a molecule its properties"),
            ("What is isomerism?", "Molecules with same formula but different structures"),
            ("What is chirality?", "A molecule that is non-superimposable on its mirror image"),
        ],
    },
    {
        "deck": "Chemistry",
        "tag": "thermodynamics",
        "cards": [
            ("What is enthalpy?", "The total heat content of a system at constant pressure"),
            ("What is entropy?", "A measure of disorder or randomness in a system"),
            ("What is Gibbs free energy?", "Energy available to do useful work: G = H - TS"),
            ("What is an exothermic reaction?", "A reaction that releases heat to surroundings"),
            ("What is equilibrium?", "When forward and reverse reaction rates are equal"),
        ],
    },
    {
        "deck": "Physics",
        "tag": "mechanics",
        "cards": [
            ("What is Newton's first law?", "An object at rest stays at rest unless acted on"),
            ("What is momentum?", "The product of an object's mass and velocity"),
            ("What is kinetic energy?", "Energy of motion: KE = 0.5 * m * v^2"),
            ("What is friction?", "A force opposing the relative motion of surfaces"),
            ("What is acceleration?", "The rate of change of velocity over time"),
        ],
    },
    {
        "deck": "Physics",
        "tag": "electromagnetism",
        "cards": [
            ("What is Coulomb's law?", "Force between charges is proportional to q1*q2/r^2"),
            ("What is an electric field?", "A region where a charge experiences a force"),
            ("What is magnetic flux?", "The total magnetic field passing through a surface"),
            ("What is Faraday's law?", "A changing magnetic flux induces an electromotive force"),
            ("What is capacitance?", "The ability to store electric charge: C = Q/V"),
        ],
    },
    {
        "deck": "Mathematics",
        "tag": "calculus",
        "cards": [
            ("What is a derivative?", "The instantaneous rate of change of a function"),
            ("What is an integral?", "The accumulation of quantities over an interval"),
            ("What is the chain rule?", "d/dx[f(g(x))] = f'(g(x)) * g'(x)"),
            ("What is a limit?", "The value a function approaches as input approaches a point"),
            ("What is the fundamental theorem?", "Integration and differentiation are inverses"),
        ],
    },
    {
        "deck": "Mathematics",
        "tag": "linear-algebra",
        "cards": [
            ("What is a matrix?", "A rectangular array of numbers arranged in rows and columns"),
            ("What is an eigenvalue?", "A scalar lambda where Av = lambda*v for some vector v"),
            ("What is a determinant?", "A scalar value computed from a square matrix"),
            ("What is linear independence?", "No vector is a combination of others"),
            ("What is a vector space?", "A set closed under addition and scaling"),
        ],
    },
    {
        "deck": "Computer Science",
        "tag": "algorithms",
        "cards": [
            ("What is Big-O notation?", "Describes the upper bound of an algorithm's growth rate"),
            ("What is a binary search?", "Searching a sorted array by halving the range"),
            ("What is a hash table?", "A structure mapping keys to values via hashing"),
            ("What is recursion?", "A function that calls itself to solve subproblems"),
            ("What is dynamic programming?", "Solving overlapping subproblems"),
        ],
    },
]


@pytest.fixture(scope="session")
def collection_server(server_factory, embedding_model: Path) -> ServerInfo:
    """ONE embedding-enabled server with a 50-note seeded collection, shared by
    every read-only embedding/semantic class (#441 — was two servers: a bare
    `embedding_server` plus a per-module seeded one). Session-scoped: the
    embedding halves run serially (xdist=None in BUILD.bazel), and mutating
    tests clean up after themselves.

    No explicit rebuild: an empty-at-boot server materializes a ready index at
    boot (#148), so the seeding upserts index incrementally.
    """
    srv = server_factory("semantic", embedding_model=str(embedding_model))

    # Poll for embedding availability: on a fresh CI runner llama-server's
    # first boot builds its model-preset cache (~6s, single-threaded), which
    # can stall /status past a single short read timeout (warm locally, so it
    # never reproduces). Mirrors the poll loop of the fixtures this replaced.
    status_url = srv.url.rsplit("/", 1)[0] + "/status"
    status: dict[str, Any] = {}
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            status = httpx.get(status_url, timeout=10.0).json()
        except (httpx.ReadTimeout, httpx.ConnectError):
            time.sleep(0.05)
            continue
        if status.get("embedding", {}).get("available"):
            break
        time.sleep(0.05)
    if not status.get("embedding", {}).get("available"):
        log_dir = Path(srv.log_dir)
        stderr_log = log_dir / "llama-server-stderr.log"
        stderr_content = stderr_log.read_text() if stderr_log.exists() else "(no stderr log)"
        server_log = log_dir / "shrike.log"
        server_content = server_log.read_text() if server_log.exists() else "(no server log)"
        raise RuntimeError(
            f"Embedding service not available within 120s of server start.\n"
            f"Status: {status}\n"
            f"--- llama-server stderr ---\n{stderr_content}\n"
            f"--- shrike server log ---\n{server_content}"
        )

    mcp = MCPClient(srv.url)
    all_notes = []
    for concept in CONCEPTS:
        for front, back in concept["cards"]:
            all_notes.append(
                {
                    "deck": concept["deck"],
                    "note_type": "Basic",
                    "fields": {"Front": front, "Back": back},
                    "tags": [concept["tag"]],
                }
            )
    # Chunked (#441): one 50-note call embeds 50 texts inside a single HTTP
    # call window; chunks keep each call comfortably inside the client timeout
    # even when the runner's first llama embed is cold.
    created = 0
    for i in range(0, len(all_notes), 10):
        result = mcp("upsert_notes", {"notes": all_notes[i : i + 10]})
        created += sum(1 for r in result["results"] if r["status"] == "created")
    assert created == 50, f"Expected 50 notes created, got {created}"
    wait_for_index_ready(srv)
    return srv
