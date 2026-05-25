"""Integration test fixtures.

Spins up a dedicated Shrike MCP server per test session, fully isolated
from any user daemon: own port, own temp collection, own log directory.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


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


@pytest.fixture(scope="session")
def test_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create isolated temp directories for the test session.

    Returns paths for 'root', 'collection', and 'logs'. All are under
    pytest's session-scoped temp directory (cleaned up automatically).
    """
    root = tmp_path_factory.mktemp("shrike")
    log_dir = root / "logs"
    log_dir.mkdir()
    collection_path = root / "collection.anki2"
    return {
        "root": root,
        "collection": collection_path,
        "logs": log_dir,
    }


@pytest.fixture(scope="session")
def server(test_dirs: dict[str, Path]) -> dict:
    """Start an isolated Shrike MCP server for the test session.

    Uses a random free port, a fresh temp collection, and a dedicated
    log directory — no interference with any user daemon.

    Yields a dict with 'url', 'collection_path', 'log_dir', and 'port'.
    """
    port = _free_port()
    collection_path = str(test_dirs["collection"])
    log_dir = str(test_dirs["logs"])
    url = f"http://127.0.0.1:{port}/mcp"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "shrike.server",
            "--collection",
            collection_path,
            "--port",
            str(port),
            "--log-dir",
            log_dir,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server(url)
    except TimeoutError:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        raise RuntimeError(
            f"Server failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        ) from None

    yield {
        "url": url,
        "port": port,
        "collection_path": collection_path,
        "log_dir": log_dir,
    }

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture()
def mcp(server: dict):
    """Return a callable that invokes an MCP tool and returns the structured result.

    Usage:
        result = mcp("collection_info", {})
        result = mcp("upsert_notes", {"notes": [...]})
    """
    url = server["url"]

    def call(tool_name: str, arguments: dict | None = None) -> dict:
        resp = httpx.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {},
                },
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"JSON-RPC error: {body['error']}")
        result: dict = body["result"]["structuredContent"]
        return result

    return call
