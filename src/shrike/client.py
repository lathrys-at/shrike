"""Standalone Shrike client.

A reusable, dependency-light client for a Shrike server: MCP tool calls over
JSON-RPC, the custom HTTP endpoints (`/status`, `/index/rebuild`,
`/embedding/*`, `/shutdown`), and daemon lifecycle (start/stop/liveness).

This module is deliberately **free of `click`** and of CLI config parsing so it
can be used as a library outside the CLI. Callers that want to auto-start a
local daemon pass a :class:`ServerSpec` describing how to launch it; resolving
that spec from config/env/flags is the caller's concern (the CLI does it).

Errors are raised as typed exceptions (:class:`ShrikeError` subclasses); the CLI
translates them into user-facing messages.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from shrike import daemon

# -- Exceptions --------------------------------------------------------------


class ShrikeError(Exception):
    """Base class for all client-raised errors."""


class ServerError(ShrikeError):
    """The server accepted the request but a tool returned an error."""


class ServerUnreachableError(ShrikeError):
    """The server could not be reached (connection refused or timed out)."""


class ServerHTTPError(ShrikeError):
    """The server returned a non-2xx HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class ServerStartError(ShrikeError):
    """Auto-starting the local daemon failed."""


# -- Launch spec -------------------------------------------------------------


@dataclass
class ServerSpec:
    """How to launch a local Shrike daemon.

    The client stays config-agnostic: the caller resolves these values (from
    config, env, and flags) and hands over a fully-formed spec. ``embedding_args``
    is the already-built list of ``--embedding-*`` / ``--no-embedding`` flags.
    """

    collection: str
    host: str = "127.0.0.1"
    port: int = 8372
    log_dir: str | None = None
    log_level: str = "info"
    state_dir: str | None = None
    cache_dir: str | None = None
    embedding_args: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"


# -- Client ------------------------------------------------------------------


class ShrikeClient:
    """HTTP client for a Shrike server, with optional daemon auto-start.

    If *spec* is provided, the client will auto-start a local daemon on the
    first connection failure (unless *autostart* is False) and retry.
    """

    def __init__(
        self,
        url: str,
        *,
        spec: ServerSpec | None = None,
        autostart: bool = True,
    ) -> None:
        self.url = url
        self.spec = spec
        self.autostart = autostart and spec is not None
        self._request_id = 0
        self._autostarted = False

    @property
    def _base_url(self) -> str:
        """The server root (URL without the trailing ``/mcp``)."""
        return self.url.rsplit("/", 1)[0]

    # -- MCP tool calls ------------------------------------------------------

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke an MCP tool and return its structured result.

        Raises:
            ServerError: the tool returned an error response.
            ServerHTTPError: the server returned a non-2xx status.
            ServerUnreachableError: the server could not be reached.
        """
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        }
        resp = self._post_mcp(payload)
        self._raise_for_status(resp)
        body = resp.json()

        if "error" in body:
            raise ServerError(f"Server error: {body['error']}")

        result = body.get("result", {})
        content = result.get("structuredContent", {})
        if isinstance(content, dict) and "error" in content:
            raise ServerError(content["error"])
        return content  # type: ignore[no-any-return]

    def _post_mcp(self, payload: dict[str, Any], *, timeout: float = 30.0) -> httpx.Response:
        """POST to the MCP endpoint, auto-starting the daemon on first failure."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            return httpx.post(self.url, json=payload, headers=headers, timeout=timeout)
        except httpx.ConnectError as err:
            if self.autostart and not self._autostarted:
                assert self.spec is not None
                self.ensure_running(self.spec)
                try:
                    return httpx.post(self.url, json=payload, headers=headers, timeout=timeout)
                except (httpx.ConnectError, httpx.TimeoutException) as err2:
                    raise ServerUnreachableError(self._unreachable_msg()) from err2
            raise ServerUnreachableError(self._unreachable_msg()) from err
        except httpx.TimeoutException as err:
            raise ServerUnreachableError(f"Request to {self.url} timed out") from err

    # -- Custom HTTP endpoints ----------------------------------------------

    def server_status(self) -> dict[str, Any] | None:
        """Probe ``GET /status``. Returns the status dict, or None if unreachable.

        A non-raising liveness probe — does NOT auto-start.
        """
        try:
            resp = httpx.get(f"{self._base_url}/status", timeout=5.0)
        except (httpx.ConnectError, httpx.TimeoutException):
            return None
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            return data
        return None

    def ping(self) -> bool:
        """True if the server responds to ``/status``. Does not auto-start."""
        return self.server_status() is not None

    def status(self) -> dict[str, Any]:
        """``GET /status`` — raises if unreachable."""
        return self._request("GET", "/status", timeout=5.0)

    def index_status(self) -> dict[str, Any]:
        idx = self.status().get("index")
        return idx if isinstance(idx, dict) else {}

    def embedding_status(self) -> dict[str, Any]:
        emb = self.status().get("embedding")
        return emb if isinstance(emb, dict) else {}

    def index_rebuild(self) -> dict[str, Any]:
        return self._request("POST", "/index/rebuild", timeout=30.0)

    def embedding_start(self, **overrides: Any) -> dict[str, Any]:
        body = {k: v for k, v in overrides.items() if v is not None}
        return self._request("POST", "/embedding/start", json=body, timeout=120.0)

    def embedding_stop(self) -> dict[str, Any]:
        return self._request("POST", "/embedding/stop", timeout=30.0)

    def shutdown(self) -> dict[str, Any]:
        return self._request("POST", "/shutdown", timeout=5.0)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Call a custom endpoint, raising typed errors on failure."""
        try:
            resp = httpx.request(method, f"{self._base_url}{path}", json=json, timeout=timeout)
        except httpx.ConnectError as err:
            raise ServerUnreachableError(self._unreachable_msg()) from err
        except httpx.TimeoutException as err:
            raise ServerUnreachableError(f"Request to {path} timed out") from err
        self._raise_for_status(resp)
        data: dict[str, Any] = resp.json()
        return data

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        message = f"Server returned HTTP {resp.status_code}"
        try:
            body = resp.json()
            if isinstance(body, dict) and body.get("error"):
                message = str(body["error"])
        except ValueError:
            pass
        raise ServerHTTPError(resp.status_code, message)

    def _unreachable_msg(self) -> str:
        return (
            f"Cannot connect to server at {self.url}\n"
            "Is the server running? Start it with: shrike server start"
        )

    # -- Daemon lifecycle ----------------------------------------------------

    def is_alive(self) -> bool:
        """True if a local daemon currently holds the server lock."""
        return daemon.is_server_alive()

    def stop(self, timeout: float = 5.0) -> dict[str, Any]:
        """Stop the local daemon (HTTP → SIGTERM → SIGKILL). Delegates to daemon."""
        return daemon.stop_server(timeout=timeout)

    def wait_until_ready(self, timeout: float = 15.0) -> dict[str, Any] | None:
        """Poll ``/status`` until the daemon responds. Returns status or None."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.server_status()
            if status is not None:
                return status
            time.sleep(0.2)
        return None

    def ensure_running(self, spec: ServerSpec) -> str:
        """Start the local daemon if it isn't already running. Returns the URL.

        Raises ServerStartError if the daemon exits before becoming ready.
        """
        self.url = spec.url
        if daemon.is_server_alive():
            meta = daemon.read_server_meta()
            self._autostarted = True
            if meta and meta.get("url"):
                self.url = str(meta["url"])
            return self.url

        # Clean up any stale state from a crashed server before spawning.
        daemon.cleanup_state()

        proc = self._spawn(spec)
        self._autostarted = True

        if self.wait_until_ready() is None and proc.poll() is not None:
            daemon.cleanup_state()
            raise ServerStartError(
                f"Server process exited with code {proc.returncode}. "
                f"Check the log in {spec.log_dir}."
            )
        return self.url

    def _spawn(self, spec: ServerSpec) -> subprocess.Popen[bytes]:
        """Spawn the daemon subprocess. The bootstrap log handle is closed in the
        parent right after spawn — the child keeps its own dup'd fd (fixes the
        leaked-handle audit item)."""
        cmd = [
            sys.executable,
            "-m",
            "shrike.server",
            "--collection",
            spec.collection,
            "--port",
            str(spec.port),
            "--host",
            spec.host,
            "--log-level",
            spec.log_level,
        ]
        if spec.log_dir:
            cmd += ["--log-dir", spec.log_dir]
        if spec.state_dir:
            cmd += ["--state-dir", spec.state_dir]
        if spec.cache_dir:
            cmd += ["--cache-dir", spec.cache_dir]
        cmd += spec.embedding_args

        daemon.STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_dir = Path(spec.log_dir) if spec.log_dir else daemon.STATE_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / "shrike-bootstrap.log", "a") as bootstrap_log:
            return subprocess.Popen(
                cmd,
                stdout=bootstrap_log,
                stderr=bootstrap_log,
                start_new_session=True,
            )

    # -- Convenience tool wrappers ------------------------------------------

    def collection_info(
        self,
        include: list[str] | None = None,
        note_type_details: list[str] | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if include:
            args["include"] = include
        if note_type_details:
            args["note_type_details"] = note_type_details
        return self.call("collection_info", args)

    def list_notes(self, **kwargs: Any) -> dict[str, Any]:
        args = {k: v for k, v in kwargs.items() if v is not None}
        return self.call("list_notes", args)

    def search_notes(self, **kwargs: Any) -> dict[str, Any]:
        args = {k: v for k, v in kwargs.items() if v is not None}
        return self.call("search_notes", args)

    def upsert_notes(self, notes: list[dict]) -> dict[str, Any]:
        """Upsert notes, transparently batching if over the server limit."""
        return self._batched_call(
            "upsert_notes",
            items=notes,
            param_key="notes",
            result_key="results",
            batch_size=100,
        )

    def upsert_note_types(self, note_types: list[dict]) -> dict[str, Any]:
        return self._batched_call(
            "upsert_note_types",
            items=note_types,
            param_key="note_types",
            result_key="results",
            batch_size=10,
        )

    def delete_note_types(self, ids: list[int]) -> dict[str, Any]:
        return self.call("delete_note_types", {"ids": ids})

    def delete_notes(self, ids: list[int]) -> dict[str, Any]:
        """Delete notes, transparently batching if over the server limit."""
        if len(ids) <= 100:
            return self.call("delete_notes", {"ids": ids})

        all_deleted: list[int] = []
        all_not_found: list[int] = []
        for i in range(0, len(ids), 100):
            chunk = ids[i : i + 100]
            result = self.call("delete_notes", {"ids": chunk})
            all_deleted.extend(result.get("deleted", []))
            all_not_found.extend(result.get("not_found", []))
        return {"deleted": all_deleted, "not_found": all_not_found}

    def _batched_call(
        self,
        tool_name: str,
        *,
        items: list,
        param_key: str,
        result_key: str,
        batch_size: int,
    ) -> dict[str, Any]:
        """Split a list of items into batches and merge the results."""
        if len(items) <= batch_size:
            return self.call(tool_name, {param_key: items})

        all_results: list = []
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            result = self.call(tool_name, {param_key: chunk})
            all_results.extend(result.get(result_key, []))
        return {result_key: all_results}
