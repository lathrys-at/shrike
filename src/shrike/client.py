"""Standalone Shrike client.

A reusable, dependency-light client for a Shrike server: MCP tool calls over
JSON-RPC, the custom HTTP endpoints (`/status`, `/index/rebuild`,
`/embedding/*`, `/shutdown`), and daemon lifecycle (start/stop/liveness).

This module is deliberately **free of `click`** and of CLI config parsing so it
can be used as a library outside the CLI. Callers that want to auto-start a
local daemon pass a :class:`ServerSpec` describing how to launch it; resolving
that spec from config/env/flags is the caller's concern (the CLI does it).

Tool and status methods return the Pydantic models from :mod:`shrike.schemas`
(the wire contract's single source of truth). The untyped escape hatch is
:meth:`ShrikeClient._call`, for tools not yet wrapped in a typed method.

Errors are raised as typed exceptions (:class:`ShrikeError` subclasses); the CLI
translates them into user-facing messages.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from shrike import daemon
from shrike.schemas import (
    CollectionInfo,
    DeleteNotesResponse,
    DeleteNoteTypesResponse,
    EmbeddingStartResponse,
    EmbeddingStatus,
    EmbeddingStopResponse,
    IndexRebuildResponse,
    IndexStatus,
    ListNotesResponse,
    NoteInput,
    NoteTypeInput,
    SearchResponse,
    ServerStatus,
    ShutdownResponse,
    StopResponse,
    UpsertNotesResponse,
    UpsertNoteTypesResponse,
)

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


def _error_text(content: Any) -> str | None:
    """Extract the first text payload from an MCP content list, if any."""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                return str(item["text"])
    return None


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

    def _call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke an MCP tool and return its raw structured result.

        This is the untyped escape hatch — the typed convenience methods
        (``list_notes``, ``search_notes``, …) wrap it and validate the result
        into a response model. Reach for ``_call`` directly only when a tool has
        no typed wrapper.

        Raises:
            ServerError: the tool failed — an MCP ``isError`` result (bad input
                or an unhandled exception) or a JSON-RPC error.
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
        if result.get("isError"):
            # Tool failure: the message lives in the text content. Tools no
            # longer embed an error field in structuredContent — failures are
            # MCP isError results.
            raise ServerError(_error_text(result.get("content")) or "Tool returned an error")

        content = result.get("structuredContent", {})
        return content if isinstance(content, dict) else {}

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

    # -- Typed tool wrappers -------------------------------------------------

    def collection_info(
        self,
        include: list[str] | None = None,
        note_type_details: list[str] | None = None,
    ) -> CollectionInfo:
        args: dict[str, Any] = {}
        if include:
            args["include"] = include
        if note_type_details:
            args["note_type_details"] = note_type_details
        return CollectionInfo.model_validate(self._call("collection_info", args))

    def list_notes(
        self,
        *,
        ids: list[int] | None = None,
        deck: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
        modified_since: str | None = None,
        query: str | None = None,
        fields: str | None = None,
        limit: int = 50,
    ) -> ListNotesResponse:
        args: dict[str, Any] = {"limit": limit}
        for key, value in (
            ("ids", ids),
            ("deck", deck),
            ("tags", tags),
            ("note_type", note_type),
            ("modified_since", modified_since),
            ("query", query),
            ("fields", fields),
        ):
            if value is not None:
                args[key] = value
        return ListNotesResponse.model_validate(self._call("list_notes", args))

    def search_notes(
        self,
        *,
        queries: list[str] | None = None,
        ids: list[int] | None = None,
        top_k: int = 10,
        threshold: float = 0.5,
        deck: str | None = None,
        tags: list[str] | None = None,
        exclude_ids: list[int] | None = None,
    ) -> SearchResponse:
        args: dict[str, Any] = {"top_k": top_k, "threshold": threshold}
        for key, value in (
            ("queries", queries),
            ("ids", ids),
            ("deck", deck),
            ("tags", tags),
            ("exclude_ids", exclude_ids),
        ):
            if value is not None:
                args[key] = value
        return SearchResponse.model_validate(self._call("search_notes", args))

    def upsert_notes(
        self,
        notes: Sequence[NoteInput | dict[str, Any]],
        *,
        top_k_neighbors: int = 5,
        neighbor_threshold: float = 0.5,
    ) -> UpsertNotesResponse:
        """Upsert notes, transparently batching if over the server limit."""
        payload = [_as_dict(n) for n in notes]
        merged = self._batched_call(
            "upsert_notes",
            items=payload,
            param_key="notes",
            result_key="results",
            batch_size=100,
            extra={
                "top_k_neighbors": top_k_neighbors,
                "neighbor_threshold": neighbor_threshold,
            },
        )
        return UpsertNotesResponse.model_validate(merged)

    def upsert_note_types(
        self, note_types: Sequence[NoteTypeInput | dict[str, Any]]
    ) -> UpsertNoteTypesResponse:
        payload = [_as_dict(nt) for nt in note_types]
        merged = self._batched_call(
            "upsert_note_types",
            items=payload,
            param_key="note_types",
            result_key="results",
            batch_size=10,
        )
        return UpsertNoteTypesResponse.model_validate(merged)

    def delete_note_types(self, ids: list[int]) -> DeleteNoteTypesResponse:
        return DeleteNoteTypesResponse.model_validate(self._call("delete_note_types", {"ids": ids}))

    def delete_notes(self, ids: list[int]) -> DeleteNotesResponse:
        """Delete notes, transparently batching if over the server limit."""
        if len(ids) <= 100:
            return DeleteNotesResponse.model_validate(self._call("delete_notes", {"ids": ids}))

        all_deleted: list[int] = []
        all_not_found: list[int] = []
        for i in range(0, len(ids), 100):
            chunk = ids[i : i + 100]
            result = self._call("delete_notes", {"ids": chunk})
            all_deleted.extend(result.get("deleted", []))
            all_not_found.extend(result.get("not_found", []))
        return DeleteNotesResponse(deleted=all_deleted, not_found=all_not_found)

    def _batched_call(
        self,
        tool_name: str,
        *,
        items: list[Any],
        param_key: str,
        result_key: str,
        batch_size: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Split a list of items into batches and merge the results."""
        extra = extra or {}
        if len(items) <= batch_size:
            return self._call(tool_name, {param_key: items, **extra})

        all_results: list[Any] = []
        message: str | None = None
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            result = self._call(tool_name, {param_key: chunk, **extra})
            all_results.extend(result.get(result_key, []))
            message = result.get("message") or message
        merged: dict[str, Any] = {result_key: all_results}
        if message:
            merged["message"] = message
        return merged

    # -- Custom HTTP endpoints ----------------------------------------------

    def server_status(self) -> ServerStatus | None:
        """Probe ``GET /status``. Returns the status, or None if unreachable.

        A non-raising liveness probe — does NOT auto-start.
        """
        try:
            resp = httpx.get(f"{self._base_url}/status", timeout=5.0)
        except (httpx.ConnectError, httpx.TimeoutException):
            return None
        if resp.status_code == 200:
            return ServerStatus.model_validate(resp.json())
        return None

    def ping(self) -> bool:
        """True if the server responds to ``/status``. Does not auto-start."""
        return self.server_status() is not None

    def status(self) -> ServerStatus:
        """``GET /status`` — raises if unreachable."""
        return ServerStatus.model_validate(self._request("GET", "/status", timeout=5.0))

    def index_status(self) -> IndexStatus:
        return self.status().index or IndexStatus()

    def embedding_status(self) -> EmbeddingStatus:
        return self.status().embedding or EmbeddingStatus()

    def index_rebuild(self) -> IndexRebuildResponse:
        return IndexRebuildResponse.model_validate(
            self._request("POST", "/index/rebuild", timeout=30.0)
        )

    def embedding_start(self, **overrides: Any) -> EmbeddingStartResponse:
        body = {k: v for k, v in overrides.items() if v is not None}
        return EmbeddingStartResponse.model_validate(
            self._request("POST", "/embedding/start", json=body, timeout=120.0)
        )

    def embedding_stop(self) -> EmbeddingStopResponse:
        return EmbeddingStopResponse.model_validate(
            self._request("POST", "/embedding/stop", timeout=30.0)
        )

    def shutdown(self) -> ShutdownResponse:
        return ShutdownResponse.model_validate(self._request("POST", "/shutdown", timeout=5.0))

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

    def stop(self, timeout: float = 5.0) -> StopResponse:
        """Stop the local daemon (HTTP → SIGTERM → SIGKILL). Delegates to daemon."""
        return StopResponse.model_validate(daemon.stop_server(timeout=timeout))

    def wait_until_ready(self, timeout: float = 15.0) -> ServerStatus | None:
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


def _as_dict(item: NoteInput | NoteTypeInput | dict[str, Any]) -> dict[str, Any]:
    """Normalize a request item (model or dict) to a JSON-RPC argument dict."""
    if isinstance(item, NoteInput | NoteTypeInput):
        return item.model_dump(exclude_none=True)
    return item
