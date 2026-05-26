from __future__ import annotations

from typing import Any

import click
import httpx


class ServerError(Exception):
    """Raised when the server returns a tool-level error."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ShrikeClient:
    """Thin HTTP client that makes MCP JSON-RPC tool calls to a Shrike server.

    If *config* is provided and *autostart* is True (the default), the
    client will automatically launch the daemon on the first connection
    failure and retry.
    """

    def __init__(
        self,
        url: str,
        config: dict[str, Any] | None = None,
        *,
        autostart: bool = True,
    ):
        self.url = url
        self.config = config
        self.autostart = autostart and config is not None
        self._request_id = 0
        self._autostarted = False

    def _post(self, payload: dict[str, Any], *, timeout: float = 30.0) -> httpx.Response:
        """POST to the server, auto-starting the daemon on first failure."""
        try:
            return httpx.post(
                self.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=timeout,
            )
        except httpx.ConnectError:
            if self.autostart and not self._autostarted:
                self._do_autostart()
                return httpx.post(
                    self.url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=timeout,
                )
            raise

    def _do_autostart(self) -> None:
        """Launch the daemon and update our URL."""
        from shrike.cli.server_cmd import ensure_server

        assert self.config is not None
        self.url = ensure_server(self.config)
        self._autostarted = True

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke an MCP tool and return the structured result.

        Raises:
            ServerError: if the tool returns an error response
            click.ClickException: if the server is unreachable
        """
        self._request_id += 1
        try:
            resp = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments or {},
                    },
                }
            )
        except httpx.ConnectError as err:
            raise click.ClickException(
                f"Cannot connect to server at {self.url}\n"
                "Is the server running? Start it with: shrike server start"
            ) from err
        except httpx.TimeoutException as err:
            raise click.ClickException(f"Request to {self.url} timed out") from err

        resp.raise_for_status()
        body = resp.json()

        if "error" in body:
            raise ServerError(f"Server error: {body['error']}")

        result = body.get("result", {})
        content = result.get("structuredContent", {})

        # Check for tool-level errors
        if isinstance(content, dict) and "error" in content:
            raise ServerError(content["error"])

        return content  # type: ignore[no-any-return]

    def ping(self) -> bool:
        """Check if the server is reachable and responding.

        Does NOT auto-start — this is a probe, not an operation.
        """
        return self.server_status() is not None

    def server_status(self) -> dict[str, Any] | None:
        """Fetch status from the daemon's /status endpoint.

        Returns the status dict, or None if the server is unreachable.
        Does NOT auto-start.
        """
        status_url = self.url.rsplit("/", 1)[0] + "/status"
        try:
            resp = httpx.get(status_url, timeout=5.0)
            if resp.status_code == 200:
                result: dict[str, Any] = resp.json()
                return result
            return None
        except (httpx.ConnectError, httpx.TimeoutException):
            return None

    # -- Convenience methods --

    def collection_info(
        self,
        include: list[str] | None = None,
        note_type_details: list[str] | None = None,
    ) -> dict:
        args: dict[str, Any] = {}
        if include:
            args["include"] = include
        if note_type_details:
            args["note_type_details"] = note_type_details
        return self.call("collection_info", args)

    def list_notes(self, **kwargs: Any) -> dict:
        # Strip None values
        args = {k: v for k, v in kwargs.items() if v is not None}
        return self.call("list_notes", args)

    def search_notes(self, **kwargs: Any) -> dict:
        args = {k: v for k, v in kwargs.items() if v is not None}
        return self.call("search_notes", args)

    def upsert_notes(self, notes: list[dict]) -> dict:
        """Upsert notes, transparently batching if over the server limit."""
        return self._batched_call(
            "upsert_notes",
            items=notes,
            param_key="notes",
            result_key="results",
            batch_size=100,
        )

    def upsert_note_types(self, note_types: list[dict]) -> dict:
        """Upsert note types, transparently batching if over the server limit."""
        return self._batched_call(
            "upsert_note_types",
            items=note_types,
            param_key="note_types",
            result_key="results",
            batch_size=10,
        )

    def delete_note_types(self, ids: list[int]) -> dict:
        """Delete note types by ID."""
        return self.call("delete_note_types", {"ids": ids})

    def delete_notes(self, ids: list[int]) -> dict:
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
    ) -> dict:
        """Split a list of items into batches and merge the results."""
        if len(items) <= batch_size:
            return self.call(tool_name, {param_key: items})

        all_results: list = []
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            result = self.call(tool_name, {param_key: chunk})
            all_results.extend(result.get(result_key, []))
        return {result_key: all_results}
