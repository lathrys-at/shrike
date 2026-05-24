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
    """Thin HTTP client that makes MCP JSON-RPC tool calls to a Shrike server."""

    def __init__(self, url: str):
        self.url = url
        self._request_id = 0

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke an MCP tool and return the structured result.

        Raises:
            ServerError: if the tool returns an error response
            click.ClickException: if the server is unreachable
        """
        self._request_id += 1
        try:
            resp = httpx.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._request_id,
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
                timeout=30.0,
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
        """Check if the server is reachable and responding."""
        try:
            resp = httpx.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "shrike-cli", "version": "0.1.0"},
                    },
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=5.0,
            )
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

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
        return self.call("upsert_notes", {"notes": notes})

    def upsert_note_types(self, note_types: list[dict]) -> dict:
        return self.call("upsert_note_types", {"note_types": note_types})

    def delete_notes(self, ids: list[int]) -> dict:
        return self.call("delete_notes", {"ids": ids})
