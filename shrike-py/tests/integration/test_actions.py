"""Integration tests for the actions-over-HTTP edge (#505).

`POST /actions/{name}` is the UI edge of the action catalog — the *same* named
ops the MCP tools bind, served through the *same* `_safe_tool`-wrapped impl,
behind the *same* Host/Origin guard, but schema-first without the MCP JSON-RPC
envelope. These tests pin the load-bearing property: **strict parity** — for the
same input, the actions route's body equals the MCP tool's `structuredContent`
(the dict FastMCP emits) — plus the #392 wire-version header round-trip.

Guard coverage (forged-Host 421 / cross-origin 403 / GET-on-POST 405) lives in
`test_security.py`, which parametrizes a concrete `/actions/collection_info`
route through its per-route suite; the non-leaking error envelope is asserted
there too. This file is the parity + version-handshake half.
"""

from __future__ import annotations

import httpx
import pytest

from .conftest import ServerInfo

pytestmark = pytest.mark.integration

WIRE_VERSION_HEADER = "X-Shrike-Wire-Version"


def _base_url(server: ServerInfo) -> str:
    return server.url.rsplit("/", 1)[0]


def _mcp_structured(server: ServerInfo, name: str, arguments: dict) -> dict:
    """The `structuredContent` the MCP path emits for a successful tool call —
    the exact dict the actions route must reproduce."""
    resp = httpx.post(
        server.url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30.0,
    )
    resp.raise_for_status()
    result = resp.json()["result"]
    assert not result.get("isError"), result
    return result["structuredContent"]


def _action(server: ServerInfo, name: str, arguments: dict) -> httpx.Response:
    return httpx.post(
        f"{_base_url(server)}/actions/{name}",
        json=arguments,
        timeout=30.0,
    )


# Representative ops: a read (collection_info), a filtered read (list_notes), the
# raw-query escape hatch (collection_query), and a no-write validation pass
# (upsert_notes dry_run). Each must round-trip byte-identically to MCP.
_PARITY_CASES = [
    ("collection_info", {}),
    ("collection_info", {"include": ["summary", "decks"]}),
    ("list_notes", {"deck": "Default", "limit": 5}),
    ("collection_query", {"query": "deck:Default", "limit": 5}),
    (
        "upsert_notes",
        {
            "notes": [
                {"note_type": "Basic", "deck": "Default", "fields": {"Front": "Q", "Back": "A"}}
            ],
            "dry_run": True,
        },
    ),
]


class TestActionsParity:
    @pytest.mark.parametrize(("name", "arguments"), _PARITY_CASES)
    def test_actions_body_equals_mcp_structured(
        self, server: ServerInfo, name: str, arguments: dict
    ) -> None:
        mcp_body = _mcp_structured(server, name, arguments)
        resp = _action(server, name, arguments)
        assert resp.status_code == 200, resp.text
        assert resp.json() == mcp_body

    def test_no_body_means_no_arguments(self, server: ServerInfo) -> None:
        # An empty POST (no JSON body) is the same as `{}` — every collection_info
        # arg has a default, so it serves the summary.
        resp = httpx.post(f"{_base_url(server)}/actions/collection_info", timeout=30.0)
        assert resp.status_code == 200, resp.text
        assert resp.json() == _mcp_structured(server, "collection_info", {})


class TestActionsInputErrorParity:
    """An input error on the actions edge carries the same `ToolInputError`
    rejection text the MCP tool raises — but in the typed envelope, *without*
    MCP's `Error executing tool <name>: ` wrapper (the actions edge is
    schema-first without the MCP envelope). Same cause, cleaner surface."""

    def test_input_error_message_is_the_bare_rejection(self, server: ServerInfo) -> None:
        # MCP side: a no-filter list_notes is an isError whose text is the
        # ToolInputError message wrapped in FastMCP's "Error executing tool…"
        # prefix.
        mcp_resp = httpx.post(
            server.url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_notes", "arguments": {}},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30.0,
        )
        mcp_result = mcp_resp.json()["result"]
        assert mcp_result.get("isError")
        mcp_text = next(
            c["text"] for c in mcp_result["content"] if isinstance(c, dict) and c.get("text")
        )

        action_resp = _action(server, "list_notes", {})
        assert action_resp.status_code == 400
        body = action_resp.json()
        assert body["code"] == "input_error"
        # The actions edge carries the bare rejection (no MCP wrapper); the MCP
        # text is that same message behind the "Error executing tool…" prefix.
        assert body["message"] in mcp_text
        assert body["message"] == (
            "At least one filter (ids, deck, tags, note_type, or modified_since) must be provided."
        )


class TestWireVersionHeader:
    """#392: every /actions/* response echoes the server's wire version; a
    request may assert it, and a mismatch is refused before the op runs."""

    def test_response_echoes_wire_version(self, server: ServerInfo) -> None:
        # Discover the server's version from /status (its canonical report).
        status = httpx.get(f"{_base_url(server)}/status", timeout=5.0).json()
        version = str(status["wire_protocol_version"])

        resp = _action(server, "collection_info", {})
        assert resp.status_code == 200
        assert resp.headers.get(WIRE_VERSION_HEADER) == version

    def test_matching_request_version_accepted(self, server: ServerInfo) -> None:
        version = str(
            httpx.get(f"{_base_url(server)}/status", timeout=5.0).json()["wire_protocol_version"]
        )
        resp = httpx.post(
            f"{_base_url(server)}/actions/collection_info",
            json={},
            headers={WIRE_VERSION_HEADER: version},
            timeout=30.0,
        )
        assert resp.status_code == 200, resp.text

    def test_mismatched_request_version_refused(self, server: ServerInfo) -> None:
        resp = httpx.post(
            f"{_base_url(server)}/actions/collection_info",
            json={},
            headers={WIRE_VERSION_HEADER: "999"},
            timeout=30.0,
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "input_error"
        assert "wire protocol version" in body["message"].lower()
        # The error envelope still carries the version header.
        assert WIRE_VERSION_HEADER in resp.headers
