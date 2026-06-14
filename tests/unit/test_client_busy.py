"""ShrikeClient maps the busy wire code to CollectionBusyError (#65, #598)."""

from __future__ import annotations

import asyncio

import mcp.types as types
import pytest
from mcp.server.fastmcp import FastMCP

from shrike.client import CollectionBusyError, ServerError, ShrikeClient
from shrike.collection import CollectionBusyError as ServerCollectionBusyError
from shrike.schemas import COLLECTION_BUSY_CODE


def _client(monkeypatch, text: str) -> ShrikeClient:
    c = ShrikeClient("http://x/mcp", autostart=False)
    body = {"result": {"isError": True, "content": [{"type": "text", "text": text}]}}

    class _Resp:
        def json(self):
            return body

    monkeypatch.setattr(c, "_post_mcp", lambda payload: _Resp())
    monkeypatch.setattr(c, "_raise_for_status", lambda resp: None)
    return c


def test_busy_code_maps_to_collection_busy(monkeypatch):
    c = _client(monkeypatch, f"{COLLECTION_BUSY_CODE}: in use by another process")
    with pytest.raises(CollectionBusyError, match="in use by another process") as exc:
        c._call("collection_info", {})
    # The human message is surfaced; the code prefix is stripped.
    assert COLLECTION_BUSY_CODE not in str(exc.value)


def test_generic_error_still_server_error(monkeypatch):
    c = _client(monkeypatch, "some other failure")
    with pytest.raises(ServerError, match="some other failure"):
        c._call("collection_info", {})

    # CollectionBusyError is a ShrikeError, so the CLI's generic handler catches it.
    from shrike.client import ShrikeError

    assert issubclass(CollectionBusyError, ShrikeError)


def _busy_wire_text_via_real_handler() -> str:
    """Produce the busy error text exactly as the server emits it on the wire.

    This drives the REAL FastMCP low-level CallToolRequest handler (not a
    hand-crafted body), so the wrapping is genuine: FastMCP's ``Tool.run`` wraps
    a raised exception as ``"Error executing tool <name>: <exc>"`` and the
    low-level handler serializes that into the ``isError`` text content. With the
    server-side ``CollectionBusyError`` (whose str is ``"collection_busy: …"``)
    the result is ``"Error executing tool boom: collection_busy: …"`` — the
    mid-string sentinel a ``startswith`` check misses (#598). The two existing
    busy tests hand-craft the wire body and never see this wrapping.
    """
    mcp = FastMCP("test")

    @mcp.tool()
    def boom() -> dict:  # the exact error _safe_tool re-raises on busy
        raise ServerCollectionBusyError()

    handler = mcp._mcp_server.request_handlers[types.CallToolRequest]

    async def _drive() -> types.CallToolResult:
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="boom", arguments={}),
        )
        return (await handler(req)).root

    result = asyncio.run(_drive())
    assert result.isError
    return result.content[0].text


def test_real_fastmcp_wrapping_maps_to_collection_busy(monkeypatch):
    """The FastMCP-wrapped busy text must still raise the client-side busy error.

    Regression for #598: ``_safe_tool`` re-raises ``CollectionBusyError``, but
    FastMCP prefixes it with ``"Error executing tool <name>: "`` so the sentinel
    is no longer at index 0. The old ``startswith`` check fell through to
    ``ServerError``, silently defeating every ``except CollectionBusyError:
    retry`` (and the CLI-against-a-daemon path) under ``--cooperative-lock``.
    """
    wire_text = _busy_wire_text_via_real_handler()
    # Guard the premise: this really is the wrapped form, sentinel mid-string.
    assert wire_text.startswith("Error executing tool")
    assert wire_text.index(f"{COLLECTION_BUSY_CODE}:") > 0

    c = _client(monkeypatch, wire_text)
    with pytest.raises(CollectionBusyError) as exc:
        c._call("collection_info", {})
    # The human message survives; the wrapper + sentinel are stripped, not kept.
    msg = str(exc.value)
    assert "is in use by another process" in msg
    assert COLLECTION_BUSY_CODE not in msg
    assert "Error executing tool" not in msg
