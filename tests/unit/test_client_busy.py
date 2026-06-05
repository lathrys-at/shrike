"""ShrikeClient maps the busy wire code to CollectionBusyError (#65)."""

from __future__ import annotations

import pytest

from shrike.client import CollectionBusyError, ServerError, ShrikeClient
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
