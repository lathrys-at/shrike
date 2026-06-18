"""ShrikeClient maps the busy action-error code to CollectionBusyError.

The actions edge returns a typed ``ActionError{code, message}`` envelope with an
HTTP status. A ``collection_busy`` code (409) means the collection couldn't be
acquired (another process holds it under cooperative locking) — the op never
ran, so the caller can retry. The client maps that ``code`` to a distinct,
catchable ``CollectionBusyError``. The ``code`` is the authority.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shrike.client import CollectionBusyError, ServerError, ShrikeClient
from shrike.schemas import ActionErrorCode


def _resp(status: int, body: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    return r


def _client_returning(status: int, body: dict) -> tuple[ShrikeClient, object]:
    c = ShrikeClient("http://x/mcp", autostart=False)
    return c, patch("httpx.Client.post", return_value=_resp(status, body))


def test_busy_code_maps_to_collection_busy() -> None:
    c, post = _client_returning(
        409,
        {"code": ActionErrorCode.COLLECTION_BUSY, "message": "in use by another process"},
    )
    with post, pytest.raises(CollectionBusyError, match="in use by another process") as exc:
        c._action("collection_info", {})
    # The human message is surfaced; the code is not leaked into it.
    assert "collection_busy" not in str(exc.value)


def test_input_error_code_is_server_error() -> None:
    c, post = _client_returning(
        400, {"code": ActionErrorCode.INPUT_ERROR, "message": "some other failure"}
    )
    with post, pytest.raises(ServerError, match="some other failure"):
        c._action("collection_info", {})


def test_collection_busy_is_a_shrike_error() -> None:
    # CollectionBusyError is a ShrikeError, so the CLI's generic handler catches
    # it (and renders the human message without a traceback).
    from shrike.client import ShrikeError

    assert issubclass(CollectionBusyError, ShrikeError)
