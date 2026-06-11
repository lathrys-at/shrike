"""The _safe_tool logging policy (#328): completion durations at INFO and the
rejected-input/busy lines — pinned with caplog against sync and async tools."""

from __future__ import annotations

import logging

import pytest

from shrike.actions import ToolInputError
from shrike.collection import CollectionBusyError
from shrike.mcp_adapter import _safe_tool


def _ok_tool(x: int) -> int:
    """A fine tool."""
    return x + 1


async def _ok_tool_async(x: int) -> int:
    """A fine async tool."""
    return x + 1


def _bad_input_tool() -> None:
    """Rejects its input."""
    raise ToolInputError("limit must be positive")


def _busy_tool() -> None:
    """Hits a held collection."""
    raise CollectionBusyError()


def _broken_tool() -> None:
    """A genuine bug."""
    raise RuntimeError("boom")


class TestDurations:
    def test_success_logs_duration_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        # Every completed call carries its duration at INFO — like the route
        # access lines, durations are operator information, never noise.
        wrapped = _safe_tool(_ok_tool)
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
            assert wrapped(1) == 2
        records = [r for r in caplog.records if "completed" in r.message]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO
        assert "_ok_tool" in records[0].message
        assert "ms)" in records[0].message

    async def test_async_success_logs_duration(self, caplog: pytest.LogCaptureFixture) -> None:
        wrapped = _safe_tool(_ok_tool_async)
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
            assert await wrapped(1) == 2
        assert any("completed" in r.message and r.levelno == logging.INFO for r in caplog.records)


class TestFailures:
    def test_tool_input_error_logs_warning_without_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wrapped = _safe_tool(_bad_input_tool)
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"), pytest.raises(ToolInputError):
            wrapped()
        records = [r for r in caplog.records if "rejected" in r.message]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert "limit must be positive" in records[0].message
        assert records[0].exc_info is None  # no traceback for expected bad input
        # No duration line for a failed call.
        assert not any("completed" in r.message for r in caplog.records)

    def test_busy_logs_warning_without_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        wrapped = _safe_tool(_busy_tool)
        with (
            caplog.at_level(logging.DEBUG, logger="shrike.tools"),
            pytest.raises(CollectionBusyError),
        ):
            wrapped()
        records = [r for r in caplog.records if "collection_busy" in r.message]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is None

    def test_unhandled_error_logs_with_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        wrapped = _safe_tool(_broken_tool)
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"), pytest.raises(RuntimeError):
            wrapped()
        records = [r for r in caplog.records if "Unhandled error" in r.message]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].exc_info is not None  # genuine bugs carry the traceback
