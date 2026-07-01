"""The _safe_tool logging policy: ONE INFO line per served call — name +
given params + recorded outcome + duration — and the rejected-input/busy lines.
Pinned with caplog against sync and async tools."""

from __future__ import annotations

import logging

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from shrike.api.actions import ToolInputError, note_outcome
from shrike.api.mcp_adapter import _safe_tool
from shrike.harness.collection import CollectionBusyError


def _ok_tool(x: int) -> int:
    """A fine tool."""
    return x + 1


def _outcome_tool(*, deck: str, limit: int = 50, tags: list | None = None) -> str:
    """A tool that records its outcome fragment."""
    note_outcome("3/3 notes")
    return deck


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


class TestCompletionLine:
    def test_one_info_line_with_params_outcome_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # THE line: one INFO record per served call, carrying the tool name,
        # the given params, the action's recorded outcome, and the duration.
        wrapped = _safe_tool(_outcome_tool)
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
            assert wrapped(deck="Test", limit=50) == "Test"
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        msg = infos[0].message
        assert msg.startswith("_outcome_tool ")
        assert "deck='Test'" in msg and "limit=50" in msg
        assert "tags=" not in msg  # None params are omitted (not given)
        assert "-> 3/3 notes (" in msg and msg.endswith("ms)")

    def test_default_outcome_is_ok(self, caplog: pytest.LogCaptureFixture) -> None:
        wrapped = _safe_tool(_ok_tool)
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
            assert wrapped(1) == 2
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        assert "-> ok (" in infos[0].message

    def test_outcome_never_leaks_between_calls(self, caplog: pytest.LogCaptureFixture) -> None:
        # A call that records no outcome must not inherit the previous call's.
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
            _safe_tool(_outcome_tool)(deck="D")
            _safe_tool(_ok_tool)(1)
        infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 2
        assert "-> 3/3 notes (" in infos[0]
        assert "-> ok (" in infos[1]

    async def test_async_success_logs_single_line(self, caplog: pytest.LogCaptureFixture) -> None:
        wrapped = _safe_tool(_ok_tool_async)
        with caplog.at_level(logging.DEBUG, logger="shrike.tools"):
            assert await wrapped(1) == 2
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        assert "-> ok (" in infos[0].message


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
        # The warning IS the line for a failed call — no completion line too.
        assert not any(r.levelno == logging.INFO for r in caplog.records)

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


class TestMcpActionRecording:
    """The MCP transport must count a pre-impl validation rejection as
    input_error — the same label the HTTP edge records — even though FastMCP
    validates args inside Tool.run before the _safe_tool impl runs."""

    async def test_validation_rejection_recorded_as_input_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mcp.server.fastmcp import FastMCP

        from shrike.api import mcp_adapter

        mcp = FastMCP("test")

        @mcp.tool()
        def typed(x: int) -> int:
            """A tool with a typed arg."""
            return x + 1

        mcp_adapter._instrument_tool_manager(mcp)

        calls: list[tuple] = []
        monkeypatch.setattr(mcp_adapter.metrics, "observe_action", lambda *a: calls.append(a))

        # A non-coercible argument fails FastMCP's func_metadata validation inside
        # Tool.run, before the impl — the case that previously went uncounted.
        with pytest.raises(ToolError):
            await mcp._tool_manager.call_tool("typed", {"x": "not-an-int"})

        # name, transport="mcp", result="input_error", duration
        assert any(c[1] == "mcp" and c[2] == "input_error" for c in calls), calls

    async def test_successful_call_not_double_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mcp.server.fastmcp import FastMCP

        from shrike.api import mcp_adapter

        mcp = FastMCP("test")
        mcp.tool()(_safe_tool(_ok_tool))
        mcp_adapter._instrument_tool_manager(mcp)

        calls: list[tuple] = []
        monkeypatch.setattr(mcp_adapter.metrics, "observe_action", lambda *a: calls.append(a))
        await mcp._tool_manager.call_tool("_ok_tool", {"x": 1})
        # _safe_tool records the ok outcome once; the wrapper adds nothing on success.
        assert [c[2] for c in calls] == ["ok"], calls
