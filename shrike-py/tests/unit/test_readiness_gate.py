"""The data-plane readiness gate (#856 / Theme C): every action awaits the
harness readiness barrier before running, so the data plane serves only once
boot/reload/re-acquire maintenance has settled. The control plane (the
operational HTTP routes) is not an action and never reaches the gate."""

from __future__ import annotations

import asyncio
import inspect

import pytest
from mcp.server.fastmcp import FastMCP

from shrike.api import mcp_adapter
from shrike.api.actions import CONTROL_PLANE_ACTIONS, build_actions
from shrike.api.mcp_adapter import ServerNotReadyError, _gate_ready
from shrike.api.tools import register_tools


async def _impl(x: int, *, y: int = 0) -> int:
    """A data-plane action."""
    return x + y


def test_gate_blocks_until_ready() -> None:
    # The gated impl must not run its body until readiness resolves.
    ran = asyncio.Event()
    ready = asyncio.Event()

    async def readiness() -> None:
        await ready.wait()

    async def body(x: int, *, y: int = 0) -> int:
        ran.set()
        return x + y

    gated = _gate_ready(body, readiness)

    async def flow() -> int:
        task = asyncio.ensure_future(gated(2, y=3))
        # Let the task reach the gate; it must be parked, not run.
        for _ in range(10):
            await asyncio.sleep(0)
        assert not ran.is_set(), "the action ran before readiness resolved"
        ready.set()
        result = await task
        assert ran.is_set()
        return result

    assert asyncio.run(flow()) == 5


def test_gate_passes_through_when_ready() -> None:
    # An already-resolved readiness lets the action run immediately.
    async def readiness() -> None:
        return None

    gated = _gate_ready(_impl, readiness)
    assert asyncio.run(gated(4, y=1)) == 5


def test_none_readiness_is_a_pass_through() -> None:
    # Standalone / tests: no gate, the impl is returned unwrapped.
    assert _gate_ready(_impl, None) is _impl


def test_gate_preserves_the_signature() -> None:
    # FastMCP's func_metadata reads the signature to build the input schema, so
    # the wrapper must carry the impl's params through functools.wraps.
    async def readiness() -> None:
        return None

    gated = _gate_ready(_impl, readiness)
    assert inspect.signature(gated) == inspect.signature(_impl)


def test_gate_propagates_the_action_result_and_errors() -> None:
    async def readiness() -> None:
        return None

    async def boom(_x: int) -> int:
        raise ValueError("from the action")

    gated = _gate_ready(boom, readiness)
    with pytest.raises(ValueError, match="from the action"):
        asyncio.run(gated(1))


def test_gate_fails_safe_on_a_wedged_barrier(monkeypatch) -> None:
    # The DEFENSIVE bound: a barrier that never resolves must surface a clear
    # ServerNotReadyError instead of hanging the call forever, so a future
    # readiness regression can't wedge every data-plane call.
    monkeypatch.setattr(mcp_adapter, "READINESS_GATE_TIMEOUT_S", 0.05)

    async def never_ready() -> None:
        await asyncio.Event().wait()

    gated = _gate_ready(_impl, never_ready)
    with pytest.raises(ServerNotReadyError, match="did not become ready"):
        asyncio.run(gated(1))


class TestControlPlaneAllowlist:
    """The data/control split is EXPLICIT and TESTED: every action gates on
    readiness; the control plane (/status, /reload, /shutdown, /embedding/*,
    /index/rebuild) is the operational HTTP routes, which are not actions — so
    the action allowlist is empty. Pinned so the split can't drift silently."""

    def test_no_action_is_control_plane(self, kharness) -> None:
        # Every registered action is data-plane: none bypasses the gate.
        from shrike.api.actions import ActionContext

        actions = build_actions(ActionContext(wrapper=kharness.wrapper, kernel=kharness.kernel))
        action_names = {a.name for a in actions}
        assert not CONTROL_PLANE_ACTIONS, (
            "the control plane lives on the HTTP routes, not actions — keep this empty "
            "unless a genuine control-plane ACTION is added (and gate-test it)"
        )
        # A guard against a stale name: every allowlisted action must exist.
        assert action_names >= CONTROL_PLANE_ACTIONS

    def test_a_control_plane_action_would_bypass_the_gate(self, kharness) -> None:
        # If a control-plane action existed, it must NOT be gated. Inject one and
        # assert it bypasses while a sibling data-plane action still gates.
        from shrike.api.actions import ActionDef

        async def never_ready() -> None:
            await asyncio.Event().wait()

        async def admin() -> str:
            return "served while not ready"

        async def data() -> str:
            return "data"

        control = ActionDef(name="_admin_probe", impl=admin, doc=None)
        ordinary = ActionDef(name="_data_probe", impl=data, doc=None)

        def flow() -> None:
            import shrike.api.mcp_adapter as adapter

            with pytest.MonkeyPatch.context() as m:
                m.setattr(adapter, "CONTROL_PLANE_ACTIONS", frozenset({"_admin_probe"}))
                m.setattr(adapter, "READINESS_GATE_TIMEOUT_S", 0.05)
                bound_control = adapter._bound_impl(control, never_ready)
                bound_data = adapter._bound_impl(ordinary, never_ready)

                async def run() -> None:
                    # The control-plane action serves despite the wedged barrier.
                    assert await bound_control() == "served while not ready"
                    # The data-plane action fails safe on the same barrier.
                    with pytest.raises(ServerNotReadyError):
                        await bound_data()

                asyncio.run(run())

        flow()


class TestServerWiring:
    """The gate is WIRED into the serve path: register_tools(readiness=...)
    threads the barrier into every bound tool, so a real MCP call parks until
    readiness resolves. The H1 finding was that the barrier was defined but had
    no serve-path consumer — this pins that it now does."""

    def test_a_data_plane_tool_awaits_the_wired_readiness_gate(self, kharness) -> None:
        ready = asyncio.Event()

        async def readiness() -> None:
            await ready.wait()

        mcp = FastMCP("test")
        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel, readiness=readiness)

        async def flow() -> None:
            call = asyncio.ensure_future(mcp.call_tool("collection_info", {}))
            for _ in range(10):
                await asyncio.sleep(0)
                if call.done():
                    break
            assert not call.done(), "the tool ran before the readiness gate opened"
            ready.set()
            await call  # now resolves
            await kharness.kernel.settle()

        kharness.run(flow())

    def test_without_a_gate_a_tool_runs_immediately(self, kharness) -> None:
        # readiness=None (the default) is a pass-through: no gate, the tool runs.
        mcp = FastMCP("test")
        register_tools(mcp, kharness.wrapper, kernel=kharness.kernel)
        result = kharness.call_tool(mcp, "collection_info", {})
        assert result is not None
