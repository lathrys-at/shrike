"""The MCP binding for the action registry.

Iterates :func:`shrike.api.actions.build_actions`'s registry and generates the
FastMCP ``@mcp.tool`` bindings, applying the ``_safe_tool`` policy — docstring
``inspect.cleandoc`` (so advertised descriptions carry no source indentation)
and the error→``isError`` mapping (``ToolInputError``/``CollectionBusyError``
logged without tracebacks, genuine bugs with one). FastMCP's ``outputSchema``
emission follows from the impls' response models.

This is one adapter among possible several: the registry itself is
FastMCP-free, so future agent-runtime adapters (on-device function-calling)
bind the same actions without touching this module.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

import shrike_native
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool
from pydantic import ValidationError

from shrike.api.actions import (
    CONTROL_PLANE_ACTIONS,
    ActionDef,
    ToolInputError,
    _call_outcome,
)
from shrike.harness.collection import CollectionBusyError
from shrike.observability.metrics import metrics

logger = logging.getLogger("shrike.tools")

# The same wrapped action implementation serves MCP and actions-over-HTTP.  The
# HTTP handler sets this for the duration of Tool.run; MCP naturally retains the
# default.  ContextVar keeps concurrent requests isolated.
_action_transport: ContextVar[str] = ContextVar("shrike_action_transport", default="mcp")

# The readiness gate: an async callable that resolves once the data plane is
# open (boot/reload/re-acquire maintenance has settled). None disables gating
# (standalone / tests with no harness barrier).
ReadinessGate = Callable[[], Awaitable[None]]

# A DEFENSIVE bound on the readiness wait: await_ready is deterministic-and-
# bounded today (boot/reload/re-acquire settle in seconds), but the gate must
# fail SAFE — a future readiness regression must surface a clear, loud error
# rather than wedging every data-plane call forever. Generous, so a normal slow
# settle (a large derived rebuild) never trips it.
READINESS_GATE_TIMEOUT_S = 30.0


class ServerNotReadyError(RuntimeError):
    """The data plane did not become ready within the gate's defensive bound.

    Raised only when [`READINESS_GATE_TIMEOUT_S`] elapses awaiting the readiness
    barrier — a sign the server is wedged settling (a regression or extreme
    load), not the normal park-until-ready path. Surfaced loudly (an MCP
    ``isError``) so the condition is visible, never silently retried."""


# Cap on a single rendered param value in the completion line (long field
# bodies and queries get elided, never dropped).
_PARAM_VALUE_MAX = 60


def _compact_value(value: Any) -> str:
    """A short rendering of one call param for the completion log line."""
    if isinstance(value, list):
        return f"[{len(value)} item(s)]" if len(value) > 3 else repr(value)[:_PARAM_VALUE_MAX]
    if isinstance(value, dict):
        return f"{{{len(value)} key(s)}}"
    rendered = repr(value)
    return rendered if len(rendered) <= _PARAM_VALUE_MAX else rendered[: _PARAM_VALUE_MAX - 1] + "…"


def _format_params(kwargs: dict[str, Any]) -> str:
    """``k=v`` fragments for the params that were actually given (None = omitted)."""
    return " ".join(f"{k}={_compact_value(v)}" for k, v in kwargs.items() if v is not None)


def _log_completed(name: str, kwargs: dict[str, Any], started: float) -> None:
    # THE log line for a served call: one INFO line carrying the tool name, its
    # params, the action-recorded outcome, and the duration. Actions contribute
    # the outcome via note_outcome(); anything else they log during the call is
    # a warning/error (exceptional) or DEBUG (internals).
    elapsed_ms = (time.perf_counter() - started) * 1000
    outcome = _call_outcome.get() or "ok"
    params = _format_params(kwargs)
    logger.info("%s%s%s -> %s (%.0fms)", name, " " if params else "", params, outcome, elapsed_ms)


def _record_action(name: str, started: float, result: str) -> None:
    metrics.observe_action(name, _action_transport.get(), result, time.perf_counter() - started)


def _safe_tool(fn: Any) -> Any:
    """Wrap a tool to log unhandled exceptions, then re-raise.

    A re-raised exception becomes an MCP ``isError`` result (FastMCP converts
    it), which the client surfaces as a ``ServerError``. Tools therefore never
    embed an ``error`` field in a success payload — protocol errors live in the
    protocol, and response models stay clean. ``ToolInputError`` (expected bad
    input) logs the rejection without a traceback; anything else logs with one.

    Every served call emits exactly ONE INFO line, here: tool name + given
    params + the action's recorded outcome (``note_outcome``) + duration.
    Failures log their warning/error instead (never both).

    The wrapped function's docstring is dedented with ``inspect.cleandoc`` so the
    tool description FastMCP advertises to clients has no source indentation.
    """
    cleaned_doc = inspect.cleandoc(fn.__doc__) if fn.__doc__ else None

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            _call_outcome.set(None)  # never inherit a previous call's outcome
            try:
                result = await fn(*args, **kwargs)
            except ToolInputError as e:
                # Expected bad input — the caller's mistake, surfaced without a trace.
                logger.warning("%s rejected: %s", fn.__name__, e)
                _record_action(fn.__name__, started, "input_error")
                raise
            except CollectionBusyError as e:
                # Expected under cooperative locking — another process holds the
                # collection. Surface the coded message (client maps it), no trace.
                logger.warning("%s in %s", e, fn.__name__)
                _record_action(fn.__name__, started, "collection_busy")
                raise
            except shrike_native.NativeBusyError as e:
                # The same contention, surfaced by a kernel-routed op (the
                # kernel's ensure_open hit a held file): normalize to the
                # typed busy surface so the client's retry contract holds.
                busy = CollectionBusyError()
                logger.warning("%s in %s", busy, fn.__name__)
                _record_action(fn.__name__, started, "collection_busy")
                raise busy from e
            except ServerNotReadyError:
                _record_action(fn.__name__, started, "not_ready")
                raise
            except Exception:
                logger.exception("Unhandled error in %s", fn.__name__)
                _record_action(fn.__name__, started, "internal_error")
                raise
            _log_completed(fn.__name__, kwargs, started)
            _record_action(fn.__name__, started, "ok")
            return result

        async_wrapper.__doc__ = cleaned_doc
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        _call_outcome.set(None)  # never inherit a previous call's outcome
        try:
            result = fn(*args, **kwargs)
        except ToolInputError as e:
            logger.warning("%s rejected: %s", fn.__name__, e)
            _record_action(fn.__name__, started, "input_error")
            raise
        except CollectionBusyError as e:
            logger.warning("%s in %s", e, fn.__name__)
            _record_action(fn.__name__, started, "collection_busy")
            raise
        except shrike_native.NativeBusyError as e:
            busy = CollectionBusyError()
            logger.warning("%s in %s", busy, fn.__name__)
            _record_action(fn.__name__, started, "collection_busy")
            raise busy from e
        except ServerNotReadyError:
            _record_action(fn.__name__, started, "not_ready")
            raise
        except Exception:
            logger.exception("Unhandled error in %s", fn.__name__)
            _record_action(fn.__name__, started, "internal_error")
            raise
        _log_completed(fn.__name__, kwargs, started)
        _record_action(fn.__name__, started, "ok")
        return result

    wrapper.__doc__ = cleaned_doc
    return wrapper


def _gate_ready(impl: Any, readiness: ReadinessGate | None) -> Any:
    """Wrap a data-plane action impl so it AWAITS data-plane readiness before
    running (Theme C / #833: serve only ``/status`` + the control plane until
    boot/reload/re-acquire maintenance has settled). Every action is data-plane,
    so the gate is uniform; the control plane (``/status``/``/reload``/
    ``/shutdown``/``/embedding/*``) is the operational HTTP routes, which never
    reach here. ``None`` (standalone / tests) is a pass-through.

    The wrapper preserves the impl's signature (``functools.wraps``) so FastMCP's
    ``func_metadata`` still generates the right input schema, and it sits BENEATH
    ``_safe_tool`` so the await is inside the one-INFO-line/error-policy frame.

    The wait carries a DEFENSIVE timeout ([`READINESS_GATE_TIMEOUT_S`]): the
    normal path parks until ready, but a wedged barrier surfaces a clear
    [`ServerNotReadyError`] instead of hanging the call forever.
    """
    if readiness is None:
        return impl

    @functools.wraps(impl)
    async def gated(*args: Any, **kwargs: Any) -> Any:
        try:
            await asyncio.wait_for(readiness(), timeout=READINESS_GATE_TIMEOUT_S)
        except TimeoutError as e:
            raise ServerNotReadyError(
                f"the data plane did not become ready within {READINESS_GATE_TIMEOUT_S:.0f}s"
            ) from e
        return await impl(*args, **kwargs)

    return gated


def _bound_impl(action: ActionDef, readiness: ReadinessGate | None) -> Any:
    """The wrapped impl both edges bind: ``_safe_tool`` over the readiness gate,
    EXCEPT for a control-plane action (in [`CONTROL_PLANE_ACTIONS`]), which
    bypasses the gate so it can serve before the data plane is ready."""
    gate = None if action.name in CONTROL_PLANE_ACTIONS else readiness
    return _safe_tool(_gate_ready(action.impl, gate))


def _instrument_tool_manager(mcp: FastMCP) -> None:
    """Count an MCP-transport **validation rejection** as ``input_error``.

    FastMCP validates a call's arguments inside ``Tool.run`` (``fn_metadata``),
    BEFORE the ``_safe_tool``-wrapped impl runs — so a bad/out-of-range argument
    raises a pydantic ``ValidationError`` (wrapped in ``ToolError``) that never
    reaches ``_record_action``. The HTTP edge records that case as ``input_error``;
    without this the two transports disagree. Wrap the tool manager's ``call_tool``
    (the single seam every MCP ``tools/call`` flows through) to record it, with
    the ``mcp`` transport from the ContextVar. Impl-path outcomes are already
    recorded by ``_safe_tool``, so only the pre-impl validation gap is filled.
    Idempotent — wrapping once per manager.
    """
    manager = mcp._tool_manager
    if getattr(manager, "_shrike_instrumented", False):
        return
    original_call_tool = manager.call_tool

    async def call_tool(name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original_call_tool(name, arguments, **kwargs)
        except Exception as exc:
            cause = exc.__cause__ if exc.__cause__ is not None else exc
            if isinstance(cause, ValidationError):
                _record_action(name, started, "input_error")
            raise

    manager.call_tool = call_tool  # type: ignore[assignment]
    manager._shrike_instrumented = True  # type: ignore[attr-defined]


def register_actions(
    mcp: FastMCP, actions: list[ActionDef], readiness: ReadinessGate | None = None
) -> None:
    """Bind every registry action as an MCP tool (the decoration order:
    ``mcp.tool()`` over the ``_safe_tool``-wrapped, readiness-gated impl)."""
    for action in actions:
        mcp.tool()(_bound_impl(action, readiness))
    _instrument_tool_manager(mcp)


def build_action_tools(
    actions: list[ActionDef], readiness: ReadinessGate | None = None
) -> dict[str, Tool]:
    """Build a ``name -> Tool`` map for the actions-over-HTTP edge.

    Strict parity by construction: each ``Tool`` is built from the *exact same*
    ``_safe_tool``-wrapped, readiness-gated impl that :func:`register_actions`
    binds to MCP (``mcp.tool()`` calls ``Tool.from_function`` on the same wrapped
    callable). So the UI edge inherits the identical ``arg_model`` (func_metadata's
    JSON→typed coercion + ``pre_parse_json``), the same ``_safe_tool`` error
    policy + one-INFO-line logging, the same readiness gate, and the same
    ``output_model`` — the structured payload it serializes is byte-identical to
    the ``structuredContent`` the MCP path emits, minus the JSON-RPC envelope.
    MCP is the agent edge; these tools are the UI edge; same catalog, two adapters.
    """
    return {action.name: Tool.from_function(_bound_impl(action, readiness)) for action in actions}
