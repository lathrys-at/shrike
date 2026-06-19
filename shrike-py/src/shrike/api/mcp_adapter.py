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

import functools
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import shrike_native
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool

from shrike.api.actions import ActionDef, ToolInputError, _call_outcome
from shrike.harness.collection import CollectionBusyError

logger = logging.getLogger("shrike.tools")

# The readiness gate: an async callable that resolves once the data plane is
# open (boot/reload/re-acquire maintenance has settled). None disables gating
# (standalone / tests with no harness barrier).
ReadinessGate = Callable[[], Awaitable[None]]

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
                raise
            except CollectionBusyError as e:
                # Expected under cooperative locking — another process holds the
                # collection. Surface the coded message (client maps it), no trace.
                logger.warning("%s in %s", e, fn.__name__)
                raise
            except shrike_native.NativeBusyError as e:
                # The same contention, surfaced by a kernel-routed op (the
                # kernel's ensure_open hit a held file): normalize to the
                # typed busy surface so the client's retry contract holds.
                busy = CollectionBusyError()
                logger.warning("%s in %s", busy, fn.__name__)
                raise busy from e
            except Exception:
                logger.exception("Unhandled error in %s", fn.__name__)
                raise
            _log_completed(fn.__name__, kwargs, started)
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
            raise
        except CollectionBusyError as e:
            logger.warning("%s in %s", e, fn.__name__)
            raise
        except shrike_native.NativeBusyError as e:
            busy = CollectionBusyError()
            logger.warning("%s in %s", busy, fn.__name__)
            raise busy from e
        except Exception:
            logger.exception("Unhandled error in %s", fn.__name__)
            raise
        _log_completed(fn.__name__, kwargs, started)
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
    """
    if readiness is None:
        return impl

    @functools.wraps(impl)
    async def gated(*args: Any, **kwargs: Any) -> Any:
        await readiness()
        return await impl(*args, **kwargs)

    return gated


def register_actions(
    mcp: FastMCP, actions: list[ActionDef], readiness: ReadinessGate | None = None
) -> None:
    """Bind every registry action as an MCP tool (the decoration order:
    ``mcp.tool()`` over the ``_safe_tool``-wrapped, readiness-gated impl)."""
    for action in actions:
        mcp.tool()(_safe_tool(_gate_ready(action.impl, readiness)))


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
    return {
        action.name: Tool.from_function(_safe_tool(_gate_ready(action.impl, readiness)))
        for action in actions
    }
