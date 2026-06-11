"""The MCP binding for the action registry (#276, implements #225).

Iterates :func:`shrike.actions.build_actions`'s registry and generates the
FastMCP ``@mcp.tool`` bindings, applying the ``_safe_tool`` policy — docstring
``inspect.cleandoc`` (so advertised descriptions carry no source indentation)
and the error→``isError`` mapping (``ToolInputError``/``CollectionBusyError``
logged without tracebacks, genuine bugs with one). FastMCP's ``outputSchema``
emission is unchanged because the impls' response models are unchanged.

This is one adapter among possible several: the registry itself is
FastMCP-free, so #225's future agent-runtime adapters (on-device
function-calling) bind the same actions without touching this module.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from shrike.actions import ActionDef, ToolInputError, _call_outcome
from shrike.collection import CollectionBusyError

logger = logging.getLogger("shrike.tools")

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
    # THE log line for a served call (#328): one INFO line carrying the tool
    # name, its params, the action-recorded outcome, and the duration. Actions
    # contribute the outcome via note_outcome(); anything else they log during
    # the call is a warning/error (exceptional) or DEBUG (internals).
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
        except Exception:
            logger.exception("Unhandled error in %s", fn.__name__)
            raise
        _log_completed(fn.__name__, kwargs, started)
        return result

    wrapper.__doc__ = cleaned_doc
    return wrapper


def register_actions(mcp: FastMCP, actions: list[ActionDef]) -> None:
    """Bind every registry action as an MCP tool (the old inline decoration order:
    ``mcp.tool()`` over the ``_safe_tool``-wrapped impl)."""
    for action in actions:
        mcp.tool()(_safe_tool(action.impl))
