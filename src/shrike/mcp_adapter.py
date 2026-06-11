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

from shrike.actions import ActionDef, ToolInputError
from shrike.collection import CollectionBusyError

logger = logging.getLogger("shrike.tools")

# A tool call slower than this logs its duration at INFO instead of DEBUG, so
# slowness is visible in a default-level log without per-call noise.
SLOW_TOOL_SECONDS = 1.0


def _log_duration(name: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    if elapsed >= SLOW_TOOL_SECONDS:
        logger.info("%s completed (%.1fs)", name, elapsed)
    else:
        logger.debug("%s completed (%.0fms)", name, elapsed * 1000)


def _safe_tool(fn: Any) -> Any:
    """Wrap a tool to log unhandled exceptions, then re-raise.

    A re-raised exception becomes an MCP ``isError`` result (FastMCP converts
    it), which the client surfaces as a ``ServerError``. Tools therefore never
    embed an ``error`` field in a success payload — protocol errors live in the
    protocol, and response models stay clean. ``ToolInputError`` (expected bad
    input) logs the rejection without a traceback; anything else logs with one.

    Every call is timed: completion duration logs at DEBUG, escalating to INFO
    for a slow call (>= 1s) so slowness surfaces without per-call log noise.

    The wrapped function's docstring is dedented with ``inspect.cleandoc`` so the
    tool description FastMCP advertises to clients has no source indentation.
    """
    cleaned_doc = inspect.cleandoc(fn.__doc__) if fn.__doc__ else None

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
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
            _log_duration(fn.__name__, started)
            return result

        async_wrapper.__doc__ = cleaned_doc
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
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
        _log_duration(fn.__name__, started)
        return result

    wrapper.__doc__ = cleaned_doc
    return wrapper


def register_actions(mcp: FastMCP, actions: list[ActionDef]) -> None:
    """Bind every registry action as an MCP tool (the old inline decoration order:
    ``mcp.tool()`` over the ``_safe_tool``-wrapped impl)."""
    for action in actions:
        mcp.tool()(_safe_tool(action.impl))
