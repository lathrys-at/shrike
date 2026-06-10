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
from typing import Any

from mcp.server.fastmcp import FastMCP

from shrike.actions import ActionDef, ToolInputError
from shrike.collection import CollectionBusyError

logger = logging.getLogger("shrike.tools")


def _safe_tool(fn: Any) -> Any:
    """Wrap a tool to log unhandled exceptions, then re-raise.

    A re-raised exception becomes an MCP ``isError`` result (FastMCP converts
    it), which the client surfaces as a ``ServerError``. Tools therefore never
    embed an ``error`` field in a success payload — protocol errors live in the
    protocol, and response models stay clean. ``ToolInputError`` (expected bad
    input) re-raises quietly; anything else logs with a traceback.

    The wrapped function's docstring is dedented with ``inspect.cleandoc`` so the
    tool description FastMCP advertises to clients has no source indentation.
    """
    cleaned_doc = inspect.cleandoc(fn.__doc__) if fn.__doc__ else None

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolInputError:
                raise
            except CollectionBusyError as e:
                # Expected under cooperative locking — another process holds the
                # collection. Surface the coded message (client maps it), no trace.
                logger.warning("%s in %s", e, fn.__name__)
                raise
            except Exception:
                logger.exception("Unhandled error in %s", fn.__name__)
                raise

        async_wrapper.__doc__ = cleaned_doc
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolInputError:
            raise
        except CollectionBusyError as e:
            logger.warning("%s in %s", e, fn.__name__)
            raise
        except Exception:
            logger.exception("Unhandled error in %s", fn.__name__)
            raise

    wrapper.__doc__ = cleaned_doc
    return wrapper


def register_actions(mcp: FastMCP, actions: list[ActionDef]) -> None:
    """Bind every registry action as an MCP tool (the old inline decoration order:
    ``mcp.tool()`` over the ``_safe_tool``-wrapped impl)."""
    for action in actions:
        mcp.tool()(_safe_tool(action.impl))
