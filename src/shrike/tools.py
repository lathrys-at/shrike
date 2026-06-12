"""The tool surface — now a composition shim over the action core (#276).

The 24 tool implementations live in ``actions.py`` as a transport-neutral
registry (the #225 design); ``mcp_adapter.py`` binds them to FastMCP with the
``_safe_tool`` policy. :func:`register_tools` keeps its signature as the
composition of the two, so every existing caller — the server boot path and the
tools-layer test files — is unchanged, and the wire surface (tools/list
schemas, behaviour) is byte-identical.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

# Re-exports: these lived here pre-split and are part of the module's surface
# (ACTIVATION_MARGIN is imported by tests; ToolInputError is the error
# contract's home name).
from shrike.actions import (
    ACTIVATION_MARGIN,
    ActionContext,
    ToolInputError,
    build_actions,
)
from shrike.collection import CollectionWrapper
from shrike.derived import DerivedTextStore
from shrike.mcp_adapter import _safe_tool, register_actions

__all__ = [
    "ACTIVATION_MARGIN",
    "ToolInputError",
    "_safe_tool",
    "register_tools",
]


def register_tools(
    mcp: FastMCP,
    wrapper: CollectionWrapper,
    index: Any | None = None,
    *,
    derived: DerivedTextStore | None = None,
    kernel: Any | None = None,
    dedup_stats: Any | None = None,
    allow_private_fetch: bool = False,
    server_path_roots: list[str] | None = None,
    media_base_url: str | None = None,
) -> None:
    """Build the action registry against this server's context and bind it to MCP.

    ``kernel`` (the AsyncKernel) is required (#355): write paths route through
    the maintained kernel ops, and ``index`` carries the search-facing
    ``KernelIndexView`` (or None when embedding is unconfigured).
    """
    context = ActionContext(
        wrapper=wrapper,
        index=index,
        derived=derived,
        kernel=kernel,
        dedup_stats=dedup_stats,
        allow_private_fetch=allow_private_fetch,
        server_path_roots=server_path_roots,
        media_base_url=media_base_url,
    )
    register_actions(mcp, build_actions(context))
