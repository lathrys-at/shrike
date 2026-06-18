"""The tool surface — a composition shim over the action core.

The tool implementations live in ``actions.py`` as a transport-neutral
registry; ``mcp_adapter.py`` binds them to FastMCP with the ``_safe_tool``
policy. :func:`register_tools` keeps its signature as the composition of the
two, so every existing caller — the server boot path and the tools-layer test
files — is unchanged, and the wire surface (tools/list schemas, behaviour) is
byte-identical.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool

# Re-exports that are part of the module's surface (ACTIVATION_MARGIN is
# imported by tests; ToolInputError is the error contract's home name).
from shrike.api.actions import (
    ACTIVATION_MARGIN,
    ActionContext,
    ToolInputError,
    build_actions,
)
from shrike.api.mcp_adapter import _safe_tool, build_action_tools, register_actions
from shrike.harness.collection import CollectionWrapper
from shrike.harness.derived import DerivedTextStore

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
    server_import_path_roots: list[str] | None = None,
    media_base_url: str | None = None,
    export_path_roots: list[str] | None = None,
    export_store: Any | None = None,
    server_purely_local: bool = False,
    registry: Any | None = None,
    resolver: Any | None = None,
) -> dict[str, Tool]:
    """Build the action registry against this server's context and bind it to MCP.

    ``kernel`` (the AsyncKernel) is required: write paths route through the
    maintained kernel ops, and ``index`` carries the search-facing
    ``KernelIndexView`` (or None when embedding is unconfigured). ``registry``
    is the collection/profile registry snapshot the ``list_profiles``
    enumeration reads; None means an empty registry. ``resolver`` is the
    per-call collection router: an async ``selector -> CollectionBundle``;
    None keeps single-collection mode (the fixed handles are the one bundle).

    Returns the ``name -> Tool`` map for the actions-over-HTTP edge, built
    from the *same* action registry as the MCP binding — so the host can register
    ``POST /actions/{name}`` over an identical catalog (one core, two adapters).
    """
    context = ActionContext(
        wrapper=wrapper,
        index=index,
        derived=derived,
        kernel=kernel,
        dedup_stats=dedup_stats,
        allow_private_fetch=allow_private_fetch,
        server_path_roots=server_path_roots,
        server_import_path_roots=server_import_path_roots,
        media_base_url=media_base_url,
        export_path_roots=export_path_roots,
        export_store=export_store,
        server_purely_local=server_purely_local,
        registry=registry,
        resolver=resolver,
    )
    actions = build_actions(context)
    register_actions(mcp, actions)
    return build_action_tools(actions)
