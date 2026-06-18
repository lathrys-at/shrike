"""Performance shim for the MCP SDK's per-call JSON Schema validation.

The low-level MCP server validates **every** tool call's input against its
``inputSchema`` and output against its ``outputSchema`` with
``jsonschema.validate(instance, schema)`` (see ``mcp/server/lowlevel/server.py``).
That one-shot API rebuilds the validator and **re-resolves every ``$ref`` on each
call** — measured at ~5.8 ms per call for Shrike's nested response schemas,
dwarfing the actual tool work (~0.1 ms) and the MCP dispatch (~0.3 ms). It's pure
redundant overhead: Shrike's tools take typed params and return Pydantic models,
so both ends are already validated by Pydantic before the SDK re-checks them.

The original shim (#140) cached validator *compilation*, which fixed the small
payloads — but the data-proportional instance WALK survived, and at scale it
dominates: measured 6.5 ms (output, 200-note response) + 2.5 ms (input,
100-note upsert) per call against ~0.5 ms for everything else in the Python
layer combined (#445). Both ends are enforced by Pydantic regardless (typed
params in, response models out — with better error messages), so the walk is
now skipped entirely. A malformed input still fails the tool call: FastMCP's
Pydantic parse rejects it; the SDK's jsonschema pass was a second, slower
spelling of the same check.

Scope: we replace the ``jsonschema`` *name* inside the SDK's low-level server
module with a thin proxy, so the global ``jsonschema.validate`` is untouched —
only the MCP call path uses the cache.
"""

from __future__ import annotations

from typing import Any

_installed = False


class _CachingJsonschema:
    """Proxy over the ``jsonschema`` module whose ``validate`` is a no-op.

    (Name kept from the #140 compile-caching shim so the install seam stays
    recognizable.) Every other attribute (``ValidationError``, ``exceptions``,
    …) forwards to the real module.
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def validate(self, instance: Any, schema: Any) -> None:
        # Deliberately a no-op (#445): both ends of every Shrike tool call are
        # already Pydantic-validated; the SDK's per-call jsonschema walk is a
        # redundant O(payload) pass (~9 ms on the large calls).
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def install_validator_cache() -> None:
    """Swap the MCP low-level server's ``jsonschema`` for the caching proxy.

    Idempotent and best-effort: if the SDK's internals ever move, this quietly
    does nothing rather than breaking startup (the slow path still works).
    """
    global _installed
    if _installed:
        return
    try:
        import jsonschema
        from mcp.server.lowlevel import server as lls

        if not isinstance(lls.jsonschema, _CachingJsonschema):
            lls.jsonschema = _CachingJsonschema(jsonschema)  # type: ignore[assignment]
        _installed = True
    except Exception:  # pragma: no cover - defensive: never block startup
        pass
