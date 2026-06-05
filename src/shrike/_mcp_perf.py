"""Performance shim for the MCP SDK's per-call JSON Schema validation.

The low-level MCP server validates **every** tool call's input against its
``inputSchema`` and output against its ``outputSchema`` with
``jsonschema.validate(instance, schema)`` (see ``mcp/server/lowlevel/server.py``).
That one-shot API rebuilds the validator and **re-resolves every ``$ref`` on each
call** — measured at ~5.8 ms per call for Shrike's nested response schemas,
dwarfing the actual tool work (~0.1 ms) and the MCP dispatch (~0.3 ms). It's pure
redundant overhead: Shrike's tools take typed params and return Pydantic models,
so both ends are already validated by Pydantic before the SDK re-checks them.

We don't *disable* the SDK's validation (it's a cheap belt-and-suspenders once
fast, and there's no toggle for output validation anyway) — we make it fast by
reusing one compiled validator per schema. The SDK caches its ``Tool``
definitions, so the schema dicts are stable for the server's lifetime; we key on
identity and pin the schema object so its ``id`` can't be recycled. Behavior is
identical: a bad instance still raises ``jsonschema.ValidationError``.

Scope: we replace the ``jsonschema`` *name* inside the SDK's low-level server
module with a thin proxy, so the global ``jsonschema.validate`` is untouched —
only the MCP call path uses the cache.
"""

from __future__ import annotations

from typing import Any

_installed = False


class _CachingJsonschema:
    """Proxy over the ``jsonschema`` module that compiles each schema once.

    ``validate`` reuses a per-schema validator (keyed on object identity, with
    the schema pinned so the id is stable); every other attribute
    (``ValidationError``, ``exceptions``, …) forwards to the real module.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self._cache: dict[int, tuple[Any, Any]] = {}

    def validate(self, instance: Any, schema: Any) -> None:
        entry = self._cache.get(id(schema))
        if entry is None or entry[0] is not schema:
            cls = self._real.validators.validator_for(schema)
            entry = (schema, cls(schema))
            self._cache[id(schema)] = entry
        entry[1].validate(instance)

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
