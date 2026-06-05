"""The JSON Schema validator cache shim (shrike._mcp_perf).

Keeps the MCP SDK's per-call input/output validation behaviour but compiles each
schema once. These tests pin the behaviour-preserving contract: still raises on
bad data, reuses one validator per schema, forwards other module attributes, and
the install is idempotent and scoped to the SDK's low-level server module.
"""

from __future__ import annotations

import jsonschema
import pytest

from shrike import _mcp_perf
from shrike._mcp_perf import _CachingJsonschema, install_validator_cache

_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
}


class TestCachingProxy:
    def test_accepts_valid_instance(self) -> None:
        _CachingJsonschema(jsonschema).validate({"x": 1}, _SCHEMA)

    def test_rejects_invalid_instance(self) -> None:
        with pytest.raises(jsonschema.ValidationError):
            _CachingJsonschema(jsonschema).validate({"x": "nope"}, _SCHEMA)

    def test_reuses_one_validator_per_schema(self) -> None:
        proxy = _CachingJsonschema(jsonschema)
        proxy.validate({"x": 1}, _SCHEMA)
        proxy.validate({"x": 2}, _SCHEMA)
        assert len(proxy._cache) == 1

    def test_rebuilds_on_id_reuse(self) -> None:
        # A different schema object that happens to reuse a freed id must not get
        # the stale validator — the pinned-identity guard rebuilds.
        proxy = _CachingJsonschema(jsonschema)
        proxy.validate({"x": 1}, _SCHEMA)
        (cached_schema, _) = next(iter(proxy._cache.values()))
        proxy._cache[id(cached_schema)] = ({"type": "string"}, object())  # poison
        # Same schema object again: entry[0] is not _SCHEMA → rebuild, validates ok.
        proxy.validate({"x": 3}, _SCHEMA)

    def test_forwards_other_attributes(self) -> None:
        proxy = _CachingJsonschema(jsonschema)
        assert proxy.ValidationError is jsonschema.ValidationError


class TestInstall:
    def test_swaps_and_is_idempotent(self) -> None:
        from mcp.server.lowlevel import server as lls

        original = lls.jsonschema
        was_installed = _mcp_perf._installed
        _mcp_perf._installed = False
        try:
            install_validator_cache()
            assert isinstance(lls.jsonschema, _CachingJsonschema)
            first = lls.jsonschema
            install_validator_cache()  # idempotent — no re-wrap
            assert lls.jsonschema is first
        finally:
            lls.jsonschema = original
            _mcp_perf._installed = was_installed
