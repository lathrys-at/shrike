"""The JSON Schema validation shim (shrike._mcp_perf).

Since #445 the proxy's ``validate`` is a deliberate NO-OP: both ends of every
tool call are already Pydantic-validated, and the SDK's data-proportional
jsonschema walk measured ~9 ms on large calls. These tests pin that contract:
validate never raises (Pydantic is the enforcement layer), other module
attributes forward, and the install is idempotent and scoped to the SDK's
low-level server module.
"""

from __future__ import annotations

import jsonschema

from shrike import _mcp_perf
from shrike._mcp_perf import _CachingJsonschema, install_validator_cache

_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
}


class TestNoOpProxy:
    def test_accepts_valid_instance(self) -> None:
        _CachingJsonschema(jsonschema).validate({"x": 1}, _SCHEMA)

    def test_does_not_raise_on_invalid_instance(self) -> None:
        # The walk is skipped by design (#445): Pydantic rejects malformed
        # input/output with better errors before/after the SDK would.
        _CachingJsonschema(jsonschema).validate({"x": "nope"}, _SCHEMA)

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
