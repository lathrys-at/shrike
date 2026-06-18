"""Numeric-bounds parity between schemas.py (Pydantic) and shrike-schemas (Rust).

The structural contract test (``test_schema_contract.py``) deliberately leaves
numeric bounds out of the shape comparison; this module pins the two cases so
the drift can't silently return:

(a) **Defensive direction** — fields the kernel (Rust) produces and Python only
    receives. Rust's ``u32``/``u64`` reject negatives/overflow at the serde
    boundary; Python enforces ``ge=0`` too, so both sides reject the same bad
    values (``ExportPackageResult.note_count``, ``ExportPackage{Path,Url}.bytes``
    + ``note_count``, ``ServerStatus.wire_protocol_version``).

(b) **Advertised-schema direction** — fields Python enforces (``ge=``) whose
    Rust ``schemars`` schema (what MCP ``tools/list`` advertises) must also
    declare the ``minimum`` (``FieldMetadataInput.size`` ≥ 1;
    ``FieldOp``/``TemplateOp`` ``position`` ≥ 0), or a client reading that
    schema gets inaccurate type info. The Rust *serde* round-trip still accepts
    any integer (the bound lives in the schema document, not in
    deserialization) — the contract is the advertised schema, asserted here.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

import shrike.schemas as schemas

shrike_native = pytest.importorskip("shrike_native")


# ── (a) defensive direction: Python now rejects what Rust already rejected ────


class TestExportPackageResultNoteCount:
    def test_python_rejects_negative_note_count(self) -> None:
        with pytest.raises(ValidationError):
            schemas.ExportPackageResult(note_count=-1, out_path="/tmp/x.apkg")

    def test_rust_rejects_negative_note_count(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResult", json.dumps({"note_count": -1, "out_path": "/tmp/x.apkg"})
            )

    def test_rust_rejects_out_of_u32_range(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResult",
                json.dumps({"note_count": 4294967296, "out_path": "/tmp/x.apkg"}),
            )

    def test_both_accept_a_valid_count(self) -> None:
        r = schemas.ExportPackageResult(note_count=42, out_path="/tmp/x.apkg")
        assert r.note_count == 42
        back = json.loads(
            shrike_native.schema_roundtrip(
                "ExportPackageResult", json.dumps({"note_count": 42, "out_path": "/tmp/x.apkg"})
            )
        )
        assert back["note_count"] == 42


class TestExportPackageResponseBounds:
    def test_python_rejects_negative_bytes_path(self) -> None:
        with pytest.raises(ValidationError):
            schemas.ExportPackagePath(
                delivery="path", note_count=5, bytes=-1, format="apkg", path="/x"
            )

    def test_python_rejects_negative_bytes_url(self) -> None:
        with pytest.raises(ValidationError):
            schemas.ExportPackageUrl(
                delivery="url", note_count=5, bytes=-1, format="apkg", url="http://x"
            )

    def test_python_rejects_negative_note_count_response(self) -> None:
        ta = TypeAdapter(schemas.ExportPackageResponse)
        with pytest.raises(ValidationError):
            ta.validate_python(
                {"delivery": "path", "note_count": -1, "bytes": 100, "format": "apkg", "path": "/x"}
            )

    def test_rust_rejects_negative_bytes_path(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResponse",
                json.dumps(
                    {
                        "delivery": "path",
                        "note_count": 5,
                        "bytes": -1,
                        "format": "apkg",
                        "path": "/x",
                    }
                ),
            )

    def test_rust_rejects_negative_note_count_response(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResponse",
                json.dumps(
                    {
                        "delivery": "url",
                        "note_count": -1,
                        "bytes": 100,
                        "format": "colpkg",
                        "url": "http://x",
                    }
                ),
            )


class TestServerStatusWireProtocolVersion:
    def _valid_kwargs(self, **over: object) -> dict[str, object]:
        base: dict[str, object] = {
            "wire_protocol_version": 1,
            "pid": 123,
            "url": "http://127.0.0.1:8372",
            "collection": "/tmp/test.anki2",
            "log_level": "INFO",
            "log_dir": "/tmp/logs",
            "embedding": {"state": "not_configured", "available": False},
            "index": {"state": "unavailable"},
        }
        base.update(over)
        return base

    def test_python_rejects_negative_version(self) -> None:
        with pytest.raises(ValidationError):
            schemas.ServerStatus(**self._valid_kwargs(wire_protocol_version=-1))

    def test_python_accepts_valid_version(self) -> None:
        s = schemas.ServerStatus(**self._valid_kwargs(wire_protocol_version=1))
        assert s.wire_protocol_version == 1

    def test_rust_rejects_negative_version(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ServerStatus", json.dumps(self._valid_kwargs(wire_protocol_version=-1))
            )


# ── (b) advertised-schema direction: the Rust schema now declares the bound ───


def _property_minimum(schema: dict, prop: str) -> object:
    """The ``minimum`` declared on a property, unwrapping an ``Option`` anyOf."""
    node = schema.get("properties", {}).get(prop, {})
    if "minimum" in node:
        return node["minimum"]
    for branch in node.get("anyOf", []):
        if branch.get("type") != "null" and "minimum" in branch:
            return branch["minimum"]
    return None


def _tagged_variant(schema: dict, tag: str, value: str) -> dict:
    branches = schema.get("oneOf") or schema.get("anyOf") or []
    for branch in branches:
        node = branch.get("properties", {}).get(tag, {})
        values = [node["const"]] if "const" in node else node.get("enum", [])
        if value in values:
            return branch
    raise AssertionError(f"no {tag}={value} variant in {schema}")


class TestAdvertisedSchemaBounds:
    """The Python ``ge=`` bounds these fields enforce must now appear as
    ``minimum`` in the Rust schema served via MCP ``tools/list``."""

    def test_field_metadata_size_schema_declares_minimum_1(self) -> None:
        # Python enforces ge=1.
        with pytest.raises(ValidationError):
            schemas.FieldMetadataInput(name="Front", size=0)
        # The advertised Rust schema now declares it too.
        schema = json.loads(shrike_native.schema_catalog()["FieldMetadataInput"])
        assert _property_minimum(schema, "size") == 1

    def test_field_op_position_schema_declares_minimum_0(self) -> None:
        with pytest.raises(ValidationError):
            schemas.FieldAdd(op="add", name="F", position=-1)
        with pytest.raises(ValidationError):
            schemas.FieldReposition(op="reposition", name="F", position=-1)
        schema = json.loads(shrike_native.schema_catalog()["FieldOp"])
        assert _property_minimum(_tagged_variant(schema, "op", "add"), "position") == 0
        assert _property_minimum(_tagged_variant(schema, "op", "reposition"), "position") == 0

    def test_template_op_position_schema_declares_minimum_0(self) -> None:
        with pytest.raises(ValidationError):
            schemas.TemplateOpAdd(op="add", name="T", front="F", back="B", position=-1)
        with pytest.raises(ValidationError):
            schemas.TemplateOpReposition(op="reposition", name="T", position=-1)
        schema = json.loads(shrike_native.schema_catalog()["TemplateOp"])
        assert _property_minimum(_tagged_variant(schema, "op", "add"), "position") == 0
        assert _property_minimum(_tagged_variant(schema, "op", "reposition"), "position") == 0
