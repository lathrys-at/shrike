"""S4 surface review: failing tests for confirmed schema parity defects.

Each test documents a real, reproducible defect between schemas.py (Python)
and shrike-schemas/src/lib.rs (Rust canonical). Run with:

    SHRIKE_SKIP_NATIVE_STALE_CHECK=1 \
        /Users/lupine/Development/shrike/.venv/bin/python -m pytest \
        tests/unit/test_s4_review_findings.py -v -p no:cacheprovider
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

import shrike.schemas as schemas

shrike_native = pytest.importorskip("shrike_native")


# ---------------------------------------------------------------------------
# FINDING S4-1: ExportPackageResult.note_count type drift (Python int vs Rust u32)
# Rust rejects negative note_count; Python accepts it silently.
# ---------------------------------------------------------------------------


class TestS4_1_ExportPackageResultNoteCount:
    """Python ExportPackageResult accepts negative note_count; Rust rejects it.

    Python: note_count: int  (unbounded)
    Rust:   note_count: u32  (minimum: 0, maximum: 4294967295)

    Path: kernel (Rust) -> action layer (Python) -> MCP client.
    Any Python mock or future direct construction can produce note_count=-1.
    A Rust consumer of that wire payload will fail to parse it.
    """

    def test_python_accepts_negative_note_count(self) -> None:
        # Python does NOT reject note_count=-1 — no bound enforced.
        r = schemas.ExportPackageResult(note_count=-1, out_path="/tmp/x.apkg")
        assert r.note_count == -1  # silently accepted

    def test_rust_rejects_negative_note_count(self) -> None:
        # Rust u32 cannot represent -1.
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResult", json.dumps({"note_count": -1, "out_path": "/tmp/x.apkg"})
            )

    def test_python_accepts_out_of_u32_range(self) -> None:
        # u32 max is 4294967295; Python accepts 4294967296 (u32 + 1).
        r = schemas.ExportPackageResult(note_count=4294967296, out_path="/tmp/x.apkg")
        assert r.note_count == 4294967296

    def test_rust_rejects_out_of_u32_range(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResult",
                json.dumps({"note_count": 4294967296, "out_path": "/tmp/x.apkg"}),
            )


# ---------------------------------------------------------------------------
# FINDING S4-2: ExportPackageResponse.bytes type drift (Python int vs Rust u64)
# and ExportPackageResponse.note_count type drift (Python int vs Rust u32).
# Both Python variants (ExportPackagePath, ExportPackageUrl) accept negative values
# that Rust rejects.
# ---------------------------------------------------------------------------


class TestS4_2_ExportPackageResponseBytes:
    """Python ExportPackageResponse accepts negative bytes; Rust rejects it.

    Python ExportPackagePath/ExportPackageUrl: bytes: int  (unbounded)
    Rust ExportPackageResponse::Path / ::Url:  bytes: u64  (minimum: 0)
    """

    def test_python_path_accepts_negative_bytes(self) -> None:
        r = schemas.ExportPackagePath(
            delivery="path", note_count=5, bytes=-1, format="apkg", path="/x"
        )
        assert r.bytes == -1  # silently accepted

    def test_rust_rejects_negative_bytes_in_path_variant(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResponse",
                json.dumps(
                    {"delivery": "path", "note_count": 5, "bytes": -1, "format": "apkg", "path": "/x"}
                ),
            )

    def test_python_url_accepts_negative_bytes(self) -> None:
        r = schemas.ExportPackageUrl(
            delivery="url", note_count=5, bytes=-1, format="apkg", url="http://x"
        )
        assert r.bytes == -1  # silently accepted

    def test_rust_rejects_negative_bytes_in_url_variant(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResponse",
                json.dumps(
                    {
                        "delivery": "url",
                        "note_count": 5,
                        "bytes": -1,
                        "format": "apkg",
                        "url": "http://x",
                    }
                ),
            )

    def test_python_accepts_negative_note_count_in_response(self) -> None:
        ta = TypeAdapter(schemas.ExportPackageResponse)
        r = ta.validate_python(
            {"delivery": "path", "note_count": -1, "bytes": 100, "format": "apkg", "path": "/x"}
        )
        assert isinstance(r, schemas.ExportPackagePath)
        assert r.note_count == -1

    def test_rust_rejects_negative_note_count_in_response(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip(
                "ExportPackageResponse",
                json.dumps(
                    {
                        "delivery": "path",
                        "note_count": -1,
                        "bytes": 100,
                        "format": "apkg",
                        "path": "/x",
                    }
                ),
            )


# ---------------------------------------------------------------------------
# FINDING S4-3: ServerStatus.wire_protocol_version type drift
# (Python int vs Rust u32).
# ---------------------------------------------------------------------------


class TestS4_3_WireProtocolVersionType:
    """Python ServerStatus accepts negative wire_protocol_version; Rust rejects it.

    Python: wire_protocol_version: int  (unbounded)
    Rust:   wire_protocol_version: u32  (minimum: 0)

    This field appears in GET /status responses. Python produces it; the
    Python CLI client validates it without bounds. A Rust client would fail.
    """

    def test_python_accepts_negative_version(self) -> None:
        # Python ServerStatus doesn't validate this field's range.
        s = schemas.ServerStatus(
            wire_protocol_version=-1,
            pid=123,
            url="http://127.0.0.1:8372",
            collection="/tmp/test.anki2",
            log_level="INFO",
            log_dir="/tmp/logs",
            embedding={"state": "not_configured", "available": False},
            index={"state": "unavailable"},
        )
        assert s.wire_protocol_version == -1

    def test_rust_rejects_negative_version(self) -> None:
        payload = {
            "wire_protocol_version": -1,
            "pid": 123,
            "url": "http://127.0.0.1:8372",
            "collection": "/tmp/test.anki2",
            "log_level": "INFO",
            "log_dir": "/tmp/logs",
            "embedding": {"state": "not_configured", "available": False},
            "index": {"state": "unavailable"},
        }
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip("ServerStatus", json.dumps(payload))


# ---------------------------------------------------------------------------
# FINDING S4-4: FieldMetadataInput.size bound present in Python (ge=1) but
# absent from the Rust schema, meaning the published JSON Schema (from Rust's
# schemars output, which appears in MCP `tools/list`) omits the constraint.
# NOTE: The MCP / actions-over-HTTP paths both run Python Pydantic validation,
# so size=0 is REJECTED at runtime. The defect is that the Rust schema
# (which clients use for type information) doesn't advertise the bound.
# ---------------------------------------------------------------------------


class TestS4_4_FieldMetadataInputSizeBound:
    """Python FieldMetadataInput.size has ge=1; Rust schema has no minimum.

    Python: size: int | None = Field(default=None, ge=1, ...)
    Rust:   size: Option<i64>  (no minimum constraint)

    The Rust schema (served via MCP tools/list) claims size can be any integer.
    Python Pydantic rejects size <= 0 on the actual call path.
    An MCP client following the Rust schema strictly could send size=0 and
    expect it to be accepted — but the server rejects it at 400.
    """

    def test_python_rejects_zero_size(self) -> None:
        with pytest.raises(ValidationError):
            schemas.FieldMetadataInput(name="Front", size=0)

    def test_python_rejects_negative_size(self) -> None:
        with pytest.raises(ValidationError):
            schemas.FieldMetadataInput(name="Front", size=-1)

    def test_rust_accepts_zero_size_in_schema(self) -> None:
        # Rust does NOT reject size=0 during roundtrip (no bound in its schema).
        result = shrike_native.schema_roundtrip(
            "FieldMetadataInput", json.dumps({"name": "Front", "size": 0})
        )
        parsed = json.loads(result)
        assert parsed["size"] == 0  # Rust accepted it

    def test_rust_accepts_negative_size_in_schema(self) -> None:
        result = shrike_native.schema_roundtrip(
            "FieldMetadataInput", json.dumps({"name": "Front", "size": -5})
        )
        parsed = json.loads(result)
        assert parsed["size"] == -5  # Rust accepted it — bound is missing

    def test_rust_schema_lacks_minimum_for_size(self) -> None:
        """The Rust JSON Schema has no 'minimum' constraint on size."""
        rust_schema_json = shrike_native.schema_catalog()["FieldMetadataInput"]
        rust_schema = json.loads(rust_schema_json)
        size_field = rust_schema.get("properties", {}).get("size", {})
        # If size is anyOf/oneOf (Option), unwrap the non-null branch.
        if "anyOf" in size_field:
            non_null = [b for b in size_field["anyOf"] if b.get("type") != "null"]
            size_field = non_null[0] if non_null else {}
        assert "minimum" not in size_field, (
            f"Expected no 'minimum' in Rust FieldMetadataInput.size schema, "
            f"but got: {size_field}"
        )


# ---------------------------------------------------------------------------
# FINDING S4-5: FieldOp/TemplateOp position fields have ge=0 in Python but
# no bound in Rust schemas, with the same schema-mismatch consequence as S4-4.
# ---------------------------------------------------------------------------


class TestS4_5_PositionFieldBounds:
    """FieldAdd/FieldReposition/TemplateOpAdd/TemplateOpReposition position bounds
    present in Python (ge=0) but absent from the Rust schema.

    Python enforces the bound at runtime; the Rust schema (advertised to clients)
    does not declare it, so MCP clients have inaccurate type information.
    """

    def test_python_rejects_negative_field_add_position(self) -> None:
        with pytest.raises(ValidationError):
            schemas.FieldAdd(op="add", name="F", position=-1)

    def test_rust_accepts_negative_field_add_position(self) -> None:
        result = json.loads(
            shrike_native.schema_roundtrip(
                "FieldOp", json.dumps({"op": "add", "name": "F", "position": -1})
            )
        )
        assert result["position"] == -1  # Rust accepted it

    def test_python_rejects_negative_field_reposition(self) -> None:
        with pytest.raises(ValidationError):
            schemas.FieldReposition(op="reposition", name="F", position=-1)

    def test_rust_accepts_negative_field_reposition(self) -> None:
        result = json.loads(
            shrike_native.schema_roundtrip(
                "FieldOp", json.dumps({"op": "reposition", "name": "F", "position": -1})
            )
        )
        assert result["position"] == -1

    def test_python_rejects_negative_template_op_add_position(self) -> None:
        with pytest.raises(ValidationError):
            schemas.TemplateOpAdd(op="add", name="T", front="F", back="B", position=-1)

    def test_rust_accepts_negative_template_op_add_position(self) -> None:
        result = json.loads(
            shrike_native.schema_roundtrip(
                "TemplateOp",
                json.dumps({"op": "add", "name": "T", "front": "F", "back": "B", "position": -1}),
            )
        )
        assert result["position"] == -1

    def test_python_rejects_negative_template_op_reposition(self) -> None:
        with pytest.raises(ValidationError):
            schemas.TemplateOpReposition(op="reposition", name="T", position=-1)

    def test_rust_accepts_negative_template_op_reposition(self) -> None:
        result = json.loads(
            shrike_native.schema_roundtrip(
                "TemplateOp",
                json.dumps({"op": "reposition", "name": "T", "position": -1}),
            )
        )
        assert result["position"] == -1
