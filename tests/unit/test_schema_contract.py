"""The schema contract test (#330, kernel inversion S1).

``shrike/schemas.py`` (Pydantic) and ``shrike-schemas`` (serde + schemars) must
describe the same wire. Two gates:

1. **Structural schema equivalence** — every Pydantic model/union is normalized
   to a canonical shape (properties, requiredness, literal/enum values, tagged-
   union variants keyed by tag value) and compared against the normalized
   schemars output. Descriptions/titles/defaults/numeric bounds are not part of
   the contract (bounds live on tool *params*, validated by FastMCP).
2. **Instance round-trips** — representative payloads serialize from Pydantic,
   ride through the Rust types (``shrike_native.schema_roundtrip`` = parse +
   re-emit), and come back wire-identical; so field names, tag values, and
   null-vs-absent semantics agree in practice, not just in schema.

Coverage is structural: the test *enumerates* schemas.py's models, so adding a
model without a Rust counterpart (or vice versa) fails here.
"""

from __future__ import annotations

import json
from typing import Annotated, get_origin

import pytest
from pydantic import BaseModel, TypeAdapter

import shrike.schemas as schemas

shrike_native = pytest.importorskip("shrike_native")


def test_wire_protocol_version_mirrors_rust() -> None:
    # The #392 constant: one number, two homes, never allowed to drift.
    assert shrike_native.wire_protocol_version() == schemas.WIRE_PROTOCOL_VERSION


# Union aliases (Annotated[..., Field(discriminator=...)]) and their tag field.
UNIONS: dict[str, str] = {
    "UpsertNoteResult": "status",
    "NoteTypeResult": "status",
    "FieldOp": "op",
    "TemplateOp": "op",
    "UpsertDeckResult": "status",
    "DeleteNoteTypeResult": "status",
    "StoreMediaResult": "status",
    "MediaFetchResult": "status",
    "EmbeddingStatus": "state",
    "IndexStatus": "state",
    "IndexRebuildResponse": "status",
    "IndexSaveResponse": "status",
    "EmbeddingStartResponse": "status",
    "EmbeddingStopResponse": "status",
    "StopResponse": "stopped",
}

# Pydantic models that are union *variants* (covered through their union's
# entry) or deliberately Python-side-only.
VARIANT_OR_LOCAL = {
    # UpsertNoteResult
    "UpsertNoteOk",
    "UpsertNoteValidated",
    "UpsertNoteSkipped",
    "UpsertNoteError",
    # NoteTypeResult
    "NoteTypeOk",
    "NoteTypeError",
    # FieldOp / TemplateOp
    "FieldAdd",
    "FieldRemove",
    "FieldRename",
    "FieldReposition",
    "TemplateOpAdd",
    "TemplateOpRemove",
    "TemplateOpRename",
    "TemplateOpReposition",
    # UpsertDeckResult / DeleteNoteTypeResult
    "UpsertDeckOk",
    "UpsertDeckError",
    "NoteTypeDeleted",
    "NoteTypeNotFound",
    "NoteTypeDeleteError",
    # StoreMediaResult / MediaFetchResult
    "StoreMediaOk",
    "StoreMediaError",
    "MediaFile",
    "MediaMissing",
    # EmbeddingStatus / IndexStatus
    "EmbeddingRunning",
    "EmbeddingDown",
    "IndexUnavailable",
    "IndexBuilding",
    "IndexReady",
    "IndexErrored",
    # custom-endpoint unions
    "IndexRebuildStarted",
    "IndexRebuildComplete",
    "IndexRebuildAlreadyBuilding",
    "IndexSaved",
    "IndexSaveEmpty",
    "IndexSaveBuilding",
    "EmbeddingStarted",
    "EmbeddingAlreadyRunning",
    "EmbeddingStopped",
    "EmbeddingNotRunning",
    "StopSucceeded",
    "StopFailed",
}


def _python_models() -> dict[str, TypeAdapter]:
    """Every contract-bearing name in schemas.py → its TypeAdapter."""
    out: dict[str, TypeAdapter] = {}
    for name in dir(schemas):
        if name.startswith("_") or name in VARIANT_OR_LOCAL:
            continue
        obj = getattr(schemas, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            if obj.__module__ != "shrike.schemas":
                continue  # imported names (BaseModel itself), not contract models
            out[name] = TypeAdapter(obj)
        elif get_origin(obj) is Annotated and name in UNIONS:
            out[name] = TypeAdapter(obj)
    return out


# ── the normalizer ──────────────────────────────────────────────────────────
# Reduces both schema dialects (Pydantic v2 / schemars 1) to one canonical,
# comparable structure. Hashable so shapes can be dict keys / set members.


def _resolve(node: dict, defs: dict) -> dict:
    """Follow a $ref / unwrap a single-element allOf (Pydantic's default+ref form)."""
    seen = 0
    while seen < 50:
        seen += 1
        if "$ref" in node:
            ref_name = node["$ref"].split("/")[-1]
            merged = dict(defs[ref_name])
            # A ref carrying siblings (e.g. default) — the ref target wins shape-wise.
            node = merged
            continue
        if "allOf" in node and len(node["allOf"]) == 1:
            extra = {k: v for k, v in node.items() if k != "allOf"}
            node = {**node["allOf"][0], **extra}
            continue
        return node
    raise AssertionError("ref/allOf resolution did not converge")


def _is_null(node: dict) -> bool:
    return node.get("type") == "null"


_TAG_CANDIDATES = ("status", "state", "op", "stopped")


def _detect_tag(branches: list[dict], defs: dict) -> str | None:
    """The discriminator of a union met without its alias name, if it has one."""
    for candidate in _TAG_CANDIDATES:
        ok = True
        for branch in branches:
            props = branch.get("properties")
            if not props or candidate not in props:
                ok = False
                break
            tag_schema = _resolve(props[candidate], defs)
            if "const" not in tag_schema and "enum" not in tag_schema:
                ok = False
                break
        if ok:
            return candidate
    return None


def _norm(node: dict, defs: dict, tag: str | None = None):  # noqa: C901
    node = _resolve(node, defs)

    # const / single-value enum (the two spellings are the same contract)
    if "const" in node:
        return ("const", json.dumps(node["const"]))
    if "enum" in node and "properties" not in node:
        values = frozenset(json.dumps(v) for v in node["enum"])
        if len(values) == 1:
            return ("const", next(iter(values)))
        return ("enum", values)

    # union forms
    branches = node.get("anyOf") or node.get("oneOf")
    if branches is not None:
        resolved = [_resolve(b, defs) for b in branches]
        non_null = [b for b in resolved if not _is_null(b)]
        if len(non_null) == len(resolved) - 1:
            # X | None
            return ("optional", _norm(non_null[0], defs, tag))
        if tag is None:
            # A nested tagged union (reached through a field, where the alias
            # name — and its declared tag — is gone): detect the tag as the
            # const/enum property every branch requires.
            tag = _detect_tag(non_null, defs)
        if tag is not None:
            # A tagged union: key each branch's shape by its tag value(s).
            variants: dict[str, object] = {}
            for branch in non_null:
                props = branch.get("properties", {})
                tag_schema = _resolve(props.get(tag, {}), defs)
                values = (
                    [tag_schema["const"]] if "const" in tag_schema else tag_schema.get("enum", [])
                )
                assert values, f"tagged-union branch has no {tag} values: {branch}"
                without_tag = {
                    "type": "object",
                    "properties": {k: v for k, v in props.items() if k != tag},
                    "required": [r for r in branch.get("required", []) if r != tag],
                }
                shape = _norm(without_tag, defs)
                for v in values:
                    variants[json.dumps(v)] = shape
            return ("tagged", tag, tuple(sorted(variants.items())))
        # An untagged union of scalar alternates — normalize each branch.
        return ("union", frozenset(_norm(b, defs) for b in non_null))

    # type: [T, "null"] (schemars' Option form)
    t = node.get("type")
    if isinstance(t, list):
        non_null_t = [x for x in t if x != "null"]
        if len(non_null_t) == 1 and len(t) == 2:
            return ("optional", _norm({**node, "type": non_null_t[0]}, defs, tag))
        return ("types", frozenset(non_null_t))

    if t == "object" or "properties" in node:
        props = node.get("properties")
        if props is not None:
            required = frozenset(node.get("required", []))
            fields = tuple(sorted((k, _norm(v, defs), k in required) for k, v in props.items()))
            return ("object", fields)
        ap = node.get("additionalProperties")
        if isinstance(ap, dict):
            return ("map", _norm(ap, defs))
        return ("object", ())

    if t == "array":
        items = node.get("items")
        return ("list", _norm(items, defs) if isinstance(items, dict) else ("any",))

    if t in ("integer", "number", "string", "boolean"):
        return ("scalar", t)
    if t == "null":
        return ("scalar", "null")
    return ("any",)


def _normalize_root(schema: dict, tag: str | None) -> object:
    defs = schema.get("$defs", schema.get("definitions", {}))
    return _norm(schema, defs, tag)


# ── the gates ────────────────────────────────────────────────────────────────


class TestCoverage:
    def test_every_python_model_has_a_rust_counterpart(self) -> None:
        rust = set(shrike_native.schema_catalog())
        python = set(_python_models())
        missing = python - rust
        assert not missing, f"schemas.py models with no Rust counterpart: {sorted(missing)}"

    def test_every_rust_type_has_a_python_counterpart(self) -> None:
        rust = set(shrike_native.schema_catalog())
        python = set(_python_models())
        extra = rust - python
        assert not extra, f"Rust types with no schemas.py counterpart: {sorted(extra)}"


class TestStructuralEquivalence:
    @pytest.mark.parametrize("name", sorted(_python_models()))
    def test_shapes_match(self, name: str) -> None:
        adapter = _python_models()[name]
        py_schema = adapter.json_schema()
        rust_schema = json.loads(shrike_native.schema_catalog()[name])
        tag = UNIONS.get(name)
        py_shape = _normalize_root(py_schema, tag)
        rust_shape = _normalize_root(rust_schema, tag)
        assert py_shape == rust_shape, (
            f"{name} diverges.\n--- pydantic ---\n{py_shape}\n--- rust ---\n{rust_shape}"
        )


# ── instance round-trips (wire parity in practice) ───────────────────────────

ROUNDTRIP_CASES: list[tuple[str, dict]] = [
    (
        "Note",
        {
            "id": 5,
            "note_type": "Basic",
            "deck": "D",
            "tags": ["a"],
            "modified": "2026-06-10",
            "content": None,
        },
    ),
    (
        "SearchMatch",
        {
            "id": 5,
            "note_type": "Basic",
            "deck": "D",
            "tags": [],
            "modified": "m",
            "content": {"Front": "Q"},
            "score": 0.75,
            "substring": {
                "matched_fields": ["Front"],
                "snippet": "…Q…",
                "source": "field",
                "ref": None,
            },
            "fuzzy": None,
            "provenance": [{"signal": "text", "rank": 1}],
        },
    ),
    (
        "UpsertNoteResult",
        {"status": "created", "id": 9, "neighbors": [], "neighbors_unavailable": False},
    ),
    (
        "UpsertNoteResult",
        {
            "status": "updated",
            "id": 9,
            "neighbors": [{"id": 1, "score": 0.9, "tags": []}],
            "neighbors_unavailable": False,
        },
    ),
    ("UpsertNoteResult", {"status": "ok", "index": 0, "action": "create"}),
    ("UpsertNoteResult", {"status": "skipped", "index": 1, "reason": "duplicate"}),
    ("UpsertNoteResult", {"status": "error", "index": 2, "error": "boom", "reason": "empty"}),
    ("UpsertNoteResult", {"status": "error", "index": 2, "error": "boom", "reason": None}),
    ("FieldOp", {"op": "add", "name": "Extra", "position": None}),
    ("FieldOp", {"op": "reposition", "name": "Front", "position": 1}),
    ("TemplateOp", {"op": "rename", "name": "Card 1", "new_name": "Recall"}),
    (
        "StoreMediaResult",
        {
            "status": "stored",
            "index": 0,
            "filename": "a.png",
            "mime": "image/png",
            "size_bytes": 12,
            "deduped": False,
        },
    ),
    ("MediaFetchResult", {"status": "missing", "filename": "gone.png"}),
    (
        "EmbeddingStatus",
        {
            "state": "running",
            "available": True,
            "pid": 4,
            "url": "http://x",
            "model": "m",
            "provider": None,
            "batch_safe": True,
            "batch": "batched",
            "modalities": ["text", "image"],
        },
    ),
    ("EmbeddingStatus", {"state": "not_configured", "available": False}),
    (
        "IndexStatus",
        {
            "state": "building",
            "available": False,
            "size": 0,
            "ndim": None,
            "path": None,
            "col_mod": None,
            "model_id": None,
            "activation": None,
            "progress": {"indexed": 1, "total": 5},
        },
    ),
    (
        "IndexStatus",
        {
            "state": "ready",
            "available": True,
            "size": 3,
            "ndim": 8,
            "path": "/p",
            "col_mod": 7,
            "model_id": "m",
            "activation": {"image": {"n": 4.0, "mean": 0.2, "std": 0.1}},
        },
    ),
    ("IndexSaveResponse", {"status": "empty"}),
    ("EmbeddingStopResponse", {"status": "not_running"}),
    ("ReloadResponse", {"status": "reloaded", "col_mod": 12, "rebuilding": True}),
    ("StopResponse", {"stopped": True, "pid": 7, "forced": False}),
    ("StopResponse", {"stopped": False, "reason": "not running"}),
    (
        "CollectionPruneResponse",
        {
            "dry_run": True,
            "unused_tags": {"removed": 1, "tags": ["x"]},
            "empty_notes": None,
            "empty_cards": None,
            "unused_media": None,
        },
    ),
    # The actions-over-HTTP error envelope (#505): one case per code, so the
    # snake_case wire values agree on both sides in practice.
    ("ActionError", {"code": "input_error", "message": "bad query"}),
    ("ActionError", {"code": "collection_busy", "message": "in use"}),
    ("ActionError", {"code": "unknown_action", "message": "no such action"}),
    ("ActionError", {"code": "internal_error", "message": "the server failed"}),
    # The collection/profile registry enumeration (#66).
    ("ProfileEntry", {"name": "work", "path": "/decks/work.anki2", "is_default": True}),
    (
        "ListProfilesResponse",
        {
            "profiles": [
                {"name": "work", "path": "/w.anki2", "is_default": False},
                {"name": "home", "path": "/h.anki2", "is_default": True},
            ],
            "default": "home",
        },
    ),
    ("ListProfilesResponse", {"profiles": [], "default": None}),
]


class TestInstanceRoundTrips:
    @pytest.mark.parametrize(
        ("name", "payload"),
        ROUNDTRIP_CASES,
        ids=[f"{n}-{i}" for i, (n, _) in enumerate(ROUNDTRIP_CASES)],
    )
    def test_pydantic_wire_rides_through_rust(self, name: str, payload: dict) -> None:
        # The payload is what Pydantic would emit (model_dump mode="json").
        adapter = _python_models()[name]
        validated = adapter.validate_python(payload)
        py_wire = adapter.dump_python(validated, mode="json")
        # Rust parses the Pydantic wire and re-emits it unchanged.
        rust_wire = json.loads(shrike_native.schema_roundtrip(name, json.dumps(py_wire)))
        assert rust_wire == py_wire
        # And the Rust emission validates back into the Pydantic model.
        adapter.validate_python(rust_wire)

    def test_unknown_type_is_an_input_error(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip("NoSuchType", "{}")

    def test_bad_payload_is_an_input_error(self) -> None:
        with pytest.raises(shrike_native.NativeInputError):
            shrike_native.schema_roundtrip("Note", '{"id": "not-an-int"}')
