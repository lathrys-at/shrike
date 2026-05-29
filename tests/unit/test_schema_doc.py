"""The committed docs/mcp-schema.json must match scripts/gen_schema.py output.

This is the drift guard for audit 7.6: the schema doc is generated from the
Pydantic models and tool signatures, never hand-edited. If this fails, run
``python scripts/gen_schema.py`` and commit the result.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN_SCRIPT = ROOT / "scripts" / "gen_schema.py"
SCHEMA_DOC = ROOT / "docs" / "mcp-schema.json"


def _load_generator():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("gen_schema", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_doc_matches_generator() -> None:
    gen = _load_generator()
    expected = gen.generate()
    actual = SCHEMA_DOC.read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/mcp-schema.json is stale. Regenerate it with: python scripts/gen_schema.py"
    )
