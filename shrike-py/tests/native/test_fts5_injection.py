"""FTS5 injection safety through the live lexical search path."""

from __future__ import annotations

import asyncio

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness  # noqa: E402


class TestFts5InjectionThroughLiveSearch:
    """The live lexical search path quotes user text safely.

    The host Python ``_fts_quote`` is gone; the live ``search_notes`` path's
    sole FTS5 quoting is now the Rust ``fts_quote`` inside ``DerivedEngine``.
    A pure-FTS5-syntax query must be matched as literal text (so it finds
    nothing), never parsed as a MATCH expression — an unquoted ``"`` or operator
    would raise an FTS5 syntax error or, worse, run as syntax. This drives those
    queries through ``kernel.search`` (which embeds nothing without a backend, so
    it exercises the lexical ``exact``/``fuzzy`` signals where the quoting lives).
    """

    def test_pure_fts5_syntax_queries_are_quoted_not_parsed(self, tmp_path) -> None:
        async def flow() -> None:
            runtime = EmbeddingRuntime(model=None)
            derived = DerivedTextStore(
                path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
            )
            harness = await Harness.assemble(
                collection_path=str(tmp_path / "collection.anki2"),
                cache_dir=str(tmp_path / "cache"),
                runtime=runtime,
                derived=derived,
                cooperative=False,
                hold_seconds=5.0,
                media_read=None,
                media_exists=None,
            )
            await harness.boot(start_embedding=False)
            await harness.settle_background()
            await harness.wrapper.upsert_notes(
                [
                    {
                        "note_type": "Basic",
                        "deck": "Default",
                        "fields": {"Front": "mitochondria powerhouse", "Back": "b"},
                    }
                ]
            )
            # Land the field rows in the derived store (the boot build ran against
            # the empty collection), so the lexical signals have something to hit.
            _rows, dmod = await harness.kernel.rebuild_derived()
            harness.derived.settle_external_build(dmod)

            # Pure FTS5 syntax / operators / unbalanced quotes: each must be
            # treated as a literal substring (zero hits — none of it appears in
            # the note), and crucially must NOT raise out of the FTS5 MATCH.
            for q in (
                "AND AND (",
                'foo" OR "bar',
                "col:term",
                "a* NEAR/2 b",
                '"unterminated',
                "NOT (x)",
                "^anchor",
            ):
                hits = await harness.kernel.search(q, 5)
                assert hits == [], f"syntax query {q!r} must match literally (0 hits), got {hits}"

            # The control: a benign literal query still finds the note — so the
            # zero-hit assertions above can't be vacuously passing on a dead path.
            benign = await harness.kernel.search("mitochondria", 5)
            assert any(nid for nid, _dist, _snips in benign), (
                "a benign lexical query still retrieves the note through the live path"
            )

            await harness.close()

        asyncio.run(flow())
