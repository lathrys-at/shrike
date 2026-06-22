"""The ``unsafe_assume_normalized`` engine-config opt-out, end to end over a
real kernel.

#971 normalizes every embedder's output at the kernel boundary so the index's
inner-product metric equals cosine. A backend that GUARANTEES unit output opts
out via ``EmbedderBackend.assume_normalized``, and the harness threads that into
the kernel attach (``unsafe_assume_normalized``).

The load-bearing invariant is stored↔query CONSISTENCY. There is ONE embedder on
the live path — the kernel's ``EmbedService.embedder`` serves BOTH stored embeds
AND in-kernel query embedding (``kernel.search``). The attach flag skips the
boundary wrap on that single embedder, so stored and query vectors take the same
(raw) path by construction. This test pins it end to end: a deliberately NON-unit
backend with the flag set still retrieves its note through ``kernel.search`` —
which only holds if the query embedded in the same raw space as the stored
vectors. The wrap-skip itself is also pinned directly in the Rust kernel test
``assume_normalized_skips_the_boundary_wrap``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

shrike_native = pytest.importorskip("shrike_native")

from shrike.harness.derived import DerivedTextStore, NativeDerivedEngine  # noqa: E402
from shrike.harness.engines.embedding.runtime import EmbeddingRuntime  # noqa: E402
from shrike.harness.harness import Harness, _assume_normalized  # noqa: E402


class _NonUnitBackend:
    """A backend whose vectors are NOT unit length, advertising whether it
    guarantees unit output.

    Each text embeds to a one-hot vector scaled by ``scale`` on an 8-dim axis
    chosen deterministically from the first token, so a query and a note sharing
    their first token land on the same axis — and the non-unit magnitude makes a
    stray normalize observable."""

    def __init__(self, *, assume_normalized: bool, scale: float = 5.0) -> None:
        self._assume = assume_normalized
        self._scale = scale

    @property
    def assume_normalized(self) -> bool:
        return self._assume

    @property
    def running(self) -> bool:
        return True

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            tok = t.split()[0] if t.split() else ""
            axis = sum(tok.encode()) % 8
            v = [0.0] * 8
            v[axis] = self._scale
            out.append(v)
        return out

    def embedding_dim(self) -> int:
        return 8

    def model_fingerprint(self) -> str:
        return f"nonunit:{'norm' if self._assume else 'raw'}:v1"

    def health(self) -> dict[str, object]:
        return {"available": True}


class _KeyedBackend:
    """A backend mapping a text's first token to an explicit (non-unit,
    non-collinear, different-magnitude) vector — the shape that makes a
    STORED-side normalize observable in the ranking.

    One-hot/single-note setups can't witness a stored↔query desync: a positive
    query scale never changes argmax, and a lone note on a unique axis is the
    nearest neighbour regardless. Different-magnitude, non-collinear stored
    vectors do: with raw inner product the higher-magnitude note wins, but if the
    store path normalized while the query did not, the more-aligned (unit) note
    wins instead — the winner flips."""

    def __init__(self, *, assume_normalized: bool, vectors: dict[str, list[float]]) -> None:
        self._assume = assume_normalized
        self._vectors = vectors

    @property
    def assume_normalized(self) -> bool:
        return self._assume

    @property
    def running(self) -> bool:
        return True

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Map by whichever key token appears in the text — the stored note text is
        # the rendered note (field labels + values), so position-based extraction
        # is unreliable; a substring probe is robust to that framing.
        out: list[list[float]] = []
        for t in texts:
            base = next((v for key, v in self._vectors.items() if key in t), [0.0, 0.0])
            out.append([*base, *([0.0] * (8 - len(base)))])
        return out

    def embedding_dim(self) -> int:
        return 8

    def model_fingerprint(self) -> str:
        return f"keyed:{'norm' if self._assume else 'raw'}:v1"

    def health(self) -> dict[str, object]:
        return {"available": True}


async def _assemble(tmp_path) -> Harness:
    runtime = EmbeddingRuntime(model=None)
    derived = DerivedTextStore(
        path=tmp_path / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
    )
    return await Harness.assemble(
        collection_path=str(tmp_path / "collection.anki2"),
        cache_dir=str(tmp_path / "cache"),
        runtime=runtime,
        derived=derived,
        cooperative=False,
        hold_seconds=5.0,
        media_read=None,
        media_exists=None,
    )


def test_assume_normalized_helper_reads_the_backend_property() -> None:
    """The single source of truth for the opt-out, the predicate the attach path
    consults. A backend without the property is treated as non-unit."""
    assert _assume_normalized(_NonUnitBackend(assume_normalized=True)) is True
    assert _assume_normalized(_NonUnitBackend(assume_normalized=False)) is False
    assert _assume_normalized(SimpleNamespace()) is False


def test_stored_query_ranking_agree_under_the_flag(tmp_path) -> None:
    """The stored↔query consistency witness, end to end on the live path. Two
    notes with NON-collinear, different-magnitude raw vectors and a query whose
    correct (raw inner-product) winner is the higher-magnitude note:

        stored P = [10, 0]   stored Q = [3, 3]   query = [2, 2]
        raw IP:  q·P = 20  >  q·Q = 12     → P wins  (flag honored both sides)

    If the flag were dropped on the STORE path (vectors normalized) but not the
    query, P→[1,0] and Q→[.707,.707] give q·P = 2 < q·Q ≈ 2.83 → Q wins: the
    winner FLIPS. So P-outranks-Q can only hold if the kernel embedded the query
    in the SAME raw space the stored vectors live in — the consistency invariant.
    The kernel embeds the query in-core (``kernel.search``), and the query text
    shares no trigram with either note, so the semantic signal is the sole
    discriminator (no lexical tier muddies the order)."""

    async def flow() -> None:
        backend = _KeyedBackend(
            assume_normalized=True,
            vectors={"ppp": [10.0, 0.0], "qqq": [3.0, 3.0], "zzz": [2.0, 2.0]},
        )
        harness = await _assemble(tmp_path)
        await harness.boot(start_embedding=False)
        harness._attach(backend)  # type: ignore[arg-type]

        notes = await harness.wrapper.upsert_notes(
            [
                {
                    "note_type": "Basic",
                    "deck": "Default",
                    "fields": {"Front": "ppp aaa bbb", "Back": "x"},
                },
                {
                    "note_type": "Basic",
                    "deck": "Default",
                    "fields": {"Front": "qqq ccc ddd", "Back": "x"},
                },
            ]
        )
        p_id, q_id = notes[0]["id"], notes[1]["id"]
        await harness.kernel.reindex_if_needed()
        await harness.kernel.settle()

        # Query embeds raw to [2, 2]; its text shares no trigram with either note,
        # so only the semantic (raw-IP) order ranks P vs Q.
        hits = await harness.kernel.search("zzz eee fff", 5)
        ranked = [nid for nid, _dist, _snips in hits]
        assert p_id in ranked and q_id in ranked, "both notes retrieved"
        assert ranked.index(p_id) < ranked.index(q_id), (
            "the higher-magnitude note outranks under raw inner product — the "
            "kernel embedded the query in the same un-normalized space as the "
            "stored vectors (a store-side-only normalize would flip the winner)"
        )

        await harness.kernel.close()

    asyncio.run(flow())
