"""Integration tests for semantic search, contextual upsert, and index management.

Runs in-process on the ONNX MiniLM backend (the `collection_server` fixture) —
the semantic behaviour under test (ranking, filters, neighbours, index updates)
is backend-agnostic, so it needs an embedder, not specifically llama-server. The
out-of-process GGUF/llama-server lifecycle is proved separately in
test_embedding.py. Tests exercise the full pipeline: embedding, indexing, search.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import (
    CLIRunner,
    MCPClient,
    requires_onnxruntime,
    requires_shrike_native,
    search_until,
    wait_for_index_ready,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.embedding,
    requires_onnxruntime,
    requires_shrike_native,
]

# ---------------------------------------------------------------------------
# Test collection: 50 notes across 10 concepts, 5 per concept
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def semantic_mcp(collection_server) -> MCPClient:
    return MCPClient(collection_server.url)


@pytest.fixture(scope="module")
def semantic_cli_config(collection_server, tmp_path_factory):
    config_dir = tmp_path_factory.mktemp("semantic-cli-config")
    config_path = config_dir / "config.yml"
    config_path.write_text(
        f"server:\n"
        f"  host: 127.0.0.1\n"
        f"  port: {collection_server.port}\n"
        f"collection: {collection_server.collection_path}\n"
        f"logging:\n"
        f"  dir: {collection_server.log_dir}\n"
    )
    return config_path


@pytest.fixture(scope="module")
def semantic_runner(collection_server, semantic_cli_config) -> CLIRunner:
    return CLIRunner(
        collection_server.url, str(semantic_cli_config), state_dir=collection_server.state_dir
    )


_wait_for_index_ready = wait_for_index_ready


# ---------------------------------------------------------------------------
# Index build and status
# ---------------------------------------------------------------------------


class TestIndexBuild:
    """Build the index and verify it completes."""

    def test_rebuild_endpoint_and_becomes_ready(self, collection_server):
        # Trigger + wait as ONE test: a test that starts a rebuild must wait it
        # out before returning, or the running rebuild leaks into whichever test
        # samples /status next (the 'building' != 'ready' flake).
        resp = collection_server.control_request("POST", "/index/rebuild", timeout=30.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") in ("started", "already_building", "complete")
        idx = _wait_for_index_ready(collection_server)
        assert idx["size"] >= 50
        assert idx["available"] is True

    def test_status_endpoint_index_and_embedding_blocks(self, collection_server):
        # One settled /status response; both blocks are properties of it.
        _wait_for_index_ready(collection_server)
        body = collection_server.control_request("GET", "/status", timeout=5.0).json()
        idx = body["index"]
        assert idx["state"] == "ready"
        assert idx["size"] >= 50
        assert idx["ndim"] is not None
        assert idx["ndim"] > 0
        assert body["embedding"]["available"] is True
        # The cross-modal coverage matrix golden: a live text-only backend makes
        # text→text native; with no recognizers attached, every media target is
        # unavailable (no native space, no derived-text path).
        assert body["coverage"] == {
            "text": {"text": "native", "image": "unavailable", "audio": "unavailable"},
            "image": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
            "audio": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
        }
        assert body["embedding"]["modalities"] == ["text"]

    def test_save_endpoint(self, collection_server):
        _wait_for_index_ready(collection_server)
        resp = collection_server.control_request("POST", "/index/save", timeout=30.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "saved"
        assert body["size"] >= 50
        # NOTE: no `pending == 0` assert — whether a flush was pending depends
        # on which mutating tests ran before this one on the shared server.


# ---------------------------------------------------------------------------
# search_notes MCP tool
# ---------------------------------------------------------------------------


class TestSearchNotes:
    """Semantic search over the populated collection."""

    def test_text_query_returns_relevant_results(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp("search_notes", {"queries": ["mitochondria ATP energy"]})
        assert len(result["results"]) == 1
        matches = result["results"][0]["matches"]
        assert len(matches) > 0
        tags = {t for m in matches for t in m.get("tags", [])}
        assert "cell-biology" in tags

    def test_similar_concepts_rank_higher(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # This is a *ranking* check: calculus notes should be among the nearest
        # matches for a calculus query. Pass threshold=0 so it doesn't hinge on
        # the absolute score clearing the default 0.5 cutoff — with the small
        # quantized test model the right answers score just under 0.5, which
        # would otherwise drop them and leave nothing to rank.
        #
        # Assert calculus is present in the top-k tags rather than pinned to an
        # exact slot: "rate of change" legitimately pulls in mechanics notes
        # (velocity/acceleration), so which concept lands at slot 0 flips run to
        # run on this borderline model. Membership in the top-k is robust to
        # that perturbation while still failing if the ranking is actually broken.
        result = semantic_mcp(
            "search_notes",
            {"queries": ["derivative calculus rate of change"], "limit": 5, "threshold": 0.0},
        )
        matches = result["results"][0]["matches"]
        assert len(matches) > 0
        top_tags = {t for m in matches for t in m.get("tags", [])}
        assert "calculus" in top_tags

    def test_id_based_search(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        listed = semantic_mcp("list_notes", {"tags": ["cell-biology"], "limit": 1})
        note_id = listed["notes"][0]["id"]

        result = semantic_mcp("search_notes", {"ids": [note_id], "limit": 5})
        assert len(result["results"]) == 1
        matches = result["results"][0]["matches"]
        assert all(m["id"] != note_id for m in matches)

    def test_deck_filter(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "search_notes",
            {"queries": ["energy force motion"], "deck": "Physics", "limit": 10},
        )
        matches = result["results"][0]["matches"]
        assert all(m["deck"] == "Physics" for m in matches)

    def test_tags_filter(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "search_notes",
            {"queries": ["chemical bonds"], "tags": ["organic"], "limit": 5},
        )
        matches = result["results"][0]["matches"]
        assert all("organic" in m["tags"] for m in matches)

    def test_exclude_ids(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        first = semantic_mcp("search_notes", {"queries": ["DNA genetics"], "limit": 3})
        first_ids = [m["id"] for m in first["results"][0]["matches"]]

        second = semantic_mcp(
            "search_notes",
            {"queries": ["DNA genetics"], "limit": 3, "exclude_ids": first_ids},
        )
        second_ids = [m["id"] for m in second["results"][0]["matches"]]
        assert not set(first_ids) & set(second_ids)

    def test_multiple_queries(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "search_notes",
            {"queries": ["Newton's laws", "eigenvalue matrix"], "limit": 3},
        )
        assert len(result["results"]) == 2
        assert result["results"][0]["source"] == "Newton's laws"
        assert result["results"][1]["source"] == "eigenvalue matrix"

    def test_result_shape(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # Use a descriptive multi-word query: a bare single word like
        # "algorithm" embeds too thinly to clear the default 0.5 threshold
        # against this collection, leaving no matches to inspect.
        result = semantic_mcp(
            "search_notes", {"queries": ["Big-O notation algorithm complexity"], "limit": 3}
        )
        # Each query is matched semantically and by exact substring; inspect a
        # semantically-ranked hit (exact-only hits carry no score).
        scored = [m for m in result["results"][0]["matches"] if m.get("score") is not None]
        assert scored
        match = scored[0]
        for key in ("id", "score", "deck", "note_type", "tags", "content"):
            assert key in match
        assert 0 < match["score"] <= 1.0


# ---------------------------------------------------------------------------
# Delete + index updates
# ---------------------------------------------------------------------------


class TestDeleteIndexUpdate:
    """Verify deleting notes removes them from search results."""

    def test_deleted_note_not_in_search(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)

        created = semantic_mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Biology",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "Unique xyzzy placeholder question",
                            "Back": "Unique xyzzy placeholder answer",
                        },
                    }
                ]
            },
        )
        note_id = created["results"][0]["id"]

        # The upsert is write-only — the index reflects it after the ingest queue
        # drains — so retry until the note is searchable.
        before = search_until(
            semantic_mcp,
            ["unique xyzzy placeholder"],
            lambda m: note_id in [x["id"] for x in m],
            limit=5,
        )
        assert note_id in [m["id"] for m in before]

        semantic_mcp("delete_notes", {"ids": [note_id]})

        # The remove drains off the same queue — retry until it's gone.
        after = search_until(
            semantic_mcp,
            ["unique xyzzy placeholder"],
            lambda m: note_id not in [x["id"] for x in m],
            limit=5,
        )
        assert note_id not in [m["id"] for m in after]


# NOTE: the shared collection_server fixture IS the empty-boot-indexing contract
# at 50-note scale — it boots against an empty collection with NO explicit
# rebuild (an empty ready index materializes at boot), seeds via incremental
# upserts, and waits for size >= 50; every search test then proves
# searchability. test_rebuild_endpoint_and_becomes_ready asserts
# `available is True` explicitly.


# ---------------------------------------------------------------------------
# CLI commands: index and embedding
# ---------------------------------------------------------------------------


class TestIndexCLI:
    """Test shrike index rebuild and shrike index status via CLI."""

    def test_index_status_json_and_pretty(self, semantic_runner, collection_server):
        # Same response, two renderings — one settled index, two invocations.
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["server", "index", "status"])
        assert data["state"] == "ready"
        assert data["size"] >= 50
        assert data["ndim"] is not None
        # Per-modality breakdown: the breakdown carries a `text` sub-index whose
        # ndim mirrors the aggregate (the aggregate ndim IS the text modality's),
        # and the per-modality sizes sum to the aggregate.
        mods = {m["modality"]: m for m in data["modalities"]}
        assert "text" in mods
        assert mods["text"]["ndim"] == data["ndim"]
        assert sum(m["size"] for m in data["modalities"]) == data["size"]
        result = semantic_runner.invoke(["server", "index", "status"])
        assert result.exit_code == 0
        # The header is the section identity, `Status:` is its own line, and the
        # per-modality breakdown reads `Vectors (text)`.
        assert "Index" in result.output
        assert "Status:" in result.output
        assert "Vectors (text)" in result.output
        assert "ready" in result.output.lower()

    def test_index_rebuild_background(self, semantic_runner, collection_server):
        # ONE rebuild via the CLI, json mode (the richer assertion), waited out
        # before returning (rebuild leaks are a CI race). The pretty variant adds
        # only exit-code/format checks, covered by status above.
        data = semantic_runner.json(["server", "index", "rebuild", "--background"])
        assert "status" in data or "total" in data
        _wait_for_index_ready(collection_server)

    def test_index_save_json_and_pretty(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["server", "index", "save"])
        assert data["status"] == "saved"
        assert data["size"] >= 50
        result = semantic_runner.invoke(["server", "index", "save"])
        assert result.exit_code == 0
        assert "saved" in result.output.lower()


class TestEmbeddingCLI:
    """Test shrike server embedding status via CLI."""

    def test_embedding_status_json_and_pretty(self, semantic_runner):
        # Per-space: JSON is the per-space LIST (a one-element list on a
        # single-space server), each entry the same EmbeddingStatus shape.
        # This is the in-process ONNX backend, so it carries no `url`/`pid`
        # (those are the out-of-process llama service's fields, asserted in
        # test_embedding.py) — assert the shape it does report.
        data = semantic_runner.json(["server", "embedding", "status"])
        assert isinstance(data, list)
        assert data[0]["available"] is True
        assert data[0]["modalities"] == ["text"]
        result = semantic_runner.invoke(["server", "embedding", "status"])
        assert result.exit_code == 0
        # `Embedding [text]` header, `Status:` on its own line.
        assert "Embedding [text]" in result.output
        assert "Status:" in result.output
        assert "available" in result.output.lower()


class TestNoteSearchCLI:
    """Test shrike search via CLI."""

    def test_search_pretty(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_runner.invoke(["search", "mitochondria energy"])
        assert result.exit_code == 0
        assert "Results for:" in result.output

    def test_search_json(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["search", "DNA genetics"])
        assert "results" in data
        assert len(data["results"]) > 0
        assert len(data["results"][0]["matches"]) > 0

    def test_search_similar_to(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        listed = semantic_runner.json(["note", "list", "--tags", "mechanics"])
        note_id = str(listed["notes"][0]["id"])

        data = semantic_runner.json(["search", "--similar-to", note_id])
        assert len(data["results"]) > 0
        result_ids = [m["id"] for m in data["results"][0]["matches"]]
        assert int(note_id) not in result_ids

    def test_search_with_deck_filter(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["search", "energy", "--deck", "Physics"])
        for m in data["results"][0]["matches"]:
            assert m["deck"] == "Physics"


# ---------------------------------------------------------------------------
# Server status CLI with index details
# ---------------------------------------------------------------------------


class TestServerStatusWithIndex:
    """Verify shrike server status shows index and embedding info.

    Index and embedding blocks are properties of the SAME response — one
    pretty invocation and one json invocation assert both."""

    def test_server_status_pretty_shows_index_and_embedding(
        self, semantic_runner, collection_server
    ):
        _wait_for_index_ready(collection_server)
        result = semantic_runner.invoke(["server", "status"])
        assert result.exit_code == 0
        # Section headers are identities; `Status:` is its own line; the
        # Embedding header carries the modalities (`Embedding [text]`).
        assert "Index" in result.output
        assert "Embedding [text]" in result.output
        assert "Status:" in result.output
        assert "ready" in result.output.lower()
        assert "available" in result.output.lower()

    def test_server_status_json_includes_index_and_embedding(
        self, semantic_runner, collection_server
    ):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["server", "status"])
        assert data["index"]["state"] == "ready"
        assert data["index"]["size"] >= 50
        assert data["embedding"]["available"] is True


class TestSearchQuality:
    """Result-QUALITY pins across the full signal mix (real model): the tag
    centroid layer, the activation floor, the exact-match override, and
    cross-tag pollution — the places where a new signal could quietly distort
    rankings. Each test cleans up what it creates (tag centroids refresh on
    the upsert/delete tails)."""

    def test_tag_surfaces_off_topic_member_below_direct_hits(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # A cell-biology-tagged note whose own text says nothing about the
        # topic: only the tag can surface it for a mitochondria query.
        r = semantic_mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Biology",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "Remember to review the chapter twelve summary notes",
                            "Back": "Personal reminder",
                        },
                        "tags": ["cell-biology"],
                    }
                ]
            },
        )
        nid = r["results"][0]["id"]
        try:
            # The note embeds off the ingest queue and the centroid refresh runs
            # off the upsert tail — retry until the new member's tag state lands.
            matches = search_until(
                semantic_mcp,
                ["mitochondria ATP production"],
                lambda m: nid in [x["id"] for x in m],
            )
            ids = [m["id"] for m in matches]
            assert nid in ids, "the tag signal surfaces the off-topic member"
            # …but never ABOVE the directly-relevant notes: the top hit is
            # semantically (or literally) on-topic, not tag-boosted filler.
            top_signals = {p["signal"] for p in matches[0]["provenance"]}
            assert "text" in top_signals or "exact" in top_signals
            assert matches[0]["id"] != nid, "tag-only filler must not take rank 1"
        finally:
            semantic_mcp("delete_notes", {"ids": [nid]})

    def test_off_topic_query_activates_no_tags(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # Nothing in the corpus is about this; with real cosines the
        # activation floor must hold — no result may carry tag provenance.
        result = semantic_mcp(
            "search_notes",
            {"queries": ["purple elephant umbrella dancing"], "limit": 10},
        )
        for group in result["results"]:
            for m in group["matches"]:
                signals = {p["signal"] for p in m.get("provenance", [])}
                assert "tag" not in signals, (
                    f"off-topic query activated a tag for note {m['id']}: {signals}"
                )

    def test_exact_match_pins_above_tag_boosted_siblings(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # "citric acid cycle" is literal text of exactly one card; the
        # cell-biology tag plausibly activates and boosts its siblings — the
        # literal hit must still take rank 1 (the exact-match override).
        result = semantic_mcp("search_notes", {"queries": ["citric acid cycle"], "limit": 5})
        matches = result["results"][0]["matches"]
        assert matches, "the literal hit must be found"
        assert matches[0].get("substring"), "rank 1 is the literal hit"

    def test_no_cross_tag_pollution(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # A clearly genetics-shaped query: the top hits must be genetics
        # cards, not members of OTHER activated tags riding the fusion.
        result = semantic_mcp(
            "search_notes",
            {"queries": ["enzyme that transcribes DNA into RNA"], "limit": 3},
        )
        matches = result["results"][0]["matches"]
        assert matches
        assert "genetics" in matches[0].get("tags", []), (
            f"rank 1 should be a genetics card, got tags={matches[0].get('tags')}"
        )

    def test_blocklisted_tag_never_contributes(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # Notes tagged ONLY `leech` (hygiene-blocklisted) on a topic of their
        # own: a query for that topic must carry no tag provenance — the
        # blocklist keeps meta-tags out of the centroid space.
        r = semantic_mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Misc",
                        "note_type": "Basic",
                        "fields": {"Front": f"Volcanic eruption fact {i}", "Back": "Geology"},
                        "tags": ["leech"],
                    }
                    for i in range(3)
                ]
            },
        )
        ids = [item["id"] for item in r["results"]]
        try:
            # The upsert indexes off the async ingest drain — poll until the new
            # notes are searchable, so the negative assertion (no tag provenance)
            # can't pass vacuously against a not-yet-indexed write.
            matches = search_until(
                semantic_mcp,
                ["volcanic eruption geology"],
                lambda ms: any(m["id"] in ids for m in ms),
            )
            for m in matches:
                signals = {p["signal"] for p in m.get("provenance", [])}
                assert "tag" not in signals, f"blocklisted tag contributed for note {m['id']}"
        finally:
            semantic_mcp("delete_notes", {"ids": ids})
