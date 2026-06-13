"""Integration tests for semantic search, contextual upsert, and index management.

Requires llama-server on PATH and downloads a small GGUF model.
Tests exercise the full pipeline: embedding, indexing, search, neighbors.
"""

from __future__ import annotations

import time

import httpx
import pytest

from tests.integration.conftest import (
    CLIRunner,
    MCPClient,
    ServerInfo,
    requires_llama_server,
    wait_for_index_ready,
)

pytestmark = [pytest.mark.integration, pytest.mark.embedding, requires_llama_server]

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
    return CLIRunner(collection_server.url, str(semantic_cli_config))


def _base_url(server: ServerInfo) -> str:
    return server.url.rsplit("/", 1)[0]


_wait_for_index_ready = wait_for_index_ready


# ---------------------------------------------------------------------------
# Index build and status
# ---------------------------------------------------------------------------


class TestIndexBuild:
    """Build the index and verify it completes."""

    def test_rebuild_endpoint_and_becomes_ready(self, collection_server):
        # Trigger + wait as ONE test: a test that starts a rebuild must wait it
        # out before returning, or the running rebuild leaks into whichever test
        # samples /status next (#441 — the 'building' != 'ready' CI flake).
        base = _base_url(collection_server)
        resp = httpx.post(f"{base}/index/rebuild", timeout=30.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") in ("started", "already_building", "complete")
        idx = _wait_for_index_ready(collection_server)
        assert idx["size"] >= 50
        assert idx["available"] is True

    def test_status_endpoint_index_and_embedding_blocks(self, collection_server):
        # One settled /status response; both blocks are properties of it.
        _wait_for_index_ready(collection_server)
        base = _base_url(collection_server)
        body = httpx.get(f"{base}/status", timeout=5.0).json()
        idx = body["index"]
        assert idx["state"] == "ready"
        assert idx["size"] >= 50
        assert idx["ndim"] is not None
        assert idx["ndim"] > 0
        assert body["embedding"]["available"] is True
        # The cross-modal coverage matrix golden (#498/#235): a live text-only
        # backend makes text→text native; with no recognizers attached, every
        # media target is unavailable (no native space, no derived-text path).
        assert body["coverage"] == {
            "text": {"text": "native", "image": "unavailable", "audio": "unavailable"},
            "image": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
            "audio": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
        }
        assert body["embedding"]["modalities"] == ["text"]

    def test_save_endpoint(self, collection_server):
        _wait_for_index_ready(collection_server)
        base = _base_url(collection_server)
        resp = httpx.post(f"{base}/index/save", timeout=30.0)
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
        # would otherwise drop them and leave nothing to rank (#91).
        #
        # Assert calculus is present in the top-k tags rather than pinned to an
        # exact slot: "rate of change" legitimately pulls in mechanics notes
        # (velocity/acceleration), so which concept lands at slot 0 flips run to
        # run on this borderline model. Membership in the top-k is robust to
        # that perturbation while still failing if the ranking is actually broken.
        result = semantic_mcp(
            "search_notes",
            {"queries": ["derivative calculus rate of change"], "top_k": 5, "threshold": 0.0},
        )
        matches = result["results"][0]["matches"]
        assert len(matches) > 0
        top_tags = {t for m in matches for t in m.get("tags", [])}
        assert "calculus" in top_tags

    def test_id_based_search(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        listed = semantic_mcp("list_notes", {"tags": ["cell-biology"], "limit": 1})
        note_id = listed["notes"][0]["id"]

        result = semantic_mcp("search_notes", {"ids": [note_id], "top_k": 5})
        assert len(result["results"]) == 1
        matches = result["results"][0]["matches"]
        assert all(m["id"] != note_id for m in matches)

    def test_deck_filter(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "search_notes",
            {"queries": ["energy force motion"], "deck": "Physics", "top_k": 10},
        )
        matches = result["results"][0]["matches"]
        assert all(m["deck"] == "Physics" for m in matches)

    def test_tags_filter(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "search_notes",
            {"queries": ["chemical bonds"], "tags": ["organic"], "top_k": 5},
        )
        matches = result["results"][0]["matches"]
        assert all("organic" in m["tags"] for m in matches)

    def test_exclude_ids(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        first = semantic_mcp("search_notes", {"queries": ["DNA genetics"], "top_k": 3})
        first_ids = [m["id"] for m in first["results"][0]["matches"]]

        second = semantic_mcp(
            "search_notes",
            {"queries": ["DNA genetics"], "top_k": 3, "exclude_ids": first_ids},
        )
        second_ids = [m["id"] for m in second["results"][0]["matches"]]
        assert not set(first_ids) & set(second_ids)

    def test_multiple_queries(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "search_notes",
            {"queries": ["Newton's laws", "eigenvalue matrix"], "top_k": 3},
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
            "search_notes", {"queries": ["Big-O notation algorithm complexity"], "top_k": 3}
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
# Contextual upsert — neighbors
# ---------------------------------------------------------------------------


class TestUpsertNeighbors:
    """Verify that upsert returns similar-note neighbors when index is available."""

    def test_create_returns_neighbors(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Biology",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "What is cellular respiration?",
                            "Back": "The process cells use to convert glucose into ATP",
                        },
                        "tags": ["cell-biology"],
                    }
                ],
            },
        )
        r = result["results"][0]
        assert r["status"] == "created"
        assert "neighbors" in r
        neighbors = r["neighbors"]
        assert len(neighbors) > 0
        n = neighbors[0]
        assert "id" in n
        assert "score" in n
        assert "tags" in n
        assert "content" not in n
        assert 0 < n["score"] <= 1.0

        # Cleanup
        semantic_mcp("delete_notes", {"ids": [r["id"]]})

    def test_neighbors_are_topically_relevant(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Physics",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "What is angular momentum?",
                            "Back": "The rotational equivalent of linear momentum: L = Iw",
                        },
                        "tags": ["mechanics"],
                    }
                ],
            },
        )
        r = result["results"][0]
        neighbors = r.get("neighbors", [])
        if neighbors:
            neighbor_tags = {t for n in neighbors for t in n["tags"]}
            assert "mechanics" in neighbor_tags or "electromagnetism" in neighbor_tags

        semantic_mcp("delete_notes", {"ids": [r["id"]]})

    def test_threshold_filters_irrelevant(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Misc",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "What is the capital of Burkina Faso?",
                            "Back": "Ouagadougou",
                        },
                    }
                ],
                "neighbor_threshold": 0.95,
            },
        )
        r = result["results"][0]
        neighbors = r.get("neighbors", [])
        assert len(neighbors) == 0

        semantic_mcp("delete_notes", {"ids": [r["id"]]})

    def test_bulk_upsert_with_neighbors(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Biology",
                        "note_type": "Basic",
                        "fields": {"Front": f"Bulk bio question {i}", "Back": "About cells"},
                        "tags": ["cell-biology"],
                    }
                    for i in range(5)
                ],
                "top_k_neighbors": 3,
            },
        )
        created = [r for r in result["results"] if r["status"] == "created"]
        assert len(created) == 5
        for r in created:
            assert "neighbors" in r
            assert len(r["neighbors"]) <= 3

        ids = [r["id"] for r in created]
        semantic_mcp("delete_notes", {"ids": ids})


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

        before = semantic_mcp("search_notes", {"queries": ["unique xyzzy placeholder"], "top_k": 5})
        assert note_id in [m["id"] for m in before["results"][0]["matches"]]

        semantic_mcp("delete_notes", {"ids": [note_id]})

        after = semantic_mcp("search_notes", {"queries": ["unique xyzzy placeholder"], "top_k": 5})
        after_ids = [m["id"] for m in after["results"][0]["matches"]]
        assert note_id not in after_ids


# NOTE: the old TestEmptyBootIndexing (its own server boot, ~5-8s) was deleted
# in the #441 audit: the shared collection_server fixture IS that contract at
# 50-note scale — it boots against an empty collection with NO explicit rebuild
# (#148 materializes an empty ready index at boot), seeds via incremental
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
        data = semantic_runner.json(["index", "status"])
        assert data["state"] == "ready"
        assert data["size"] >= 50
        assert data["ndim"] is not None
        result = semantic_runner.invoke(["index", "status"])
        assert result.exit_code == 0
        assert "Index:" in result.output
        assert "ready" in result.output.lower()

    def test_index_rebuild_background(self, semantic_runner, collection_server):
        # ONE rebuild via the CLI, json mode (the richer assertion), waited out
        # before returning (#441 — rebuild leaks were the CI race). The pretty
        # variant added only exit-code/format checks, covered by status above.
        data = semantic_runner.json(["index", "rebuild", "--background"])
        assert "status" in data or "total" in data
        _wait_for_index_ready(collection_server)

    def test_index_save_json_and_pretty(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["index", "save"])
        assert data["status"] == "saved"
        assert data["size"] >= 50
        result = semantic_runner.invoke(["index", "save"])
        assert result.exit_code == 0
        assert "saved" in result.output.lower()


class TestEmbeddingCLI:
    """Test shrike embedding status via CLI."""

    def test_embedding_status_json_and_pretty(self, semantic_runner):
        data = semantic_runner.json(["embedding", "status"])
        assert data["available"] is True
        assert "url" in data
        result = semantic_runner.invoke(["embedding", "status"])
        assert result.exit_code == 0
        assert "Embedding:" in result.output
        assert "available" in result.output.lower()


class TestNoteSearchCLI:
    """Test shrike note search via CLI."""

    def test_search_pretty(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_runner.invoke(["note", "search", "mitochondria energy"])
        assert result.exit_code == 0
        assert "Results for:" in result.output

    def test_search_json(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["note", "search", "DNA genetics"])
        assert "results" in data
        assert len(data["results"]) > 0
        assert len(data["results"][0]["matches"]) > 0

    def test_search_similar_to(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        listed = semantic_runner.json(["note", "list", "--tags", "mechanics"])
        note_id = str(listed["notes"][0]["id"])

        data = semantic_runner.json(["note", "search", "--similar-to", note_id])
        assert len(data["results"]) > 0
        result_ids = [m["id"] for m in data["results"][0]["matches"]]
        assert int(note_id) not in result_ids

    def test_search_with_deck_filter(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["note", "search", "energy", "--deck", "Physics"])
        for m in data["results"][0]["matches"]:
            assert m["deck"] == "Physics"


# ---------------------------------------------------------------------------
# Server status CLI with index details
# ---------------------------------------------------------------------------


class TestServerStatusWithIndex:
    """Verify shrike server status shows index and embedding info.

    Index and embedding blocks are properties of the SAME response — one
    pretty invocation and one json invocation assert both (#441)."""

    def test_server_status_pretty_shows_index_and_embedding(
        self, semantic_runner, collection_server
    ):
        _wait_for_index_ready(collection_server)
        result = semantic_runner.invoke(["server", "status"])
        assert result.exit_code == 0
        assert "Index:" in result.output
        assert "ready" in result.output.lower()
        assert "Embedding:" in result.output
        assert "available" in result.output.lower()

    def test_server_status_json_includes_index_and_embedding(
        self, semantic_runner, collection_server
    ):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["server", "status"])
        assert data["index"]["state"] == "ready"
        assert data["index"]["size"] >= 50
        assert data["embedding"]["available"] is True


# ---------------------------------------------------------------------------
# Embedding service lifecycle — start/stop independently of the server
# ---------------------------------------------------------------------------


class TestEmbeddingLifecycle:
    """The full embedding lifecycle as ONE woven flow on ONE server (#441).

    Replaces the old TestEmbeddingLifecycle + TestNoEmbeddingBoot pair (two
    dedicated servers, four llama-server boots) with one server booted
    `--no-embedding` and two llama boots. The flow is a single ordered test:
    every step's contract depends on the state the previous step left, which
    is exactly why they were flaky as separate tests and cheap as one.
    """

    @pytest.fixture(scope="class")
    def lifecycle_server(self, server_factory, embedding_model) -> ServerInfo:
        """Booted with --no-embedding though a model IS configured — the cold
        no-auto-start contract is the fixture's own first state."""
        return server_factory(
            "emb-lifecycle",
            embedding_model=str(embedding_model),
            extra_args=["--no-embedding"],
        )

    def test_full_lifecycle_flow(
        self,
        lifecycle_server: ServerInfo,
        embedding_model,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        base = _base_url(lifecycle_server)
        mcp = MCPClient(lifecycle_server.url)

        # (a) Cold: --no-embedding suppressed auto-start despite the configured
        # model (the old TestNoEmbeddingBoot contract).
        status = httpx.get(f"{base}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is False
        assert status["index"]["state"] == "unavailable"

        # (b) Empty-body start uses the boot-configured model (llama boot #1).
        resp = httpx.post(f"{base}/embedding/start", json={}, timeout=120.0)
        assert resp.json()["status"] == "started"
        status = httpx.get(f"{base}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is True

        # (c) Seed two notes; the empty-at-boot index materialized at start, so
        # incremental upserts index them (#148) — searchable.
        mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Bio",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "What is a mitochondrion?",
                            "Back": "An organelle that produces ATP",
                        },
                        "tags": ["cell-biology"],
                    },
                    {
                        "deck": "Bio",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "What is ATP synthase?",
                            "Back": "Enzyme that synthesizes ATP from a proton gradient",
                        },
                        "tags": ["cell-biology"],
                    },
                ]
            },
        )
        _wait_for_index_ready(lifecycle_server)
        before = mcp("search_notes", {"queries": ["mitochondria ATP energy"], "top_k": 5})
        assert before["results"][0]["matches"]

        # (d) Stop: embedding unavailable, index unavailable, search degrades to
        # the exact tier (no index needed; "ATP" is literal in the seeds).
        resp = httpx.post(f"{base}/embedding/stop", timeout=30.0)
        assert resp.json()["status"] == "stopped"
        status = httpx.get(f"{base}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is False
        assert status["index"]["state"] == "unavailable"
        degraded = mcp("search_notes", {"queries": ["ATP"], "top_k": 5})
        matches = degraded["results"][0]["matches"]
        assert matches
        assert all(m["score"] is None for m in matches)
        assert matches[0]["substring"]["matched_fields"]
        assert "not running" in degraded["message"].lower()
        # Stopping again is a no-op.
        assert httpx.post(f"{base}/embedding/stop", timeout=10.0).json()["status"] == "not_running"

        # (e) CLI start with explicit model + port (llama boot #2) — the CLI
        # wiring half of the old test_cli_stop_and_start.
        cfg_dir = tmp_path_factory.mktemp("emb-lifecycle-cli")
        cfg = cfg_dir / "config.yml"
        cfg.write_text(
            f"server:\n  host: 127.0.0.1\n  port: {lifecycle_server.port}\n"
            f"collection: {lifecycle_server.collection_path}\n"
            f"logging:\n  dir: {lifecycle_server.log_dir}\n"
        )
        runner = CLIRunner(lifecycle_server.url, str(cfg))
        started = runner.invoke(
            [
                "embedding",
                "start",
                "--embedding-model",
                str(embedding_model),
                "--embedding-port",
                str(lifecycle_server.embedding_port),
                "--background",
            ]
        )
        assert started.exit_code == 0, started.output
        _wait_for_index_ready(lifecycle_server)
        after = mcp("search_notes", {"queries": ["mitochondria ATP energy"], "top_k": 5})
        assert after["results"][0]["matches"]

        # (f) Idempotent start while running.
        resp = httpx.post(f"{base}/embedding/start", json={}, timeout=30.0)
        assert resp.json()["status"] == "already_running"

        # (g) The fingerprint came from llama-server's /v1/models meta block,
        # not the file-size fallback.
        idx = httpx.get(f"{base}/status", timeout=5.0).json()["index"]
        assert idx.get("model_id", "").startswith("meta:")

        # (h) CLI stop — the other half of the CLI wiring.
        stopped = runner.invoke(["embedding", "stop"])
        assert stopped.exit_code == 0, stopped.output
        assert "stopped" in stopped.output.lower()
        status = httpx.get(f"{base}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is False


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
            # The centroid refresh runs off the upsert tail since #445 —
            # retry briefly until the new member's tag state lands.
            deadline = time.monotonic() + 10
            while True:
                result = semantic_mcp(
                    "search_notes",
                    {"queries": ["mitochondria ATP production"], "top_k": 10},
                )
                matches = result["results"][0]["matches"]
                ids = [m["id"] for m in matches]
                if nid in ids or time.monotonic() > deadline:
                    break
                time.sleep(0.2)
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
            {"queries": ["purple elephant umbrella dancing"], "top_k": 10},
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
        result = semantic_mcp("search_notes", {"queries": ["citric acid cycle"], "top_k": 5})
        matches = result["results"][0]["matches"]
        assert matches, "the literal hit must be found"
        assert matches[0].get("substring"), "rank 1 is the literal hit"

    def test_no_cross_tag_pollution(self, semantic_mcp, collection_server):
        _wait_for_index_ready(collection_server)
        # A clearly genetics-shaped query: the top hits must be genetics
        # cards, not members of OTHER activated tags riding the fusion.
        result = semantic_mcp(
            "search_notes",
            {"queries": ["enzyme that transcribes DNA into RNA"], "top_k": 3},
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
            result = semantic_mcp(
                "search_notes", {"queries": ["volcanic eruption geology"], "top_k": 10}
            )
            for group in result["results"]:
                for m in group["matches"]:
                    signals = {p["signal"] for p in m.get("provenance", [])}
                    assert "tag" not in signals, f"blocklisted tag contributed for note {m['id']}"
        finally:
            semantic_mcp("delete_notes", {"ids": ids})
