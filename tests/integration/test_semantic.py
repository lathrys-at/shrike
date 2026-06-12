"""Integration tests for semantic search, contextual upsert, and index management.

Requires llama-server on PATH and downloads a small GGUF model.
Tests exercise the full pipeline: embedding, indexing, search, neighbors.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from tests.integration.conftest import CLIRunner, MCPClient, ServerInfo, requires_llama_server

pytestmark = [pytest.mark.integration, pytest.mark.embedding, requires_llama_server]

# ---------------------------------------------------------------------------
# Test collection: 50 notes across 10 concepts, 5 per concept
# ---------------------------------------------------------------------------

CONCEPTS: list[dict[str, Any]] = [
    {
        "deck": "Biology",
        "tag": "cell-biology",
        "cards": [
            ("What is a mitochondrion?", "An organelle that produces ATP"),
            ("What is the inner mitochondrial membrane?", "Site of electron transport"),
            ("What is ATP synthase?", "Enzyme that synthesizes ATP using proton gradient"),
            ("What is the citric acid cycle?", "A metabolic pathway in the matrix"),
            ("What is oxidative phosphorylation?", "ATP production via electron transport"),
        ],
    },
    {
        "deck": "Biology",
        "tag": "genetics",
        "cards": [
            ("What is DNA?", "A double-stranded molecule encoding genetic information"),
            ("What is RNA polymerase?", "The enzyme that transcribes DNA into RNA"),
            ("What is a codon?", "A three-nucleotide sequence coding for an amino acid"),
            ("What is mRNA?", "Messenger RNA carries genetic code from DNA to ribosomes"),
            ("What is translation?", "The process of synthesizing protein from mRNA"),
        ],
    },
    {
        "deck": "Biology",
        "tag": "evolution",
        "cards": [
            ("What is natural selection?", "Differential survival and reproduction of organisms"),
            ("What is genetic drift?", "Random changes in allele frequency in a population"),
            ("What is speciation?", "The formation of new and distinct species"),
            ("What is fitness?", "An organism's ability to survive and reproduce"),
            ("What is adaptation?", "A trait that increases fitness in a given environment"),
        ],
    },
    {
        "deck": "Chemistry",
        "tag": "organic",
        "cards": [
            ("What is a covalent bond?", "A chemical bond formed by sharing electron pairs"),
            ("What is an alkane?", "A saturated hydrocarbon with single bonds only"),
            ("What is a functional group?", "An atom or group giving a molecule its properties"),
            ("What is isomerism?", "Molecules with same formula but different structures"),
            ("What is chirality?", "A molecule that is non-superimposable on its mirror image"),
        ],
    },
    {
        "deck": "Chemistry",
        "tag": "thermodynamics",
        "cards": [
            ("What is enthalpy?", "The total heat content of a system at constant pressure"),
            ("What is entropy?", "A measure of disorder or randomness in a system"),
            ("What is Gibbs free energy?", "Energy available to do useful work: G = H - TS"),
            ("What is an exothermic reaction?", "A reaction that releases heat to surroundings"),
            ("What is equilibrium?", "When forward and reverse reaction rates are equal"),
        ],
    },
    {
        "deck": "Physics",
        "tag": "mechanics",
        "cards": [
            ("What is Newton's first law?", "An object at rest stays at rest unless acted on"),
            ("What is momentum?", "The product of an object's mass and velocity"),
            ("What is kinetic energy?", "Energy of motion: KE = 0.5 * m * v^2"),
            ("What is friction?", "A force opposing the relative motion of surfaces"),
            ("What is acceleration?", "The rate of change of velocity over time"),
        ],
    },
    {
        "deck": "Physics",
        "tag": "electromagnetism",
        "cards": [
            ("What is Coulomb's law?", "Force between charges is proportional to q1*q2/r^2"),
            ("What is an electric field?", "A region where a charge experiences a force"),
            ("What is magnetic flux?", "The total magnetic field passing through a surface"),
            ("What is Faraday's law?", "A changing magnetic flux induces an electromotive force"),
            ("What is capacitance?", "The ability to store electric charge: C = Q/V"),
        ],
    },
    {
        "deck": "Mathematics",
        "tag": "calculus",
        "cards": [
            ("What is a derivative?", "The instantaneous rate of change of a function"),
            ("What is an integral?", "The accumulation of quantities over an interval"),
            ("What is the chain rule?", "d/dx[f(g(x))] = f'(g(x)) * g'(x)"),
            ("What is a limit?", "The value a function approaches as input approaches a point"),
            ("What is the fundamental theorem?", "Integration and differentiation are inverses"),
        ],
    },
    {
        "deck": "Mathematics",
        "tag": "linear-algebra",
        "cards": [
            ("What is a matrix?", "A rectangular array of numbers arranged in rows and columns"),
            ("What is an eigenvalue?", "A scalar lambda where Av = lambda*v for some vector v"),
            ("What is a determinant?", "A scalar value computed from a square matrix"),
            ("What is linear independence?", "No vector is a combination of others"),
            ("What is a vector space?", "A set closed under addition and scaling"),
        ],
    },
    {
        "deck": "Computer Science",
        "tag": "algorithms",
        "cards": [
            ("What is Big-O notation?", "Describes the upper bound of an algorithm's growth rate"),
            ("What is a binary search?", "Searching a sorted array by halving the range"),
            ("What is a hash table?", "A structure mapping keys to values via hashing"),
            ("What is recursion?", "A function that calls itself to solve subproblems"),
            ("What is dynamic programming?", "Solving overlapping subproblems"),
        ],
    },
]


@pytest.fixture(scope="module")
def collection_server(server_factory, embedding_model):
    """Server with embedding + a pre-populated 50-note collection.

    Module-scoped so all test classes share one server and one llama-server
    process. Tests that create notes must clean up after themselves.
    """
    srv = server_factory("semantic", embedding_model=str(embedding_model))

    status_url = srv.url.rsplit("/", 1)[0] + "/status"
    resp = httpx.get(status_url, timeout=5.0)
    status = resp.json()
    if not status.get("embedding", {}).get("available"):
        pytest.skip("Embedding service not available")

    mcp = MCPClient(srv.url)
    all_notes = []
    for concept in CONCEPTS:
        for front, back in concept["cards"]:
            all_notes.append(
                {
                    "deck": concept["deck"],
                    "note_type": "Basic",
                    "fields": {"Front": front, "Back": back},
                    "tags": [concept["tag"]],
                }
            )
    result = mcp("upsert_notes", {"notes": all_notes})
    created = sum(1 for r in result["results"] if r["status"] == "created")
    assert created == 50, f"Expected 50 notes created, got {created}"

    # Build the index here so the fixture is self-sufficient: this server boots
    # against an empty collection, and an empty-at-boot server does NOT index
    # notes added by incremental upsert (#148 — the upsert path is gated on
    # index.available, which is False until the index is materialized). Without
    # this, the index only gets built as a side effect of TestIndexBuild running
    # first, so a single TestSearchNotes test run in isolation (-k) would hang the
    # full _wait_for_index_ready timeout. Remove this explicit rebuild once #148
    # lands (incremental upsert will materialize the index on its own).
    httpx.post(f"{_base_url(srv)}/index/rebuild", timeout=30.0)
    _wait_for_index_ready(srv)

    return srv


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


def _wait_for_index_ready(server: ServerInfo, timeout: float = 60.0) -> dict:
    """Poll /status until the index is ready."""
    base = _base_url(server)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = httpx.get(f"{base}/status", timeout=5.0)
        idx = resp.json().get("index", {})
        if idx.get("state") == "ready" and idx.get("size", 0) > 0:
            return idx
        time.sleep(0.05)
    raise TimeoutError("Index did not become ready")


# ---------------------------------------------------------------------------
# Index build and status
# ---------------------------------------------------------------------------


class TestIndexBuild:
    """Build the index and verify it completes."""

    def test_rebuild_endpoint(self, collection_server):
        base = _base_url(collection_server)
        resp = httpx.post(f"{base}/index/rebuild", timeout=30.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") in ("started", "already_building", "complete")

    def test_index_becomes_ready(self, collection_server):
        idx = _wait_for_index_ready(collection_server)
        assert idx["size"] >= 50

    def test_status_endpoint_includes_index(self, collection_server):
        base = _base_url(collection_server)
        resp = httpx.get(f"{base}/status", timeout=5.0)
        body = resp.json()
        assert "index" in body
        idx = body["index"]
        assert idx["state"] == "ready"
        assert idx["size"] >= 50
        assert idx["ndim"] is not None
        assert idx["ndim"] > 0

    def test_status_endpoint_includes_embedding(self, collection_server):
        base = _base_url(collection_server)
        resp = httpx.get(f"{base}/status", timeout=5.0)
        body = resp.json()
        assert "embedding" in body
        assert body["embedding"]["available"] is True

    def test_save_endpoint(self, collection_server):
        _wait_for_index_ready(collection_server)
        base = _base_url(collection_server)
        resp = httpx.post(f"{base}/index/save", timeout=30.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "saved"
        assert body["size"] >= 50
        # Built at boot, no edits since, so nothing was pending to flush.
        assert body["pending"] == 0


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

        before = semantic_mcp("search_notes", {"ids": [note_id], "top_k": 1})
        assert len(before["results"][0]["matches"]) >= 0

        semantic_mcp("delete_notes", {"ids": [note_id]})

        after = semantic_mcp("search_notes", {"queries": ["unique xyzzy placeholder"], "top_k": 5})
        after_ids = [m["id"] for m in after["results"][0]["matches"]]
        assert note_id not in after_ids


# ---------------------------------------------------------------------------
# Empty-at-boot indexing (#148)
# ---------------------------------------------------------------------------


class TestEmptyBootIndexing:
    """A server booted against an empty collection should still index notes
    added later in the same session (#148)."""

    def test_upsert_into_empty_boot_collection_is_indexed(
        self, server_factory, embedding_model
    ) -> None:
        # A dedicated server with its own empty collection (not the shared,
        # rebuilt collection_server). An empty collection's index is trivially
        # complete, so boot materializes an empty, ready index (#148).
        srv = server_factory("empty-boot-index", embedding_model=str(embedding_model))
        base = _base_url(srv)

        # The embedding service must be up before we upsert; otherwise a skip
        # would be for the wrong reason (no embedder) rather than the bug.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if httpx.get(f"{base}/status", timeout=5.0).json()["embedding"]["available"]:
                break
            time.sleep(0.05)
        else:
            pytest.skip("embedding service did not become available")

        mcp = MCPClient(srv.url)
        result = mcp(
            "upsert_notes",
            {
                "notes": [
                    {
                        "deck": "Bio",
                        "note_type": "Basic",
                        "fields": {
                            "Front": "What is a ribosome?",
                            "Back": "The cellular machine that synthesizes proteins",
                        },
                        "tags": ["cell-biology"],
                    }
                ]
            },
        )
        assert result["results"][0]["status"] == "created"

        # No explicit /index/rebuild: boot materialized an empty, ready index, so
        # `index.available` is True and the incremental upsert path indexed the
        # note (index.add() is self-sufficient — _ensure_index() sizes the USearch
        # index from the embedding dimension).
        idx = httpx.get(f"{base}/status", timeout=5.0).json()["index"]
        assert idx["available"] is True
        assert idx["size"] >= 1

        # And the note is actually semantically searchable in the same session.
        # threshold=0 so the assertion doesn't hinge on the small quantized test
        # model clearing the default 0.5 cutoff (see test_similar_concepts_rank_higher).
        search = mcp(
            "search_notes",
            {"queries": ["protein synthesis organelle"], "top_k": 5, "threshold": 0.0},
        )
        matches = search["results"][0]["matches"]
        assert any("cell-biology" in m.get("tags", []) for m in matches)


# ---------------------------------------------------------------------------
# CLI commands: index and embedding
# ---------------------------------------------------------------------------


class TestIndexCLI:
    """Test shrike index rebuild and shrike index status via CLI."""

    def test_index_status_json(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["index", "status"])
        assert data["state"] == "ready"
        assert data["size"] >= 50
        assert data["ndim"] is not None

    def test_index_status_pretty(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_runner.invoke(["index", "status"])
        assert result.exit_code == 0
        assert "Index:" in result.output
        assert "ready" in result.output.lower()

    def test_index_rebuild_background(self, semantic_runner, collection_server):
        result = semantic_runner.invoke(["index", "rebuild", "--background"])
        assert result.exit_code == 0
        _wait_for_index_ready(collection_server)

    def test_index_rebuild_json(self, semantic_runner, collection_server):
        data = semantic_runner.json(["index", "rebuild", "--background"])
        assert "status" in data or "total" in data

    def test_index_save_json(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["index", "save"])
        assert data["status"] == "saved"
        assert data["size"] >= 50

    def test_index_save_pretty(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_runner.invoke(["index", "save"])
        assert result.exit_code == 0
        assert "saved" in result.output.lower()


class TestEmbeddingCLI:
    """Test shrike embedding status via CLI."""

    def test_embedding_status_json(self, semantic_runner):
        data = semantic_runner.json(["embedding", "status"])
        assert data["available"] is True
        assert "url" in data

    def test_embedding_status_pretty(self, semantic_runner):
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
    """Verify shrike server status shows index and embedding info."""

    def test_server_status_shows_index(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        result = semantic_runner.invoke(["server", "status"])
        assert result.exit_code == 0
        assert "Index:" in result.output
        assert "ready" in result.output.lower()

    def test_server_status_shows_embedding(self, semantic_runner):
        result = semantic_runner.invoke(["server", "status"])
        assert result.exit_code == 0
        assert "Embedding:" in result.output
        assert "available" in result.output.lower()

    def test_server_status_json_includes_index(self, semantic_runner, collection_server):
        _wait_for_index_ready(collection_server)
        data = semantic_runner.json(["server", "status"])
        assert "index" in data
        assert data["index"]["state"] == "ready"
        assert data["index"]["size"] >= 50

    def test_server_status_json_includes_embedding(self, semantic_runner):
        data = semantic_runner.json(["server", "status"])
        assert "embedding" in data
        assert data["embedding"]["available"] is True


# ---------------------------------------------------------------------------
# Embedding service lifecycle — start/stop independently of the server
# ---------------------------------------------------------------------------


class TestEmbeddingLifecycle:
    """Stop and start the embedding service while the server keeps running."""

    @pytest.fixture(scope="class")
    def lifecycle_server(self, server_factory, embedding_model) -> ServerInfo:
        """A dedicated embedding-enabled server (mutated by these tests)."""
        srv = server_factory("emb-lifecycle", embedding_model=str(embedding_model))
        mcp = MCPClient(srv.url)
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
        # No explicit rebuild needed: an empty-at-boot server materializes a
        # ready index at boot (#148), so the upserts above index incrementally.
        _wait_for_index_ready(srv)
        return srv

    def test_stop_then_start_cycle(self, lifecycle_server: ServerInfo) -> None:
        base = _base_url(lifecycle_server)
        mcp = MCPClient(lifecycle_server.url)

        # Searchable to begin with.
        before = mcp("search_notes", {"queries": ["mitochondria ATP energy"], "top_k": 5})
        assert before["results"][0]["matches"]

        # Stop: embedding unavailable, index marked unavailable, search degrades.
        resp = httpx.post(f"{base}/embedding/stop", timeout=30.0)
        assert resp.json()["status"] == "stopped"
        status = httpx.get(f"{base}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is False
        assert status["index"]["state"] == "unavailable"

        # Semantic ranking is gone, but exact substring matching needs no index
        # and still works ("ATP" appears literally in the seeded notes).
        degraded = mcp("search_notes", {"queries": ["ATP"], "top_k": 5})
        matches = degraded["results"][0]["matches"]
        assert matches
        assert all(m["score"] is None for m in matches)
        assert matches[0]["substring"]["matched_fields"]
        assert "not running" in degraded["message"].lower()

        # Stopping again is a no-op.
        assert httpx.post(f"{base}/embedding/stop", timeout=10.0).json()["status"] == "not_running"

        # Start again (server reuses its own configured model — empty body).
        resp = httpx.post(f"{base}/embedding/start", json={}, timeout=120.0)
        assert resp.json()["status"] == "started"
        _wait_for_index_ready(lifecycle_server)

        after = mcp("search_notes", {"queries": ["mitochondria ATP energy"], "top_k": 5})
        assert after["results"][0]["matches"]

    def test_start_when_running_is_idempotent(self, lifecycle_server: ServerInfo) -> None:
        base = _base_url(lifecycle_server)
        httpx.post(f"{base}/embedding/start", json={}, timeout=120.0)
        _wait_for_index_ready(lifecycle_server)
        resp = httpx.post(f"{base}/embedding/start", json={}, timeout=30.0)
        assert resp.json()["status"] == "already_running"

    def test_status_exposes_meta_model_id(self, lifecycle_server: ServerInfo) -> None:
        base = _base_url(lifecycle_server)
        httpx.post(f"{base}/embedding/start", json={}, timeout=120.0)
        _wait_for_index_ready(lifecycle_server)
        idx = httpx.get(f"{base}/status", timeout=5.0).json()["index"]
        # Proves the fingerprint came from llama-server's /v1/models meta block,
        # not the file-size fallback.
        assert idx.get("model_id", "").startswith("meta:")

    def test_cli_stop_and_start(
        self,
        lifecycle_server: ServerInfo,
        embedding_model,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        base = _base_url(lifecycle_server)
        httpx.post(f"{base}/embedding/start", json={}, timeout=120.0)
        _wait_for_index_ready(lifecycle_server)

        cfg_dir = tmp_path_factory.mktemp("emb-lifecycle-cli")
        cfg = cfg_dir / "config.yml"
        cfg.write_text(
            f"server:\n  host: 127.0.0.1\n  port: {lifecycle_server.port}\n"
            f"collection: {lifecycle_server.collection_path}\n"
            f"logging:\n  dir: {lifecycle_server.log_dir}\n"
        )
        runner = CLIRunner(lifecycle_server.url, str(cfg))

        stopped = runner.invoke(["embedding", "stop"])
        assert stopped.exit_code == 0, stopped.output
        assert "stopped" in stopped.output.lower()

        # Pass explicit model + port so they match how this test server was launched.
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


class TestNoEmbeddingBoot:
    """A server can boot with embedding off though a model is configured."""

    @pytest.fixture(scope="class")
    def no_embedding_server(self, server_factory, embedding_model) -> ServerInfo:
        return server_factory(
            "no-embedding",
            embedding_model=str(embedding_model),
            extra_args=["--no-embedding"],
        )

    def test_boots_without_embedding(self, no_embedding_server: ServerInfo) -> None:
        status = httpx.get(f"{_base_url(no_embedding_server)}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is False
        assert status["index"]["state"] == "unavailable"

    def test_start_uses_configured_model(self, no_embedding_server: ServerInfo) -> None:
        base = _base_url(no_embedding_server)
        # Empty body: the server starts with the model it was configured with at
        # boot, even though --no-embedding skipped auto-start. The POST returns
        # only once llama-server is healthy, so embedding is available right away
        # (the collection is empty, so there's nothing to index).
        resp = httpx.post(f"{base}/embedding/start", json={}, timeout=120.0)
        assert resp.json()["status"] == "started"
        status = httpx.get(f"{base}/status", timeout=5.0).json()
        assert status["embedding"]["available"] is True


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
            result = semantic_mcp(
                "search_notes",
                {"queries": ["mitochondria ATP production"], "top_k": 10},
            )
            matches = result["results"][0]["matches"]
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
