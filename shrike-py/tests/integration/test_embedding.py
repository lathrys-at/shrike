"""Integration tests for the embedding service.

These tests require llama-server on PATH and download a small (~20MB)
GGUF model on first run. They are skipped automatically when
llama-server is not available.

Audited for cost (#441): the read-only classes share the session
`collection_server` (no dedicated boot), each class is one test against one
response, and the orphan-reap test fakes the orphan — the reap LOGIC is pinned
by shrike-core/managed/shrike-llama-server's Rust tests; this lane proves only the wiring,
which needs one real boot, not two.
"""

from __future__ import annotations

import httpx
import pytest
import shrike_native

from tests.integration.conftest import requires_llama_server

pytestmark = [pytest.mark.integration, pytest.mark.embedding, requires_llama_server]


class TestEmbeddingHealth:
    """The /status embedding block is wired through from the live service."""

    def test_status_reports_embedding_fields(self, collection_server):
        # One /status fetch, every field of the embedding block asserted together —
        # these are all properties of the same response. `available is True` also
        # subsumes the old separate llama /health probe (availability requires it).
        status_url = collection_server.url.rsplit("/", 1)[0] + "/status"
        resp = httpx.get(status_url, timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert "embedding" in body
        emb = body["embedding"]
        assert emb["available"] is True
        assert isinstance(emb["pid"], int)
        assert emb["url"] == f"http://127.0.0.1:{collection_server.embedding_port}"
        assert emb["model"].endswith(".gguf")


class TestEmbeddings:
    """The pinned llama binary + model actually embed (a canary for the
    pinned externals, not for Shrike code). One batch covers shape, dim
    consistency, float types, and semantic ordering."""

    def test_batch_embeds_with_consistent_dims_and_semantics(self, collection_server):
        texts = [
            "the weather is sunny today",
            "it is a bright and sunny day",
            "quantum chromodynamics describes strong nuclear force",
        ]
        resp = httpx.post(
            f"{collection_server.embedding_url}/v1/embeddings",
            json={"input": texts},
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        vecs = [item["embedding"] for item in data["data"]]
        dims = len(vecs[0])
        assert dims > 0
        assert all(len(v) == dims for v in vecs)
        assert all(isinstance(x, float) for v in vecs for x in v)

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        assert cosine_sim(vecs[0], vecs[1]) > cosine_sim(vecs[0], vecs[2])


class TestEmbeddingServiceViaShrike:
    """LlamaServerBackend.embed_texts() through a real shrike_native
    RemoteEmbedder against live llama-server — the wire path the unit
    tests stub."""

    def test_embed_method_types_and_dims(self, collection_server):
        from shrike.harness.engines.embedding.runtime import EmbeddingService

        svc = EmbeddingService.__new__(EmbeddingService)
        svc._base_url = collection_server.embedding_url
        svc._model = "test"
        svc._model_name = None
        svc._manager = type("FakeMgr", (), {"running": lambda self: True, "pid": lambda self: 1})()
        svc._safe_batch = 16  # bypassing start()/__init__, simulate a batch-safe probe
        svc._batch_cap = None
        # The native client pair __init__/start() would have built (#342 P4):
        # the unpinned fallback drives REAL requests against live llama-server.
        svc._client = shrike_native.RemoteEmbedder(collection_server.embedding_url)
        svc._remote = None

        r1 = svc.embed_texts(["a single sentence"])
        r2 = svc.embed_texts(["another sentence", "and one more", "three total"])
        assert len(r1) == 1 and len(r2) == 3
        assert all(isinstance(v, list) for v in r1 + r2)
        assert all(isinstance(v[0], float) for v in r1 + r2)
        assert len(r1[0]) == len(r2[0]) == len(r2[1])


class TestOrphanReaping:
    """A llama-server orphaned by an unclean shutdown is reaped on next start.

    The reap decision logic (recycled-PID guard, bind/held probes, SIGTERM →
    SIGKILL escalation, pid-file lifecycle) is pinned by the Rust tests in
    shrike-core/managed/shrike-llama-server. What only this lane can prove is the WIRING:
    pid_file threads into the native manager, the reap runs inside start()
    before binding, and the new PID is written. That needs one real boot; the
    orphan itself only needs to be a process that holds the port with its PID
    in the pid file — the exact dual signal the reap checks (it neither knows
    nor cares that a real orphan would be llama-server).
    """

    def test_start_reaps_orphaned_llama_server(self, embedding_model, tmp_path):
        import socket
        import subprocess
        import sys
        import time

        from shrike.harness.engines.embedding.runtime import EmbeddingService

        # Pick a free port for the stub orphan and the real service to contend over.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        pid_file = tmp_path / "embedding.pid"

        # The "orphan": a stub that listens on the port, PID recorded — as a
        # SIGKILLed Shrike would have left llama-server.
        orphan = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import socket, time\n"
                "s = socket.socket()\n"
                f"s.bind(('127.0.0.1', {port}))\n"
                "s.listen(1)\n"
                "time.sleep(120)\n",
            ],
        )
        pid_file.write_text(str(orphan.pid))
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with socket.socket() as probe:
                    probe.settimeout(0.2)
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.05)
            else:
                pytest.fail("stub orphan never bound the port")

            # The real service on the SAME port + pid_file must reap the orphan
            # and come up healthy — a port collision would otherwise fail the bind.
            b = EmbeddingService(
                model=str(embedding_model), port=port, log_dir=tmp_path / "b", pid_file=pid_file
            )
            try:
                b.start()
                assert b.running
                assert b.health()["available"] is True
                assert b._manager.pid() != orphan.pid
                assert pid_file.read_text() == str(b._manager.pid())
                # The reap terminated the stub.
                assert orphan.wait(timeout=5) is not None
            finally:
                b.stop()
        finally:
            if orphan.poll() is None:
                orphan.kill()
                orphan.wait()


class TestAttachMode:
    """managed.llama_server.manage: attach (#498): a daemon that embeds via a
    llama-server another process owns — here, the collection_server's child.
    The attaching server never spawns, reaps, or stops it."""

    def test_attach_serves_embeddings_from_a_server_it_does_not_own(
        self, collection_server, server_factory, tmp_path_factory
    ):
        cfg = tmp_path_factory.mktemp("attach-config") / "config.yml"
        cfg.write_text(
            "embedders:\n"
            "  - modalities: [text]\n"
            "    runtime: remote\n"
            "managed:\n"
            "  llama_server:\n"
            "    manage: attach\n"
            f"    port: {collection_server.embedding_port}\n"
        )
        # Generous boot deadline: the attach boot embeds against the upstream
        # (connectivity proof + batch probe) before serving — fast on a warm
        # server, but the factory's model-keyed heuristic can't see it.
        attached = server_factory(
            "attach-client", extra_args=["--config", str(cfg)], boot_timeout=120.0
        )

        status = httpx.get(attached.url.rsplit("/", 1)[0] + "/status", timeout=5.0).json()
        emb = status["embedding"]
        assert emb["available"] is True
        # Attached, not owned: it points at the upstream's port and reports no
        # pid (there is no child process to manage).
        assert emb["url"] == f"http://127.0.0.1:{collection_server.embedding_port}"
        assert emb.get("pid") is None
        # The cross-modal coverage matrix golden for the attach shape (#498/#235):
        # text-only space → text→text native, every other pair unavailable.
        assert emb["modalities"] == ["text"]
        assert status["coverage"] == {
            "text": {"text": "native", "image": "unavailable", "audio": "unavailable"},
            "image": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
            "audio": {"text": "unavailable", "image": "unavailable", "audio": "unavailable"},
        }

        # And the upstream is untouched — still serving its own daemon.
        upstream = httpx.get(
            collection_server.url.rsplit("/", 1)[0] + "/status", timeout=5.0
        ).json()
        assert upstream["embedding"]["available"] is True


class TestSharedRouterWiring:
    """The shared managed router (#567): ONE LlamaServerManager.router serving a
    directory of GGUFs, with N model-pinned RemoteEmbedder clients routing by
    the request `model` field.

    The router/single ModelSpec command construction is pinned by the Rust
    tests; per-request model pinning by the shrike-embed-remote tests; the
    profiles collapse + the harness owner-only stop by the unit/native suites.
    What only this lane proves is the live WIRING: one spawn (not N) on one
    port serving two distinct model names without a collision, both pinned
    clients embedding, owner-only stop. Without Wave-2 model materialization we
    have one real GGUF — so we serve it under TWO filenames in a temp
    models_dir; both spaces embed identically, which is fine (the point is the
    one-server / two-pinned-clients plumbing, not two distinct models)."""

    def test_one_router_serves_two_pinned_clients_on_one_port(self, embedding_model, tmp_path):
        import shutil
        import socket

        # A models_dir with the single test GGUF under two distinct names — the
        # router lists both; each pinned client routes to its own.
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for name in ("text-a.gguf", "text-b.gguf"):
            shutil.copy(embedding_model, models_dir / name)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        mgr = shrike_native.LlamaServerManager.router(
            str(models_dir),
            host="127.0.0.1",
            port=port,
            log_dir=str(tmp_path / "log"),
            pid_file=str(tmp_path / "embedding.pid"),
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            # ONE spawn for the whole directory; /health is 200 before any model
            # lazy-loads, so this returns quickly.
            mgr.start()
            assert mgr.running()
            first_pid = mgr.pid()

            # The router's /v1/models lists the served model names — pin each
            # client to one so the test is robust to the exact alias convention.
            models = httpx.get(f"{base_url}/v1/models", timeout=30.0).json()["data"]
            served = [m["id"] for m in models]
            assert len(served) >= 2, f"router should list both GGUFs: {served}"

            # Two model-pinned clients against the ONE endpoint — both embed,
            # proving request-`model`-field routing works over a shared server.
            dims = []
            for model_name in served[:2]:
                client = shrike_native.RemoteEmbedder(base_url, model=model_name)
                vectors = client.embed_chunk(["hello router"])
                assert vectors and vectors[0], f"no vector for {model_name}"
                dims.append(len(vectors[0]))
            # Same underlying GGUF → same dim; the wiring is what we assert.
            assert dims[0] == dims[1]

            # Still ONE process — no second spawn, no port collision.
            assert mgr.running()
            assert mgr.pid() == first_pid
        finally:
            mgr.stop()
        # Owner stop terminated the single router.
        assert not mgr.running()
