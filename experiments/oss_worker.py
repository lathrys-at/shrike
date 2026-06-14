"""#650 angle-D oversubscription worker.

A faithful standalone replica of
`test_dedup_search_covers_ocr_vectors_max_over_items` (test_harness.py:661):
assemble a harness, attach a text-only fake embedder, upsert two notes (one
image-only via an <img>), attach a stub OCR recognizer, run the recognition
sweep fully, then search and assert the OCR vector is present (score >= 0.6).

Adds in-process diagnostics the test lacks: at the search point it reads the
note's text-modality vector COUNT via the kernel engine when such an
introspection exists, so a failure is classified absent-vs-unrecalled rather
than guessed.

Run standalone (one flow), or spawned M-wide by oss_runner.py to crank the
process oversubscription factor (each worker spins its own process-global
tokio runtime: worker_threads = available_parallelism, blocking pool too).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import shrike_native
from shrike.derived import DerivedTextStore, NativeDerivedEngine
from shrike.embedding import EmbeddingRuntime
from shrike.harness import Harness, KernelIndexView


class _TokenHash:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * 32
            for tok in t.lower().split():
                h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16)
                v[h % 32] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out

    def model_fingerprint(self) -> str:
        return "tok:v1"

    def embedding_dim(self) -> int:
        return 32


class _StubOcr:
    def recognize(self, items: list[bytes]) -> list[tuple[str, float, str]]:
        return [(data.decode(), 0.9, "") for data in items]

    def model_fingerprint(self) -> str:
        return "stub-ocr:v1"


def _ocr_vector_count(kernel, note_id: int) -> int | None:
    """Best-effort: how many text-modality vectors live under the note key."""
    for attr in ("modality_get", "engine_modality_get"):
        fn = getattr(kernel, attr, None)
        if fn is not None:
            try:
                got = fn("text", note_id)
                if got is None:
                    return 0
                return len(got)
            except Exception:
                pass
    return None


async def flow(tmp: Path) -> dict:
    media = {"cycle.png": b"oxaloacetate condenses with acetyl coa"}
    runtime = EmbeddingRuntime(model=None)
    derived = DerivedTextStore(
        path=tmp / "cache" / "shrike.db", engine_factory=NativeDerivedEngine
    )
    harness = await Harness.assemble(
        collection_path=str(tmp / "collection.anki2"),
        cache_dir=str(tmp / "cache"),
        runtime=runtime,
        derived=derived,
        cooperative=False,
        hold_seconds=5.0,
        media_read=media.get,
        media_exists=lambda name: name in media,
    )
    await harness.boot(start_embedding=False)
    backend = _TokenHash()
    harness.kernel.attach_embedder(shrike_native.PyEmbedder.capture(backend))
    await harness.kernel.reindex_if_needed()

    notes = await harness.wrapper.upsert_notes(
        [
            {
                "note_type": "Basic",
                "deck": "Default",
                "fields": {"Front": 'See diagram <img src="cycle.png">', "Back": "b"},
            },
            {
                "note_type": "Basic",
                "deck": "Default",
                "fields": {"Front": "qq unrelated filler card qq", "Back": "b"},
            },
        ]
    )
    diagram_id = notes[0]["id"]

    harness.attach_recognizer(_StubOcr())
    report = await harness.recognition_sweep(batch_size=4)

    view = KernelIndexView(harness.kernel, SimpleNamespace(backend=backend))  # type: ignore[arg-type]
    draft = "oxaloacetate condenses with acetyl coa today"
    hits = view.search([draft], top_k=5)[0]
    scores = {h["note_id"]: 1.0 - h["distance"] for h in hits}
    score = scores.get(diagram_id)
    vcount = _ocr_vector_count(harness.kernel, diagram_id)
    try:
        status = harness.kernel.index_status_json()
    except Exception:
        status = None

    await harness.close()
    return {
        "total_stored": report.get("total_stored"),
        "report": report,
        "in_scores": diagram_id in scores,
        "score": score,
        "vector_count": vcount,
        "index_status": status,
        "ok": score is not None and score >= 0.6,
    }


def main() -> int:
    # Replicate xdist: ONE long-lived interpreter runs the flow REPS times,
    # reusing the process-global tokio runtime + blocking pool across runs
    # (each fresh asyncio loop per run, like each test's asyncio.run).
    reps = int(os.environ.get("OSS_REPS", "1"))
    rc = 0
    for i in range(reps):
        tmp = Path(tempfile.mkdtemp(prefix=f"oss650-{os.getpid()}-{i}-"))
        try:
            res = asyncio.run(flow(tmp))
        except Exception as e:  # noqa: BLE001
            print(f"PID {os.getpid()} rep {i} EXC {type(e).__name__}: {e}", file=sys.stderr)
            rc = 3
            continue
        if not res["ok"]:
            print(
                f"PID {os.getpid()} rep {i} FAIL "
                f"stored={res['total_stored']} in_scores={res['in_scores']} "
                f"score={res['score']} vcount={res['vector_count']} "
                f"idx={res['index_status']} report={res['report']}"
            )
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
