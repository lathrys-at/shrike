"""#650 angle-D: explain the exact 0.154 failure score.

Shows that when the diagram note has ONLY its field-text vector indexed (OCR
vector absent), the dedup search scores it 0.154 (< 0.6) against the draft —
byte-identically reproducing the reported failure
(`assert 0.154... >= 0.6`). When the OCR vector is also present, the
max-over-items dedup lifts it to 0.913. So the failure IS "OCR vector absent,
field-text vector present", not a recall miss.

This pins the symptom to the missing 2nd text vector, with the precise score.
"""
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import shrike_native
from shrike.derived import DerivedTextStore, NativeDerivedEngine
from shrike.embedding import EmbeddingRuntime
from shrike.harness import Harness, KernelIndexView

import sys
sys.path.insert(0, str(Path(__file__).parent))
from oss_worker import _StubOcr, _TokenHash  # noqa: E402


async def go(run_ocr: bool) -> dict:
    media = {"cycle.png": b"oxaloacetate condenses with acetyl coa"}
    tmp = Path(tempfile.mkdtemp())
    runtime = EmbeddingRuntime(model=None)
    derived = DerivedTextStore(path=tmp / "cache" / "shrike.db", engine_factory=NativeDerivedEngine)
    h = await Harness.assemble(
        collection_path=str(tmp / "collection.anki2"),
        cache_dir=str(tmp / "cache"),
        runtime=runtime, derived=derived, cooperative=False, hold_seconds=5.0,
        media_read=media.get, media_exists=lambda n: n in media,
    )
    await h.boot(start_embedding=False)
    b = _TokenHash()
    h.kernel.attach_embedder(shrike_native.PyEmbedder.capture(b))
    await h.kernel.reindex_if_needed()
    # Kernel maintained upsert so the field-text vector is indexed at upsert,
    # isolating the OCR-vector contribution from the sweep.
    raw = await h.kernel.upsert_notes_json(json.dumps([
        {"note_type": "Basic", "deck": "Default",
         "fields": {"Front": 'See diagram <img src="cycle.png">', "Back": "b"}},
        {"note_type": "Basic", "deck": "Default",
         "fields": {"Front": "qq unrelated filler card qq", "Back": "b"}},
    ]), "error", False)
    res = json.loads(raw)
    did = res[0]["id"]
    if run_ocr:
        h.attach_recognizer(_StubOcr())
        await h.recognition_sweep(batch_size=4)
    view = KernelIndexView(h.kernel, SimpleNamespace(backend=b))
    hits = view.search(["oxaloacetate condenses with acetyl coa today"], top_k=5)[0]
    scores = {x["note_id"]: 1.0 - x["distance"] for x in hits}
    await h.close()
    return {"run_ocr": run_ocr, "diagram_score": scores.get(did), "in_scores": did in scores}


async def main() -> None:
    no_ocr = await go(False)
    with_ocr = await go(True)
    print("OCR vector ABSENT (field-text only):", no_ocr,
          "-> FAILS >=0.6:", (no_ocr["diagram_score"] or 0) < 0.6)
    print("OCR vector PRESENT (max-over-items): ", with_ocr,
          "-> PASSES >=0.6:", (with_ocr["diagram_score"] or 0) >= 0.6)


if __name__ == "__main__":
    asyncio.run(main())
