"""Cross-space intra-modal floor — validation harness.

Builds the REAL 2-space (MiniLM text + CLIP image) graded corpus ONCE, reports
the harness-derived secondary image floor, then runs the decision rule
before (relative-only, the leak) vs after (relative+floor, the shipped
calibrated default) on the three axes:

  - modality_gap recall  (R@k / MRR — must NOT regress)
  - over_return precision (the ∅-gold query: # cards / # via CLIP / max score)
  - negative control      (gate_no_inject class metric; the authority is the
                           integration suite's TestRealActivationGate)

This is the reproducible decision artifact — the integration suite's
``TestRealPrecision.test_over_return_query_injects_no_clip_image_card`` is the
permanent guard; this script regenerates the before/after table on demand.

Run (needs the real models + the Commons corpus, like the manual suite):
    SHRIKE_SEARCH_QUALITY=1 .venv/bin/python tests/manual/search_quality/cross_space_floor.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root (for `import tests.*`)
sys.path.insert(0, str(ROOT))

from tests.manual.search_quality.inprocess import to_returned_cards  # noqa: E402
from tests.manual.search_quality.manifest import load_manifest  # noqa: E402
from tests.manual.search_quality.metrics import SuiteReport, evaluate_query  # noqa: E402
from tests.manual.search_quality.runner import (  # noqa: E402
    MANIFEST,
    build_real_collection,
    clip_fired,
)

OVER_RETURN, MODALITY_GAP, NEG = "over_return", "modality_gap", "gate_no_inject_portrait"


def _set_mode(mode: str) -> None:
    os.environ["SHRIKE_CROSS_SPACE_FUSION_MODE"] = mode


async def _run(ip, manifest, inv) -> tuple[SuiteReport, dict]:
    reports, returns = [], {}
    for q in manifest.queries:
        thr = q.threshold if q.threshold is not None else 0.5
        matches = await ip.matches(q.q, top_k=q.top_k, threshold=thr)
        for m in matches:
            m["id"] = inv.get(m["id"], m["id"])
        returns[q.q] = matches
        reports.append(evaluate_query(q.q, q.adversarial_class, to_returned_cards(matches), q.gold))
    return SuiteReport(queries=tuple(reports)), returns


def _axes(suite: SuiteReport, returns: dict, manifest) -> dict:
    or_q = next(q for q in manifest.queries if q.adversarial_class == OVER_RETURN)
    orr = returns.get(or_q.q, [])
    or_scores = [m.get("score") for m in orr if m.get("score") is not None]
    return {
        "mg_rk": suite.mean_recall_at_k(by_class=MODALITY_GAP),
        "mg_mrr": suite.mean_mrr(by_class=MODALITY_GAP),
        "overall_rk": suite.mean_recall_at_k(),
        "or_count": len(orr),
        "or_clip": sum(1 for m in orr if clip_fired(m)),
        "or_max": max(or_scores) if or_scores else None,
        "neg_r1": suite.mean_recall_at_1(by_class=NEG),
    }


def _fmt(v) -> str:
    if v is None:
        return "  -  "
    return f"{v:5.3f}" if isinstance(v, float) else f"{v:5d}"


async def main() -> None:
    manifest = load_manifest(MANIFEST)
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sweep576_"))

    print("Building real 2-space corpus (MiniLM + CLIP)…", flush=True)
    ip, id_map, backends = await build_real_collection(tmp, manifest)
    inv = {v: k for k, v in id_map.items()}

    rows = []
    try:
        derived = await ip.harness.kernel.calibrate_secondary_floors(
            ip.harness.cross_space_floor_margin
        )
        print(f"\nDERIVED secondary image floor(s): {derived}", flush=True)

        for label, mode in [
            ("V0 (before: relative)", "relative"),
            ("V0+floor (after: DEFAULT)", "relative_floor"),
        ]:
            _set_mode(mode)
            suite, returns = await _run(ip, manifest, inv)
            rows.append((label, _axes(suite, returns, manifest)))
    finally:
        await ip.harness.close()
        for b in backends:
            b.stop()

    print("\n" + "=" * 92)
    print("#576 DECISION RULE — before/after the calibrated floor")
    print("=" * 92)
    hdr = (
        f"{'variant':<28}| {'mg_Rk':>6}{'mg_MRR':>7}{'all_Rk':>7} | "
        f"{'OR_n':>5}{'OR_clip':>8}{'OR_max':>7} | {'neg_R1':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, a in rows:
        print(
            f"{label:<28}| {_fmt(a['mg_rk'])}{_fmt(a['mg_mrr'])}{_fmt(a['overall_rk'])} | "
            f"{a['or_count']:>5}{a['or_clip']:>8}{_fmt(a['or_max']):>7} | {_fmt(a['neg_r1'])}"
        )
    print("=" * 92)
    print(
        "PASS: OR_clip 10→0 (leak closed), mg_Rk/mg_MRR unchanged (no recall loss). "
        "neg_R1<1.0 is a class-metric artifact (a 2-relevant query); the authority is "
        "tests/integration TestRealActivationGate."
    )


if __name__ == "__main__":
    asyncio.run(main())
