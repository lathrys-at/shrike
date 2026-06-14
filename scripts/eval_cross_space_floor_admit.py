"""#580 cross-space FLOOR-ADMISSION — validation harness.

Builds the REAL 2-space (MiniLM text + CLIP image) graded corpus ONCE, then
sweeps the production baseline against the floor-admission prototype modes on
the FOUR #580 axes (single-image-space case — the production-valid config; the
N>=2 multiplicity case is a kernel unit test + a config assertion, not a corpus
axis):

  1a. CORROBORATION (recall, the WIN): the 7 filename-collision cards
      (heart/skeleton/plant_cell/animal_cell/citric_acid/periodic_table/
      great_wall) — their <img src> filename lexically wins the text space, so
      the RELATIVE gate shuts CLIP out. Under floor-admission, does the on-topic
      CLIP hit reach RRF provenance (image#clip present), and does the card's
      rank hold/improve vs the relative-gate baseline?
  1b. SPURIOUS FILENAME (precision, the GUARD): the 3 homonym matched pairs
      (jaguar animal/car, crane bird/machine, bass fish/guitar) — one filename
      word, two visual senses. For the disambiguated query, the floor must
      ADMIT CLIP for the on-topic image (image#clip present) and REJECT it for
      the off-topic one (no image#clip). The floor is the SOLE discriminator
      now that the relative gate is dropped — this is the load-bearing test.
  2.  modality_gap recall (no regression): R@1/@5/@k, MRR, nDCG@10.
  3.  over-return leak (#576, must stay closed): 0 image#clip on the ∅-gold
      query.

Run (needs the real models + the Commons corpus, like the manual suite):
    SHRIKE_SEARCH_QUALITY=1 .venv/bin/python scripts/eval_cross_space_floor_admit.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.search_quality.inprocess import to_returned_cards  # noqa: E402
from tests.search_quality.manifest import load_manifest  # noqa: E402
from tests.search_quality.metrics import SuiteReport, evaluate_query  # noqa: E402
from tests.search_quality.runner import (  # noqa: E402
    MANIFEST,
    build_real_collection,
    clip_fired,
)

OVER_RETURN = "over_return"
MODALITY_GAP = "modality_gap"
SPURIOUS = "spurious_filename"

# The 7 filename-collision cards (manifest ids) — their <img src> filename word
# is in the query, so the relative gate stays shut on the pre-#580 baseline.
COLLISION_IDS = {1: "heart", 2: "skeleton", 6: "plant_cell", 7: "animal_cell",
                 8: "citric_acid", 10: "periodic_table", 15: "great_wall"}

# The modes to sweep: the pre-#580 production gate (now eval-only) vs the #580
# floor-admission family. `FloorAdmit` is the PRODUCTION default since #580.
# (FloorAdmitBudget B=1.0 is included to show N=1 — the production-valid single-
# image-space case — is unaffected by the budget; the budget only ever bites at
# N>=2, which the single-image-space config error makes impossible.)
MODES = [
    ("RelativeFloor (pre-#580 gate)", "relative_floor"),
    ("FloorAdmit (PROD default #580)", "floor_admit"),
    ("FloorAdmitBudget B=1.0 (N=1)", "floor_admit_budget"),
    ("SoftFloorAdmit (tau=0.05)", "soft_floor_admit"),
    ("SoftFloorAdmit (tau=0.10)", "soft_floor_admit"),
]
# τ overrides per row index (None = leave default 0.05).
TAU_OVERRIDE = {4: "0.10"}


def _set_mode(mode: str, tau: str | None) -> None:
    os.environ["SHRIKE_CROSS_SPACE_FUSION_MODE"] = mode
    if tau is not None:
        os.environ["SHRIKE_CROSS_SPACE_TAU"] = tau
    else:
        os.environ.pop("SHRIKE_CROSS_SPACE_TAU", None)


async def _run(ip, manifest, inv) -> tuple[SuiteReport, dict]:
    """Run every manifest query; return the graded suite + the raw returns
    (manifest-id-keyed, provenance intact) for the per-card provenance probes."""
    reports, returns = [], {}
    for q in manifest.queries:
        thr = q.threshold if q.threshold is not None else 0.5
        matches = await ip.matches(q.q, top_k=q.top_k, threshold=thr)
        for m in matches:
            m["id"] = inv.get(m["id"], m["id"])
        returns[q.q] = matches
        reports.append(evaluate_query(q.q, q.adversarial_class, to_returned_cards(matches), q.gold))
    return SuiteReport(queries=tuple(reports)), returns


def _rank_of(matches: list, note_id: int) -> int | None:
    for i, m in enumerate(matches, start=1):
        if int(m["id"]) == note_id:
            return i
    return None


def _clip_present(matches: list, note_id: int) -> bool:
    for m in matches:
        if int(m["id"]) == note_id:
            return clip_fired(m)
    return False


def _collision_axis(returns: dict, manifest) -> dict:
    """Axis 1a: per collision card, did image#clip reach provenance, and at what
    rank. Keyed by the modality_gap query that targets each collision id."""
    out = {}
    for q in manifest.queries:
        if q.adversarial_class != MODALITY_GAP:
            continue
        targets = [nid for nid in q.gold.relevant_ids if nid in COLLISION_IDS]
        if not targets:
            continue
        nid = targets[0]
        matches = returns.get(q.q, [])
        out[nid] = {
            "name": COLLISION_IDS[nid],
            "rank": _rank_of(matches, nid),
            "clip": _clip_present(matches, nid),
        }
    return out


def _score_of(matches: list, note_id: int) -> float | None:
    for m in matches:
        if int(m["id"]) == note_id:
            return m.get("score")
    return None


def _spurious_axis(returns: dict, manifest) -> list:
    """Axis 1b: per homonym pair, the on-topic (grade-3) and off-topic (grade-0)
    cards' rank + score + clip-presence.

    NOTE the precision metric is ORDERING, not provenance-presence: the image
    activation signal is SPACE-LEVEL (once the floor opens, the WHOLE image
    ranking enters as one `image#clip` signal — so both senses carry the
    provenance whenever either does; this is the established #201b behaviour, NOT
    a floor-admission change). The discrimination that matters is whether the
    on-topic sense RANKS ABOVE the off-topic one (cosine separation) and whether
    the on-topic card wins rank-1 overall."""
    out = []
    for q in manifest.queries:
        if q.adversarial_class != SPURIOUS:
            continue
        on = [nid for nid, g in q.gold.grades.items() if g >= 2]
        off = [nid for nid, g in q.gold.grades.items() if g == 0]
        matches = returns.get(q.q, [])
        row = {"q": q.q[:30]}
        if on:
            row["on_id"] = on[0]
            row["on_rank"] = _rank_of(matches, on[0])
            row["on_clip"] = _clip_present(matches, on[0])
            row["on_score"] = _score_of(matches, on[0])
        if off:
            row["off_id"] = off[0]
            row["off_rank"] = _rank_of(matches, off[0])
            row["off_clip"] = _clip_present(matches, off[0])
            row["off_score"] = _score_of(matches, off[0])
        out.append(row)
    return out


def _axes(suite: SuiteReport, returns: dict, manifest) -> dict:
    or_q = next(q for q in manifest.queries if q.adversarial_class == OVER_RETURN)
    orr = returns.get(or_q.q, [])
    return {
        "mg_r1": suite.mean_recall_at_1(by_class=MODALITY_GAP),
        "mg_r5": suite.mean_recall_at_5(by_class=MODALITY_GAP),
        "mg_rk": suite.mean_recall_at_k(by_class=MODALITY_GAP),
        "mg_mrr": suite.mean_mrr(by_class=MODALITY_GAP),
        "mg_ndcg": suite.mean_ndcg(by_class=MODALITY_GAP),
        "or_count": len(orr),
        "or_clip": sum(1 for m in orr if clip_fired(m)),
        "collision": _collision_axis(returns, manifest),
        "spurious": _spurious_axis(returns, manifest),
    }


def _fmt(v) -> str:
    if v is None:
        return "  -  "
    return f"{v:5.3f}" if isinstance(v, float) else f"{v:>5}"


async def main() -> None:
    manifest = load_manifest(MANIFEST)
    tmp = Path(tempfile.mkdtemp(prefix="sweep580_"))
    print("Building real 2-space corpus (MiniLM + CLIP)…", flush=True)
    ip, id_map, backends = await build_real_collection(tmp, manifest)
    inv = {v: k for k, v in id_map.items()}

    rows = []
    try:
        derived = await ip.harness.kernel.calibrate_secondary_floors(ip.harness.cross_space_floor_margin)
        print(f"\nDERIVED secondary image floor(s): {derived}\n", flush=True)
        for idx, (label, mode) in enumerate(MODES):
            _set_mode(mode, TAU_OVERRIDE.get(idx))
            suite, returns = await _run(ip, manifest, inv)
            rows.append((label, _axes(suite, returns, manifest)))
    finally:
        await ip.harness.close()
        for b in backends:
            b.stop()

    # ── Axis 2/3 summary table ────────────────────────────────────────────
    print("=" * 100)
    print("#580 FLOOR-ADMISSION — modality_gap recall (axis 2) + over-return leak (axis 3)")
    print("=" * 100)
    hdr = (f"{'mode':<32}| {'mg_R1':>6}{'mg_R5':>6}{'mg_Rk':>6}{'mg_MRR':>7}{'mg_nDCG':>8} | "
           f"{'OR_n':>5}{'OR_clip':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label, a in rows:
        print(f"{label:<32}| {_fmt(a['mg_r1'])}{_fmt(a['mg_r5'])}{_fmt(a['mg_rk'])}"
              f"{_fmt(a['mg_mrr'])}{_fmt(a['mg_ndcg'])} | {a['or_count']:>5}{a['or_clip']:>8}")

    # ── Axis 1a: collision corroboration (clip present + rank) ─────────────
    print("\n" + "=" * 100)
    print("AXIS 1a — filename-collision CORROBORATION (clip=image#clip reached RRF, r=rank)")
    print("=" * 100)
    names = [COLLISION_IDS[i] for i in sorted(COLLISION_IDS)]
    print(f"{'mode':<32}| " + "".join(f"{n[:9]:>11}" for n in names))
    print("-" * (34 + 11 * len(names)))
    for label, a in rows:
        cells = []
        for nid in sorted(COLLISION_IDS):
            c = a["collision"].get(nid, {})
            flag = "C" if c.get("clip") else "·"
            cells.append(f"{flag}r{c.get('rank')}")
        print(f"{label:<32}| " + "".join(f"{c:>11}" for c in cells))

    # ── Axis 1b: spurious-filename precision (ORDERING, not provenance) ─────
    print("\n" + "=" * 100)
    print("AXIS 1b — spurious-filename PRECISION (homonym pairs). The image signal is SPACE-LEVEL,")
    print("so both senses carry image#clip whenever either does (the #201b gate, not a #580 change).")
    print("The precision that matters is ORDERING: on-topic must rank ABOVE off-topic (cosine sep).")
    print("=" * 100)
    for label, a in rows:
        print(f"\n  {label}")
        for row in a["spurious"]:
            os_, ofs = row.get("on_score"), row.get("off_score")
            sep = (os_ - ofs) if (os_ is not None and ofs is not None) else None
            on = f"on(id{row.get('on_id')}) r={row.get('on_rank')} cos={_fmt(os_)}"
            off = f"off(id{row.get('off_id')}) r={row.get('off_rank')} cos={_fmt(ofs)}"
            # The discrimination: on-topic ranks above off-topic AND wins rank-1.
            ordered = (row.get("on_rank") is not None and row.get("off_rank") is not None
                       and row["on_rank"] < row["off_rank"])
            ok = "PASS" if (ordered and row.get("on_rank") == 1) else "FAIL"
            print(f"    [{ok}] {row['q']:<32} {on:<28} {off:<28} sep={_fmt(sep)}")
    print("\n" + "=" * 100)
    print("DECISION RULE: a floor-admission mode is GO iff — axis1a: image#clip reaches the")
    print("collision cards (≥ baseline) and ranks hold/improve; axis1b: on-topic outranks off-topic")
    print("and wins rank-1 on every pair (no precision regression vs baseline); axis2: mg recall does")
    print("not regress vs RelativeFloor; axis3: OR_clip stays 0.")


if __name__ == "__main__":
    asyncio.run(main())
