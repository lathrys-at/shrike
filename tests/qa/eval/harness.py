#!/usr/bin/env python3
"""QA eval harness — capture, grade, and report. Server-touching glue around the
pure grader in ``grade.py``.

The orchestration (reset fixture → spawn a cold Haiku agent → capture → grade)
is driven externally (an interactive session, or a future claude -p run.py).
This CLI provides the deterministic, reusable steps:

    harness.py prompt   --scenario 01 --config with_skill   # emit the exact agent prompt
    harness.py baseline --out runs/<batch>/01/with_skill/r1  # snapshot before the run
    harness.py grade    --scenario 01 --dir runs/<batch>/01/with_skill/r1 \
                        --transcript <file>                  # capture + grade after the run
    harness.py report   --batch runs/<batch>                 # aggregate → report.md

Per-run dir holds: baseline.json, transcript.txt, run.json, grading.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from grade import DEFAULT_DUP_THRESHOLD, grade_run, summarize
from judge import DEFAULT_JUDGE_MODEL, run_judge
from prompts import SHRIKE_BIN, baseline_prompt, with_skill_prompt

HERE = Path(__file__).resolve().parent
SCENARIOS_DIR = HERE.parent / "scenarios"
SCENARIOS_YAML = HERE / "scenarios.yaml"
LOG_PATH = HERE.parent / "run" / "logs" / "shrike.log"


def _shrike_json(*args: str) -> Any:
    out = subprocess.run(
        [SHRIKE_BIN, "--json", *args], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)


def _load_spec() -> dict[str, Any]:
    return yaml.safe_load(SCENARIOS_YAML.read_text())


def _scenario(spec: dict[str, Any], sid: str) -> dict[str, Any]:
    for s in spec["scenarios"]:
        if s["id"] == sid:
            return s
    raise SystemExit(f"unknown scenario id: {sid}")


def _read_prompt_md(sid: str) -> str:
    """Pull the fenced prompt block from scenarios/<id>-*.md (single source)."""
    matches = sorted(SCENARIOS_DIR.glob(f"{sid}-*.md"))
    if not matches:
        raise SystemExit(f"no scenario markdown for id {sid} in {SCENARIOS_DIR}")
    text = matches[0].read_text()
    after = text.split("## Prompt", 1)
    if len(after) != 2:
        raise SystemExit(f"{matches[0].name} has no '## Prompt' section")
    fence = after[1].split("```")
    if len(fence) < 3:
        raise SystemExit(f"{matches[0].name} has no fenced prompt block")
    return fence[1].lstrip("\n").rstrip()


def _observe_agent_calls(log_offset: int) -> dict[str, Any]:
    """Parse the agent's actual tool calls from the server log slice written
    since ``log_offset`` (the baseline). Surfaces *how* it worked — orientation
    calls, the real search query strings (DEBUG), and post-write thrash — which
    the outcome-based grader can't see. Call this BEFORE the harness makes its
    own queries, so the slice is the agent's activity alone.
    """
    if not LOG_PATH.exists():
        return {}
    lines = LOG_PATH.read_text()[log_offset:].splitlines()

    orientation = sum(1 for ln in lines if "collection_info sections=" in ln)
    search_idxs = [i for i, ln in enumerate(lines) if "search_notes queries=" in ln]
    upsert_idx = next((i for i, ln in enumerate(lines) if "upsert_notes count=" in ln), None)

    queries: list[str] = []
    marker = "search_notes query strings: "
    for ln in lines:
        if marker in ln:
            frag = ln.split(marker, 1)[1].strip()
            try:
                queries.extend(ast.literal_eval(frag))
            except (ValueError, SyntaxError):
                queries.append(frag)

    after = lambda i: upsert_idx is not None and i > upsert_idx  # noqa: E731
    return {
        "orientation_calls": orientation,
        "search_calls": len(search_idxs),
        "search_queries": queries,
        "post_upsert_searches": sum(1 for i in search_idxs if after(i)),
        "post_upsert_readbacks": sum(
            1 for i, ln in enumerate(lines) if "list_notes ids=" in ln and after(i)
        ),
    }


# -- subcommands -------------------------------------------------------------


def cmd_prompt(args: argparse.Namespace) -> int:
    user = _read_prompt_md(args.scenario)
    builder = with_skill_prompt if args.config == "with_skill" else baseline_prompt
    print(builder(user))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    info = _shrike_json("info", "--decks")
    decks = [d["name"] for d in info.get("decks", [])]
    summary = _shrike_json("info")["summary"]
    baseline = {
        "decks": decks,
        "note_count": summary["notes"],
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        # byte offset into the server log, so grade can slice out exactly the
        # agent's tool calls (everything logged after this point, before the
        # harness makes its own queries).
        "log_offset": LOG_PATH.stat().st_size if LOG_PATH.exists() else 0,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print(f"baseline: {baseline['note_count']} notes, decks={decks}, since={baseline['timestamp']}")
    return 0


def _nearest_existing_score(note_id: int, sibling_ids: set[int]) -> float | None:
    """Top cosine of a created note against PRE-EXISTING notes (siblings excluded).
    None if the index/embeddings aren't available."""
    try:
        resp = _shrike_json("note", "search", "--similar-to", str(note_id), "--top-k", "10")
    except subprocess.CalledProcessError:
        return None
    best: float | None = None
    for group in resp.get("results", []):
        for m in group.get("matches", []):
            if m["id"] in sibling_ids or m["id"] == note_id:
                continue
            best = m["score"] if best is None else max(best, m["score"])
    return best


def cmd_grade(args: argparse.Namespace) -> int:
    spec = _load_spec()
    scn = _scenario(spec, args.scenario)
    dup_threshold = float(spec.get("dup_threshold", DEFAULT_DUP_THRESHOLD))
    run_dir = Path(args.dir)
    baseline = json.loads((run_dir / "baseline.json").read_text())

    # Observe the agent's calls from the log FIRST — before the harness's own
    # queries below append to it.
    observed = _observe_agent_calls(baseline.get("log_offset", 0))

    listing = _shrike_json("note", "list", "--since", baseline["timestamp"], "--limit", "200")
    created_raw = listing.get("notes", [])
    new_ids = {n["id"] for n in created_raw}

    created = []
    for n in created_raw:
        fields = n.get("content") or {}
        front = fields.get("Front") or fields.get("Text") or next(iter(fields.values()), "")
        created.append(
            {
                "id": n["id"],
                "note_type": n["note_type"],
                "deck": n["deck"],
                "tags": n.get("tags", []),
                "front": front,
                "content": fields,  # full fields, for the advisory judge
                "nearest_existing_score": _nearest_existing_score(n["id"], new_ids),
            }
        )

    transcript = Path(args.transcript).read_text() if args.transcript else ""
    run = {
        "scenario_id": args.scenario,
        "config": run_dir.parent.name,
        "repeat": run_dir.name,
        "baseline": baseline,
        "created_notes": created,
        "observed": observed,
        "transcript": transcript,
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2))

    results = grade_run(run, scn, dup_threshold)
    s = summarize(results)
    grading: dict[str, Any] = {"scenario_id": args.scenario, "summary": s, "expectations": results}

    # Advisory LLM judge (Sonnet by default). Never gates: it's stored alongside
    # the mechanical result and surfaced in the report as a separate column.
    judge: dict[str, Any] | None = None
    if not args.no_judge:
        judge = run_judge(scn, _read_prompt_md(args.scenario), created, model=args.judge_model)
        grading["judge"] = judge

    (run_dir / "grading.json").write_text(json.dumps(grading, indent=2))

    mark = "PASS" if s["failed"] == 0 else "FAIL"
    hdr = f"[{mark}] {args.scenario} {run['config']} {run['repeat']}"
    print(f"{hdr}: {s['passed']}/{s['total']} assertions, {len(created)} cards")
    for r in results:
        if not r["passed"]:
            print(f"    ✗ {r['text']} — {r['evidence']}")
    if judge is not None:
        verdict = judge.get("verdict", "?")
        detail = judge.get("rubric") or judge.get("error") or judge.get("raw", "")
        print(f"    judge[{judge.get('model')}]: {verdict} — {str(detail)[:200]}")
        for issue in judge.get("issues", [])[:4]:
            print(f"      issue: {issue}")
    if observed:
        print(
            f"    behavior: orient={observed['orientation_calls']} "
            f"searches={observed['search_calls']} "
            f"post-upsert-search={observed['post_upsert_searches']} "
            f"readback={observed['post_upsert_readbacks']}"
        )
        for q in observed.get("search_queries", []):
            print(f"      query: {q!r}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    batch = Path(args.batch)
    # cell -> (scenario, config) -> list of run-level pass bools
    cells: dict[tuple[str, str], list[bool]] = {}
    # cell -> list of advisory judge verdicts ("pass"/"mixed"/"fail"/...)
    judges: dict[tuple[str, str], list[str]] = {}
    fail_lines: list[str] = []
    for gpath in sorted(batch.rglob("grading.json")):
        g = json.loads(gpath.read_text())
        sid = g["scenario_id"]
        config = gpath.parent.parent.name
        run_pass = g["summary"]["failed"] == 0
        cells.setdefault((sid, config), []).append(run_pass)
        if "judge" in g:
            judges.setdefault((sid, config), []).append(g["judge"].get("verdict", "?"))
        if not run_pass:
            fails = [e["text"] for e in g["expectations"] if not e["passed"]]
            fail_lines.append(f"- {sid} {config} {gpath.parent.name}: {', '.join(fails)}")

    def judge_cell(key: tuple[str, str]) -> str:
        verdicts = judges.get(key, [])
        if not verdicts:
            return "—"
        return f"{verdicts.count('pass')}/{len(verdicts)}✓"

    sids = sorted({sid for sid, _ in cells})
    lines = [
        "# QA eval report",
        "",
        "Mechanical run-level pass rate (all assertions passed) per scenario — the",
        "gate. The `judge` column is the advisory Sonnet read (pass count), which",
        "does not gate.",
        "",
    ]
    lines.append("| Scenario | with_skill | baseline | delta | judge (ws) |")
    lines.append("|---|---|---|---|---|")
    for sid in sids:
        ws = cells.get((sid, "with_skill"), [])
        bl = cells.get((sid, "baseline"), [])
        ws_rate = sum(ws) / len(ws) if ws else 0.0
        bl_rate = sum(bl) / len(bl) if bl else 0.0
        ws_s = f"{sum(ws)}/{len(ws)}" if ws else "—"
        bl_s = f"{sum(bl)}/{len(bl)}" if bl else "—"
        delta = f"{(ws_rate - bl_rate) * 100:+.0f}pp" if ws and bl else "—"
        lines.append(f"| {sid} | {ws_s} | {bl_s} | {delta} | {judge_cell((sid, 'with_skill'))} |")
    if fail_lines:
        lines += ["", "## Failing runs", "", *fail_lines]

    report = "\n".join(lines) + "\n"
    (batch / "report.md").write_text(report)
    print(report)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prompt", help="emit the canonical agent prompt for a scenario")
    sp.add_argument("--scenario", required=True)
    sp.add_argument("--config", choices=["with_skill", "baseline"], required=True)
    sp.set_defaults(func=cmd_prompt)

    sb = sub.add_parser("baseline", help="snapshot the collection before a run")
    sb.add_argument("--out", required=True)
    sb.set_defaults(func=cmd_baseline)

    sg = sub.add_parser("grade", help="capture created notes and grade a finished run")
    sg.add_argument("--scenario", required=True)
    sg.add_argument("--dir", required=True, help="run dir containing baseline.json")
    sg.add_argument("--transcript", help="file with the agent's final report")
    sg.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"model alias for the advisory LLM judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    sg.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the advisory LLM judge (mechanical grade only)",
    )
    sg.set_defaults(func=cmd_grade)

    sr = sub.add_parser("report", help="aggregate a batch of graded runs")
    sr.add_argument("--batch", required=True)
    sr.set_defaults(func=cmd_report)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
