#!/usr/bin/env python3
"""Automated QA eval runner.

Drives the full loop end-to-end, no human in the chair: for every
scenario × config × repeat it resets the fixture, spawns a **cold author agent**
via ``claude -p`` (Haiku 4.5 by default — the deliberately-weak model the eval
measures), captures the agent's final report, then grades it (mechanical gate +
advisory Sonnet judge) and writes a batch report.

This is the automated counterpart to driving ``harness.py`` by hand. The author
runs with ``--dangerously-skip-permissions`` because the QA collection is a
disposable fixture rebuilt every cell — fine here, never point it at a real one.

Prerequisites (same as the manual harness — see ../README.md):
    export LLAMA_SERVER_PATH=/path/to/llama-server
    export SHRIKE_EMBEDDING_MODEL=/path/to/embedding-model.gguf

Examples:
    # 1x sweep, with_skill only, Sonnet judge (the default):
    tests/qa/eval/run.py --repeats 1 --configs with_skill

    # 3x depth across two scenarios, no judge (mechanical only, fast):
    tests/qa/eval/run.py --scenarios 01,03 --repeats 3 --no-judge

    # full matrix with a different judge model:
    tests/qa/eval/run.py --configs with_skill,baseline --judge-model opus
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from judge import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_THINKING
from prompts import SHRIKE_BIN, baseline_prompt, with_skill_prompt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # tests/qa/eval -> repo root
LAUNCH = ROOT / "scripts" / "launch-qa-server.sh"
HARNESS = HERE / "harness.py"
PY = sys.executable

ALL_SCENARIOS = ["01", "02", "03", "04", "05", "06"]
DEFAULT_AUTHOR_MODEL = "haiku"
# Author thinking is on by default: the weak model benefits most from reasoning
# through the type/dedup/cue decisions. Generous budget — it's a multi-turn agent.
DEFAULT_AUTHOR_THINKING = 8000
INDEX_TIMEOUT = 180.0
AUTHOR_TIMEOUT = 900.0


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {msg}", flush=True)


def _preflight() -> None:
    """Fail fast on missing prerequisites instead of hanging in the index-ready
    wait — a missing embedder leaves the index ``unavailable``, which the wait
    loop can't distinguish from a slow build until it times out."""
    problems: list[str] = []

    if shutil.which("claude") is None:
        problems.append("`claude` CLI not on PATH — needed to spawn the author and judge")

    model = os.environ.get("SHRIKE_EMBEDDING_MODEL", "")
    if not model:
        problems.append(
            "SHRIKE_EMBEDDING_MODEL is unset — the eval needs embeddings; "
            "without one the index never reaches `ready`"
        )
    elif not Path(model).is_file():
        problems.append(f"SHRIKE_EMBEDDING_MODEL points at a missing file: {model}")

    llama = os.environ.get("LLAMA_SERVER_PATH", "")
    if llama and not Path(llama).is_file():
        problems.append(f"LLAMA_SERVER_PATH points at a missing file: {llama}")
    elif not llama and shutil.which("llama-server") is None:
        problems.append("LLAMA_SERVER_PATH is unset and `llama-server` is not on PATH")

    if problems:
        raise SystemExit(
            "preflight failed:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nSee tests/qa/README.md for the env setup."
        )


def _read_prompt_md(sid: str) -> str:
    # Same single-source read as harness.py, duplicated to keep run.py standalone.
    matches = sorted((HERE.parent / "scenarios").glob(f"{sid}-*.md"))
    if not matches:
        raise SystemExit(f"no scenario markdown for id {sid}")
    fence = matches[0].read_text().split("## Prompt", 1)[1].split("```")
    return fence[1].lstrip("\n").rstrip()


def _reset_fixture() -> None:
    """Clean rebuild + restart the QA server (stops any running one first)."""
    _log("reset: launching clean QA server…")
    proc = subprocess.run(["bash", str(LAUNCH)], capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("launch-qa-server.sh failed")


def _wait_index_ready(timeout: float = INDEX_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = subprocess.run(
            [SHRIKE_BIN, "--json", "index", "status"], capture_output=True, text=True
        ).stdout
        if '"state": "ready"' in out:
            _log("index ready")
            return
        time.sleep(2)
    raise SystemExit(f"index not ready within {timeout:.0f}s")


def _parse_author_stream(stdout: str) -> tuple[str, dict[str, Any]]:
    """Parse the author's ``--output-format stream-json`` NDJSON into (final
    report text, run stats). Tool calls are counted from ``tool_use`` content
    blocks. Token counts come from the terminal ``result`` event's usage — the
    authoritative cumulative figures. (Per-turn ``assistant`` events carry an
    *early* usage snapshot with output not yet finalized, so summing them
    undercounts output badly — the bug this replaces.)"""
    text = ""
    parts: list[str] = []
    tool_calls = num_turns = 0
    usage: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for b in ev.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    tool_calls += 1
                elif b.get("type") == "text":
                    parts.append(b.get("text", ""))
        elif ev.get("type") == "result":
            text = ev.get("result", "") or text
            num_turns = ev.get("num_turns") or num_turns
            usage = ev.get("usage", {}) or {}
    # Sum per-call iterations (cumulative across the whole run) if present;
    # otherwise fall back to the top-level result usage.
    iters = usage.get("iterations") or ([usage] if usage else [])
    in_tok = sum(i.get("input_tokens", 0) for i in iters)
    out_tok = sum(i.get("output_tokens", 0) for i in iters)
    cache_tok = sum(i.get("cache_read_input_tokens", 0) for i in iters)
    if not text:
        text = "\n".join(p for p in parts if p).strip()
    return text, {
        "tool_calls": tool_calls,
        "num_turns": num_turns,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_read_tokens": cache_tok,
        "total_tokens": in_tok + out_tok,
    }


def _author(config: str, sid: str, model: str, thinking: int) -> tuple[str, dict[str, Any], str]:
    """Spawn the cold author agent; return (final report, run stats, raw stream).
    ``thinking`` sets the author's MAX_THINKING_TOKENS budget (0 disables)."""
    user = _read_prompt_md(sid)
    builder = with_skill_prompt if config == "with_skill" else baseline_prompt
    prompt = builder(user)
    _log(f"author: claude -p --model {model} (thinking={thinking}) ({config} {sid})…")
    t0 = time.monotonic()
    env = {**os.environ, "MAX_THINKING_TOKENS": str(max(thinking, 0))}
    proc = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=AUTHOR_TIMEOUT,
        env=env,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"author claude -p failed (exit {proc.returncode})")
    text, stats = _parse_author_stream(proc.stdout)
    stats.update(model=model, thinking=thinking, duration_s=round(time.monotonic() - t0, 1))
    _log(
        f"author done in {stats['duration_s']:.0f}s — {stats['tool_calls']} tools, "
        f"{stats['num_turns']} turns, {stats['total_tokens']:,} tokens "
        f"({stats['output_tokens']:,} out)"
    )
    return text, stats, proc.stdout


def _baseline(run_dir: Path) -> None:
    subprocess.run([PY, str(HARNESS), "baseline", "--out", str(run_dir)], check=True, cwd=str(HERE))


def _grade(
    sid: str,
    run_dir: Path,
    transcript: Path,
    judge_model: str,
    judge_thinking: int,
    no_judge: bool,
) -> None:
    cmd = [
        PY,
        str(HARNESS),
        "grade",
        "--scenario",
        sid,
        "--dir",
        str(run_dir),
        "--transcript",
        str(transcript),
    ]
    if no_judge:
        cmd.append("--no-judge")
    else:
        cmd += ["--judge-model", judge_model, "--judge-thinking", str(judge_thinking)]
    subprocess.run(cmd, check=True, cwd=str(HERE))


def _report(batch: Path) -> None:
    subprocess.run([PY, str(HARNESS), "report", "--batch", str(batch)], check=True, cwd=str(HERE))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenarios", default=",".join(ALL_SCENARIOS), help="comma-separated ids")
    p.add_argument("--configs", default="with_skill", help="with_skill and/or baseline")
    p.add_argument("--repeats", type=int, default=1, help="runs per cell (depth)")
    p.add_argument("--batch", default=None, help="batch name (default: timestamp)")
    p.add_argument(
        "--author-model",
        default=DEFAULT_AUTHOR_MODEL,
        help=f"author model alias (default: {DEFAULT_AUTHOR_MODEL})",
    )
    p.add_argument(
        "--author-thinking",
        type=int,
        default=DEFAULT_AUTHOR_THINKING,
        help=f"author MAX_THINKING_TOKENS (default: {DEFAULT_AUTHOR_THINKING}; 0 off)",
    )
    p.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"judge model alias (default: {DEFAULT_JUDGE_MODEL})",
    )
    p.add_argument(
        "--judge-thinking",
        type=int,
        default=DEFAULT_JUDGE_THINKING,
        help=f"judge thinking budget (default: {DEFAULT_JUDGE_THINKING}; 0 off)",
    )
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="continue after a cell errors")
    args = p.parse_args()
    _preflight()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    batch_name = args.batch or f"auto-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    batch = HERE / "runs" / batch_name

    cells = [(s, c, r) for s in scenarios for c in configs for r in range(1, args.repeats + 1)]
    judge_desc = "off" if args.no_judge else f"{args.judge_model}/think={args.judge_thinking}"
    _log(
        f"batch {batch_name}: {len(cells)} cells "
        f"({len(scenarios)} scenarios × {len(configs)} configs × {args.repeats} repeats), "
        f"author={args.author_model}/think={args.author_thinking}, judge={judge_desc}"
    )

    # Drop a self-documenting record of the config that produced this batch, so a
    # haiku vs haiku+thinking vs sonnet run is never ambiguous after the fact.
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "config.json").write_text(
        json.dumps(
            {
                "batch": batch_name,
                "started": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                "scenarios": scenarios,
                "configs": configs,
                "repeats": args.repeats,
                "author_model": args.author_model,
                "author_thinking": args.author_thinking,
                "judge_model": None if args.no_judge else args.judge_model,
                "judge_thinking": None if args.no_judge else args.judge_thinking,
            },
            indent=2,
        )
    )

    errors = 0
    for sid, config, rep in cells:
        run_dir = batch / sid / config / f"r{rep}"
        _log(f"=== {sid} {config} r{rep} ===")
        try:
            _reset_fixture()
            _wait_index_ready()
            _baseline(run_dir)
            report_text, author_stats, author_raw = _author(
                config, sid, args.author_model, args.author_thinking
            )
            transcript = run_dir / "transcript.txt"
            transcript.write_text(report_text)
            (run_dir / "author_stats.json").write_text(json.dumps(author_stats, indent=2))
            (run_dir / "author_raw.jsonl").write_text(author_raw)  # for verifying token parse
            _grade(sid, run_dir, transcript, args.judge_model, args.judge_thinking, args.no_judge)
        except (SystemExit, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            errors += 1
            _log(f"!! cell {sid} {config} r{rep} failed: {e}")
            if not args.keep_going:
                raise SystemExit(
                    f"aborting after {errors} error(s); --keep-going to push through"
                ) from e

    _log(f"=== batch complete ({errors} cell error(s)) ===")
    _report(batch)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
