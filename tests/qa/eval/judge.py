"""Advisory LLM judge for QA eval runs.

The mechanical grader (``grade.py``) is the gate; this judge does **not** gate.
It reads the cards an agent actually created and rates the qualitative bits the
``assert`` block can't see — best card type per fact, atomicity, recall-framing —
against the scenario's ``judge`` rubric. It runs a cold ``claude -p`` (Sonnet by
default), self-contained: the prompt carries the material, the rubric, and the
cards, so the judge needs no tools and touches nothing.

Kept separate from ``grade.py`` so the deterministic grader stays pure and
unit-testable; this module is the one that shells out.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

DEFAULT_JUDGE_MODEL = "sonnet"

_INSTRUCTIONS = """\
You are grading Anki flashcards that a *weaker* AI model created from a user's
study material. Your job is a qualitative read the mechanical checks can't do:
did it pick the right kind of card for each fact, are the cards atomic (one fact
each), and do they demand recall rather than recognition? Judge the cards as
they'd actually be reviewed, not the model's explanation of them.

You are advisory — you do not pass or fail the run, you give an honest read.
Be concrete and a little exacting: name the specific card when you flag something.
If no cards were created, judge whether that was the right call for the material."""

_OUTPUT = """\
Return ONLY a JSON object, no prose around it:

{
  "verdict": "pass" | "mixed" | "fail",
  "rubric": "<2-4 sentences answering the rubric questions directly>",
  "strengths": ["<short>", ...],
  "issues": ["<short, each naming the card>", ...]
}

"pass" = cards are well-formed and the rubric is satisfied; "mixed" = mostly good
with real but minor problems; "fail" = the rubric's core ask is missed. Do not
use any tools — everything you need is in this message."""


def build_judge_prompt(
    scenario: dict[str, Any], user_request: str, created: list[dict[str, Any]]
) -> str:
    rubric = (scenario.get("judge") or "").strip() or "(no rubric provided)"
    if created:
        blocks = []
        for i, n in enumerate(created, 1):
            content = n.get("content") or {}
            fields = "\n".join(f"      {k}: {v}" for k, v in content.items()) or "      (empty)"
            blocks.append(
                f"  Card {i} — type={n.get('note_type')} deck={n.get('deck')} "
                f"tags={n.get('tags', [])}\n{fields}"
            )
        cards_block = "\n".join(blocks)
    else:
        cards_block = "  (the agent created no cards)"

    return (
        f"{_INSTRUCTIONS}\n\n"
        f"<study_material>\n{user_request}\n</study_material>\n\n"
        f"<rubric>\n{rubric}\n</rubric>\n\n"
        f"<cards_created>\n{cards_block}\n</cards_created>\n\n"
        f"{_OUTPUT}\n"
    )


def run_judge(
    scenario: dict[str, Any],
    user_request: str,
    created: list[dict[str, Any]],
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run the advisory judge via ``claude -p``. Always returns a dict; failure
    modes (``error``/``unparsed``) are encoded in ``verdict`` rather than raised,
    so a flaky judge never sinks a run's mechanical grade."""
    prompt = build_judge_prompt(scenario, user_request, created)
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"model": model, "verdict": "error", "error": f"timeout after {timeout:.0f}s"}
    except FileNotFoundError:
        return {"model": model, "verdict": "error", "error": "claude CLI not found on PATH"}

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()[:500] or f"exit {proc.returncode}"
        return {"model": model, "verdict": "error", "error": err}

    return _parse_result(proc.stdout, model)


def _parse_result(stdout: str, model: str) -> dict[str, Any]:
    # `--output-format json` wraps the model's text in {"result": "...", ...}.
    result_text = stdout
    try:
        outer = json.loads(stdout)
        if isinstance(outer, dict) and "result" in outer:
            result_text = outer["result"]
    except json.JSONDecodeError:
        pass

    obj = _extract_json(result_text)
    if obj is None:
        return {"model": model, "verdict": "unparsed", "raw": result_text[:1000]}
    obj["model"] = model
    if obj.get("verdict") not in {"pass", "mixed", "fail"}:
        obj["verdict"] = "mixed"
    return obj


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of model output (handles ``` fences and prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        candidates.append(text[i : j + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
