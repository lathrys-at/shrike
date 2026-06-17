"""Deterministic grader for QA eval runs.

Pure functions: given a *run record* (what an agent actually did, captured from
the collection) and a scenario's ``assert`` spec, return one result per
assertion. No I/O, no server — so it's unit-testable in isolation. The
server-touching capture lives in ``harness.py``.

Run record shape (see harness.py):

    {
      "scenario_id": "01",
      "config": "with_skill",
      "repeat": 1,
      "baseline": {"decks": [...], "note_count": 66, "timestamp": "..."},
      "created_notes": [
        {"id": 1, "note_type": "Cloze", "deck": "Biology",
         "tags": ["biology", "cardiology"], "front": "...",
         "nearest_existing_score": 0.42}  # vs PRE-EXISTING notes, or None
      ],
      "transcript": "the agent's final report text"
    }

Each result is ``{"text": ..., "passed": bool, "evidence": ...}`` (the field
names a reviewer/viewer expects).
"""

from __future__ import annotations

import re
from typing import Any

DEFAULT_DUP_THRESHOLD = 0.85


def grade_run(
    run: dict[str, Any], spec: dict[str, Any], dup_threshold: float
) -> list[dict[str, Any]]:
    created = run.get("created_notes", [])
    baseline_decks = set(run.get("baseline", {}).get("decks", []))
    transcript = run.get("transcript", "") or ""
    a = spec.get("assert", {})
    results: list[dict[str, Any]] = []

    def add(text: str, passed: bool, evidence: Any) -> None:
        results.append({"text": text, "passed": bool(passed), "evidence": evidence})

    decks_used = sorted({n["deck"] for n in created})
    types_used = sorted({n["note_type"] for n in created})
    new_decks = sorted(set(decks_used) - baseline_decks)
    all_tags = sorted({t.lower() for n in created for t in n.get("tags", [])})

    if "decks_subset" in a:
        allowed = set(a["decks_subset"])
        bad = [d for d in decks_used if d not in allowed]
        add(f"decks ⊆ {a['decks_subset']}", not bad, f"used={decks_used} unexpected={bad}")

    if "new_decks_max" in a:
        add(
            f"≤ {a['new_decks_max']} new deck(s)",
            len(new_decks) <= a["new_decks_max"],
            f"new_decks={new_decks}",
        )

    if "new_decks_min" in a:
        add(
            f"≥ {a['new_decks_min']} new deck(s)",
            len(new_decks) >= a["new_decks_min"],
            f"new_decks={new_decks}",
        )

    if "note_types_subset" in a:
        allowed = set(a["note_types_subset"])
        bad = [t for t in types_used if t not in allowed]
        add(
            f"note types ⊆ {a['note_types_subset']}", not bad, f"used={types_used} unexpected={bad}"
        )

    if "note_types_include" in a:
        missing = [t for t in a["note_types_include"] if t not in types_used]
        add(
            f"note types include {a['note_types_include']}",
            not missing,
            f"used={types_used} missing={missing}",
        )

    if "new_cards" in a:
        lo, hi = a["new_cards"]
        n = len(created)
        add(f"new cards ∈ [{lo}, {hi}]", lo <= n <= hi, f"created={n}")

    if "duplicates_max" in a:
        dups = [
            n
            for n in created
            if n.get("nearest_existing_score") is not None
            and n["nearest_existing_score"] >= dup_threshold
        ]
        dup_list = [(n["id"], round(n["nearest_existing_score"], 2)) for n in dups]
        add(
            f"≤ {a['duplicates_max']} duplicate(s) of existing notes",
            len(dups) <= a["duplicates_max"],
            f"dups={dup_list} (thr={dup_threshold})",
        )

    if "tags_include" in a:
        missing = [t for t in a["tags_include"] if t.lower() not in all_tags]
        add(f"tags include {a['tags_include']}", not missing, f"tags={all_tags} missing={missing}")

    if "tags_forbidden" in a:
        present = [t for t in a["tags_forbidden"] if t.lower() in all_tags]
        add(f"no forbidden tags {a['tags_forbidden']}", not present, f"forbidden_present={present}")

    if a.get("report_mentions_new_deck"):
        # Heuristic: the report should say it created/added a new deck.
        m = re.search(
            r"(new|creat\w+)[^.\n]{0,40}\bdeck\b|\bdeck\b[^.\n]{0,40}(creat\w+|new)",
            transcript,
            re.I,
        )
        add(
            "report flags the new deck",
            m is not None,
            m.group(0) if m else "(no new-deck mention found)",
        )

    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(1 for r in results if r["passed"])
    return {"passed": passed, "total": len(results), "failed": len(results) - passed}
