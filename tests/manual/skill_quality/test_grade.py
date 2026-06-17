"""Unit tests for the pure eval grader. Run with: pytest tests/qa/eval/test_grade.py
(not part of the CI gate — QA tooling)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade import grade_run, summarize  # noqa: E402

BASELINE = {"decks": ["Biology", "Default", "History"], "note_count": 66}


def _run(created, transcript=""):
    return {"baseline": BASELINE, "created_notes": created, "transcript": transcript}


def _by_text(results):
    return {r["text"]: r for r in results}


def test_all_pass_for_clean_scenario01_like_run():
    spec = {
        "assert": {
            "decks_subset": ["Biology"],
            "new_decks_max": 0,
            "note_types_subset": ["Basic", "Cloze"],
            "new_cards": [3, 7],
            "duplicates_max": 0,
        }
    }
    created = [
        {
            "id": 1,
            "note_type": "Cloze",
            "deck": "Biology",
            "tags": ["biology"],
            "nearest_existing_score": 0.4,
        },
        {
            "id": 2,
            "note_type": "Basic",
            "deck": "Biology",
            "tags": ["biology"],
            "nearest_existing_score": 0.3,
        },
        {
            "id": 3,
            "note_type": "Basic",
            "deck": "Biology",
            "tags": ["biology"],
            "nearest_existing_score": None,
        },
    ]
    results = grade_run(_run(created), spec, dup_threshold=0.85)
    assert summarize(results)["failed"] == 0


def test_new_deck_and_subset_violation_fail():
    spec = {"assert": {"decks_subset": ["Biology"], "new_decks_max": 0}}
    created = [
        {"id": 1, "note_type": "Basic", "deck": "Music", "tags": [], "nearest_existing_score": None}
    ]
    r = _by_text(grade_run(_run(created), spec, 0.85))
    assert r["decks ⊆ ['Biology']"]["passed"] is False
    assert r["≤ 0 new deck(s)"]["passed"] is False


def test_duplicate_detection():
    spec = {"assert": {"duplicates_max": 0}}
    created = [
        {
            "id": 9,
            "note_type": "Basic",
            "deck": "Biology",
            "tags": [],
            "nearest_existing_score": 0.93,
        }
    ]
    r = _by_text(grade_run(_run(created), spec, 0.85))
    assert r["≤ 0 duplicate(s) of existing notes"]["passed"] is False


def test_forbidden_and_required_tags():
    spec = {"assert": {"tags_include": ["world-war-2"], "tags_forbidden": ["wwii", "ww2"]}}
    bad = [
        {
            "id": 1,
            "note_type": "Basic",
            "deck": "History",
            "tags": ["WWII"],
            "nearest_existing_score": None,
        }
    ]
    r = _by_text(grade_run(_run(bad), spec, 0.85))
    assert r["tags include ['world-war-2']"]["passed"] is False
    assert r["no forbidden tags ['wwii', 'ww2']"]["passed"] is False  # case-insensitive WWII

    good = [
        {
            "id": 1,
            "note_type": "Basic",
            "deck": "History",
            "tags": ["world-war-2"],
            "nearest_existing_score": None,
        }
    ]
    r = _by_text(grade_run(_run(good), spec, 0.85))
    assert r["tags include ['world-war-2']"]["passed"] is True
    assert r["no forbidden tags ['wwii', 'ww2']"]["passed"] is True


def test_new_deck_min_and_report_mention():
    spec = {"assert": {"new_decks_min": 1, "report_mentions_new_deck": True}}
    created = [
        {
            "id": 1,
            "note_type": "Basic",
            "deck": "Music Theory",
            "tags": [],
            "nearest_existing_score": None,
        }
    ]
    r = _by_text(
        grade_run(_run(created, "I created a new Music Theory deck since none fit."), spec, 0.85)
    )
    assert r["≥ 1 new deck(s)"]["passed"] is True
    assert r["report flags the new deck"]["passed"] is True

    r2 = _by_text(grade_run(_run(created, "Added some cards."), spec, 0.85))
    assert r2["report flags the new deck"]["passed"] is False
