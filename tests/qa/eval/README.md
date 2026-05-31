# QA eval harness

Quantitative, repeatable evaluation of the **anki-cards** skill: feed each
scenario prompt to a cold weak agent (Haiku 4.5), with the skill and without it,
and grade what actually lands in the collection. Reusable as the skill evolves.

Manual — not part of `pytest`/CI. (The pure grader has its own quick test:
`pytest tests/qa/eval/test_grade.py`.)

## Pieces

| File | Role |
|---|---|
| `scenarios.yaml` | Machine spec: per-scenario `assert` block (deterministic checks) + `judge` rubric. Prompts are read from `../scenarios/<id>-*.md` (single source). |
| `grade.py` | Pure grader: `(run record, assert spec) → per-assertion pass/fail`. No I/O. |
| `prompts.py` | Canonical `with_skill` / `baseline` agent prompts (so runs are comparable). |
| `harness.py` | Glue CLI: `prompt`, `baseline`, `grade`, `report`. Talks to the running server. |
| `runs/` | Per-run artifacts (gitignored): `baseline.json`, `transcript.txt`, `run.json`, `grading.json`, `report.md`. |

## Matrix

6 scenarios × 2 configs (`with_skill`, `baseline`) × 3 repeats = 36 runs. Serial
only — one shared `server.lock`, and every run mutates the collection, so each
needs its own fresh fixture.

## The loop (per run)

Driven externally — currently an interactive Claude session spawns the agents
(the `Agent` tool with `model: haiku`); a `claude -p` `run.py` could automate it
later using the same prompts + harness steps.

```
B=tests/qa/eval/runs/<batch>; D=$B/<scenario>/<config>/r<n>

# 1. fresh fixture (clean 66-note corpus + index)
export LLAMA_SERVER_PATH=... SHRIKE_EMBEDDING_MODEL=...   # see ../README.md
scripts/launch-qa-server.sh

# 2. snapshot before
tests/qa/eval/harness.py baseline --out "$D"

# 3. get the exact prompt, run a COLD Haiku agent with it, save its final report:
tests/qa/eval/harness.py prompt --scenario <id> --config <config>
#    → (spawn the agent; write its final message to $D/transcript.txt)

# 4. capture what landed + grade
tests/qa/eval/harness.py grade --scenario <id> --dir "$D" --transcript "$D/transcript.txt"
```

Repeat for every cell (reset between each), then:

```
tests/qa/eval/harness.py report --batch "$B"      # → $B/report.md
```

## Grading

- **Mechanical (the gate)** — `grade.py` checks the `assert` block against the
  note-delta + transcript: deck placement, new decks, note types, card count,
  duplicates (nearest pre-existing cosine ≥ `dup_threshold`), required/forbidden
  tags, new-deck-flagged. Deterministic.
- **LLM judge (advisory)** — the `judge` rubric is handed to a grader agent that
  reads the created cards and rates the qualitative bits (best card type? atomic?
  recall-framed?). Doesn't gate; it's a sanity read alongside the numbers.

The `report` table shows run-level pass rate per scenario for `with_skill` vs
`baseline`, and the delta — the skill's measured lift over an unguided model.
