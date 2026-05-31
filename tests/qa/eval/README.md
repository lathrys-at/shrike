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
| `judge.py` | Advisory LLM judge: builds a rubric+cards prompt, runs `claude -p` (Sonnet by default), parses a verdict. Never gates. |
| `prompts.py` | Canonical `with_skill` / `baseline` agent prompts (so runs are comparable). |
| `harness.py` | Glue CLI: `prompt`, `baseline`, `grade` (mechanical + judge), `report`. Talks to the running server. |
| `run.py` | **Automated runner**: loops `scenario × config × repeat`, resets the fixture, spawns the cold author via `claude -p`, grades + judges, writes the report. The hands-free counterpart to driving `harness.py` by hand. |
| `runs/` | Per-run artifacts (gitignored): `baseline.json`, `transcript.txt`, `author_stats.json`, `run.json`, `grading.json`, `report.md`. |

Each graded cell also prints an **author line** — tool calls, turns, and token
counts (in / out / cache) — parsed from the author's `claude -p` stream by
`run.py` (`author_stats.json`). It's the quickest read on what a config spends:
e.g. whether thinking buys better cards at the price of more output tokens.

## Matrix

6 scenarios × 2 configs (`with_skill`, `baseline`) × 3 repeats = 36 runs. Serial
only — one shared `server.lock`, and every run mutates the collection, so each
needs its own fresh fixture.

## Running it

**Automated (`run.py`)** — the whole matrix, hands-free:

```
export LLAMA_SERVER_PATH=... SHRIKE_EMBEDDING_MODEL=...   # see ../README.md

# 1x sweep, with_skill only, Sonnet judge (the defaults):
tests/qa/eval/run.py --repeats 1 --configs with_skill

# 3x depth on two scenarios, mechanical only (fast, no judge):
tests/qa/eval/run.py --scenarios 01,03 --repeats 3 --no-judge

# plain Haiku with no reasoning, to compare against the thinking default:
tests/qa/eval/run.py --author-thinking 0
```

Per cell it resets the fixture, waits for the index, snapshots the baseline,
spawns the cold author (`claude -p --model haiku`, the deliberately-weak model
the eval measures), grades + judges, and at the end writes `report.md`. Flags:
`--scenarios`, `--configs` (`with_skill`/`baseline`), `--repeats` (depth),
`--author-model`, `--author-thinking`, `--judge-model` (default `sonnet`),
`--judge-thinking`, `--no-judge`, `--keep-going`.

**Extended thinking is on by default** — author `8000` tokens, judge `4000`
(`MAX_THINKING_TOKENS`); `0` disables. Haiku 4.5 is weak but *can* reason, and
card design (type choice, dedup, cue framing) is reasoning work, so letting it
think is the cheap lever before reaching for a stronger author. `haiku`,
`haiku --author-thinking 0`, and `--author-model sonnet` are distinct configs;
each batch records which it was in `runs/<batch>/config.json`.

The author runs under `--dangerously-skip-permissions` — safe
because the QA collection is a disposable fixture rebuilt every cell, but it's an
autonomous agent loop, so run it deliberately.

**By hand (`harness.py`)** — when you want to drive the author yourself (e.g. an
interactive session spawning a sub-agent) and just use the deterministic steps:

```
B=tests/qa/eval/runs/<batch>; D=$B/<scenario>/<config>/r<n>
scripts/launch-qa-server.sh                                   # 1. fresh fixture
tests/qa/eval/harness.py baseline --out "$D"                  # 2. snapshot before
tests/qa/eval/harness.py prompt --scenario <id> --config <config>
#    → spawn a COLD Haiku agent with that prompt; write its final report to $D/transcript.txt
tests/qa/eval/harness.py grade --scenario <id> --dir "$D" --transcript "$D/transcript.txt"
tests/qa/eval/harness.py report --batch "$B"                  # after all cells → $B/report.md
```

`grade` runs the advisory judge too (Sonnet); add `--no-judge` to skip it or
`--judge-model <alias>` to swap models.

## Grading

- **Mechanical (the gate)** — `grade.py` checks the `assert` block against the
  note-delta + transcript: deck placement, new decks, note types, card count,
  duplicates (nearest pre-existing cosine ≥ `dup_threshold`), required/forbidden
  tags, new-deck-flagged. Deterministic.
- **LLM judge (advisory)** — `judge.py` hands the `judge` rubric and the actual
  created cards to a cold `claude -p` (Sonnet by default) that rates the
  qualitative bits (best card type? atomic? recall-framed?) and returns a
  `pass`/`mixed`/`fail` verdict with strengths and issues. It does **not** gate —
  stored in `grading.json` and surfaced as a separate `report` column, a sanity
  read alongside the mechanical numbers. A flaky/unparsable judge is recorded as
  `error`/`unparsed`, never sinking the run.

The `report` table shows mechanical run-level pass rate per scenario for
`with_skill` vs `baseline`, the delta — the skill's measured lift over an
unguided model — and the advisory judge's pass count.
