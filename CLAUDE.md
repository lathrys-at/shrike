# CLAUDE.md

Orientation for working on Shrike. This file is deliberately short; the deep
material lives in [`docs/dev/`](docs/dev/), one concern per file. When you need
detail, follow the links rather than expecting it here.

## What is Shrike?

Shrike manages Anki flashcard collections without running Anki's GUI. It exposes
Anki's collection operations through an MCP server and a `shrike` CLI, with
semantic search over notes computed locally.

License: AGPL-3.0.

## Architecture at a glance

Shrike is a **Rust compute core** (`shrike-core/`) wrapped by a **Python harness**
(`shrike-py/`), bridged by one compiled extension, `shrike_native`.

```
CLI (shrike)  ──HTTP/JSON-RPC──▶  MCP Server (server/ host; api/ verb surface)
                                      └──▶ Harness (assembly + verbs)
                                              └──▶ AsyncKernel (Rust, via shrike-pyo3)
                                                      ├── collection.anki2 (anki protobuf services)
                                                      ├── vector index (per-modality USearch)
                                                      ├── derived-text store (FTS5 sidecar)
                                                      ├── Embed slot  ◀── shrike-engine
                                                      └── Recognize slot ◀── shrike-engine
```

The kernel owns the collection, the indexes, and search fusion; it is a **plugin
host** that runs whatever embedder/recognizer the harness attaches. The full
picture — the plugin contracts, the tokio runtime, the action exchange, and the
load-bearing invariants — is in [`docs/dev/architecture.md`](docs/dev/architecture.md).

## Development

One-step, idempotent setup (re-run any time as a repair button):

```bash
scripts/dev-setup.sh        # creates .venv, installs shrike-py[dev], builds shrike_native
source .venv/bin/activate
```

The full local gate before a native change:

```bash
(cd shrike-core && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings)
(cd shrike-core && cargo test --workspace)
scripts/build-native.sh && pytest shrike-py/tests/unit shrike-py/tests/native -q
./bazel test //...      # the authoritative CI lane (Bazel is not on PATH; use ./bazel)
```

Lint and type-check (all three must be clean):

```bash
ruff check  shrike-py/src/shrike/ shrike-py/tests/ shrike-core/shrike-pyo3/python/
ruff format --check shrike-py/src/shrike/ shrike-py/tests/ shrike-core/shrike-pyo3/python/
mypy --config-file shrike-py/pyproject.toml shrike-py/src/shrike/
```

`pip install` does **not** build the native extension — that is a separate cargo
step, run for you by direnv or `scripts/build-native.sh`; `pytest` aborts loudly
if the `.so` is stale. Python 3.12 (managed via pyenv; `.python-version` at the
repo root). Coverage, the test-sharing model, and the Bazel lanes are in
[`docs/dev/testing.md`](docs/dev/testing.md).

## Conventions you must follow

- **Type annotations on every function** (mypy `disallow_untyped_defs`); **ruff**
  for lint + format, line length 100; **`raise ... from err`**;
  **`contextlib.suppress`** over bare `try/except/pass`; **`datetime.UTC`**.
- **Make illegal states unrepresentable** in `schemas.py` — correlated fields are a
  discriminated union, never a bag of optionals.
- **Ground performance at a 100k-note collection** — no collection reads inside
  per-item loops; one transaction per batch; never hold a lock across file writes.
- **One INFO log line per served call**; per-module loggers; `%s` formatting, not
  f-strings.

The detail and the reasoning are in [`docs/dev/conventions.md`](docs/dev/conventions.md).

## Documentation map

Developer docs — how the code works — live in [`docs/dev/`](docs/dev/):

| Doc | Covers |
|-----|--------|
| [`architecture.md`](docs/dev/architecture.md) | The Rust/Python split, the plugin kernel, the runtime, the action exchange. |
| [`layout.md`](docs/dev/layout.md) | Where every crate and package lives; the `scripts`/`tools`/`bin` boundary. |
| [`testing.md`](docs/dev/testing.md) | Dev setup, the suites, the native build, coverage, linting. |
| [`server-runtime.md`](docs/dev/server-runtime.md) | Collection lifecycle and locking, the transport trust boundary, the daemon, config. |
| [`embedding-and-recognition.md`](docs/dev/embedding-and-recognition.md) | The embedding service, its backends, and OCR/recognition. |
| [`indexing-and-search.md`](docs/dev/indexing-and-search.md) | Vector-index consistency, the derived-text sidecar, search fusion (RRF). |
| [`tools.md`](docs/dev/tools.md) | The 26 MCP tools: where they live and the behaviours to preserve. |
| [`conventions.md`](docs/dev/conventions.md) | Code style, the schema house style, performance rules, logging. |
| [`decisions.md`](docs/dev/decisions.md) | The "why" behind non-obvious choices, and the alternatives rejected. |
| [`build-bazel.md`](docs/dev/build-bazel.md) | The Bazel build graph, the two lanes, caching, the locks. |

Reference docs for *users and integrators* live at [`docs/`](docs/): the
[CLI reference](docs/cli-reference.md), the [MCP tool reference](docs/mcp-tools.md),
and the [distribution profiles](docs/distribution.md).

## Working in this repo

Trunk-based: every change goes through a `‹type›/‹issue#›-‹slug›` branch → PR →
squash merge, and the roadmap and all tracked work live in GitHub issues and
milestones, not in prose. The human-facing conventions — branching model,
versioning, releasing, repo settings — are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
What follows is the operating procedure for driving work to merge, and it is
must-read.

### Review and audit gates (mandatory)

These run *in addition* to the CI lint/test gates, and they are required, not
optional:

- **Code review on every significant change** before merge — anything that adds or
  changes behaviour (not trivial typo/doc/dep-bump PRs). Use `/code-review`
  (escalate to `ultra` for larger changes).
- **Security review whenever the server API surface changes** — a new or changed
  MCP tool or HTTP route, auth/transport/SSRF/path handling, anything touching the
  trust boundary. Run it in addition to the code review, via `/security-review`.
- **Before cutting a release**, run a fresh security audit and performance audit
  over the release candidate. Apply the `rc` label first so the cross-platform CI
  legs also run.

The billed `ultra`/cloud passes and the pre-release audits are **launched by the
user** — the agent can't start them. The agent's job is to surface that a change
crosses one of these thresholds and to act on the findings.

### The single-agent PR loop

For work the user has delegated, own the whole PR cycle and keep it pipelined
rather than serial.

1. **PR at each natural checkpoint, as a draft.** Open every PR with
   `--draft` (CI runs on it immediately — there is no `ci` gate). Don't contort
   in-progress work to make it mergeable when going a bit further lands a coherent
   section, but don't hoard mergeable work either.
2. **Self-review while it's still a draft.** Run a self-check review against the
   requirements — via a subagent (prefer the latest Opus model) — for substantive
   changes; skip it for mechanical ones. Keep working while the review is in flight.
3. **Mark ready, then auto-merge, move on.** Once findings are addressed and CI is
   green, take it out of draft (`gh pr ready`) and set it to merge when green
   (`gh pr merge --auto --squash`). Don't poll. Add a cross-platform label (`rc`,
   `macos`, or `linux-arm`) when the change warrants that coverage.

**Never enable auto-merge on a draft or before CI has run.** Subagents assist with
research, orientation, and review — never authorship. All code and tests are
written by the agent itself.

This composes with the gates above: the user-triggered passes stay user-triggered;
the agent's self-review is the floor, not a replacement, for anything crossing
those thresholds.

### Multi-agent team development

When the user points you at a milestone or a set of issues to develop in parallel,
you act as **team lead**: decompose, dispatch workers, keep boundaries intact,
review, and drive to a user-gated merge. You orchestrate; you don't implement the
issues yourself. The full playbook is the team-development skill; the load-bearing
rules:

- **Cap at 3–6 concurrent agents per wave**, sized to the set's natural parallelism
  (disjoint surfaces, no inter-dependency). More ready work than the cap → run in
  waves, each ending in a joint review and user-gated merge before the next.
- **Pre-flight:** read the governing design doc, fix the hard boundaries, build the
  real dependency graph (verify cross-stream state live with `gh`), and assign one
  owner per shared surface plus one lockfile owner per wave, so two agents never
  edit the same file.
- **Workers** run in their own worktree under `bypassPermissions` (a background
  agent can't answer a permission prompt), branch off `origin/main`, work in slices,
  run the full local gate, open a PR, report **"READY FOR REVIEW"**, then **HOLD**.
  In team mode workers do **not** self-review and **never** self-merge. Workers are
  expected to ask the lead for clarification, re-partition, or blockers.
- **Lead incremental review** (per PR, directly — not via a subagent): review each
  as it lands for alignment to the plan plus a correctness pass, verifying
  load-bearing claims against the code. Hold for the joint review.
- **Joint cross-review — the gate before any merge:** resume all authoring agents
  to peer-review each other's PRs along four axes: correctness/plan, performance,
  security, and cross-PR alignment.
- **Consolidate → user signoff → batch merge:** fold every review into one report
  (per-PR verdicts, must-fixes + owners, rebases, merge order). Nothing merges until
  the user's signoff. On the go, batch-merge in dependency order, then unblock
  downstream or start the next wave.

This joint review is the team review floor; it does not replace the user-triggered
`ultra`/cloud review or the pre-release audits.

### Defect workflow

When you hit a bug, a limitation, or a missing API surface that is **out of scope**
for the task in hand, do not silently fix it inline and do not leave it as a prose
note. Capture it as resumable state:

1. Open a GitHub issue with a clear problem statement (repro / expected vs actual,
   or the intended API surface that's missing).
2. Create a branch `fix/‹issue#›-‹slug›` (or `feat/…` for a missing capability;
   `xfail/‹issue#›-‹slug›` when you're only capturing the red reproducing spec).
3. Add failing test(s) that exercise the defect or pin the intended API, marked
   `@pytest.mark.xfail(strict=True, reason="#‹n›: …")` so the branch's CI stays
   green while the test is red-by-design, and a future fix that makes it pass forces
   the marker's removal.
4. Push the branch to origin and link it from the issue.

The failing test is the spec; the pushed branch is the handoff.

### Approved-plan check-in

When the user **approves** a plan (plan mode / ExitPlanMode), post it as a comment
on the GitHub issue it's homed in, so the issue carries the agreed design as
resumable state. End the comment with the same attribution line used on PR bodies.
Post only *approved* plans, never drafts, and **strip notes-to-self** before posting
— drop the process/meta sections (e.g. "Workflow updates", "Branch"); the comment
carries the substantive plan only.
