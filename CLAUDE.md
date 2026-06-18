# CLAUDE.md

Orientation for working on Shrike. Kept short on purpose; the depth is in
[`docs/dev/`](docs/dev/), one concern per file — follow the links.

## What is Shrike?

Shrike manages Anki collections without Anki's GUI, exposing Anki's collection
operations through an MCP server and a `shrike` CLI, with semantic search over notes
computed locally. License: AGPL-3.0.

## Architecture at a glance

A **Rust compute core** (`shrike-core/`) wrapped by a **Python harness**
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

The kernel owns the collection, the indexes, and search fusion; it is a **plugin host**
that runs whatever embedder/recognizer the harness attaches. The full picture — plugin
contracts, the tokio runtime, the action exchange, the load-bearing invariants — is in
[`docs/dev/architecture.md`](docs/dev/architecture.md).

## Development

```bash
scripts/dev-setup.sh        # idempotent: .venv, shrike-py[dev], builds shrike_native
source .venv/bin/activate
```

The full local gate before a native change:

```bash
(cd shrike-core && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings)
(cd shrike-core && cargo test --workspace)
scripts/build-native.sh && pytest shrike-py/tests/unit shrike-py/tests/native -q
./bazel test //...      # authoritative CI lane (Bazel isn't on PATH — use ./bazel)
```

Lint + type-check (all three clean):

```bash
ruff check  shrike-py/src/shrike/ shrike-py/tests/ shrike-core/bindings/shrike-pyo3/python/
ruff format --check shrike-py/src/shrike/ shrike-py/tests/ shrike-core/bindings/shrike-pyo3/python/
mypy --config-file shrike-py/pyproject.toml shrike-py/src/shrike/
```

`pip install` does **not** build the native extension (a separate Bazel build via
`scripts/build-native.sh`, also run for you by direnv; `pytest` aborts if the `.so` is
stale). Python 3.12 via pyenv. Coverage, the test-sharing model, and the Bazel lanes are in
[`docs/dev/testing.md`](docs/dev/testing.md).

## Conventions you must follow

- **Type annotations on every function** (mypy `disallow_untyped_defs`); **ruff** lint +
  format, line length 100; **`raise ... from err`**; **`contextlib.suppress`** over bare
  `try/except/pass`; **`datetime.UTC`**.
- **Make illegal states unrepresentable** in `schemas.py` — correlated fields are a
  discriminated union, never a bag of optionals.
- **Ground performance at a 100k-note collection** — no collection reads in per-item
  loops; one transaction per batch; never hold a lock across file writes.
- **One INFO log line per served call**; per-module loggers; `%s` formatting, not
  f-strings.
- **Comments document the code, not its history** — keep future-facing code docs
  (invariants, `// SAFETY:`) and interface docstrings; `(#NNN)`/"as of today"
  history goes in issues, PRs, and `decisions.md`.
- **Write for a developer, not a session** — code comments, commit messages, issues,
  and PR bodies are for human developers who lack session context but know the code:
  direct, terse, no "Claude-in-a-session" voice, no addressing the prompter, no
  quoting the user conversation, no context-vomit.

The reasoning is in [`docs/dev/conventions.md`](docs/dev/conventions.md).

## Documentation map

Developer docs — how the code works — in [`docs/dev/`](docs/dev/):

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

Reference docs for users and integrators are at [`docs/`](docs/): the
[CLI reference](docs/cli-reference.md), the [MCP tool reference](docs/mcp-tools.md), and
the [distribution profiles](docs/distribution.md).

## Working in this repo

Trunk-based: every change goes through a `‹type›/‹issue#›-‹slug›` branch → PR → squash
merge; the roadmap and tracked work live in GitHub issues and milestones. Human
conventions (branching, versioning, releasing, repo settings) are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). The rest of this section is the operating procedure
for driving work to merge — must-read.

### Review and audit gates (mandatory)

In addition to CI lint/test, and required:

- **Code review on every behavioural change** before merge (skip trivial typo/doc/dep
  PRs) — `/code-review`, escalate to `ultra` for larger changes. **Conformance to
  [`docs/dev/conventions.md`](docs/dev/conventions.md) is a required check** — style, the
  schema house style, performance rules, logging, and comment discipline. A convention
  breach is a review finding, not a nit.
- **Security review whenever the server API surface changes** — a new/changed MCP tool or
  HTTP route, anything touching auth/transport/SSRF/path handling — via `/security-review`,
  on top of the code review.
- **Before a release**, a fresh security + performance audit over the RC; apply the `rc`
  label first so the cross-platform legs run.

The `ultra`/cloud passes and pre-release audits are **launched by the user**; the agent
surfaces that a change crosses a threshold and acts on the findings.

### The single-agent PR loop

For delegated work, own the whole cycle, pipelined not serial:

1. **PR at each checkpoint, as a draft** (`--draft`; CI runs immediately). Don't contort
   in-progress work to be mergeable, but don't hoard mergeable work either.
2. **Self-review while still a draft** — a subagent self-check (latest Opus) for
   substantive changes, skipped for mechanical ones, covering correctness *and*
   conformance to [`docs/dev/conventions.md`](docs/dev/conventions.md). Keep working while
   it's in flight.
3. **Mark ready, auto-merge, move on** — once findings are addressed and CI is green,
   `gh pr ready` then `gh pr merge --auto --squash`; don't poll. Add a cross-platform
   label (`rc`/`macos`/`linux-arm`) when warranted.

**Never auto-merge on a draft or before CI has run.** Subagents assist with research and
review — never authorship; all code and tests are the agent's own.

### Multi-agent team development

When pointed at a milestone or issue set to develop in parallel, act as **team lead** —
decompose, dispatch, keep boundaries, review, drive to a user-gated merge; orchestrate,
don't implement. The full playbook is the team-development skill; the load-bearing rules:

- **Cap at 3–6 agents per wave**, sized to disjoint, independent surfaces; more ready work
  than the cap runs in waves, each ending in a joint review and user-gated merge.
- **Pre-flight**: read the design doc, fix the boundaries, build the real dependency graph
  (verify live with `gh`), assign one owner per shared surface and one lockfile owner per
  wave — no two agents edit the same file.
- **Workers** run in their own worktree under `bypassPermissions`, branch off
  `origin/main`, work in slices, run the full gate, open a PR, report **"READY FOR
  REVIEW"**, then **HOLD** — no self-review, no self-merge; ask the lead when blocked.
- **Lead incremental review** per PR (directly): alignment to plan + a correctness pass.
- **Joint cross-review is the gate before any merge** — resume all authors to peer-review
  along correctness/plan, convention adherence (`docs/dev/conventions.md`), performance,
  security, and cross-PR alignment.
- **Consolidate → user signoff → batch merge** in dependency order; nothing merges before
  signoff.

This joint review is the team floor; it doesn't replace the user's `ultra`/cloud pass or
the pre-release audits.

### Defect workflow

A bug, limitation, or missing API **out of scope** for the task in hand isn't fixed inline
or left as a prose note — capture it as resumable state:

1. Open an issue with a clear problem statement (repro / expected vs actual).
2. Branch `fix/‹issue#›-‹slug›` (or `feat/…`; `xfail/…` for a red reproducing spec).
3. Add failing test(s) marked `@pytest.mark.xfail(strict=True, reason="#‹n›: …")`, so CI
   stays green while red-by-design and a future fix forces the marker's removal.
4. Push and link from the issue.

The failing test is the spec; the pushed branch is the handoff.

### Approved-plan check-in

When the user **approves** a plan, post it as a comment on its home issue (ending with the
PR attribution line) so the issue carries the agreed design. Post only approved plans, and
**strip notes-to-self** (process/meta sections) first — the substantive plan only.
