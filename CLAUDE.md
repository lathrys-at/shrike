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

**In a git worktree, run `scripts/dev-setup.sh` first, before anything else.** It
roots at the checkout you stand in, so it builds a `.venv` and native extension
*local to that worktree* and prints the absolute `source …/.venv/bin/activate` for
it. Activate that one. Never reuse, activate, or build into another checkout's
`.venv`: a shared venv cross-wires the native extension and its staleness stamp
between worktrees, so `pytest` silently imports the wrong `.so`. The tooling now
refuses a cross-checkout venv loudly rather than corrupting it.

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

Enforced, not aspirational: conformance is a required check in every code review (see
[Working in this repo](#working-in-this-repo)), so a breach is a review finding.

### Style

- **Type annotations on every function** (mypy runs with `disallow_untyped_defs`).
- **Ruff** for linting (rules E, F, W, I, UP, B, SIM) and formatting, line length 100.
- **`raise ... from err`** in except blocks (ruff B904).
- **`contextlib.suppress`** instead of bare `try/except/pass`.
- **`datetime.UTC`**, not `timezone.utc` (ruff UP017).
- Batch operations use **per-item try/except** so one failure doesn't block the batch;
  results carry a per-item `status`.

### Comments document the code, not its history

Comments document the code; **rationale and history live elsewhere** — in issues, PR
bodies, and [`docs/dev/decisions.md`](docs/dev/decisions.md). `git blame` already traces
when and why a line changed, so a comment that re-states that goes stale the moment it is
written ("when was *recently*?").

**Keep** a comment only if it is:

- **future-facing code documentation** — a non-obvious invariant, a
  looks-wrong-but-correct-because-X guard, a `// SAFETY:` justification, or algorithmic
  rationale a future reader genuinely needs; or
- **interface documentation** — a docstring on a public module, class, or function (the
  contract).

**Drop** the historical/contextual narrative: `(#NNN)` issue citations, "changed in the X
pass", "as of today / for now", and session-generated context-vomit.

The test is the comment's *value*, not a keyword. A comment that explains the code stays
even if it happens to cite an issue — strip the historical scaffolding (the `(#NNN)`, the
"as of" framing) and keep the explanation. A comment whose value is *purely* historical
drops entirely. The same "no historical narrative" trim applies *within* an interface
docstring. It is judgment per comment, never a blunt regex sweep — a naive "delete every
line with `#NNN`" would destroy genuinely valuable explanations (the `// SAFETY:` blocks,
the CVE-class rationale in path handling).

### Write for a developer, not a session

Comments, commit messages, issue bodies, and PR descriptions are read by **human
developers who lack your session's context but know the codebase**. Write to that
audience, always — never to the person driving a coding session:

- **No session voice.** Never address "you" the prompter, never narrate what an agent did
  this session, and never quote or paraphrase the conversation with a user ("as we
  discussed", "per your request", "the user wanted…"). The reader wasn't there; that
  framing is noise to them.
- **Direct and terse.** State what is true and why it's non-obvious, in as few words as it
  takes. Cut filler, hedging, throat-clearing, and anything that just restates the code.
- **No context-vomit.** Don't disgorge the reasoning trail, the alternatives tried this
  session, or a play-by-play. The durable *why* goes in
  [`docs/dev/decisions.md`](docs/dev/decisions.md); the rest is not written down.

Trust the reader to read the code. Tell them the part they can't infer, plainly.

### Schema house style: make illegal states unrepresentable

The wire models in `schemas.py` follow one rule: when a field's presence is *correlated*
with another (a hidden state), model it as a **discriminated union**, never a bag of
optionals.

The pattern is an `Annotated` type alias. Each variant is a `BaseModel` with a `Literal`
discriminator field, and:

```python
Thing = Annotated[VariantA | VariantB, Field(discriminator="status")]
```

Validate the alias with `TypeAdapter(Thing).validate_python(...)` (a model *field* typed
as `Thing` validates automatically). Examples: per-item results (`UpsertNoteResult` —
success has `id` + `neighbors`, error has `index` + `error`), `IndexStatus`
(`IndexUnavailable | IndexBuilding | IndexReady | IndexErrored`), and the `/index/rebuild`
+ `/embedding/*` endpoint responses.

Two fields that always appear or vanish *as a pair* are the same smell at smaller scale —
group them into a nested sub-model, not two optionals. A bare `X | None` is reserved for
*genuinely independent* optionality (a datum absent on its own); annotate why, so it reads
as deliberate.

Response models carry **no `error` field**. A whole-call failure (bad input, unhandled
exception) is raised in the tool and surfaces as an MCP `isError` result, which
`ShrikeClient._call` turns into a `ServerError`. Expected bad input raises `ToolInputError`
(logged without a traceback); genuine bugs log with one. The only optional advisory on a
success response is `message`.

Input bounds (`limit` 1–200, `top_k` 1–50, batch sizes ≤100/≤10) are declared as
`Annotated[..., Field(ge=, le=, min_length=, max_length=)]` on the tool params, so FastMCP
**rejects** out-of-range input rather than silently clamping. Optional list filters use
`Field(default_factory=list)` so they render as a plain array, not a noisy
`anyOf:[array, null]`.

### Performance

Ground performance decisions at a **100k-note collection**. These rules came out of the
kernel performance audit; the recurring failure modes and the rules that prevent them:

- **No collection reads inside per-item loops.** The N+1 is the repeat offender: a
  singleton `note_dicts`/`note_texts` per candidate pays two SQL queries plus a full
  deck/notetype enumeration each, serialized on the collection actor. Discover the id set
  first, then do ONE batched read (`read_notes_batch`, `note_dicts(&ids)`,
  `texts_for_source_for_notes`) and assemble from the map. **When porting policy between
  layers, port its batching with it.**
- **Read only what the op needs.** Prefer scoped variants over full-collection renders
  (`note_image_refs`, the `any_tagged` probe, notes-scoped derived reads). Push a
  pre-filter into SQL only when its semantics match the Rust side exactly.
- **Per-op tails do no O(collection) work.** Derived signals (tag centroids) refresh in a
  coalescing background task behind a cheap relevance probe; the op tail only *requests*.
  Boot/rebuild paths keep synchronous refreshes so "ready" means ready.
- **Never hold a lock across file writes or compute.** Snapshot the small shared state
  under the lock, write outside it; serialize savers with a dedicated guard; blocking fs
  work rides `spawn_blocking`.
- **One transaction per batch; prepared statements in row loops.** A journal commit (fsync)
  per item is the hidden cost — `ingest_many` and `set_note_tags_bulk` batch it away.
  `Connection::execute` re-prepares per call, so loops use `prepare_cached`.
- **Skip provably-identity work — but prove it from the pinned source.** The strip-skip (no
  `<` and no `&` → Anki's stripper is byte-identity) was verified against Anki's own gate
  and is pinned by a test. A skip predicate justified only empirically is a future
  correctness bug.
- **Hand out views and `Arc`s, not clones; bound unbounded expansions.** Arc'd
  per-notetype field lists, `Cow` pass-throughs, per-batch lookup memos, and ceilings with
  deterministic sampling where an input scales with the collection.

### Logging

Logging is configured in `platform/log.py`. Format, parsing, and styling all live there —
formatting knowledge should not spread across CLI commands.

Use **per-module loggers**: `shrike.server`, `shrike.kernel`, `shrike.tools`,
`shrike.collection`, `shrike.embedding`, `shrike.index`, `shrike.derived`, `shrike.daemon`.
This is what makes per-logger level overrides (`logging.levels.shrike.collection: debug`)
work. Never log under a bare `shrike` logger. Native (Rust) tracing forwards through
pyo3-log under the crate's module path, so the same overrides govern it.

Principles:

1. **Say what happened, with the key context.** "Collection ready: 847 notes, 5 decks, 12
   note types", not "Collection opened". Include counts, IDs, paths, durations.
2. **Log operational boundaries at INFO** — startup, shutdown, configuration loaded, server
   listening. These are the anchors you scan for.
3. **One INFO line per served call.** The adapter (`_safe_tool`) emits it: tool name +
   params + outcome + duration, e.g. `list_notes deck='Test' limit=50 -> 3/3 notes (12ms)`.
   Actions contribute the outcome fragment via `note_outcome(...)`; custom routes get the
   same from `_guard`. Anything else logged while serving is WARNING/ERROR (exceptional) or
   DEBUG (internals) — never a second INFO line.
4. **DEBUG for internals** — individual note creates, query construction, index lookups.
5. **WARNING for recoverable failures that deserve attention** — a single note failing in a
   batch, a rejected note-type update. Not normal empty results.
6. **`%s` formatting, not f-strings** — lazy evaluation.
7. **Don't repeat what the logger name says.** The line already shows `shrike.tools`; don't
   prefix the message with "tools:".
8. **Log the signal name on shutdown**, so you know whether it was SIGTERM (normal), SIGINT
   (Ctrl+C), or something else.

The log file format is fixed (`parse_log_line` / `style_log_line` know it):

```
2025-05-24T10:30:00 INFO  shrike.tools  list_notes deck=Test limit=50
```

Timestamp is `%Y-%m-%dT%H:%M:%S` (19 chars), level left-padded to 5 chars, logger and
message separated by a double space. Keep the parser and styler in sync if you change it.

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
  PRs) — `/code-review`, escalate to `ultra` for larger changes. **Conformance to the
  [Conventions](#conventions-you-must-follow) above is a required check** — style, the
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
   conformance to the [Conventions](#conventions-you-must-follow). Keep working while it's
   in flight.
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
  along correctness/plan, convention adherence ([Conventions](#conventions-you-must-follow)),
  performance, security, and cross-PR alignment.
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
