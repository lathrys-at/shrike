# Working in this repo

This is the operating procedure for an agent (or a contributor working like one)
developing changes here. The human-facing conventions — branching model,
versioning, releasing, issue tracking — live in
[`../../CONTRIBUTING.md`](../../CONTRIBUTING.md); this doc is the parts specific to
how work gets driven to merge.

## Review and audit gates (mandatory)

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

## The single-agent PR loop

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

## Multi-agent team development

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

## Defect workflow

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

## Approved-plan check-in

When the user **approves** a plan (plan mode / ExitPlanMode), post it as a comment
on the GitHub issue it's homed in, so the issue carries the agreed design as
resumable state. End the comment with the same attribution line used on PR bodies.
Post only *approved* plans, never drafts, and **strip notes-to-self** before posting
— drop the process/meta sections (e.g. "Workflow updates", "Branch"); the comment
carries the substantive plan only.
