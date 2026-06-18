# Contributing to Shrike

This is the canonical reference for how work flows through Shrike: branching,
versioning, releasing, and how defects get captured. The agent-facing summary in
`CLAUDE.md` points here; this file is the source of truth.

## Branching model

Trunk-based. `main` is always releasable — no `develop` branch, no Gitflow.

- Every change lands through a short-lived branch and a pull request. No direct
  pushes to `main`. **Open PRs as drafts** (`gh pr create --draft`) and take them
  out of draft only once complete and past initial review. A draft can't merge or
  auto-merge — the discipline that stops a PR landing before it's reviewed.
  (Enforced: CI always runs, so an *un-tested* PR can't merge. Review-before-merge
  is convention — the ruleset requires no approvals — so keep PRs drafts until
  reviewed.)
- **Squash merge** PRs, so `main` keeps a linear, one-commit-per-change history
  that bisects cleanly.
- `main` is protected: the single required status check is `ci-ok` (`CI passed`).
  **CI always runs on every PR** — there is no `ci` label gate (retired in #678);
  the suite runs on every push, including on drafts, and `CI passed` reports the
  real verdict. Because CI actually runs, a PR is pending/blocked until it goes
  green.
- The cross-platform lanes (macOS + Linux ARM) stay opt-in by label: `rc`
  (release candidate) selects **all** legs — apply it before tagging a release;
  `macos` selects just the macos-latest leg (for macOS-specific platform-glue
  changes); `linux-arm` selects just the ubuntu-24.04-arm leg. `rc` subsumes the
  per-leg labels. None run on plain PRs or on merge to `main` — the gating keeps
  the iterate-and-merge loop fast and the signal clean. See
  [`.github/workflows/test.yml`](.github/workflows/test.yml).
- Branch names are `‹type›/‹issue#›-‹slug›`, where `‹type›` is one of `feat`,
  `fix`, `docs`, `chore`, `refactor`, `test`, `xfail` — e.g.
  `fix/44-version-tag-drift`, `feat/33-ankiweb-sync`,
  `xfail/214-search-notes-neighbor-n1`. The issue number ties the branch to its
  tracking issue. An `xfail/` branch carries **only** a red-by-design reproducing
  test (no fix) — the handoff spec produced by the performance audit and the defect
  workflow below; whoever picks it up lands the actual fix on a `fix/` branch that
  flips the test green.

Release branches (`release/0.x`) are **not** used yet. They get cut reactively,
from the release tag, the first time an old version line genuinely needs a
backported fix — not before.

## Versioning

[Semantic Versioning](https://semver.org/). While in `0.x`, the public surface
(MCP tool schemas, CLI, config) may break between minor versions — `0.x` is the
license to iterate. `1.0.0` waits until that surface is stable enough to promise
backward compatibility; don't rush it.

- Released versions are annotated, ideally signed, git tags: `vMAJOR.MINOR.PATCH`.
- Real release candidates get pre-release tags: `vX.Y.Z-rc.1`, `-rc.2`, … (these
  sort before the final per SemVer). Distinct from the `rc` *CI label*, though
  they pair: label the PR `rc` to run cross-platform CI, cut a `-rc.N` tag for
  testers.
- The package version is **derived from the git tag** by hatch-vcs (config in
  `pyproject.toml`), not hand-maintained — so it can't drift from the tag. There's
  no `__version__` constant to bump: the build writes `shrike-py/src/shrike/_version.py`
  (gitignored) from `git describe`, and `shrike/__init__.py` re-exports it. Between
  releases the version is a dev version like `0.3.2.dev7+g1a2b3c4`; on a tagged
  commit it's the clean `0.3.2`. (CI checks out with `fetch-depth: 0` so the tag is
  visible at install time.)

## Releasing

The tag *is* the version (hatch-vcs, #42) — there's no constant to bump. The
tag-triggered workflow (`.github/workflows/release.yml`) runs the full
cross-platform suite, builds the **sdist + wheel**, the **`create-cards.skill`**
bundle (`scripts/package-skill.py`), and a **`SHA256SUMS`**, and attaches them to a
GitHub Release. Final-release notes come from the matching `## [X.Y.Z]` section of
`CHANGELOG.md`; an rc tag uses auto-generated commit notes instead. A final release
(not an rc) is also published to PyPI as `shrike-mcp`, via trusted publishing.

**Cut release candidates first, with the changelog left under `[Unreleased]`:**

1. Tag an rc on the current commit and push:
   `git tag -a vX.Y.Z-rc.N -m "vX.Y.Z-rc.N" && git push origin vX.Y.Z-rc.N`. It
   publishes as a GitHub **pre-release**. Iterate (`rc.2`, …) until one is stable —
   cut each off a **new commit** (a real fix), never by re-tagging an unchanged one.

**Then cut the final release:**

2. Roll up the changelog: open a PR moving the `[Unreleased]` items under a dated
   `## [X.Y.Z]` heading (leave a fresh empty `[Unreleased]`), and merge it.
3. Tag the **roll-up merge commit** and push:
   `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z` (signs automatically
   when `tag.gpgsign` is set).

Rolling up the changelog *after* the last rc — not before — is deliberate: it
finalizes the changelog at release time **and** guarantees the final tag lands on a
different commit than the rc. Two release tags on one commit make hatch-vcs ambiguous
(it can resolve to the rc version), which the workflow's version guard then rejects.

## Issue tracking

The roadmap and all tracked work live in GitHub issues and milestones — not in
prose in the docs (that's how the old README/CLAUDE roadmaps drifted out of sync
with reality).

- Each **milestone is a themed body of work** (*Search*, *Sync*, *Terminal UI
  (TUI)*, …), not a version number — what ships in a given release is decided
  at tag time.
- Each milestone has an **epic** tracking issue (label `epic`) whose body is a
  checklist of its deliverables. Fine-grained issues are broken out for the
  next-up work and linked from the epic; later work stays as the epic until
  it comes into focus.
- Shipped-design rationale lives in [`docs/decisions.md`](docs/decisions.md), not in
  closed issues.

## Defect & limitation workflow

When you find a defect, a limitation, or a missing API surface — whether you're
about to fix it or just spotted it out of scope — capture it as **executable,
resumable state**, not a mental note or a prose TODO:

1. **Open a GitHub issue** with a clear problem statement: how to reproduce,
   expected vs. actual, and scope — or, for a missing feature, the intended API
   surface and why it's wanted.
2. **Create a branch** `fix/‹issue#›-‹slug›` (or `feat/…` when the work is a
   missing capability rather than a bug). When you're only capturing the
   reproducing spec and not fixing it yet — as the weekly performance audit does —
   use `xfail/‹issue#›-‹slug›` instead, to signal "red spec, no fix attached".
3. **Add failing test(s)** that exercise the defect, demonstrate the limitation,
   or pin the intended API. Write the test asserting the *desired* behaviour and
   mark it `@pytest.mark.xfail(strict=True, reason="#‹n›: …")`. `strict=True`
   means the branch's own CI stays green (the failure is expected) while the test
   is red-by-design — and the day someone implements the fix, the now-unexpected
   pass makes CI fail until they remove the marker. That turns "the test went
   green" into the signal that the work is done. A plain failing test is fine for
   a fix you're actively finishing.
4. **Push the branch to origin** and link it from the issue. Anyone — a later
   session, another agent, a contributor — can then pick the work up from a branch
   that already encodes exactly what "fixed" means.

The point: a defect should never exist only as a sentence. The failing test *is*
the spec, and pushing the branch makes the handoff real.

## Repository settings (manual)

Branch protection on `main` (required status checks, no direct pushes, squash-only
merges) is configured in the GitHub repo settings, not in this repo's files. Set
it once when the project goes public; it can also be scripted later via `gh api`.

## Repository layout: `scripts/` vs `tools/` vs `bin/`

Three top-level directories hold non-package code. The line between them is **who
invokes it** — each carries a `README.md` with its full inventory:

- **`bin/`** — shipped/runnable product entry points: the Bazel `py_binary`
  launchers over `//shrike-py/src/shrike:shrike` (`//shrike-py/bin:shrike`,
  `//shrike-py/bin:server`, `//shrike-py/bin:server_embedding`). Load-bearing, kept
  outside the `shrike` package so a
  binary's output path never collides with a package subdir. Not cruft.
- **`tools/`** — invoked by the build: Bazel macros (`//tools/bazel`), the
  version-pin locks + their writers/checkers, the sdist/wheel/requirements
  builders, `workspace_status.sh`, the hermetic-toolchain CI smoke tests.
- **`scripts/`** — human-facing dev/maintenance entry points: `dev-setup.sh`, the
  native build, the coverage runners, the `//scripts:serve_<profile>` launcher.

A file follows its strongest coupling. A *version-pin lock* is consumed by the
build (Bazel reads it, CI cache keys hash it, a `py_test` validates it), so the
lock, its regenerator, and its tripwire all live in `tools/` — e.g. the llama-lock
trio `tools/llama-server.lock` + `tools/update-llama-lock.sh` +
`tools/check_llama_lock.py`, beside the sibling `tools/bazel.lock` build-pin. A
script a developer runs by hand stays in `scripts/` even when its output feeds a
build (e.g. `build-native.sh`).

## Local checks before a PR

```bash
ruff check shrike-py/src/shrike/ shrike-py/tests/ shrike-core/shrike-pyo3/python/
ruff format --check shrike-py/src/shrike/ shrike-py/tests/ shrike-core/shrike-pyo3/python/
mypy --config-file shrike-py/pyproject.toml shrike-py/src/shrike/
pytest shrike-py/tests/unit -q
pytest shrike-py/tests/integration -q -m "integration and not embedding"
```

For a change touching the Rust workspace, also run the native gate:

```bash
(cd shrike-core && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings)
(cd shrike-core && cargo test --workspace)
scripts/build-native.sh && pytest shrike-py/tests/unit shrike-py/tests/native -q
./bazel test //...     # the authoritative CI lane
```

`pip install` doesn't build the `shrike_native` extension (a separate cargo step
in `scripts/build-native.sh`), so a stale `.so` used to slip into a pytest run
silently after a `shrike-core/` change. That's structurally closed now (#573): run
`scripts/dev-setup.sh` once for the whole setup, then rebuilds are automatic with
direnv (`.envrc` rebuilds a stale extension on `cd`) and `pytest` fails loud
(before importing the extension) if you skip the rebuild without direnv. The
staleness check is `scripts/native-stale.sh`; `SHRIKE_SKIP_NATIVE_STALE_CHECK=1`
bypasses the pytest backstop.

Development is deliberately two-lane: iterate on the **pip lane** (the venv,
`scripts/build-native.sh`, `pytest` with its debugging affordances), and let
the **Bazel lane** be the proof — CI runs one `bazel test` invocation over the
whole graph, and releases are Bazel-built. The committed `./bazel` wrapper
bootstraps the pinned toolchain (no install). [`docs/build-bazel.md`](docs/build-bazel.md)
covers the targets, regenerating the dependency locks, and how caching works;
the design rationale is the Bazel ADR in [`docs/decisions.md`](docs/decisions.md).
An optional `.envrc` auto-activates the venv (and keeps the native extension
fresh) if you use direnv — nothing requires it.

`scripts/coverage.sh` runs the coverage measurement locally and enforces the
`fail_under` target (the per-PR CI lane runs tests *without* the tracer for speed,
and CI only ever *reports* coverage — rc PRs + 3x/week on `main` — it never gates on
it). So check it locally as you work. See `CLAUDE.md` for the details and the
embedding/integration suites that need a local `llama-server`.

### Faster inner loop (optional)

While iterating, `pytest --testmon` (from `pytest-testmon`, in the `dev` extra)
runs only the tests affected by your uncommitted changes — it tracks which code
each test exercises in a local `.testmondata` (gitignored) and skips the rest.

It's a **local convenience only**, never run in CI, for two reasons: CI always
runs the full suite (the per-PR correctness gate, and the rc/scheduled coverage
run), and impact analysis can produce false negatives — it can miss a test whose
dependency it didn't capture, notably the
integration suite, which drives a *separate* server subprocess that testmon's
in-process tracking can't see. So always do a full `pytest` run before you push.

## Skill changes (QA eval)

The `create-cards` skill (`shrike-skills/create-cards/**`) is **not covered by `pytest`/CI** —
only the pure grader (`shrike-py/tests/manual/skill_quality/test_grade.py`) runs there. A change to the
skill's guidance or references can pass every CI check while silently regressing
card quality, so the regression guard is the manual eval harness, not the test
suite. (For scale: rewording one rule from an abstraction into a surface check
moved a weak-model scenario from 1/5 to 5/5 — the kind of swing CI can't see.)

When you change skill definitions, **run the QA eval and note the `with_skill`
result (vs `baseline`) in the PR.** It needs a local `llama-server` + `claude -p`
and costs real tokens, so it can't be a CI gate — it's expected practice, not a
hard merge blocker.

```bash
shrike-py/tests/manual/skill_quality/run.py   # see shrike-py/tests/manual/skill_quality/EVAL.md for flags
```

Scope is your judgment: run the scenarios the change plausibly affects (a targeted
`--scenarios …` sweep is fine for a narrow edit; a fuller matrix before a release).
The harness, scenarios, and grading are documented in
[`shrike-py/tests/manual/skill_quality/EVAL.md`](shrike-py/tests/manual/skill_quality/EVAL.md).
