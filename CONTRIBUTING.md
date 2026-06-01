# Contributing to Shrike

This is the canonical reference for how work flows through Shrike: branching,
versioning, releasing, and how defects get captured. The agent-facing summary in
`CLAUDE.md` points here; this file is the source of truth.

## Branching model

Trunk-based. `main` is always releasable — no `develop` branch, no Gitflow.

- Every change lands through a short-lived branch and a pull request. No direct
  pushes to `main`.
- **Squash merge** PRs, so `main` keeps a linear, one-commit-per-change history
  that bisects cleanly.
- `main` is protected: the Linux CI checks (`lint`, `test`, `embedding`) must pass
  before merge. The expensive cross-platform lanes (macOS + ARM) run at merge time
  and on PRs labelled `rc` — see [`.github/workflows/test.yml`](.github/workflows/test.yml).
- Branch names are `‹type›/‹issue#›-‹slug›`, where `‹type›` is one of `feat`,
  `fix`, `docs`, `chore`, `refactor`, `test` — e.g. `fix/44-version-tag-drift`,
  `feat/33-ankiweb-sync`. The issue number ties the branch to its tracking issue.

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
- The package version currently lives in `src/shrike/__init__.py` (`__version__`),
  read by `pyproject.toml` via `[tool.hatch.version]`. **Bump it and tag in
  lockstep** until tag-derived versioning lands (#42), which removes the manual
  step and the drift risk entirely.

## Releasing

1. Bump `__version__` and update `CHANGELOG.md` (move `Unreleased` items under the
   new version heading).
2. Create an annotated tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` (add `-s` to sign).
3. Push the tag. A tag-triggered release workflow (#43) will, once built, run the
   full cross-platform suite, build sdist + wheel, and cut a GitHub Release. Until
   then, do the build/release step by hand.

## Issue tracking

The roadmap and all tracked work live in GitHub issues and milestones — not in
prose in the docs (that's how the old README/CLAUDE roadmaps drifted out of sync
with reality).

- One **milestone per minor version** (`v0.4.0 — Sync`, …).
- Each milestone has an **epic** tracking issue (label `epic`) whose body is a
  checklist of its deliverables. Fine-grained issues are broken out for the
  next-up version and linked from the epic; later versions stay as the epic until
  they come into focus.
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
   missing capability rather than a bug).
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

## Local checks before a PR

```bash
ruff check src/shrike/ tests/
ruff format --check src/shrike/ tests/
mypy src/shrike/
pytest tests/unit -q
pytest tests/integration -q -m "integration and not embedding"
```

See `CLAUDE.md` for the coverage-gate reproduction and the embedding/integration
suites that need a local `llama-server`.
