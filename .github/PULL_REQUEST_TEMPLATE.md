<!-- Title: imperative summary. Branch: ‹type›/‹issue#›-‹slug›. -->

Closes #<!-- issue number -->

## What & why

<!-- What this changes and the reasoning. -->

## Checklist

- [ ] Tests added/updated (and any `xfail` markers for fixed defects removed)
- [ ] `ruff check` / `ruff format --check` / `mypy` pass
- [ ] `CHANGELOG.md` updated under `Unreleased` (if user-visible)
- [ ] Docs updated (`CLAUDE.md` / `docs/` / `README.md`) if behaviour changed
- [ ] If `shrike-skills/**` changed: ran the QA eval and noted the `with_skill` vs `baseline` result above

## Release candidate

<!-- Required only when this PR carries the `rc` label (a release candidate, run
     before tagging vX.Y.Z). Delete this section otherwise. -->

- [ ] Full cross-platform CI suite green (linux + macOS + ARM)
- [ ] `CHANGELOG.md` section dated and the version decided
- [ ] QA run performed against a real client
- [ ] QA report filled in below (required before merging an `rc`)

### QA report

<!-- Client/harness used, scenarios exercised, pass/fail per scenario, and any
     anomalies or follow-up issues filed. -->

