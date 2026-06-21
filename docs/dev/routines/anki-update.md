# Anki update

You are an autonomous maintenance agent for the **`lathrys-at/shrike`**
repository. You have been triggered because **Anki has published a new release**
— a stable, or a beta/rc we follow — and shrike's pin is behind. Your job has a
**hard, narrow scope**: move the Anki pin, and nothing else. Read this whole
file before doing anything.

## Why shrike tracks Anki closely (the policy)

Shrike's runtime *is* Anki's `rslib` backend, reached only through its protobuf
service layer (`../architecture.md`) and pinned by git tag. So we ride Anki's
release line deliberately:

- **Every Anki move is a shrike patch release** that bumps the pin and nothing
  else — the same posture as a routine dependency bump
  (`dependency-security-update.md`). A small bump each time Anki moves beats an
  annual multi-version slog (see the 25.09 → 26.05 catch-up, #943).
- **Follow Anki's pre-releases.** When Anki cuts a beta/rc, cut a shrike `-rc.N`
  against it (CONTRIBUTING.md → *Releasing*), so we are validated against Anki's
  release candidate *before* their stable lands — then ship our patch when their
  stable does.
- **The one exception** — a move that forces an API change on us — is *not* a
  routine patch. See *The API-change escape hatch*.

## How you were triggered (monitoring)

The `anki-watch` workflow (`.github/workflows/anki-watch.yml`, #944) polls Anki's
**GitHub Releases API** (`/repos/ankitects/anki/releases` — its `prerelease`
flag is what surfaces betas/rcs, not just stable) and cross-checks **PyPI**
(`https://pypi.org/pypi/anki/json`), compares against our pin, and opens or
updates a standing tracking issue when Anki is ahead. That issue is your trigger
and your claim anchor. A GitHub tag without a published PyPI wheel means the
runtime lane can move while the pip oracle waits for the wheel.

## Your mandate — and its limits

In order:

1. **Claim the work first** — the tracking issue *is* the claim (see
   *Claiming*), so parallel triggers never double-bump. If it is already
   claimed, stop.
2. **Make exactly the Anki version change across every pin and lockfile below —
   and only that.**
3. **Open a pull request**, get it green, and **merge it**.
4. **If the bump forces an API change, STOP and hand back** — that is the escape
   hatch, not your job here.

**You MUST NOT:** change source beyond a minimal mechanical compat shim;
refactor, reformat, or touch code unrelated to the bump; bump other dependencies
opportunistically; touch tests, CI, or docs beyond a `CHANGELOG.md` entry and
rebasing the Anki Bazel patches (below). One Anki move → one PR.

## Claiming (avoid duplicate work)

The watcher re-fires (a beta, then its stable), so two triggers can race. The
**tracking issue is the claim**; the claim key is the **target Anki version**.
Before creating anything, list open `deps` issues and open PRs/branches for one
already covering your target version — if a live claim exists, **stop**.
Otherwise claim it with a matchable title `[deps] anki <from> -> <to>`, then
re-list and, on a create-create race, concede to the **lowest issue number**
(the same lock-free tie-break as `dependency-security-update.md`).

## The Anki pin surface and its lockfiles

One Anki version, two lanes — they **must** move together.

| Lane | What it pins | Files | Regenerate |
|---|---|---|---|
| **Runtime** — correctness; the native Rust backend | the `anki` / `anki_proto` / `anki_i18n` crates + the protobuf descriptors | `shrike-core/runtime/shrike-collection/Cargo.toml` (git `tag`); `MODULE.bazel` (`anki_src` `strip_prefix` / `urls` + `sha256`); `shrike-core/third_party/anki/anki_descriptors.bin`; `shrike-core/patches/anki-*-bazel-*.patch` | `cargo update -p anki -p anki_proto`; `tools/update-bazel-lock.sh`; `scripts/update-anki-descriptors.sh` |
| **Test oracle** — parity; the pip `anki`, test-only since #278 | the Anki the cross-core oracle compares native output against | `requirements_lock.txt` (`anki==`); `shrike-py/pyproject.toml` + `shrike-py/BUILD.bazel` (`anki>=` floor, `dev` extra) | `tools/update-requirements.sh` |

Keep the oracle on the **same Anki version** as the runtime — the parity tests
compare the native backend against the pip backend, so a version mismatch tests
the wrong oracle.

**Version spelling.** Anki tags zero-pad the month (`25.09.4`, `26.05`); pip
normalizes (`25.9.4`, `26.5`). Use the **tag** spelling in the Rust/Bazel lane,
the **normalized** spelling in the pip lane.

## How to perform the bump

1. **Rust tag.** Set the new `tag` on `anki` + `anki_proto` in
   `shrike-collection/Cargo.toml`; set `anki_src`'s `strip_prefix` / `urls` in
   `MODULE.bazel` and recompute its `sha256`
   (`curl -sL <tarball-url> | sha256sum`). Then
   `(cd shrike-core && cargo update -p anki -p anki_proto)` (the transitive
   `anki_i18n` follows) and refresh the Bazel mirror with
   `tools/update-bazel-lock.sh`. Commit `shrike-core/Cargo.lock` **and**
   `MODULE.bazel.lock`.
2. **Rebase the Bazel patches.** The `anki-*-bazel-*.patch` files (proto
   buildrs / rustrs, i18n build / gather, version) are applied **only** by Bazel
   — the cargo lane builds the unpatched checkout — so a version jump that shifts
   their context will fail to apply. Re-roll any that no longer apply; do not
   widen their reach.
3. **Refresh descriptors.** Run `scripts/update-anki-descriptors.sh`, then
   **inspect the descriptor change**: a changed or removed service method we
   dispatch is an **API change, not a routine bump** — go to the escape hatch.
   (Dispatch indices are not build-derived yet, #394, so a silent service
   reordering is a live risk — scrutinize it.)
4. **Pip oracle.** Set `requirements_lock.txt` to the new normalized version via
   `tools/update-requirements.sh`; raise the `anki>=` floor only if a feature
   needs it.
5. **Gate + cross-platform.** Run the local gate (below) and apply the `rc` CI
   label so the macOS + Linux-ARM legs run — Anki carries platform-specific code.

## The API-change escape hatch (when it is NOT a routine patch)

If step 3 shows a service we dispatch changed or removed, or the cargo build,
the native tests, or the parity oracle break in a way a minimal shim cannot
absorb: **stop**. This is real work — a `0.x` minor while we are pre-1.0 — not a
tracking patch. Write up the blocker on the tracking issue (the Anki versions,
what broke, what a fix needs), leave the branch unpushed or as a draft, and hand
back. A clear handoff is success; an in-scope sprawling migration is failure.

## Verify before you open the PR (the local gate)

```bash
(cd shrike-core && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace)
scripts/build-native.sh && pytest shrike-py/tests/unit shrike-py/tests/native -q
CARGO_BAZEL_ISOLATED=0 ./bazel test //...
```

The parity oracles under `shrike-py/tests/native` drive the pip `anki` in
subprocesses and compare it against the native backend — they are the
load-bearing check that the new Anki behaves as the kernel expects. Never skip
them on an Anki bump.

## Open and merge the PR

1. Branch off the **latest `origin/main`**: `fix/<issue#>-anki-<version>`. Never
   commit onto an auto-generated `claude/*` branch. `<issue#>` is the claim.
2. Conventional-commit subject: `fix(deps): bump anki <from> -> <to>`. The body
   describes the change only. End the commit message with:
   `Co-Authored-By: Claude <noreply@anthropic.com>`
3. Open the PR **as a draft** (CI runs immediately). The body links the tracking
   issue (`Fixes #<issue#>`), names the lanes / lockfiles touched, and notes any
   rebased patches. End the PR body with:
   `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
4. **Self-review the diff against this scope.** A clean Anki bump is version
   strings + regenerated lockfiles + the refreshed descriptor `.bin` (+ at most
   rebased patches or a minimal shim). Larger than that ⇒ you have hit the escape
   hatch — stop and reassess.
5. Once reviewed and CI is green: `gh pr ready <pr#>` then
   `gh pr merge <pr#> --auto --squash`. Do not poll. Apply the `rc` label before
   tagging any release candidate against Anki's pre-release.

## Recap of the hard limits

- Claim before you work; concede to the lowest issue number on a race.
- One Anki move → one tracking issue → one PR. No bundling.
- Diff = version strings + lockfiles + descriptor `.bin` (+ rebased patches /
  minimal shim). Nothing else.
- A forced API change ⇒ stop, document on the issue, hand back. Do not migrate.
- Follow Anki's pre-releases: a shrike `-rc.N` against Anki's beta/rc, the patch
  when their stable lands.
