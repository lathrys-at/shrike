# `scripts/` — human-facing dev/maintenance entry points

`scripts/` holds the commands a **developer runs by hand**: environment setup,
the native-extension build, coverage runners, and dogfooding launchers. If a
human types it at a shell during development, it belongs here.

See the [`scripts/` vs `tools/` vs `bin/` boundary](../tools/README.md#the-boundary)
for the full rule. In short: `bin/` = shipped runnable entry points, `tools/` =
invoked by the build, `scripts/` = invoked by a human developer.

A script can produce output the build later consumes (e.g. the native `.so`,
the anki descriptor set) and still live here — what places it is that a
*developer* invokes it, not the build. By contrast a version-pin **lock** and
its regenerator live in [`tools/`](../tools/README.md) because the *build*
consumes the lock.

## Contents

### Environment & native build
- `dev-setup.sh` — the one-step, idempotent dev environment (venv, editable install, native build).
- `build-native.sh` — rebuilds the `shrike_native` extension into the venv (the pip-lane inner loop).
- `native-stale.sh` / `native-stamp.sh` — the staleness check + per-venv stamp that drive the auto-rebuild and the pytest backstop.
- `update-anki-descriptors.sh` — refreshes the checked-in anki protobuf descriptor set after an anki tag bump (writes into `shrike-core/`).

> The native-build trio (`build-native.sh`, `native-stale.sh`, `native-stamp.sh`)
> physically lives in `shrike-core/scripts/` — with the workspace it serves, so
> `shrike-core/` is a self-contained, subtree-extractable unit (#696). The entries
> here are relative symlinks back into `scripts/` so the familiar
> `scripts/build-native.sh` invocation still resolves.

### Coverage
- `coverage.sh` — the full local coverage run; enforces `fail_under`. Copies the
  single committed subprocess-capture hook (`tools/coverage_subprocess.pth`) into
  site-packages rather than carrying the hook string inline (#700).
- `coverage-bazel.sh` — the Bazel-lane coverage equivalent.

### Dogfooding launcher
- `serve.py` / `serve_test.py` / `serve.bzl` — the consolidated `//scripts:serve_<profile>` launcher (boots a real server against a fresh collection from a path-free capability profile; one per-profile target per profile, models assembled from the pinned externals at build time) and its logic test + the model-assembly macro.
- `profiles/` — the checked-in, path-free capability profiles `serve` reads.

### Packaging
- `package-skill.py` — bundles the `create-cards` skill into a `.skill` package (a symlink into the `shrike-skills/` unit, where the real file lives).

> The layout epic (#694) reshaped this directory: `serve` is canonically Bazel
> (#699), the skill packager moved into the `shrike-skills/` unit (#701), and the
> native-build trio moved to `shrike-core/scripts/` (symlinked back, above; #696).
> The dev/maintenance-scripts Bazel-ification (#700) came out narrow — the coverage
> `.pth` hook was de-duplicated into one committed source, but `sh_binary`/`genrule`
> conversion was declined (it would need a new `rules_shell` dep or touch the
> `shrike-core/` tree; see
> [`tools/README.md`](../tools/README.md#not-bazel-ified-and-why-700-verdict)).
