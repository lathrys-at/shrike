"""pytest_test — run pytest file(s) as one Bazel py_test (#244).

Wraps rules_python's py_test with the shared pytest launcher
(//tools/bazel:pytest_runner.py), so the files run under `bazel test` the same way
they do under plain pytest. All files in `srcs` run in ONE test action (one
process), so the heavy anki/usearch/pydantic import is paid once, not per file,
and `-n auto` (xdist) parallelizes across cores — mirroring the pip
`pytest -n auto` lane. Always pulls in pytest + pytest-asyncio + pytest-xdist and
the shrike library; a target needing more (a genuine third-party wheel) adds it
via `deps`, and the package's conftest goes in `deps` too (so pytest finds it in
runfiles).

`xdist` sets pytest's `-n` worker count (default "auto", matching the pip
`pytest -n auto` lane). Override per-target with a specific count (`xdist = "4"`
or `xdist = 4`), or run serially with `xdist = None` (also `0` / `False`) — e.g.
for an order-dependent target or one you want to step through under a debugger.

Every other default is overridable per-target the same way: `size` (default
"small"), `deps` / `data` / `args`, and any other `py_test` attribute
(`tags`, `timeout`, `flaky`, `env`, …) passed through via `**kwargs`. The only
non-negotiable deps are the launcher + pytest/pytest-asyncio/pytest-xdist;
the code under test comes from the caller's `deps` (since #259 the shrike
package is fine-grained sub-libraries, so a target names the specific
libraries it exercises — usually via its package's conftest — instead of
getting the whole package implicitly).
"""

load("@rules_python//python:defs.bzl", "py_test")
load("@shrike_pip//:requirements.bzl", "requirement")

_RUNNER = "//tools/bazel:pytest_runner.py"

def pytest_test(name, srcs, deps = [], data = [], args = [], size = "small", xdist = "auto", env = {}, **kwargs):
    # `-n <xdist>` (xdist worker count) unless a caller disables it with a falsy
    # value (None / 0 / False) for a serial run. str() so an int count works too.
    xdist_args = ["-n", str(xdist)] if xdist else []

    # Disarm the local native-staleness backstop (#573) under Bazel: that hook
    # (tests/conftest.py) guards the pip lane's per-venv .so against a missing
    # rebuild, but Bazel builds the extension hermetically and has no venv stamp
    # to read, so the check is meaningless here. A caller's own `env` still wins.
    test_env = {"SHRIKE_SKIP_NATIVE_STALE_CHECK": "1"}
    test_env.update(env)

    py_test(
        name = name,
        size = size,
        srcs = srcs + [_RUNNER],
        main = _RUNNER,
        # Each test file is passed to the launcher as a positional arg; cwd is the
        # runfiles root at test time, so $(location ...) resolves correctly.
        args = ["$(location {})".format(s) for s in srcs] + xdist_args + args,
        deps = deps + [
            requirement("pytest"),
            requirement("pytest-asyncio"),
            requirement("pytest-xdist"),
        ],
        data = data,
        env = test_env,
        **kwargs
    )
