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

`xdist` is the `-n` value (default "auto"); pass `xdist = None` to run serial.
"""

load("@rules_python//python:defs.bzl", "py_test")
load("@shrike_pip//:requirements.bzl", "requirement")

_RUNNER = "//tools/bazel:pytest_runner.py"

def pytest_test(name, srcs, deps = [], data = [], args = [], size = "small", xdist = "auto", **kwargs):
    xdist_args = ["-n", xdist] if xdist else []
    py_test(
        name = name,
        size = size,
        srcs = srcs + [_RUNNER],
        main = _RUNNER,
        # Each test file is passed to the launcher as a positional arg; cwd is the
        # runfiles root at test time, so $(location ...) resolves correctly.
        args = ["$(location {})".format(s) for s in srcs] + xdist_args + args,
        deps = deps + [
            "//src/shrike",
            requirement("pytest"),
            requirement("pytest-asyncio"),
            requirement("pytest-xdist"),
        ],
        data = data,
        **kwargs
    )
