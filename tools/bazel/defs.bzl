"""pytest_test — run a pytest file as a Bazel py_test (#244).

Wraps rules_python's py_test with the shared pytest launcher
(//tools/bazel:pytest_runner.py), so a test file runs under `bazel test` the same
way it does under plain pytest. Always pulls in pytest + pytest-asyncio and the
shrike library; a test needing more (a genuine third-party wheel) adds it via
`deps`, and the package's conftest goes in `deps` too (so pytest finds it in
runfiles).
"""

load("@rules_python//python:defs.bzl", "py_test")
load("@shrike_pip//:requirements.bzl", "requirement")

_RUNNER = "//tools/bazel:pytest_runner.py"

def pytest_test(name, src, deps = [], data = [], args = [], size = "small", **kwargs):
    py_test(
        name = name,
        size = size,
        srcs = [src, _RUNNER],
        main = _RUNNER,
        # The launcher receives the test file's runfiles path; cwd is the
        # runfiles root at test time, so $(location ...) resolves correctly.
        args = ["$(location {})".format(src)] + args,
        deps = deps + [
            "//src/shrike",
            requirement("pytest"),
            requirement("pytest-asyncio"),
        ],
        data = data,
        **kwargs
    )
