"""The py-spy invocation the profiler seam builds — pure, no native/harness, so it
runs on the per-PR lane."""

from __future__ import annotations

from pathlib import Path

from tests.manual.perf.instrument import flame_path, pyspy_command


def test_flame_path_is_named_per_workload():
    assert flame_path(Path("/runs/x"), "search").name == "flame-search.svg"


def test_pyspy_command_profiles_one_workload_natively(tmp_path):
    run_py = tmp_path / "run.py"
    cmd = pyspy_command(
        run_py,
        tmp_path,
        profile="stub",
        size=500,
        variant="text",
        workload="search",
        repeats=5,
        warmup=1,
        ops=100,
        baseline=None,
    )
    # --native is the cross-boundary requirement (Python + Rust in one flamegraph).
    assert cmd[:3] == ["py-spy", "record", "--native"]
    assert str(tmp_path / "flame-search.svg") in cmd
    # The inner run is a SINGLE workload and carries NO --instrument (no re-exec).
    assert "--instrument" not in cmd
    workload_arg = cmd[cmd.index("--workloads") + 1]
    assert workload_arg == "search"
    assert "," not in workload_arg
    assert str(run_py) in cmd  # re-execs this very runner
    # The built-in profile is reproduced as --profile (not --profile-path), and N
    # is threaded so the inner run scales the same.
    assert cmd[cmd.index("--profile") + 1] == "stub"
    assert "--profile-path" not in cmd
    assert cmd[cmd.index("--ops") + 1] == "100"


def test_pyspy_command_threads_a_baseline_when_given(tmp_path):
    cmd = pyspy_command(
        tmp_path / "run.py",
        tmp_path,
        profile="real",
        size=5000,
        variant="text+image",
        workload="rebuild",
        repeats=3,
        warmup=0,
        ops=50,
        baseline=Path("/b.json"),
    )
    assert cmd[cmd.index("--baseline") + 1] == "/b.json"
    # The unsanitized variant flows to the inner run (the corpus spec wants it).
    assert cmd[cmd.index("--variant") + 1] == "text+image"


def test_pyspy_command_reproduces_a_custom_profile_path(tmp_path):
    # A custom profile is reproduced as --profile-path (NOT --profile), so the
    # inner run boots the same custom engines.
    cmd = pyspy_command(
        tmp_path / "run.py",
        tmp_path,
        profile=None,
        profile_path=Path("/cfg/my-engines.yml"),
        size=500,
        variant="text",
        workload="search",
        repeats=5,
        warmup=1,
        ops=100,
        baseline=None,
    )
    assert cmd[cmd.index("--profile-path") + 1] == "/cfg/my-engines.yml"
    assert "--profile" not in cmd
