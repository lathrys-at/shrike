"""The profiler invocations the instrument seam builds — pure, no native/harness,
so they run on the per-PR lane. The platform is passed in (never read from the
host) so both the native-capable and Python-only py-spy shapes are tested here."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.manual.perf.instrument import (
    artifact_path,
    instrument_binary,
    instrument_command,
    pyspy_native_supported,
    validate_instrument_request,
)


def _validate(**overrides):
    """A ``validate_instrument_request`` call for a well-formed py-spy request, with
    per-case overrides — returns the error message, or ``None`` when valid."""
    kwargs = dict(
        tool="py-spy",
        instrument_args=[],
        workloads=["search"],
        out_given=False,
        platform="linux",
    )
    kwargs.update(overrides)
    return validate_instrument_request(**kwargs)


def _command(tool: str, tmp_path: Path, *, platform: str = "linux", **overrides):
    """An ``instrument_command`` for ``tool`` with sensible defaults the tests
    override per case."""
    kwargs = dict(
        platform=platform,
        extra_args=[],
        profile="stub",
        profile_path=None,
        size=500,
        variant="text",
        workload="search",
        repeats=5,
        warmup=1,
        ops=100,
        seed=42,
        baseline=None,
    )
    kwargs.update(overrides)
    return instrument_command(tool, tmp_path / "run.py", tmp_path, **kwargs)


def test_artifact_path_is_named_per_workload_and_tool():
    assert artifact_path(Path("/runs/x"), "search", "py-spy").name == "flame-search.svg"
    assert artifact_path(Path("/runs/x"), "search", "samply").name == "profile-search.json"
    assert artifact_path(Path("/runs/x"), "rebuild", "xctrace").name == "profile-rebuild.trace"


def test_instrument_binary_maps_xctrace_to_xcrun():
    assert instrument_binary("py-spy") == "py-spy"
    assert instrument_binary("samply") == "samply"
    assert instrument_binary("xctrace") == "xcrun"


def test_pyspy_native_supported_only_on_linux_and_windows():
    assert pyspy_native_supported("linux")
    assert pyspy_native_supported("win32")
    assert not pyspy_native_supported("darwin")
    assert not pyspy_native_supported("freebsd13")


def test_pyspy_uses_native_on_linux(tmp_path):
    cmd = _command("py-spy", tmp_path, platform="linux")
    # --native is the cross-boundary requirement (Python + Rust in one flamegraph).
    assert cmd[:3] == ["py-spy", "record", "--native"]
    assert str(tmp_path / "flame-search.svg") in cmd
    # The inner run is a SINGLE workload and carries NO --instrument (no re-exec).
    assert "--instrument" not in cmd
    assert cmd[cmd.index("--workloads") + 1] == "search"
    assert str(tmp_path / "run.py") in cmd  # re-execs this very runner
    # The built-in profile is reproduced as --profile (not --profile-path), and N
    # is threaded so the inner run scales the same.
    assert cmd[cmd.index("--profile") + 1] == "stub"
    assert "--profile-path" not in cmd
    assert cmd[cmd.index("--ops") + 1] == "100"


def test_pyspy_drops_native_on_macos(tmp_path):
    # macOS has no py-spy native unwinding; passing --native there aborts the run,
    # so it is omitted and the flamegraph is Python-only.
    cmd = _command("py-spy", tmp_path, platform="darwin")
    assert cmd[:2] == ["py-spy", "record"]
    assert "--native" not in cmd
    assert str(tmp_path / "flame-search.svg") in cmd


def test_samply_command_saves_without_a_server(tmp_path):
    cmd = _command("samply", tmp_path, workload="rebuild")
    assert cmd[:3] == ["samply", "record", "--save-only"]
    assert cmd[cmd.index("--output") + 1] == str(tmp_path / "profile-rebuild.json")
    # The inner run follows a -- separator and is a single workload.
    assert "--" in cmd
    assert cmd[cmd.index("--workloads") + 1] == "rebuild"
    assert "--native" not in cmd  # samply has no such flag


def test_xctrace_command_launches_the_time_profiler(tmp_path):
    cmd = _command("xctrace", tmp_path, workload="upsert-batch")
    assert cmd[:3] == ["xcrun", "xctrace", "record"]
    assert cmd[cmd.index("--template") + 1] == "Time Profiler"
    assert cmd[cmd.index("--output") + 1] == str(tmp_path / "profile-upsert-batch.trace")
    # --launch precedes the -- target separator and the inner run.
    assert cmd.index("--launch") < cmd.index("--")
    assert cmd[cmd.index("--workloads") + 1] == "upsert-batch"


def test_extra_args_pass_through_to_the_tool(tmp_path):
    # --instrument-arg tokens reach the tool verbatim and contiguous, and stay on
    # the tool's side of the -- target separator (so they tune the tool, never leak
    # into the inner run).
    extra = ["--rate", "2000"]
    for tool in ("py-spy", "samply", "xctrace"):
        cmd = _command(tool, tmp_path, platform="darwin", extra_args=extra)
        sep = cmd.index("--")
        start = next(i for i in range(len(cmd)) if cmd[i : i + len(extra)] == extra)
        assert start + len(extra) <= sep, tool  # on the tool's side of --
        assert "--rate" not in cmd[sep + 1 :], tool  # and never leaks into the inner run


def test_threads_a_baseline_and_unsanitized_variant(tmp_path):
    cmd = _command(
        "py-spy",
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


def test_reproduces_a_custom_profile_path(tmp_path):
    # A custom profile is reproduced as --profile-path (NOT --profile), so the
    # inner run boots the same custom engines.
    cmd = _command(
        "py-spy",
        tmp_path,
        profile=None,
        profile_path=Path("/cfg/my-engines.yml"),
    )
    assert cmd[cmd.index("--profile-path") + 1] == "/cfg/my-engines.yml"
    assert "--profile" not in cmd


def test_rejects_an_unknown_tool(tmp_path):
    with pytest.raises(ValueError, match="unknown instrumenter"):
        _command("perf", tmp_path)


def test_validate_passes_a_well_formed_request():
    assert _validate() is None
    assert _validate(tool="samply") is None
    # xctrace is fine on macOS.
    assert _validate(tool="xctrace", platform="darwin") is None
    # No --instrument at all (a plain timed run) is valid.
    assert _validate(tool=None) is None


def test_validate_rejects_instrument_arg_without_a_tool():
    assert "needs --instrument" in _validate(tool=None, instrument_args=["--rate", "2000"])


def test_validate_rejects_multiple_workloads():
    assert "ONE workload" in _validate(workloads=["search", "rebuild"])


def test_validate_rejects_out_under_instrument():
    assert "--out is ignored" in _validate(out_given=True)


def test_validate_rejects_xctrace_off_macos():
    assert "macOS-only" in _validate(tool="xctrace", platform="linux")
    assert "macOS-only" in _validate(tool="xctrace", platform="win32")


def test_validate_checks_arg_without_tool_before_workload_count():
    # The arg-without-a-tool mistake is reported even when the workload list is also
    # wrong (it's the more fundamental error).
    msg = _validate(tool=None, instrument_args=["-r"], workloads=["a", "b"])
    assert "needs --instrument" in msg
