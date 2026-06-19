"""Unit tests for the server's transport-security helpers."""

from __future__ import annotations

import ipaddress
import logging

import pytest
from mcp.server.transport_security import TransportSecurityMiddleware

from shrike.server.server import (
    _EXEC_SHAPING_OVERRIDES,
    _build_transport_security,
    _ControlListener,
    _is_loopback,
    _rejected_exec_overrides,
    _server_is_purely_local,
    _validate_media_path_root,
    create_mcp,
)


def _mcp_protection(host: str, security: object) -> bool:
    """The DNS-rebinding-protection state FastMCP applies to /mcp for these inputs."""
    app = create_mcp(host=host, port=8372, transport_security=security)  # type: ignore[arg-type]
    ts = app.settings.transport_security
    assert ts is not None
    return ts.enable_dns_rebinding_protection


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"],
)
def test_loopback_hosts_are_recognized(host: str) -> None:
    assert _is_loopback(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.10", "10.0.0.5", "example.com", "::"],
)
def test_non_loopback_hosts_are_rejected(host: str) -> None:
    assert _is_loopback(host) is False


def test_transport_security_enabled_for_loopback() -> None:
    settings = _build_transport_security("127.0.0.1")
    assert settings is not None
    assert settings.enable_dns_rebinding_protection is True
    # Loopback hosts/origins only; the port is wildcarded.
    assert settings.allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert settings.allowed_origins == [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    ]


def test_transport_security_none_for_non_loopback() -> None:
    # A deliberately remote bind with no explicit allow-list has no fixed Host
    # set to validate against; protection is left to the network boundary /
    # (roadmap) auth layer. This preserves the historical --allow-remote default.
    assert _build_transport_security("0.0.0.0") is None
    assert _build_transport_security("192.168.1.10") is None


def test_extra_allowed_host_extends_loopback_list() -> None:
    # The reverse-proxy / VPN case: loopback bind, guard on, but additionally
    # trust the hostname the proxy forwards (e.g. a Tailscale name).
    settings = _build_transport_security(
        "127.0.0.1",
        allowed_hosts=["host.tailnet.ts.net"],
        allowed_origins=["https://host.tailnet.ts.net"],
    )
    assert settings is not None
    assert settings.allowed_hosts == [
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        "host.tailnet.ts.net",
    ]
    assert "https://host.tailnet.ts.net" in settings.allowed_origins


@pytest.mark.parametrize(
    ("host", "expected_host", "expected_origin"),
    [
        ("127.0.0.2", "127.0.0.2:*", "http://127.0.0.2:*"),
        ("127.0.0.53", "127.0.0.53:*", "http://127.0.0.53:*"),
        # An expanded ::1 is still loopback; it canonicalizes to the bracketed
        # "[::1]:*" form, which the fixed trio already carries — so no duplicate.
        ("0:0:0:0:0:0:0:1", "[::1]:*", "http://[::1]:*"),
    ],
)
def test_loopback_bind_host_folded_into_allow_list(
    host: str, expected_host: str, expected_origin: str
) -> None:
    # is_loopback accepts ALL of 127/8, so main() accepts a bind like
    # --host 127.0.0.2 with no --allow-remote. The guard must therefore answer
    # that bind's own Host header rather than 421 every request (self-brick).
    settings = _build_transport_security(host)
    assert settings is not None
    assert expected_host in settings.allowed_hosts
    assert expected_origin in settings.allowed_origins
    # No duplicate entries (an already-fixed literal like [::1]:* isn't re-added).
    assert settings.allowed_hosts.count(expected_host) == 1


def test_non_127_0_0_1_loopback_bind_is_reachable() -> None:
    # A legitimate client to a 127.0.0.2 bind must validate. 127.0.0.1
    # stays valid too (the fixed loopback literals are not dropped).
    settings = _build_transport_security("127.0.0.2")
    mw = TransportSecurityMiddleware(settings)
    assert mw._validate_host("127.0.0.2:8372") is True
    assert mw._validate_host("127.0.0.1:8372") is True


def test_no_dns_rebinding_protection_disables_guard_on_any_bind() -> None:
    # The network-is-the-boundary mode: guard off regardless of bind address,
    # even on loopback (Shrike behind a local reverse proxy / tailscale serve).
    assert _build_transport_security("127.0.0.1", disable=True) is None
    assert _build_transport_security("0.0.0.0", disable=True) is None


# -- --no-dns-rebinding-protection must reach /mcp, not just the custom routes --


def test_disabled_guard_is_honored_on_mcp_endpoint_loopback() -> None:
    # With the guard disabled on a loopback bind, _build_transport_security
    # returns None — and FastMCP would SILENTLY re-enable DNS-rebinding protection
    # on /mcp for 127.0.0.1 when handed None (mcp fastmcp/server.py auto-enables for
    # 127.0.0.1/localhost/::1). The custom routes honor the off-state; /mcp must too.
    # create_mcp must translate None into explicit protection-off so /mcp obeys the flag.
    security = _build_transport_security("127.0.0.1", disable=True)
    assert security is None
    assert _mcp_protection("127.0.0.1", security) is False


def test_disabled_guard_matches_custom_routes_on_mcp() -> None:
    # The two layers must AGREE: when the guard is off, both /mcp (FastMCP) and the
    # custom routes (TransportSecurityMiddleware(None)) report protection disabled.
    security = _build_transport_security("localhost", disable=True)
    custom_route_protection = TransportSecurityMiddleware(
        security
    ).settings.enable_dns_rebinding_protection
    assert custom_route_protection is False  # custom routes already honored the flag
    assert _mcp_protection("localhost", security) is False  # /mcp now agrees


def test_disabled_guard_not_reenabled_for_any_loopback_spelling() -> None:
    # The SDK's auto-re-enable keys on the exact loopback literals; cover them all so
    # a refactor can't reopen the divergence for one spelling.
    for host in ("127.0.0.1", "localhost", "::1"):
        security = _build_transport_security(host, disable=True)
        assert security is None
        assert _mcp_protection(host, security) is False


def test_default_loopback_guard_stays_on_at_mcp() -> None:
    # Boundary: the fix only changes the EXPLICITLY-disabled case. The default
    # loopback bind (no --no-dns-rebinding-protection) keeps the guard ON at /mcp.
    security = _build_transport_security("127.0.0.1")
    assert security is not None
    assert _mcp_protection("127.0.0.1", security) is True


def test_explicit_allow_list_builds_guard_even_when_non_loopback() -> None:
    # A non-loopback bind WITH an explicit allow-list keeps the guard, trusting
    # only the named hosts (no loopback defaults, since the bind isn't loopback).
    settings = _build_transport_security("0.0.0.0", allowed_hosts=["proxy.internal"])
    assert settings is not None
    assert settings.allowed_hosts == ["proxy.internal"]
    assert settings.allowed_origins == []


def test_disable_takes_precedence_over_allow_list() -> None:
    # disable=True wins even when explicit hosts are given — the operator opted
    # out of the guard entirely.
    assert (
        _build_transport_security("127.0.0.1", allowed_hosts=["proxy.internal"], disable=True)
        is None
    )


def test_origins_only_non_loopback_is_fail_closed(caplog: pytest.LogCaptureFixture) -> None:
    # The footgun: a non-loopback bind given allowed_origins but NO allowed_hosts
    # builds a guard whose Host allow-list is empty — so every request's Host is
    # rejected (421) and the server answers nothing. Fail-*closed* (not a hole),
    # but a config trap, so it must warn at build time.
    with caplog.at_level(logging.WARNING, logger="shrike.server"):
        settings = _build_transport_security("0.0.0.0", allowed_origins=["https://app.example"])
    assert settings is not None
    assert settings.allowed_hosts == []  # nothing matches → 421 for everything
    assert settings.allowed_origins == ["https://app.example"]
    assert any("no allowed Host values" in r.message for r in caplog.records)


def test_loopback_guard_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    # The normal loopback path has a populated Host list — no footgun, no warning.
    with caplog.at_level(logging.WARNING, logger="shrike.server"):
        _build_transport_security("127.0.0.1")
    assert not any("no allowed Host values" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0:0:0:0:0:0:0:1", True),  # expanded ::1 — still loopback
        ("LOCALHOST", False),  # exact "localhost" match only; uppercase is NOT loopback
    ],
)
def test_is_loopback_stable_edges(host: str, expected: bool) -> None:
    # Pin the Shrike-controlled boundary so a refactor can't silently shift it.
    assert _is_loopback(host) is expected


def test_is_loopback_ipv4_mapped_delegates_to_stdlib() -> None:
    # ::ffff:127.0.0.1 — Shrike adds no special handling, it defers to
    # ipaddress.is_loopback, whose result for IPv4-mapped addresses is
    # *Python-version-dependent* (False <3.12.4, True after). Pin the delegation,
    # not a fixed bool, so the test is robust across the supported range.
    mapped = "::ffff:127.0.0.1"
    assert _is_loopback(mapped) is ipaddress.ip_address(mapped).is_loopback


# -- _server_is_purely_local: gates the store_media server-local `path` --


def _purely_local(**overrides) -> bool:
    base = {
        "host": "127.0.0.1",
        "allow_remote": False,
        "no_dns_rebinding_protection": False,
        "allowed_hosts": None,
        "allowed_origins": None,
    }
    base.update(overrides)
    host = base.pop("host")
    return _server_is_purely_local(host, **base)


def test_default_loopback_is_purely_local() -> None:
    assert _purely_local() is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "0.0.0.0"},  # non-loopback bind
        {"host": "192.168.1.10"},
        {"allow_remote": True},
        {"no_dns_rebinding_protection": True},  # behind a proxy/tailnet → peer may be the proxy
        {"allowed_hosts": ["proxy.internal"]},  # added a proxy/VPN host
        {"allowed_origins": ["https://app.example"]},
    ],
)
def test_any_remote_exposure_signal_disables_server_paths(overrides) -> None:
    assert _purely_local(**overrides) is False


# -- _validate_media_path_root: startup validation of --media-path-root --


def test_media_path_root_valid_dir_returns_realpath(tmp_path) -> None:
    import os

    root = tmp_path / "media"
    root.mkdir()
    assert _validate_media_path_root(str(root)) == os.path.realpath(str(root))


def test_media_path_root_rejects_filesystem_root() -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        _validate_media_path_root("/")


def test_media_path_root_rejects_nonexistent(tmp_path) -> None:
    with pytest.raises(ValueError, match="not an existing directory"):
        _validate_media_path_root(str(tmp_path / "does-not-exist"))


def test_media_path_root_rejects_a_file(tmp_path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="not an existing directory"):
        _validate_media_path_root(str(f))


# -- _rejected_exec_overrides: gates the /embedding/start execution params (#791) --


def test_purely_local_allows_every_exec_override() -> None:
    # The local operator's CLI flow (`shrike embedding start --llama-server ...`):
    # a purely-local server forwards execution-shaping overrides untouched.
    overrides = {k: "x" for k in _EXEC_SHAPING_OVERRIDES}
    assert _rejected_exec_overrides(overrides, purely_local=True) == []


@pytest.mark.parametrize("key", _EXEC_SHAPING_OVERRIDES)
def test_non_local_rejects_each_exec_override(key: str) -> None:
    # The capability hole: a non-purely-local server must refuse a body that
    # chooses the spawned binary/model/args/providers/backend.
    assert _rejected_exec_overrides({key: "x"}, purely_local=False) == [key]


def test_non_local_reports_all_rejected_keys_in_order() -> None:
    overrides = {k: "x" for k in _EXEC_SHAPING_OVERRIDES}
    assert _rejected_exec_overrides(overrides, purely_local=False) == list(_EXEC_SHAPING_OVERRIDES)


def test_llama_server_is_gated() -> None:
    # The direct-RCE param specifically: a caller-supplied binary path is never
    # honoured on a non-purely-local server.
    assert _rejected_exec_overrides({"llama_server": "/tmp/evil"}, purely_local=False) == [
        "llama_server"
    ]


@pytest.mark.parametrize("key", ["port", "context_size", "threads", "gpu_layers", "pooling"])
def test_non_local_allows_runtime_knobs(key: str) -> None:
    # Numeric/enum runtime knobs don't choose what executes, so they pass through
    # even when the server is not purely-local.
    assert _rejected_exec_overrides({key: 1}, purely_local=False) == []


def test_empty_body_is_never_rejected() -> None:
    # A no-override start ("start what the daemon was configured with") is always
    # allowed, even on a non-purely-local server.
    assert _rejected_exec_overrides({}, purely_local=False) == []


# -- _ControlListener.operator_gated drives the /embedding/start exec gate --
#
# The gate keys on the control transport's trust: a filesystem-gated UDS confines
# callers to the operator (overrides allowed); the Windows loopback-TCP fallback
# is reachable by any local user (overrides refused — the #791 RCE class).


def test_uds_control_listener_is_operator_gated(tmp_path) -> None:
    listener = _ControlListener(tmp_path)
    try:
        assert listener.uds is not None  # POSIX default
        assert listener.operator_gated is True
        # The gate therefore ACCEPTS exec overrides on a UDS control plane.
        assert (
            _rejected_exec_overrides({"llama_server": "/x"}, purely_local=listener.operator_gated)
            == []
        )
    finally:
        listener.cleanup()


def test_loopback_tcp_control_listener_is_not_operator_gated(monkeypatch, tmp_path) -> None:
    # Force the Windows branch: a pre-bound loopback-TCP socket, reachable by any
    # local user — so it must NOT be operator-gated, and the gate must REFUSE
    # caller-supplied exec params there.
    monkeypatch.setattr("shrike.server.server.sys.platform", "win32")
    listener = _ControlListener(tmp_path)
    try:
        assert listener.uds is None
        assert listener.host == "127.0.0.1" and listener.port  # ephemeral TCP
        assert listener.operator_gated is False
        assert _rejected_exec_overrides(
            {"llama_server": "/x"}, purely_local=listener.operator_gated
        ) == ["llama_server"]
    finally:
        listener.cleanup()
