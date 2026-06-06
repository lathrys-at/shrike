from __future__ import annotations

import base64

import pytest

from shrike import collection as collection_mod
from shrike.collection import _resolve_public_ip, _safe_media_name
from shrike.schemas import CollectionPruneResponse, FetchMediaResponse, StoreMediaResponse

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n-fake-image-bytes").decode("ascii")


class TestStoreMedia:
    async def test_store_data(self, wrapper):
        results = await wrapper.store_media(
            [{"data": PNG, "filename": "cell.png"}], allow_private_fetch=False
        )
        # validates against the wire schema
        resp = StoreMediaResponse.model_validate({"results": results})
        assert resp.results[0].status == "stored"
        assert resp.results[0].filename == "cell.png"
        assert resp.results[0].mime == "image/png"
        assert resp.results[0].deduped is False

    async def test_bad_base64_is_per_item_error(self, wrapper):
        results = await wrapper.store_media(
            [
                {"data": PNG, "filename": "ok.png"},
                {"data": "!!not-base64!!", "filename": "bad.png"},
            ],
            allow_private_fetch=False,
        )
        assert [r["status"] for r in results] == ["stored", "error"]
        assert results[1]["index"] == 1
        assert results[1]["filename"] == "bad.png"

    async def test_identical_content_dedupes(self, wrapper):
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        again = await wrapper.store_media(
            [{"data": PNG, "filename": "a.png"}], allow_private_fetch=False
        )
        assert again[0]["deduped"] is True
        assert again[0]["filename"] == "a.png"

    async def test_collision_renames(self, wrapper):
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        other = base64.b64encode(b"totally different bytes").decode("ascii")
        clash = await wrapper.store_media(
            [{"data": other, "filename": "a.png"}], allow_private_fetch=False
        )
        assert clash[0]["status"] == "stored"
        assert clash[0]["filename"] != "a.png"
        assert clash[0]["deduped"] is False

    async def test_store_url_derives_name_and_extension(self, wrapper, monkeypatch):
        def fake_fetch(url, *, allow_private, **kwargs):
            assert allow_private is False
            return b"downloaded-bytes", "image/png"

        monkeypatch.setattr(collection_mod, "_fetch_media_url", fake_fetch)
        # no filename and a URL path without an extension -> derived from Content-Type
        results = await wrapper.store_media(
            [{"url": "https://example.com/asset"}], allow_private_fetch=False
        )
        assert results[0]["status"] == "stored"
        assert results[0]["filename"].endswith(".png")

    async def test_store_url_failure_is_per_item_error(self, wrapper, monkeypatch):
        def boom(url, *, allow_private, **kwargs):
            raise ValueError("refusing to fetch from non-public address 10.0.0.1")

        monkeypatch.setattr(collection_mod, "_fetch_media_url", boom)
        results = await wrapper.store_media(
            [{"url": "http://10.0.0.1/x.png"}], allow_private_fetch=False
        )
        assert results[0]["status"] == "error"
        assert "non-public" in results[0]["error"]


class TestFetchMedia:
    async def test_found_reports_path_never_bytes(self, wrapper):
        # fetch never returns bytes — only where they live (path; url added by the
        # tool layer). The model/CLI fetches from there.
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        results = await wrapper.fetch_media(["a.png", "nope.png"])
        resp = FetchMediaResponse.model_validate({"results": results})
        assert resp.results[0].status == "found"
        assert "data" not in results[0]
        assert resp.results[0].path.endswith("a.png")
        assert resp.results[0].size_bytes > 0
        assert resp.results[1].status == "missing"

    async def test_path_traversal_is_missing(self, wrapper):
        results = await wrapper.fetch_media(["../../etc/passwd"])
        assert results[0]["status"] == "missing"


class TestListMedia:
    async def test_list_and_glob(self, wrapper):
        await wrapper.store_media(
            [
                {"data": PNG, "filename": "a.png"},
                {"data": PNG, "filename": "b.jpg"},
            ],
            allow_private_fetch=False,
        )
        allm = await wrapper.list_media(pattern=None, limit=None)
        assert allm["count"] == 2
        pngs = await wrapper.list_media(pattern="*.png", limit=None)
        assert [f["filename"] for f in pngs["files"]] == ["a.png"]

    async def test_limit_caps_files_not_count(self, wrapper):
        await wrapper.store_media(
            [{"data": PNG, "filename": f"f{i}.png"} for i in range(3)],
            allow_private_fetch=False,
        )
        listed = await wrapper.list_media(pattern=None, limit=2)
        assert listed["count"] == 3
        assert len(listed["files"]) == 2


class TestDeleteMedia:
    async def test_trash_and_not_found(self, wrapper):
        await wrapper.store_media([{"data": PNG, "filename": "a.png"}], allow_private_fetch=False)
        result = await wrapper.delete_media(["a.png", "ghost.png"])
        assert result["deleted"] == ["a.png"]
        assert result["not_found"] == ["ghost.png"]
        assert (await wrapper.list_media(pattern="a.png", limit=None))["count"] == 0


class TestMediaCheck:
    async def test_unused_and_missing(self, wrapper):
        # an unreferenced file -> unused; a note referencing an absent file -> missing
        await wrapper.store_media(
            [{"data": PNG, "filename": "orphan.png"}], allow_private_fetch=False
        )
        wrapper.run_sync(
            lambda _c: wrapper._upsert_notes(
                [
                    {
                        "deck": "Test",
                        "note_type": "Basic",
                        "fields": {"Front": '<img src="ghost.png">', "Back": "x"},
                    }
                ]
            )
        )
        check = await wrapper.media_check()
        assert "orphan.png" in check["unused"]
        assert "ghost.png" in check["missing"]
        assert check["missing_media_notes"]


class TestPruneUnusedMedia:
    async def test_dry_run_then_apply(self, wrapper):
        await wrapper.store_media(
            [{"data": PNG, "filename": "orphan.png"}], allow_private_fetch=False
        )
        preview, removed = await wrapper.prune(
            unused_tags=False,
            empty_notes=False,
            empty_cards=False,
            unused_media=True,
            dry_run=True,
        )
        resp = CollectionPruneResponse.model_validate(preview)
        assert resp.unused_media is not None
        assert "orphan.png" in resp.unused_media.files
        assert resp.unused_tags is None  # not requested
        assert removed == []  # media removal isn't an index concern
        assert (await wrapper.list_media(pattern="orphan.png", limit=None))["count"] == 1

        applied, _ = await wrapper.prune(
            unused_tags=False,
            empty_notes=False,
            empty_cards=False,
            unused_media=True,
            dry_run=False,
        )
        assert applied["unused_media"]["removed"] == 1
        assert (await wrapper.list_media(pattern="orphan.png", limit=None))["count"] == 0


class TestSsrfGuard:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "169.254.169.254",  # cloud metadata
            "::1",
            "100.64.0.1",  # carrier-grade NAT — a denylist misses this; is_global catches it
            "192.0.0.1",
            "0.0.0.0",
            "224.0.0.1",  # multicast — is_global is True for it, so checked explicitly
            "240.0.0.1",  # reserved/future
            "::ffff:169.254.169.254",  # IPv4-mapped IPv6
            "fd00::1",  # unique-local
            "fe80::1",  # link-local
            "::",  # unspecified
        ],
    )
    def test_non_global_addresses_blocked(self, host):
        with pytest.raises(ValueError, match="non-public address"):
            _resolve_public_ip(host)

    @pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
    def test_public_address_allowed(self, host):
        _resolve_public_ip(host)  # numeric literal: no DNS, no network

    def test_split_horizon_mixed_records_refused(self, monkeypatch):
        # A name resolving to [public, private] must be rejected — the guard checks
        # *every* record, so an internal A-record can't ride alongside a public one.
        def fake_getaddrinfo(host, *a, **k):
            return [
                (0, 0, 0, "", ("1.1.1.1", 0)),
                (0, 0, 0, "", ("127.0.0.1", 0)),
            ]

        monkeypatch.setattr(collection_mod.socket, "getaddrinfo", fake_getaddrinfo)
        with pytest.raises(ValueError, match="non-public address"):
            _resolve_public_ip("split.example")

    @pytest.mark.parametrize(
        "literal", ["http://2130706433/", "http://0x7f000001/", "http://127.1/"]
    )
    def test_obfuscated_ipv4_literal_refused(self, monkeypatch, literal):
        # These decimal/hex/short forms all denote 127.0.0.1; getaddrinfo parsing
        # varies by platform, so pin it to keep the test hermetic and cross-platform.
        from urllib.parse import urlparse

        monkeypatch.setattr(
            collection_mod.socket,
            "getaddrinfo",
            lambda *a, **k: [(0, 0, 0, "", ("127.0.0.1", 0))],
        )
        with pytest.raises(ValueError, match="non-public address"):
            _resolve_public_ip(urlparse(literal).hostname or literal)

    def test_unresolvable_host_raises(self, monkeypatch):
        import socket as _socket

        def boom(*a, **k):
            raise _socket.gaierror("nope")

        monkeypatch.setattr(collection_mod.socket, "getaddrinfo", boom)
        with pytest.raises(ValueError, match="could not resolve host"):
            _resolve_public_ip("nonexistent.invalid")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("../../etc/passwd", "passwd"), ("a/b/c.png", "c.png"), ("..", ""), ("", "")],
    )
    def test_safe_media_name(self, raw, expected):
        assert _safe_media_name(raw) == expected


# -- Manual redirect-hop handling (the SSRF fix). Fake httpx.Client so no network;
# every test asserts the forbidden hop is BOTH refused AND never streamed (the
# "never connected" half pins guard-before-fetch ordering). socket.getaddrinfo on
# IP literals is network-free, so the address check runs for real. --------------


def _make_fake_client(script, *, requested, kwargs_sink, calls=None):
    """Return a fake httpx.Client class driving _fetch_media_url's redirect loop.

    ``script[i]`` drives the i-th ``.stream()`` call:
      {"kind": "redirect", "location": "..."}  or
      {"kind": "ok", "headers": {...}, "chunks": [b".."]}
    ``requested`` collects every URL actually streamed (the pinned IP URL);
    ``kwargs_sink`` captures the ``httpx.Client(**kwargs)`` constructor kwargs (to
    pin ``follow_redirects=False``); ``calls`` (optional) records each stream's
    ``{"url", "headers", "extensions"}`` so a test can assert the Host header / SNI.
    """
    import httpx

    class _Resp:
        def __init__(self, entry, url):
            self._e = entry
            self.url = httpx.URL(url)
            self.is_redirect = entry["kind"] == "redirect"
            self.headers = dict(entry.get("headers", {}))
            if self.is_redirect and "location" in entry:
                self.headers["location"] = entry["location"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield from self._e.get("chunks", [b""])

    class _Client:
        def __init__(self, *a, **k):
            kwargs_sink.update(k)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, **kwargs):
            requested.append(url)
            if calls is not None:
                calls.append({"url": url, **kwargs})
            return _Resp(script[len(requested) - 1], url)

    return _Client


class TestRedirectHandling:
    def _run(self, monkeypatch, script, *, url="http://1.1.1.1/start", **kw):
        """Drive _fetch_media_url over a scripted fake client; return (result_or_exc,
        requested, kwargs_sink). Never raises — caller inspects the first element."""
        import httpx

        from shrike.collection import _fetch_media_url

        requested: list[str] = []
        kwargs_sink: dict = {}
        monkeypatch.setattr(
            httpx, "Client", _make_fake_client(script, requested=requested, kwargs_sink=kwargs_sink)
        )
        try:
            result: object = _fetch_media_url(url, allow_private=False, **kw)
        except Exception as e:  # noqa: BLE001 — tests inspect the exception
            result = e
        return result, requested, kwargs_sink

    def _assert_blocked(self, result, requested, forbidden, *, match="non-public address"):
        assert isinstance(result, ValueError), f"expected ValueError, got {result!r}"
        assert match in str(result)
        assert not any(forbidden in u for u in requested), (
            f"guard ran too late — {forbidden} was streamed: {requested}"
        )

    # 1. Guard runs on *every* hop
    def test_redirect_chain_public_then_private(self, monkeypatch):
        result, requested, _ = self._run(
            monkeypatch,
            [
                {"kind": "redirect", "location": "http://1.0.0.1/a"},
                {"kind": "redirect", "location": "http://169.254.169.254/latest/meta-data/"},
            ],
        )
        self._assert_blocked(result, requested, "169.254.169.254")

    @pytest.mark.parametrize("private", ["127.0.0.1", "10.0.0.1", "192.168.1.1"])
    def test_redirect_to_private(self, monkeypatch, private):
        result, requested, _ = self._run(
            monkeypatch, [{"kind": "redirect", "location": f"http://{private}/x"}]
        )
        self._assert_blocked(result, requested, private)

    def test_protocol_relative_redirect(self, monkeypatch):
        # //host inherits the current scheme; the joined host must be re-validated.
        result, requested, _ = self._run(
            monkeypatch, [{"kind": "redirect", "location": "//169.254.169.254/latest/"}]
        )
        self._assert_blocked(result, requested, "169.254.169.254")

    def test_userinfo_in_redirect_host(self, monkeypatch):
        # http://expected.public@127.0.0.1/ — hostname is 127.0.0.1, not the userinfo.
        result, requested, _ = self._run(
            monkeypatch, [{"kind": "redirect", "location": "http://expected.public@127.0.0.1/"}]
        )
        self._assert_blocked(result, requested, "127.0.0.1")

    # 2. Scheme & Location robustness
    @pytest.mark.parametrize(
        "location",
        ["file:///etc/passwd", "gopher://127.0.0.1:6379/_INFO", "dict://127.0.0.1/", "ftp://x/"],
    )
    def test_redirect_to_dangerous_scheme(self, monkeypatch, location):
        result, requested, _ = self._run(monkeypatch, [{"kind": "redirect", "location": location}])
        self._assert_blocked(result, requested, location, match="unsupported URL scheme")

    def test_redirect_missing_location(self, monkeypatch):
        # fake-only: real httpx's is_redirect implies a Location.
        result, _, _ = self._run(monkeypatch, [{"kind": "redirect"}])
        assert isinstance(result, ValueError)
        assert "Location" in str(result)

    def test_redirect_empty_location(self, monkeypatch):
        result, _, _ = self._run(monkeypatch, [{"kind": "redirect", "location": ""}])
        assert isinstance(result, ValueError)
        assert "Location" in str(result)

    def test_scheme_change_http_to_https_allowed(self, monkeypatch):
        result, requested, _ = self._run(
            monkeypatch,
            [
                {"kind": "redirect", "location": "https://1.1.1.1/secure"},
                {"kind": "ok", "headers": {"content-type": "image/png"}, "chunks": [b"ok"]},
            ],
        )
        assert result == (b"ok", "image/png")
        assert requested == ["http://1.1.1.1/start", "https://1.1.1.1/secure"]

    def test_relative_redirect_resolves_same_host(self, monkeypatch):
        result, requested, _ = self._run(
            monkeypatch,
            [
                {"kind": "redirect", "location": "/b"},
                {"kind": "ok", "chunks": [b"body"]},
            ],
        )
        assert result == (b"body", None)
        assert requested == ["http://1.1.1.1/start", "http://1.1.1.1/b"]

    # 3. Redirect counting / loop
    def test_exactly_max_redirects_succeeds(self, monkeypatch):
        script = [{"kind": "redirect", "location": f"http://1.1.1.1/{i}"} for i in range(5)]
        script.append({"kind": "ok", "chunks": [b"final"]})
        result, requested, _ = self._run(monkeypatch, script)
        assert result == (b"final", None)
        assert len(requested) == 6

    def test_one_over_max_redirects_raises(self, monkeypatch):
        script = [{"kind": "redirect", "location": f"http://1.1.1.1/{i}"} for i in range(6)]
        result, requested, _ = self._run(monkeypatch, script)
        assert isinstance(result, ValueError)
        assert "too many redirects" in str(result)
        assert len(requested) == 6

    def test_redirect_cycle_terminates(self, monkeypatch):
        script = [
            {"kind": "redirect", "location": "http://1.1.1.1/a" if i % 2 else "http://1.0.0.1/b"}
            for i in range(6)
        ]
        result, requested, _ = self._run(monkeypatch, script, url="http://1.1.1.1/a")
        assert isinstance(result, ValueError)
        assert "too many redirects" in str(result)
        assert len(requested) == 6

    # 4. allow_private semantics
    def test_allow_private_follows_redirect_to_private(self, monkeypatch):
        import httpx

        from shrike.collection import _fetch_media_url

        requested: list[str] = []
        kwargs_sink: dict = {}
        script = [
            {"kind": "redirect", "location": "http://127.0.0.1/internal"},
            {"kind": "ok", "chunks": [b"private-ok"]},
        ]
        monkeypatch.setattr(
            httpx, "Client", _make_fake_client(script, requested=requested, kwargs_sink=kwargs_sink)
        )
        raw, _ = _fetch_media_url("http://1.1.1.1/start", allow_private=True)
        assert raw == b"private-ok"
        assert "http://127.0.0.1/internal" in requested  # the flag disables the check on every hop

    def test_allow_private_still_enforces_scheme(self, monkeypatch):
        # allow_private gates only the address check, not scheme/host-presence.
        import httpx

        from shrike.collection import _fetch_media_url

        monkeypatch.setattr(
            httpx,
            "Client",
            _make_fake_client(
                [{"kind": "redirect", "location": "file:///etc/passwd"}],
                requested=[],
                kwargs_sink={},
            ),
        )
        with pytest.raises(ValueError, match="unsupported URL scheme"):
            _fetch_media_url("http://1.1.1.1/start", allow_private=True)

    # 6. Body size cap
    def test_size_cap_across_chunks(self, monkeypatch):
        result, _, _ = self._run(
            monkeypatch,
            [{"kind": "ok", "chunks": [b"x" * 6, b"y" * 6]}],
            max_bytes=10,
        )
        assert isinstance(result, ValueError)
        assert "exceeds" in str(result)

    def test_size_cap_terminal_after_redirects(self, monkeypatch):
        result, _, _ = self._run(
            monkeypatch,
            [
                {"kind": "redirect", "location": "http://1.1.1.1/a"},
                {"kind": "ok", "chunks": [b"z" * 50]},
            ],
            max_bytes=10,
        )
        assert isinstance(result, ValueError)
        assert "exceeds" in str(result)

    # 7. Invariant the fake can't enforce: follow_redirects must be False.
    def test_client_constructed_without_follow_redirects(self, monkeypatch):
        _, _, kwargs_sink = self._run(monkeypatch, [{"kind": "ok", "chunks": [b"x"]}])
        assert kwargs_sink.get("follow_redirects") is False


class TestConnectionPinning:
    """The connection is pinned to the vetted IP (closes the DNS-rebinding TOCTOU):
    the request dials the resolved IP, the Host header carries the name, and HTTPS
    SNI/cert validation uses the name via the sni_hostname extension."""

    def _fake(self, monkeypatch, *, resolves_to: str, script):
        import httpx

        # The name resolves (once, for vetting) to a public IP; pinning must then
        # dial THAT IP rather than re-resolve the name at connect time.
        monkeypatch.setattr(
            collection_mod.socket,
            "getaddrinfo",
            lambda host, *a, **k: [(0, 0, 0, "", (resolves_to, 0))],
        )
        calls: list[dict] = []
        monkeypatch.setattr(
            httpx,
            "Client",
            _make_fake_client(script, requested=[], kwargs_sink={}, calls=calls),
        )
        return calls

    def test_https_pins_ip_sets_host_and_sni(self, monkeypatch):
        from shrike.collection import _fetch_media_url

        calls = self._fake(
            monkeypatch,
            resolves_to="1.1.1.1",
            script=[{"kind": "ok", "headers": {"content-type": "image/png"}, "chunks": [b"x"]}],
        )
        raw, ct = _fetch_media_url("https://cdn.example/img.png", allow_private=False)
        assert raw == b"x"
        # Dialed the vetted IP, not the name — no connect-time re-resolution.
        assert calls[0]["url"] == "https://1.1.1.1/img.png"
        assert calls[0]["headers"]["Host"] == "cdn.example"
        assert calls[0]["extensions"]["sni_hostname"] == "cdn.example"

    def test_http_pins_ip_with_host_no_sni(self, monkeypatch):
        from shrike.collection import _fetch_media_url

        calls = self._fake(
            monkeypatch,
            resolves_to="1.1.1.1",
            script=[{"kind": "ok", "chunks": [b"y"]}],
        )
        _fetch_media_url("http://cdn.example:8080/a/b.png?q=1", allow_private=False)
        assert calls[0]["url"] == "http://1.1.1.1:8080/a/b.png?q=1"
        assert calls[0]["headers"]["Host"] == "cdn.example:8080"
        # Plain HTTP has no TLS, so no sni_hostname extension.
        assert not calls[0]["extensions"]

    def test_allow_private_does_not_pin(self, monkeypatch):
        # With the guard off, connect to the URL as given (no IP rewrite / Host
        # override) — the operator opted into trusted internal hosts.
        import httpx

        from shrike.collection import _fetch_media_url

        calls: list[dict] = []
        monkeypatch.setattr(
            httpx,
            "Client",
            _make_fake_client(
                [{"kind": "ok", "chunks": [b"z"]}], requested=[], kwargs_sink={}, calls=calls
            ),
        )
        _fetch_media_url("http://intranet.local/x.png", allow_private=True)
        assert calls[0]["url"] == "http://intranet.local/x.png"
        assert not calls[0]["headers"]
        assert not calls[0]["extensions"]
