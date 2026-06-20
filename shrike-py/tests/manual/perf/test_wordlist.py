"""Offline tests for the perf wordlist loader: parsing/filtering, the
cache-vs-fallback selection, the env override, and the cache-key fingerprint.
No network — the download path (`ensure_wordlist`) is exercised by real perf
runs, not here."""

from __future__ import annotations

from tests.manual.perf import wordlist


def test_parse_strips_crlf_and_filters():
    # CRLF endings, case-folding, and the length / a-z filters all in one pass.
    text = "Foo\r\nab\r\nBARbaz\r\nhas123\r\nco-op\r\nabcdefghijklmnop\r\n"
    # "ab" too short; "has123"/"co-op" not pure-alpha; 16-char word over the cap.
    assert wordlist._parse(text) == ["foo", "barbaz"]


def test_load_reads_cache_when_present(tmp_path, monkeypatch):
    cache = tmp_path / "words_alpha.txt"
    cache.write_text("alpha\r\nBETA\r\ngamma\r\n")
    monkeypatch.setattr(wordlist, "cache_path", lambda: cache)
    monkeypatch.delenv("SHRIKE_PERF_NO_WORDLIST", raising=False)
    assert wordlist.load_wordlist(["fallbackword"]) == ["alpha", "beta", "gamma"]


def test_load_falls_back_without_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(wordlist, "cache_path", lambda: tmp_path / "absent.txt")
    monkeypatch.delenv("SHRIKE_PERF_NO_WORDLIST", raising=False)
    # The fallback is filtered the same way: "xy" too short, "n0pe" not alpha.
    assert wordlist.load_wordlist(["mitochondria", "xy", "n0pe", "hello"]) == [
        "mitochondria",
        "hello",
    ]


def test_env_forces_fallback_even_with_cache(tmp_path, monkeypatch):
    cache = tmp_path / "words_alpha.txt"
    cache.write_text("alpha\r\nbeta\r\n")
    monkeypatch.setattr(wordlist, "cache_path", lambda: cache)
    monkeypatch.setenv("SHRIKE_PERF_NO_WORDLIST", "1")
    assert wordlist.load_wordlist(["mitochondria"]) == ["mitochondria"]


def test_fingerprint_distinguishes_and_is_stable():
    small = ["alpha", "beta", "gamma"]
    large = [f"word{i:05d}" for i in range(1000)]
    assert wordlist.fingerprint(small) == wordlist.fingerprint(list(small))  # stable
    assert wordlist.fingerprint(small) != wordlist.fingerprint(large)  # distinguishes
    assert wordlist.fingerprint([]) == "empty"
