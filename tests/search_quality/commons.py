"""Wikimedia Commons image resolution for the manual search-quality corpus (#559).

Reuses the proven ``scripts/eval_multimodal.py`` resolve→pin→cache mechanism,
generalized so a corpus entry can be pinned by its **committed URL** OR
resolved fresh from a ``search_term``, and so the per-image **licensing
metadata** (Commons page / license / author) is captured for ``ASSETS.md``.

The contract that keeps the AGPL repo clean of redistributed assets:

  - **URLs are committed** (``eval/search_quality/resolved_urls.json``) — the
    pinned selection, so a replay is reproducible.
  - **Bytes are NEVER committed** — downloaded on demand into a *gitignored*
    cache (``eval/search_quality/cache/``), exactly the ``eval/multimodal/``
    pattern. The suite is env-gated and manual, so the bytes only ever exist on
    a developer's machine that opted in with ``SHRIKE_SEARCH_QUALITY=1``.

A corpus entry's ``media[].spec`` carries either a ``url`` (already pinned) or a
``search_term`` (resolve on first sight, then pin), plus the licensing fields
(``commons_page`` / ``license`` / ``author``) that ``ASSETS.md`` is generated
from. PD/PD-art/CC0 are preferred; CC-BY-SA is allowed but must be attributed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "shrike-search-quality-eval/0.1 (https://github.com/lathrys-at/shrike)"}


def _get(url: str, *, params: dict | None = None, retries: int = 4) -> httpx.Response:
    """GET with raise_for_status and backoff on transient failures (429/5xx/network)."""
    last: Exception | None = None
    for i in range(retries):
        try:
            r = httpx.get(url, params=params, headers=UA, timeout=30, follow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 — retry transient API/network errors
            last = e
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


@dataclass(frozen=True)
class CommonsAsset:
    """A resolved Commons image: the pinned URL + licensing metadata.

    ``commons_page`` / ``license`` / ``author`` populate ``ASSETS.md`` and the
    manifest's per-image attribution. They come from the MediaWiki
    ``extmetadata`` block; a field absent there is recorded as ``"unknown"`` so
    a human review of ``ASSETS.md`` flags it (never silently empty)."""

    url: str
    commons_page: str
    license: str
    author: str


def _clean(text: str | None) -> str:
    """Strip HTML out of an extmetadata value (Artist often carries an <a> tag)."""
    if not text:
        return "unknown"
    stripped = re.sub(r"<[^>]+>", "", text).strip()
    return stripped or "unknown"


def resolve_asset(
    term_or_url: str,
    *,
    is_url: bool = False,
    width_hint: int = 640,
) -> CommonsAsset:
    """Resolve a Commons ``search_term`` (or verify a pinned ``url``) to a
    :class:`CommonsAsset` with licensing metadata.

    With ``is_url`` the URL is taken verbatim (the pinned-replay path) and only
    its file's metadata is looked up; otherwise a search picks the first
    bitmap/drawing match. PD-first selection is a *human* choice at corpus
    authoring time — this just records whatever license the chosen file carries
    so ``ASSETS.md`` is honest.
    """
    if is_url:
        title = _title_from_url(term_or_url)
        meta = _file_metadata(title) if title else {}
        url = term_or_url
    else:
        url, meta = _search_and_metadata(term_or_url, width_hint)
    ext = meta.get("extmetadata", {})
    return CommonsAsset(
        url=url,
        commons_page=meta.get("descriptionurl", "unknown"),
        license=_clean(ext.get("LicenseShortName", {}).get("value")),
        author=_clean(ext.get("Artist", {}).get("value")),
    )


def _search_and_metadata(term: str, width_hint: int) -> tuple[str, dict[str, Any]]:
    r = _get(
        COMMONS_API,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap|drawing {term}",
            "gsrnamespace": 6,
            "gsrlimit": 1,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": width_hint,
        },
    )
    pages = r.json().get("query", {}).get("pages", {})
    for p in pages.values():
        ii = p.get("imageinfo")
        if ii:
            info = ii[0]
            url = info.get("thumburl") or info.get("url")
            if url:
                return url, info
    raise RuntimeError(f"no Commons image found for {term!r}")


def _title_from_url(url: str) -> str | None:
    """Recover a ``File:Name`` title from a commons upload URL (best-effort).

    A plain upload URL ends in the filename; a ``/thumb/X/XX/<File>/<NNNpx-…>``
    rendition carries the real file as the segment *before* the ``NNNpx-``
    rendition. Pick that segment for a thumb URL, else the last."""
    from urllib.parse import unquote

    parts = [p for p in url.split("/") if p]
    # A thumb URL is .../thumb/X/XX/<File>/<NNNpx-rendition> → take the <File>
    # segment; a plain upload URL's last segment IS the file.
    name = unquote(parts[-2] if "thumb" in parts else parts[-1])
    return f"File:{name}" if name else None


def _file_metadata(title: str) -> dict[str, Any]:
    try:
        r = _get(
            COMMONS_API,
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
            },
        )
        pages = r.json().get("query", {}).get("pages", {})
        for p in pages.values():
            ii = p.get("imageinfo")
            if ii:
                return ii[0]
    except Exception:  # noqa: BLE001 — metadata is best-effort; bytes still resolve
        pass
    return {}


@dataclass
class CommonsCache:
    """The resolve→pin→cache store: committed URL pins + a gitignored byte cache.

    ``resolved_urls.json`` maps a corpus image *handle* to its pinned URL (the
    committed selection); the cache holds bytes keyed by a URL hash. A handle is
    used as the key (not the search term) so the same logical image keeps a
    stable pin even if a card's search term is reworded."""

    resolved_path: Path
    cache_dir: Path

    def load_pins(self) -> dict[str, str]:
        if self.resolved_path.exists():
            return json.loads(self.resolved_path.read_text())
        return {}

    def save_pins(self, pins: dict[str, str]) -> None:
        self.resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolved_path.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")

    def fetch_bytes(self, url: str) -> bytes:
        """Pinned-URL bytes, locally cached (gitignored). Never committed."""
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{key}.img"
        if cache_file.exists():
            return cache_file.read_bytes()
        data = _get(url).content
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
        return data
