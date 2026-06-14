#!/usr/bin/env python3
"""Resolve + pin the search-quality corpus's Wikimedia Commons images (#559 PR2).

Corpus tooling (NOT a test): for every ``source: commons`` image in
``eval/search_quality/manifest.json`` it resolves a real Commons file, **pins
the URL** in ``eval/search_quality/resolved_urls.json`` (committed), and writes
``eval/search_quality/ASSETS.md`` — the per-image attribution table (Commons
page / license / author) the AGPL repo needs since it redistributes no bytes.

Image **bytes are never committed**: the manual suite downloads them on demand
into the gitignored ``eval/search_quality/cache/`` at run time, exactly the
``eval/multimodal/`` pattern. PD / PD-art / CC0 are preferred at corpus design
time; CC-BY-SA is allowed but is attributed here.

Run::

    python scripts/eval_search_quality_corpus.py            # resolve missing, write ASSETS.md
    python scripts/eval_search_quality_corpus.py --refresh  # re-resolve every term
    python scripts/eval_search_quality_corpus.py --backfill-multimodal  # + eval/multimodal pins

The pinned URLs make a replay reproducible; ``--refresh`` re-resolves (a Commons
file can be deleted/renamed). The licensing block ALWAYS re-reads metadata so
``ASSETS.md`` stays current even when a URL is already pinned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "eval" / "search_quality"
MANIFEST = EVAL_DIR / "manifest.json"
RESOLVED = EVAL_DIR / "resolved_urls.json"  # committed: pins image selection
ASSETS = EVAL_DIR / "ASSETS.md"
MULTIMODAL_RESOLVED = ROOT / "eval" / "multimodal" / "resolved_urls.json"

# Canonical pins for handles where a fuzzy Commons search picks the wrong file
# (a desmosome instead of the cell, a rock formation instead of a violin). These
# handles back explicit recall queries, so the image MUST be the right subject —
# pin a specific well-known file by its ``File:`` page TITLE. Title resolution is
# robust where a hardcoded thumb URL isn't: Commons renames thumb paths and
# rejects arbitrary thumb sizes (HTTP 400/404), but the API always returns the
# file's CURRENT valid URL for a title. PD/PD-art/CC0 preferred (see ASSETS.md).
CANONICAL_TITLES = {
    "animal_cell": "File:Animal_cell_structure_en.svg",  # Public domain — LadyofHats' labeled cell
    "sunflower": "File:Sunflower_sky_backdrop.jpg",  # GFDL/CC — Fir0002's sunflower
    "violin": "File:Violin_VL100.png",  # CC0 — a clean violin render
    "coffee": "File:A_small_cup_of_coffee.JPG",  # CC BY-SA — a real coffee cup
}


def _commons_entries(manifest: dict) -> list[tuple[str, str]]:
    """(handle, search_term) for every commons image in the manifest, deduped on handle."""
    seen: dict[str, str] = {}
    for card in manifest.get("cards", []):
        for media in card.get("media", []):
            if media.get("source") != "commons":
                continue
            handle = media["handle"]
            spec = media.get("spec", {})
            term = spec.get("search_term") or spec.get("url", "")
            seen.setdefault(handle, term)
    return sorted(seen.items())


def resolve_corpus(refresh: bool) -> dict[str, dict]:
    """Resolve every corpus image → {handle: {url, commons_page, license, author}}.

    A handle already pinned in resolved_urls.json keeps its URL (unless
    --refresh), but its metadata is always re-read so ASSETS.md is current."""
    from tests.search_quality.commons import resolve_asset

    manifest = json.loads(MANIFEST.read_text())
    pins: dict[str, str] = json.loads(RESOLVED.read_text()) if RESOLVED.exists() else {}
    out: dict[str, dict] = {}
    for handle, term in _commons_entries(manifest):
        pinned = pins.get(handle)
        title = CANONICAL_TITLES.get(handle)
        try:
            if pinned and not refresh:
                # An existing pin replays verbatim (canonical or not — once a
                # title resolved to a working URL it's pinned like any other).
                asset = resolve_asset(pinned, is_url=True)
            elif title is not None:
                # A canonical handle resolves by File: title (robust) → pin the
                # CURRENT valid URL the API returns, overriding a noisy search.
                asset = resolve_asset(title, is_title=True, width_hint=800)
                pins[handle] = asset.url
            else:
                asset = resolve_asset(term)
                pins[handle] = asset.url
            out[handle] = {
                "url": asset.url,
                "commons_page": asset.commons_page,
                "license": asset.license,
                "author": asset.author,
            }
            flag = "PIN" if (pinned and not refresh) else "NEW"
            print(f"  [{flag}] {handle:16} {asset.license:18} {term[:40]}")
        except Exception as e:  # noqa: BLE001 — report + continue; a missing image is skipped
            print(f"  [ERR] {handle:16} {e}")
    RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    return out


def _multimodal_backfill() -> list[dict]:
    """Re-read attribution for the reused eval/multimodal images (closes the
    pre-existing attribution gap — they were pinned with no license record)."""
    from tests.search_quality.commons import resolve_asset

    if not MULTIMODAL_RESOLVED.exists():
        return []
    pins: dict[str, str] = json.loads(MULTIMODAL_RESOLVED.read_text())
    rows = []
    for term, url in sorted(pins.items()):
        try:
            asset = resolve_asset(url, is_url=True)
            rows.append(
                {
                    "handle": term,
                    "url": url,
                    "commons_page": asset.commons_page,
                    "license": asset.license,
                    "author": asset.author,
                }
            )
            print(f"  [MM ] {term[:28]:28} {asset.license}")
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {term}: {e}")
    return rows


def write_assets(resolved: dict[str, dict], multimodal: list[dict]) -> None:
    lines = [
        "# Search-quality corpus image assets (#559)",
        "",
        "These images are **not redistributed** in this repository. The manual",
        "search-quality suite (`SHRIKE_SEARCH_QUALITY=1`) resolves each via the",
        "Wikimedia Commons API and downloads the bytes on demand into the",
        "gitignored `eval/search_quality/cache/` — only the pinned URLs",
        "(`resolved_urls.json`) and this attribution table are committed.",
        "",
        "Licensing preference is **public domain / PD-art / CC0**; a few",
        "CC-BY-SA images are used where no public-domain file fits the corpus",
        "need, and are attributed below. Each row links its Wikimedia Commons",
        "page, where the full license terms and authorship live.",
        "",
        "## Corpus images (`eval/search_quality/`)",
        "",
        "| Handle | License | Author | Commons page |",
        "| --- | --- | --- | --- |",
    ]
    for handle in sorted(resolved):
        a = resolved[handle]
        page = a["commons_page"]
        page_md = f"[{page.rsplit('/', 1)[-1][:48]}]({page})" if page.startswith("http") else page
        lines.append(f"| `{handle}` | {a['license']} | {a['author']} | {page_md} |")

    if multimodal:
        lines += [
            "",
            "## Reused multimodal-eval images (`eval/multimodal/`)",
            "",
            "Attribution for the images the multimodal eval (#162) pinned without a",
            "license record — backfilled here (the suites share these Commons files).",
            "",
            "| Search term | License | Author | Commons page |",
            "| --- | --- | --- | --- |",
        ]
        for a in multimodal:
            page = a["commons_page"]
            page_md = (
                f"[{page.rsplit('/', 1)[-1][:48]}]({page})" if page.startswith("http") else page
            )
            lines.append(f"| {a['handle']} | {a['license']} | {a['author']} | {page_md} |")

    lines.append("")
    ASSETS.write_text("\n".join(lines))
    print(f"\nwrote {ASSETS.relative_to(ROOT)} ({len(resolved)} corpus images)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-resolve every term (re-pin URLs)")
    ap.add_argument(
        "--backfill-multimodal",
        action="store_true",
        help="also attribute the reused eval/multimodal images in ASSETS.md",
    )
    args = ap.parse_args()

    print(f"Resolving corpus images from {MANIFEST.relative_to(ROOT)} ...")
    resolved = resolve_corpus(args.refresh)
    multimodal = _multimodal_backfill() if args.backfill_multimodal else []
    write_assets(resolved, multimodal)


if __name__ == "__main__":
    main()
