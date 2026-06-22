"""Real common-misspellings pairs for the fuzzy-recall eval.

The synthetic typo injector exercises the trigram-overlap mechanism with
keyboard/transposition/deletion edits, but real human spelling errors
(``recieve`` for ``receive``, ``definately`` for ``definitely``) follow patterns a
synthetic model does not reproduce. This module supplies a curated
``misspelling -> correction`` list so the eval can inject genuine misspellings
wherever a corpus note contains the corrected word.

Following the perf wordlist's no-redistribute pattern, the list bytes are **not
committed**: only the pinned source (``misspellings_source.json``: a fixed
Wikipedia revision ``oldid`` + URL + SHA-256) and the ``ASSETS.md`` attribution
are. The source is Wikipedia's "Lists of common misspellings/For machines"
(CC-BY-SA), so the bytes must not enter the AGPL tree; :func:`ensure_misspellings`
downloads the pinned revision once into the gitignored cache and verifies the
hash. When the cache is absent (offline / first run), :func:`load_misspellings`
returns a small embedded fallback of obvious, uncopyrightable spelling facts so
the eval still runs (with fewer real-misspelling queries).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# shrike-py/ (mirrors corpus.py's anchor); the gitignored cache is repo-root/.cache.
_SHRIKE_PY = Path(__file__).resolve().parents[3]
_CACHE_DIR = _SHRIKE_PY.parent / ".cache" / "fuzzy_recall" / "misspellings"
_MANIFEST_PATH = Path(__file__).resolve().parent / "misspellings_source.json"

# The trigram floor (a pair must have >= 3 chars to produce any trigram) and a
# ceiling that drops the rare ultra-long entries, matching the corpus vocabulary's
# shape so injected misspellings read like the rest of the text.
_MIN_LEN = 3
_MAX_LEN = 20

# A tiny embedded fallback of obvious spelling facts (uncopyrightable — these are
# not the licensed list, just the canonical errors anyone would name), so an
# offline run still injects SOME real misspellings. Lowercase a-z only.
_FALLBACK_PAIRS: dict[str, str] = {
    "recieve": "receive",
    "definately": "definitely",
    "seperate": "separate",
    "occured": "occurred",
    "untill": "until",
    "wierd": "weird",
    "accomodate": "accommodate",
    "acheive": "achieve",
    "beleive": "believe",
    "neccessary": "necessary",
    "occassion": "occasion",
    "publically": "publicly",
    "tommorow": "tomorrow",
    "wich": "which",
    "thier": "their",
    "alot": "lot",
    "embarass": "embarrass",
    "goverment": "government",
    "enviroment": "environment",
    "calender": "calendar",
}


def _manifest() -> dict[str, object]:
    data: dict[str, object] = json.loads(_MANIFEST_PATH.read_text())
    return data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path() -> Path:
    """Where the downloaded misspellings list is cached (gitignored)."""
    return _CACHE_DIR / "common_misspellings.txt"


def ensure_misspellings(timeout: float = 60.0) -> Path:
    """Ensure the pinned misspellings list is present in the cache and verified,
    downloading the fixed Wikipedia revision once if absent.

    Raises on a download/verification failure rather than silently degrading, so a
    full eval run does not quietly fall back to the small embedded list.
    """
    dest = cache_path()
    manifest = _manifest()
    want = str(manifest["sha256"])
    if dest.is_file() and _sha256(dest) == want:
        return dest
    url = str(manifest["url"])
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("fetching common-misspellings list %s -> %s", url, dest)
    # Wikipedia rejects the default urllib User-Agent (403); its policy requires a
    # descriptive one identifying the client and a contact, so send that.
    request = urllib.request.Request(  # noqa: S310 — pinned https URL
        url,
        headers={
            "User-Agent": "shrike-fuzzy-recall-eval/1.0 (https://github.com/lathrys-at/shrike)"
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 — pinned https URL
        tmp.write_bytes(resp.read())
    got = _sha256(tmp)
    if got != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"misspellings list sha256 mismatch: got {got}, expected {want} (from {url})"
        )
    tmp.replace(dest)  # atomic: a half-written download never looks complete
    return dest


def _parse(text: str) -> dict[str, str]:
    """``misspelling -> correction`` pairs from the source's
    ``misspelling->correction`` lines. The source carries a wiki-markup preamble
    (skipped: only ``a->b`` lines parse) and some entries list multiple
    comma-separated corrections — the FIRST correction is taken (the primary, and
    the one whose presence in a note keys the injection). Both sides are
    lowercased a-z within the length range; an entry whose two sides share no
    differing characters (or are equal) is dropped (not a usable typo)."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if "->" not in line:
            continue
        wrong, _, rest = line.partition("->")
        wrong = wrong.strip().lower()
        # Multiple corrections are comma-separated; take the first.
        right = rest.split(",")[0].strip().lower()
        if not (wrong.isascii() and wrong.isalpha() and right.isascii() and right.isalpha()):
            continue
        if not (_MIN_LEN <= len(wrong) <= _MAX_LEN and _MIN_LEN <= len(right) <= _MAX_LEN):
            continue
        if wrong == right:
            continue
        # First correction wins on a duplicate misspelling key (deterministic: the
        # source lists each misspelling once, but guard against a stray repeat).
        out.setdefault(wrong, right)
    return out


def load_misspellings() -> dict[str, str]:
    """The active ``misspelling -> correction`` map: the cached full list when
    present, else the embedded fallback. Never downloads — call
    :func:`ensure_misspellings` first for the full list.
    ``SHRIKE_FUZZY_NO_MISSPELLINGS=1`` forces the fallback (used by tests that must
    stay offline and fast)."""
    if os.environ.get("SHRIKE_FUZZY_NO_MISSPELLINGS") != "1":
        dest = cache_path()
        if dest.is_file():
            pairs = _parse(dest.read_text())
            if pairs:
                return pairs
            logger.warning("misspellings cache %s parsed empty; using fallback", dest)
    # Drop any identity pair (key == value) so the fallback can never carry a no-op
    # "misspelling" that would inject nothing.
    return {wrong: right for wrong, right in _FALLBACK_PAIRS.items() if wrong != right}
