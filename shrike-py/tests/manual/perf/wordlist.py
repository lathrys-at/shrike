"""Large public-domain wordlist for the perf corpus.

The synthetic corpus and the search workload draw their vocabulary from a large
real English wordlist so character trigrams are distributed like real text. A
tiny vocabulary made trigrams pathologically shared — a fuzzy trigram ``OR``
matched almost the whole collection — which is not how a real Anki collection
behaves and made the search benchmark a degenerate worst case.

Following the search-quality lane's no-redistribute pattern, the wordlist bytes
are not committed: only the pinned source (``wordlist_source.json``: URL +
immutable commit + SHA-256) and the ``ASSETS.md`` attribution are.
:func:`ensure_wordlist` downloads it once into the gitignored cache and verifies
the hash; :func:`load_wordlist` reads the cache and, when it is absent, returns a
caller-supplied fallback so offline/manual runs still work (with reduced realism).
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
_CACHE_DIR = _SHRIKE_PY.parent / ".cache" / "perf" / "wordlist"
_MANIFEST_PATH = Path(__file__).resolve().parent / "wordlist_source.json"

# Word filter shared by the cache parse and the fallback so both vocabularies are
# shaped identically: lowercase a-z only (the trigram tokenizer is case-folding
# and the corpus text is ASCII), length in range. The 3-char floor is the
# trigram minimum; the ceiling drops the rare ultra-long words that don't read
# like study-note terms.
_MIN_LEN = 3
_MAX_LEN = 15


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
    """Where the downloaded wordlist is cached (gitignored)."""
    return _CACHE_DIR / "words_alpha.txt"


def ensure_wordlist(timeout: float = 60.0) -> Path:
    """Ensure the pinned wordlist is present in the cache and verified, downloading
    it once if absent. Real perf runs call this at startup so the corpus is built
    from the full vocabulary; tests do not (they fall back).

    Raises on a download/verification failure rather than silently degrading —
    a perf run must not quietly profile against the small fallback vocabulary.
    """
    dest = cache_path()
    manifest = _manifest()
    want = str(manifest["sha256"])
    if dest.is_file() and _sha256(dest) == want:
        return dest
    url = str(manifest["url"])
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("fetching perf wordlist %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — pinned https URL
        tmp.write_bytes(resp.read())
    got = _sha256(tmp)
    if got != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"perf wordlist sha256 mismatch: got {got}, expected {want} (from {url})"
        )
    tmp.replace(dest)  # atomic: a half-written download never looks complete
    return dest


def _parse(text: str) -> list[str]:
    """Lowercase a-z words within the length range, in file order (deterministic).
    Handles the source's CRLF line endings."""
    out: list[str] = []
    for raw in text.splitlines():
        w = raw.strip().lower()
        if _MIN_LEN <= len(w) <= _MAX_LEN and w.isascii() and w.isalpha():
            out.append(w)
    return out


def load_wordlist(fallback: list[str]) -> list[str]:
    """The active vocabulary: the cached full wordlist when present, else
    ``fallback`` (filtered to the same shape). Never downloads — call
    :func:`ensure_wordlist` first for the full list. ``SHRIKE_PERF_NO_WORDLIST=1``
    forces the fallback (used by tests that must stay offline and fast)."""
    if os.environ.get("SHRIKE_PERF_NO_WORDLIST") != "1":
        dest = cache_path()
        if dest.is_file():
            words = _parse(dest.read_text())
            if words:
                return words
            logger.warning("perf wordlist cache %s parsed empty; using fallback", dest)
    # Filter the fallback the same way so the two vocabularies are interchangeable.
    return _parse("\n".join(fallback))


def fingerprint(words: list[str]) -> str:
    """A short content fingerprint of the active vocabulary, folded into the
    corpus cache key so a fallback-built corpus never aliases a full-built one
    (and a wordlist change invalidates cached corpora). Cheap: size + boundary
    and midpoint words, not the whole list."""
    if not words:
        return "empty"
    probe = f"{len(words)}:{words[0]}:{words[len(words) // 2]}:{words[-1]}"
    return hashlib.sha256(probe.encode()).hexdigest()[:12]
