# Fuzzy-recall eval assets (#927)

The fuzzy-recall eval injects real human misspellings (alongside the synthetic
keyboard/transposition/deletion edits) into queries drawn from the perf corpus, so
recall is measured against the spelling errors a synthetic model cannot reproduce.

The misspellings list bytes are **not redistributed** in this repository. The
source is Wikipedia's "Lists of common misspellings/For machines", which is
CC-BY-SA — so, following the perf wordlist's pin-and-cache pattern, only the
pinned source (below, in `misspellings_source.json`: a fixed revision `oldid` +
the raw-export URL + SHA-256) and this attribution are committed. `misspellings.py`
downloads that exact revision on demand into the gitignored
`.cache/fuzzy_recall/misspellings/` and verifies it against the pinned SHA-256.
When the cache is absent (offline / first run of a manual test), a small embedded
fallback of obvious, uncopyrightable spelling facts is used instead, with fewer
real-misspelling queries.

The corpus vocabulary asset (`words_alpha.txt`) is the perf lane's; see
`tests/manual/perf/ASSETS.md`.

| Asset | License | Source | Pin |
| --- | --- | --- | --- |
| Common misspellings (~4.3k pairs) | CC-BY-SA 4.0 | [Wikipedia: Lists of common misspellings/For machines](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines) | revision `oldid=1199637275`, SHA-256 `3b6a9290e5aaad968da7ec769a5174adb45247688ef7ae097fd2dd80237a3050` |
