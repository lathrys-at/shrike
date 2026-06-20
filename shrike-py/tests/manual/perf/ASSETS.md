# Perf-corpus wordlist asset (#911)

The synthetic perf corpus and its search workload draw their vocabulary from a
large real English wordlist, so character trigrams are distributed like real
text rather than collapsing onto a tiny shared set (an 84-word vocabulary made a
fuzzy trigram `OR` match almost the whole collection — unrepresentative of a real
Anki collection).

The wordlist bytes are **not redistributed** in this repository. Following the
search-quality lane's pattern, only the pinned source (below, in
`wordlist_source.json`) and this attribution are committed; `wordlist.py`
downloads the bytes on demand into the gitignored
`.cache/perf/wordlist/` and verifies them against the pinned SHA-256. When the
cache is absent (offline / first run of a manual test), a small embedded
fallback vocabulary is used instead, with reduced realism.

| Asset | License | Source | Pin |
| --- | --- | --- | --- |
| `words_alpha.txt` (~370k words) | Unlicense (public domain) | [dwyl/english-words](https://github.com/dwyl/english-words) | commit `8179fe68775df3f553ef19520db065228e65d1d3`, SHA-256 `3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48` |
