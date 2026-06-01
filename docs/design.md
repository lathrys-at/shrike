# Design decisions

A log of non-obvious design choices for *shipped* work — the "why", kept out of
both the code (which shows the "what") and the issue tracker (which holds *future*
work). New entries go on top of their section. For the mechanics of how a feature
works, see `CLAUDE.md`'s "Key technical details"; this file is the reasoning that
isn't reconstructable from the code.

## Semantic search & the vector index

### The index is a derived cache, never a co-equal store

The Anki collection (SQLite) is always the source of truth; the USearch index is
a rebuildable projection of it. The index may lag the collection (stale search
results) but the collection never lags the index (data ops are always correct).
This is what lets drift detection be a single `col.mod` comparison with a
background rebuild, rather than per-note reconciliation. The full rationale,
drift-detection scheme, and model-fingerprint design live under "Vector index and
consistency" in `CLAUDE.md` — not duplicated here.

### Contextual upsert returns neighbours; it makes no suggestions

`upsert_notes` returns, for each created/updated note, the *k* most similar
existing notes (`{id, score, tags}`, defaults `top_k_neighbors=5`,
`neighbor_threshold=0.5`) — the same similarity operation `search_notes` runs,
queried by the upserted note's own content, with the batch's own notes excluded
from each other's results.

It returns **raw neighbour data and stops there.** The server suggests no tags,
flags no duplicates, makes no decisions. The reasoning: the server can't know the
caller's intent, and baking in a policy (auto-tagging, dup-rejection thresholds)
would be a guess that's wrong half the time and impossible to override cleanly.
Handing back the neighbours lets an LLM-driven caller ground new cards in the
collection's existing taxonomy (reuse tags that already exist) and spot
near-duplicates — while the *policy* for what to do with that lives in the skill,
not the server.

### Duplicate detection is not a separate feature

There is no dedicated duplicate-detection endpoint, and there won't be one. A high
similarity score in `search_notes` results or in upsert neighbours *is* the
duplicate signal; the caller applies its own threshold. Adding a second code path
that re-implements the same cosine-similarity lookup with a built-in cutoff would
be redundant surface area with a worse interface (a hard-coded threshold the
caller can't see or tune).

## Tags

### Tags are a full replace, never an add/remove merge

Everywhere Shrike sets tags — `upsert_notes` partial updates (`{id, tags}`),
`shrike note update --tags`, and the bulk `shrike note tag <ids> --set a,b`
(`--set ""` clears) — the note ends up with *exactly* the set you sent. There is
no additive or subtractive mode.

We considered and rejected an additive/subtractive `mode` parameter on
`upsert_notes`. It's precisely the "bag of optionals / hidden state" the
`schemas.py` house style warns against — the meaning of the `tags` field would
silently depend on a second field. And it doesn't match the actual workflow: the
skill's tag work is a create-time, full-set decision ("this card's tags are X, Y,
Z"), not a retroactive merge into whatever was already there.

### Retroactive collection-wide tag cleanup is deliberately unbuilt

A true add/remove sweep over many *existing* notes (backed by Anki's `bulk_add` /
`bulk_remove`) is a different operation from create-time tagging, and it's out of
scope until a concrete need appears. `shrike note tag --set` is bulk *replace*
sugar, not cleanup — it sets the same exact tag set across many notes, which is
still a full-set decision, just applied widely.
