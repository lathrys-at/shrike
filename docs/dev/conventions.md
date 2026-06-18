# Code conventions

These are enforced, not aspirational: conformance is a required check in every code
review (see CLAUDE.md), so a breach is a review finding.

## Style

- **Type annotations on every function** (mypy runs with `disallow_untyped_defs`).
- **Ruff** for linting (rules E, F, W, I, UP, B, SIM) and formatting, line length
  100.
- **`raise ... from err`** in except blocks (ruff B904).
- **`contextlib.suppress`** instead of bare `try/except/pass`.
- **`datetime.UTC`**, not `timezone.utc` (ruff UP017).
- Batch operations use **per-item try/except** so one failure doesn't block the
  batch; results carry a per-item `status`.

## Comments document the code, not its history

Comments document the code; **rationale and history live elsewhere** — in issues,
PR bodies, and [`decisions.md`](decisions.md). `git blame` already traces when and
why a line changed, so a comment that re-states that goes stale the moment it is
written ("when was *recently*?").

**Keep** a comment only if it is:

- **future-facing code documentation** — a non-obvious invariant, a
  looks-wrong-but-correct-because-X guard, a `// SAFETY:` justification, or
  algorithmic rationale a future reader genuinely needs; or
- **interface documentation** — a docstring on a public module, class, or function
  (the contract).

**Drop** the historical/contextual narrative: `(#NNN)` issue citations, "changed in
the X pass", "as of today / for now", and session-generated context-vomit.

The test is the comment's *value*, not a keyword. A comment that explains the code
stays even if it happens to cite an issue — strip the historical scaffolding (the
`(#NNN)`, the "as of" framing) and keep the explanation. A comment whose value is
*purely* historical drops entirely. The same "no historical narrative" trim applies
*within* an interface docstring. It is judgment per comment, never a blunt regex
sweep — a naive "delete every line with `#NNN`" would destroy genuinely valuable
explanations (the `// SAFETY:` blocks, the CVE-class rationale in path handling).

## Schema house style: make illegal states unrepresentable

The wire models in `schemas.py` follow one rule: when a field's presence is
*correlated* with another (a hidden state), model it as a **discriminated union**,
never a bag of optionals.

The pattern is an `Annotated` type alias. Each variant is a `BaseModel` with a
`Literal` discriminator field, and:

```python
Thing = Annotated[VariantA | VariantB, Field(discriminator="status")]
```

Validate the alias with `TypeAdapter(Thing).validate_python(...)` (a model *field*
typed as `Thing` validates automatically). Examples: per-item results
(`UpsertNoteResult` — success has `id` + `neighbors`, error has `index` + `error`),
`IndexStatus` (`IndexUnavailable | IndexBuilding | IndexReady | IndexErrored`), and
the `/index/rebuild` + `/embedding/*` endpoint responses.

Two fields that always appear or vanish *as a pair* are the same smell at smaller
scale — group them into a nested sub-model, not two optionals. A bare `X | None` is
reserved for *genuinely independent* optionality (a datum absent on its own);
annotate why, so it reads as deliberate.

Response models carry **no `error` field**. A whole-call failure (bad input,
unhandled exception) is raised in the tool and surfaces as an MCP `isError` result,
which `ShrikeClient._call` turns into a `ServerError`. Expected bad input raises
`ToolInputError` (logged without a traceback); genuine bugs log with one. The only
optional advisory on a success response is `message`.

Input bounds (`limit` 1–200, `top_k` 1–50, batch sizes ≤100/≤10) are declared as
`Annotated[..., Field(ge=, le=, min_length=, max_length=)]` on the tool params, so
FastMCP **rejects** out-of-range input rather than silently clamping. Optional list
filters use `Field(default_factory=list)` so they render as a plain array, not a
noisy `anyOf:[array, null]`.

## Performance

Ground performance decisions at a **100k-note collection**. These rules came out
of the kernel performance audit; the recurring failure modes and the rules that
prevent them:

- **No collection reads inside per-item loops.** The N+1 is the repeat offender: a
  singleton `note_dicts`/`note_texts` per candidate pays two SQL queries plus a full
  deck/notetype enumeration each, serialized on the collection actor. Discover the
  id set first, then do ONE batched read (`read_notes_batch`, `note_dicts(&ids)`,
  `texts_for_source_for_notes`) and assemble from the map. **When porting policy
  between layers, port its batching with it.**
- **Read only what the op needs.** Prefer scoped variants over full-collection
  renders (`note_image_refs`, the `any_tagged` probe, notes-scoped derived reads).
  Push a pre-filter into SQL only when its semantics match the Rust side exactly.
- **Per-op tails do no O(collection) work.** Derived signals (tag centroids) refresh
  in a coalescing background task behind a cheap relevance probe; the op tail only
  *requests*. Boot/rebuild paths keep synchronous refreshes so "ready" means ready.
- **Never hold a lock across file writes or compute.** Snapshot the small shared
  state under the lock, write outside it; serialize savers with a dedicated guard;
  blocking fs work rides `spawn_blocking`.
- **One transaction per batch; prepared statements in row loops.** A journal commit
  (fsync) per item is the hidden cost — `ingest_many` and `set_note_tags_bulk` batch
  it away. `Connection::execute` re-prepares per call, so loops use `prepare_cached`.
- **Skip provably-identity work — but prove it from the pinned source.** The
  strip-skip (no `<` and no `&` → Anki's stripper is byte-identity) was verified
  against Anki's own gate and is pinned by a test. A skip predicate justified only
  empirically is a future correctness bug.
- **Hand out views and `Arc`s, not clones; bound unbounded expansions.** Arc'd
  per-notetype field lists, `Cow` pass-throughs, per-batch lookup memos, and
  ceilings with deterministic sampling where an input scales with the collection.

## Logging

Logging is configured in `platform/log.py`. Format, parsing, and styling all live
there — formatting knowledge should not spread across CLI commands.

Use **per-module loggers**: `shrike.server`, `shrike.kernel`, `shrike.tools`,
`shrike.collection`, `shrike.embedding`, `shrike.index`, `shrike.derived`,
`shrike.daemon`. This is what makes per-logger level overrides
(`logging.levels.shrike.collection: debug`) work. Never log under a bare `shrike`
logger. Native (Rust) tracing forwards through pyo3-log under the crate's module
path, so the same overrides govern it.

Principles:

1. **Say what happened, with the key context.** "Collection ready: 847 notes, 5
   decks, 12 note types", not "Collection opened". Include counts, IDs, paths,
   durations.
2. **Log operational boundaries at INFO** — startup, shutdown, configuration
   loaded, server listening. These are the anchors you scan for.
3. **One INFO line per served call.** The adapter (`_safe_tool`) emits it: tool
   name + params + outcome + duration, e.g.
   `list_notes deck='Test' limit=50 -> 3/3 notes (12ms)`. Actions contribute the
   outcome fragment via `note_outcome(...)`; custom routes get the same from
   `_guard`. Anything else logged while serving is WARNING/ERROR (exceptional) or
   DEBUG (internals) — never a second INFO line.
4. **DEBUG for internals** — individual note creates, query construction, index
   lookups.
5. **WARNING for recoverable failures that deserve attention** — a single note
   failing in a batch, a rejected note-type update. Not normal empty results.
6. **`%s` formatting, not f-strings** — lazy evaluation.
7. **Don't repeat what the logger name says.** The line already shows
   `shrike.tools`; don't prefix the message with "tools:".
8. **Log the signal name on shutdown**, so you know whether it was SIGTERM
   (normal), SIGINT (Ctrl+C), or something else.

The log file format is fixed (`parse_log_line` / `style_log_line` know it):

```
2025-05-24T10:30:00 INFO  shrike.tools  list_notes deck=Test limit=50
```

Timestamp is `%Y-%m-%dT%H:%M:%S` (19 chars), level left-padded to 5 chars, logger
and message separated by a double space. Keep the parser and styler in sync if you
change it.
