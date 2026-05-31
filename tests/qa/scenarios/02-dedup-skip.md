# 02 — Dedup (already covered)

**Exercises:** the look-before-you-write habit — when the source material is
already covered by existing notes, the skill should detect that (via pre-create
search and/or the upsert neighbors) and **skip or merge** rather than create
near-duplicates.

**Why this material:** every fact below is already a note in the fixture's
`Biology` deck, so the correct outcome is *roughly zero new cards*.

## Prompt

```
Make Anki cards from this summary of cellular respiration: Mitochondria generate
most of the cell's ATP through aerobic respiration. The proton gradient that
drives ATP synthesis is maintained across the inner mitochondrial membrane. The
citric acid cycle takes place in the mitochondrial matrix.
```

## Expected outcome

- The skill **searches first** and recognizes existing coverage:
  - "mitochondria produce ATP" ↔ *"What is the primary function of the
    mitochondria?"*
  - "proton gradient … inner membrane" ↔ *"Across which mitochondrial structure
    is the proton gradient … maintained?"*
  - "citric acid cycle … matrix" ↔ the existing matrix cloze.
- **No duplicate cards created.** Acceptable outcomes: create nothing and report
  the existing coverage; or, if a fact is phrased to add genuinely new detail,
  *update* the existing note rather than adding a parallel one.
- It does **not** rely on the similarity score alone — it reads the matched
  content and reasons about overlap.
- **Report** clearly states what already existed and that it avoided
  duplicating.

## Fixture dependencies

Relies on the planted Biology dedup targets (see `_dedup_targets` in
`../collection.json`): mitochondria→ATP, inner-membrane proton gradient, and the
citric-acid-cycle matrix cloze.
