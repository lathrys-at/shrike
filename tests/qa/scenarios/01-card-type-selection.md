# 01 — Card-type selection

**Exercises:** whether the skill reasons about the *right kind* of card per fact
(one-way Q/A vs. cloze with separate deletions vs. ordered-sequence cloze)
instead of defaulting everything to front/back — and lands the cards in the
existing `Biology` deck with tags rather than spawning a sub-deck.

**Why this material:** human-heart anatomy isn't in the fixture, so there's
little dedup noise — this isolates the type-selection decision.

## Prompt

```
Here are my notes on the human heart — make some Anki cards and add them to my collection:
- The heart has four chambers: left and right atria (upper), left and right ventricles (lower).
- Blood flows: right atrium → right ventricle → lungs → left atrium → left ventricle → body.
- The sinoatrial (SA) node is the heart's natural pacemaker.
- The left ventricle has the thickest wall because it pumps blood to the whole body.
```

## Expected outcome

- **Type choices fit the material:**
  - The four chambers (a set) → either one cloze note with a *separate* deletion
    per chamber (`{{c1}}`–`{{c4}}`), or atomic Q/A — not one "name all four"
    card.
  - The blood-flow path (an ordered sequence) → cloze that preserves the order,
    or a small number of atomic steps — not a single dump.
  - SA node = pacemaker, and "thickest wall + why" → one-way Q/A (discrete
    facts).
- **Placement:** all cards in the existing **`Biology`** deck. **No** new deck
  (no `Heart`, no `Biology::Anatomy::…`).
- **Tags:** drawn from / consistent with the existing biology vocabulary
  (`biology`, plus a reasonable topic tag); not a tag explosion.
- **Report** names what was created, where, and any judgment calls.

## Fixture dependencies

`Biology` deck and its tag vocabulary exist; the heart facts themselves do not,
so nothing should be flagged as a duplicate.
