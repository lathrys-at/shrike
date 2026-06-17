# 04 — Two-way / reversed cards for vocabulary

**Exercises:** the skill should pick a **reversed** note type for genuinely
bidirectional associations (a vocabulary word ↔ its meaning, needed both ways),
using the existing `Basic (and reversed card)` type rather than one-way Q/A or a
newly-invented type — and it should *not* reverse associations that are only
useful one way.

**Why this material:** the words below aren't in the fixture, so this isolates
the type choice. (`el libro = book` would collide with an existing card — left
out on purpose so dedup doesn't muddy the signal.)

## Prompt

```
Add these Spanish words to my collection: gracias = thank you, por favor = please,
la mesa = table, la ventana = window.
```

## Expected outcome

- **Two-way cards:** uses the existing **`Basic (and reversed card)`** note type
  (front ⇄ back), matching how the rest of the `Spanish` deck is built — so each
  word is reviewable Spanish→English *and* English→Spanish.
- It does **not** invent a new note type, and does not settle for one-way Q/A for
  vocabulary.
- **Placement:** `Spanish` deck; tags consistent with the existing vocab
  vocabulary (`spanish`, `vocabulary`, and a sensible topic tag like
  `food`/`household` where it fits the established pattern).
- **Watch-point (the negative case):** if you instead feed an *asymmetric* fact
  (e.g. "the Spanish for 'to eat' is 'comer'" framed as needing only one
  direction, or a definitional fact), the skill should keep it **one-way** and
  not reflexively reverse it. Worth a follow-up prompt to confirm it doesn't
  over-apply reversal.

## Fixture dependencies

`Spanish` deck uses the `Basic (and reversed card)` type with `spanish` /
`vocabulary` (+ `food`/`animals`/`verbs`) tags. The four prompt words are not in
the fixture.
