# 03 — Honor a requested type, but keep discipline

**Exercises:** when the user explicitly asks for a card *type*, the skill honors
it — but still applies the writing standards, and still checks for existing
coverage. Here the requested type (cloze) is a slightly awkward fit for discrete
date facts, and three of the four facts already live in the `History` deck.

## Prompt

```
Make cloze cards for these: the French Revolution began in 1789; the Bastille
was stormed on 14 July 1789; the US declared independence in 1776; Napoleon was
crowned Emperor in 1804.
```

## Expected outcome

- **Honors the cloze request** — produces cloze notes (not Q/A), one disciplined
  deletion per fact (the date/event), each answerable from its sentence; no
  over-clozing.
- **Catches the overlap:** "French Revolution began in 1789", "Bastille stormed
  14 July 1789", and "US declared independence in 1776" already exist in
  `History`. The skill should skip/flag those rather than duplicate — even
  though they exist as Q/A or a different phrasing, it recognizes the same fact.
- **Creates the genuinely new one:** Napoleon crowned Emperor in 1804 (not in
  the fixture) → a new cloze card in `History`.
- **Placement/tags:** `History` deck, existing history/revolution tag
  vocabulary; no new deck.
- If cloze genuinely fights a fact, it may say so — but the user's request sets
  the type.
- **Report** distinguishes created vs. skipped-as-existing.

## Fixture dependencies

`History` deck contains 1789 (French Revolution), the Bastille (14 July 1789),
and 1776 (US independence). Napoleon's 1804 coronation is **not** present.
