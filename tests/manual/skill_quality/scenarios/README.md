# QA scenarios

Hand-run test scenarios for the **create-cards** skill. Each one is a realistic
user prompt plus the behaviour to look for — they exercise the skill's judgment
(card-type choice, dedup, deck/tag restraint), which is qualitative and not a
fit for the automated suites.

These are **manual**. Nothing here runs in `pytest` or CI.

## How to run one

1. Start a **fresh** QA server so the fixture is in a known state:
   ```bash
   ./bazel run //scripts:serve_text_onnx -- --seed qa --daemon
   ```
   (Or, for a GGUF model, drive the eval harness `../run.py`, which sets up
   the fixture + server with `SHRIKE_EMBEDDING_MODEL`. See `../README.md`.)
2. In a session that has the **create-cards** skill loaded and can reach the QA
   server (the `shrike` CLI, or the MCP tools via `mcp-remote`), paste the
   scenario's **Prompt**.
3. Check the result against **Expected outcome**. Useful inspection:
   ```bash
   shrike note list --since <today>          # what got created
   shrike info --decks --tags                # deck sprawl? new/synonym tags?
   shrike note search "<concept>" --json      # what the skill would have seen
   ```
4. **Reset between scenarios** — relaunch (step 1) so the fixture is clean
   again. The dedup scenarios especially depend on starting from the seed
   corpus, not the leftovers of a previous run.

## The scenarios

| # | Exercises |
|---|---|
| 01 | Card-type selection (Q/A vs. cloze-enumeration vs. ordered cloze), deck reuse |
| 02 | Dedup — material already covered; skip/merge rather than duplicate |
| 03 | Honor a requested type but apply discipline, and catch existing overlap |
| 04 | Two-way / reversed-card selection for genuine bidirectional vocab |
| 05 | Tag-vocabulary reuse — adopt existing tags, don't coin synonyms |
| 06 | Create a new deck only when justified, and flag it for the user |

01–03 are the original set, confirmed passing. 04–06 cover behaviours the first
three don't reach yet; verify and tighten the expected outcomes as you run them.

Outcomes are described in terms of **decks, card types, tags, and create/skip
decisions** — never note IDs, which are regenerated on every launch.
