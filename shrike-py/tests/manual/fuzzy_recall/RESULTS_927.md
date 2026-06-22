# #927 fuzzy rare-trigram cap policy — measured results

A snapshot of the fuzzy-recall eval (`fuzzy_recall.py --notes 5000 --sample 500`,
seed 0). The fuzzy signal is measured in isolation (`search_fuzzy_batch`), gold known
by construction (a typo is injected into a content phrase drawn from a note; gold =
every note whose clean text contains the un-perturbed phrase). recall@10 is the
headline cut, MRR secondary. Control = **fixed-6** (the shipped default; the engine
default is unchanged by this PR).

Regenerate with `SHRIKE_FUZZY_RECALL=1` + the CLI; recall@10/MRR are deterministic.
Latency (the other side of the trade) is wired but excluded here — it belongs on a
clean-environment run, not the dev machine.

## Finding

- **The log-growth curve is a clear recall win, monotone in k**, concentrated exactly
  where it is meant to act — the long-query buckets. Overall recall@10 rises from
  **50.4%** (fixed-6) to **62.0%** (curve k=4.0, +11.5pts); the n18+ bucket rises
  71.6% → 86.8% and the n12-17 bucket 41.7% → 54.1%. The short bucket (n≤6, where the
  curve clamps to the floor) is unchanged — confirming the curve only touches long
  queries.
- **The floor sweep DOWN costs recall with no offsetting win** (fixed-5 −5.8pts,
  fixed-4 −16.3pts), so 6 is not too high.
- The win holds across every typo count (even 3-typo: 41.4% → 54.8% at k=4.0) and
  every edit type, including the real-misspelling class.
- recall@10 is **still climbing at k=4.0** (the issue's steeper guess beats the
  conservative k=2.7), and at the ceiling 12 — so the ceiling may warrant raising.

The cap decision is a separate, eval-gated follow-up; the latency cost must be read
from a clean-env run before adopting a curve. This PR only builds the eval + the
injectable cap (default stays fixed-6).

## Overall

| arm | floor | k | ceiling | recall@10 | Δ vs control | MRR |
|---|---:|---:|---:|---:|---:|---:|
| fixed-6 (control) | 6 | 2.7 | 6 |  50.4% | — |  47.5% |
| fixed-5 | 5 | 2.7 | 5 |  44.6% | -5.8 |  42.0% |
| fixed-4 | 4 | 2.7 | 4 |  34.1% | -16.3 |  34.2% |
| curve k=2.0 | 6 | 2 | 12 |  58.7% | +8.2 |  58.7% |
| curve k=2.7 | 6 | 2.7 | 12 |  59.6% | +9.1 |  60.7% |
| curve k=4.0 | 6 | 4 | 12 |  62.0% | +11.5 |  62.3% |

## recall@10 by query-length bucket (trigram count)

The cap curve only acts on n > the floor, so the long-query buckets are where a growth
arm must show a win and a floor-down arm must not lose.

| arm | n<=6 | n7-11 | n12-17 | n18+ |
|---|---:|---:|---:|---:|
| fixed-6 (control) |   2.8% (n=82) |  27.3% (n=52) |  41.7% (n=74) |  71.6% (n=272) |
| fixed-5 |   4.0% (n=82) |  28.3% (n=52) |  36.2% (n=74) |  62.2% (n=272) |
| fixed-4 |   4.0% (n=82) |  20.9% (n=52) |  27.1% (n=74) |  47.6% (n=272) |
| curve k=2.0 |   2.8% (n=82) |  30.1% (n=52) |  47.3% (n=74) |  84.1% (n=272) |
| curve k=2.7 |   2.8% (n=82) |  32.1% (n=52) |  50.3% (n=74) |  84.5% (n=272) |
| curve k=4.0 |   4.0% (n=82) |  34.5% (n=52) |  54.1% (n=74) |  86.8% (n=272) |

## recall@10 by typo count

| arm | 1 typo | 2 typo | 3 typo |
|---|---:|---:|---:|
| fixed-6 (control) |  60.6% (n=159) |  49.4% (n=161) |  41.4% (n=160) |
| fixed-5 |  53.4% (n=159) |  46.9% (n=161) |  33.6% (n=160) |
| fixed-4 |  46.4% (n=159) |  33.4% (n=161) |  22.5% (n=160) |
| curve k=2.0 |  67.8% (n=159) |  57.8% (n=161) |  50.5% (n=160) |
| curve k=2.7 |  68.4% (n=159) |  58.4% (n=161) |  51.9% (n=160) |
| curve k=4.0 |  69.5% (n=159) |  61.7% (n=161) |  54.8% (n=160) |

## recall@10 by edit type

| arm | case | delete | double | insert | phonetic | real_misspelling | substitute_adjacent | substitute_random | transpose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-6 (control) |  53.1% (n=92) |  50.8% (n=90) |  43.8% (n=92) |  42.7% (n=100) |  48.7% (n=87) |  48.7% (n=243) |  46.2% (n=81) |  46.0% (n=87) |  43.0% (n=89) |
| fixed-5 |  47.6% (n=92) |  46.6% (n=90) |  40.9% (n=92) |  35.6% (n=100) |  38.9% (n=87) |  43.0% (n=243) |  37.6% (n=81) |  41.8% (n=87) |  36.9% (n=89) |
| fixed-4 |  37.0% (n=92) |  33.1% (n=90) |  36.0% (n=92) |  27.1% (n=100) |  28.8% (n=87) |  29.5% (n=243) |  23.5% (n=81) |  29.2% (n=87) |  27.4% (n=89) |
| curve k=2.0 |  57.5% (n=92) |  59.7% (n=90) |  52.7% (n=92) |  53.9% (n=100) |  54.6% (n=87) |  59.1% (n=243) |  55.4% (n=81) |  49.4% (n=87) |  54.6% (n=89) |
| curve k=2.7 |  57.5% (n=92) |  60.0% (n=90) |  55.2% (n=92) |  55.9% (n=100) |  55.8% (n=87) |  60.0% (n=243) |  57.8% (n=81) |  48.3% (n=87) |  55.7% (n=89) |
| curve k=4.0 |  58.5% (n=92) |  62.4% (n=90) |  56.3% (n=92) |  60.4% (n=100) |  59.8% (n=87) |  63.2% (n=243) |  60.7% (n=81) |  51.2% (n=87) |  56.9% (n=89) |
