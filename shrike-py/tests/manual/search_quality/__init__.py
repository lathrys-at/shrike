"""Shared search-quality support: the metric engine + manifest loader.

One source of truth for what a search-quality run computes, imported by two
consumers:

- ``tests/native/test_search_quality.py`` — the DETERMINISTIC,
  CI-stable classes: a stub embedder gives exactly-controlled vectors, so the
  RRF fusion arithmetic, the exact-override tier, the activation gate, and the
  graceful-degradation paths are pinned on every PR. No model, no network.
- ``tests/integration/test_search_quality.py`` (manual/off-CI) — the
  real-model recall/precision suite over a Wikimedia Commons corpus.

The metric engine (:mod:`metrics`) is a PURE function of ``(returned, gold)``,
so it doubles as a threshold-tuning harness (re-run over
``threshold``/``ACTIVATION_MARGIN``/weights/``RRF_K``). The manifest loader
(:mod:`manifest`) parses the reconciled cards/queries schema both consumers share.
"""
