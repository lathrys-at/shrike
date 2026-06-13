"""Search-quality adversarial suite support (#559).

The metric engine (``metrics``), manifest loader (``manifest``), and the build +
run harness (``harness``) shared by the pytest suite
(``tests/integration/test_search_quality.py``) and the ``scripts/eval_search_quality``
runner — one source of truth for what a search-quality run computes.

Off-CI: every consumer is gated by ``pytest.mark.search_quality`` +
``SHRIKE_SEARCH_QUALITY=1`` and excluded from the Bazel ``:integration`` glob;
see the suite module docstring.
"""
