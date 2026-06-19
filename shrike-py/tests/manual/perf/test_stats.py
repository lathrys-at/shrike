"""Pure distribution math for the perf harness — no native, no I/O, so it
runs on every lane."""

from __future__ import annotations

import pytest

from tests.manual.perf.stats import Distribution, _percentile, summarize


def test_summarize_basic_percentiles():
    d = summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert d.n == 5
    assert d.min_ms == 10.0
    assert d.max_ms == 50.0
    assert d.p50_ms == 30.0
    assert d.mean_ms == 30.0


def test_warmup_discards_the_cold_samples():
    d = summarize([1000.0, 10.0, 20.0, 30.0], warmup=1)
    assert d.n == 3
    assert d.max_ms == 30.0  # the cold 1000.0 is gone


def test_warmup_consuming_everything_raises():
    with pytest.raises(ValueError, match="no samples"):
        summarize([1.0, 2.0], warmup=2)


def test_negative_warmup_raises():
    with pytest.raises(ValueError, match="warmup"):
        summarize([1.0], warmup=-1)


def test_percentile_linear_interpolation():
    assert _percentile([0.0, 100.0], 0.5) == 50.0
    assert _percentile([0.0, 10.0, 20.0, 30.0], 0.9) == pytest.approx(27.0)


def test_single_sample_collapses_all_percentiles():
    d = summarize([42.0])
    assert d.p50_ms == d.p90_ms == d.p99_ms == d.max_ms == d.min_ms == 42.0


def test_distribution_round_trips_through_dict():
    d = summarize([1.0, 2.0, 3.0, 4.0])
    assert Distribution.from_dict(d.as_dict()) == d
