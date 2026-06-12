"""The host-side index policy that survived the #355 facade retirement.

The index itself (drift, reconcile, persistence, the debounced saver,
calibration) is kernel-owned and pinned by the shrike-kernel/shrike-index Rust
suites plus tests/native; what remains host-side is the pure #201b gate math
the search action applies to kernel-calibrated stats.
"""

from __future__ import annotations

import pytest

from shrike.index import IndexState, activation_floor


class TestActivationFloor:
    def test_uncalibrated_disables_the_gate(self) -> None:
        assert activation_floor(None, 2.0) is None
        assert activation_floor({}, 2.0) is None

    def test_floor_is_mean_plus_margin_std(self) -> None:
        assert activation_floor({"n": 40.0, "mean": 0.3, "std": 0.1}, 2.0) == pytest.approx(0.5)
        assert activation_floor({"n": 40.0, "mean": 0.2, "std": 0.05}, 1.0) == pytest.approx(0.25)

    def test_zero_margin_floors_at_the_mean(self) -> None:
        assert activation_floor({"n": 10.0, "mean": 0.42, "std": 0.2}, 0.0) == pytest.approx(0.42)


class TestIndexState:
    def test_values_match_the_wire_vocabulary(self) -> None:
        # The kernel's index_status_json state strings round-trip through the
        # enum (KernelIndexView.state does IndexState(name)).
        assert {s.value for s in IndexState} == {"ready", "building", "unavailable", "error"}
