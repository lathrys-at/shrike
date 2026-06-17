"""Host-side vector-index policy: the state enum + the activation-gate floor.

The index itself lives in the kernel since the harness rebase (#332/#353):
``shrike-kernel``'s ``IndexOrchestrator`` (``shrike-core/shrike-kernel/src/
index_orchestrator.rs``) owns the per-modality USearch engine, drift detection,
incremental reconcile, per-note fingerprints, persistence, the debounced saver,
and activation calibration. The ``VectorIndex``/``IndexSaver`` facade that
mirrored it for standalone/test contexts retired with #355 — the unit suites
drive a real ``AsyncKernel`` now (``tests/unit/conftest.py``).

What stays here is the host's slice:

- :class:`IndexState` — the state machine's names, shared by the wire status
  shapes (``schemas.IndexStatus``), the search action's gating, and the
  derived store (which deliberately reuses the same vocabulary).
- :func:`activation_floor` — the pure #201b gate math the search action
  applies to the kernel-calibrated stats.
- ``CALIB_MIN`` — the calibration minimum, mirroring the kernel constant
  (``shrike_kernel::index_orchestrator::CALIB_MIN``) for the CLIP
  integration suite's assertions.
"""

from __future__ import annotations

import enum

__all__ = ["CALIB_MIN", "IndexState", "activation_floor"]

# Minimum non-self best-matches a modality needs for activation stats — the
# host-side mirror of the kernel's CALIB_MIN (kept in sync by hand; it only
# feeds test assertions, never the gate itself, which runs on kernel stats).
CALIB_MIN = 30


class IndexState(enum.Enum):
    READY = "ready"
    BUILDING = "building"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


def activation_floor(stats: dict[str, float] | None, margin: float) -> float | None:
    """Similarity a modality's best match must exceed to activate: ``mean + margin·std``.

    ``stats`` is one modality's calibrated ``{n, mean, std}`` (or ``None`` when uncalibrated — then
    there is no floor and the gate is disabled, i.e. the modality always contributes). Pure: no
    index or embedding state, so it is unit-testable in isolation and shared by the gate in
    the search action (#201b).
    """
    if not stats:
        return None
    return stats["mean"] + margin * stats["std"]
