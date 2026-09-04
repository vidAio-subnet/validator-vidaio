"""Finiteness guard — scoring fails CLOSED on NaN/inf, never maps them to a score.

NaN compares False against every threshold, so before this guard a non-finite metric
sailed through gate comparisons and formula case-checks (a NaN VMAF composed to a
formula final; a NaN PieAPP mapped to perfect quality 1.0). Formulas and aggregation
reject non-finite inputs with :class:`ValueError` at the boundary; gates translate
them into ``METRIC_NON_FINITE`` violations instead (see :mod:`vidaio.scoring.gates`).
"""

from __future__ import annotations

import math


def require_finite(name: str, value: float) -> float:
    """Return ``value`` unchanged, or raise ``ValueError`` when it is NaN or +/-inf."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value
