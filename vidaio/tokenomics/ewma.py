"""EWMA score accumulation (design spec §03): new = decay * old + (1 - decay) * score.

EXCLUDED_SCORE (-1.0) is the documented exclusion sentinel: a cycle score of -1 marks
the miner excluded and the accumulator latches to -1 (the miner takes no weight while
excluded — see rank_curve). The next genuine score re-enters the accumulator from 0.0,
so exclusion never carries phantom history forward.
"""

from __future__ import annotations

EXCLUDED_SCORE = -1.0


def is_excluded(score: float) -> bool:
    """True when `score` is the exclusion sentinel."""
    return score == EXCLUDED_SCORE


def accumulate(old: float, score: float, decay: float) -> float:
    """Fold one cycle score into the EWMA accumulator.

    `decay` comes from TokenomicsConfig.ewma_decay (0.75). A sentinel `score`
    returns the sentinel; a genuine score after exclusion restarts from 0.0.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1)")
    if is_excluded(score):
        return EXCLUDED_SCORE
    if score < 0.0:
        raise ValueError("cycle scores must be >= 0 (or exactly -1 as the exclusion sentinel)")
    base = 0.0 if is_excluded(old) else old
    return decay * base + (1.0 - decay) * score
