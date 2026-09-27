"""Locate the committed competition result behind a reward window.

A reward window folded before schema v18 does not carry its payout policy. The first
v18 fold re-resolves it (:func:`vidaio.tokenomics.breakthrough.upgrade_legacy_window`)
from the ``competition_result`` committed in the epoch log that applied it. That log
is found by walking the log hash chain backwards from the predecessor log: every step
is read under the digest its successor committed, so the authority and every auditor
reach the same bytes or none at all.
"""

from __future__ import annotations

from collections.abc import Callable

from vidaio.epoch.log import EpochLog
from vidaio.tokenomics.state import CompetitionResult, RewardWindowState

#: Upper bound on the walk. A 168-hour window spans at most 140 epochs of 72 minutes;
#: the bound leaves room for gap epochs and never reads the whole history.
WINDOW_SOURCE_MAX_STEPS = 160

#: Reads one finalized epoch log by (epoch id, expected log digest). It must verify
#: the digest and raise when the bytes are unreadable.
EpochLogReader = Callable[[int, str], EpochLog]


class WindowSourceUnavailable(LookupError):
    """The applying epoch log of a window could not be reached through the chain."""


def find_window_source_result(
    read_log: EpochLogReader,
    start: EpochLog,
    window: RewardWindowState,
    *,
    max_steps: int = WINDOW_SOURCE_MAX_STEPS,
) -> CompetitionResult:
    """The ``competition_result`` of the epoch log that applied ``window``.

    ``start`` is the log that carries ``window`` (the predecessor of the epoch being
    composed or audited). The walk follows ``prior_log_digest`` while the logs still
    carry the same window and stops at the first log whose committed result has the
    window's competition id and cycle. It raises :class:`WindowSourceUnavailable`
    when the chain leaves the window, ends, or exceeds ``max_steps`` without it.
    """
    log = start
    for _ in range(max_steps + 1):
        carried = log.reward_window_state
        if (
            carried.source_cycle != window.source_cycle
            or carried.source_competition_id != window.source_competition_id
        ):
            break
        result = log.competition_result
        if (
            result is not None
            and result.cycle == window.source_cycle
            and result.competition_id == window.source_competition_id
        ):
            return result
        prior_id = log.prior_epoch_id
        if prior_id is None or log.prior_log_digest is None:
            break
        log = read_log(prior_id, log.prior_log_digest)
    raise WindowSourceUnavailable(
        f"no epoch log applying competition {window.source_competition_id!r} "
        f"cycle {window.source_cycle} within {max_steps} chained predecessors"
    )
