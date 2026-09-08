"""Short SQL exclusion for round completion and coherent epoch selection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from typing import Callable, Sequence

from vidaio.chain.adapter import ChainNeuron
from vidaio.tokenomics import MinerSnapshot


class RoundCommitDeferred(RuntimeError):
    """No mutation occurred: retry only after a new real best-head observation."""


class EpochCaptureUnavailable(RuntimeError):
    """The captured selection was rolled back and its close was not sealed."""


def block_height(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("block height must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class CapturedEpochInputs:
    packets: tuple[dict, ...]
    availability: tuple[dict, ...]
    content_rounds: tuple[dict, ...]
    round_commits: tuple[dict, ...]
    snapshots: tuple[MinerSnapshot, ...]


def capture_epoch_inputs(
    conn: sqlite3.Connection,
    *,
    prior_close_block: int | None,
    close_block: int,
    chain_neurons: Sequence[ChainNeuron],
    read_best_head: Callable[[], int],
    max_head: int | None = None,
) -> CapturedEpochInputs:
    """Detach all epoch inputs under the same exclusion used by round writers.

    Media/archive I/O happens after this function returns. A real observed head
    at least at close seals the selection; writers must subsequently observe a
    head strictly above the durable watermark. The optional anchor limit is
    exclusive. Failed reads and stale heads roll back the fence as one unit.
    """
    from vidaio.validator import miner_manager
    from vidaio.validator.evidence import AvailabilityFoldEvidence, ScorePacketEvidence

    block_height(close_block)
    if prior_close_block is not None:
        block_height(prior_close_block)
        if prior_close_block >= close_block:
            raise ValueError("prior close must precede close")
    if max_head is not None:
        block_height(max_head)
        if max_head <= close_block:
            raise ValueError("anchor limit must follow close")
    with miner_manager.transaction(conn):
        seal = conn.execute("SELECT sealed_close FROM round_seal WHERE singleton = 1").fetchone()
        if seal is None:
            raise EpochCaptureUnavailable("durable close watermark is missing")
        if close_block < seal[0]:
            raise EpochCaptureUnavailable("selected close regresses the durable watermark")
        if conn.execute("SELECT 1 FROM rounds WHERE committed_at IS NOT NULL AND commit_block IS NULL LIMIT 1").fetchone():
            raise EpochCaptureUnavailable("completed round is missing its commit height")
        scores = ScorePacketEvidence(conn)
        bounds = dict(through_block=close_block, after_block=prior_close_block)
        captured = CapturedEpochInputs(
            packets=tuple(dict(row) for row in scores.packets(**bounds)),
            availability=tuple(dict(row) for row in AvailabilityFoldEvidence(conn).observations(**bounds)),
            content_rounds=tuple(dict(row) for row in scores.content_rounds(**bounds)),
            round_commits=tuple(dict(row) for row in scores.round_commits(**bounds)),
            snapshots=tuple(miner_manager.snapshot_at(
                conn, chain_neurons, close_block, datetime.now(timezone.utc))),
        )
        observed = block_height(read_best_head())
        if observed < close_block:
            raise EpochCaptureUnavailable("fresh best head is below the selected close")
        if max_head is not None and observed >= max_head:
            raise EpochCaptureUnavailable("fresh best head is outside the open anchor window")
        changed = conn.execute("UPDATE round_seal SET sealed_close = MAX(sealed_close, ?) WHERE singleton = 1", (close_block,))
        if changed.rowcount != 1:
            raise EpochCaptureUnavailable("durable close watermark is missing")
    return captured
