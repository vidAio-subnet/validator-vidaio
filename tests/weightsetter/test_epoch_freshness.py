"""Regression coverage for latest-epoch convergence freshness.

An anchored epoch is valid historical evidence forever, so anchor verification
alone cannot establish that ``GET /epoch/latest`` is current.  These tests model
the archive boundary independently of the authority API and prove that stale or
regressed vectors never reach ``set_weights`` while explicit historical reads and
same-epoch retries remain available.
"""

from __future__ import annotations

import pytest

from vidaio.chain import ChainNeuron, EpochBoundary, InMemoryChain
from vidaio.core.db import connect
from vidaio.weightsetter import WeightSetter, intents
from vidaio.weightsetter.shared_snapshot import (
    SharedSnapshotProvider,
    SnapshotUnavailable,
)

from weightsetter_support import (
    NETUID,
    NOW,
    AuthorityHarness,
    FakeScoringAuthorityClient,
)


class ArchiveBoundaryReader:
    """Historical anchor reader plus a controllable finalized archive boundary."""

    def __init__(self, pointers, *, latest_sequence) -> None:
        self._pointers = {pointer.epoch_id: pointer for pointer in pointers}
        self._latest_sequence = list(latest_sequence)
        self._latest_calls = 0
        self.close_blocks = {
            pointer.epoch_id: pointer.close_block for pointer in pointers
        }

    def read_epoch_anchor(self, *, netuid: int, epoch_id: int) -> str | None:
        assert netuid == NETUID
        pointer = self._pointers.get(epoch_id)
        return pointer.snapshot_digest if pointer is not None else None

    def read_epoch_anchor_at(
        self, *, netuid: int, epoch_id: int, block_number: int
    ) -> str | None:
        assert netuid == NETUID
        pointer = self._pointers.get(epoch_id)
        if pointer is None or pointer.anchor.block != block_number:
            return None
        return pointer.snapshot_digest

    def latest_closed_epoch(self, *, netuid: int) -> EpochBoundary | None:
        assert netuid == NETUID
        if not self._latest_sequence:
            return None
        index = min(self._latest_calls, len(self._latest_sequence) - 1)
        self._latest_calls += 1
        return self._latest_sequence[index]

    def epoch_close_block(self, *, netuid: int, epoch_id: int) -> int | None:
        assert netuid == NETUID
        return self.close_blocks.get(epoch_id)


async def _two_empty_epochs(tmp_path):
    harness = AuthorityHarness(tmp_path, burn_uid=7)
    first = await harness.finalize(
        epoch_id=1, close_block=100, miners=[], items=None
    )
    second = await harness.finalize(
        epoch_id=2, close_block=200, miners=[], items=None
    )
    return harness, first, second, harness.pointer(1), harness.pointer(2)


def _provider(harness, *, latest, pointers, reader) -> SharedSnapshotProvider:
    return SharedSnapshotProvider(
        client=FakeScoringAuthorityClient(
            latest=latest,
            by_epoch={pointer.epoch_id: pointer for pointer in pointers},
        ),
        store=harness.store,
        netuid=NETUID,
        anchor_reader=reader,
    )


def _live_setter(tmp_path, name, *, provider):
    conn = connect(tmp_path / f"{name}.db")
    chain = InMemoryChain(
        _neurons=[ChainNeuron(7, "owner-hk", "owner-ck", "0.0.0.0", 0.0, 0.0)]
    )
    chain.get_burn_uid = lambda: 7  # type: ignore[attr-defined]
    raw = {
        "core": {"metrics_port": 0},
        "chain": {"mode": "bittensor", "netuid": NETUID},
        "weightsetter": {
            "metrics_port": 0,
            "chain_timeout_seconds": 0.5,
            "chain_retry_attempts": 1,
            "chain_retry_base_delay_seconds": 0.01,
            "publication_enabled": False,
        },
    }
    return WeightSetter(raw, chain=chain, snapshots=provider, conn=conn), chain, conn


async def test_latest_rejects_stale_anchored_pointer_but_history_remains_readable(
    tmp_path,
) -> None:
    harness, _first, _second, pointer1, pointer2 = await _two_empty_epochs(tmp_path)
    try:
        reader = ArchiveBoundaryReader(
            (pointer1, pointer2),
            latest_sequence=(
                EpochBoundary(epoch_id=2, close_block=pointer2.close_block),
            ),
        )
        provider = _provider(
            harness,
            latest=pointer1,
            pointers=(pointer1, pointer2),
            reader=reader,
        )

        with pytest.raises(SnapshotUnavailable, match="not the archive chain's latest"):
            provider.miner_snapshots()

        # Explicit by-id resolution is an audit/recovery API, not a convergence
        # target. The same historical pointer stays verifiable and readable.
        assert provider.resolve_epoch(1).epoch_id == 1
    finally:
        harness.close()


async def test_latest_requires_independent_exact_close_block(tmp_path) -> None:
    harness, _first, _second, pointer1, pointer2 = await _two_empty_epochs(tmp_path)
    try:
        reader = ArchiveBoundaryReader(
            (pointer1, pointer2),
            latest_sequence=(
                EpochBoundary(epoch_id=2, close_block=pointer2.close_block),
            ),
        )
        reader.close_blocks[2] = pointer2.close_block + 1
        provider = _provider(
            harness,
            latest=pointer2,
            pointers=(pointer1, pointer2),
            reader=reader,
        )

        with pytest.raises(SnapshotUnavailable, match="boundary views disagree"):
            provider.miner_snapshots()
    finally:
        harness.close()


async def test_epoch_closing_during_mirror_holds_before_intent_or_write(tmp_path) -> None:
    harness, _first, _second, pointer1, pointer2 = await _two_empty_epochs(tmp_path)
    try:
        # First boundary read (before mirroring) says epoch 1. The final service
        # fence re-reads after mirror/parse and observes epoch 2.
        reader = ArchiveBoundaryReader(
            (pointer1, pointer2),
            latest_sequence=(
                EpochBoundary(epoch_id=1, close_block=pointer1.close_block),
                EpochBoundary(epoch_id=2, close_block=pointer2.close_block),
            ),
        )
        provider = _provider(
            harness,
            latest=pointer1,
            pointers=(pointer1, pointer2),
            reader=reader,
        )
        setter, chain, conn = _live_setter(
            tmp_path, "close-during-mirror", provider=provider
        )

        assert await setter.attempt_once() is False
        assert chain.weight_calls == []
        assert intents.intents(conn) == []
    finally:
        harness.close()


async def test_bittensor_mode_holds_when_boundary_capability_is_missing(tmp_path) -> None:
    harness, _first, _second, _pointer1, pointer2 = await _two_empty_epochs(tmp_path)
    try:
        # The ordinary InMemory anchor reader authenticates historical bytes but
        # intentionally has no latest_closed_epoch/epoch_close_block capability.
        provider = SharedSnapshotProvider(
            client=FakeScoringAuthorityClient(latest=pointer2),
            store=harness.store,
            netuid=NETUID,
            anchor_reader=harness.anchor_reader(),
        )
        setter, chain, conn = _live_setter(
            tmp_path, "missing-boundary", provider=provider
        )

        assert await setter.attempt_once() is False
        assert chain.weight_calls == []
        assert intents.intents(conn) == []
        assert (
            setter.metric_chain_state_skips.labels(
                reason="snapshot_epoch_boundary_unverified"
            )._value.get()
            == 1
        )
    finally:
        harness.close()


async def test_same_current_epoch_can_be_resubmitted(tmp_path) -> None:
    harness, _first, _second, pointer1, pointer2 = await _two_empty_epochs(tmp_path)
    try:
        reader = ArchiveBoundaryReader(
            (pointer1, pointer2),
            latest_sequence=(
                EpochBoundary(epoch_id=2, close_block=pointer2.close_block),
            ),
        )
        provider = _provider(
            harness,
            latest=pointer2,
            pointers=(pointer1, pointer2),
            reader=reader,
        )
        setter, chain, conn = _live_setter(
            tmp_path, "same-epoch-resubmit", provider=provider
        )

        assert await setter.attempt_once() is True
        assert intents.latest_snapshot_epoch_id(conn) == 2
        chain.advance_blocks(chain.tempo + 1)
        assert await setter.attempt_once() is True
        assert len(chain.weight_calls) == 2
        assert [row["snapshot_epoch_id"] for row in intents.intents(conn)] == [2, 2]
    finally:
        harness.close()


async def test_durable_epoch_floor_blocks_joint_api_rpc_regression(tmp_path) -> None:
    harness, _first, _second, pointer1, pointer2 = await _two_empty_epochs(tmp_path)
    try:
        # Model a transient view in which both the authority API and archive RPC
        # claim epoch 1 is current. The durable DB proves this validator already
        # admitted epoch 2 and must never roll back.
        reader = ArchiveBoundaryReader(
            (pointer1, pointer2),
            latest_sequence=(
                EpochBoundary(epoch_id=1, close_block=pointer1.close_block),
            ),
        )
        provider = _provider(
            harness,
            latest=pointer1,
            pointers=(pointer1, pointer2),
            reader=reader,
        )
        setter, chain, conn = _live_setter(
            tmp_path, "durable-regression", provider=provider
        )
        prior = intents.record_intent(
            conn,
            created_at=NOW.isoformat(),
            attempt_block=1,
            version_key=setter.config.version_key,
            weights={7: 65535.0},
            packet_digests=(),
            snapshot_digest=pointer2.snapshot_digest,
            snapshot_epoch_id=2,
        )
        intents.mark_published(
            conn, prior, at=NOW.isoformat(), resolution="fixture_high_watermark"
        )

        assert await setter.attempt_once() is False
        assert chain.weight_calls == []
        assert len(intents.intents(conn)) == 1
        assert (
            setter.metric_chain_state_skips.labels(
                reason="snapshot_epoch_regression"
            )._value.get()
            == 1
        )
    finally:
        harness.close()
