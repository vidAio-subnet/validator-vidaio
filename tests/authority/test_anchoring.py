"""anchor_payload + anchor_epoch — the on-chain tamper-evidence root."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path

import pytest

from vidaio.authority import EpochIndex, anchor_epoch, anchor_payload
from vidaio.authority.anchoring import ANCHOR_DOMAIN
from vidaio.chain.adapter import CommitmentCapacity, InMemoryChain
from vidaio.services.commitment_capacity import CommitmentCapacityError

from test_index import _finalized  # reuse the minimal valid FinalizedEpoch builder

NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
DIGEST = "a" * 64


def test_anchor_payload_is_domain_tagged_and_under_128() -> None:
    payload = anchor_payload(41822, 85, DIGEST)
    assert payload.decode("ascii") == f"{ANCHOR_DOMAIN}:85:41822:{DIGEST}"
    assert len(payload) <= 128
    # the anchored digest is readable straight out of the payload bytes.
    assert payload.decode("ascii").split(":")[-1] == DIGEST


@pytest.fixture
def index(tmp_path: Path) -> EpochIndex:
    return EpochIndex.open(tmp_path / "authority.db")


async def test_anchor_epoch_records_txid_and_block(index: EpochIndex) -> None:
    fin = _finalized(41822)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = InMemoryChain()
    chain.advance_blocks(9)  # block 10

    record = await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW)

    assert record.epoch_id == 41822
    assert record.digest == fin.log_digest
    assert record.txid is not None
    assert record.block == 10
    # the chain recorded the payload binding the log_digest.
    assert len(chain.anchored) == 1
    assert chain.anchored[0].decode("ascii").endswith(fin.log_digest)
    # persisted to the index.
    assert index.get(41822).anchor_txid == record.txid


async def test_anchor_epoch_records_exact_commitment_inclusion_not_later_head(
    index: EpochIndex,
) -> None:
    fin = _finalized(99)
    index.record_finalized(fin, finalized_at=NOW.isoformat())

    class FinalizedWriteChain:
        def __init__(self):
            self.read_calls = []

        async def anchor_commitment(self, payload):
            return "0xreceipt"

        def read_anchor_block(self, *, netuid, epoch_id, domain):
            self.read_calls.append((netuid, epoch_id, domain))
            return 41

        def current_block(self):
            return 47  # finalization wait observed a later head

    chain = FinalizedWriteChain()
    record = await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW)
    assert record.block == 41
    assert record.block != chain.current_block()
    assert chain.read_calls == [(85, 99, ANCHOR_DOMAIN)]


async def test_anchor_epoch_is_idempotent(index: EpochIndex) -> None:
    fin = _finalized(7)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = InMemoryChain()

    first = await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW)
    second = await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW)

    assert first.txid == second.txid
    assert len(chain.anchored) == 1  # NO second on-chain write


async def test_anchor_epoch_holds_writer_lane_through_inclusion_readback(
    index: EpochIndex, tmp_path: Path
) -> None:
    fin = _finalized(101)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    lock_path = tmp_path / "anchor.lock"

    class ProbingChain:
        async def anchor_commitment(self, _payload):
            return "0xreceipt"

        def read_anchor_block(self, **_kwargs):
            fd = os.open(lock_path, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
            return 55

    record = await anchor_epoch(
        fin,
        chain=ProbingChain(),
        index=index,
        netuid=85,
        now=NOW,
        writer_lock_path=lock_path,
        writer_lock_timeout_seconds=1,
    )
    assert record.block == 55


async def test_anchor_epoch_refuses_mismatched_finalized_archive_state(
    index: EpochIndex, tmp_path: Path
) -> None:
    fin = _finalized(102)
    index.record_finalized(fin, finalized_at=NOW.isoformat())

    class ReplacedChain:
        async def anchor_commitment(self, _payload):
            return "0xreceipt"

        def read_anchor_block(self, **_kwargs):
            return 55

        def finalized_block(self):
            return 55

        def read_anchor_at(self, **_kwargs):
            return "f" * 64

    with pytest.raises(RuntimeError, match="finalized archive state did not contain"):
        await anchor_epoch(
            fin,
            chain=ReplacedChain(),
            index=index,
            netuid=85,
            now=NOW,
            writer_lock_path=tmp_path / "anchor.lock",
            writer_lock_timeout_seconds=1,
        )
    assert index.get(fin.epoch_id).anchored is False


async def test_anchor_epoch_waits_for_independent_finality_without_resubmitting(
    index: EpochIndex,
) -> None:
    fin = _finalized(104)
    index.record_finalized(fin, finalized_at=NOW.isoformat())

    class LaggingFinalityChain(InMemoryChain):
        finality_reads = 0

        def finalized_block(self) -> int:
            self.finality_reads += 1
            return 0 if self.finality_reads == 1 else self.current_block()

    chain = LaggingFinalityChain()
    record = await anchor_epoch(
        fin,
        chain=chain,
        index=index,
        netuid=85,
        now=NOW,
        verification_timeout_seconds=1,
        verification_poll_seconds=0.001,
    )

    assert record.block == 1
    assert chain.finality_reads == 2
    assert len(chain.anchored) == 1


async def test_anchor_epoch_finality_timeout_never_resubmits(
    index: EpochIndex,
) -> None:
    fin = _finalized(105)
    index.record_finalized(fin, finalized_at=NOW.isoformat())

    class NeverFinalChain(InMemoryChain):
        def finalized_block(self) -> int:
            return 0

    chain = NeverFinalChain()
    with pytest.raises(RuntimeError, match="timed out.*not finalized"):
        await anchor_epoch(
            fin,
            chain=chain,
            index=index,
            netuid=85,
            now=NOW,
            verification_timeout_seconds=0.01,
            verification_poll_seconds=0.001,
        )

    assert len(chain.anchored) == 1
    assert index.get(fin.epoch_id).anchored is False


async def test_anchor_epoch_refuses_exhausted_runtime_capacity_before_write(
    index: EpochIndex,
) -> None:
    fin = _finalized(103)
    index.record_finalized(fin, finalized_at=NOW.isoformat())

    class ExhaustedChain(InMemoryChain):
        def commitment_capacity(self, netuid: int, hotkey: str) -> CommitmentCapacity:
            return CommitmentCapacity(
                netuid=netuid,
                hotkey=hotkey,
                block=44,
                current_epoch=8,
                usage_epoch=8,
                max_space=3_100,
                reported_used_space=3_050,
                used_space=3_050,
            )

    chain = ExhaustedChain()
    with pytest.raises(CommitmentCapacityError, match="epoch 103 anchor"):
        await anchor_epoch(
            fin,
            chain=chain,
            index=index,
            netuid=85,
            now=NOW,
            anchor_hotkey="authority-hotkey",
        )

    assert chain.anchored == []
    assert index.get(fin.epoch_id).anchored is False
