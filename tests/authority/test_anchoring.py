"""anchor_payload + anchor_epoch — the on-chain tamper-evidence root."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import os
import hashlib
import sqlite3
import asyncio
from pathlib import Path

import pytest

from vidaio.authority import EpochIndex, anchor_epoch, anchor_payload
from vidaio.authority.anchoring import (
    ANCHOR_DOMAIN, AnchorSubmissionHold, acknowledge_anchor_submission,
    clear_confirmed_anchor_writer,
)
from vidaio.authority.index import EpochIndexConflict
from vidaio.chain.adapter import CommitmentCapacity, InMemoryChain
from vidaio.chain.anchor_writer import (
    AnchorWriterPendingError, anchor_writer_lock, mark_anchor_writer_pending,
    read_anchor_writer_pending,
)
from vidaio.chain.bittensor_adapter import AuthorityAnchorExpiredAbsent
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


def _submission(encoded: bytes = b"signed authority epoch anchor") -> dict:
    return {
        "extrinsic_hash": "0x" + hashlib.blake2b(encoded, digest_size=32).hexdigest(),
        "extrinsic_hex": "0x" + encoded.hex(), "submit_block": 50,
        "submit_time": NOW.timestamp(), "nonce": 9, "mortal_era": 128,
    }


class SplitAuthorityChain(InMemoryChain):
    """Signed preparation and one-shot dispatch with independently observed finality."""

    def __init__(self, digest: str, confirmations: list | None = None):
        super().__init__()
        self.digest = digest
        self.submission = _submission()
        self.prepares = 0
        self.submits = 0
        self.confirmations = confirmations if confirmations is not None else [55]
        self.confirmed_hashes = []
        self.before_send = lambda: None
        self.send_error = None

    async def prepare_authority_anchor(self, payload):
        self.prepares += 1
        assert payload.decode().endswith(self.digest)
        return self.submission

    async def submit_authority_anchor(self, submission):
        self.before_send()
        self.submits += 1
        assert submission == self.submission
        if self.send_error:
            raise self.send_error

    async def confirm_authority_anchor(self, submission, **_kwargs):
        self.confirmed_hashes.append(submission["extrinsic_hash"])
        observation = self.confirmations.pop(0)
        if isinstance(observation, Exception):
            raise observation
        return observation

    def finalized_block(self):
        return 250

    def read_anchor_block(self, **_kwargs):
        raise AssertionError("durable confirmation must not depend on the mutable head slot")

    def read_anchor_at(self, **_kwargs):
        raise AssertionError("authority receipts must use the canonical finalized archive seam")

    def read_authority_anchor_at(self, *, block_number, **_kwargs):
        assert block_number == 55
        return self.digest


def _marker(fin, submission):
    return {
        "extrinsic_hash": submission["extrinsic_hash"], "epoch_id": fin.epoch_id,
        "netuid": 85, "log_digest": fin.log_digest,
        "submit_block": submission["submit_block"], "pid": 12345,
        "time": NOW.timestamp(), "submission": submission,
    }


async def test_signed_evidence_and_writer_fence_are_durable_before_send(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(106)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = SplitAuthorityChain(fin.log_digest)
    lock_path = tmp_path / "anchor.lock"
    phases = []

    def before_send():
        observer = EpochIndex.open(tmp_path / "authority.db")
        try:
            stored = observer.get_anchor_submission(fin.epoch_id)
            assert stored.submission == chain.submission
            assert stored.outcome is None
            assert read_anchor_writer_pending(lock_path)["extrinsic_hash"] == stored.extrinsic_hash
            assert phases[-1] == "anchor_submit"
        finally:
            observer.close()

    chain.before_send = before_send
    result = await anchor_epoch(
        fin, chain=chain, index=index, netuid=85, now=NOW,
        writer_lock_path=lock_path, on_phase=phases.append,
    )
    assert result.txid == chain.submission["extrinsic_hash"]
    assert result.block == 55
    assert index.get_anchor_submission(fin.epoch_id).outcome == "confirmed"
    assert read_anchor_writer_pending(lock_path) is None
    assert phases.index("anchor_submit") < phases.index("anchor_confirm")
    assert phases[-1] == "index"


async def test_restart_resumes_same_hash_and_never_prepares_or_sends_again(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(107)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    lock_path = tmp_path / "anchor.lock"
    first = SplitAuthorityChain(fin.log_digest, [TimeoutError("finality RPC timed out")])
    first.send_error = TimeoutError("submission acknowledgement lost")
    with pytest.raises(AnchorSubmissionHold, match="finality RPC timed out"):
        await anchor_epoch(fin, chain=first, index=index, netuid=85, now=NOW,
                           writer_lock_path=lock_path)
    assert first.submits == 1
    assert index.get_anchor_submission(fin.epoch_id).outcome is None
    assert read_anchor_writer_pending(lock_path) is not None
    with pytest.raises(AnchorWriterPendingError):
        async with anchor_writer_lock(lock_path, timeout_seconds=1):
            pytest.fail("an unrelated writer must HOLD across process restart")

    index.close()
    resumed_index = EpochIndex.open(tmp_path / "authority.db")
    resumed = SplitAuthorityChain(fin.log_digest)
    try:
        result = await anchor_epoch(
            fin, chain=resumed, index=resumed_index, netuid=85, now=NOW,
            writer_lock_path=lock_path,
        )
        assert result.txid == first.submission["extrinsic_hash"]
        assert resumed.prepares == resumed.submits == 0
        assert resumed.confirmed_hashes == [first.submission["extrinsic_hash"]]
        assert read_anchor_writer_pending(lock_path) is None
    finally:
        resumed_index.close()


async def test_cancelled_dispatch_retains_signed_evidence_and_writer_fence(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(113)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    sending = asyncio.Event()

    class CancelledDispatchChain(SplitAuthorityChain):
        async def submit_authority_anchor(self, submission):
            self.submits += 1
            sending.set()
            await asyncio.Event().wait()

    chain = CancelledDispatchChain(fin.log_digest)
    lock_path = tmp_path / "anchor.lock"
    phases = []
    task = asyncio.create_task(anchor_epoch(
        fin, chain=chain, index=index, netuid=85, now=NOW,
        writer_lock_path=lock_path, on_phase=phases.append,
    ))
    await sending.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert index.get_anchor_submission(fin.epoch_id).outcome is None
    assert read_anchor_writer_pending(lock_path)["extrinsic_hash"] == chain.submission["extrinsic_hash"]
    assert phases[-1] == "anchor_confirm"
    assert chain.submits == 1


async def test_fence_publication_failure_never_dispatches_unprotected_transaction(
    index: EpochIndex, tmp_path: Path, monkeypatch,
) -> None:
    fin = _finalized(114)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = SplitAuthorityChain(fin.log_digest)

    def unavailable_disk(*_args):
        raise OSError("pending marker fsync failed")

    monkeypatch.setattr("vidaio.authority.anchoring.mark_anchor_writer_pending", unavailable_disk)
    with pytest.raises(OSError, match="marker fsync failed"):
        await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW,
                           writer_lock_path=tmp_path / "anchor.lock")
    assert index.get_anchor_submission(fin.epoch_id).outcome is None
    assert chain.submits == 0


async def test_fresh_prepared_head_refuses_late_dispatch_despite_stale_tick_head(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(115)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = SplitAuthorityChain(fin.log_digest)
    chain.submission["submit_block"] = fin.close_block + 20
    assert chain.current_block() < fin.close_block + 20
    lock_path = tmp_path / "anchor.lock"
    with pytest.raises(AnchorSubmissionHold, match="deadline elapsed before dispatch"):
        await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW,
                           writer_lock_path=lock_path, confirmation_depth=20)
    assert chain.prepares == 1
    assert chain.submits == 0
    assert index.get_anchor_submission(fin.epoch_id) is None
    assert read_anchor_writer_pending(lock_path) is None


async def test_marker_only_restart_restores_evidence_without_dispatch(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(108)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = SplitAuthorityChain(fin.log_digest)
    lock_path = tmp_path / "anchor.lock"
    async with anchor_writer_lock(lock_path, timeout_seconds=1):
        mark_anchor_writer_pending(lock_path, _marker(fin, chain.submission))
    assert index.get_anchor_submission(fin.epoch_id) is None

    result = await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW,
                                writer_lock_path=lock_path)
    assert result.block == 55
    assert chain.prepares == chain.submits == 0
    assert index.get_anchor_submission(fin.epoch_id).outcome == "confirmed"


async def test_restart_clears_fence_after_indexed_confirmation(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(109)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    prepared = _submission()
    index.record_anchor_submission(fin.epoch_id, netuid=85, log_digest=fin.log_digest,
                                   submission=prepared)
    lock_path = tmp_path / "anchor.lock"
    async with anchor_writer_lock(lock_path, timeout_seconds=1):
        mark_anchor_writer_pending(lock_path, _marker(fin, prepared))
    index.set_anchor(fin.epoch_id, txid=prepared["extrinsic_hash"], block=55)

    assert await clear_confirmed_anchor_writer(index=index, netuid=85, writer_lock_path=lock_path)
    assert index.get_anchor_submission(fin.epoch_id).outcome == "confirmed"
    assert read_anchor_writer_pending(lock_path) is None


def _expired_evidence(submission):
    return {"extrinsic_hash": submission["extrinsic_hash"], "finalized_block": 199,
            "search_start": 50, "search_end": 198, "mortal_era": 128,
            "confirmation_depth": 20}


async def test_proven_expiry_records_outcome_and_releases_fence_without_new_send(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(110)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = SplitAuthorityChain(fin.log_digest)
    chain.confirmations = [AuthorityAnchorExpiredAbsent(_expired_evidence(chain.submission))]
    lock_path = tmp_path / "anchor.lock"
    expired = []
    with pytest.raises(AnchorSubmissionHold):
        await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW,
                           writer_lock_path=lock_path, on_expired=lambda: expired.append(True))
    assert expired == [True]
    assert chain.prepares == chain.submits == 1
    assert index.get_anchor_submission(fin.epoch_id).outcome == "expired_absent"
    assert index.get(fin.epoch_id).anchored is False
    assert read_anchor_writer_pending(lock_path) is None
    with pytest.raises(AnchorSubmissionHold, match="reused a terminal signed hash"):
        await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW,
                           writer_lock_path=lock_path)
    assert chain.submits == 1


async def test_operator_acknowledgement_clears_only_uncertainty_never_sends(
    index: EpochIndex, tmp_path: Path,
) -> None:
    fin = _finalized(111)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    chain = SplitAuthorityChain(fin.log_digest, [TimeoutError("era unreadable")])
    lock_path = tmp_path / "anchor.lock"
    with pytest.raises(AnchorSubmissionHold):
        await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW,
                           writer_lock_path=lock_path)
    await acknowledge_anchor_submission(index=index,
        extrinsic_hash=chain.submission["extrinsic_hash"], writer_lock_path=lock_path, now=NOW)
    assert read_anchor_writer_pending(lock_path) is None
    with pytest.raises(AnchorSubmissionHold, match="operator acknowledged"):
        await anchor_epoch(fin, chain=chain, index=index, netuid=85, now=NOW,
                           writer_lock_path=lock_path)
    with pytest.raises(ValueError, match="must name a pending hash"):
        await acknowledge_anchor_submission(index=index,
            extrinsic_hash=chain.submission["extrinsic_hash"], writer_lock_path=lock_path)
    assert chain.submits == 1
    assert index.get(fin.epoch_id).anchored is False


def test_submission_evidence_is_immutable_and_new_hash_needs_proven_expiry(
    index: EpochIndex,
) -> None:
    fin = _finalized(112)
    index.record_finalized(fin, finalized_at=NOW.isoformat())
    prepared = _submission()
    first = index.record_anchor_submission(fin.epoch_id, netuid=85, log_digest=fin.log_digest,
                                          submission=prepared)
    with pytest.raises(EpochIndexConflict, match="without proven expiry"):
        index.record_anchor_submission(fin.epoch_id, netuid=85, log_digest=fin.log_digest,
                                       submission=_submission(b"different signed anchor"))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        index._conn.execute("UPDATE authority_anchor_submissions SET submission_json='{}'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        index._conn.execute("DELETE FROM authority_anchor_submissions")
    incomplete = _expired_evidence(prepared)
    incomplete["search_end"] = 110
    with pytest.raises(EpochIndexConflict, match="full finalized mortal-era evidence"):
        index.finish_anchor_submission(first.extrinsic_hash, outcome="expired_absent",
                                        outcome_at=NOW.isoformat(), evidence=incomplete)
    index.finish_anchor_submission(first.extrinsic_hash, outcome="expired_absent",
                                    outcome_at=NOW.isoformat(), evidence=_expired_evidence(prepared))
    new = index.record_anchor_submission(fin.epoch_id, netuid=85, log_digest=fin.log_digest,
                                         submission=_submission(b"different signed anchor"))
    assert new.extrinsic_hash != first.extrinsic_hash
    assert index.anchor_submission_by_hash(first.extrinsic_hash).outcome == "expired_absent"
    assert index.pending_anchor_submissions() == [new]
