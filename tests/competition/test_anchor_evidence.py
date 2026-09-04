from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

from vidaio.audit.canonical import sha256_hex
from vidaio.chain.adapter import InMemoryChain
from vidaio.competition.anchor_evidence import (
    CompetitionAnchorMismatch,
    CompetitionAnchorUnavailable,
    competition_anchor_payload,
    verify_competition_anchor_on_chain,
)


START = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
ANCHOR_AT = START - timedelta(minutes=30)
ROOT = "a" * 64
PAYLOAD = competition_anchor_payload(ROOT)
ANCHOR_BLOCK = 10


@dataclass(frozen=True)
class _Receipt:
    commitment_root: str = ROOT
    anchor_netuid: int = 85
    anchor_payload_hex: str = PAYLOAD.hex()
    anchor_payload_digest: str = sha256_hex(PAYLOAD)
    anchor_block: int = ANCHOR_BLOCK
    anchor_block_hash: str = ""
    anchor_finalized_block: int = ANCHOR_BLOCK


def _chain(payload: bytes = PAYLOAD) -> InMemoryChain:
    return InMemoryChain(
        _block=1_000,
        anchored=[payload],
        _anchor_blocks=[ANCHOR_BLOCK],
        block_time_anchor=(ANCHOR_BLOCK, ANCHOR_AT),
    )


def _receipt(chain: InMemoryChain) -> _Receipt:
    block_hash = chain.block_hash(ANCHOR_BLOCK)
    assert block_hash is not None
    return _Receipt(anchor_block_hash=block_hash)


def _verify(chain: object | None, receipt: _Receipt) -> None:
    verify_competition_anchor_on_chain(
        chain,
        receipt,
        expected_netuid=85,
        competition_start_time=START,
        epoch_close_block=500,
    )


def test_exact_finalized_archive_receipt_passes() -> None:
    chain = _chain()
    _verify(chain, _receipt(chain))


def test_missing_rpc_seam_is_unavailable_not_a_mismatch() -> None:
    with pytest.raises(CompetitionAnchorUnavailable, match="missing"):
        _verify(object(), _receipt(_chain()))


def test_readable_different_raw_payload_is_a_mismatch() -> None:
    chain = _chain(b"readable-but-wrong")
    with pytest.raises(CompetitionAnchorMismatch, match="exact committed payload"):
        _verify(chain, _receipt(chain))


def test_readable_different_block_hash_is_a_mismatch() -> None:
    chain = _chain()
    receipt = replace(_receipt(chain), anchor_block_hash="b" * 64)
    with pytest.raises(CompetitionAnchorMismatch, match="block hash differs"):
        _verify(chain, receipt)


def test_anchor_and_finality_must_precede_enrollment_and_epoch_close() -> None:
    chain = _chain()
    receipt = _receipt(chain)
    with pytest.raises(CompetitionAnchorMismatch, match="enrollment start"):
        verify_competition_anchor_on_chain(
            chain,
            receipt,
            expected_netuid=85,
            competition_start_time=ANCHOR_AT,
            epoch_close_block=500,
        )
    with pytest.raises(CompetitionAnchorMismatch, match="epoch close"):
        verify_competition_anchor_on_chain(
            chain,
            replace(receipt, anchor_block=500, anchor_finalized_block=500),
            expected_netuid=85,
            competition_start_time=START,
            epoch_close_block=500,
        )
