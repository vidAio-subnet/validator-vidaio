"""Bounded, no-resubmit verification of one finalized commitment receipt.

The Bittensor write socket and the independent archive/read socket can observe
GRANDPA finality a few blocks apart.  A successful ``set_commitment`` must not be
submitted again merely because the read socket is one block behind: doing so
burns another Commitments-pallet capacity charge and moves the mutable slot.

Callers hold their existing cross-process anchor-writer lane around this helper.
It only reads.  Transient head/finality/archive unavailability is polled within a
bounded window; an exact finalized archive mismatch is definitive and fails
immediately.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from vidaio.chain.adapter import ChainAdapter, ChainCommitmentRecord


SHA256_HEX = re.compile(r"[0-9a-f]{64}")

# Testnet GRANDPA/archive visibility was observed 2--3 blocks behind best head.
# Ninety seconds leaves ample room for that independent read path while staying
# inside the production challenge request timeout (>=240 seconds) and the epoch
# anchor's close+K window (K>=20 blocks).
DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS = 90.0
DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS = 1.0


class AnchorReceiptVerificationError(RuntimeError):
    """A submitted commitment cannot be given a finalized archive receipt."""


class AnchorReceiptMismatch(AnchorReceiptVerificationError):
    """Exact finalized archive state proves a different commitment."""


class AnchorReceiptTimeout(AnchorReceiptVerificationError):
    """Independent receipt visibility did not converge inside the bounded wait."""


@dataclass(frozen=True)
class FinalizedAnchorReceipt:
    """The exact finalized inclusion point independently verified from archive state."""

    block: int
    block_hash: str | None


@dataclass(frozen=True)
class FinalizedCommitmentReceipt:
    """Finalized proof for one exact raw commitment payload."""

    block: int
    block_hash: str
    finalized_block: int


async def wait_for_finalized_anchor_receipt(
    chain: ChainAdapter,
    *,
    netuid: int,
    anchor_id: int,
    domain: str,
    expected_digest: str,
    operation: str,
    timeout_seconds: float = DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS,
    require_head_digest: bool = False,
    require_block_hash: bool = False,
) -> FinalizedAnchorReceipt:
    """Wait for one already-submitted anchor to become independently provable.

    This function never calls ``anchor_commitment``.  A stale current-slot read,
    a finalized head below the inclusion block, and raised read/RPC errors are
    availability observations and are retried until the deadline.  Once the
    inclusion block is finalized, a successful exact-block archive query carrying
    any digest other than ``expected_digest`` is a hard mismatch and is never
    retried.

    ``require_head_digest`` is used by challenge receipts, whose response binds
    both the current read-back and exact inclusion block.  Epoch anchors can retain
    compatibility with small report/test adapters whose block reader already
    domain-matches the payload but which do not expose a separate head-digest seam.
    ``require_block_hash`` similarly reflects the challenge receipt schema; epoch
    pointers persist the inclusion height rather than a block hash.
    """
    if timeout_seconds <= 0:
        raise ValueError("anchor receipt timeout must be positive")
    if poll_seconds <= 0:
        raise ValueError("anchor receipt poll interval must be positive")
    if not SHA256_HEX.fullmatch(expected_digest):
        raise ValueError("expected anchor digest must be lowercase sha256 hex")

    head_reader = getattr(chain, "read_anchor", None)
    block_reader = getattr(chain, "read_anchor_block", None)
    finalized_reader = getattr(chain, "finalized_block", None)
    archive_reader = getattr(chain, "read_anchor_at", None)
    block_hash_reader = getattr(chain, "block_hash", None)

    required = {
        "read_anchor_block": block_reader,
        "finalized_block": finalized_reader,
        "read_anchor_at": archive_reader,
    }
    if require_head_digest:
        required["read_anchor"] = head_reader
    if require_block_hash:
        required["block_hash"] = block_hash_reader
    missing = [name for name, reader in required.items() if not callable(reader)]
    if missing:
        raise AnchorReceiptVerificationError(
            f"{operation}: chain adapter cannot prove a finalized archive receipt; "
            f"missing {', '.join(sorted(missing))}"
        )

    assert callable(block_reader)
    assert callable(finalized_reader)
    assert callable(archive_reader)
    deadline = time.monotonic() + timeout_seconds
    last_observation = "receipt was not yet visible"
    inclusion_block: int | None = None

    async def _retry(reason: str) -> None:
        nonlocal last_observation
        last_observation = reason
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AnchorReceiptTimeout(
                f"{operation}: timed out after {timeout_seconds:g}s waiting for "
                f"independent finalized/archive receipt visibility; last observation: "
                f"{last_observation}"
            )
        await asyncio.sleep(min(poll_seconds, remaining))

    while True:
        if inclusion_block is None:
            try:
                observed = (
                    await asyncio.to_thread(
                        head_reader,
                        netuid=netuid,
                        epoch_id=anchor_id,
                        domain=domain,
                    )
                    if callable(head_reader)
                    else expected_digest
                )
                raw_block = await asyncio.to_thread(
                    block_reader,
                    netuid=netuid,
                    epoch_id=anchor_id,
                    domain=domain,
                )
            except Exception as exc:  # transient independent read/RPC visibility
                await _retry(
                    "current anchor read-back unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if observed != expected_digest or raw_block is None:
                await _retry(
                    "current commitment slot has not exposed the submitted anchor "
                    f"(digest_match={observed == expected_digest}, block={raw_block!r})"
                )
                continue
            if isinstance(raw_block, bool):
                raise AnchorReceiptVerificationError(
                    f"{operation}: anchor inclusion block is boolean, not an integer"
                )
            inclusion_block = int(raw_block)
            if inclusion_block < 0:
                raise AnchorReceiptVerificationError(
                    f"{operation}: anchor inclusion block is negative "
                    f"({inclusion_block})"
                )

        try:
            raw_finalized = await asyncio.to_thread(finalized_reader)
        except Exception as exc:
            await _retry(
                f"finalized-head read unavailable: {type(exc).__name__}: {exc}"
            )
            continue
        if isinstance(raw_finalized, bool):
            raise AnchorReceiptVerificationError(
                f"{operation}: finalized block is boolean, not an integer"
            )
        finalized = int(raw_finalized)
        if finalized < inclusion_block:
            await _retry(
                f"anchor block {inclusion_block} is not finalized "
                f"(finalized={finalized})"
            )
            continue

        block_hash: str | None = None
        if require_block_hash:
            assert callable(block_hash_reader)
            try:
                raw_hash = await asyncio.to_thread(block_hash_reader, inclusion_block)
            except Exception as exc:
                await _retry(
                    f"finalized block-hash read unavailable: {type(exc).__name__}: {exc}"
                )
                continue
            if raw_hash is None:
                await _retry(
                    f"finalized anchor block {inclusion_block} has no visible block hash"
                )
                continue
            if not isinstance(raw_hash, str) or not SHA256_HEX.fullmatch(raw_hash):
                raise AnchorReceiptVerificationError(
                    f"{operation}: anchor block {inclusion_block} has a non-canonical "
                    "finalized block hash"
                )
            block_hash = raw_hash

        try:
            historical = await asyncio.to_thread(
                archive_reader,
                netuid=netuid,
                epoch_id=anchor_id,
                domain=domain,
                block_number=inclusion_block,
            )
        except Exception as exc:
            await _retry(
                f"archive read at finalized inclusion block {inclusion_block} "
                f"unavailable: {type(exc).__name__}: {exc}"
            )
            continue
        if historical != expected_digest:
            raise AnchorReceiptMismatch(
                f"{operation}: finalized archive state did not contain the submitted "
                f"anchor digest at inclusion block {inclusion_block}"
            )
        return FinalizedAnchorReceipt(
            block=inclusion_block,
            block_hash=block_hash,
        )


async def wait_for_finalized_commitment_receipt(
    chain: ChainAdapter,
    *,
    netuid: int,
    expected_payload: bytes,
    operation: str,
    timeout_seconds: float = DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS,
) -> FinalizedCommitmentReceipt:
    """Prove one already-submitted raw payload without ever resubmitting it.

    Competition commitments deliberately retain their public v1 payload bytes,
    which are not shaped like an epoch anchor.  The adapter therefore exposes one
    raw ``read_commitment_record`` seam carrying payload + original inclusion
    height atomically.  This verifier waits for independent head visibility,
    finality and a canonical block hash, then requires the exact same record from
    archive state at that inclusion block.
    """

    if timeout_seconds <= 0:
        raise ValueError("commitment receipt timeout must be positive")
    if poll_seconds <= 0:
        raise ValueError("commitment receipt poll interval must be positive")
    if not isinstance(expected_payload, bytes):
        raise TypeError("expected commitment payload must be bytes")
    if not expected_payload or len(expected_payload) > 128:
        raise ValueError("expected commitment payload must contain 1..128 bytes")

    record_reader = getattr(chain, "read_commitment_record", None)
    finalized_reader = getattr(chain, "finalized_block", None)
    block_hash_reader = getattr(chain, "block_hash", None)
    required = {
        "read_commitment_record": record_reader,
        "finalized_block": finalized_reader,
        "block_hash": block_hash_reader,
    }
    missing = [name for name, reader in required.items() if not callable(reader)]
    if missing:
        raise AnchorReceiptVerificationError(
            f"{operation}: chain adapter cannot prove a finalized raw commitment "
            f"receipt; missing {', '.join(sorted(missing))}"
        )

    assert callable(record_reader)
    assert callable(finalized_reader)
    assert callable(block_hash_reader)
    deadline = time.monotonic() + timeout_seconds
    last_observation = "commitment record was not yet visible"
    inclusion_block: int | None = None

    async def _retry(reason: str) -> None:
        nonlocal last_observation
        last_observation = reason
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AnchorReceiptTimeout(
                f"{operation}: timed out after {timeout_seconds:g}s waiting for "
                "independent finalized/archive commitment visibility; last "
                f"observation: {last_observation}"
            )
        await asyncio.sleep(min(poll_seconds, remaining))

    while True:
        if inclusion_block is None:
            try:
                record = await asyncio.to_thread(
                    record_reader, netuid=netuid, block_number=None
                )
            except Exception as exc:
                await _retry(
                    "current commitment record unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if record is not None and not isinstance(record, ChainCommitmentRecord):
                raise AnchorReceiptVerificationError(
                    f"{operation}: read_commitment_record returned "
                    f"{type(record).__name__}, expected ChainCommitmentRecord"
                )
            if record is None or record.payload != expected_payload:
                await _retry(
                    "current commitment slot has not exposed the submitted exact "
                    f"payload (present={record is not None})"
                )
                continue
            inclusion_block = record.block

        try:
            raw_finalized = await asyncio.to_thread(finalized_reader)
        except Exception as exc:
            await _retry(
                f"finalized-head read unavailable: {type(exc).__name__}: {exc}"
            )
            continue
        if isinstance(raw_finalized, bool):
            raise AnchorReceiptVerificationError(
                f"{operation}: finalized block is boolean, not an integer"
            )
        finalized = int(raw_finalized)
        if finalized < inclusion_block:
            await _retry(
                f"commitment block {inclusion_block} is not finalized "
                f"(finalized={finalized})"
            )
            continue

        try:
            raw_hash = await asyncio.to_thread(block_hash_reader, inclusion_block)
        except Exception as exc:
            await _retry(
                "finalized commitment block-hash read unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if raw_hash is None:
            await _retry(
                f"finalized commitment block {inclusion_block} has no visible block hash"
            )
            continue
        if not isinstance(raw_hash, str) or not SHA256_HEX.fullmatch(raw_hash):
            raise AnchorReceiptVerificationError(
                f"{operation}: commitment block {inclusion_block} has a "
                "non-canonical finalized block hash"
            )

        try:
            historical = await asyncio.to_thread(
                record_reader,
                netuid=netuid,
                block_number=inclusion_block,
            )
        except Exception as exc:
            await _retry(
                f"archive commitment read at finalized inclusion block "
                f"{inclusion_block} unavailable: {type(exc).__name__}: {exc}"
            )
            continue
        if (
            not isinstance(historical, ChainCommitmentRecord)
            or historical.payload != expected_payload
            or historical.block != inclusion_block
        ):
            raise AnchorReceiptMismatch(
                f"{operation}: finalized archive state did not contain the exact "
                f"submitted commitment record at inclusion block {inclusion_block}"
            )
        return FinalizedCommitmentReceipt(
            block=inclusion_block,
            block_hash=raw_hash,
            finalized_block=finalized,
        )


__all__ = [
    "AnchorReceiptMismatch",
    "AnchorReceiptTimeout",
    "AnchorReceiptVerificationError",
    "DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS",
    "DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS",
    "FinalizedAnchorReceipt",
    "FinalizedCommitmentReceipt",
    "wait_for_finalized_anchor_receipt",
    "wait_for_finalized_commitment_receipt",
]
