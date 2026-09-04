"""Independent proof of an earning competition's pre-enrollment chain anchor.

The orchestrator's SQLite receipt is useful provenance, but it is not chain proof.
Schema-v14 carries that receipt inside :class:`CompetitionInput`; the authority and
every auditor call this module against their own archive adapter before accepting the
earning evidence.  Availability failures and positive mismatches are deliberately
separate so auditors can report INCONCLUSIVE versus FAIL without making findings an
automatic weight-setting interlock.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Protocol

from vidaio.audit.canonical import SHA256_HEX_PATTERN, sha256_hex
from vidaio.audit.commitments import COMMITMENT_DOMAIN
from vidaio.chain.adapter import ChainCommitmentRecord


class CompetitionAnchorUnavailable(RuntimeError):
    """The independent archive/RPC proof could not be completed."""


class CompetitionAnchorMismatch(Exception):
    """Readable independent chain state contradicts the committed receipt."""


class CompetitionAnchorInput(Protocol):
    """Receipt fields required from the schema-v14 ``CompetitionInput``."""

    commitment_root: str
    anchor_netuid: int
    anchor_payload_hex: str
    anchor_payload_digest: str
    anchor_block: int
    anchor_block_hash: str
    anchor_finalized_block: int


def competition_anchor_payload(root: str) -> bytes:
    """Return the one canonical raw pallet payload for a competition root."""

    if not isinstance(root, str) or re.fullmatch(SHA256_HEX_PATTERN, root) is None:
        raise CompetitionAnchorMismatch(
            "competition commitment root is not canonical lowercase sha256 hex"
        )
    return f"{COMMITMENT_DOMAIN}:competition:{root}".encode("ascii")


def validate_competition_anchor_input(
    receipt: CompetitionAnchorInput,
) -> bytes:
    """Validate the self-contained payload/root/receipt relationships.

    This is a structural check only.  :func:`verify_competition_anchor_on_chain`
    adds the independent archive proof.
    """

    expected_payload = competition_anchor_payload(receipt.commitment_root)
    try:
        payload = bytes.fromhex(receipt.anchor_payload_hex)
    except (TypeError, ValueError) as exc:
        raise CompetitionAnchorMismatch(
            "competition anchor payload is not canonical hex"
        ) from exc
    if payload != expected_payload:
        raise CompetitionAnchorMismatch(
            "competition anchor payload does not bind commitment_root"
        )
    if sha256_hex(payload) != receipt.anchor_payload_digest:
        raise CompetitionAnchorMismatch(
            "competition anchor payload digest does not bind its exact bytes"
        )
    if receipt.anchor_finalized_block < receipt.anchor_block:
        raise CompetitionAnchorMismatch(
            "competition anchor finalized height precedes its inclusion height"
        )
    return payload


def _read(reader, operation: str, *args, **kwargs):
    try:
        return reader(*args, **kwargs)
    except CompetitionAnchorMismatch:
        raise
    except Exception as exc:  # archive/RPC transport, decoding, or pruned state
        raise CompetitionAnchorUnavailable(
            f"{operation} is unavailable: {type(exc).__name__}: {exc}"
        ) from exc


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise CompetitionAnchorUnavailable(f"{name} returned boolean, not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CompetitionAnchorUnavailable(
            f"{name} returned an invalid integer: {value!r}"
        ) from exc
    if result < 0:
        raise CompetitionAnchorUnavailable(f"{name} returned a negative height")
    return result


def verify_competition_anchor_on_chain(
    chain: object | None,
    receipt: CompetitionAnchorInput,
    *,
    expected_netuid: int,
    competition_start_time: datetime,
    epoch_close_block: int,
) -> None:
    """Independently prove the exact finalized payload and its chronology.

    Required adapter seams are intentionally read-only: ``finalized_block``,
    ``block_hash``, ``block_time`` and the raw archive
    ``read_commitment_record``.  A missing/failed seam is UNAVAILABLE; a readable
    different payload, hash, inclusion height, subnet, or chronology is MISMATCH.
    """

    payload = validate_competition_anchor_input(receipt)
    if receipt.anchor_netuid != expected_netuid:
        raise CompetitionAnchorMismatch(
            f"competition anchor netuid {receipt.anchor_netuid} differs from the "
            f"audited subnet {expected_netuid}"
        )
    if receipt.anchor_block >= epoch_close_block:
        raise CompetitionAnchorMismatch(
            "competition anchor inclusion block does not precede the earning epoch close"
        )
    if receipt.anchor_finalized_block >= epoch_close_block:
        raise CompetitionAnchorMismatch(
            "competition anchor finalized receipt does not precede the earning epoch close"
        )
    if competition_start_time.tzinfo is None or competition_start_time.utcoffset() is None:
        raise CompetitionAnchorMismatch(
            "competition manifest start_time is timezone-naive"
        )

    required = {
        "finalized_block": getattr(chain, "finalized_block", None),
        "block_hash": getattr(chain, "block_hash", None),
        "block_time": getattr(chain, "block_time", None),
        "read_commitment_record": getattr(chain, "read_commitment_record", None),
    }
    missing = sorted(name for name, reader in required.items() if not callable(reader))
    if missing:
        raise CompetitionAnchorUnavailable(
            "chain adapter cannot independently prove the competition anchor; missing "
            + ", ".join(missing)
        )

    finalized_reader = required["finalized_block"]
    hash_reader = required["block_hash"]
    time_reader = required["block_time"]
    record_reader = required["read_commitment_record"]
    finalized = _integer(
        _read(finalized_reader, "finalized_block"), name="finalized_block"
    )
    if finalized < receipt.anchor_finalized_block:
        raise CompetitionAnchorUnavailable(
            f"independent finalized head {finalized} has not reached the committed "
            f"receipt height {receipt.anchor_finalized_block}"
        )

    block_hash = _read(hash_reader, "anchor block_hash", receipt.anchor_block)
    if block_hash is None:
        raise CompetitionAnchorUnavailable(
            f"anchor block {receipt.anchor_block} has no archive-readable block hash"
        )
    if (
        not isinstance(block_hash, str)
        or re.fullmatch(SHA256_HEX_PATTERN, block_hash) is None
    ):
        raise CompetitionAnchorUnavailable(
            "archive returned a non-canonical competition anchor block hash"
        )
    if block_hash != receipt.anchor_block_hash:
        raise CompetitionAnchorMismatch(
            "competition anchor block hash differs from independent archive state"
        )

    record = _read(
        record_reader,
        "raw commitment archive read",
        netuid=receipt.anchor_netuid,
        block_number=receipt.anchor_block,
    )
    if record is None:
        raise CompetitionAnchorMismatch(
            "competition anchor slot is empty at the claimed inclusion block"
        )
    if not isinstance(record, ChainCommitmentRecord):
        raise CompetitionAnchorUnavailable(
            "read_commitment_record returned an unsupported record type"
        )
    if record.block != receipt.anchor_block:
        raise CompetitionAnchorMismatch(
            "archived commitment record's inclusion height differs from the receipt"
        )
    if record.payload != payload:
        raise CompetitionAnchorMismatch(
            "archived commitment record does not contain the exact committed payload"
        )

    anchor_time = _read(time_reader, "anchor block_time", receipt.anchor_block)
    finalized_time = _read(
        time_reader, "receipt finalized block_time", receipt.anchor_finalized_block
    )
    if anchor_time is None or finalized_time is None:
        raise CompetitionAnchorUnavailable(
            "anchor inclusion/finality chronology has no archive-readable block time"
        )
    for name, value in (
        ("anchor inclusion block_time", anchor_time),
        ("anchor finalized block_time", finalized_time),
    ):
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise CompetitionAnchorUnavailable(f"{name} is not timezone-aware")
    if anchor_time >= competition_start_time:
        raise CompetitionAnchorMismatch(
            "competition anchor inclusion did not precede the manifest enrollment start"
        )
    if finalized_time >= competition_start_time:
        raise CompetitionAnchorMismatch(
            "competition anchor was not finalized before the manifest enrollment start"
        )


__all__ = [
    "CompetitionAnchorMismatch",
    "CompetitionAnchorUnavailable",
    "competition_anchor_payload",
    "validate_competition_anchor_input",
    "verify_competition_anchor_on_chain",
]
