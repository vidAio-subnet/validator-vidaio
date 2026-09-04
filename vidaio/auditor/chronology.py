"""Independent proof that a miner result happened after challenge commitment.

The chain's commitment slot is mutable, so a receipt is meaningful only at its
exact finalized archive block. The miner's response signature binds that receipt,
including the finalized block hash; this is the unpredictable fact that prevents a
compromised authority from pre-signing a result and anchoring the challenge later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from vidaio.audit import AuditBundle, AuditStore, IntegrityError, sha256_hex
from vidaio.challenge import CHALLENGE_ANCHOR_DOMAIN, ChallengeCommitment
from vidaio.scoring.duplicate_evidence import (
    InvalidDuplicateEvidence,
    canonical_receipt_digest,
    duplicate_order_key,
    duplicate_witness_from_packet,
    is_duplicate_identity,
)
from vidaio.scoring.config import ScoringConfig
from vidaio.services.artifact_auth import (
    MinerArtifactReceipt,
    verify_miner_artifact_receipt,
)

CHALLENGE_CHRONOLOGY_INVALID = "CHALLENGE_CHRONOLOGY_INVALID"
CHALLENGE_CHRONOLOGY_UNVERIFIED = "CHALLENGE_CHRONOLOGY_UNVERIFIED"
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_VALIDATOR_ZERO_SCORER_NAME = "validator-zero/1"


class ChronologyKind(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class ChronologyResult:
    kind: ChronologyKind
    code: str = ""
    detail: str = ""


def _pass() -> ChronologyResult:
    return ChronologyResult(ChronologyKind.PASS)


def _fail(detail: str) -> ChronologyResult:
    return ChronologyResult(ChronologyKind.FAIL, CHALLENGE_CHRONOLOGY_INVALID, detail)


def _skip(detail: str) -> ChronologyResult:
    return ChronologyResult(
        ChronologyKind.SKIP, CHALLENGE_CHRONOLOGY_UNVERIFIED, detail
    )


def _is_validator_zero_identity(identity: object) -> bool:
    value = str(identity or "")
    return value == _VALIDATOR_ZERO_SCORER_NAME or value.startswith(
        f"{_VALIDATOR_ZERO_SCORER_NAME}+"
    )


def _read_reveal(store: AuditStore, bundle: AuditBundle) -> bytes | None:
    if bundle.dag_reveal is None:
        return None
    try:
        reveal = store.get_limited(bundle.dag_reveal, _MAX_METADATA_BYTES)
    except (IntegrityError, FileNotFoundError, OSError):
        return None
    if sha256_hex(reveal) != bundle.commitment_hash:
        return None
    return reveal


def _read_packet(store: AuditStore, bundle: AuditBundle) -> dict | None:
    try:
        raw = store.get_limited(bundle.score_packet, _MAX_METADATA_BYTES)
        packet = json.loads(raw)
    except (IntegrityError, FileNotFoundError, OSError, ValueError, TypeError):
        return None
    return packet if isinstance(packet, dict) else None


def verify_challenge_chronology(
    bundle: AuditBundle,
    store: AuditStore,
    chain: object | None,
    *,
    require_anchor: bool,
    expected_netuid: int,
    scoring: ScoringConfig,
    receipt_verifier: Callable[[MinerArtifactReceipt], bool] = (
        verify_miner_artifact_receipt
    ),
) -> ChronologyResult:
    """Verify finalized archive anchor + miner-signed result ordering."""
    anchor = bundle.challenge_anchor
    if anchor is None:
        if require_anchor:
            return _skip("audit bundle has no external pre-dispatch challenge receipt")
        return _pass()
    if anchor.netuid != expected_netuid:
        return _fail(
            f"challenge anchor netuid {anchor.netuid} != expected {expected_netuid}"
        )
    if anchor.commitment_hash != bundle.commitment_hash:
        return _fail("challenge anchor digest does not equal the bundle commitment")
    if anchor.block_hash is None:
        return _skip("challenge anchor receipt has no finalized block hash")
    if chain is None:
        return _skip("no independent chain adapter is wired for challenge receipt")

    finalized_reader = getattr(chain, "finalized_block", None)
    block_hash_reader = getattr(chain, "block_hash", None)
    archive_reader = getattr(chain, "read_anchor_at", None)
    if not all(
        callable(reader)
        for reader in (
            finalized_reader,
            block_hash_reader,
            archive_reader,
        )
    ):
        return _skip(
            "chain adapter lacks finalized_block/block_hash/read_anchor_at chronology seams"
        )
    try:
        finalized = int(finalized_reader())
        observed_block_hash = block_hash_reader(anchor.block)
        observed_digest = archive_reader(
            netuid=anchor.netuid,
            epoch_id=anchor.dispatch_ordering_key,
            domain=CHALLENGE_ANCHOR_DOMAIN,
            block_number=anchor.block,
        )
    except Exception as exc:
        return _skip(
            f"challenge receipt archive/finality read unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    if finalized < anchor.block:
        return _skip(
            f"challenge anchor block {anchor.block} is not finalized (finalized={finalized})"
        )
    if observed_block_hash is None:
        return _skip(f"block hash {anchor.block} is not independently readable")
    if str(observed_block_hash).lower().removeprefix("0x") != anchor.block_hash:
        return _fail(
            "challenge anchor finalized block hash does not match chain history"
        )
    if observed_digest != anchor.commitment_hash:
        return _fail(
            "archive state at the claimed finalized block does not contain the "
            "challenge commitment"
        )

    receipt = bundle.miner_receipt
    if receipt is None:
        packet = _read_packet(store, bundle)
        if packet is not None and _is_validator_zero_identity(
            packet.get("scorer_version")
        ):
            return _fail(
                "validator-zero packets are not launch-valid economic evidence; "
                "authority-observed failures must be non-punitive skips"
            )
        return _skip("measured packet has no miner-signed response receipt")

    reveal = _read_reveal(store, bundle)
    committed = (
        None
        if reveal is None
        else ChallengeCommitment.committed_dispatch_from_preimage(reveal)
    )
    if committed is None:
        return _skip("challenge reveal cannot bind receipt track/ordering key")
    committed_track, committed_key = committed
    allowed_item_ids = {
        receipt.metadata.task_id,
        f"{receipt.metadata.task_id}-c{anchor.dispatch_ordering_key}",
    }
    if (
        receipt.metadata.commitment_anchor != anchor
        or committed_key != anchor.dispatch_ordering_key
        or receipt.metadata.track != committed_track
        or not receipt.metadata.task_id.startswith(f"{bundle.challenge_id}:")
        or bundle.item_id not in allowed_item_ids
        or receipt.metadata.input_digest != bundle.challenge_input.digest
        or receipt.input_size != bundle.challenge_input.byte_size
        or receipt.output_digest != bundle.miner_output.digest
        or receipt.output_size != bundle.miner_output.byte_size
        or receipt.miner_hotkey != bundle.miner_hotkey
    ):
        return _fail(
            "miner receipt is not bound to the exact anchor, committed track/order, "
            "challenge task, miner and media"
        )
    try:
        signature_ok = bool(receipt_verifier(receipt))
    except Exception as exc:
        return _skip(
            f"miner receipt signature verifier unavailable: {type(exc).__name__}: {exc}"
        )
    if not signature_ok:
        return _fail("miner response signature is invalid")

    packet = _read_packet(store, bundle)
    if packet is not None and is_duplicate_identity(packet.get("scorer_version")):
        try:
            witness = duplicate_witness_from_packet(packet)
        except InvalidDuplicateEvidence as exc:
            return _fail(f"duplicate witness is invalid: {exc}")
        winner_receipt = witness.winner_receipt
        if (
            canonical_receipt_digest(receipt) != witness.loser_receipt_digest
            or witness.loser_hotkey != receipt.miner_hotkey
            or witness.loser_output_digest != receipt.output_digest
            or receipt.metadata.task_id != f"{bundle.challenge_id}:{witness.loser_uid}"
            or winner_receipt.metadata.commitment_anchor != anchor
            or winner_receipt.metadata.task_id
            != f"{bundle.challenge_id}:{witness.winner_uid}"
            or winner_receipt.metadata.track != committed_track
            or winner_receipt.metadata.input_digest != bundle.challenge_input.digest
            or winner_receipt.input_size != bundle.challenge_input.byte_size
            or winner_receipt.output_digest != witness.winner_output.digest
            or winner_receipt.output_size != witness.winner_output.byte_size
            or winner_receipt.miner_hotkey != witness.winner_hotkey
            or winner_receipt.validator_hotkey != receipt.validator_hotkey
        ):
            return _fail(
                "duplicate winner receipt is not bound to the same anchor, challenge,"
                " validator, miner identity and media"
            )
        try:
            winner_order = duplicate_order_key(anchor.block_hash, witness.winner_hotkey)
            loser_order = duplicate_order_key(anchor.block_hash, witness.loser_hotkey)
        except InvalidDuplicateEvidence as exc:
            return _fail(f"duplicate deterministic ordering is invalid: {exc}")
        if winner_order >= loser_order:
            return _fail(
                "duplicate winner violates anchor_hash_hotkey/1 deterministic ordering"
            )
        try:
            winner_signature_ok = bool(receipt_verifier(winner_receipt))
        except Exception as exc:
            return _skip(
                "duplicate winner receipt signature verifier unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        if not winner_signature_ok:
            return _fail("duplicate winner response signature is invalid")
    return _pass()


__all__ = [
    "CHALLENGE_CHRONOLOGY_INVALID",
    "CHALLENGE_CHRONOLOGY_UNVERIFIED",
    "ChronologyKind",
    "ChronologyResult",
    "verify_challenge_chronology",
]
