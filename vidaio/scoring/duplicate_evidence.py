"""Auditable byte-exact cross-miner duplicate score packets.

A launch duplicate penalty needs two real miner-signed outputs whose verified
SHA-256 digests are identical. Perceptual similarity is deliberately excluded:
honest restorations of the same scene are expected to look similar. The losing
receipt lives in the normal audit bundle; this canonical witness embeds the kept
peer's receipt and content-addressed output reference.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, ArtifactRef
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.gates import ReasonCode, ValidityViolation
from vidaio.scoring.result import ItemScore, compose_item_score, config_digest
from vidaio.services.artifact_auth import MinerArtifactReceipt

DUPLICATE_SCORER_NAME = "validator-exact-duplicate/1"
DUPLICATE_SCORER_PREFIX = f"{DUPLICATE_SCORER_NAME}+"
DUPLICATE_WITNESS_METRIC = "duplicate_witness"
DUPLICATE_WITNESS_VERSION = 2
DUPLICATE_SELECTION_RULE = "anchor_hash_hotkey/1"
DUPLICATE_EVIDENCE_RULE = "sha256_exact_output/1"
DUPLICATE_ORDER_DOMAIN = b"vidaio:duplicate-order:anchor-hash-hotkey:v1\x00"


class InvalidDuplicateEvidence(ValueError):
    """A claimed duplicate packet or witness violates the launch convention."""


class DuplicateWitness(BaseModel):
    """Canonical proof facts for one byte-exact loser and its kept peer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = DUPLICATE_WITNESS_VERSION
    evidence_rule: Literal["sha256_exact_output/1"] = DUPLICATE_EVIDENCE_RULE
    selection_rule: Literal["anchor_hash_hotkey/1"] = DUPLICATE_SELECTION_RULE
    committed_scorer_version: str
    track: str

    loser_uid: int = Field(ge=0)
    loser_hotkey: str
    loser_output_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    loser_output_size: int = Field(ge=1)
    loser_receipt_digest: str = Field(pattern=SHA256_HEX_PATTERN)

    winner_uid: int = Field(ge=0)
    winner_hotkey: str
    winner_output: ArtifactRef
    winner_receipt: MinerArtifactReceipt


def is_duplicate_identity(identity: str | None) -> bool:
    return bool(identity) and str(identity).startswith(DUPLICATE_SCORER_PREFIX)


def duplicate_identity(
    *, committed_scorer_version: str, track: str, scoring_config_digest: str
) -> str:
    """Derived identity bound to the worker and exact-only launch convention."""
    digest = sha256_hex(
        canonical_json_bytes(
            {
                "committed_scorer_version": committed_scorer_version,
                "convention": DUPLICATE_SCORER_NAME,
                "evidence_rule": DUPLICATE_EVIDENCE_RULE,
                "scoring_config_digest": scoring_config_digest,
                "selection_rule": DUPLICATE_SELECTION_RULE,
                "track": track,
            }
        )
    )
    return f"{DUPLICATE_SCORER_PREFIX}{digest[:12]}"


def canonical_receipt_digest(receipt: MinerArtifactReceipt) -> str:
    return sha256_hex(canonical_json_bytes(receipt.model_dump(mode="json")))


def duplicate_order_key(block_hash: str, hotkey: str) -> str:
    """Salt a signed identity with finalized, pre-dispatch chain state."""
    if not isinstance(block_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", block_hash):
        raise InvalidDuplicateEvidence(
            "duplicate anchor block hash is not lowercase sha256 hex"
        )
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or len(hotkey) > 128
        or any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in hotkey)
    ):
        raise InvalidDuplicateEvidence(
            "duplicate miner hotkey is not canonical printable ASCII"
        )
    rank = sha256_hex(
        DUPLICATE_ORDER_DOMAIN
        + bytes.fromhex(block_hash)
        + b"\x00"
        + hotkey.encode("ascii")
    )
    return f"{rank}\x00{hotkey}"


def encode_duplicate_witness(witness: DuplicateWitness) -> str:
    return canonical_json_bytes(witness.model_dump(mode="json")).decode("utf-8")


def parse_duplicate_witness(value: object) -> DuplicateWitness:
    if not isinstance(value, str) or not value:
        raise InvalidDuplicateEvidence(
            "duplicate_witness metric is missing or not text"
        )
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise InvalidDuplicateEvidence(f"duplicate witness is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidDuplicateEvidence("duplicate witness must be a JSON object")
    if canonical_json_bytes(payload).decode("utf-8") != value:
        raise InvalidDuplicateEvidence("duplicate witness is not canonical JSON")
    try:
        witness = DuplicateWitness.model_validate(payload)
    except ValidationError as exc:
        raise InvalidDuplicateEvidence(
            f"duplicate witness is malformed: {exc}"
        ) from exc
    if witness.winner_output.kind is not ArtifactKind.MINER_OUTPUT:
        raise InvalidDuplicateEvidence("winner_output must be a miner_output artifact")
    if witness.winner_hotkey == witness.loser_hotkey:
        raise InvalidDuplicateEvidence(
            "winner and loser must be distinct miner identities"
        )
    winner_anchor = witness.winner_receipt.metadata.commitment_anchor
    if winner_anchor is None or winner_anchor.block_hash is None:
        raise InvalidDuplicateEvidence(
            "duplicate winner receipt has no finalized anchor block hash"
        )
    if (
        witness.winner_output.digest != witness.winner_receipt.output_digest
        or witness.winner_output.byte_size != witness.winner_receipt.output_size
        or witness.winner_hotkey != witness.winner_receipt.miner_hotkey
    ):
        raise InvalidDuplicateEvidence(
            "winner output reference, identity and signed receipt do not agree"
        )
    if witness.loser_output_digest != witness.winner_output.digest:
        raise InvalidDuplicateEvidence(
            "economic duplicate evidence requires byte-exact output digests"
        )
    if witness.loser_output_size != witness.winner_output.byte_size:
        raise InvalidDuplicateEvidence("byte-exact duplicate output sizes do not agree")
    return witness


def duplicate_witness_from_packet(
    packet: ItemScore | Mapping[str, Any],
) -> DuplicateWitness:
    try:
        item = (
            packet
            if isinstance(packet, ItemScore)
            else ItemScore.model_validate(packet)
        )
    except Exception as exc:
        raise InvalidDuplicateEvidence(
            f"duplicate score packet is malformed: {exc}"
        ) from exc
    return parse_duplicate_witness(item.metrics.get(DUPLICATE_WITNESS_METRIC))


def mint_duplicate_packet(
    *,
    item_id: str,
    challenge_id: str,
    track: str,
    loser_uid: int,
    loser_hotkey: str,
    loser_output_digest: str,
    loser_output_size: int,
    loser_receipt: MinerArtifactReceipt,
    winner_uid: int,
    winner_hotkey: str,
    winner_output: ArtifactRef,
    winner_receipt: MinerArtifactReceipt,
    committed_scorer_version: str,
    config: ScoringConfig,
) -> ItemScore:
    """Mint an economic zero only from two byte-exact signed outputs."""
    if (
        loser_receipt.output_digest != loser_output_digest
        or loser_receipt.output_size != loser_output_size
        or loser_receipt.miner_hotkey != loser_hotkey
    ):
        raise InvalidDuplicateEvidence(
            "loser receipt does not bind the losing miner output"
        )
    if (
        loser_output_digest != winner_output.digest
        or loser_output_size != winner_output.byte_size
    ):
        raise InvalidDuplicateEvidence(
            "economic duplicate evidence requires byte-exact equal outputs"
        )
    loser_anchor = loser_receipt.metadata.commitment_anchor
    winner_anchor = winner_receipt.metadata.commitment_anchor
    if (
        loser_anchor is None
        or loser_anchor.block_hash is None
        or winner_anchor != loser_anchor
    ):
        raise InvalidDuplicateEvidence(
            "duplicate receipts must bind the same finalized challenge anchor"
        )
    if duplicate_order_key(
        loser_anchor.block_hash, winner_hotkey
    ) >= duplicate_order_key(loser_anchor.block_hash, loser_hotkey):
        raise InvalidDuplicateEvidence(
            "winner violates anchor_hash_hotkey/1 deterministic ordering"
        )
    witness = DuplicateWitness(
        committed_scorer_version=committed_scorer_version,
        track=track,
        loser_uid=loser_uid,
        loser_hotkey=loser_hotkey,
        loser_output_digest=loser_output_digest,
        loser_output_size=loser_output_size,
        loser_receipt_digest=canonical_receipt_digest(loser_receipt),
        winner_uid=winner_uid,
        winner_hotkey=winner_hotkey,
        winner_output=winner_output,
        winner_receipt=winner_receipt,
    )
    parse_duplicate_witness(encode_duplicate_witness(witness))
    scoring_digest = config_digest(config)
    identity = duplicate_identity(
        committed_scorer_version=committed_scorer_version,
        track=track,
        scoring_config_digest=scoring_digest,
    )
    return compose_item_score(
        item_id=item_id,
        challenge_id=challenge_id,
        track=track,
        gate_passed=False,
        violations=[
            ValidityViolation(
                code=ReasonCode.REPLAY_DUPLICATE,
                detail=(
                    "auditable byte-exact duplicate of deterministic "
                    f"anchor-salted winner {winner_hotkey}"
                ),
            )
        ],
        breakdown=None,
        config=config,
        miner_hotkey=loser_hotkey,
        content_digest=loser_output_digest,
        metrics={DUPLICATE_WITNESS_METRIC: encode_duplicate_witness(witness)},
        backend_versions={},
        scorer_version=identity,
    )


__all__ = [
    "DUPLICATE_EVIDENCE_RULE",
    "DUPLICATE_SCORER_NAME",
    "DUPLICATE_SCORER_PREFIX",
    "DUPLICATE_SELECTION_RULE",
    "DUPLICATE_WITNESS_METRIC",
    "DuplicateWitness",
    "InvalidDuplicateEvidence",
    "canonical_receipt_digest",
    "duplicate_identity",
    "duplicate_order_key",
    "duplicate_witness_from_packet",
    "encode_duplicate_witness",
    "is_duplicate_identity",
    "mint_duplicate_packet",
    "parse_duplicate_witness",
]
