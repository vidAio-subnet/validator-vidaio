"""The Scoring Authority API contract — POINTER payloads (never snapshot bytes).

These are the exact response bodies the pointer API serves (build-wave 4,
the project design record §3.1). Every one is a THIN POINTER: an epoch id, the
object-store KEY to mirror, the content DIGESTS, and the on-chain anchor. The
epoch-log bytes are NEVER in any of these bodies — a validator fetches a pointer,
then pulls the bytes directly from the object store by `snapshot_key` and verifies

    sha256(fetched bytes) == snapshot_digest == on-chain anchored digest

before trusting/submitting (the tamper-evidence chain, §5). Keeping the API a pure
index makes it cheap, cacheable, and unable to become a per-request tampering
surface — the digest and the anchor establish trust, not the API.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.canonical import SHA256_HEX_PATTERN
from vidaio.authority.index import EpochRecord


class AnchorPointer(BaseModel):
    """The on-chain anchor inside an epoch pointer: txid + the anchored digest.

    `digest` is the epoch-log `log_digest` that was anchored — the tamper-evidence
    root. `txid` is None until the digest has been anchored on chain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    txid: str | None
    digest: str = Field(pattern=SHA256_HEX_PATTERN)
    #: Chain block whose state contains this commitment. A bittensor reader uses
    #: it for an archive lookup after the account's single head slot is replaced.
    block: int | None = Field(default=None, ge=0)


class EpochPointer(BaseModel):
    """The pointer to one finalized epoch's log — keys + digests + anchor, NO bytes.

    `snapshot_key` is the object-store key a validator mirrors; `snapshot_digest`
    (== the epoch-log `log_digest`) is what `sha256(mirrored bytes)` must equal and
    what the anchor binds; `weight_vector_digest` binds the u16 vector.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch_id: int
    close_block: int
    snapshot_key: str
    snapshot_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    weight_vector_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    finalized: bool = True
    anchor: AnchorPointer


class AnchorRecord(BaseModel):
    """The standalone anchor record (`GET /epoch/{id}/anchor`) for on-chain checks.

    Everything a third party needs to independently verify the anchor: the epoch,
    the anchored `digest` (the epoch-log log_digest), the `txid`, and the `block`
    the anchor landed at (None if the adapter cannot report it cheaply).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch_id: int
    digest: str = Field(pattern=SHA256_HEX_PATTERN)
    txid: str | None
    block: int | None = None


def pointer_from_record(record: EpochRecord) -> EpochPointer:
    """Project an indexed epoch into its public pointer body."""
    return EpochPointer(
        epoch_id=record.epoch_id,
        close_block=record.close_block,
        snapshot_key=record.snapshot_key,
        snapshot_digest=record.log_digest,
        weight_vector_digest=record.weight_vector_digest,
        finalized=True,
        anchor=AnchorPointer(
            txid=record.anchor_txid,
            digest=record.log_digest,
            block=record.anchor_block,
        ),
    )


def anchor_from_record(record: EpochRecord) -> AnchorRecord:
    """Project an indexed epoch into its standalone anchor record."""
    return AnchorRecord(
        epoch_id=record.epoch_id,
        digest=record.log_digest,
        txid=record.anchor_txid,
        block=record.anchor_block,
    )
