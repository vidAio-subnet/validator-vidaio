"""Trusted read seam for current-epoch-schema CROWN promotion evidence.

The registry never accepts an ``anchor_verified`` boolean or a caller-selected
winner.  A :class:`VerifiedCrownEpochSource` implementation owns the expensive
boundary: fetch the canonical epoch snapshot, verify its digest against the
archive on-chain anchor, independently rederive the competition result and
reward-window transition, and only then return the immutable DTO below.

The DTO deliberately carries the old baseline identity, exact machine winner,
archived submission and complete packet/bundle matrix.  Promotion therefore has
no reason to consult operational ranking or human review state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidaio.audit.canonical import SHA256_HEX_PATTERN
from vidaio.audit.store import ArtifactKind, ArtifactRef

_SHA1_HEX_PATTERN = r"^[0-9a-f]{40}$"


class CrownAuditItem(BaseModel):
    """One exact packet/bundle pair behind the machine winner's aggregate."""

    model_config = ConfigDict(frozen=True)

    item_index: int = Field(ge=0)
    item_id: str = Field(min_length=1)
    challenge_id: str = Field(min_length=1)
    score_packet: ArtifactRef
    audit_bundle: ArtifactRef

    @model_validator(mode="after")
    def _typed_refs(self) -> "CrownAuditItem":
        if self.score_packet.kind is not ArtifactKind.SCORE_PACKET:
            raise ValueError("winner packet matrix must use score_packet refs")
        if self.audit_bundle.kind is not ArtifactKind.AUDIT_BUNDLE:
            raise ValueError("winner packet matrix must use audit_bundle refs")
        if self.score_packet.byte_size <= 0 or self.audit_bundle.byte_size <= 0:
            raise ValueError("winner packet/bundle evidence must be non-empty")
        return self


class VerifiedCrownEpoch(BaseModel):
    """Immutable result returned only after canonical anchor/rederivation checks.

    ``schema_version`` and ``reward_window_state`` are intentionally not Literals:
    the registry repeats those cheap fail-closed checks at its own boundary, so a
    broken source cannot accidentally turn a PODIUM or foreign-schema snapshot into
    executable state.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    reward_window_state: str = Field(min_length=1)
    epoch_id: str = Field(min_length=1)
    snapshot_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    anchor_block: int = Field(ge=0)
    anchor_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    competition_id: str = Field(min_length=1)
    track: str = Field(min_length=1)
    cycle: int = Field(ge=1)
    completed_at: datetime
    reward_starts_at: datetime
    reward_ends_at: datetime
    winner_uid: int = Field(ge=0)
    winner_hotkey: str = Field(min_length=1)
    winner_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    baseline_score: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    winner_margin: float = Field(ge=0.0, allow_inf_nan=False)
    baseline_version: int = Field(ge=0)
    baseline_artifact_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    winner_submission: ArtifactRef
    winner_image_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    winner_repo_url: str = Field(min_length=1)
    winner_commit_sha: str = Field(pattern=_SHA1_HEX_PATTERN)
    winner_tree_sha: str = Field(pattern=_SHA1_HEX_PATTERN)
    audit_items: tuple[CrownAuditItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> "VerifiedCrownEpoch":
        for field_name in ("completed_at", "reward_starts_at", "reward_ends_at"):
            value = getattr(self, field_name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        if not (self.completed_at <= self.reward_starts_at < self.reward_ends_at):
            raise ValueError(
                "reward window must start at/after completion and end after start"
            )
        if self.winner_submission.byte_size <= 0:
            raise ValueError("winner submission archive must be non-empty")
        if self.winner_submission.kind is not ArtifactKind.SUBMISSION_ARCHIVE:
            raise ValueError(
                "winner submission must use the sealed submission_archive kind"
            )
        indexes = [item.item_index for item in self.audit_items]
        if indexes != list(range(len(indexes))):
            raise ValueError(
                "winner audit matrix must be complete and ordered from item_index 0"
            )
        identities = {(item.item_id, item.challenge_id) for item in self.audit_items}
        if len(identities) != len(self.audit_items):
            raise ValueError("winner audit matrix contains duplicate item identity")
        packet_digests = {item.score_packet.digest for item in self.audit_items}
        bundle_digests = {item.audit_bundle.digest for item in self.audit_items}
        if len(packet_digests) != len(self.audit_items):
            raise ValueError("winner audit matrix reuses a score packet")
        if len(bundle_digests) != len(self.audit_items):
            raise ValueError("winner audit matrix reuses an audit bundle")
        return self

    @property
    def idempotence_key(self) -> tuple[str, str, str]:
        return self.snapshot_digest, self.competition_id, self.track

    def canonical_provenance_obj(self) -> dict[str, object]:
        """Stable promotion provenance archived by the registry after rerun."""
        return {
            "anchor_block": self.anchor_block,
            "anchor_digest": self.anchor_digest,
            "audit_items": [
                {
                    "audit_bundle": item.audit_bundle.model_dump(mode="json"),
                    "challenge_id": item.challenge_id,
                    "item_id": item.item_id,
                    "item_index": item.item_index,
                    "score_packet": item.score_packet.model_dump(mode="json"),
                }
                for item in self.audit_items
            ],
            "baseline_artifact_digest": self.baseline_artifact_digest,
            "baseline_score": self.baseline_score,
            "baseline_version": self.baseline_version,
            "competition_id": self.competition_id,
            "completed_at": self.completed_at.astimezone(timezone.utc).isoformat(),
            "cycle": self.cycle,
            "domain": "vidaio.registry.crown-promotion.v1",
            "epoch_id": self.epoch_id,
            "reward_ends_at": self.reward_ends_at.astimezone(timezone.utc).isoformat(),
            "reward_starts_at": self.reward_starts_at.astimezone(
                timezone.utc
            ).isoformat(),
            "reward_window_state": self.reward_window_state,
            "schema_version": self.schema_version,
            "snapshot_digest": self.snapshot_digest,
            "track": self.track,
            "winner_hotkey": self.winner_hotkey,
            "winner_image_digest": self.winner_image_digest,
            "winner_repo_url": self.winner_repo_url,
            "winner_commit_sha": self.winner_commit_sha,
            "winner_tree_sha": self.winner_tree_sha,
            "winner_margin": self.winner_margin,
            "winner_score": self.winner_score,
            "winner_submission": self.winner_submission.model_dump(mode="json"),
            "winner_uid": self.winner_uid,
        }


@runtime_checkable
class VerifiedCrownEpochSource(Protocol):
    """Archive/chain verifier used as the sole authority for promotion facts."""

    def verified_crown(self, snapshot_digest: str) -> VerifiedCrownEpoch | None:
        """Return a rederived anchored CROWN, or ``None`` if it is not verified.

        Implementations MUST verify canonical snapshot digest == the digest in the
        finalized on-chain archive receipt and independently rederive state CROWN,
        winner, baseline binding and packet/bundle matrix.  Merely reading the
        authority-published fields is not sufficient.
        """
        ...


__all__ = [
    "CrownAuditItem",
    "VerifiedCrownEpoch",
    "VerifiedCrownEpochSource",
]
