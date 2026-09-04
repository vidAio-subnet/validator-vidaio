"""Competition manifest — what a competition declares (spec: design spec §04 manifest table).

The manifest is strictly validated at creation and stored verbatim on the competitions
row. `manifest_digest()` is a sha256 over canonical JSON; the audit module pre-commits
that digest on chain before enrollment opens — this module only exposes the digest.

The ``baseline`` block binds the active registry version, its archived source and
provenance, and the exact pinned Git tree used to build it.  It has no participant
identity or payout field: it is an executable quality floor, evaluated over the
same hidden item matrix but excluded from ranking by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from vidaio.competition.config import CompetitionConfig
from vidaio.competition.item_commitment import (
    evaluation_item_commitment as evaluation_item_commitment,
)

_SHA1_HEX = r"^[0-9a-f]{40}$"
_SHA256_HEX = r"^[0-9a-f]{64}$"

#: Substrings that mark a field as payout/identity-bearing — forbidden on the baseline
#: block (non-earning calibration baseline, the project design record #1).
_BASELINE_FORBIDDEN_TOKENS = ("hotkey", "coldkey", "payout", "wallet", "reward", "emission")


class ManifestBoundsError(ValueError):
    """Manifest is schema-valid but outside the operator-configured envelope."""


class ScoringFactors(BaseModel):
    """quality / cost / length split — must sum to exactly 1.0 (comp-01: 0.6/0.0/0.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quality: float = Field(ge=0, le=1)
    cost_efficiency: float = Field(ge=0, le=1)
    length_coverage: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "ScoringFactors":
        total = self.quality + self.cost_efficiency + self.length_coverage
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"scoring_factors must sum to 1.0, got {total}")
        return self


class ArchivedBaseline(BaseModel):
    """Versioned archived executable floor, never a participant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=0)
    artifact_digest: str = Field(pattern=_SHA256_HEX)
    artifact_bytes: int = Field(gt=0)
    image_digest: str = Field(pattern=_SHA256_HEX)
    provenance_digest: str = Field(pattern=_SHA256_HEX)
    provenance_bytes: int = Field(gt=0)
    repo_url: str = Field(min_length=1)
    commit_sha: str = Field(pattern=_SHA1_HEX)
    tree_sha: str = Field(pattern=_SHA1_HEX)

    @model_validator(mode="before")
    @classmethod
    def _forbid_payout_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            banned = sorted(
                key
                for key in data
                if isinstance(key, str)
                and any(tok in key.lower() for tok in _BASELINE_FORBIDDEN_TOKENS)
            )
            if banned:
                raise ValueError(
                    "an archived baseline is not a participant; payout/identity fields "
                    f"are forbidden on the baseline block: {banned}"
                )
        return data


class EvaluationBatchSizeBounds(BaseModel):
    """Items per sandbox batch (comp-01: 1-5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "EvaluationBatchSizeBounds":
        if self.min > self.max:
            raise ValueError(f"evaluation_batch_size min ({self.min}) > max ({self.max})")
        return self


class CompetitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_schema_version: Literal[2] = 2
    competition_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    track: Literal["compression", "upscaling"] = "compression"

    # Lifecycle times (all timezone-aware; normalized to UTC on validation):
    #   start_time           SCHEDULED -> ENROLLING (gated on no other running)
    #   enrollment_deadline  last instant a contender may enroll (alpha-stake gate)
    #   finalization_time    ENROLLING -> FINALIZING_SUBMISSIONS
    #   end_time             AWAITING_END_TIME -> COMPLETED
    start_time: AwareDatetime
    enrollment_deadline: AwareDatetime
    finalization_time: AwareDatetime
    end_time: AwareDatetime

    minimum_alpha_stake: float = Field(ge=0)
    scoring_factors: ScoringFactors
    #: Quality-term selector for ranking (spec §18 worst-decile bottleneck aggregation).
    #: False (default): the quality factor multiplies the length-weighted mean — the
    #: aggregation the manifest's declared scoring_factors describe. True: it multiplies
    #: the worst-decile aggregate instead (one excellent item can never offset a failed
    #: one). BOTH aggregates are always computed and stored on the contender row
    #: (media_score_aggregate / worst_decile_aggregate); this flag only picks which one
    #: the final_score ranks on.
    use_worst_decile: bool = False
    vmaf_threshold: float = Field(ge=0, le=100)
    #: Sealed per-item threshold variants (comp-01: 85/89/93). Only the commitment to
    #: the per-item assignment is public before evaluation; values live here so the
    #: manifest digest commits to the variant set itself.
    sealed_vmaf_variants: list[float] = Field(min_length=1)
    #: Upscaling-only, ordered factor allow-list.  ``None`` is deliberately omitted
    #: from legacy compression canonical JSON so loading a pre-migration manifest
    #: does not change its already-anchored digest.
    allowed_upscale_factors: list[Literal[2, 4]] | None = None
    #: Ordered by ``item_index``.  Each digest commits the high-resolution reference,
    #: low-resolution miner input, and factor via ``evaluation_item_commitment``.
    #: The list itself is inside the canonical manifest digest anchored before
    #: enrollment; mutable repository metadata is therefore not an audit trust root.
    evaluation_item_commitments: list[str] | None = None
    allowed_gpus: list[str] = Field(min_length=1)
    evaluation_batch_size: EvaluationBatchSizeBounds
    #: sha256 commitment to the deterministic dataset-variant RNG seed — the hash,
    #: never the seed itself (the seed is revealed post-competition for audit).
    scoring_seed_commitment: str = Field(pattern=_SHA256_HEX)
    container_size_limit_gb: float = Field(gt=0)
    scoring_version: str = Field(min_length=1)
    baseline: ArchivedBaseline | None = None

    @field_validator("start_time", "enrollment_deadline", "finalization_time", "end_time")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        # Normalize to UTC so canonical JSON (and therefore the digest) is independent
        # of the timezone representation the manifest author used.
        return value.astimezone(timezone.utc)

    @field_validator("sealed_vmaf_variants")
    @classmethod
    def _variants_in_range(cls, value: list[float]) -> list[float]:
        for v in value:
            if not 0 <= v <= 100:
                raise ValueError(f"sealed vmaf variant {v} outside [0, 100]")
        return value

    @field_validator("allowed_gpus")
    @classmethod
    def _gpus_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [g.strip() for g in value]
        if any(not g for g in cleaned):
            raise ValueError("allowed_gpus entries must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("allowed_gpus entries must be unique")
        return cleaned

    @field_validator("allowed_upscale_factors")
    @classmethod
    def _upscale_factors_unique(
        cls, value: list[Literal[2, 4]] | None
    ) -> list[Literal[2, 4]] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("allowed_upscale_factors entries must be unique")
        return value

    @field_validator("evaluation_item_commitments")
    @classmethod
    def _item_commitments_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(re.fullmatch(_SHA256_HEX, digest) is None for digest in value):
            raise ValueError(
                "evaluation_item_commitments entries must be lowercase sha256 hex"
            )
        if len(set(value)) != len(value):
            raise ValueError("evaluation_item_commitments entries must be unique")
        return value

    @model_validator(mode="after")
    def _times_ordered(self) -> "CompetitionManifest":
        if not (self.start_time < self.enrollment_deadline):
            raise ValueError("start_time must be before enrollment_deadline")
        if not (self.enrollment_deadline <= self.finalization_time):
            raise ValueError("enrollment_deadline must be at or before finalization_time")
        if not (self.finalization_time < self.end_time):
            raise ValueError("finalization_time must be before end_time")
        if self.track == "upscaling":
            if not self.allowed_upscale_factors:
                raise ValueError(
                    "upscaling manifest requires non-empty allowed_upscale_factors"
                )
            if not self.evaluation_item_commitments:
                raise ValueError(
                    "upscaling manifest requires precommitted evaluation items"
                )
        elif (
            self.allowed_upscale_factors is not None
            or self.evaluation_item_commitments is not None
        ):
            raise ValueError(
                "upscaling factor/item commitments are valid only for the "
                "upscaling track"
            )
        return self

    # ---- canonical form & digest ----

    def canonical_json(self) -> str:
        """Deterministic JSON: sorted keys, no whitespace, UTC ISO-8601 datetimes."""
        payload = self.model_dump(mode="json")
        # Backward-compatible digest fence: these fields did not exist in the v1
        # compression manifest.  An absent value must not rewrite an anchored digest
        # merely because newer code loaded the old JSON.
        for field in ("allowed_upscale_factors", "evaluation_item_commitments"):
            if payload[field] is None:
                del payload[field]
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def manifest_digest(self) -> str:
        """sha256 over canonical JSON — the value pre-committed on chain (audit module)."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def validate_against_config(manifest: CompetitionManifest, cfg: CompetitionConfig) -> None:
    """Check a schema-valid manifest against the operator-configured envelope.

    Raises ManifestBoundsError on the first violation. Called by the engine before
    a competition row is created.
    """
    if not (cfg.vmaf_threshold_min <= manifest.vmaf_threshold <= cfg.vmaf_threshold_max):
        raise ManifestBoundsError(
            f"vmaf_threshold {manifest.vmaf_threshold} outside "
            f"[{cfg.vmaf_threshold_min}, {cfg.vmaf_threshold_max}]"
        )
    for v in manifest.sealed_vmaf_variants:
        if not (cfg.vmaf_threshold_min <= v <= cfg.vmaf_threshold_max):
            raise ManifestBoundsError(
                f"sealed vmaf variant {v} outside "
                f"[{cfg.vmaf_threshold_min}, {cfg.vmaf_threshold_max}]"
            )
    if manifest.container_size_limit_gb > cfg.container_size_limit_gb_max:
        raise ManifestBoundsError(
            f"container_size_limit_gb {manifest.container_size_limit_gb} exceeds "
            f"max {cfg.container_size_limit_gb_max}"
        )
    if manifest.evaluation_batch_size.max > cfg.evaluation_batch_size_max:
        raise ManifestBoundsError(
            f"evaluation_batch_size.max {manifest.evaluation_batch_size.max} exceeds "
            f"max {cfg.evaluation_batch_size_max}"
        )
    if manifest.minimum_alpha_stake < cfg.minimum_alpha_stake_min:
        raise ManifestBoundsError(
            f"minimum_alpha_stake {manifest.minimum_alpha_stake} below "
            f"floor {cfg.minimum_alpha_stake_min}"
        )
