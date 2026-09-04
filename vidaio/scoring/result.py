"""ItemScore — the audit-grade per-item record (spec §08: independently recomputable).

Carries everything a third party needs to recompute the score from the audit store:
every metric input value, the full formula breakdown, gate outcomes with reason codes,
backend versions, the canonicalization plan digest, the derived PieAPP start frame and
a digest of the scoring config in force. Round-trips losslessly through JSON.
"""

from __future__ import annotations

import hashlib
import math
from typing import Annotated, Union

from pydantic import BaseModel, Field, field_validator

from vidaio.scoring.compression import CompressionBreakdown
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.gates import GateSkip, ValidityViolation
from vidaio.scoring.upscaling import UpscalingBreakdown

Breakdown = Annotated[
    Union[CompressionBreakdown, UpscalingBreakdown], Field(discriminator="kind")
]


def config_digest(config: ScoringConfig) -> str:
    """sha256 of the canonical JSON dump of the scoring config in force."""
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


class ItemScore(BaseModel):
    """One scored item. ``score`` is 0.0 whenever ``gate_passed`` is False —
    gates-first is a structural invariant, not a convention (see :func:`compose_item_score`).
    """

    model_config = {"frozen": True}

    # identity
    item_id: str
    challenge_id: str
    track: str
    miner_hotkey: str | None = None
    content_digest: str | None = None

    # outcome
    #: Bounded by construction: finite and in [0, 1]. An Infinity/NaN packet must be
    #: unconstructible AND unparseable (``from_json`` rejects it too) — an unbounded
    #: score would otherwise flow into ranking before the audit recompute catches it.
    score: float
    gate_passed: bool
    violations: list[ValidityViolation] = Field(default_factory=list)
    #: Checks consciously disabled by config (e.g. ``require_secondary_vmaf=False``) —
    #: persisted so the audit packet itself shows which gates did NOT run and why,
    #: rather than leaving that only on the transient GateContext.
    skips: list[GateSkip] = Field(default_factory=list)
    breakdown: Breakdown | None = None

    # raw metric inputs (everything the recompute needs, beyond the breakdown)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)

    # provenance
    #: pins the packet to the scorer that produced it; the audit layer compares it
    #: against AuditBundle.scorer_version during verification.
    scorer_version: str | None = None
    backend_versions: dict[str, str] = Field(default_factory=dict)
    canonicalization_plan_digest: str | None = None
    pieapp_start_frame: int | None = None
    scoring_config_digest: str | None = None

    @field_validator("score")
    @classmethod
    def _score_bounded(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"score must be finite, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {value!r}")
        return value

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> "ItemScore":
        return cls.model_validate_json(payload)


def compose_item_score(
    *,
    item_id: str,
    challenge_id: str,
    track: str,
    gate_passed: bool,
    violations: list[ValidityViolation],
    breakdown: CompressionBreakdown | UpscalingBreakdown | None,
    config: ScoringConfig,
    skips: list[GateSkip] | None = None,
    miner_hotkey: str | None = None,
    content_digest: str | None = None,
    metrics: dict[str, float | int | str | None] | None = None,
    backend_versions: dict[str, str] | None = None,
    canonicalization_plan_digest: str | None = None,
    pieapp_start_frame: int | None = None,
    scorer_version: str | None = None,
) -> ItemScore:
    """Assemble the final ItemScore, enforcing gates-first zeroing.

    The score is ``breakdown.final`` ONLY when every gate passed; any violation forces
    0.0 regardless of how good the metrics were. A breakdown-internal zero reason
    (e.g. the compression rate case) also yields 0 via ``breakdown.final == 0``.

    ``skips`` is threaded from ``GateContext.skips`` so consciously-disabled checks
    become part of the persisted audit packet, not transient pipeline state.
    """
    if gate_passed and breakdown is not None:
        score = breakdown.final
    else:
        score = 0.0
    return ItemScore(
        item_id=item_id,
        challenge_id=challenge_id,
        track=track,
        miner_hotkey=miner_hotkey,
        content_digest=content_digest,
        score=score,
        gate_passed=gate_passed,
        violations=violations,
        skips=list(skips) if skips else [],
        breakdown=breakdown,
        metrics=metrics or {},
        scorer_version=scorer_version,
        backend_versions=backend_versions or {},
        canonicalization_plan_digest=canonicalization_plan_digest,
        pieapp_start_frame=pieapp_start_frame,
        scoring_config_digest=config_digest(config),
    )
