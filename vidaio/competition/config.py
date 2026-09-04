"""Competition module configuration — the `competition:` section of config/default.yaml.

Holds the engine tick cadence, the human-review window, and the validation bounds a
manifest must satisfy before a competition is created (spec: design spec §04 manifest table,
§14 production bar). The manifest schema itself enforces universal invariants
(ordering, factor sums); these bounds are the operator-tunable envelope on top.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompetitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: How often the lifecycle engine's tick() is expected to be driven.
    tick_interval_seconds: float = Field(default=30.0, gt=0)

    #: Human review / disqualification window after scores are persisted
    #: (reviews allowed up to human_review_deadline; spec §04).
    human_review_window_hours: float = Field(default=24.0, ge=0)

    #: Audit-linkage completion gate: the AWAITING_END_TIME -> COMPLETED tick
    #: additionally requires every performance_history row — including the baseline
    #: calibration rows — to carry its audit_bundle_digest
    #: (engine.audit_linkage_gaps == []) AND, when a baseline calibration contender
    #: exists, a baseline score row for every evaluation item
    #: (repository.count_missing_calibration_rows == 0 — a baseline with zero rows
    #: must never slip past the gate). A competition failing either check stays
    #: in AWAITING_END_TIME (with a structured log line) until
    #: the audit runner links every bundle. False is allowed ONLY for tests/dev
    #: environments without an audit store; the bypass is logged on every
    #: completion. Production MUST run with True (the project design record: every scored
    #: metric must be independently recomputable from the audit store).
    require_audit_linkage: bool = True

    # ---- manifest validation bounds (defaults; see manifest.validate_against_config) ----

    #: VMAF quality-gate threshold must land inside this envelope (comp-01 uses 90
    #: with sealed per-item variants 85/89/93).
    vmaf_threshold_min: float = Field(default=50.0, ge=0, le=100)
    vmaf_threshold_max: float = Field(default=100.0, ge=0, le=100)

    #: Hard ceiling on the manifest's container image cap. NOTE: the 25 GB cap is
    #: "acknowledged-but-unmeasured" on Modal (size unattested) — spec §04/§05.
    container_size_limit_gb_max: float = Field(default=25.0, gt=0)

    #: Ceiling on the manifest's evaluation batch-size upper bound (comp-01: 1-5).
    evaluation_batch_size_max: int = Field(default=16, ge=1)

    #: Floor on the enrollment alpha-stake gate (0 disables the floor).
    minimum_alpha_stake_min: float = Field(default=0.0, ge=0)
