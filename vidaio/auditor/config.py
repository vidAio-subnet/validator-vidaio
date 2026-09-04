"""Auditor configuration (config section: ``auditor``).

The beacon auditor and full own-auditor are isolated OS services, separate from the
thin weight-setter and from one another. In production each CPU-recomputes every
committed item and POSTs signed verdicts to the central Audit Results API. This module
holds the selection policy, recompute backend, report identity, and results-API pointer
used by their shared component.

Credentials follow the store's convention (the project design record open items): config
carries the NAME of the env var the token is read from, never the token value.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from vidaio.auditor.report import AuditMode
from vidaio.scoring.config import ScoringConfig
from vidaio.tokenomics.config import TokenomicsConfig


class SamplePolicy(BaseModel):
    """How many of an epoch's audit items THIS auditor recomputes, and how it picks.

    The generic component supports sampling, while both production service wrappers
    explicitly select uncapped ``all_items`` at rate 1.0:

    - ``sample_rate`` — the fraction of the item population to recompute, applied
      PER SOURCE (competition items and inference items are sampled independently,
      so both tracks always get coverage);
    - ``min_samples`` / ``max_samples`` — clamp the per-source count (never audit
      fewer than ``min`` when items exist, never more than ``max``);
    - ``all_items`` — explicit full-coverage mode that bypasses those clamps and
      recomputes the entire population;
    - selection is DETERMINISTIC, seeded from ``epoch_id + auditor_hotkey`` only
      (never wall-clock), so a run is reproducible and an operator cannot
      cherry-pick which items get audited (see ``vidaio.auditor.sampling``).

    A higher ``sample_rate`` raises whole-epoch confidence; any single provable
    FAIL is conclusive regardless (the project design record §5).
    """

    model_config = {"frozen": True}

    sample_rate: float = Field(0.10, ge=0.0, le=1.0)
    min_samples: int = Field(1, ge=0)
    max_samples: int = Field(50, ge=1)
    # Production audit workers use this explicit mode to recompute every committed media
    # item. It intentionally bypasses max_samples; sample_rate=1 alone is still capped.
    all_items: bool = False

    @model_validator(mode="after")
    def _sane(self) -> "SamplePolicy":
        if self.max_samples < self.min_samples:
            raise ValueError(
                f"max_samples ({self.max_samples}) must be >= min_samples "
                f"({self.min_samples})"
            )
        return self

    def target_count(self, population: int) -> int:
        """The number of items to draw from a ``population``-sized source.

        ``ceil(population * sample_rate)`` clamped into ``[min_samples,
        max_samples]`` and never above ``population`` (cannot sample what is not
        there). An empty population samples nothing.
        """
        if population <= 0:
            return 0
        if self.all_items:
            return population
        rate_count = math.ceil(population * self.sample_rate)
        count = max(self.min_samples, rate_count)
        count = min(count, self.max_samples)
        return min(count, population)


class AuditorConfig(BaseModel):
    """The auditor component's settings.

    ``backend`` mirrors the scoring worker: ``real`` composes CPU ffmpeg/libvmaf,
    PIQ PieAPP, and deterministic perceptual checks for both launch tracks.
    ``fake`` REQUIRES an injected recomputer (tests/CI) — the auditor never invents
    a fake engine on its own. ``strict`` governs how ``verify_bundle`` treats absent
    external anchors; it defaults to True now that the audit manifest carries the
    committed ``score_packet_merkle_root`` and per-item merkle inclusion proofs (v2),
    so strict merkle inclusion of every sampled item is PROVED (a packet outside the
    committed set → MERKLE_EXCLUSION dispute) alongside the recompute crux.
    """

    model_config = {"frozen": True}

    #: The auditor's on-chain identity; every AuditReport is attributed to it and
    #: (when a signer is wired) hotkey-signed under it.
    auditor_hotkey: str = ""

    #: Independent audit path represented by reports from this instance. The
    #: historical/default loop is BEACON; the standalone own-auditor selects OWN_AUDIT
    #: so both reports can coexist for one hotkey+epoch.
    audit_mode: AuditMode = AuditMode.BEACON

    sample_policy: SamplePolicy = Field(default_factory=SamplePolicy)

    #: "real" -> ffmpeg/libvmaf recompute; "fake" -> an injected recomputer only.
    backend: str = "real"

    #: strict verify_bundle (absent anchors count as failures). Default True — the
    #: v2 manifest carries the committed score-packet merkle root + per-item inclusion
    #: proofs, so strict merkle inclusion is proved for every sampled item (in addition
    #: to the recompute/identity/digest checks that catch substitution).
    strict: bool = True

    #: Require every earning challenge to carry a finalized historical chain
    #: receipt. Bare-model tests keep this off; the shipped production config and
    #: production preflight require it on.
    require_external_challenge_anchors: bool = False
    #: Subnet whose authority commitment account is independently archive-read.
    challenge_anchor_netuid: int = Field(default=85, ge=0)

    #: Needed to validate the one permitted no-miner-receipt case: the strict
    #: validator-attributed zero convention. Production wires the same scoring
    #: section as the worker.
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

    #: Wave-7 Audit Results API pointer + the NAME of the env var its bearer token
    #: is read from (never the token itself — the store's credential convention).
    results_api_url: str = ""
    results_api_token_env: str = "VIDAIO_AUDIT_RESULTS_TOKEN"

    #: Read-only Scoring Authority client bearer. Production audit workers receive
    #: this raw client env capability, never the server-side
    #: ``authority.api_token`` config key. Report/local mode may retain the legacy
    #: in-process fallback for the all-in-one demo.
    authority_api_token_env: str = "VIDAIO_AUTHORITY_READ_TOKEN"

    #: The tokenomics levers the auditor re-derives weights with; MUST match the
    #: Scoring Authority's (the project design record #5 locked levers) or an honest log
    #: would false-flag WEIGHT_DERIVATION_MISMATCH.
    tokenomics: TokenomicsConfig = Field(default_factory=TokenomicsConfig)

    #: Report/local fallback for the canonical empty-epoch recipient. A production
    #: auditor resolves the subnet owner's current uid through its independent chain
    #: adapter. If that read is unavailable, the verdict is INCONCLUSIVE (never CLEAN
    #: and never a guessed uid 0). Report overlays explicitly pin uid 0.
    burn_uid: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _sane(self) -> "AuditorConfig":
        if self.backend not in ("real", "fake"):
            raise ValueError(f"backend must be 'real' or 'fake', got {self.backend!r}")
        return self
