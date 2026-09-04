"""Independent recompute verifier — the check that kills score injection.

`verify_bundle` takes a post-retirement bundle, fetches every referenced
artifact by digest (integrity-verified by the store), re-runs scoring through
a `ScoreRecomputer`, and compares the result against the RECORDED score packet
under per-metric tolerances. The packet's authoritative fields are first-class
checks: its FULL identity (challenge_id, item_id and — when the bundle pins
one — miner_hotkey) must match the bundle, its scorer version must bind the
pre-dispatch DAG_REVEAL commitment, bundle, and recomputer (the two-signed-output
duplicate convention uses a deterministic committed-worker-derived identity),
its backend versions must exactly equal the bundle pins and the independent
recompute must report those same pins (BACKEND_VERSION_MISMATCH otherwise), its
internal
gates-first invariant must hold, and its TOP-LEVEL score — the value rankings
consume — must match the recompute, not merely the metrics block. It also checks the commit-reveal
(the revealed DAG must hash to the pre-enrollment commitment, and — when a
deep verifier is injected — must actually regenerate the DAG) and, when the
published merkle root is supplied, the inclusion of the score packet in the
published set.

A substituted or injected entry cannot pass: tampered score values fail
SCORE_MISMATCH against the recompute; a packet whose metrics are honest but
whose top-level score was edited fails SCORE_MISMATCH/PACKET_INCONSISTENT;
a packet minted for another challenge, item, or miner fails IDENTITY_MISMATCH;
a packet claiming different backend pins than the bundle anchored fails
BACKEND_VERSION_MISMATCH; a packet whose score is non-finite or outside [0, 1]
fails MALFORMED_SCORE_PACKET at parse time; tampered
artifacts fail the store's verify-on-read; packets minted outside the
committed set fail MERKLE_EXCLUSION; edited bundle metadata (timestamps,
versions, refs) fails DIGEST_MISMATCH against the anchored bundle digest.

Strictness: full third-party verification needs the external anchors (the
published bundle digest, the published merkle root, the DAG deep verifier).
When one is absent the corresponding check is recorded as an explicit SKIPPED
result; under `strict=True` (the default) a skipped anchor is a FAILURE —
pass `strict=False` only for partial/diagnostic verification.

Tolerances: byte-exact metrics (byte ratio / compression_rate) compare with
tolerance 0.0 — that is the default for any metric not listed. VMAF gets a
small epsilon because floating-point accumulation across environments can
differ in the last decimals even with pinned backends. Caller overrides may
only narrow these shipped ceilings; widening, non-finite values, and a positive
tolerance for an exact/unlisted metric are refused before verification starts.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vidaio.audit.bundle import AuditBundle, LifecycleStage
from vidaio.audit.commitments import verify_merkle_proof
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, ArtifactRef, AuditStore, IntegrityError
from vidaio.competition.item_commitment import evaluation_item_commitment

ArtifactPayload = bytes | Path

# These equal the shipped validator/scoring ingress limits. Large media stays on
# disk throughout audit recomputation; metadata is intentionally much tighter.
MAX_AUDIT_INPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_AUDIT_OUTPUT_BYTES = 4 * 1024 * 1024 * 1024
MAX_AUDIT_METADATA_BYTES = 16 * 1024 * 1024
_MEDIA_LIMITS: dict[ArtifactKind, int] = {
    ArtifactKind.CHALLENGE_INPUT: MAX_AUDIT_INPUT_BYTES,
    ArtifactKind.REFERENCE_ORIGINAL: MAX_AUDIT_INPUT_BYTES,
    ArtifactKind.MINER_OUTPUT: MAX_AUDIT_OUTPUT_BYTES,
}

#: Per-metric absolute tolerances; metrics not listed compare exactly (0.0).
#: The reserved key "score" is the tolerance for the packet's top-level score.
DEFAULT_TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        # libvmaf's final decimals can vary across otherwise-pinned CPU builds.
        "vmaf": 0.05,
        "vmaf_secondary": 0.05,
        "vmaf_model_delta_primary": 0.05,
        "vmaf_model_delta": 0.10,
        # Launch scoring and auditing both run the pinned PieAPP model on CPU. This
        # allowance is only for last-decimal variation across otherwise-pinned CPU
        # builds; it is NOT an uncalibrated CPU-vs-CUDA compatibility budget. A CUDA
        # PieAPP scorer is not release-valid until a separately versioned, measured
        # cross-device bound proves every CPU auditor reproduces its output.
        "pieapp": 1e-5,
        # Integer-derived CPU perceptual statistics are stable; retain a tiny
        # serialization allowance without permitting a gate-boundary change.
        "tone_manipulation_measure": 1e-9,
        "color_grayscale_measure": 1e-9,
        "chroma_uv_measure": 1e-9,
        "final_score": 1e-5,
        "score": 1e-5,
    }
)


def _resolve_tolerances(overrides: Mapping[str, float] | None) -> dict[str, float]:
    """Apply caller policy only when it is at least as strict as the release.

    Production and third-party callers can demand tighter equality without
    creating a local acceptance dialect.  An unlisted metric has a shipped
    ceiling of zero and therefore cannot be made approximate by an override.
    """

    resolved = dict(DEFAULT_TOLERANCES)
    for metric, raw_value in (overrides or {}).items():
        if not isinstance(metric, str) or not metric:
            raise ValueError("audit tolerance metric names must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Real):
            raise ValueError(
                f"audit tolerance override for {metric!r} must be a finite number"
            )
        value = float(raw_value)
        ceiling = float(DEFAULT_TOLERANCES.get(metric, 0.0))
        if not math.isfinite(value) or value < 0.0 or value > ceiling:
            raise ValueError(
                f"audit tolerance override for {metric!r} may only narrow the "
                f"shipped ceiling {ceiling!r}; got {value!r}"
            )
        resolved[metric] = value
    return resolved


# Failure codes (stable identifiers for dashboards/alerts/tests).
INCOMPLETE_BUNDLE = "INCOMPLETE_BUNDLE"
DIGEST_MISMATCH = "DIGEST_MISMATCH"
ARTIFACT_MISSING = "ARTIFACT_MISSING"
ARTIFACT_CORRUPT = "ARTIFACT_CORRUPT"
COMMITMENT_MISMATCH = "COMMITMENT_MISMATCH"
MALFORMED_SCORE_PACKET = "MALFORMED_SCORE_PACKET"
SCORER_VERSION_MISMATCH = "SCORER_VERSION_MISMATCH"
#: The packet, bundle, and independent recompute disagree about backend_versions
#: — recompute parity cannot be claimed under different or omitted backends.
BACKEND_VERSION_MISMATCH = "BACKEND_VERSION_MISMATCH"
MERKLE_EXCLUSION = "MERKLE_EXCLUSION"
RECOMPUTE_ERROR = "RECOMPUTE_ERROR"
METRIC_SET_MISMATCH = "METRIC_SET_MISMATCH"
SCORE_MISMATCH = "SCORE_MISMATCH"
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
PACKET_INCONSISTENT = "PACKET_INCONSISTENT"
MISSING_ANCHOR = "MISSING_ANCHOR"
REVEAL_UNVERIFIED = "REVEAL_UNVERIFIED"
REVEAL_INVALID = "REVEAL_INVALID"
COMPETITION_MANIFEST_INVALID = "COMPETITION_MANIFEST_INVALID"


@dataclass(frozen=True, slots=True)
class CompetitionAuditContext:
    """Pre-enrollment commitments required to audit a finalized competition item."""

    competition_id: str
    track: str
    manifest_digest: str
    threshold_commitment: str
    item_index: int | None = None
    input_sha256: str | None = None
    reference_sha256: str | None = None
    upscale_factor: int | None = None
    target_width: int | None = None
    target_height: int | None = None
    item_commitment: str | None = None


# Legacy authority-attributed zero packets are identified only so the verifier
# can reject them explicitly.  Timeouts and other authority-observed failures
# have no independently reproducible economic evidence at launch.
_VALIDATOR_ZERO_SCORER_NAME = "validator-zero/1"
_VALIDATOR_ZERO_PREFIX = f"{_VALIDATOR_ZERO_SCORER_NAME}+"
_ORCHESTRATOR_ZERO_SCORER_NAME = "orchestrator-zero/1"
_ORCHESTRATOR_ZERO_PREFIX = f"{_ORCHESTRATOR_ZERO_SCORER_NAME}+"
_DUPLICATE_SCORER_NAME = "validator-exact-duplicate/1"
_DUPLICATE_SCORER_PREFIX = f"{_DUPLICATE_SCORER_NAME}+"
_DUPLICATE_WITNESS_METRIC = "duplicate_witness"
_DUPLICATE_SELECTION_RULE = "anchor_hash_hotkey/1"
_DUPLICATE_EVIDENCE_RULE = "sha256_exact_output/1"
_DUPLICATE_ORDER_DOMAIN = b"vidaio:duplicate-order:anchor-hash-hotkey:v1\x00"


def _orchestrator_zero_identity(
    *,
    committed_scorer_version: str,
    track: str,
    scoring_config_digest: str,
) -> str:
    """Derive the reserved zero-record identity without importing the service.

    The producer lives below the orchestrator package, which imports this audit
    layer. Keeping the tiny domain-separated derivation here avoids an import
    cycle. The real recomputer separately requires the supplied config digest to
    equal its own locked CPU scoring configuration.
    """
    if (
        len(scoring_config_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in scoring_config_digest)
    ):
        raise ValueError(
            "orchestrator-zero scoring_config_digest is not lowercase sha256 hex"
        )
    digest = sha256_hex(
        canonical_json_bytes(
            {
                "committed_scoring_version": committed_scorer_version,
                "convention": _ORCHESTRATOR_ZERO_SCORER_NAME,
                "scoring_config_digest": scoring_config_digest,
                "track": track,
            }
        )
    )
    return f"{_ORCHESTRATOR_ZERO_PREFIX}{digest[:12]}"


class ScorePacketShape(BaseModel):
    """Audit-side contract for the scoring module's ItemScore JSON.

    Defined locally — audit never imports scoring — as the set of REQUIRED
    authoritative fields, mirroring ItemScore's full field set; unknown extra
    fields are ignored (loose on extras) so scoring can grow the packet without
    breaking old verifiers. A packet missing any of these keys is MALFORMED:
    it cannot be audited. Several keys are required-but-nullable: the KEY must
    be present, the value may be null (content_digest, pieapp_start_frame —
    null for compression items — and breakdown, null only when the gate
    failed). `skips` (auditable gate skips, e.g. a disabled secondary metric)
    defaults to empty so older packets parse, but is carried so packets with
    skips are visible to auditors.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    item_id: str
    challenge_id: str
    track: str
    #: The authoritative value rankings consume — gates-first: 0.0 whenever
    #: gate_passed is False. Must be finite and within [0, 1]: an Infinity/NaN
    #: or out-of-range score is MALFORMED at parse time, before any recompute.
    score: float
    gate_passed: bool
    violations: list[Any]
    #: Auditable gate skips (e.g. secondary VMAF disabled by config). Default
    #: empty; a packet carrying skips surfaces them to auditors unchanged.
    skips: list[Any] = Field(default_factory=list)
    miner_hotkey: str | None
    content_digest: str | None
    breakdown: dict[str, Any] | None
    metrics: dict[str, Any]
    scorer_version: str
    backend_versions: dict[str, str]
    pieapp_start_frame: int | None
    scoring_config_digest: str | None
    canonicalization_plan_digest: str | None

    @field_validator("score")
    @classmethod
    def _score_finite_unit_interval(cls, v: float) -> float:
        if not math.isfinite(v) or not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be finite and within [0, 1], got {v!r}")
        return v

    def numeric_metrics(self) -> dict[str, float]:
        """The recomputable subset: metric entries with real numeric values."""
        return {
            k: float(v)
            for k, v in self.metrics.items()
            if isinstance(v, numbers.Real) and not isinstance(v, bool)
        }


class DuplicateWitnessShape(BaseModel):
    """Audit-local decoder for the canonical two-output duplicate witness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    evidence_rule: Literal["sha256_exact_output/1"]
    selection_rule: Literal["anchor_hash_hotkey/1"]
    committed_scorer_version: str
    track: str
    loser_uid: int = Field(ge=0)
    loser_hotkey: str
    loser_output_digest: str
    loser_output_size: int = Field(ge=1)
    loser_receipt_digest: str
    winner_uid: int = Field(ge=0)
    winner_hotkey: str
    winner_output: ArtifactRef
    winner_receipt: dict[str, Any]


class RecomputedScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    scorer_version: str
    #: Versions detected by the independent scoring composition.  The verifier
    #: compares this complete map to the packet-bound bundle pins.
    backend_versions: dict[str, str]
    #: Recomputed top-level outcome, gates-first: score MUST be 0.0 when
    #: gate_passed is False. Compared against the packet's authoritative fields.
    score: float
    gate_passed: bool
    #: Fresh gate evidence and formula breakdown. These are not trusted packet
    #: data: a real recomputer obtains them from its independent scoring run.
    #: They let the verifier recognize a decision that crossed a numeric
    #: boundary by no more than the already-declared metric tolerance, without
    #: weakening any metric comparison or accepting non-numeric gate flips.
    violations: list[dict[str, Any]] = Field(default_factory=list)
    breakdown: dict[str, Any] | None = None


class ScoreRecomputer(Protocol):
    """Contract the scoring engine implements for independent recompute.

    `artifacts` maps metadata to integrity-verified bytes and media to verified
    local paths. Keeping multi-gigabyte video on disk prevents a valid padded
    object from exhausting an auditor's RAM. The implementation must be
    deterministic given identical artifacts + pinned backend versions.
    """

    def recompute(
        self, bundle: AuditBundle, artifacts: Mapping[ArtifactKind, ArtifactPayload]
    ) -> RecomputedScore: ...


class StaticRecomputer:
    """Test double: returns fixed metrics regardless of the artifacts.

    Used by this module's tests and available to other modules that need a
    stand-in until the real scoring engine implements ScoreRecomputer.
    The top-level score defaults to the "final_score" metric (gates-first:
    forced to 0.0 when gate_passed is False).
    """

    def __init__(
        self,
        metrics: dict[str, float],
        scorer_version: str,
        *,
        score: float | None = None,
        gate_passed: bool = True,
        backend_versions: Mapping[str, str] | None = None,
        violations: Sequence[Mapping[str, Any]] = (),
        breakdown: Mapping[str, Any] | None = None,
    ) -> None:
        self._metrics = dict(metrics)
        self._scorer_version = scorer_version
        if not gate_passed:
            self._score = 0.0
        elif score is not None:
            self._score = score
        else:
            self._score = float(self._metrics.get("final_score", 0.0))
        self._gate_passed = gate_passed
        self._backend_versions = (
            None if backend_versions is None else dict(backend_versions)
        )
        self._violations = [dict(v) for v in violations]
        self._breakdown = dict(breakdown) if breakdown is not None else None

    def recompute(
        self, bundle: AuditBundle, artifacts: Mapping[ArtifactKind, ArtifactPayload]
    ) -> RecomputedScore:
        return RecomputedScore(
            metrics=dict(self._metrics),
            scorer_version=self._scorer_version,
            # A test double with no explicit override models a recomputer built
            # from the bundle's pinned environment.  Mismatch tests pass an
            # explicit map; production recomputers always report detected values.
            backend_versions=(
                dict(bundle.backend_versions)
                if self._backend_versions is None
                else dict(self._backend_versions)
            ),
            score=self._score,
            gate_passed=self._gate_passed,
            violations=list(self._violations),
            breakdown=self._breakdown,
        )


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    code: str | None = None  # failure code; None when passed
    #: True when the check could not run (absent anchor/verifier). Skips pass
    #: only under strict=False; under strict=True they are failures with a code.
    skipped: bool = False
    reason: str = ""


class VerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenge_id: str
    bundle_digest: str
    strict: bool = True
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def skips(self) -> list[CheckResult]:
        return [c for c in self.checks if c.skipped]


def _ok(name: str, reason: str = "") -> CheckResult:
    return CheckResult(name=name, passed=True, reason=reason)


def _fail(name: str, code: str, reason: str) -> CheckResult:
    return CheckResult(name=name, passed=False, code=code, reason=reason)


def _skip(name: str, code: str, reason: str, *, strict: bool) -> CheckResult:
    """Explicit SKIPPED result: a failure under strict, a marked pass otherwise."""
    if strict:
        return CheckResult(
            name=name,
            passed=False,
            code=code,
            skipped=True,
            reason=f"skipped: {reason} (strict mode treats skipped checks as failures)",
        )
    return CheckResult(
        name=name, passed=True, skipped=True, reason=f"skipped: {reason}"
    )


def _parse_score_packet(data: bytes) -> tuple[ScorePacketShape | None, str]:
    """Returns (packet, error). packet is None on failure."""
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"score packet is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "score packet must be a JSON object"
    try:
        return ScorePacketShape.model_validate(payload), ""
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        return None, f"score packet does not match the ItemScore contract: {problems}"


def _is_validator_zero_identity(identity: str) -> bool:
    return identity == _VALIDATOR_ZERO_SCORER_NAME or identity.startswith(
        _VALIDATOR_ZERO_PREFIX
    )


def _parse_duplicate_witness(
    packet: ScorePacketShape,
) -> tuple[DuplicateWitnessShape | None, str]:
    raw = packet.metrics.get(_DUPLICATE_WITNESS_METRIC)
    if not isinstance(raw, str) or not raw:
        return None, "duplicate_witness metric is missing or not text"
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return None, f"duplicate witness is not JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "duplicate witness must be a JSON object"
    if canonical_json_bytes(payload).decode("utf-8") != raw:
        return None, "duplicate witness is not canonical JSON"
    try:
        witness = DuplicateWitnessShape.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        return None, f"duplicate witness is malformed: {problems}"
    if witness.winner_output.kind is not ArtifactKind.MINER_OUTPUT:
        return None, "duplicate winner output is not a miner_output artifact"
    if witness.winner_hotkey == witness.loser_hotkey:
        return None, "duplicate winner and loser identities are not distinct"
    return witness, ""


def _duplicate_order_key(block_hash: str, hotkey: str) -> str:
    if (
        not isinstance(block_hash, str)
        or len(block_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in block_hash)
    ):
        raise ValueError("anchor block hash is not lowercase sha256 hex")
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or len(hotkey) > 128
        or any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in hotkey)
    ):
        raise ValueError("miner hotkey is not canonical printable ASCII")
    rank = sha256_hex(
        _DUPLICATE_ORDER_DOMAIN
        + bytes.fromhex(block_hash)
        + b"\x00"
        + hotkey.encode("ascii")
    )
    return f"{rank}\x00{hotkey}"


def _parse_committed_scorer(data: bytes) -> tuple[tuple[str, str] | None, str]:
    """Read ``(scorer_version, track)`` from a canonical DAG_REVEAL preimage.

    The bytes themselves are the challenge-commitment preimage. Checking the
    content hash alone only proves that *some* bytes were committed; this parser
    exposes the scorer identity those bytes fixed before dispatch. Canonicality
    is mandatory so alternate JSON encodings cannot acquire a second meaning in
    another implementation.
    """
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        return None, f"DAG_REVEAL commitment preimage is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "DAG_REVEAL commitment preimage must be a JSON object"
    try:
        # Match challenge.dag.canonical_json_dumps byte-for-byte, including its
        # escaped non-ASCII representation. The commitment hash was produced
        # over that serializer, while audit.canonical intentionally uses a
        # different ensure_ascii policy for audit-domain objects.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        return None, f"DAG_REVEAL commitment preimage cannot be canonicalized: {exc}"
    if canonical != data:
        return None, "DAG_REVEAL commitment preimage is not canonical JSON"
    scorer_version = payload.get("scorer_version")
    track = payload.get("track")
    if not isinstance(scorer_version, str) or not scorer_version.strip():
        return None, "DAG_REVEAL commitment preimage has no committed scorer_version"
    if not isinstance(track, str) or not track.strip():
        return None, "DAG_REVEAL commitment preimage has no committed track"
    return (scorer_version, track), ""


def _expected_packet_scorer(
    packet: ScorePacketShape, *, committed_scorer: str, committed_track: str
) -> tuple[str | None, str]:
    """Translate the committed worker identity into the packet identity.

    Measured packets must name the committed worker verbatim. Reserved duplicate
    and orchestrator-zero records derive their identities from the committed
    worker, track, and locked scoring config. Legacy validator-zero packets are
    refused because authority-observed failures are not reproducible proof.
    """
    if packet.scorer_version.startswith(_DUPLICATE_SCORER_PREFIX):
        witness, error = _parse_duplicate_witness(packet)
        if witness is None:
            return None, error
        if packet.track != committed_track or witness.track != committed_track:
            return None, "duplicate packet/witness track differs from committed track"
        if witness.committed_scorer_version != committed_scorer:
            return None, "duplicate witness does not name the committed scorer identity"
        config_digest = packet.scoring_config_digest
        if not isinstance(config_digest, str) or not config_digest:
            return None, "duplicate packet has no scoring_config_digest"
        digest = sha256_hex(
            canonical_json_bytes(
                {
                    "committed_scorer_version": committed_scorer,
                    "convention": _DUPLICATE_SCORER_NAME,
                    "evidence_rule": _DUPLICATE_EVIDENCE_RULE,
                    "scoring_config_digest": config_digest,
                    "selection_rule": _DUPLICATE_SELECTION_RULE,
                    "track": committed_track,
                }
            )
        )
        return f"{_DUPLICATE_SCORER_PREFIX}{digest[:12]}", ""
    if packet.scorer_version.startswith(_ORCHESTRATOR_ZERO_PREFIX):
        if packet.track != committed_track:
            return None, "orchestrator-zero packet track differs from committed track"
        if not isinstance(packet.scoring_config_digest, str):
            return None, "orchestrator-zero packet has no scoring_config_digest"
        try:
            expected = _orchestrator_zero_identity(
                committed_scorer_version=committed_scorer,
                track=committed_track,
                scoring_config_digest=packet.scoring_config_digest,
            )
        except ValueError as exc:
            return None, str(exc)
        return expected, ""
    if _is_validator_zero_identity(packet.scorer_version):
        return None, (
            "validator-zero packets are not launch-valid economic evidence; "
            "authority-observed failures must be non-punitive skips"
        )
    return committed_scorer, ""


# Only these gates have a numeric decision surface that is also present in the
# score packet's recomputable metric set. Structural, identity, stream-shape,
# missing-metric, and duplicate failures can never receive hysteresis.
_BOUNDARY_METRICS: dict[str, tuple[str, Literal["below", "above"]]] = {
    "VMAF_BELOW_FLOOR": ("vmaf", "below"),
    "VMAF_BELOW_THRESHOLD": ("vmaf", "below"),
    "VMAF_MODEL_DELTA_EXCEEDED": ("vmaf_model_delta", "above"),
    "COMPRESSION_RATE_TOO_HIGH": ("compression_rate", "above"),
    "TONE_MANIPULATION": ("tone_manipulation_measure", "above"),
    "COLOR_GRAYSCALE": ("color_grayscale_measure", "below"),
    "CHROMA_UV_MANIPULATION": ("chroma_uv_measure", "above"),
}


def _violation_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json")
        return payload if isinstance(payload, Mapping) else None
    return None


def _formula_boundary(breakdown: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the compression threshold's numeric decision fact, if present.

    ``VMAF_BELOW_THRESHOLD`` is a formula eligibility boundary rather than a
    GatePipeline violation: both sides say ``gate_passed=True`` but one final is
    zero. Treating it here is what closes the VMAF-90 audit-griefing seam.
    """
    if not isinstance(breakdown, Mapping):
        return None
    if breakdown.get("kind") != "compression":
        return None
    if breakdown.get("zero_reason") != "VMAF_BELOW_THRESHOLD":
        return None
    return {
        "code": "VMAF_BELOW_THRESHOLD",
        "measured": breakdown.get("vmaf"),
        "limit": breakdown.get("vmaf_threshold"),
    }


def _boundary_fact_is_tolerated(
    fact: Mapping[str, Any],
    *,
    failing_metrics: Mapping[str, float],
    passing_metrics: Mapping[str, float],
    tolerances: Mapping[str, float],
) -> bool:
    code = str(fact.get("code", ""))
    mapped = _BOUNDARY_METRICS.get(code)
    if mapped is None:
        return False
    metric, direction = mapped
    measured = fact.get("measured")
    limit = fact.get("limit")
    if (
        not isinstance(measured, numbers.Real)
        or isinstance(measured, bool)
        or not isinstance(limit, numbers.Real)
        or isinstance(limit, bool)
    ):
        return False
    measured = float(measured)
    limit = float(limit)
    failing = failing_metrics.get(metric)
    passing = passing_metrics.get(metric)
    allowed = float(tolerances.get(metric, 0.0))
    if (
        failing is None
        or passing is None
        or not all(
            math.isfinite(v) for v in (measured, limit, failing, passing, allowed)
        )
        or allowed < 0.0
        or abs(failing - measured) > allowed
        or abs(failing - passing) > allowed
        or abs(measured - limit) > allowed
    ):
        return False
    if direction == "below":
        return failing < limit <= passing
    return passing <= limit < failing


def _boundary_hysteresis_reason(
    packet: ScorePacketShape,
    recomputed: RecomputedScore,
    recorded_metrics: Mapping[str, float],
    fresh_metrics: Mapping[str, float],
    tolerances: Mapping[str, float],
) -> tuple[str | None, frozenset[str]]:
    """Recognize only an epsilon-sized numeric decision-surface crossing.

    The raw metric sets must agree and every non-derived value must already be
    inside its declared tolerance. A gate flip additionally requires that the
    *only* failing violations be whitelisted numeric boundaries carrying finite
    measured+limit values. The compression threshold discontinuity is handled
    through the independently recomputed formula breakdown. Anything structural
    or beyond the tolerance band returns ``None`` and remains a hard mismatch.
    """
    if set(recorded_metrics) != set(fresh_metrics):
        return None, frozenset()

    formula_flip = False
    facts: list[Mapping[str, Any]] = []
    if packet.gate_passed != recomputed.gate_passed:
        raw_facts = (
            packet.violations if not packet.gate_passed else recomputed.violations
        )
        for raw in raw_facts:
            fact = _violation_mapping(raw)
            if fact is None:
                return None, frozenset()
            facts.append(fact)
        if not facts:
            return None, frozenset()
    else:
        recorded_formula = _formula_boundary(packet.breakdown)
        fresh_formula = _formula_boundary(recomputed.breakdown)
        if (recorded_formula is None) == (fresh_formula is None):
            return None, frozenset()
        formula_flip = True
        facts = [recorded_formula or fresh_formula]  # type: ignore[list-item]

    excused_metrics = frozenset({"final_score"} if formula_flip else set())
    for metric in recorded_metrics:
        if metric in excused_metrics:
            continue
        allowed = float(tolerances.get(metric, 0.0))
        if (
            not math.isfinite(allowed)
            or allowed < 0.0
            or abs(recorded_metrics[metric] - fresh_metrics[metric]) > allowed
        ):
            return None, frozenset()

    recorded_fails = not packet.gate_passed or (
        formula_flip and _formula_boundary(packet.breakdown) is not None
    )
    failing_metrics = recorded_metrics if recorded_fails else fresh_metrics
    passing_metrics = fresh_metrics if recorded_fails else recorded_metrics
    if not all(
        _boundary_fact_is_tolerated(
            fact,
            failing_metrics=failing_metrics,
            passing_metrics=passing_metrics,
            tolerances=tolerances,
        )
        for fact in facts
    ):
        return None, frozenset()

    codes = ",".join(sorted(str(fact.get("code")) for fact in facts))
    return (
        "accepted committed outcome: independent metrics crossed numeric boundary "
        f"{codes} only within declared audit tolerance",
        excused_metrics,
    )


def _scores_are_internally_honest(
    packet: ScorePacketShape,
    recomputed: RecomputedScore,
    *,
    tolerance: float,
) -> bool:
    """Both sides' top-level score must match their own gates/formula result."""
    recorded_final = packet.numeric_metrics().get("final_score", 0.0)
    fresh_final = recomputed.metrics.get("final_score", 0.0)
    expected_recorded = recorded_final if packet.gate_passed else 0.0
    expected_fresh = fresh_final if recomputed.gate_passed else 0.0
    return (
        abs(packet.score - expected_recorded) <= tolerance
        and abs(recomputed.score - expected_fresh) <= tolerance
    )


def verify_bundle(
    bundle: AuditBundle,
    store: AuditStore,
    recomputer: ScoreRecomputer,
    *,
    tolerances: Mapping[str, float] | None = None,
    expected_bundle_digest: str | None = None,
    expected_miner_hotkey: str | None = None,
    require_expected_miner: bool = False,
    published_root: str | None = None,
    inclusion_proof: Sequence[tuple[str, str]] | None = None,
    reveal_verifier: Callable[[bytes], bool] | None = None,
    strict: bool = True,
    competition_context: CompetitionAuditContext | None = None,
) -> VerificationReport:
    """Run every audit check on a bundle; the report lists each with pass/fail.

    - expected_bundle_digest: the digest that was published/anchored for this
      item; catches any post-hoc edit of bundle metadata.
    - expected_miner_hotkey: the identity the manifest attributes this item to
      (the uid's hotkey). When supplied, the score packet's miner_hotkey MUST
      equal it — the miner comparison is NOT skipped merely because the bundle
      pins no miner, so a packet minted for another miner
      cannot pass recompute by leaving the bundle miner null. When both this and
      the bundle's pinned miner are present they must also agree. When None, the
      check falls back to the bundle's pinned miner (baseline rows).
    - require_expected_miner: when True, a null/empty expected_miner_hotkey is
      itself an IDENTITY_MISMATCH fault, not a fall-back to the bundle's pinned
      miner. The auditor sets this for a SAMPLED item attributed
      to a nonzero-weight uid: a null expected identity means the packet's miner can
      never be bound to the uid, so it must fail closed rather than silently skip the
      miner comparison (the null-hotkey bypass). Left False for baseline rows
      (uid None) which legitimately carry no expected identity.
    - published_root / inclusion_proof: the post-evaluation publication's
      score-packet merkle root and this packet's proof.
    - reveal_verifier: deep commit-reveal check injected by the caller (the
      challenge module exposes one that rebuilds the DAG from the revealed
      seed); receives the DAG_REVEAL artifact bytes, returns True iff they
      genuinely regenerate the committed challenge.
      Independently of that injected deep check, the reveal must be canonical
      JSON and its pre-dispatch scorer identity must bind the packet, bundle,
      and recomputer (through the reserved two-output duplicate identity where
      applicable). Legacy validator-zero packets are never economic evidence.
    - strict: absent anchors/verifiers record SKIPPED results that COUNT AS
      FAILURES (default). strict=False lets skips pass for partial audits.
    """
    tol = _resolve_tolerances(tolerances)
    checks: list[CheckResult] = []

    # 1. Inference requires post-retirement reveal. A completed competition's sealed
    # input is itself the reference and its scorer/threshold policy is precommitted by
    # the competition manifest, so its original PRE_REVEAL bundle remains sufficient.
    competition_stage_ok = competition_context is not None and (
        (
            competition_context.track == "compression"
            and bundle.stage is LifecycleStage.PRE_REVEAL
        )
        or (
            competition_context.track == "upscaling"
            and bundle.stage is LifecycleStage.COMPETITION_SEALED
        )
    )
    if bundle.stage is LifecycleStage.POST_RETIREMENT or competition_stage_ok:
        checks.append(_ok("stage_recomputable"))
    else:
        checks.append(
            _fail(
                "stage_recomputable",
                INCOMPLETE_BUNDLE,
                f"stage is {bundle.stage.value}; full verification requires post_retirement",
            )
        )

    # 2. Bundle digest vs the anchored/published one.
    actual_digest = bundle.bundle_digest()
    if expected_bundle_digest is not None:
        if actual_digest == expected_bundle_digest:
            checks.append(_ok("bundle_digest"))
        else:
            checks.append(
                _fail(
                    "bundle_digest",
                    DIGEST_MISMATCH,
                    f"bundle digest {actual_digest} != published {expected_bundle_digest}"
                    " — bundle metadata was altered after publication",
                )
            )
    else:
        checks.append(
            _skip(
                "bundle_digest",
                MISSING_ANCHOR,
                "no published bundle digest supplied",
                strict=strict,
            )
        )

    # 3. Fetch every referenced artifact (store verifies content addresses).
    refs: list[ArtifactRef] = [
        r
        for r in (
            bundle.challenge_input,
            bundle.miner_output,
            bundle.manifest,
            bundle.score_packet,
            bundle.reference_original,
            bundle.dag_reveal,
        )
        if r is not None
    ]
    artifacts: dict[ArtifactKind, ArtifactPayload] = {}
    media_tmp = TemporaryDirectory(prefix="vidaio-audit-media-")
    for ref in refs:
        name = f"artifact_integrity:{ref.kind.value}"
        try:
            limit = _MEDIA_LIMITS.get(ref.kind)
            if limit is None:
                artifacts[ref.kind] = store.get_limited(ref, MAX_AUDIT_METADATA_BYTES)
            else:
                artifacts[ref.kind] = store.materialize(
                    ref, media_tmp.name, max_bytes=limit
                )
            checks.append(_ok(name))
        except IntegrityError as exc:
            checks.append(_fail(name, ARTIFACT_CORRUPT, str(exc)))
        except (FileNotFoundError, OSError) as exc:
            checks.append(_fail(name, ARTIFACT_MISSING, f"artifact unavailable: {exc}"))

    competition_manifest: Mapping[str, Any] | None = None
    if competition_context is not None:
        raw_manifest = artifacts.get(ArtifactKind.MANIFEST)
        manifest_problems: list[str] = []
        if raw_manifest is None or isinstance(raw_manifest, Path):
            manifest_problems.append("competition manifest artifact is unavailable")
        else:
            if sha256_hex(raw_manifest) != competition_context.manifest_digest:
                manifest_problems.append(
                    "competition manifest bytes do not match the committed manifest digest"
                )
            try:
                parsed_manifest = json.loads(raw_manifest)
            except (TypeError, ValueError) as exc:
                manifest_problems.append(f"competition manifest is not JSON: {exc}")
            else:
                if not isinstance(parsed_manifest, dict):
                    manifest_problems.append(
                        "competition manifest is not a JSON object"
                    )
                else:
                    canonical = json.dumps(
                        parsed_manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                    if canonical != raw_manifest:
                        manifest_problems.append(
                            "competition manifest bytes are not canonical JSON"
                        )
                    if (
                        parsed_manifest.get("competition_id")
                        != competition_context.competition_id
                    ):
                        manifest_problems.append(
                            "competition_id differs from committed input"
                        )
                    if parsed_manifest.get("track") != competition_context.track:
                        manifest_problems.append(
                            "competition track differs from committed input"
                        )
                    scorer = parsed_manifest.get("scoring_version")
                    if not isinstance(scorer, str) or not scorer:
                        manifest_problems.append(
                            "competition manifest has no scoring_version"
                        )
                    if competition_context.track == "compression":
                        threshold = parsed_manifest.get("vmaf_threshold")
                        if (
                            not isinstance(threshold, numbers.Real)
                            or isinstance(threshold, bool)
                            or not math.isfinite(float(threshold))
                            or not 0.0 <= float(threshold) <= 100.0
                        ):
                            manifest_problems.append(
                                "competition manifest has no valid VMAF threshold"
                            )
                    competition_manifest = parsed_manifest
        if bundle.commitment_hash != competition_context.threshold_commitment:
            manifest_problems.append(
                "bundle threshold commitment differs from the committed evaluation item"
            )
        if ArtifactKind.CHALLENGE_INPUT not in artifacts:
            manifest_problems.append("competition challenge input is unavailable")
        if competition_context.track == "upscaling":
            binding = bundle.competition_item
            context_binding = (
                competition_context.item_index,
                competition_context.input_sha256,
                competition_context.reference_sha256,
                competition_context.upscale_factor,
                competition_context.target_width,
                competition_context.target_height,
                competition_context.item_commitment,
            )
            if binding is None or any(value is None for value in context_binding):
                manifest_problems.append(
                    "upscaling competition item commitment preimage is incomplete"
                )
            else:
                bundle_binding = (
                    binding.item_index,
                    binding.input_sha256,
                    binding.reference_sha256,
                    binding.upscale_factor,
                    binding.target_width,
                    binding.target_height,
                    binding.item_commitment,
                )
                if bundle_binding != context_binding:
                    manifest_problems.append(
                        "bundle upscaling item preimage differs from epoch evidence"
                    )
                if (
                    bundle.challenge_input.digest != binding.input_sha256
                    or bundle.reference_original is None
                    or bundle.reference_original.digest != binding.reference_sha256
                ):
                    manifest_problems.append(
                        "bundle media refs differ from the upscaling item preimage"
                    )
                try:
                    derived_item_commitment = evaluation_item_commitment(
                        competition_id=competition_context.competition_id,
                        item_index=binding.item_index,
                        reference_sha256=binding.reference_sha256,
                        input_sha256=binding.input_sha256,
                        upscale_factor=binding.upscale_factor,
                        target_width=binding.target_width,
                        target_height=binding.target_height,
                    )
                except ValueError as exc:
                    manifest_problems.append(
                        f"upscaling item preimage is invalid: {exc}"
                    )
                else:
                    if derived_item_commitment != binding.item_commitment:
                        manifest_problems.append(
                            "upscaling item commitment does not hash its preimage"
                        )
                    if competition_manifest is not None:
                        factors = competition_manifest.get(
                            "allowed_upscale_factors"
                        )
                        commitments = competition_manifest.get(
                            "evaluation_item_commitments"
                        )
                        if (
                            not isinstance(factors, list)
                            or binding.upscale_factor not in factors
                        ):
                            manifest_problems.append(
                                "upscaling factor is not allowed by the manifest"
                            )
                        if (
                            not isinstance(commitments, list)
                            or binding.item_index >= len(commitments)
                            or commitments[binding.item_index]
                            != derived_item_commitment
                        ):
                            manifest_problems.append(
                                "upscaling item is not at its committed manifest index"
                            )
            if ArtifactKind.REFERENCE_ORIGINAL not in artifacts:
                manifest_problems.append(
                    "upscaling pristine reference is unavailable or unreleased"
                )
        if manifest_problems:
            checks.append(
                _fail(
                    "competition_manifest_commitment",
                    COMPETITION_MANIFEST_INVALID,
                    "; ".join(manifest_problems),
                )
            )
        else:
            checks.append(_ok("competition_manifest_commitment"))
            if competition_context.track == "compression":
                # Compression uses the same clean bytes as reference and miner input.
                artifacts[ArtifactKind.REFERENCE_ORIGINAL] = artifacts[
                    ArtifactKind.CHALLENGE_INPUT
                ]

    # 4. Commit-reveal: the revealed DAG must hash to the committed hash, and —
    #    when a deep verifier is injected — must actually regenerate the DAG.
    dag = artifacts.get(ArtifactKind.DAG_REVEAL)
    if isinstance(dag, Path):  # defensive; DAG_REVEAL is metadata above
        dag = None
    if competition_context is not None:
        checks.append(
            _ok(
                "commitment_reveal",
                "competition score policy is bound by the pre-enrollment manifest",
            )
        )
    elif dag is not None:
        revealed = sha256_hex(dag)
        if revealed == bundle.commitment_hash:
            checks.append(_ok("commitment_reveal"))
        else:
            checks.append(
                _fail(
                    "commitment_reveal",
                    COMMITMENT_MISMATCH,
                    f"revealed DAG hashes to {revealed}, committed {bundle.commitment_hash}",
                )
            )
    else:
        checks.append(
            _fail(
                "commitment_reveal", ARTIFACT_MISSING, "dag_reveal artifact unavailable"
            )
        )
    if competition_context is not None:
        checks.append(
            _ok(
                "dag_reveal_generation",
                "competition inputs have no degradation DAG; manifest provenance applies",
            )
        )
    elif reveal_verifier is None:
        checks.append(
            _skip(
                "dag_reveal_generation",
                REVEAL_UNVERIFIED,
                "no reveal verifier supplied — DAG regeneration not proven",
                strict=strict,
            )
        )
    elif dag is None:
        checks.append(
            _fail(
                "dag_reveal_generation",
                ARTIFACT_MISSING,
                "dag_reveal artifact unavailable",
            )
        )
    else:
        try:
            reveal_ok = bool(reveal_verifier(dag))
        except Exception as exc:  # a verifier crash is itself a finding
            checks.append(
                _fail(
                    "dag_reveal_generation",
                    REVEAL_INVALID,
                    f"reveal verifier raised: {exc}",
                )
            )
        else:
            if reveal_ok:
                checks.append(_ok("dag_reveal_generation"))
            else:
                checks.append(
                    _fail(
                        "dag_reveal_generation",
                        REVEAL_INVALID,
                        "revealed bytes do not regenerate the committed challenge DAG",
                    )
                )

    # 5. Parse the recorded score packet and check its authoritative fields.
    packet: ScorePacketShape | None = None
    duplicate_witness: DuplicateWitnessShape | None = None
    duplicate_winner_output: ArtifactPayload | None = None
    packet_bytes = artifacts.get(ArtifactKind.SCORE_PACKET)
    if isinstance(packet_bytes, Path):  # defensive; SCORE_PACKET is metadata above
        packet_bytes = None
    if packet_bytes is not None:
        packet, err = _parse_score_packet(packet_bytes)
        if packet is None:
            checks.append(_fail("score_packet_parse", MALFORMED_SCORE_PACKET, err))
        else:
            checks.append(_ok("score_packet_parse"))
            if packet.scorer_version.startswith(_DUPLICATE_SCORER_PREFIX):
                duplicate_witness, witness_error = _parse_duplicate_witness(packet)
                duplicate_problems: list[str] = []
                if duplicate_witness is None:
                    duplicate_problems.append(witness_error)
                else:
                    witness = duplicate_witness
                    violation = (
                        packet.violations[0] if len(packet.violations) == 1 else None
                    )
                    violation_code = (
                        violation.get("code") if isinstance(violation, dict) else None
                    )
                    expected_detail = (
                        "auditable byte-exact duplicate of deterministic "
                        f"anchor-salted winner {witness.winner_hotkey}"
                    )
                    violation_detail = (
                        violation.get("detail") if isinstance(violation, dict) else None
                    )
                    if set(packet.metrics) != {_DUPLICATE_WITNESS_METRIC}:
                        duplicate_problems.append(
                            "duplicate packet contains metrics outside its canonical witness"
                        )
                    if packet.score != 0.0 or packet.gate_passed:
                        duplicate_problems.append(
                            "duplicate packet must be gate-failed with score zero"
                        )
                    if packet.breakdown is not None or packet.skips:
                        duplicate_problems.append(
                            "duplicate packet must not claim a measured breakdown or gate skips"
                        )
                    if (
                        packet.canonicalization_plan_digest is not None
                        or packet.pieapp_start_frame is not None
                    ):
                        duplicate_problems.append(
                            "duplicate packet must not claim scoring canonicalization/PieAPP work"
                        )
                    if (
                        violation_code != "REPLAY_DUPLICATE"
                        or violation_detail != expected_detail
                    ):
                        duplicate_problems.append(
                            "duplicate packet violation is absent, mislabeled or noncanonical"
                        )
                    if packet.backend_versions:
                        duplicate_problems.append(
                            "exact duplicate packet must not claim measurement backends"
                        )
                    base_loser_item_id = f"{packet.challenge_id}:{witness.loser_uid}"
                    allowed_loser_item_ids = {base_loser_item_id}
                    committed_ordering_key: int | None = None
                    if bundle.challenge_anchor is not None:
                        committed_ordering_key = (
                            bundle.challenge_anchor.dispatch_ordering_key
                        )
                    elif dag is not None:
                        try:
                            raw_ordering_key = json.loads(dag).get(
                                "dispatch_ordering_key"
                            )
                            if isinstance(raw_ordering_key, int):
                                committed_ordering_key = raw_ordering_key
                        except (AttributeError, TypeError, ValueError):
                            pass
                    if committed_ordering_key is not None:
                        allowed_loser_item_ids.add(
                            f"{base_loser_item_id}-c{committed_ordering_key}"
                        )
                    if (
                        packet.item_id not in allowed_loser_item_ids
                        or packet.miner_hotkey != witness.loser_hotkey
                        or packet.track != witness.track
                        or packet.content_digest != witness.loser_output_digest
                        or bundle.miner_output.digest != witness.loser_output_digest
                        or bundle.miner_output.byte_size != witness.loser_output_size
                    ):
                        duplicate_problems.append(
                            "duplicate loser identity/media is not bound to packet and bundle"
                        )
                    if (
                        witness.loser_output_digest != witness.winner_output.digest
                        or witness.loser_output_size != witness.winner_output.byte_size
                    ):
                        duplicate_problems.append(
                            "economic duplicate witness is not byte-exact"
                        )
                    winner_receipt = witness.winner_receipt
                    if (
                        winner_receipt.get("miner_hotkey") != witness.winner_hotkey
                        or winner_receipt.get("output_digest")
                        != witness.winner_output.digest
                        or winner_receipt.get("output_size")
                        != witness.winner_output.byte_size
                    ):
                        duplicate_problems.append(
                            "duplicate winner output reference and receipt do not agree"
                        )
                    if bundle.miner_receipt is None:
                        duplicate_problems.append(
                            "duplicate loser has no miner-signed output receipt"
                        )
                    else:
                        loser_anchor = bundle.miner_receipt.metadata.commitment_anchor
                        winner_metadata = winner_receipt.get("metadata")
                        winner_anchor = (
                            winner_metadata.get("commitment_anchor")
                            if isinstance(winner_metadata, dict)
                            else None
                        )
                        loser_receipt_digest = sha256_hex(
                            canonical_json_bytes(
                                bundle.miner_receipt.model_dump(mode="json")
                            )
                        )
                        if loser_receipt_digest != witness.loser_receipt_digest:
                            duplicate_problems.append(
                                "duplicate witness does not bind the bundle's loser receipt"
                            )
                        if (
                            loser_anchor is None
                            or loser_anchor.block_hash is None
                            or winner_anchor != loser_anchor.model_dump(mode="json")
                        ):
                            duplicate_problems.append(
                                "duplicate receipts do not bind the same finalized anchor"
                            )
                        else:
                            try:
                                winner_order = _duplicate_order_key(
                                    loser_anchor.block_hash,
                                    witness.winner_hotkey,
                                )
                                loser_order = _duplicate_order_key(
                                    loser_anchor.block_hash,
                                    witness.loser_hotkey,
                                )
                            except ValueError as exc:
                                duplicate_problems.append(
                                    f"duplicate ordering inputs are malformed: {exc}"
                                )
                            else:
                                if winner_order >= loser_order:
                                    duplicate_problems.append(
                                        "duplicate winner violates anchor_hash_hotkey/1 "
                                        "ordering"
                                    )
                    # Exact duplicates share one content-addressed object.  It
                    # was already independently fetched and integrity-checked as
                    # the loser payload, so reuse it rather than materializing
                    # the same digest to the same temporary destination twice.
                    if witness.winner_output == bundle.miner_output:
                        duplicate_winner_output = artifacts.get(
                            ArtifactKind.MINER_OUTPUT
                        )
                    else:
                        try:
                            duplicate_winner_output = store.materialize(
                                witness.winner_output,
                                media_tmp.name,
                                max_bytes=MAX_AUDIT_OUTPUT_BYTES,
                            )
                        except IntegrityError as exc:
                            duplicate_problems.append(
                                f"duplicate winner artifact is corrupt: {exc}"
                            )
                        except (FileNotFoundError, OSError) as exc:
                            duplicate_problems.append(
                                f"duplicate winner artifact is unavailable: {exc}"
                            )
                if duplicate_problems:
                    checks.append(
                        _fail(
                            "duplicate_witness",
                            MALFORMED_SCORE_PACKET,
                            "; ".join(duplicate_problems),
                        )
                    )
                else:
                    checks.append(_ok("duplicate_witness"))
            # 5a. Identity: the packet must be FOR this bundle's challenge, item,
            # and miner. The miner is anchored to the EXPECTED identity when the
            # caller supplies one (`expected_miner_hotkey` — the uid this item is
            # attributed to), else to the bundle's pinned miner. Crucially the
            # miner check is NOT skipped merely because the bundle pins no miner
            #: a packet minted for another miner must not pass
            # recompute by leaving the bundle miner null. When both an expected
            # identity and a bundle-pinned miner are present they must also agree.
            identity_problems: list[str] = []
            # an internal review: a required-but-null expected identity is a fault, not a
            # fall-back to the bundle's pinned miner — a null expected hotkey must never
            # let a packet pass the miner comparison by default.
            if require_expected_miner and not expected_miner_hotkey:
                identity_problems.append(
                    f"expected miner identity is required but missing/empty "
                    f"({expected_miner_hotkey!r}) — the packet's miner cannot be bound to "
                    "the attributed uid (null-hotkey bypass)"
                )
            if packet.challenge_id != bundle.challenge_id:
                identity_problems.append(
                    f"challenge_id {packet.challenge_id!r} != bundle"
                    f" {bundle.challenge_id!r}"
                )
            if packet.item_id != bundle.item_id:
                identity_problems.append(
                    f"item_id {packet.item_id!r} != bundle {bundle.item_id!r}"
                )
            if packet.content_digest != bundle.miner_output.digest:
                identity_problems.append(
                    f"content_digest {packet.content_digest!r} != bundle miner_output "
                    f"digest {bundle.miner_output.digest!r}"
                )
            expected_miner = (
                expected_miner_hotkey
                if expected_miner_hotkey is not None
                else bundle.miner_hotkey
            )
            if expected_miner is not None and packet.miner_hotkey != expected_miner:
                identity_problems.append(
                    f"miner_hotkey {packet.miner_hotkey!r} != expected"
                    f" {expected_miner!r}"
                )
            if (
                expected_miner_hotkey is not None
                and bundle.miner_hotkey is not None
                and bundle.miner_hotkey != expected_miner_hotkey
            ):
                identity_problems.append(
                    f"bundle miner_hotkey {bundle.miner_hotkey!r} != expected"
                    f" {expected_miner_hotkey!r}"
                )
            if not identity_problems:
                checks.append(_ok("packet_identity"))
            else:
                checks.append(
                    _fail(
                        "packet_identity",
                        IDENTITY_MISMATCH,
                        "; ".join(identity_problems)
                        + " — score packet minted for a different challenge/item/miner",
                    )
                )
            # 5b. Version pinning: packet vs bundle.
            if packet.scorer_version == bundle.scorer_version:
                checks.append(_ok("scorer_version"))
            else:
                checks.append(
                    _fail(
                        "scorer_version",
                        SCORER_VERSION_MISMATCH,
                        f"packet scorer_version {packet.scorer_version!r}"
                        f" != bundle {bundle.scorer_version!r}",
                    )
                )
            # 5c. Backend pinning: the bundle must carry the packet's COMPLETE
            # backend map.  Subset matching made an empty bundle mapping pass
            # vacuously even when a measured packet named its actual backends.
            backend_problems: list[str] = []
            for backend, pinned in sorted(bundle.backend_versions.items()):
                claimed = packet.backend_versions.get(backend)
                if claimed is None and backend not in packet.backend_versions:
                    backend_problems.append(
                        f"bundle pins {backend}={pinned!r} but the packet does"
                        " not report that backend"
                    )
                elif claimed != pinned:
                    backend_problems.append(
                        f"packet {backend}={claimed!r} != bundle pin {pinned!r}"
                    )
            for backend, claimed in sorted(packet.backend_versions.items()):
                if backend not in bundle.backend_versions:
                    backend_problems.append(
                        f"packet reports unpinned backend {backend}={claimed!r}"
                    )
            if not backend_problems:
                checks.append(_ok("backend_versions"))
            else:
                checks.append(
                    _fail(
                        "backend_versions",
                        BACKEND_VERSION_MISMATCH,
                        "; ".join(backend_problems),
                    )
                )
            # 5d. Internal gates-first invariant of the packet itself.
            if not packet.gate_passed and packet.score != 0.0:
                checks.append(
                    _fail(
                        "packet_consistency",
                        PACKET_INCONSISTENT,
                        f"gate_passed is false but score is {packet.score}"
                        " — gates-first requires 0.0",
                    )
                )
            elif packet.gate_passed and packet.violations:
                checks.append(
                    _fail(
                        "packet_consistency",
                        PACKET_INCONSISTENT,
                        f"gate_passed is true but {len(packet.violations)} violation(s)"
                        " are recorded",
                    )
                )
            elif packet.gate_passed and packet.breakdown is None:
                checks.append(
                    _fail(
                        "packet_consistency",
                        PACKET_INCONSISTENT,
                        "gate_passed is true but breakdown is null — the formula"
                        " breakdown may be absent only when the gate failed",
                    )
                )
            else:
                checks.append(_ok("packet_consistency"))
    else:
        checks.append(
            _fail("score_packet_parse", ARTIFACT_MISSING, "score packet unavailable")
        )

    # 5e. Bind the pre-dispatch COMMITTED scorer to the packet and bundle.
    # Packet <-> bundle and recomputer <-> bundle checks are insufficient by
    # themselves: an authority could otherwise rewrite all three to a different
    # scorer after seeing the miner response while leaving the challenge's
    # already-anchored scorer identity ignored. Validator-zero uses its reserved
    # derived identity, which remains cryptographically bound to the committed
    # worker rather than equalling it verbatim.
    committed_identity: tuple[str, str] | None = None
    committed_identity_error = "DAG_REVEAL artifact unavailable"
    if competition_context is not None and competition_manifest is not None:
        committed_scorer = competition_manifest.get("scoring_version")
        if isinstance(committed_scorer, str) and committed_scorer:
            committed_identity = (committed_scorer, competition_context.track)
        else:
            committed_identity_error = "competition manifest has no committed scorer"
    elif dag is not None:
        committed_identity, committed_identity_error = _parse_committed_scorer(dag)
    if committed_identity is None:
        checks.append(
            _fail(
                "committed_scorer_version",
                (
                    COMPETITION_MANIFEST_INVALID
                    if competition_context is not None
                    else (REVEAL_INVALID if dag is not None else ARTIFACT_MISSING)
                ),
                committed_identity_error,
            )
        )
    elif packet is None:
        checks.append(
            _fail(
                "committed_scorer_version",
                SCORER_VERSION_MISMATCH,
                "cannot bind the committed scorer identity because the score packet"
                " is unavailable or malformed",
            )
        )
    else:
        committed_scorer, committed_track = committed_identity
        expected_packet_scorer, identity_error = _expected_packet_scorer(
            packet,
            committed_scorer=committed_scorer,
            committed_track=committed_track,
        )
        if (
            expected_packet_scorer is not None
            and packet.scorer_version == expected_packet_scorer
            and bundle.scorer_version == expected_packet_scorer
        ):
            checks.append(_ok("committed_scorer_version"))
        else:
            detail = identity_error or (
                f"DAG_REVEAL commits scorer_version {committed_scorer!r}, which expects"
                f" packet identity {expected_packet_scorer!r}; packet reports"
                f" {packet.scorer_version!r} and bundle reports {bundle.scorer_version!r}"
            )
            checks.append(
                _fail(
                    "committed_scorer_version",
                    SCORER_VERSION_MISMATCH,
                    detail
                    + " — scorer identity was not fixed by the challenge commitment",
                )
            )

    # 6. Merkle inclusion of the score packet in the published set.
    if published_root is not None:
        if inclusion_proof is None:
            checks.append(
                _fail(
                    "merkle_inclusion",
                    MERKLE_EXCLUSION,
                    "published root supplied without an inclusion proof",
                )
            )
        else:
            try:
                included = verify_merkle_proof(
                    bundle.score_packet.digest, inclusion_proof, published_root
                )
            except ValueError as exc:
                # an internal review: a malformed inclusion-proof sibling hex (non-hex bytes)
                # crashes `bytes.fromhex` inside `verify_merkle_proof`. A structural defect
                # in the AUTHORITY's own proof is a provable fault — the packet is NOT
                # provably in the committed set — so it is MERKLE_EXCLUSION (a FAIL that
                # rolls up to a SIGNED DISPUTED report), never an uncaught exception that
                # makes the auditor runner retry forever and BLOCK the cursor.
                checks.append(
                    _fail(
                        "merkle_inclusion",
                        MERKLE_EXCLUSION,
                        f"malformed inclusion proof for score packet "
                        f"{bundle.score_packet.digest}: {exc} — a structural defect in the "
                        "authority's proof; the packet is not provably in the committed set",
                    )
                )
            else:
                if included:
                    checks.append(_ok("merkle_inclusion"))
                else:
                    checks.append(
                        _fail(
                            "merkle_inclusion",
                            MERKLE_EXCLUSION,
                            f"score packet {bundle.score_packet.digest} is not included in "
                            f"the published root {published_root} — injected outside the "
                            "committed set",
                        )
                    )
    else:
        checks.append(
            _skip(
                "merkle_inclusion",
                MISSING_ANCHOR,
                "no published root supplied",
                strict=strict,
            )
        )

    # 7. Recompute and compare: metrics, top-level score, gate outcome, version.
    if packet is not None and all(
        k in artifacts
        for k in (ArtifactKind.CHALLENGE_INPUT, ArtifactKind.MINER_OUTPUT)
    ):
        try:
            if packet.scorer_version.startswith(_DUPLICATE_SCORER_PREFIX):
                duplicate_recompute = getattr(recomputer, "recompute_duplicate", None)
                if (
                    not callable(duplicate_recompute)
                    or duplicate_witness is None
                    or duplicate_winner_output is None
                ):
                    raise RuntimeError(
                        "duplicate packet requires a CPU two-output recomputer and a valid witness"
                    )
                recomputed = duplicate_recompute(
                    bundle,
                    artifacts,
                    duplicate_winner_output,
                    duplicate_witness.model_dump(mode="json"),
                )
            elif packet.scorer_version.startswith(_ORCHESTRATOR_ZERO_PREFIX):
                zero_recompute = getattr(
                    recomputer, "recompute_orchestrator_zero", None
                )
                if not callable(zero_recompute) or committed_identity is None:
                    raise RuntimeError(
                        "orchestrator-zero packet requires a CPU zero-record "
                        "recomputer and a valid committed scorer identity"
                    )
                committed_scorer, committed_track = committed_identity
                recomputed = zero_recompute(
                    bundle,
                    artifacts,
                    committed_scorer_version=committed_scorer,
                    committed_track=committed_track,
                )
            else:
                recomputed = recomputer.recompute(bundle, artifacts)
        except Exception as exc:  # scoring engine failure is itself a finding
            checks.append(
                _fail("score_recompute", RECOMPUTE_ERROR, f"recompute failed: {exc}")
            )
        else:
            # 7a. The recomputer must be running the pinned scorer version.
            if recomputed.scorer_version == bundle.scorer_version:
                checks.append(_ok("scorer_version_recompute"))
            else:
                checks.append(
                    _fail(
                        "scorer_version_recompute",
                        SCORER_VERSION_MISMATCH,
                        f"recomputer reports scorer_version {recomputed.scorer_version!r}"
                        f" != bundle {bundle.scorer_version!r}"
                        " — recompute did not run the pinned scorer",
                    )
                )
            # 7b. The independent scorer must actually be running the complete
            # backend environment pinned by the packet-bound bundle.
            if recomputed.backend_versions == bundle.backend_versions:
                checks.append(_ok("backend_versions_recompute"))
            else:
                checks.append(
                    _fail(
                        "backend_versions_recompute",
                        BACKEND_VERSION_MISMATCH,
                        "recomputer backend_versions "
                        f"{recomputed.backend_versions!r} != bundle pins "
                        f"{bundle.backend_versions!r} — recompute did not run the "
                        "pinned backend composition",
                    )
                )
            # 7c. Metric set + per-metric values (numeric entries only).
            recorded_metrics = packet.numeric_metrics()
            recorded_keys = set(recorded_metrics)
            recomputed_keys = set(recomputed.metrics)
            hysteresis_reason, hysteresis_metric_exceptions = (
                _boundary_hysteresis_reason(
                    packet,
                    recomputed,
                    recorded_metrics,
                    recomputed.metrics,
                    tol,
                )
            )
            if recorded_keys != recomputed_keys:
                checks.append(
                    _fail(
                        "metric_set",
                        METRIC_SET_MISMATCH,
                        f"recorded metrics {sorted(recorded_keys)}"
                        f" != recomputable metrics {sorted(recomputed_keys)}",
                    )
                )
            else:
                checks.append(_ok("metric_set"))
            for metric in sorted(recorded_keys & recomputed_keys):
                recorded = recorded_metrics[metric]
                fresh = recomputed.metrics[metric]
                allowed = tol.get(metric, 0.0)
                name = f"score_recompute:{metric}"
                if abs(recorded - fresh) <= allowed:
                    checks.append(_ok(name))
                elif (
                    hysteresis_reason is not None
                    and metric in hysteresis_metric_exceptions
                ):
                    checks.append(_ok(name, hysteresis_reason))
                else:
                    checks.append(
                        _fail(
                            name,
                            SCORE_MISMATCH,
                            f"recorded {metric}={recorded} but independent recompute"
                            f" yields {fresh} (tolerance {allowed})",
                        )
                    )
            # 7c. The AUTHORITATIVE top-level score — agreeing metrics with an
            # edited top-level score must fail here, not slip through.
            allowed = tol.get("score", 0.0)
            if abs(packet.score - recomputed.score) <= allowed:
                checks.append(_ok("score_recompute:score"))
            elif hysteresis_reason is not None and _scores_are_internally_honest(
                packet, recomputed, tolerance=allowed
            ):
                checks.append(_ok("score_recompute:score", hysteresis_reason))
            else:
                checks.append(
                    _fail(
                        "score_recompute:score",
                        SCORE_MISMATCH,
                        f"packet top-level score={packet.score} but independent"
                        f" recompute yields {recomputed.score} (tolerance {allowed})",
                    )
                )
            # 7d. Gate outcome, gates-first: a recomputed gate failure means the
            # packet must say gate_passed=False and score 0.0 (7c catches the
            # score; this catches a doctored gate_passed flag).
            if packet.gate_passed == recomputed.gate_passed:
                checks.append(_ok("gate_recompute"))
            elif hysteresis_reason is not None:
                checks.append(_ok("gate_recompute", hysteresis_reason))
            else:
                checks.append(
                    _fail(
                        "gate_recompute",
                        SCORE_MISMATCH,
                        f"packet gate_passed={packet.gate_passed} but independent"
                        f" recompute yields {recomputed.gate_passed}",
                    )
                )
    else:
        checks.append(
            _fail(
                "score_recompute",
                RECOMPUTE_ERROR,
                "recompute skipped: required artifacts or parsed score packet unavailable",
            )
        )

    media_tmp.cleanup()
    return VerificationReport(
        challenge_id=bundle.challenge_id,
        bundle_digest=actual_digest,
        strict=strict,
        checks=checks,
    )
