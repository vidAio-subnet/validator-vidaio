"""Population-relative source-proximity verdicts, committed as auditable round evidence.

Background: a compression output that is an encode of the sealed PRISTINE reference
rather than of the served (degraded) input scores as if it had seen the holdout. Two residuals expose it without any secret data,
both published in every measured compression packet by the scoring worker:

* ``vmaf_residual``  = VMAF(candidate, pristine) - VMAF(candidate, input)
* ``chroma_residual`` = PSNR_UV(candidate, pristine) - PSNR_UV(candidate, input)

An encoder that only saw the input sits below the round's population on both; an
encode of the pristine sits above it. Clip effects shift a whole round, so the
decision is RELATIVE to the round median, never an absolute threshold, and it needs
both signals (luma alone has honest false positives on bimodal clips; the scored
VMAF model is luma-only, so a score-driven encoder cannot move chroma toward the
pristine by accident).

The verdict is minted exactly like the content equal-share rule: the authority builds
a roster of every eligible measured packet in the round (uid-sorted, each bound to its
archived packet/output/receipt and carrying its two residuals), derives the medians and
the flagged set from that roster alone, and mints a SOURCE_PROXIMITY zero packet whose
single metric is the canonical witness. Auditors re-derive medians and flags from the
same roster, verify each roster value against the archived original packet, recompute
the flagged member's residuals from the released media, and check roster completeness
against the epoch's finalized packets.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, ArtifactRef
from vidaio.challenge import ChallengeAnchor
from vidaio.services.artifact_auth import MinerArtifactReceipt

SOURCE_SCORER_NAME = "validator-source-proximity/1"
SOURCE_SCORER_PREFIX = SOURCE_SCORER_NAME + "+"
SOURCE_WITNESS_METRIC = "source_proximity_witness"
#: Rule versions and their margins: (luma excess over the round median, chroma excess in dB,
#: minimum absolute luma residual). BOTH excesses are required, plus the absolute floor.
#: v1 (0.25 / 0.10 dB / > 0) flagged two honest encodes in one round (2026-09-13, round
#: b3d81295: luma excess +0.30 and +0.35 with chroma +0.20 and +0.68 dB) and sat within
#: 0.03 dB of a third; over the 34 rounds after its roll honest outputs reached luma excess
#: +0.30 and chroma excess +0.98 dB (denoising/sharpening recipes move both residuals toward
#: the pristine). Pristine-derived outputs measured +0.3 ... +0.9 luma and 0.5 ... 1.7 dB
#: chroma above their rounds. v2 widens both margins and requires a clearly positive
#: absolute luma residual. Archived v1 evidence re-derives with the v1 margins.
RULE_MARGINS: dict[str, tuple[float, float, float]] = {
    "source_proximity/1": (0.25, 0.10, 0.0),
    "source_proximity/2": (0.45, 0.25, 0.15),
}
SOURCE_RULE = "source_proximity/2"
#: Excess of an item's residual over the round median that flags it (BOTH required).
VMAF_RESIDUAL_EXCESS_MIN, CHROMA_RESIDUAL_EXCESS_MIN_DB, MIN_ABS_VMAF_RESIDUAL = RULE_MARGINS[SOURCE_RULE]
#: An honest encode of the served input scores at least as well against that input as
#: against the pristine (its noise is correlated with the input), so a NEGATIVE absolute
#: ``vmaf_residual`` is never flagged however far above a bimodal round's median it sits.
#: This is what keeps clip-shifted rounds (two honest recipes far apart) from producing
#: false positives; the population-relative margins alone would flag the upper mode.
REQUIRE_POSITIVE_VMAF_RESIDUAL = True
#: Fewer eligible items than this and the population carries no information.
MIN_ROSTER = 3
#: Packet metric names the roster copies (numeric, tolerance-checked by auditors).
VMAF_RESIDUAL_METRIC = "vmaf_residual"
CHROMA_RESIDUAL_METRIC = "chroma_residual"


class InvalidSourceEvidence(ValueError):
    """A source-proximity decision lacks canonical, independently auditable evidence."""


class SourceMember(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    uid: int = Field(ge=0, strict=True)
    hotkey: str
    score_packet: ArtifactRef
    output: ArtifactRef
    receipt: MinerArtifactReceipt
    vmaf_residual: float
    chroma_residual: float

    @model_validator(mode="after")
    def _bound_media(self) -> "SourceMember":
        if self.score_packet.kind is not ArtifactKind.SCORE_PACKET or self.output.kind is not ArtifactKind.MINER_OUTPUT:
            raise ValueError("source member artifact kinds are incorrect")
        if (self.output.digest != self.receipt.output_digest
                or self.output.byte_size != self.receipt.output_size
                or self.hotkey != self.receipt.miner_hotkey):
            raise ValueError("source member output/hotkey differ from signed receipt")
        return self


def median(values: Sequence[float]) -> float:
    """Plain sample median: middle element, or the mean of the two middle elements."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise InvalidSourceEvidence("median of an empty roster")
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def derive_medians(roster: Sequence[SourceMember]) -> tuple[float, float]:
    return (median([m.vmaf_residual for m in roster]), median([m.chroma_residual for m in roster]))


def rule_margins(rule: str) -> tuple[float, float, float]:
    try:
        return RULE_MARGINS[rule]
    except KeyError:
        raise InvalidSourceEvidence(f"unknown source-proximity rule {rule!r}") from None


def derive_flags(roster: Sequence[SourceMember], rule: str = SOURCE_RULE) -> tuple[int, ...]:
    """uids whose BOTH residuals exceed the round medians by the rule's margins.

    With ``REQUIRE_POSITIVE_VMAF_RESIDUAL`` the item's absolute luma residual must also
    be above the rule's floor (closer to the pristine than to its own input in absolute
    terms; v2 demands a clear margin, v1 any positive value).
    """
    if len(roster) < MIN_ROSTER:
        return ()
    vmaf_min, chroma_min, abs_min = rule_margins(rule)
    vmaf_median, chroma_median = derive_medians(roster)
    return tuple(sorted(
        m.uid for m in roster
        if m.vmaf_residual - vmaf_median >= vmaf_min
        and m.chroma_residual - chroma_median >= chroma_min
        and (m.vmaf_residual > abs_min or not REQUIRE_POSITIVE_VMAF_RESIDUAL)
    ))


class SourceRoundEvidence(BaseModel):
    """The complete eligible roster of one compression round and the verdict derived from it."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    rule: Literal["source_proximity/1", "source_proximity/2"] = SOURCE_RULE
    vmaf_residual_excess_min: float = VMAF_RESIDUAL_EXCESS_MIN
    chroma_residual_excess_min_db: float = CHROMA_RESIDUAL_EXCESS_MIN_DB
    require_positive_vmaf_residual: Literal[True] = REQUIRE_POSITIVE_VMAF_RESIDUAL
    min_roster: Literal[3] = MIN_ROSTER
    challenge_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    track: str = Field(min_length=1)
    commitment_anchor: ChallengeAnchor
    challenge_input: ArtifactRef
    reference_original: ArtifactRef
    committed_scorer_version: str = Field(min_length=1)
    scoring_config_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    roster: tuple[SourceMember, ...] = Field(min_length=1, max_length=4096)
    vmaf_residual_median: float
    chroma_residual_median: float
    flagged: tuple[int, ...]

    @model_validator(mode="after")
    def _canonical_decision(self) -> "SourceRoundEvidence":
        if self.item_id != self.challenge_id:
            raise ValueError("source round item_id must be the common challenge_id")
        uids = tuple(member.uid for member in self.roster)
        if uids != tuple(sorted(set(uids))) or len({member.hotkey for member in self.roster}) != len(uids):
            raise ValueError("source roster must be uid-sorted with distinct uids and hotkeys")
        if self.commitment_anchor.block_hash is None:
            raise ValueError("source round requires finalized anchor hash")
        if (self.challenge_input.kind is not ArtifactKind.CHALLENGE_INPUT
                or self.reference_original.kind is not ArtifactKind.REFERENCE_ORIGINAL):
            raise ValueError("source round shared artifact kinds are incorrect")
        input_bindings = set()
        validators = set()
        for member in self.roster:
            receipt = member.receipt
            if (receipt.metadata.commitment_anchor != self.commitment_anchor
                    or receipt.metadata.track != self.track
                    or receipt.metadata.task_id != f"{self.challenge_id}:{member.uid}"):
                raise ValueError("source member receipt differs from committed challenge/track/anchor")
            input_bindings.add((receipt.metadata.input_digest, receipt.input_size))
            validators.add(receipt.validator_hotkey)
        if input_bindings != {(self.challenge_input.digest, self.challenge_input.byte_size)} or len(validators) != 1:
            raise ValueError("source roster mixes challenge inputs or validator identities")
        vmaf_min, chroma_min, _abs_min = rule_margins(self.rule)
        if (self.vmaf_residual_excess_min, self.chroma_residual_excess_min_db) != (vmaf_min, chroma_min):
            raise ValueError("source round margins differ from the rule")
        if (self.vmaf_residual_median, self.chroma_residual_median) != derive_medians(self.roster):
            raise ValueError("source round medians differ from the roster")
        if self.flagged != derive_flags(self.roster, self.rule):
            raise ValueError("source round flagged set differs from the roster decision")
        return self

    def to_json(self) -> str:
        return canonical_json_bytes(self.model_dump(mode="json")).decode()

    def digest(self) -> str:
        return sha256_hex(self.to_json().encode())

    def member(self, uid: int) -> SourceMember:
        for member in self.roster:
            if member.uid == uid:
                return member
        raise InvalidSourceEvidence(f"uid {uid} absent from source roster")

    def excess(self, uid: int) -> tuple[float, float]:
        member = self.member(uid)
        return (member.vmaf_residual - self.vmaf_residual_median,
                member.chroma_residual - self.chroma_residual_median)


class SourceProximityWitness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    round_evidence: SourceRoundEvidence
    member_uid: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _flagged(self) -> "SourceProximityWitness":
        if self.member_uid not in self.round_evidence.flagged:
            raise ValueError("source witness member is not a flagged roster uid")
        return self

    def to_json(self) -> str:
        return canonical_json_bytes(self.model_dump(mode="json")).decode()


def _parse_canonical(value: object, model: type[BaseModel]) -> Any:
    if not isinstance(value, str) or not value or len(value.encode()) > 16 * 1024 * 1024:
        raise InvalidSourceEvidence("source witness is missing or exceeds metadata limit")
    try:
        obj = json.loads(value)
        if not isinstance(obj, dict) or canonical_json_bytes(obj).decode() != value:
            raise InvalidSourceEvidence("source evidence must be canonical JSON")
        return model.model_validate(obj)
    except (ValidationError, ValueError, TypeError) as exc:
        raise InvalidSourceEvidence(f"invalid source evidence: {exc}") from exc


def parse_source_round(value: object) -> SourceRoundEvidence:
    return _parse_canonical(value, SourceRoundEvidence)


def parse_source_witness(value: object) -> SourceProximityWitness:
    return _parse_canonical(value, SourceProximityWitness)


def source_witness_from_packet(packet: Any) -> SourceProximityWitness:
    metrics = packet.metrics if hasattr(packet, "metrics") else packet.get("metrics", {})
    raw = metrics.get(SOURCE_WITNESS_METRIC) if isinstance(metrics, Mapping) else None
    return parse_source_witness(raw)


def is_source_proximity_identity(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith(SOURCE_SCORER_PREFIX)


def source_proximity_identity(*, committed_scorer_version: str, track: str, scoring_config_digest: str,
                              rule: str = SOURCE_RULE) -> str:
    vmaf_min, chroma_min, abs_min = rule_margins(rule)
    fields = {
        "convention": SOURCE_SCORER_NAME, "committed_scorer_version": committed_scorer_version,
        "track": track, "scoring_config_digest": scoring_config_digest, "rule": rule,
        "vmaf_residual_excess_min": vmaf_min,
        "chroma_residual_excess_min_db": chroma_min, "min_roster": MIN_ROSTER,
        "require_positive_vmaf_residual": REQUIRE_POSITIVE_VMAF_RESIDUAL,
        "vmaf_residual_metric": VMAF_RESIDUAL_METRIC, "chroma_residual_metric": CHROMA_RESIDUAL_METRIC,
    }
    if rule != "source_proximity/1":
        # v1 identities were minted without this field; keep them reproducible byte for byte.
        fields["min_abs_vmaf_residual"] = abs_min
    return SOURCE_SCORER_PREFIX + sha256_hex(canonical_json_bytes(fields))[:12]


def packet_residuals(packet: Any) -> tuple[float, float] | None:
    """The (vmaf_residual, chroma_residual) pair a measured packet publishes, if both are numeric."""
    metrics = packet.metrics if hasattr(packet, "metrics") else packet.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    values = []
    for key in (VMAF_RESIDUAL_METRIC, CHROMA_RESIDUAL_METRIC):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return values[0], values[1]


def validate_member_packet(member: SourceMember, packet: Any, context: SourceRoundEvidence) -> None:
    """Bind a roster descriptor to the real original measured packet it was copied from."""
    obj = packet.model_dump(mode="json") if hasattr(packet, "model_dump") else dict(packet)
    expected = {
        "challenge_id": context.challenge_id, "item_id": member.receipt.metadata.task_id,
        "track": context.track, "miner_hotkey": member.hotkey, "content_digest": member.output.digest,
        "scorer_version": context.committed_scorer_version, "scoring_config_digest": context.scoring_config_digest,
    }
    if any(obj.get(key) != value for key, value in expected.items()) or obj.get("gate_passed") is not True:
        raise InvalidSourceEvidence("original packet is not bound to eligible source roster member")
    if obj.get("breakdown") is None or obj.get("violations"):
        raise InvalidSourceEvidence("source roster requires a successfully measured gate-passing packet")
    score = obj.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not score > 0:
        raise InvalidSourceEvidence("source roster requires a positively scored packet")
    residuals = packet_residuals(obj)
    if residuals is None or residuals != (member.vmaf_residual, member.chroma_residual):
        raise InvalidSourceEvidence("source roster residuals differ from the original measured packet")


def violation_detail(context: SourceRoundEvidence, uid: int) -> str:
    vmaf_excess, chroma_excess = context.excess(uid)
    return (
        "output is closer to the sealed pristine reference than to the served input: "
        f"vmaf_residual {vmaf_excess:+.4f} and chroma_residual {chroma_excess:+.4f} dB above the "
        f"round medians of {len(context.roster)} measured outputs (rule {context.rule})"
    )


def mint_source_proximity_packet(*, witness: SourceProximityWitness, config: Any) -> Any:
    from vidaio.scoring.gates import ReasonCode, ValidityViolation
    from vidaio.scoring.result import compose_item_score, config_digest

    witness = parse_source_witness(witness.to_json())
    context = witness.round_evidence
    if config_digest(config) != context.scoring_config_digest:
        raise InvalidSourceEvidence("source round config differs from active scoring config")
    member = context.member(witness.member_uid)
    vmaf_excess, chroma_excess = context.excess(witness.member_uid)
    return compose_item_score(
        item_id=member.receipt.metadata.task_id, challenge_id=context.challenge_id, track=context.track,
        miner_hotkey=member.hotkey, content_digest=member.output.digest,
        gate_passed=False, violations=[ValidityViolation(
            code=ReasonCode.SOURCE_PROXIMITY, detail=violation_detail(context, witness.member_uid),
            measured=chroma_excess, limit=rule_margins(context.rule)[1])],
        breakdown=None, config=config, metrics={SOURCE_WITNESS_METRIC: witness.to_json()},
        backend_versions={}, scorer_version=source_proximity_identity(
            committed_scorer_version=context.committed_scorer_version, track=context.track,
            scoring_config_digest=context.scoring_config_digest, rule=context.rule),
    )
