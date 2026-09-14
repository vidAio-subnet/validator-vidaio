"""Additive, auditable connected-component content duplicate decisions.

The historical exact-byte witness stays in ``duplicate_evidence`` unchanged.
A content verdict commits the whole eligible, originally-scored candidate set;
auditors recompute its graph and check membership against finalized packets.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, ArtifactRef
from vidaio.challenge import ChallengeAnchor
from vidaio.scoring.duplicate_evidence import duplicate_order_key
from vidaio.services.artifact_auth import MinerArtifactReceipt

CONTENT_SCORER_NAME = "validator-content-duplicate/4"
CONTENT_SCORER_PREFIX = CONTENT_SCORER_NAME + "+"
CONTENT_SCORER_PREFIX_V1 = "validator-content-duplicate/1+"
CONTENT_SCORER_PREFIX_V2 = "validator-content-duplicate/2+"
CONTENT_SCORER_PREFIX_V3 = "validator-content-duplicate/3+"
CONTENT_SHARE_RULE = "equal_share/1"
CONTENT_WITNESS_METRIC = "content_duplicate_witness"
CONTENT_EVIDENCE_RULE = "canonical_content/3"
CONTENT_EVIDENCE_RULE_V1 = "canonical_content/1"
CONTENT_EVIDENCE_RULE_V2 = "canonical_content/2"
CONTENT_EXACT_MATCH_RULE = "exact_canonical_digest/1"
CONTENT_SELECTION_RULE = "anchor_hash_hotkey/1"
ContentEvidenceRule = Literal["canonical_content/1", "canonical_content/2", "canonical_content/3"]


class InvalidContentEvidence(ValueError):
    """A content decision lacks canonical, independently auditable evidence."""


class ContentMember(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    uid: int = Field(ge=0, strict=True)
    hotkey: str
    score_packet: ArtifactRef
    output: ArtifactRef
    receipt: MinerArtifactReceipt
    canonical_content_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    content_fingerprint: tuple[str, ...] = Field(min_length=32, max_length=32)
    encoded_size: int = Field(gt=0, strict=True)
    canonicalization_plan_digest: str = Field(pattern=SHA256_HEX_PATTERN)

    @field_validator("content_fingerprint")
    @classmethod
    def _words(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(word) != 16 or any(c not in "0123456789abcdef" for c in word) for word in value):
            raise ValueError("fingerprint requires 32 lowercase 16-hex words")
        return value

    @model_validator(mode="after")
    def _bound_media(self) -> "ContentMember":
        if self.score_packet.kind is not ArtifactKind.SCORE_PACKET or self.output.kind is not ArtifactKind.MINER_OUTPUT:
            raise ValueError("content member artifact kinds are incorrect")
        if (self.output.digest != self.receipt.output_digest
                or self.output.byte_size != self.encoded_size
                or self.encoded_size != self.receipt.output_size
                or self.hotkey != self.receipt.miner_hotkey):
            raise ValueError("content member output/size/hotkey differ from signed receipt")
        return self


def same_content(
    left: Any, right: Any, *, evidence_rule: ContentEvidenceRule = CONTENT_EVIDENCE_RULE,
) -> bool:
    """Compare with the committed rule; new rounds group only exact canonical-digest twins."""
    if evidence_rule not in (CONTENT_EVIDENCE_RULE_V1, CONTENT_EVIDENCE_RULE_V2, CONTENT_EVIDENCE_RULE):
        raise InvalidContentEvidence("unsupported content evidence_rule")
    if left.canonicalization_plan_digest != right.canonicalization_plan_digest:
        return False
    if left.canonical_content_digest == right.canonical_content_digest:
        return True
    if evidence_rule == CONTENT_EVIDENCE_RULE:
        # canonical_content/3: no approximate-size branch. Every asset is served once
        # (retire_after_uses=1), so a re-padded copy of a committed output can never be
        # replayed; the band only ever grouped independent encodes that landed close.
        return False
    if evidence_rule == CONTENT_EVIDENCE_RULE_V2:
        if 1000 * abs(left.encoded_size - right.encoded_size) > 2 * max(left.encoded_size, right.encoded_size):
            return False
    elif 100 * abs(left.encoded_size - right.encoded_size) > max(left.encoded_size, right.encoded_size):
        return False
    return sum((int(a, 16) ^ int(b, 16)).bit_count() <= 6
               for a, b in zip(left.content_fingerprint, right.content_fingerprint, strict=True)) >= 30


def derive_edges(
    roster: Sequence[ContentMember], *, evidence_rule: ContentEvidenceRule = CONTENT_EVIDENCE_RULE,
) -> tuple[tuple[int, int], ...]:
    if evidence_rule not in (CONTENT_EVIDENCE_RULE_V1, CONTENT_EVIDENCE_RULE_V2, CONTENT_EVIDENCE_RULE):
        raise InvalidContentEvidence("unsupported content evidence_rule")
    ordered = sorted(roster, key=lambda member: member.uid)
    return tuple((left.uid, right.uid) for index, left in enumerate(ordered)
                 for right in ordered[index + 1:] if same_content(left, right, evidence_rule=evidence_rule))


def derive_components(
    roster: Sequence[ContentMember], *, evidence_rule: ContentEvidenceRule = CONTENT_EVIDENCE_RULE,
) -> tuple[tuple[int, ...], ...]:
    parents = {member.uid: member.uid for member in roster}
    if len(parents) != len(roster):
        raise InvalidContentEvidence("duplicate uid in content roster")

    def root(uid: int) -> int:
        while parents[uid] != uid:
            parents[uid] = parents[parents[uid]]
            uid = parents[uid]
        return uid

    for left, right in derive_edges(roster, evidence_rule=evidence_rule):
        a, b = root(left), root(right)
        parents[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for uid in sorted(parents):
        groups.setdefault(root(uid), []).append(uid)
    return tuple(sorted(tuple(group) for group in groups.values()))


class _ContentRoundEvidenceBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    evidence_rule: ContentEvidenceRule
    selection_rule: Literal["anchor_hash_hotkey/1"] = CONTENT_SELECTION_RULE
    fingerprint_version: Literal[1] = 1
    hamming_max: Literal[6] = 6
    matched_frames_min: Literal[30] = 30
    challenge_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    track: str = Field(min_length=1)
    commitment_anchor: ChallengeAnchor
    challenge_input: ArtifactRef
    reference_original: ArtifactRef
    committed_scorer_version: str = Field(min_length=1)
    scoring_config_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    roster: tuple[ContentMember, ...] = Field(min_length=1, max_length=4096)
    edges: tuple[tuple[int, int], ...]
    components: tuple[tuple[int, ...], ...]
    skipped_components: tuple[tuple[int, ...], ...] = ()

    @model_validator(mode="after")
    def _canonical_graph(self) -> "_ContentRoundEvidenceBase":
        # Miner packet item ids carry a uid suffix, but one dispatch is the
        # common comparison domain. An authority-selected alternate item name
        # must not split one eligible roster into separately winning subsets.
        if self.item_id != self.challenge_id:
            raise ValueError("content round item_id must be the common challenge_id")
        uids = tuple(member.uid for member in self.roster)
        if uids != tuple(sorted(set(uids))) or len({member.hotkey for member in self.roster}) != len(uids):
            raise ValueError("content roster must be uid-sorted with distinct uids and hotkeys")
        if self.commitment_anchor.block_hash is None:
            raise ValueError("content round requires finalized anchor hash")
        if self.challenge_input.kind is not ArtifactKind.CHALLENGE_INPUT or self.reference_original.kind is not ArtifactKind.REFERENCE_ORIGINAL:
            raise ValueError("content round shared artifact kinds are incorrect")
        input_bindings = set()
        validators = set()
        for member in self.roster:
            receipt = member.receipt
            if (receipt.metadata.commitment_anchor != self.commitment_anchor
                    or receipt.metadata.track != self.track
                    or receipt.metadata.task_id != f"{self.challenge_id}:{member.uid}"):
                raise ValueError("content member receipt differs from committed challenge/track/anchor")
            duplicate_order_key(self.commitment_anchor.block_hash, member.hotkey)
            input_bindings.add((receipt.metadata.input_digest, receipt.input_size))
            validators.add(receipt.validator_hotkey)
        if input_bindings != {(self.challenge_input.digest, self.challenge_input.byte_size)} or len(validators) != 1:
            raise ValueError("content roster mixes challenge inputs or validator identities")
        if (self.edges != derive_edges(self.roster, evidence_rule=self.evidence_rule)
                or self.components != derive_components(self.roster, evidence_rule=self.evidence_rule)):
            raise ValueError("content round edges/components differ from complete pair graph")
        if (self.skipped_components != tuple(sorted(set(self.skipped_components)))
                or any(component not in self.components for component in self.skipped_components)):
            raise ValueError("content skips must be distinct complete derived components")
        return self

    def to_json(self) -> str:
        return canonical_json_bytes(self.model_dump(mode="json")).decode()

    def digest(self) -> str:
        return sha256_hex(self.to_json().encode())

    @classmethod
    def from_json(cls, value: str) -> "ContentRoundContext":
        return parse_content_round(value)

    def winner(self, component: tuple[int, ...]) -> int:
        if component not in self.components:
            raise InvalidContentEvidence("component absent from full content graph")
        hotkeys = {member.uid: member.hotkey for member in self.roster}
        return min(component, key=lambda uid: duplicate_order_key(self.commitment_anchor.block_hash, hotkeys[uid]))


class ContentRoundEvidenceV1(_ContentRoundEvidenceBase):
    """Historical 1% evidence, retaining its original canonical JSON fields."""

    evidence_rule: Literal["canonical_content/1"] = CONTENT_EVIDENCE_RULE_V1
    size_delta_percent: Literal[1] = 1


class ContentRoundEvidenceV2(_ContentRoundEvidenceBase):
    """Historical 0.2% evidence (2026-09-10 21:43Z until the canonical_content/3 roll)."""

    evidence_rule: Literal["canonical_content/2"] = CONTENT_EVIDENCE_RULE_V2
    size_delta_permille: Literal[2] = 2


class ContentRoundEvidence(_ContentRoundEvidenceBase):
    """Evidence for new rounds: only exact canonical-digest twins are the same content."""

    evidence_rule: Literal["canonical_content/3"] = CONTENT_EVIDENCE_RULE
    match_rule: Literal["exact_canonical_digest/1"] = CONTENT_EXACT_MATCH_RULE


ContentRoundContext = Annotated[
    ContentRoundEvidenceV1 | ContentRoundEvidenceV2 | ContentRoundEvidence,
    Field(discriminator="evidence_rule"),
]
_CONTENT_ROUND_ADAPTER = TypeAdapter(ContentRoundContext)


def _parse_canonical(value: object, model: type[BaseModel] | TypeAdapter[Any]) -> Any:
    if not isinstance(value, str) or not value or len(value.encode()) > 16 * 1024 * 1024:
        raise InvalidContentEvidence("content witness is missing or exceeds metadata limit")
    try:
        obj = json.loads(value)
        if not isinstance(obj, dict) or canonical_json_bytes(obj).decode() != value:
            raise InvalidContentEvidence("content evidence must be canonical JSON")
        return model.validate_python(obj) if isinstance(model, TypeAdapter) else model.model_validate(obj)
    except (ValidationError, ValueError, TypeError) as exc:
        raise InvalidContentEvidence(f"invalid content evidence: {exc}") from exc


def parse_content_round(value: object) -> ContentRoundContext:
    return _parse_canonical(value, _CONTENT_ROUND_ADAPTER)


class ContentDuplicateWitnessV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    round_evidence: ContentRoundEvidenceV1
    component_uids: tuple[int, ...]
    winner_uid: int = Field(ge=0, strict=True)
    loser_uid: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _selection(self) -> "ContentDuplicateWitnessV1":
        context = self.round_evidence
        if (self.component_uids not in context.components or len(self.component_uids) < 2
                or self.component_uids in context.skipped_components
                or self.loser_uid not in self.component_uids
                or self.winner_uid != context.winner(self.component_uids)
                or self.loser_uid == self.winner_uid):
            raise ValueError("content witness has an invalid component/winner/loser")
        return self

    def to_json(self) -> str:
        return canonical_json_bytes(self.model_dump(mode="json")).decode()


ContentDuplicateWitness = ContentDuplicateWitnessV1


class ContentShareWitness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[2] = 2
    round_evidence: ContentRoundContext
    component_uids: tuple[int, ...]
    winner_uid: int = Field(ge=0, strict=True)
    member_uid: int = Field(ge=0, strict=True)
    winner_score: float = Field(gt=0.0, le=1.0)
    share: float

    @model_validator(mode="after")
    def _selection(self) -> "ContentShareWitness":
        context = self.round_evidence
        if (self.component_uids not in context.components or len(self.component_uids) < 2
                or self.component_uids in context.skipped_components
                or self.member_uid not in self.component_uids
                or self.winner_uid != context.winner(self.component_uids)):
            raise ValueError("content share witness has an invalid component/winner/member")
        if self.share != self.winner_score / len(self.component_uids):
            raise ValueError("content share must equal winner_score / len(component_uids) exactly")
        return self

    @property
    def role(self) -> Literal["winner", "loser"]:
        return "winner" if self.member_uid == self.winner_uid else "loser"

    def to_json(self) -> str:
        return canonical_json_bytes(self.model_dump(mode="json")).decode()


def parse_content_witness(value: object) -> ContentDuplicateWitnessV1:
    """Parse a legacy v1 zero witness without changing its audit contract."""
    return _parse_canonical(value, ContentDuplicateWitnessV1)


def content_witness_from_packet(packet: Any) -> ContentDuplicateWitnessV1 | ContentShareWitness:
    metrics = packet.metrics if hasattr(packet, "metrics") else packet.get("metrics", {})
    raw = metrics.get(CONTENT_WITNESS_METRIC) if isinstance(metrics, Mapping) else None
    if not isinstance(raw, str) or not raw or len(raw.encode()) > 16 * 1024 * 1024:
        raise InvalidContentEvidence("content witness is missing or exceeds metadata limit")
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidContentEvidence(f"invalid content evidence: {exc}") from exc
    version = obj.get("schema_version") if isinstance(obj, dict) else None
    if type(version) is int and version == 1:
        return parse_content_witness(raw)
    if type(version) is int and version == 2:
        return _parse_canonical(raw, ContentShareWitness)
    raise InvalidContentEvidence("unsupported content witness schema_version")


def is_content_duplicate_identity(value: str | None) -> bool:
    return content_identity_version(value) is not None


def content_identity_version(value: str | None) -> Literal[1, 2, 3, 4] | None:
    if isinstance(value, str):
        if value.startswith(CONTENT_SCORER_PREFIX_V1):
            return 1
        if value.startswith(CONTENT_SCORER_PREFIX_V2):
            return 2
        if value.startswith(CONTENT_SCORER_PREFIX_V3):
            return 3
        if value.startswith(CONTENT_SCORER_PREFIX):
            return 4
    return None


def content_duplicate_identity_v1(*, committed_scorer_version: str, track: str, scoring_config_digest: str) -> str:
    digest = sha256_hex(canonical_json_bytes({
        "convention": "validator-content-duplicate/1", "committed_scorer_version": committed_scorer_version,
        "track": track, "scoring_config_digest": scoring_config_digest, "evidence_rule": CONTENT_EVIDENCE_RULE_V1,
        "selection_rule": CONTENT_SELECTION_RULE, "fingerprint_version": 1,
        "hamming_max": 6, "matched_frames_min": 30, "size_delta_percent": 1,
    }))
    return CONTENT_SCORER_PREFIX_V1 + digest[:12]


def content_duplicate_identity(
    *, committed_scorer_version: str, track: str, scoring_config_digest: str,
    evidence_rule: ContentEvidenceRule = CONTENT_EVIDENCE_RULE,
) -> str:
    if evidence_rule == CONTENT_EVIDENCE_RULE_V1:
        convention, prefix = "validator-content-duplicate/2", CONTENT_SCORER_PREFIX_V2
        size_rule = {"size_delta_percent": 1}
    elif evidence_rule == CONTENT_EVIDENCE_RULE_V2:
        convention, prefix = "validator-content-duplicate/3", CONTENT_SCORER_PREFIX_V3
        size_rule = {"size_delta_permille": 2}
    elif evidence_rule == CONTENT_EVIDENCE_RULE:
        convention, prefix = CONTENT_SCORER_NAME, CONTENT_SCORER_PREFIX
        size_rule = {"match_rule": CONTENT_EXACT_MATCH_RULE}
    else:
        raise InvalidContentEvidence("unsupported content evidence_rule")
    digest = sha256_hex(canonical_json_bytes({
        "convention": convention, "committed_scorer_version": committed_scorer_version,
        "track": track, "scoring_config_digest": scoring_config_digest, "evidence_rule": evidence_rule,
        "selection_rule": CONTENT_SELECTION_RULE, "fingerprint_version": 1,
        "hamming_max": 6, "matched_frames_min": 30, **size_rule,
        "share_rule": CONTENT_SHARE_RULE,
    }))
    return prefix + digest[:12]


def validate_member_packet(member: ContentMember, packet: Any, context: ContentRoundContext) -> None:
    """Bind a roster descriptor to the real original measured packet."""
    obj = packet.model_dump(mode="json") if hasattr(packet, "model_dump") else dict(packet)
    expected = {
        "challenge_id": context.challenge_id, "item_id": member.receipt.metadata.task_id,
        "track": context.track, "miner_hotkey": member.hotkey, "content_digest": member.output.digest,
        "scorer_version": context.committed_scorer_version, "scoring_config_digest": context.scoring_config_digest,
        "canonical_content_digest": member.canonical_content_digest, "content_fingerprint": list(member.content_fingerprint),
        "encoded_size": member.encoded_size, "canonicalization_plan_digest": member.canonicalization_plan_digest,
    }
    if any(obj.get(key) != value for key, value in expected.items()) or obj.get("gate_passed") is not True:
        raise InvalidContentEvidence("original content packet is not bound to eligible roster member")
    if obj.get("breakdown") is None or obj.get("violations"):
        raise InvalidContentEvidence("content roster requires a successfully measured gate-passing packet")
    score = obj.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not score > 0:
        raise InvalidContentEvidence("content roster requires a positively scored packet; a zero cannot claim the slot")


def mint_content_duplicate_packet(*, witness: ContentDuplicateWitness, config: Any) -> Any:
    from vidaio.scoring.gates import ReasonCode, ValidityViolation
    from vidaio.scoring.result import compose_item_score, config_digest

    # Apply the same bound used by readers before emitting a stored verdict.
    witness = parse_content_witness(witness.to_json())
    context = witness.round_evidence
    if config_digest(config) != context.scoring_config_digest:
        raise InvalidContentEvidence("content round config differs from active scoring config")
    member = next(member for member in context.roster if member.uid == witness.loser_uid)
    return compose_item_score(
        item_id=member.receipt.metadata.task_id, challenge_id=context.challenge_id, track=context.track,
        miner_hotkey=member.hotkey, content_digest=member.output.digest,
        gate_passed=False, violations=[ValidityViolation(code=ReasonCode.DUPLICATE_CONTENT,
            detail=f"auditable same-content component; anchor-salted winner uid {witness.winner_uid}")],
        breakdown=None, config=config, metrics={CONTENT_WITNESS_METRIC: witness.to_json()},
        backend_versions={}, scorer_version=content_duplicate_identity_v1(
            committed_scorer_version=context.committed_scorer_version, track=context.track,
            scoring_config_digest=context.scoring_config_digest),
    )


def mint_content_share_packet(*, witness: ContentShareWitness, config: Any, measured_packet: Any = None) -> Any:
    from vidaio.scoring.gates import ReasonCode, ValidityViolation
    from vidaio.scoring.result import compose_item_score, config_digest

    witness = _parse_canonical(witness.to_json(), ContentShareWitness)
    context = witness.round_evidence
    if config_digest(config) != context.scoring_config_digest:
        raise InvalidContentEvidence("content round config differs from active scoring config")
    member = next(member for member in context.roster if member.uid == witness.member_uid)
    identity = content_duplicate_identity(
        committed_scorer_version=context.committed_scorer_version, track=context.track,
        scoring_config_digest=context.scoring_config_digest, evidence_rule=context.evidence_rule)
    common = dict(
        item_id=member.receipt.metadata.task_id, challenge_id=context.challenge_id, track=context.track,
        miner_hotkey=member.hotkey, content_digest=member.output.digest, config=config,
        scorer_version=identity, shared_score=witness.share,
    )
    if witness.role == "loser":
        n = len(witness.component_uids)
        return compose_item_score(
            **common, gate_passed=False, violations=[ValidityViolation(code=ReasonCode.DUPLICATE_CONTENT,
                detail=f"auditable same-content component of {n}; equal share 1/{n}; anchor-salted reference uid {witness.winner_uid}")],
            breakdown=None, metrics={CONTENT_WITNESS_METRIC: witness.to_json()}, backend_versions={},
        )
    if measured_packet is None:
        raise InvalidContentEvidence("content share winner requires its original measured packet")
    if measured_packet.score != witness.winner_score:
        raise InvalidContentEvidence("content share winner_score differs from original measured packet")
    validate_member_packet(member, measured_packet, context)
    return compose_item_score(
        **common, gate_passed=True, violations=[], breakdown=measured_packet.breakdown,
        metrics={**measured_packet.metrics, CONTENT_WITNESS_METRIC: witness.to_json()},
        backend_versions=measured_packet.backend_versions,
        canonical_content_digest=measured_packet.canonical_content_digest,
        content_fingerprint=measured_packet.content_fingerprint, encoded_size=measured_packet.encoded_size,
        canonicalization_plan_digest=measured_packet.canonicalization_plan_digest,
        pieapp_start_frame=measured_packet.pieapp_start_frame, skips=measured_packet.skips,
    )
