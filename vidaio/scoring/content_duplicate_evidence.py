"""Additive, auditable connected-component content duplicate decisions.

The historical exact-byte witness stays in ``duplicate_evidence`` unchanged.
A content verdict commits the whole eligible, originally-scored candidate set;
auditors recompute its graph and check membership against finalized packets.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, ArtifactRef
from vidaio.challenge import ChallengeAnchor
from vidaio.scoring.duplicate_evidence import duplicate_order_key
from vidaio.services.artifact_auth import MinerArtifactReceipt

CONTENT_SCORER_NAME = "validator-content-duplicate/1"
CONTENT_SCORER_PREFIX = CONTENT_SCORER_NAME + "+"
CONTENT_WITNESS_METRIC = "content_duplicate_witness"
CONTENT_EVIDENCE_RULE = "canonical_content/1"
CONTENT_SELECTION_RULE = "anchor_hash_hotkey/1"


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


def same_content(left: Any, right: Any) -> bool:
    """The D-024 pair predicate, with an exact integer 1-percent boundary."""
    if left.canonicalization_plan_digest != right.canonicalization_plan_digest:
        return False
    if left.canonical_content_digest == right.canonical_content_digest:
        return True
    if 100 * abs(left.encoded_size - right.encoded_size) > max(left.encoded_size, right.encoded_size):
        return False
    return sum((int(a, 16) ^ int(b, 16)).bit_count() <= 6
               for a, b in zip(left.content_fingerprint, right.content_fingerprint, strict=True)) >= 30


def derive_edges(roster: Sequence[ContentMember]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(roster, key=lambda member: member.uid)
    return tuple((left.uid, right.uid) for index, left in enumerate(ordered)
                 for right in ordered[index + 1:] if same_content(left, right))


def derive_components(roster: Sequence[ContentMember]) -> tuple[tuple[int, ...], ...]:
    parents = {member.uid: member.uid for member in roster}
    if len(parents) != len(roster):
        raise InvalidContentEvidence("duplicate uid in content roster")

    def root(uid: int) -> int:
        while parents[uid] != uid:
            parents[uid] = parents[parents[uid]]
            uid = parents[uid]
        return uid

    for left, right in derive_edges(roster):
        a, b = root(left), root(right)
        parents[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for uid in sorted(parents):
        groups.setdefault(root(uid), []).append(uid)
    return tuple(sorted(tuple(group) for group in groups.values()))


class ContentRoundEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    evidence_rule: Literal["canonical_content/1"] = CONTENT_EVIDENCE_RULE
    selection_rule: Literal["anchor_hash_hotkey/1"] = CONTENT_SELECTION_RULE
    fingerprint_version: Literal[1] = 1
    hamming_max: Literal[6] = 6
    matched_frames_min: Literal[30] = 30
    size_delta_percent: Literal[1] = 1
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
    def _canonical_graph(self) -> "ContentRoundEvidence":
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
        if self.edges != derive_edges(self.roster) or self.components != derive_components(self.roster):
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
    def from_json(cls, value: str) -> "ContentRoundEvidence":
        return parse_content_round(value)

    def winner(self, component: tuple[int, ...]) -> int:
        if component not in self.components:
            raise InvalidContentEvidence("component absent from full content graph")
        hotkeys = {member.uid: member.hotkey for member in self.roster}
        return min(component, key=lambda uid: duplicate_order_key(self.commitment_anchor.block_hash, hotkeys[uid]))


def _parse_canonical(value: object, model: type[BaseModel]) -> Any:
    if not isinstance(value, str) or not value or len(value.encode()) > 16 * 1024 * 1024:
        raise InvalidContentEvidence("content witness is missing or exceeds metadata limit")
    try:
        obj = json.loads(value)
        if not isinstance(obj, dict) or canonical_json_bytes(obj).decode() != value:
            raise InvalidContentEvidence("content evidence must be canonical JSON")
        return model.model_validate(obj)
    except (ValidationError, ValueError, TypeError) as exc:
        raise InvalidContentEvidence(f"invalid content evidence: {exc}") from exc


def parse_content_round(value: object) -> ContentRoundEvidence:
    return _parse_canonical(value, ContentRoundEvidence)


class ContentDuplicateWitness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    round_evidence: ContentRoundEvidence
    component_uids: tuple[int, ...]
    winner_uid: int = Field(ge=0, strict=True)
    loser_uid: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _selection(self) -> "ContentDuplicateWitness":
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


def parse_content_witness(value: object) -> ContentDuplicateWitness:
    return _parse_canonical(value, ContentDuplicateWitness)


def content_witness_from_packet(packet: Any) -> ContentDuplicateWitness:
    metrics = packet.metrics if hasattr(packet, "metrics") else packet.get("metrics", {})
    return parse_content_witness(metrics.get(CONTENT_WITNESS_METRIC))


def is_content_duplicate_identity(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith(CONTENT_SCORER_PREFIX)


def content_duplicate_identity(*, committed_scorer_version: str, track: str, scoring_config_digest: str) -> str:
    digest = sha256_hex(canonical_json_bytes({
        "convention": CONTENT_SCORER_NAME, "committed_scorer_version": committed_scorer_version,
        "track": track, "scoring_config_digest": scoring_config_digest, "evidence_rule": CONTENT_EVIDENCE_RULE,
        "selection_rule": CONTENT_SELECTION_RULE, "fingerprint_version": 1,
        "hamming_max": 6, "matched_frames_min": 30, "size_delta_percent": 1,
    }))
    return CONTENT_SCORER_PREFIX + digest[:12]


def validate_member_packet(member: ContentMember, packet: Any, context: ContentRoundEvidence) -> None:
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
        backend_versions={}, scorer_version=content_duplicate_identity(
            committed_scorer_version=context.committed_scorer_version, track=context.track,
            scoring_config_digest=context.scoring_config_digest),
    )
