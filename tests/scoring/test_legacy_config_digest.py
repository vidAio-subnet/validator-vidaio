"""Witness evidence minted before the object-removal config fields existed stays verifiable.

A roll that adds ScoringConfig fields changes ``config_digest``. The next finalized epoch
re-verifies every packet committed since the last persisted epoch log, including witness
packets minted by the previous release, so their recorded digest must still be accepted.
"""
from __future__ import annotations

import hashlib

import pytest

from tests.auditor.test_content_evidence import CONTENT_EVIDENCE_RULE, content_case, shared_case
from vidaio.audit import canonical_json_bytes
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.content_duplicate_evidence import (
    InvalidContentEvidence,
    content_witness_from_packet,
    mint_content_duplicate_packet,
    mint_content_share_packet,
)
from vidaio.scoring.gates import ReasonCode, ValidityViolation
from vidaio.scoring.result import (
    REMOVAL_CONFIG_FIELDS,
    accepted_config_digests,
    compose_item_score,
    config_digest,
)


def _legacy(config: ScoringConfig) -> str:
    return hashlib.sha256(
        config.model_dump_json(exclude=set(REMOVAL_CONFIG_FIELDS)).encode("utf-8")
    ).hexdigest()


def test_removal_field_set_is_the_frozen_history():
    # The legacy digest names exactly the fields the removal release added. A later field
    # needs its own legacy entry; widening this set would break the pre-removal digest.
    assert REMOVAL_CONFIG_FIELDS == {
        "removal_weights", "removal_psnr_margin_db", "removal_psnr_span_db",
        "removal_warp_cap_factor", "removal_warp_floor", "removal_outside_tolerance",
        "removal_lpips_stride", "removal_lpips_max_side",
    }
    assert REMOVAL_CONFIG_FIELDS <= set(ScoringConfig.model_fields)


def test_accepted_digests_are_current_and_pre_removal():
    config = ScoringConfig()
    assert _legacy(config) != config_digest(config)
    assert accepted_config_digests(config) == {config_digest(config), _legacy(config)}


def test_compose_accepts_only_accepted_digests():
    config = ScoringConfig()
    common = dict(item_id="i", challenge_id="c", track="compression", gate_passed=False,
                  violations=[ValidityViolation(code=ReasonCode.DUPLICATE_CONTENT, detail="d")],
                  breakdown=None, config=config)
    assert compose_item_score(**common).scoring_config_digest == config_digest(config)
    assert compose_item_score(**common, scoring_config_digest=_legacy(config)).scoring_config_digest \
        == _legacy(config)
    with pytest.raises(ValueError, match="accepted digest"):
        compose_item_score(**common, scoring_config_digest="0" * 64)


def test_pre_removal_duplicate_witness_remints_exactly(tmp_path):
    _, scoring, _, _, context, _, _, witness, _ = content_case(tmp_path)
    old = witness.model_copy(update={
        "round_evidence": context.model_copy(update={"scoring_config_digest": _legacy(scoring)})})
    packet = mint_content_duplicate_packet(witness=old, config=scoring)
    assert packet.scoring_config_digest == _legacy(scoring)
    stored = packet.model_dump(mode="json")
    again = mint_content_duplicate_packet(witness=content_witness_from_packet(stored), config=scoring)
    assert canonical_json_bytes(again.model_dump(mode="json")) == canonical_json_bytes(stored)

    unknown = witness.model_copy(update={
        "round_evidence": context.model_copy(update={"scoring_config_digest": "0" * 64})})
    with pytest.raises(InvalidContentEvidence, match="config differs"):
        mint_content_duplicate_packet(witness=unknown, config=scoring)


@pytest.mark.parametrize("role", ["winner", "loser"])
def test_pre_removal_share_witness_remints_exactly(tmp_path, role):
    case = shared_case(tmp_path, 2, evidence_rule=CONTENT_EVIDENCE_RULE)
    uid = next(u for u, w in case.witnesses.items() if (w.member_uid == w.winner_uid) == (role == "winner"))
    witness = case.witnesses[uid]
    old = witness.model_copy(update={
        "round_evidence": witness.round_evidence.model_copy(
            update={"scoring_config_digest": _legacy(case.scoring)})})
    # the previous release stamped the same legacy digest on the winner's measured packet
    measured = (case.originals[uid].model_copy(update={"scoring_config_digest": _legacy(case.scoring)})
                if role == "winner" else None)
    packet = mint_content_share_packet(witness=old, config=case.scoring, measured_packet=measured)
    assert packet.scoring_config_digest == _legacy(case.scoring)
    again = mint_content_share_packet(
        witness=content_witness_from_packet(packet.model_dump(mode="json")),
        config=case.scoring, measured_packet=measured)
    assert canonical_json_bytes(again.model_dump(mode="json")) == canonical_json_bytes(
        packet.model_dump(mode="json"))
