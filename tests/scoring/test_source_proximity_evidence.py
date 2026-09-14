"""Population-relative source-proximity verdicts derive only from the committed roster."""

import math

import pytest

from tests.auditor.test_duplicate_evidence import _case, _signed_receipt
from vidaio.audit import ArtifactKind
from vidaio.scoring import ItemScore
from vidaio.scoring.compression import score_compression
from vidaio.scoring.gates import ReasonCode
from vidaio.scoring.result import compose_item_score, config_digest
from vidaio.scoring.source_proximity_evidence import (
    CHROMA_RESIDUAL_EXCESS_MIN_DB,
    SOURCE_WITNESS_METRIC,
    VMAF_RESIDUAL_EXCESS_MIN,
    InvalidSourceEvidence,
    SourceMember,
    SourceProximityWitness,
    SourceRoundEvidence,
    derive_flags,
    derive_medians,
    is_source_proximity_identity,
    median,
    mint_source_proximity_packet,
    packet_residuals,
    parse_source_round,
    rule_margins,
    source_proximity_identity,
    source_witness_from_packet,
    validate_member_packet,
)


def build_round(tmp_path, residuals):
    """residuals: {uid: (vmaf_residual, chroma_residual)} -> (context, packets, scoring)."""
    store, scoring, real, commitment, template, *_ = _case(tmp_path)
    members, packets = [], {}
    for uid, (vres, cres) in sorted(residuals.items()):
        output = store.put(b"encode-" + str(uid).encode(), ArtifactKind.MINER_OUTPUT)
        receipt = _signed_receipt(validator="validator-hotkey", miner=f"miner-{uid}", uid=uid,
            challenge_id=template.challenge_id, track="compression", anchor=template.challenge_anchor,
            input_digest=template.challenge_input.digest, input_size=template.challenge_input.byte_size,
            output_digest=output.digest, output_size=output.byte_size)
        packet = compose_item_score(item_id=receipt.metadata.task_id, challenge_id=template.challenge_id,
            track="compression", miner_hotkey=receipt.miner_hotkey, content_digest=output.digest,
            gate_passed=True, violations=[], breakdown=score_compression(candidate_bytes=output.byte_size,
                reference_bytes=1000, vmaf=95.0, config=scoring), config=scoring,
            metrics={"vmaf": 95.0, "vmaf_residual": vres, "chroma_residual": cres,
                     "psnr_uv_reference": 40.0 + cres, "psnr_uv_input": 40.0},
            scorer_version=real.scorer_version)
        ref = store.put(packet.to_json().encode(), ArtifactKind.SCORE_PACKET)
        packets[uid] = packet
        members.append(SourceMember(uid=uid, hotkey=receipt.miner_hotkey, score_packet=ref, output=output,
                                    receipt=receipt, vmaf_residual=vres, chroma_residual=cres))
    roster = tuple(members)
    vmed, cmed = derive_medians(roster)
    context = SourceRoundEvidence(challenge_id=template.challenge_id, item_id=template.challenge_id,
        track="compression", commitment_anchor=template.challenge_anchor,
        challenge_input=template.challenge_input, reference_original=template.reference_original,
        committed_scorer_version=real.scorer_version, scoring_config_digest=config_digest(scoring),
        roster=roster, vmaf_residual_median=vmed, chroma_residual_median=cmed, flagged=derive_flags(roster))
    return context, packets, scoring, store


HONEST = {1: (-0.20, -0.9), 2: (-0.10, -1.1), 3: (-0.30, -0.7), 4: (-0.05, -1.0)}


def test_median_definition():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5
    with pytest.raises(InvalidSourceEvidence):
        median([])


def test_flags_need_both_signals_and_a_population(tmp_path):
    pristine = {9: (0.55, 0.30)}          # both excesses above the margins
    luma_only = {8: (0.60, -1.0)}         # honest chroma: a clip-positive luma reading
    chroma_only = {7: (-0.10, 0.50)}
    context, *_ = build_round(tmp_path, {**HONEST, **pristine, **luma_only, **chroma_only})
    assert context.flagged == (9,)
    vmaf_excess, chroma_excess = context.excess(9)
    assert vmaf_excess >= VMAF_RESIDUAL_EXCESS_MIN and chroma_excess >= CHROMA_RESIDUAL_EXCESS_MIN_DB
    # A bimodal clip round: the honest upper mode sits far above the median on both
    # residuals but its absolute luma residual is still negative -> never flagged.
    bimodal, *_ = build_round(tmp_path / "bimodal", {
        1: (-0.50, -0.90), 2: (-0.52, -0.80), 3: (-0.48, -1.00), 4: (-0.06, -0.30), 5: (-0.09, -0.35),
        9: (0.47, 0.73)})
    assert bimodal.flagged == (9,)
    # Two items are no population: nothing is ever flagged.
    tiny, *_ = build_round(tmp_path / "tiny", {1: (-0.2, -1.0), 9: (5.0, 5.0)})
    assert tiny.flagged == ()


def test_v2_margins_spare_sharpened_honest_outputs_and_v1_evidence_still_verifies(tmp_path):
    # Round b3d81295 (2026-09-13): honest outputs at luma excess ~+0.35 / chroma ~+0.20 dB were
    # zeroed under v1; v2 needs +0.45 / +0.25 dB and an absolute luma residual above +0.15.
    context, *_ = build_round(tmp_path, {**HONEST, 8: (0.25, -0.70), 7: (0.30, -0.65)})
    assert context.rule == "source_proximity/2" and context.flagged == ()
    context, *_ = build_round(tmp_path / "abs", {**HONEST, 9: (0.14, 0.30)})   # excesses pass, absolute floor does not
    assert context.flagged == ()
    context, *_ = build_round(tmp_path / "v2", {**HONEST, 9: (0.55, 0.30)})
    assert context.flagged == (9,) and rule_margins(context.rule) == (0.45, 0.25, 0.15)
    payload = context.model_dump()
    # An archived v1 document carries the v1 margins and re-derives with them (uid 9 would also
    # flag under v1); a v2 document claiming v1 margins is rejected.
    v1 = {**payload, "rule": "source_proximity/1", "vmaf_residual_excess_min": 0.25,
          "chroma_residual_excess_min_db": 0.1}
    assert SourceRoundEvidence.model_validate(v1).flagged == (9,)
    assert parse_source_round(SourceRoundEvidence.model_validate(v1).to_json()).rule == "source_proximity/1"
    with pytest.raises(ValueError, match="margins differ"):
        SourceRoundEvidence.model_validate({**payload, "vmaf_residual_excess_min": 0.25})
    assert source_proximity_identity(committed_scorer_version="s", track="compression", scoring_config_digest="0" * 64) \
        != source_proximity_identity(committed_scorer_version="s", track="compression", scoring_config_digest="0" * 64,
                                     rule="source_proximity/1")


def test_round_evidence_rejects_tampered_decisions(tmp_path):
    context, *_ = build_round(tmp_path, {**HONEST, 9: (0.55, 0.30)})
    payload = context.model_dump()
    for change, match in (
        ({"flagged": ()}, "flagged set"),
        ({"vmaf_residual_median": context.vmaf_residual_median + 0.01}, "medians"),
        ({"item_id": "subset"}, "common challenge_id"),
        ({"roster": tuple(reversed(context.roster))}, "uid-sorted"),
    ):
        with pytest.raises(ValueError, match=match):
            SourceRoundEvidence.model_validate({**payload, **change})
    assert parse_source_round(context.to_json()) == context
    with pytest.raises(InvalidSourceEvidence):
        parse_source_round(context.to_json().replace('"flagged":[9]', '"flagged":[9] '))


def test_witness_and_minted_packet_are_canonical(tmp_path):
    context, packets, scoring, _ = build_round(tmp_path, {**HONEST, 9: (0.40, 0.30)})
    with pytest.raises(ValueError, match="flagged"):
        SourceProximityWitness(round_evidence=context, member_uid=1)
    witness = SourceProximityWitness(round_evidence=context, member_uid=9)
    packet = mint_source_proximity_packet(witness=witness, config=scoring)
    assert packet.score == 0.0 and not packet.gate_passed and packet.breakdown is None
    assert packet.violations[0].code is ReasonCode.SOURCE_PROXIMITY
    assert packet.violations[0].limit == CHROMA_RESIDUAL_EXCESS_MIN_DB
    assert packet.violations[0].measured == pytest.approx(0.30 - context.chroma_residual_median)
    assert "vmaf_residual +" in packet.violations[0].detail
    assert packet.metrics == {SOURCE_WITNESS_METRIC: witness.to_json()}
    assert packet.backend_versions == {} and packet.canonical_content_digest is None
    assert packet.miner_hotkey == "miner-9" and packet.content_digest == context.member(9).output.digest
    assert is_source_proximity_identity(packet.scorer_version)
    assert packet.scorer_version == source_proximity_identity(
        committed_scorer_version=context.committed_scorer_version, track="compression",
        scoring_config_digest=context.scoring_config_digest)
    assert source_witness_from_packet(ItemScore.from_json(packet.to_json())) == witness
    # A different committed scorer or track yields a different identity.
    assert packet.scorer_version != source_proximity_identity(
        committed_scorer_version="other", track="compression", scoring_config_digest=context.scoring_config_digest)


def test_member_packet_binding_and_residual_extraction(tmp_path):
    context, packets, scoring, _ = build_round(tmp_path, {**HONEST, 9: (0.40, 0.30)})
    member = context.member(9)
    validate_member_packet(member, packets[9], context)
    assert packet_residuals(packets[9]) == (0.40, 0.30)
    assert packet_residuals({"metrics": {"vmaf_residual": 0.1}}) is None
    assert packet_residuals({"metrics": {"vmaf_residual": True, "chroma_residual": 0.1}}) is None
    drifted = packets[9].model_copy(update={"metrics": {**packets[9].metrics, "chroma_residual": 0.31}})
    with pytest.raises(InvalidSourceEvidence, match="residuals differ"):
        validate_member_packet(member, drifted, context)
    with pytest.raises(InvalidSourceEvidence, match="not bound"):
        validate_member_packet(member, packets[1], context)
