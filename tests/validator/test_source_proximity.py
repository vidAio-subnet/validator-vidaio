"""Source-proximity verdicts are round-relative, archived, and precede content grouping."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from tests.auditor.test_duplicate_evidence import _case, _signed_receipt
from vidaio.audit import ArtifactKind, ArtifactRef
from vidaio.scoring import ItemScore
from vidaio.scoring.compression import score_compression
from vidaio.scoring.result import compose_item_score
from vidaio.scoring.source_proximity_evidence import (
    SOURCE_WITNESS_METRIC, is_source_proximity_identity, source_witness_from_packet,
)
from vidaio.validator.inference import InferenceValidator, PacketEvidence, RoundReport


def source_round(tmp_path, residuals, *, archive=True):
    store, scoring, real, commitment, template, *_ = _case(tmp_path)
    rows = []
    for uid, pair in sorted(residuals.items()):
        output = store.put(b"encode-" + str(uid).encode(), ArtifactKind.MINER_OUTPUT)
        receipt = _signed_receipt(validator="validator-hotkey", miner=f"miner-{uid}", uid=uid,
            challenge_id=template.challenge_id, track="compression", anchor=template.challenge_anchor,
            input_digest=template.challenge_input.digest, input_size=template.challenge_input.byte_size,
            output_digest=output.digest, output_size=output.byte_size)
        metrics = {"vmaf": 95.0 - uid / 100, "candidate_bytes": output.byte_size, "reference_bytes": 1000}
        if pair is not None:
            metrics.update(vmaf_residual=pair[0], chroma_residual=pair[1])
        packet = compose_item_score(item_id=receipt.metadata.task_id, challenge_id=template.challenge_id,
            track="compression", miner_hotkey=receipt.miner_hotkey, content_digest=output.digest,
            gate_passed=True, violations=[], breakdown=score_compression(candidate_bytes=output.byte_size,
                reference_bytes=1000, vmaf=95.0 - uid / 100, config=scoring), config=scoring,
            metrics=metrics, canonical_content_digest=str(uid % 10) * 64,
            content_fingerprint=("0000000000000000",) * 32, encoded_size=output.byte_size,
            canonicalization_plan_digest="c" * 64, scorer_version=real.scorer_version)
        raw = packet.to_json(); ref = store.put(raw.encode(), ArtifactKind.SCORE_PACKET)
        rows.append(PacketEvidence(uid=uid, item_id=packet.item_id, challenge_id=packet.challenge_id,
            track="compression", miner_hotkey=packet.miner_hotkey, content_digest=output.digest,
            packet_digest=ref.digest, packet_json=raw, scorer_version=real.scorer_version, score=packet.score,
            audit_ref=ref.backend_key, challenge_input_ref=template.challenge_input.model_dump_json(),
            miner_output_ref=output.model_dump_json(), reference_original_ref=template.reference_original.model_dump_json(),
            miner_receipt_json=receipt.model_dump_json()))

    def archive_packet(raw, digest):
        if not archive:
            raise RuntimeError("audit store down")
        return store.put(raw.encode(), ArtifactKind.SCORE_PACKET).backend_key

    authority = SimpleNamespace(_store=store, scoring=scoring, scoring_pin=lambda: real.scorer_version,
        log=logging.getLogger("test-source"), _archive_packet=archive_packet)
    item = SimpleNamespace(dispatch=SimpleNamespace(challenge_id=template.challenge_id),
        commitment_anchor=template.challenge_anchor, track="compression")
    return authority, item, rows, store


HONEST = {1: (-0.20, -0.9), 2: (-0.10, -1.1), 3: (-0.30, -0.7), 4: (-0.05, -1.0)}


def test_flagged_output_is_zeroed_with_archived_witness_and_others_untouched(tmp_path):
    authority, item, rows, store = source_round(tmp_path, {**HONEST, 9: (0.45, 0.30)})
    report = RoundReport()
    for row in rows:
        report.scored[row.uid] = row.score
    result = InferenceValidator._apply_source_proximity(authority, item, list(reversed(rows)), report)
    by_uid = {row.uid: row for row in result}
    assert set(by_uid) == {1, 2, 3, 4, 9}
    for uid in HONEST:
        assert by_uid[uid] == next(row for row in rows if row.uid == uid)  # byte-identical evidence
    zero = by_uid[9]
    assert zero.score == 0.0 and is_source_proximity_identity(zero.scorer_version)
    packet = ItemScore.from_json(zero.packet_json)
    assert packet.violations[0].code.value == "SOURCE_PROXIMITY" and not packet.gate_passed
    witness = source_witness_from_packet(packet)
    assert witness.member_uid == 9 and witness.round_evidence.flagged == (9,)
    assert {m.uid for m in witness.round_evidence.roster} == {1, 2, 3, 4, 9}
    # Roster residuals are the originals' published metrics, bound to the archived packets.
    for member in witness.round_evidence.roster:
        original = ItemScore.from_json(next(row.packet_json for row in rows if row.uid == member.uid))
        assert (member.vmaf_residual, member.chroma_residual) == (
            original.metrics["vmaf_residual"], original.metrics["chroma_residual"])
        assert member.score_packet.digest == next(row.packet_digest for row in rows if row.uid == member.uid)
    stored = ArtifactRef(kind=ArtifactKind.SCORE_PACKET, digest=zero.packet_digest,
        byte_size=len(zero.packet_json.encode()), backend_key=zero.audit_ref)
    assert store.get(stored) == zero.packet_json.encode()
    assert report.zeroed == {9: "source_proximity"} and 9 not in report.scored
    assert report.non_punitive_skips == {} and report.scoring_failed == []


def test_no_flag_means_no_evidence_and_identical_packets(tmp_path):
    authority, item, rows, _ = source_round(tmp_path, {**HONEST, 8: (0.60, -1.0), 7: (-0.1, 0.5)})
    report = RoundReport()
    assert InferenceValidator._apply_source_proximity(authority, item, rows, report) == rows
    assert report.zeroed == {} and report.non_punitive_skips == {}


def test_packets_without_residuals_are_not_part_of_the_population(tmp_path):
    # Two measured packets with residuals cannot form a population, whatever the third says.
    authority, item, rows, _ = source_round(tmp_path, {1: (-0.2, -1.0), 9: (5.0, 5.0), 5: None})
    report = RoundReport()
    assert InferenceValidator._apply_source_proximity(authority, item, rows, report) == rows


def test_upscaling_and_storeless_rounds_are_untouched(tmp_path):
    authority, item, rows, _ = source_round(tmp_path, {**HONEST, 9: (0.45, 0.30)})
    report = RoundReport()
    item.track = "upscaling"
    assert InferenceValidator._apply_source_proximity(authority, item, rows, report) == rows
    item.track = "compression"; authority._store = None
    assert InferenceValidator._apply_source_proximity(authority, item, rows, report) == rows


def test_archive_failure_becomes_a_non_punitive_skip(tmp_path):
    authority, item, rows, _ = source_round(tmp_path, {**HONEST, 9: (0.45, 0.30)}, archive=False)
    report = RoundReport(); report.scored[9] = 0.5
    result = InferenceValidator._apply_source_proximity(authority, item, rows, report)
    assert {row.uid for row in result} == set(HONEST)
    assert report.non_punitive_skips == {9: "source_evidence_unavailable"}
    assert report.scoring_failed == [9] and 9 not in report.scored and report.zeroed == {}
