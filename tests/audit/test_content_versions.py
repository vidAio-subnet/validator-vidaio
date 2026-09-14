"""Audit entrypoint dispatch and commitment binding across content-rule versions."""

from __future__ import annotations

import pytest

from tests.auditor.test_content_evidence import content_case, shared_case
from vidaio.audit.canonical import canonical_json_bytes
from vidaio.audit.recompute import (
    SCORER_VERSION_MISMATCH,
    ScorePacketShape,
    _expected_packet_scorer,
    verify_bundle,
)
from vidaio.audit.store import ArtifactKind
from vidaio.challenge import deep_reveal_verifier
from vidaio.scoring.content_duplicate_evidence import (
    CONTENT_EVIDENCE_RULE,
    CONTENT_EVIDENCE_RULE_V1,
    CONTENT_EVIDENCE_RULE_V2,
    content_duplicate_identity,
    content_duplicate_identity_v1,
    content_identity_version,
)


def test_audit_entrypoint_recomputes_mixed_history_with_one_recomputer(tmp_path, monkeypatch):
    historical = shared_case(tmp_path / "historical", evidence_rule=CONTENT_EVIDENCE_RULE_V1)
    intermediate = shared_case(tmp_path / "intermediate", evidence_rule=CONTENT_EVIDENCE_RULE_V2)
    current = shared_case(tmp_path / "current", evidence_rule=CONTENT_EVIDENCE_RULE)
    recomputer = historical.real
    monkeypatch.setattr(recomputer, "recompute", historical.byte.recompute)
    recompute_content = recomputer.recompute_content_duplicate
    dispatched = []

    def record_dispatch(bundle, artifacts, store):
        dispatched.append(content_identity_version(bundle.scorer_version))
        return recompute_content(bundle, artifacts, store)

    monkeypatch.setattr(recomputer, "recompute_content_duplicate", record_dispatch)
    for case, version in ((historical, 2), (intermediate, 3), (current, 4), (historical, 2)):
        for uid, bundle in case.bundles.items():
            original_bytes = case.store.get(bundle.score_packet)
            report = verify_bundle(
                bundle, case.store, recomputer,
                expected_bundle_digest=bundle.bundle_digest(),
                reveal_verifier=deep_reveal_verifier, strict=False,
            )
            assert report.passed, report.failures()
            assert dispatched[-1] == version
            assert case.store.get(bundle.score_packet) == original_bytes
            assert case.packets[uid].score == case.witnesses[uid].share
            checks = {check.name: check for check in report.checks}
            for name in ("packet_consistency", "committed_scorer_version",
                         "scorer_version_recompute", "score_recompute:score"):
                assert checks[name].passed and not checks[name].skipped
    assert dispatched == [2, 2, 3, 3, 4, 4, 2, 2]


@pytest.mark.parametrize("evidence_rule", [CONTENT_EVIDENCE_RULE_V1, CONTENT_EVIDENCE_RULE])
@pytest.mark.parametrize("role", ["winner", "loser"])
def test_audit_rejects_identity_from_another_content_rule(tmp_path, monkeypatch, evidence_rule, role):
    case = shared_case(tmp_path, evidence_rule=evidence_rule)
    uid = next(uid for uid, witness in case.witnesses.items() if witness.role == role)
    other_rule = CONTENT_EVIDENCE_RULE if evidence_rule == CONTENT_EVIDENCE_RULE_V1 else CONTENT_EVIDENCE_RULE_V1
    identity = content_duplicate_identity(
        committed_scorer_version=case.context.committed_scorer_version,
        track=case.context.track, scoring_config_digest=case.context.scoring_config_digest,
        evidence_rule=other_rule,
    )
    forged = case.packets[uid].model_copy(update={"scorer_version": identity})
    ref = case.store.put(canonical_json_bytes(forged.model_dump(mode="json")), ArtifactKind.SCORE_PACKET)
    bundle = case.bundles[uid].model_copy(update={"score_packet": ref, "scorer_version": identity})
    monkeypatch.setattr(case.real, "recompute", case.byte.recompute)

    report = verify_bundle(bundle, case.store, case.real, strict=False)
    commitment = next(check for check in report.checks if check.name == "committed_scorer_version")
    assert not commitment.passed
    assert commitment.code == SCORER_VERSION_MISMATCH
    assert "identity and round evidence rule differ" in commitment.reason


def test_content_identity_cannot_exchange_legacy_zero_and_share_witnesses(tmp_path):
    case = shared_case(tmp_path / "share", evidence_rule=CONTENT_EVIDENCE_RULE_V1)
    packet = next(iter(case.packets.values()))
    zero_identity = content_duplicate_identity_v1(
        committed_scorer_version=case.context.committed_scorer_version,
        track=case.context.track, scoring_config_digest=case.context.scoring_config_digest,
    )
    forged_share = ScorePacketShape.model_validate(
        packet.model_dump(mode="json") | {"scorer_version": zero_identity}
    )
    expected, error = _expected_packet_scorer(
        forged_share, committed_scorer=case.context.committed_scorer_version,
        committed_track=case.context.track,
    )
    assert expected is None and "schema-v1 witness" in error

    store, _, _, _, context, _, _, _, bundle = content_case(tmp_path / "zero")
    zero = ScorePacketShape.model_validate_json(store.get(bundle.score_packet))
    expected, error = _expected_packet_scorer(
        zero, committed_scorer=context.committed_scorer_version, committed_track=context.track,
    )
    assert expected == zero.scorer_version and not error
    share_identity = content_duplicate_identity(
        committed_scorer_version=context.committed_scorer_version,
        track=context.track, scoring_config_digest=context.scoring_config_digest,
        evidence_rule=CONTENT_EVIDENCE_RULE_V1,
    )
    expected, error = _expected_packet_scorer(
        zero.model_copy(update={"scorer_version": share_identity}),
        committed_scorer=context.committed_scorer_version, committed_track=context.track,
    )
    assert expected is None and "schema-v2 witness" in error
