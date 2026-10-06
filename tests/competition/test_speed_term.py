"""Frame-interpolation speed term: anchored rules, the score curve, the committed run time
and the digest rule that keeps every anchored manifest verifiable."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vidaio.competition import CompetitionManifest, interpolation_item_commitment, migrate
from vidaio.competition import repository as repo
from vidaio.competition.economic_result import (
    CompetitionEconomicResultError,
    speed_adjusted_score,
    speed_score,
)
from vidaio.competition.epoch_evidence import CompetitionEvidenceError, _subject_run_seconds
from vidaio.competition.manifest import ResultRules
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase
from vidaio.core import connect
from vidaio.epoch.log import CompetitionAuditSubject, CompetitionRulesInput, EpochLogInvalid

FIXTURES = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 10, 5, tzinfo=timezone.utc)
SPEED = {"speed_weight": 0.4, "speed_fast_spf": 1 / 30, "speed_slow_spf": 2.0, "speed_inserted_frames": 3600}
RESOURCES = {"cpu": 16.0, "memory_mb": 65536, "batch_timeout_seconds": 1200}


@pytest.mark.parametrize(
    "name, anchored",
    [
        # anchored before the v3 payout policy existed: no policy keys at all
        ("manifest-sn85-compression-001c.json", "0f26eee7c38bfa48f07a9f79f76e756329c6e751b44a6233dc2a15fd382157b2"),
        # anchored with the policy keys, some explicitly null
        ("manifest-sn85-removal-002.json", "9a1dc346c22de13878746b7eb1909735f7bec936ccbb7469eb305fe1ca1d3ccd"),
    ],
)
def test_anchored_manifests_reproduce_their_digests(name: str, anchored: str) -> None:
    manifest = CompetitionManifest.model_validate(json.loads((FIXTURES / name).read_text()))
    assert manifest.manifest_digest() == anchored
    # and a reload of the canonical bytes is a fixed point
    again = CompetitionManifest.model_validate_json(manifest.canonical_json())
    assert again.canonical_json() == manifest.canonical_json()


def _interp_manifest(rules: dict, resources: dict | None = RESOURCES) -> CompetitionManifest:
    commitment = interpolation_item_commitment(
        competition_id="sn85-interpolation-test", item_index=0, input_sha256="1" * 64, reference_sha256="2" * 64
    )
    return CompetitionManifest.model_validate(
        {
            "competition_id": "sn85-interpolation-test", "track": "interpolation",
            "start_time": T0 + timedelta(hours=1), "enrollment_deadline": T0 + timedelta(hours=2),
            "finalization_time": T0 + timedelta(hours=3), "end_time": T0 + timedelta(hours=4),
            "minimum_alpha_stake": 1.0,
            "scoring_factors": {"quality": 0.6, "cost_efficiency": 0.0, "length_coverage": 0.4},
            "vmaf_threshold": 90.0, "sealed_vmaf_variants": [85.0, 89.0, 93.0], "allowed_gpus": ["L40S"],
            "evaluation_item_commitments": [commitment], "evaluation_batch_size": {"min": 4, "max": 4},
            "scoring_seed_commitment": "a" * 64, "container_size_limit_gb": 25.0, "scoring_version": "v-test",
            "baseline": None, "result_rules": rules, **({} if resources is None else {"sandbox_resources": resources}),
        }
    )


def test_speed_fields_enter_the_canonical_bytes_only_when_set() -> None:
    plain = _interp_manifest({"crown_margin": 0.05, "crown_min_score": 0.85, "podium_min_score": 0.72})
    timed = _interp_manifest({"crown_margin": 0.05, "crown_min_score": 0.85, "podium_min_score": 0.72, **SPEED})
    assert "speed_weight" not in plain.canonical_json()
    rules = json.loads(timed.canonical_json())["result_rules"]
    assert rules["speed_weight"] == 0.4 and rules["speed_inserted_frames"] == 3600
    assert plain.manifest_digest() != timed.manifest_digest()


def test_speed_rules_are_all_or_nothing_and_ordered() -> None:
    with pytest.raises(ValueError, match="together"):
        ResultRules(crown_margin=0.05, speed_weight=0.4)
    with pytest.raises(ValueError, match="below"):
        ResultRules(crown_margin=0.05, **{**SPEED, "speed_fast_spf": 3.0})
    with pytest.raises(ValueError, match="sandbox_resources"):
        _interp_manifest({"crown_margin": 0.05, **SPEED}, resources=None)
    with pytest.raises(EpochLogInvalid):
        CompetitionRulesInput(crown_margin=0.05, speed_weight=0.4)


def test_speed_score_curve() -> None:
    kw = {"inserted_frames": 100, "fast_spf": 1 / 30, "slow_spf": 2.0}
    assert speed_score(100 / 30, **kw) == 1.0          # real time
    assert speed_score(1.0, **kw) == 1.0               # faster than real time
    assert speed_score(200.0, **kw) == 0.0             # 2 s per frame
    assert speed_score(10_000.0, **kw) == 0.0
    mid = math.sqrt((1 / 30) * 2.0) * 100              # geometric middle -> 0.5
    assert abs(speed_score(mid, **kw) - 0.5) < 1e-12
    values = [speed_score(t, **kw) for t in (5, 10, 20, 40, 80, 160)]
    assert values == sorted(values, reverse=True)
    # every 2x faster is worth the same
    steps = [a - b for a, b in zip(values, values[1:])]
    assert max(steps) - min(steps) < 1e-12
    assert speed_adjusted_score(0.8, 1.0, weight=0.4) == pytest.approx(0.8)
    assert speed_adjusted_score(0.8, 0.0, weight=0.4) == pytest.approx(0.48)
    with pytest.raises(CompetitionEconomicResultError):
        speed_score(float("nan"), **kw)


def test_subject_canonical_carries_run_seconds_only_when_set() -> None:
    base = dict(subject_id="contender:hk", role="contender", uid=1, hotkey="hk",
                submission_archive_digest="5" * 64, submission_archive_bytes=10, execution_image_digest="3" * 64,
                repo_url="https://github.com/a/b", commit_sha="e" * 40, tree_sha="1" * 40,
                packet_digests=("6" * 64,), audit_bundle_digests=("7" * 64,))
    assert "run_seconds" not in CompetitionAuditSubject(**base)._canonical_obj()
    assert CompetitionAuditSubject(**base, run_seconds=12.5)._canonical_obj()["run_seconds"] == 12.5
    rules = CompetitionRulesInput(crown_margin=0.05)
    assert "speed_weight" not in rules._canonical_obj()
    assert CompetitionRulesInput(crown_margin=0.05, **SPEED)._canonical_obj()["speed_slow_spf"] == 2.0


def _db_with_contender():
    manifest = _interp_manifest({"crown_margin": 0.05, **SPEED})
    conn = connect(":memory:")
    migrate(conn)
    repo.insert_competition(conn, manifest, T0)
    repo.set_status(conn, manifest.competition_id, Phase.ENROLLING, T0)
    contender = repo.enroll_contender(
        conn, manifest.competition_id, hotkey="hk", repo_url="https://github.com/a/b",
        commit_sha="e" * 40, tree_sha="1" * 40, stake=10.0, now=T0,
    )
    pers.ensure_batches(conn, manifest.competition_id, contender, n_items=12, batch_size=4, now=T0)
    rows = conn.execute("SELECT batch_id FROM batches WHERE contender_id = ? ORDER BY batch_index", (contender,)).fetchall()
    return conn, manifest.competition_id, contender, [r["batch_id"] for r in rows]


def test_completed_batches_record_their_wall_time_and_failures_cost_the_timeout() -> None:
    conn, cid, contender, batches = _db_with_contender()
    pers.complete_batch(conn, cid, batches[0], contender, [], T0, wall_seconds=31.25)
    pers.complete_batch(conn, cid, batches[1], contender, [], T0, wall_seconds=40.0)
    pers.fail_batch_contender_fault(conn, cid, batches[2], contender, "SOLUTION_EXIT", "exit 1", T0)
    assert pers.batch_wall_seconds(conn, cid, contender) == {batches[0]: 31.25, batches[1]: 40.0}
    total = _subject_run_seconds(conn, cid, contender, batch_timeout_seconds=1200, subject_id="contender:hk")
    assert total == 31.25 + 40.0 + 1200


def test_run_time_fails_closed_on_open_or_unmeasured_batches() -> None:
    conn, cid, contender, batches = _db_with_contender()
    pers.complete_batch(conn, cid, batches[0], contender, [], T0, wall_seconds=10.0)
    with pytest.raises(CompetitionEvidenceError, match="not terminal"):
        _subject_run_seconds(conn, cid, contender, batch_timeout_seconds=1200, subject_id="s")
    pers.complete_batch(conn, cid, batches[1], contender, [], T0)  # no measurement
    pers.complete_batch(conn, cid, batches[2], contender, [], T0, wall_seconds=10.0)
    with pytest.raises(CompetitionEvidenceError, match="without a measured run time"):
        _subject_run_seconds(conn, cid, contender, batch_timeout_seconds=1200, subject_id="s")
