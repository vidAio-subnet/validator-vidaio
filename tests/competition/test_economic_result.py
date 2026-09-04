from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import pytest

from vidaio.competition.economic_result import (
    CompetitionDedupCandidate,
    CompetitionEconomicResultError,
    aggregate_subject_scores,
    competition_dedup_losers,
    derive_competition_economics,
    derive_competition_result,
)
from vidaio.audit.commitments import COMMITMENT_DOMAIN
from vidaio.epoch.log import (
    CompetitionAuditItem,
    CompetitionAuditSubject,
    CompetitionInput,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _subject(
    subject_id: str,
    role: str,
    packet_indexes: tuple[int, ...],
    *,
    uid: int | None = None,
    hotkey: str | None = None,
) -> CompetitionAuditSubject:
    common = {
        "subject_id": subject_id,
        "role": "baseline" if role == "baseline" else "contender",
        "uid": uid,
        "hotkey": hotkey,
        "execution_image_digest": _digest(20_000 + packet_indexes[0]),
        "packet_digests": tuple(_digest(index) for index in packet_indexes),
        "audit_bundle_digests": tuple(
            _digest(index + 10_000) for index in packet_indexes
        ),
    }
    if role != "baseline":
        common.update(
            {
                "submission_archive_digest": _digest(30_000 + packet_indexes[0]),
                "submission_archive_bytes": 512,
                "repo_url": f"https://example.invalid/{subject_id}",
                "commit_sha": f"{40_000 + packet_indexes[0]:040x}",
                "tree_sha": f"{50_000 + packet_indexes[0]:040x}",
            }
        )
    subject = CompetitionAuditSubject(**common)
    return subject if role in {"baseline", "contender"} else subject.model_copy(
        update={"role": role}
    )


def _input(
    *subjects: CompetitionAuditSubject,
    item_count: int = 2,
) -> CompetitionInput:
    baseline = next(subject for subject in subjects if subject.role == "baseline")
    commitment_root = _digest(50_001)
    anchor_payload = (
        f"{COMMITMENT_DOMAIN}:competition:{commitment_root}".encode("ascii")
    )
    value = CompetitionInput(
        competition_id="competition-7",
        track="compression",
        cycle=7,
        completed_at=datetime(2026, 8, 24, 12, 30, tzinfo=timezone.utc),
        applied_at=datetime(2026, 8, 24, 12, 45, tzinfo=timezone.utc),
        manifest_digest=_digest(50_000),
        commitment_root=commitment_root,
        anchor_netuid=85,
        anchor_payload_hex=anchor_payload.hex(),
        anchor_payload_digest=hashlib.sha256(anchor_payload).hexdigest(),
        anchor_block=100,
        anchor_block_hash=_digest(50_010),
        anchor_finalized_block=101,
        baseline_version=3,
        baseline_artifact_digest=_digest(50_002),
        baseline_artifact_bytes=1024,
        baseline_execution_image_digest=baseline.execution_image_digest,
        baseline_provenance_digest=_digest(50_003),
        baseline_provenance_bytes=512,
        items=tuple(
            CompetitionAuditItem(
                challenge_id=f"challenge-{index}",
                item_id=f"item-{index}",
                threshold_commitment=_digest(60_000 + index),
            )
            for index in range(item_count)
        ),
        subjects=subjects,
    )
    return value


def _scores(competition: CompetitionInput, value: float = 0.5) -> dict[str, float]:
    return {
        digest: value
        for subject in competition.subjects
        for digest in subject.packet_digests
    }


def test_derives_means_scores_provenance_and_stable_tie_order() -> None:
    competition = _input(
        _subject("baseline", "baseline", (1, 2)),
        _subject("b-uid9", "contender", (3, 4), uid=9, hotkey="hk-b"),
        _subject("a-uid8", "contender", (5, 6), uid=8, hotkey="hk-a"),
        _subject("c-uid2", "contender", (7, 8), uid=2, hotkey="hk-c"),
    )
    scores = {
        _digest(1): 0.4,
        _digest(2): 0.6,
        _digest(3): 0.8,
        _digest(4): 1.0,
        _digest(5): 0.9,
        _digest(6): 0.9,
        _digest(7): 0.9,
        _digest(8): 0.9,
    }

    derivation = derive_competition_economics(
        competition, dict(reversed(tuple(scores.items())))
    )

    assert derivation.baseline_score == pytest.approx(0.5)
    assert derivation.aggregate_by_subject_id() == pytest.approx(
        {"baseline": 0.5, "b-uid9": 0.9, "a-uid8": 0.9, "c-uid2": 0.9}
    )
    result = derivation.result
    assert result.cycle == competition.cycle
    assert result.applied_at == competition.applied_at
    assert result.competition_id == competition.competition_id
    assert result.track == competition.track
    assert result.baseline_score == pytest.approx(0.5)
    assert result.baseline_version == 3
    assert result.baseline_artifact_digest == _digest(50_002)
    assert [(entry.hotkey, entry.uid) for entry in result.contenders] == [
        ("hk-a", 8),
        ("hk-b", 9),
        ("hk-c", 2),
    ]
    assert [entry.score for entry in result.contenders] == pytest.approx([0.9] * 3)
    assert derive_competition_result(competition, scores) == result


def test_previous_champion_role_is_rejected() -> None:
    valid = _input(
        _subject("baseline", "baseline", (10, 11)),
        _subject("previous", "contender", (12, 13), uid=3, hotkey="hk-old"),
        _subject("new", "contender", (14, 15), uid=4, hotkey="hk-new"),
    )
    competition = valid.model_copy(
        update={
            "subjects": (
                valid.subjects[0],
                valid.subjects[1].model_copy(update={"role": "previous_champion"}),
                valid.subjects[2],
            )
        }
    )
    scores = _scores(competition)
    scores.update(
        {
            _digest(10): 0.4,
            _digest(11): 0.6,
            _digest(12): 0.7,
            _digest(13): 0.7,
            _digest(14): 0.8,
            _digest(15): 0.8,
        }
    )

    with pytest.raises(
        CompetitionEconomicResultError, match="only be baseline or contender"
    ):
        derive_competition_economics(competition, scores)


def test_zero_baseline_is_preserved_for_retryable_window_failure() -> None:
    competition = _input(
        _subject("baseline", "baseline", (20, 21)),
        _subject("new", "contender", (22, 23), uid=4, hotkey="hk-new"),
    )
    scores = _scores(competition, 0.0)
    scores[_digest(22)] = scores[_digest(23)] = 0.8

    derivation = derive_competition_economics(competition, scores)

    assert derivation.baseline_score == 0.0
    assert derivation.result.contenders[0].score == pytest.approx(0.8)


def test_packet_score_coverage_must_be_exact() -> None:
    competition = _input(
        _subject("baseline", "baseline", (30, 31)),
        _subject("new", "contender", (32, 33), uid=4, hotkey="hk-new"),
    )
    complete = _scores(competition)

    missing = dict(complete)
    missing.pop(_digest(30))
    with pytest.raises(
        CompetitionEconomicResultError, match="coverage mismatch.*missing"
    ):
        derive_competition_result(competition, missing)

    extra = dict(complete, unexpected=0.5)
    with pytest.raises(
        CompetitionEconomicResultError, match="coverage mismatch.*extra"
    ):
        derive_competition_result(competition, extra)


@pytest.mark.parametrize(
    "bad",
    [True, False, None, "0.5", float("nan"), float("inf"), -0.0001, 1.0001],
)
def test_packet_scores_must_be_real_finite_unit_values(bad: object) -> None:
    competition = _input(
        _subject("baseline", "baseline", (40, 41)),
        _subject("new", "contender", (42, 43), uid=4, hotkey="hk-new"),
    )
    scores: dict[str, object] = _scores(competition)
    scores[_digest(42)] = bad

    with pytest.raises(CompetitionEconomicResultError, match="score"):
        derive_competition_result(competition, scores)  # type: ignore[arg-type]


def test_every_subject_requires_one_packet_per_committed_item() -> None:
    competition = _input(
        _subject("baseline", "baseline", (50,)),
        _subject("new", "contender", (51, 52), uid=4, hotkey="hk-new"),
        item_count=2,
    )

    with pytest.raises(CompetitionEconomicResultError, match="one per audit item"):
        aggregate_subject_scores(competition, _scores(competition))


def test_deriver_defends_baseline_and_contender_cardinality() -> None:
    valid = _input(
        _subject("baseline", "baseline", (60, 61)),
        _subject("new", "contender", (62, 63), uid=4, hotkey="hk-new"),
    )

    no_baseline = valid.model_copy(update={"subjects": valid.subjects[1:]})
    with pytest.raises(CompetitionEconomicResultError, match="exactly one.*baseline"):
        derive_competition_result(no_baseline, _scores(no_baseline))

    second_baseline = _subject("baseline-2", "baseline", (64, 65))
    two_baselines = valid.model_copy(
        update={"subjects": (*valid.subjects, second_baseline)}
    )
    with pytest.raises(CompetitionEconomicResultError, match="exactly one.*baseline"):
        derive_competition_result(two_baselines, _scores(two_baselines))

    no_contender = valid.model_copy(update={"subjects": valid.subjects[:1]})
    with pytest.raises(CompetitionEconomicResultError, match="at least one contender"):
        derive_competition_result(no_contender, _scores(no_contender))


def test_competition_dedup_is_lowest_uid_and_exempts_unspecified_ip() -> None:
    candidates = (
        CompetitionDedupCandidate(
            "winner", 2, "ck-2", "198.51.100.2", (_digest(1), _digest(2))
        ),
        CompetitionDedupCandidate(
            "same-ip", 4, "ck-4", "198.51.100.2", (_digest(3), _digest(4))
        ),
        CompetitionDedupCandidate(
            "same-coldkey", 5, "ck-2", "198.51.100.5", (_digest(5), _digest(6))
        ),
        CompetitionDedupCandidate(
            "same-output", 6, "ck-6", "198.51.100.6", (_digest(1), _digest(2))
        ),
        CompetitionDedupCandidate(
            "unspecified-a", 7, "ck-7", "0.0.0.0", (_digest(7), _digest(8))
        ),
        CompetitionDedupCandidate(
            "unspecified-b", 8, "ck-8", "0.0.0.0", (_digest(9), _digest(10))
        ),
    )

    assert competition_dedup_losers(candidates) == frozenset(
        {"same-ip", "same-coldkey", "same-output"}
    )


def test_dedup_excluded_subject_is_audited_but_not_paid() -> None:
    baseline = _subject("baseline", "baseline", (70, 71))
    winner = _subject("winner", "contender", (72, 73), uid=2, hotkey="hk-2")
    duplicate = _subject(
        "duplicate", "contender", (74, 75), uid=3, hotkey="hk-3"
    ).model_copy(update={"dedup_excluded": True})
    competition = _input(baseline, winner, duplicate)
    scores = _scores(competition, 0.9)
    scores[_digest(70)] = scores[_digest(71)] = 0.5

    derivation = derive_competition_economics(competition, scores)

    assert {entry.subject_id for entry in derivation.subject_aggregates} == {
        "baseline",
        "winner",
        "duplicate",
    }
    assert [(entry.uid, entry.hotkey) for entry in derivation.result.contenders] == [
        (2, "hk-2")
    ]
