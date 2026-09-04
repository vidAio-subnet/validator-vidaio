"""The orchestrator's half of THE SCORER-IDENTITY CONTRACT (services.protocol).

`CompetitionManifest.scoring_version` is audit-critical: the manifest digest is
built into the pre-enrollment commitment and ANCHORED before anyone enrolls, so
the competition publicly commits to the scorer that will measure it. Scoring it
with a different one produces packets no audit can reconcile — the bundle's
scorer version would name a scorer the manifest never promised.

So the orchestrator compares the live worker's advertised identity against the
PERSISTED manifest at competition start and again before SCORING. Disagreement
is an INFRA fault by the module's own classification table (the same worker
scores everyone — it is systemic), so it HALTS with a readable operator reason
rather than failing the competition or surfacing as a 409 mid-scoring.

A worker that is merely UNREACHABLE is not a disagreement: nothing is proven, so
the check defers and the scoring stage's bounded retry/halt path owns it.
"""

from __future__ import annotations

import json
import logging

import pytest

from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.interfaces import ScorePacket
from vidaio.competition.states import Phase
from vidaio.services.protocol import ScorerIdentityUnavailable, ScorerRuntimeMismatch

from orchestrator_support import (
    END,
    FINALIZATION,
    M,
    FakeScoringClient,
    build_manifest,
    events_of,
    phase,
    seed_items,
    start_and_enroll,
)

OTHER_IDENTITY = "vidaio-scorer/1+deadbeefcafe"


# --- authoring: the manifest is written against the live worker ----------------


async def test_scorer_identity_helper_reports_the_workers_own_identity(
    orchestrator_factory,
):
    """The helper callers use to AUTHOR a manifest — no guessing an identity."""
    orch = orchestrator_factory()

    assert orch.scorer_identity() == FakeScoringClient.IDENTITY


async def test_scorer_identity_refuses_a_client_that_cannot_report_one(
    orchestrator_factory,
):
    class NoDiscovery:
        conn = object()  # not None: the factory must not attach a connection

        def score_item(self, *args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("not used")

    orch = orchestrator_factory(scoring_client=NoDiscovery())

    with pytest.raises(ScorerIdentityUnavailable):
        orch.scorer_identity()


# --- agreement: business as usual ----------------------------------------------


async def test_agreeing_identity_lets_the_competition_run(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest(scoring_version=FakeScoringClient.IDENTITY)
    cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")

    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 10 * M)
    await orch.step(FINALIZATION + 15 * M)

    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    assert pers.is_halted(orch.conn, cid) is False
    assert orch.scoring_client.identity_calls >= 1


# --- disagreement: HALT, early and readably ------------------------------------


async def test_disagreement_halts_at_competition_start_not_mid_scoring(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """The whole point of checking early: nothing is built, run or scored."""
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest(scoring_version=OTHER_IDENTITY)
    cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")

    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION)

    assert pers.is_halted(orch.conn, cid) is True
    # Halted, NEVER failed: an infra blocker does not fail a competition.
    assert phase(orch, cid) is Phase.FINALIZING_SUBMISSIONS
    # The halt is CRITICAL-logged and recorded in the append-only event log, and
    # its reason names both scorers plus the operator action.
    assert any(r.levelno >= logging.CRITICAL for r in caplog.records)
    halted = events_of(orch, cid, pers.EVENT_HALTED)
    assert len(halted) == 1
    reason = halted[0]["payload_json"]
    assert OTHER_IDENTITY in reason and FakeScoringClient.IDENTITY in reason
    assert "clear_halt" in reason
    # No scoring was ever attempted against the wrong scorer.
    assert orch.scoring_client.calls == []


async def test_a_halted_competition_does_no_phase_work(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest(scoring_version=OTHER_IDENTITY)
    cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")
    await orch.step(FINALIZATION)
    assert pers.is_halted(orch.conn, cid) is True

    for at in (FINALIZATION + 2 * M, FINALIZATION + 10 * M, END + M):
        await orch.step(at)

    assert phase(orch, cid) is Phase.FINALIZING_SUBMISSIONS
    assert orch.scoring_client.calls == []


async def test_clearing_the_halt_after_aligning_the_scorer_resumes_the_pipeline(
    orchestrator_factory, fixture_repos, tmp_path
):
    """The operator action the halt reason names: fix the blocker, clear_halt."""
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest(scoring_version=OTHER_IDENTITY)
    cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")
    await orch.step(FINALIZATION)
    assert pers.is_halted(orch.conn, cid) is True

    # The committed scorer is brought up (the manifest never changes — it is
    # anchored — so it is the WORKER that has to be the one it names).
    orch.scoring_client.identity = OTHER_IDENTITY
    assert (
        orch.clear_halt(
            cid, "ops", FINALIZATION + M, reason="committed scorer restored"
        )
        is True
    )

    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 4 * M)
    await orch.step(FINALIZATION + 10 * M)
    await orch.step(FINALIZATION + 15 * M)

    assert pers.is_halted(orch.conn, cid) is False
    assert phase(orch, cid) is Phase.AWAITING_END_TIME


# --- unreachable != disagreeing -------------------------------------------------


async def test_an_unreachable_worker_defers_the_check_and_does_not_halt(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch = orchestrator_factory(repos=fixture_repos)
    orch.scoring_client.identity_unavailable = True
    manifest = build_manifest(scoring_version=FakeScoringClient.IDENTITY)
    cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")

    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)

    assert pers.is_halted(orch.conn, cid) is False
    assert phase(orch, cid) is Phase.BUILDING


async def test_a_positive_runtime_mismatch_halts_immediately(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch = orchestrator_factory(repos=fixture_repos)

    def _mismatched_runtime() -> str:
        raise ScorerRuntimeMismatch("backend_versions differs")

    orch.scoring_client.scorer_identity = _mismatched_runtime
    manifest = build_manifest(scoring_version=FakeScoringClient.IDENTITY)
    cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")

    await orch.step(FINALIZATION)

    assert pers.is_halted(orch.conn, cid) is True
    assert phase(orch, cid) is Phase.FINALIZING_SUBMISSIONS
    assert orch.scoring_client.calls == []
    reason = events_of(orch, cid, pers.EVENT_HALTED)[0]["payload_json"]
    assert "canonical scorer runtime disagreement" in reason


async def test_the_check_is_repeated_before_scoring(
    orchestrator_factory, fixture_repos, tmp_path
):
    """A worker swapped after the start check must still be caught before SCORING."""
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest(scoring_version=FakeScoringClient.IDENTITY)
    cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-b"])
    seed_items(orch, cid, tmp_path / "items")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 10 * M)
    assert phase(orch, cid) is Phase.SCORING

    orch.scoring_client.identity = OTHER_IDENTITY  # the worker was swapped
    await orch.step(FINALIZATION + 15 * M)

    assert pers.is_halted(orch.conn, cid) is True
    assert phase(orch, cid) is Phase.SCORING  # nothing scored by the wrong scorer
    assert orch.scoring_client.calls == []


# --- response binding: fail before any packet can become economic state -------


async def _drive_through_first_scoring(orch, cid: str) -> None:
    for at in (
        FINALIZATION,
        FINALIZATION + 2 * M,
        FINALIZATION + 3 * M,
        FINALIZATION + 10 * M,
        FINALIZATION + 15 * M,
    ):
        await orch.step(at)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("challenge_id", "chal-moved"),
        ("item_id", "item-moved"),
        ("track", "upscaling"),
        ("miner_hotkey", "hk-moved"),
        ("content_digest", "f" * 64),
        ("scorer_version", OTHER_IDENTITY),
        ("scoring_config_digest", "e" * 64),
    ],
)
async def test_persistence_guard_rejects_moved_packet_before_score_or_bundle_write(
    orchestrator_factory, fixture_repos, tmp_path, field, value
):
    class MovedPacketClient(FakeScoringClient):
        def score_item(self, competition_id, contender_id, item, output):
            packet = super().score_item(competition_id, contender_id, item, output)
            moved = json.loads(packet.packet_bytes)
            moved[field] = value
            return ScorePacket(
                item_id=packet.item_id,
                contender_id=packet.contender_id,
                packet_bytes=json.dumps(
                    moved, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
            )

    client = MovedPacketClient()
    orch = orchestrator_factory(
        scoring_client=client,
        repos=fixture_repos,
        scoring_retry_attempts=1,
    )
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "items")

    await _drive_through_first_scoring(orch, cid)

    assert pers.is_halted(orch.conn, cid) is True
    assert phase(orch, cid) is Phase.SCORING
    assert (
        orch.conn.execute(
            "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?",
            (cid,),
        ).fetchone()["n"]
        == 0
    )
    assert events_of(orch, cid, "audit_bundle_built") == []
    reason = events_of(orch, cid, pers.EVENT_HALTED)[0]["payload_json"]
    assert field in reason and "not bound" in reason


async def test_persistence_guard_rechecks_complete_health_backend_map(
    orchestrator_factory, fixture_repos, tmp_path
):
    expected = {
        "pieapp": "piq/0.8.0:cpu",
        "runtime": "vidaio-payout-runtime/1+" + "a" * 64,
    }

    class MovedBackendClient(FakeScoringClient):
        def __init__(self) -> None:
            super().__init__(
                backend_versions={
                    **expected,
                    "pieapp": "piq/0.8.0:cuda",
                }
            )

        def expected_backend_versions(self):
            return dict(expected)

    client = MovedBackendClient()
    orch = orchestrator_factory(
        scoring_client=client,
        repos=fixture_repos,
        scoring_retry_attempts=1,
    )
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "items")

    await _drive_through_first_scoring(orch, cid)

    assert pers.is_halted(orch.conn, cid) is True
    assert phase(orch, cid) is Phase.SCORING
    assert (
        orch.conn.execute(
            "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?",
            (cid,),
        ).fetchone()["n"]
        == 0
    )
    reason = events_of(orch, cid, pers.EVENT_HALTED)[0]["payload_json"]
    assert "backend_versions" in reason and "pieapp" in reason


async def test_persistence_guard_rehashes_output_before_packet_bundle_or_row(
    orchestrator_factory, fixture_repos, tmp_path
):
    class OutputRaceClient(FakeScoringClient):
        outputs_dir = None

        def score_item(self, competition_id, contender_id, item, output):
            packet = super().score_item(competition_id, contender_id, item, output)
            assert self.outputs_dir is not None
            path = self.outputs_dir / output.output_sha256
            path.write_bytes(path.read_bytes() + b"post-score-tamper")
            return packet

    client = OutputRaceClient()
    orch = orchestrator_factory(
        scoring_client=client,
        repos=fixture_repos,
        scoring_retry_attempts=1,
    )
    client.outputs_dir = orch.outputs_dir
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "items")

    await _drive_through_first_scoring(orch, cid)

    assert pers.is_halted(orch.conn, cid) is True
    assert phase(orch, cid) is Phase.SCORING
    assert (
        orch.conn.execute(
            "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?",
            (cid,),
        ).fetchone()["n"]
        == 0
    )
    assert events_of(orch, cid, "audit_bundle_built") == []
    reason = events_of(orch, cid, pers.EVENT_HALTED)[0]["payload_json"]
    assert "materialized miner output differs" in reason
