"""Schema-v14 baseline registry and verified-CROWN promotion semantics."""

from __future__ import annotations

import inspect
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from vidaio.audit.store import ArtifactKind, ArtifactRef, LocalFsStore
from vidaio.competition.interfaces import logical_build_identity
from vidaio.epoch import EPOCH_LOG_SCHEMA_VERSION
from vidaio.registry.baseline import (
    SUPPORTED_TRACKS,
    BaselineRollbackError,
    GenesisBaseline,
    GenesisBaselineError,
    PendingPromotionError,
    baseline_events,
    baseline_history,
    baseline_invariant_violations,
    current_baseline,
    pending_promotion,
    require_no_pending_promotion,
    rollback_baseline,
    seed_genesis_baselines,
)
from vidaio.registry.baseline_promotion import (
    BaselineBuildError,
    BaselinePromotionPipeline,
    BaselineRerunResult,
    CrownArchiveError,
    CrownEpochNotVerifiedError,
    CrownProofMismatchError,
    ForeignEpochSchemaError,
    NonCrownEpochError,
)
from vidaio.registry.crown_source import CrownAuditItem, VerifiedCrownEpoch

NOW = datetime(2026, 8, 27, 0, 0, tzinfo=timezone.utc)


def _genesis(store: LocalFsStore) -> list[GenesisBaseline]:
    seeds: list[GenesisBaseline] = []
    for track in SUPPORTED_TRACKS:
        repo_url = f"https://example.invalid/vidaio-{track}-baseline.git"
        commit_sha = ("11" if track == "compression" else "22") * 20
        tree_sha = ("12" if track == "compression" else "23") * 20
        seeds.append(
            GenesisBaseline(
                track=track,
                artifact=store.put(
                    f"public-reference-{track}-v0".encode(),
                    ArtifactKind.SUBMISSION_ARCHIVE,
                ),
                image_digest=logical_build_identity(
                    repo_url=repo_url,
                    commit_sha=commit_sha,
                    tree_sha=tree_sha,
                ),
                provenance=store.put(
                    f"public-reference-{track}-v0-provenance".encode(),
                    ArtifactKind.MANIFEST,
                ),
                repo_url=repo_url,
                commit_sha=commit_sha,
                tree_sha=tree_sha,
            )
        )
    return seeds


def _seed(conn: sqlite3.Connection, store: LocalFsStore):
    return seed_genesis_baselines(conn, store, _genesis(store), NOW)


def _proof(
    conn: sqlite3.Connection,
    store: LocalFsStore,
    *,
    track: str = "compression",
    snapshot_byte: str = "a1",
    competition_id: str = "competition-7",
    reward_state: str = "CROWN",
    schema_version: int = EPOCH_LOG_SCHEMA_VERSION,
) -> VerifiedCrownEpoch:
    active = current_baseline(conn, track)
    assert active is not None
    submission = store.put(
        f"winner-submission-{track}-{competition_id}".encode(),
        ArtifactKind.SUBMISSION_ARCHIVE,
    )
    packet = store.put(
        f"score-packet-{track}-{competition_id}".encode(), ArtifactKind.SCORE_PACKET
    )
    bundle = store.put(
        f"audit-bundle-{track}-{competition_id}".encode(), ArtifactKind.AUDIT_BUNDLE
    )
    return VerifiedCrownEpoch(
        schema_version=schema_version,
        reward_window_state=reward_state,
        epoch_id="epoch-1007",
        snapshot_digest=snapshot_byte * 32,
        anchor_block=8_888,
        anchor_digest=snapshot_byte * 32,
        competition_id=competition_id,
        track=track,
        cycle=7,
        completed_at=NOW,
        reward_starts_at=NOW,
        reward_ends_at=NOW + timedelta(days=7),
        winner_uid=42,
        winner_hotkey=f"hk-{track}-winner",
        winner_score=0.66,
        baseline_score=0.60,
        winner_margin=0.10,
        baseline_version=active.version,
        baseline_artifact_digest=active.artifact_digest,
        winner_submission=submission,
        winner_image_digest="31" * 32,
        winner_repo_url=f"https://example.invalid/{track}-winner.git",
        winner_commit_sha="51" * 20,
        winner_tree_sha="52" * 20,
        audit_items=(
            CrownAuditItem(
                item_index=0,
                item_id="item-0",
                challenge_id="challenge-0",
                score_packet=packet,
                audit_bundle=bundle,
            ),
        ),
    )


class Source:
    def __init__(self, proof: VerifiedCrownEpoch | None) -> None:
        self.proof = proof
        self.calls = 0

    def verified_crown(self, snapshot_digest: str) -> VerifiedCrownEpoch | None:
        self.calls += 1
        return self.proof


class Runner:
    def __init__(
        self, store: LocalFsStore, *, fail: bool = False, drift_image: bool = False
    ) -> None:
        self.store = store
        self.fail = fail
        self.drift_image = drift_image
        self.calls = 0

    def build_and_rerun(self, proof, serving_baseline) -> BaselineRerunResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("fresh sandbox build failed")
        receipt = self.store.put(
            f"rerun:{proof.snapshot_digest}:{serving_baseline.version}".encode(),
            ArtifactKind.MANIFEST,
        )
        return BaselineRerunResult(
            submission_digest=proof.winner_submission.digest,
            source_image_digest=proof.winner_image_digest,
            built_image_digest=(
                "41" * 32 if self.drift_image else proof.winner_image_digest
            ),
            reproduced_score=proof.winner_score,
            receipt=receipt,
        )


def _pipeline(store: LocalFsStore, proof: VerifiedCrownEpoch | None, **runner_kwargs):
    source = Source(proof)
    runner = Runner(store, **runner_kwargs)
    return BaselinePromotionPipeline(store, source, runner), source, runner


# ---- genesis -------------------------------------------------------------------


def test_genesis_seeds_both_tracks_at_v0_and_is_idempotent(conn, store) -> None:
    seeds = _genesis(store)
    first = seed_genesis_baselines(conn, store, seeds, NOW)
    second = seed_genesis_baselines(conn, store, seeds, NOW + timedelta(days=1))

    assert set(first) == set(SUPPORTED_TRACKS)
    assert {track: row.version for track, row in first.items()} == {
        "compression": 0,
        "upscaling": 0,
    }
    assert {track: row.baseline_id for track, row in second.items()} == {
        track: row.baseline_id for track, row in first.items()
    }
    assert baseline_invariant_violations(conn) == []
    for track, row in first.items():
        assert row.status == "active" and row.source_kind == "genesis"
        assert store.get(row.artifact_ref()).startswith(b"public-reference-")
        assert store.is_released(row.artifact_ref())
        assert [event["event_type"] for event in baseline_events(conn, track)] == [
            "baseline_genesis_seeded"
        ]


def test_genesis_requires_complete_nonconflicting_archived_pair(conn, store) -> None:
    seeds = _genesis(store)
    with pytest.raises(GenesisBaselineError, match="exactly"):
        seed_genesis_baselines(conn, store, seeds[:1], NOW)
    assert conn.execute("SELECT COUNT(*) AS n FROM baselines").fetchone()["n"] == 0

    _seed(conn, store)
    conflicting = _genesis(store)
    different = store.put(b"different-v0", ArtifactKind.SUBMISSION_ARCHIVE)
    conflicting[0] = conflicting[0].model_copy(update={"artifact": different})
    with pytest.raises(GenesisBaselineError, match="different archived identity"):
        seed_genesis_baselines(conn, store, conflicting, NOW)


def test_unseeded_registry_reports_both_missing_tracks(conn) -> None:
    assert baseline_invariant_violations(conn) == [
        "track 'compression' has 0 active baselines, expected exactly 1",
        "track 'upscaling' has 0 active baselines, expected exactly 1",
    ]


# ---- proof boundary -------------------------------------------------------------


def test_unverified_foreign_and_non_crown_epochs_write_nothing(conn, store) -> None:
    _seed(conn, store)
    selector = "a1" * 32
    cases = [
        (None, CrownEpochNotVerifiedError),
        (
            _proof(conn, store, schema_version=EPOCH_LOG_SCHEMA_VERSION - 1),
            ForeignEpochSchemaError,
        ),
        (_proof(conn, store, reward_state="PODIUM"), NonCrownEpochError),
    ]
    for proof, error in cases:
        pipeline, _source, _runner = _pipeline(store, proof)
        with pytest.raises(error):
            pipeline.promote_verified_crown(conn, snapshot_digest=selector, now=NOW)
        assert pending_promotion(conn, "compression") is None
        assert [row.version for row in baseline_history(conn, "compression")] == [0]


def test_anchor_selector_margin_and_baseline_binding_are_rechecked(conn, store) -> None:
    seeded = _seed(conn, store)
    proof = _proof(conn, store)
    broken = [
        proof.model_copy(update={"anchor_digest": "ff" * 32}),
        proof.model_copy(update={"snapshot_digest": "bb" * 32}),
        proof.model_copy(update={"winner_margin": 0.2}),
        proof.model_copy(update={"baseline_version": 9}),
        proof.model_copy(update={"baseline_artifact_digest": "ee" * 32}),
    ]
    for candidate in broken:
        pipeline, _source, _runner = _pipeline(store, candidate)
        with pytest.raises(CrownProofMismatchError):
            pipeline.promote_verified_crown(
                conn, snapshot_digest=proof.snapshot_digest, now=NOW
            )
        assert pending_promotion(conn, "compression") is None
        assert current_baseline(conn, "compression") == seeded["compression"]


def test_subthreshold_margin_cannot_be_labeled_crown(conn, store) -> None:
    _seed(conn, store)
    proof = _proof(conn, store).model_copy(
        update={"winner_score": 0.624, "winner_margin": 0.04}
    )
    pipeline, _source, _runner = _pipeline(store, proof)
    with pytest.raises(NonCrownEpochError, match="threshold"):
        pipeline.promote_verified_crown(
            conn, snapshot_digest=proof.snapshot_digest, now=NOW
        )


def test_crown_promotion_uses_the_canonical_inclusive_decimal_floor(
    conn, store
) -> None:
    _seed(conn, store)
    exact_floor = _proof(conn, store).model_copy(
        update={"winner_score": 0.105, "baseline_score": 0.1, "winner_margin": 0.05}
    )
    pipeline, _source, _runner = _pipeline(store, exact_floor)

    promoted = pipeline.promote_verified_crown(
        conn, snapshot_digest=exact_floor.snapshot_digest, now=NOW
    )

    assert promoted.winner_score == 0.105
    assert promoted.winner_margin == 0.05


def test_crown_promotion_rejects_a_decimal_score_just_below_the_floor(
    conn, store
) -> None:
    _seed(conn, store)
    below_floor = _proof(conn, store).model_copy(
        update={
            "winner_score": 0.104999999999,
            "baseline_score": 0.1,
            "winner_margin": 0.04999999999,
        }
    )
    pipeline, _source, _runner = _pipeline(store, below_floor)

    with pytest.raises(NonCrownEpochError, match="threshold"):
        pipeline.promote_verified_crown(
            conn, snapshot_digest=below_floor.snapshot_digest, now=NOW
        )


def test_public_promotion_api_has_no_winner_rank_or_artifact_assertions() -> None:
    params = inspect.signature(
        BaselinePromotionPipeline.promote_verified_crown
    ).parameters
    assert set(params) == {"self", "conn", "snapshot_digest", "now"}
    assert "final_rank" not in VerifiedCrownEpoch.model_fields
    assert "manual_disqualified" not in VerifiedCrownEpoch.model_fields


# ---- promotion / latch ----------------------------------------------------------


def test_verified_crown_promotes_exact_archive_and_archives_provenance(
    conn, store
) -> None:
    seeded = _seed(conn, store)
    proof = _proof(conn, store)
    pipeline, source, runner = _pipeline(store, proof)

    promoted = pipeline.promote_verified_crown(
        conn, snapshot_digest=proof.snapshot_digest, now=NOW + timedelta(days=8)
    )

    assert promoted.version == 1 and promoted.status == "active"
    assert promoted.artifact_digest == proof.winner_submission.digest
    assert promoted.winner_hotkey == proof.winner_hotkey
    assert promoted.winner_uid == proof.winner_uid
    assert promoted.compared_baseline_version == 0
    assert promoted.compared_baseline_score == proof.baseline_score
    assert promoted.compared_baseline_digest == seeded["compression"].artifact_digest
    assert store.get(promoted.provenance_ref())
    assert store.is_released(proof.winner_submission)
    provenance = json.loads(store.get(promoted.provenance_ref()))
    assert provenance["proof"]["snapshot_digest"] == proof.snapshot_digest
    assert (
        provenance["proof"]["winner_submission"]["digest"] == promoted.artifact_digest
    )
    assert provenance["rerun"]["built_image_digest"] == promoted.image_digest
    assert [
        (row.version, row.status) for row in baseline_history(conn, "compression")
    ] == [
        (0, "superseded"),
        (1, "active"),
    ]
    assert current_baseline(conn, "upscaling").version == 0
    assert pending_promotion(conn, "compression") is None
    require_no_pending_promotion(conn, "compression")
    # Promotion is deliberately independent of the already-ended seven-day window.
    assert promoted.activated_at > proof.reward_ends_at
    assert source.calls == 1 and runner.calls == 1


def test_same_crown_is_idempotent_without_second_build(conn, store) -> None:
    _seed(conn, store)
    proof = _proof(conn, store)
    pipeline, _source, runner = _pipeline(store, proof)
    first = pipeline.promote_verified_crown(
        conn, snapshot_digest=proof.snapshot_digest, now=NOW
    )
    second = pipeline.promote_verified_crown(
        conn, snapshot_digest=proof.snapshot_digest, now=NOW + timedelta(days=1)
    )
    assert second.baseline_id == first.baseline_id
    assert runner.calls == 1
    assert len(baseline_history(conn, "compression")) == 2


def test_build_failure_keeps_prior_baseline_and_pending_interlock(conn, store) -> None:
    seeded = _seed(conn, store)
    proof = _proof(conn, store)
    pipeline, _source, runner = _pipeline(store, proof, fail=True)

    with pytest.raises(BaselineBuildError, match="fresh sandbox build failed"):
        pipeline.promote_verified_crown(
            conn, snapshot_digest=proof.snapshot_digest, now=NOW
        )

    assert current_baseline(conn, "compression") == seeded["compression"]
    assert [row.version for row in baseline_history(conn, "compression")] == [0]
    latch = pending_promotion(conn, "compression")
    assert latch is not None and latch.snapshot_digest == proof.snapshot_digest
    with pytest.raises(PendingPromotionError, match="next competition"):
        require_no_pending_promotion(conn, "compression")
    require_no_pending_promotion(conn, "upscaling")
    assert runner.calls == 1
    assert not store.is_released(proof.winner_submission)

    # Same latch is retryable; success resolves it without creating a second latch.
    runner.fail = False
    promoted = pipeline.promote_verified_crown(
        conn, snapshot_digest=proof.snapshot_digest, now=NOW + timedelta(minutes=1)
    )
    assert promoted.version == 1 and pending_promotion(conn, "compression") is None


def test_fresh_promotion_build_cannot_replace_logical_identity_with_provider_id(
    conn, store
) -> None:
    seeded = _seed(conn, store)
    proof = _proof(conn, store)
    pipeline, _source, _runner = _pipeline(store, proof, drift_image=True)

    with pytest.raises(BaselineBuildError, match="stable logical image identity"):
        pipeline.promote_verified_crown(
            conn, snapshot_digest=proof.snapshot_digest, now=NOW
        )

    assert current_baseline(conn, "compression") == seeded["compression"]
    assert pending_promotion(conn, "compression") is not None


def test_missing_winner_archive_leaves_latch_pending_and_baseline_untouched(
    conn, store
) -> None:
    seeded = _seed(conn, store)
    proof = _proof(conn, store)
    missing = ArtifactRef(
        digest="fe" * 32,
        kind=ArtifactKind.SUBMISSION_ARCHIVE,
        byte_size=123,
        backend_key="ignored/by/content/addressing",
    )
    proof = proof.model_copy(update={"winner_submission": missing})
    pipeline, _source, runner = _pipeline(store, proof)
    with pytest.raises(CrownArchiveError, match="not verifiably archived"):
        pipeline.promote_verified_crown(
            conn, snapshot_digest=proof.snapshot_digest, now=NOW
        )
    assert current_baseline(conn, "compression") == seeded["compression"]
    assert pending_promotion(conn, "compression") is not None
    assert runner.calls == 0


def test_second_crown_for_same_track_refused_while_first_is_pending(
    conn, store
) -> None:
    _seed(conn, store)
    first = _proof(conn, store, snapshot_byte="a1", competition_id="competition-a")
    second = _proof(conn, store, snapshot_byte="b2", competition_id="competition-b")
    p1, _source, _runner = _pipeline(store, first)
    p1.latch_verified_crown(conn, snapshot_digest=first.snapshot_digest, now=NOW)
    p2, _source, _runner = _pipeline(store, second)
    with pytest.raises(PendingPromotionError, match="already has pending"):
        p2.latch_verified_crown(conn, snapshot_digest=second.snapshot_digest, now=NOW)


def test_tracks_latch_and_promote_independently(conn, store) -> None:
    _seed(conn, store)
    compression = _proof(
        conn, store, track="compression", snapshot_byte="a1", competition_id="c"
    )
    upscaling = _proof(
        conn, store, track="upscaling", snapshot_byte="b2", competition_id="u"
    )
    cp, _source, _runner = _pipeline(store, compression, fail=True)
    with pytest.raises(BaselineBuildError):
        cp.promote_verified_crown(
            conn, snapshot_digest=compression.snapshot_digest, now=NOW
        )
    up, _source, _runner = _pipeline(store, upscaling)
    promoted = up.promote_verified_crown(
        conn, snapshot_digest=upscaling.snapshot_digest, now=NOW
    )
    assert promoted.track == "upscaling" and promoted.version == 1
    assert current_baseline(conn, "compression").version == 0
    assert pending_promotion(conn, "compression") is not None
    assert pending_promotion(conn, "upscaling") is None


# ---- transactional activation / rollback --------------------------------------


class Boom(RuntimeError):
    pass


class CrashingConn:
    def __init__(self, conn: sqlite3.Connection, crash_after: int) -> None:
        self.conn = conn
        self.crash_after = crash_after
        self.writes = 0

    @property
    def in_transaction(self):
        return self.conn.in_transaction

    def execute(self, sql, *args, **kwargs):
        if sql.lstrip().upper().startswith(("INSERT", "UPDATE")):
            self.writes += 1
            if self.writes == self.crash_after:
                raise Boom(f"crash at write {self.writes}")
        return self.conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.conn, name)


@pytest.mark.parametrize("crash_after", [1, 2, 3, 4, 5])
def test_activation_crash_keeps_prior_baseline_and_latch(
    conn, store, crash_after: int
) -> None:
    seeded = _seed(conn, store)
    proof = _proof(conn, store)
    pipeline, _source, _runner = _pipeline(store, proof)
    pipeline.latch_verified_crown(conn, snapshot_digest=proof.snapshot_digest, now=NOW)

    with pytest.raises(Boom):
        pipeline.promote_verified_crown(
            CrashingConn(conn, crash_after),
            snapshot_digest=proof.snapshot_digest,
            now=NOW,
        )

    assert current_baseline(conn, "compression") == seeded["compression"]
    assert [row.version for row in baseline_history(conn, "compression")] == [0]
    assert pending_promotion(conn, "compression") is not None
    assert baseline_invariant_violations(conn) == []


def test_rollback_appends_version_and_does_not_reopen_old_row(conn, store) -> None:
    _seed(conn, store)
    proof = _proof(conn, store)
    pipeline, _source, _runner = _pipeline(store, proof)
    promoted = pipeline.promote_verified_crown(
        conn, snapshot_digest=proof.snapshot_digest, now=NOW
    )
    rolled = rollback_baseline(
        conn,
        "compression",
        0,
        "promoted executable regressed under serving load",
        NOW + timedelta(hours=1),
    )
    assert rolled.version == 2 and rolled.reinstated_version == 0
    assert (
        rolled.artifact_digest
        == baseline_history(conn, "compression")[0].artifact_digest
    )
    assert [
        (row.version, row.status) for row in baseline_history(conn, "compression")
    ] == [
        (0, "superseded"),
        (1, "rolled_back"),
        (2, "active"),
    ]
    assert promoted.status == "active"  # immutable value object returned earlier
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE baselines SET status = 'active' WHERE track='compression' AND version=1"
        )
    with pytest.raises(BaselineRollbackError, match="already active"):
        rollback_baseline(conn, "compression", 2, "pointless", NOW)
    with pytest.raises(BaselineRollbackError, match="non-empty"):
        rollback_baseline(conn, "compression", 0, " ", NOW)
    assert baseline_invariant_violations(conn) == []


def test_baseline_events_are_append_only(conn, store) -> None:
    _seed(conn, store)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM baseline_events")
