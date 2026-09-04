from __future__ import annotations

from datetime import timedelta

from vidaio.audit.store import LocalFsStore
from vidaio.auditor import (
    AuditorConfig,
    AuditStatus,
    InMemoryBundleSource,
    ItemVerdictKind,
    REWARD_WINDOW_MISMATCH,
    SamplePolicy,
)
from vidaio.authority import EpochFinalizer
from vidaio.epoch import AuditManifest
from vidaio.tokenomics import EmissionState, RewardWindowState, TokenomicsConfig

from tests.auditor.fakes import BURN_UID, NOW, SCORER, MetagraphAuditor


def _window() -> RewardWindowState:
    return RewardWindowState(
        kind=EmissionState.PODIUM,
        starts_at=NOW - timedelta(hours=1),
        ends_at=NOW + timedelta(hours=167),
        podium_hotkeys=("hk7",),
        winner_hotkey="hk7",
        winner_uid=7,
        winner_score=0.51,
        winner_margin=0.02,
        baseline_score=0.50,
        baseline_version=0,
        baseline_artifact_digest="ab" * 32,
        source_competition_id="competition-1",
        source_track="compression",
        source_cycle=7,
        last_applied_cycle=7,
    )


def _empty_epoch(
    *,
    epoch_id: int,
    prior=None,
    prior_reward_window_state: RewardWindowState | None = None,
):
    manifest = AuditManifest(
        fold_cursors=(
            dict(prior.audit_manifest.fold_cursors) if prior is not None else {}
        )
    )
    carried = (
        prior.reward_window_state
        if prior is not None
        else prior_reward_window_state
    )
    return EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER).build_log(
        epoch_id=epoch_id,
        close_block=epoch_id * 3_600,
        snapshots=(),
        miner_census=(),
        burn_uid=BURN_UID,
        audit_manifest=manifest,
        now=NOW,
        prior_log_digest=(prior.log_digest() if prior is not None else None),
        prior_fold_cursors=(
            prior.audit_manifest.fold_cursors if prior is not None else {}
        ),
        prior_reward_window_state=carried,
    )


def _audit(current, tmp_path, *, prior):
    auditor = MetagraphAuditor(
        AuditorConfig(auditor_hotkey="auditor-reward-window", burn_uid=BURN_UID),
        InMemoryBundleSource(),
    )
    return auditor.audit_epoch(
        current,
        LocalFsStore(tmp_path / "audit"),
        SamplePolicy(sample_rate=0.0, min_samples=0),
        None,
        NOW,
        prior_log=prior,
        is_genesis=False,
    )


def test_reward_window_carries_exactly_across_an_idle_epoch(tmp_path) -> None:
    prior = _empty_epoch(epoch_id=1, prior_reward_window_state=_window())
    current = _empty_epoch(epoch_id=2, prior=prior)

    report = _audit(current, tmp_path, prior=prior)

    reward = next(
        verdict
        for verdict in report.earning_verdicts
        if verdict.item_id == "reward-window-state"
    )
    assert reward.verdict is ItemVerdictKind.PASS
    assert current.reward_window_state == prior.reward_window_state
    assert report.overall is AuditStatus.CLEAN


def test_substituted_reward_window_disputes_the_chained_epoch(tmp_path) -> None:
    prior = _empty_epoch(epoch_id=1)
    current = _empty_epoch(epoch_id=2, prior=prior)
    tampered = current.model_copy(update={"reward_window_state": _window()})

    report = _audit(tampered, tmp_path, prior=prior)

    reward = next(
        verdict
        for verdict in report.earning_verdicts
        if verdict.item_id == "reward-window-state"
    )
    assert reward.verdict is ItemVerdictKind.FAIL
    assert reward.code == REWARD_WINDOW_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_missing_predecessor_makes_reward_window_inconclusive(tmp_path) -> None:
    prior = _empty_epoch(epoch_id=1, prior_reward_window_state=_window())
    current = _empty_epoch(epoch_id=2, prior=prior)

    report = _audit(current, tmp_path, prior=None)

    reward = next(
        verdict
        for verdict in report.earning_verdicts
        if verdict.item_id == "reward-window-state"
    )
    assert reward.verdict is ItemVerdictKind.SKIP
    assert "prior epoch is unavailable" in reward.detail
    assert report.overall is AuditStatus.INCONCLUSIVE
