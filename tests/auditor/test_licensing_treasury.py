"""Tokenomics v4: a gap epoch that pays the licensing treasury audits CLEAN, and only the
treasury the auditor itself is configured with can be paid that way."""

from __future__ import annotations

from vidaio.audit.store import LocalFsStore
from vidaio.auditor import AuditorConfig, AuditStatus, InMemoryBundleSource, SamplePolicy
from vidaio.epoch.log import AuditManifest
from vidaio.tokenomics import EMISSION_PROFILES, TokenomicsConfig
from vidaio.tokenomics.state import MinerSnapshot

from tests.auditor.fakes import BURN_UID, CLOSE_BLOCK, NOW, SCORER, MetagraphAuditor
from tests.auditor.fakes import FakeEpochFinalizer as EpochFinalizer

NO_SAMPLE = SamplePolicy(sample_rate=0.0, min_samples=0)
TREASURY = "hk-treasury"


def _v4(treasury: str | None = TREASURY) -> TokenomicsConfig:
    return TokenomicsConfig(**EMISSION_PROFILES["v4"], treasury_hotkey=treasury, result_window_hours=336.0)


def _gap_log(config: TokenomicsConfig):
    treasury_row = MinerSnapshot(
        uid=7, hotkey=TREASURY, coldkey="ck-treasury", ip="10.0.0.7",
        track="compression", accumulate_score=0.0,
    )
    return EpochFinalizer(config, scorer_version=SCORER).build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(treasury_row,),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )


def _audit(log, config: TokenomicsConfig, tmp_path):
    auditor = MetagraphAuditor(
        AuditorConfig(auditor_hotkey="hkAuditor", tokenomics=config, burn_uid=BURN_UID),
        InMemoryBundleSource(),
    )
    return auditor.audit_epoch(log, LocalFsStore(tmp_path / "s"), NO_SAMPLE, None, NOW)


def test_a_gap_epoch_paying_the_treasury_audits_clean(tmp_path) -> None:
    config = _v4()
    log = _gap_log(config)
    assert log.weight_shares == {7: 0.5, BURN_UID: 0.5}
    report = _audit(log, config, tmp_path)
    assert report.overall is AuditStatus.CLEAN, [
        (v.source, v.code, v.detail)
        for v in (*report.earning_verdicts, *report.item_verdicts)
        if v.verdict.value != "PASS"
    ]


def test_only_the_auditors_own_treasury_can_be_paid(tmp_path) -> None:
    log = _gap_log(_v4())
    report = _audit(log, _v4(treasury="hk-someone-else"), tmp_path)
    assert report.overall is AuditStatus.DISPUTED


def test_the_treasury_survives_the_published_bytes_and_v3_logs_do_not_carry_it() -> None:
    from vidaio.epoch.log import EpochLog

    log = _gap_log(_v4())
    reparsed = EpochLog.from_json(log.to_json())
    assert reparsed.treasury_hotkey == TREASURY and reparsed.log_digest() == log.log_digest()
    v3_log = EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER).build_log(
        epoch_id=100, close_block=CLOSE_BLOCK, snapshots=(),
        burn_uid=BURN_UID, audit_manifest=AuditManifest(), now=NOW,
    )
    assert b"treasury_hotkey" not in v3_log.to_json()
