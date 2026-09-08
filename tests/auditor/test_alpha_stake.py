"""Stake eligibility must bind even zero-weight and census-only identities."""

from dataclasses import replace

import pytest

from vidaio.audit.store import LocalFsStore
from vidaio.auditor import AuditorConfig, AuditStatus, InMemoryBundleSource
from tests.auditor.fakes import FakeChronologyAuditor as Auditor
from vidaio.epoch.log import MinerCensusEntry, weight_vector_digest
from vidaio.tokenomics import TokenomicsConfig, build_weight_vector, quantize_u16
from tests.auditor.fakes import BURN_UID, NOW, metagraph_chain, rebuild_log
from tests.auditor.test_census import NO_SAMPLE, _honest


@pytest.mark.parametrize("forged_burn", [False, True])
def test_false_alpha_in_economic_snapshot_and_census_is_disputed(tmp_path, forged_burn):
    store = LocalFsStore(tmp_path / "store")
    source = InMemoryBundleSource()
    log, _, original = _honest(store, source, {1: 0.8})
    true_miners = [replace(original[0], alpha_stake=10.0)]
    claimed_miners = [replace(original[0], alpha_stake=0.0 if forged_burn else 9.0)]
    config = TokenomicsConfig(payout_min_alpha_stake=5)
    shares = build_weight_vector(config, claimed_miners, burn_uid=BURN_UID)
    u16 = quantize_u16(shares)
    claimed = rebuild_log(
        log, miners=tuple(claimed_miners),
        miner_census=tuple(MinerCensusEntry.from_miner(m) for m in claimed_miners),
        weight_shares=shares, weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16),
    )
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor", tokenomics=config, burn_uid=BURN_UID),
        source, chain=metagraph_chain(true_miners),
    )
    report = auditor.audit_epoch(claimed, store, NO_SAMPLE, None, NOW)
    assert report.overall is AuditStatus.DISPUTED
    assert any(v.source == "snapshot" and v.verdict.value == "FAIL" for v in report.earning_verdicts)
    assert any(v.source == "census" and v.verdict.value == "FAIL" for v in report.earning_verdicts)


def test_census_only_alpha_is_verified_when_default_floor_is_zero(tmp_path):
    store = LocalFsStore(tmp_path / "store")
    source = InMemoryBundleSource()
    log, _, original = _honest(store, source, {1: 0.8})
    other = replace(original[0], uid=2, hotkey="hk2", coldkey="ck2", ip="10.0.0.2", alpha_stake=10)
    census = tuple(log.miner_census) + (MinerCensusEntry.from_miner(replace(other, alpha_stake=0)),)
    manifest = log.audit_manifest.model_copy(update={
        "fold_cursors": {**log.audit_manifest.fold_cursors, 2: None},
    })
    claimed = rebuild_log(log, miner_census=census, audit_manifest=manifest)
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor", burn_uid=BURN_UID),
        source, chain=metagraph_chain(original + [other]),
    )
    report = auditor.audit_epoch(claimed, store, NO_SAMPLE, None, NOW)
    assert report.overall is AuditStatus.DISPUTED
    assert any(v.uid == 2 and v.source == "census" and v.verdict.value == "FAIL" for v in report.earning_verdicts)


def test_auditor_rederives_with_the_archived_floor_not_its_own_config(tmp_path):
    """D-025: the vector follows from the floor the log ARCHIVED; policy drift is report-only."""
    from vidaio.auditor.report import PAYOUT_POLICY_MISMATCH

    store = LocalFsStore(tmp_path / "store")
    source = InMemoryBundleSource()
    log, _, original = _honest(store, source, {1: 0.8, 2: 0.9})
    miners = [replace(original[0], alpha_stake=10.0), replace(original[1], alpha_stake=1.0)]
    archived = TokenomicsConfig(payout_min_alpha_stake=5.0)
    shares = build_weight_vector(archived, miners, burn_uid=BURN_UID)
    assert shares.get(2, 0.0) == 0.0 and shares[1] > 0.0  # the floor excludes uid 2
    u16 = quantize_u16(shares)
    claimed = rebuild_log(
        log, miners=tuple(miners),
        miner_census=tuple(MinerCensusEntry.from_miner(m) for m in miners),
        weight_shares=shares, weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16),
        payout_min_alpha_stake=5.0,
    )
    chain = metagraph_chain(miners)

    # An auditor still configured with floor 0 re-derives from the archived 5.0: honest.
    drifted = Auditor(
        AuditorConfig(auditor_hotkey="auditor", burn_uid=BURN_UID), source, chain=chain,
    )
    report = drifted.audit_epoch(claimed, store, NO_SAMPLE, None, NOW)
    assert report.overall is AuditStatus.CLEAN
    assert report.weight_verdict.verdict.value == "PASS"
    assert report.weight_verdict.code == PAYOUT_POLICY_MISMATCH

    # An auditor whose policy matches the archived floor reports no note.
    aligned = Auditor(
        AuditorConfig(auditor_hotkey="auditor", tokenomics=archived, burn_uid=BURN_UID),
        source, chain=chain,
    )
    aligned_report = aligned.audit_epoch(claimed, store, NO_SAMPLE, None, NOW)
    assert aligned_report.overall is AuditStatus.CLEAN
    assert aligned_report.weight_verdict.code == ""

    # A log that archives floor 0 but published the floor-5 vector is a provable fault.
    lying = rebuild_log(claimed, payout_min_alpha_stake=0.0)
    lying_report = aligned.audit_epoch(lying, store, NO_SAMPLE, None, NOW)
    assert lying_report.overall is AuditStatus.DISPUTED
    assert lying_report.weight_verdict.verdict.value == "FAIL"
