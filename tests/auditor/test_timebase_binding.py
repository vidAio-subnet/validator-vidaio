"""Time-base weight-input binding.

The weight re-derivation reuses an authority-attested input verbatim: ``created_at`` binds the
epoch's wall-clock time. The auditor binds it to independently-verifiable state — ``created_at``
against the epoch's CLOSE BLOCK time read from the chain ITSELF (a disagreement => DISPUTED; an
unreadable close-block time => INCONCLUSIVE, never PASS). Media-free (`sample_rate=0`).

Schema-v13 competition cycle/completion chronology is covered by the dedicated competition
evidence tests; this module isolates the general ``created_at`` versus close-block binding.
"""

from __future__ import annotations

from datetime import timedelta

from vidaio.audit.store import LocalFsStore
from vidaio.auditor import (
    Auditor,
    AuditorConfig,
    AuditStatus,
    CREATED_AT_MISMATCH,
    CREATED_AT_UNVERIFIED,
    InMemoryBundleSource,
    ItemVerdictKind,
    SamplePolicy,
)
from vidaio.tokenomics import TokenomicsConfig

from tests.auditor.fakes import (
    BURN_UID,
    CLOSE_BLOCK,
    NOW,
    fold,
    honest_log,
    make_miner,
    metagraph_chain,
    rebuild_log,
)

CFG = TokenomicsConfig()
NO_SAMPLE = SamplePolicy(sample_rate=0.0, min_samples=0)


def _created_at_verdict(report):
    return next(v for v in report.earning_verdicts if v.item_id == "created_at")


# --- created_at bound to the close-block time ----------------------------------------


def test_honest_created_at_matches_close_block_time_is_clean(tmp_path) -> None:
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [make_miner(1, fold(0.0, [0.8]))]
    from vidaio.authority import build_audit_manifest
    from tests.auditor.test_earning_state import _bundle, _scored

    b = _bundle(store, source, 1, "i1", 0.8)
    manifest = build_audit_manifest([_scored(b, 1, 0.8)], store=store)
    log = honest_log(miners, manifest, close_block=CLOSE_BLOCK)
    # Explicit chain whose block clock puts close_block at the log's (honest) created_at.
    chain = metagraph_chain(log.miners, close_block=log.close_block, close_block_time=NOW)
    report = Auditor(
        AuditorConfig(
            auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID
        ),
        source,
        chain=chain,
    ).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    assert _created_at_verdict(report).verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_backdated_created_at_disputes(tmp_path) -> None:
    """A created_at backdated 2h before the true close-block time disagrees with the chain's
    block_time(close_block) => DISPUTED."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [make_miner(1, fold(0.0, [0.8]))]
    from vidaio.authority import build_audit_manifest
    from tests.auditor.test_earning_state import _bundle, _scored

    b = _bundle(store, source, 1, "i1", 0.8)
    manifest = build_audit_manifest([_scored(b, 1, 0.8)], store=store)
    honest = honest_log(miners, manifest, close_block=CLOSE_BLOCK)
    backdated = rebuild_log(honest, created_at=NOW - timedelta(hours=2))
    # The TRUE close-block time is NOW (independent of the log's backdated created_at).
    chain = metagraph_chain(
        backdated.miners, close_block=backdated.close_block, close_block_time=NOW
    )
    report = Auditor(
        AuditorConfig(
            auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID
        ),
        source,
        chain=chain,
    ).audit_epoch(backdated, store, NO_SAMPLE, None, NOW)

    v = _created_at_verdict(report)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == CREATED_AT_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_naive_created_at_fails_closed_without_crashing(tmp_path) -> None:
    """an internal review: a timezone-NAIVE created_at (EpochLog._validate rejects it, but a log
    handed in directly / legacy bytes could carry one) must NOT crash the auditor's aware
    close-block-time subtraction — it fails closed as DISPUTED (a signed verdict, never a
    TypeError that blocks the cursor)."""
    from datetime import datetime

    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [make_miner(1, fold(0.0, [0.8]))]
    from vidaio.authority import build_audit_manifest
    from tests.auditor.test_earning_state import _bundle, _scored

    b = _bundle(store, source, 1, "i1", 0.8)
    manifest = build_audit_manifest([_scored(b, 1, 0.8)], store=store)
    log = honest_log(miners, manifest, close_block=CLOSE_BLOCK)
    # Bypass EpochLog validation (model_construct) to inject a NAIVE created_at the auditor guard
    # must survive — the exact TypeError-crash scenario the fix fails closed on.
    naive = log.model_construct(**{**dict(log), "created_at": datetime(2026, 8, 21, 12, 0, 0)})
    chain = metagraph_chain(naive.miners, close_block=naive.close_block, close_block_time=NOW)
    report = Auditor(
        AuditorConfig(
            auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID
        ),
        source,
        chain=chain,
    ).audit_epoch(naive, store, NO_SAMPLE, None, NOW)

    v = _created_at_verdict(report)
    assert v.verdict is ItemVerdictKind.FAIL and v.code == CREATED_AT_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_unreadable_close_block_time_is_inconclusive(tmp_path) -> None:
    """The close-block time cannot be read (the chain exposes no block clock) => the created_at
    binding is INCONCLUSIVE (fail-closed), never a PASS on the authority's self-attested time."""
    store = LocalFsStore(tmp_path / "s")
    source = InMemoryBundleSource()
    miners = [make_miner(1, fold(0.0, [0.8]))]
    from vidaio.authority import build_audit_manifest
    from tests.auditor.test_earning_state import _bundle, _scored

    b = _bundle(store, source, 1, "i1", 0.8)
    manifest = build_audit_manifest([_scored(b, 1, 0.8)], store=store)
    log = honest_log(miners, manifest, close_block=CLOSE_BLOCK)
    # A metagraph chain WITHOUT a block clock: identities bind, but block_time returns None.
    chain = metagraph_chain(log.miners, close_block=log.close_block)
    chain.block_time_anchor = None
    report = Auditor(
        AuditorConfig(
            auditor_hotkey="hkAuditor", tokenomics=CFG, burn_uid=BURN_UID
        ),
        source,
        chain=chain,
    ).audit_epoch(log, store, NO_SAMPLE, None, NOW)

    v = _created_at_verdict(report)
    assert v.verdict is ItemVerdictKind.SKIP and v.code == CREATED_AT_UNVERIFIED
    assert report.overall is AuditStatus.INCONCLUSIVE


# Competition-specific application/completion chronology lives in the schema-v14 competition
# audit modules; this file intentionally contains only the general epoch time-base cases.
