"""The validator's OWN-AUDIT verdict builder.

Production uses report-only mode after set_weights: every CLEAN / DISPUTED /
INCONCLUSIVE result is reported for manual remediation and never gates emissions. The
legacy contiguous classifier remains covered because its ledger/cursor integrity is
still useful to historical tooling, but it is not wired into the submission path.
"""

from __future__ import annotations

from vidaio.auditor import AuditorConfig, AuditStatus, SamplePolicy
from vidaio.auditor.report import ItemVerdictKind
from vidaio.authority import build_audit_manifest
from vidaio.epoch import EpochLog, MinerCensusEntry, weight_vector_digest
from vidaio.tokenomics import TokenomicsConfig
from vidaio.tokenomics.quantize import quantize_u16
from vidaio.weightsetter.own_audit import OwnAuditGate

from tests.weightsetter.weightsetter_support import (
    NOW,
    SCORER,
    AuthorityHarness,
    make_item,
    make_miner,
)


class _UnsupportedRecomputer:
    """A recomputer that declares every item un-recomputable (e.g. GPU unavailable).

    The auditor's honest-refusal probe reads ``unsupported_reason`` and SKIPs the item
    before any media work — the faithful "media cannot be recomputed" case, which yields
    INCONCLUSIVE (not a false CLEAN, not a false DISPUTED)."""

    def unsupported_reason(self, bundle, artifacts, *, track=None) -> str:
        return "media backend unavailable (test)"

    def recompute(self, bundle, artifacts):  # pragma: no cover - never reached after SKIP
        raise AssertionError("recompute must not run once the item is probed unsupported")


def _gate(
    store,
    *,
    sample_rate: float = 0.0,
    recomputer: object | None = None,
    is_genesis_for=None,
    prior_log_for=None,
    ledger=None,
    burn_uid: int = 999,
    all_items: bool = False,
    report_only: bool = False,
) -> OwnAuditGate:
    from tests.auditor.fakes import MetagraphAuditor

    # MetagraphAuditor auto-wires the close-block metagraph from the reviewed log's honest
    # identities so the own-audit gate clears honest epochs (CLEAN).
    # burn_uid: the CANONICAL burn recipient the gate's audit binds the log's burn uid to
    # — the harness's honest burn epochs use 999.
    auditor = MetagraphAuditor.over_store(
        AuditorConfig(
            auditor_hotkey="v-own", backend="fake", tokenomics=TokenomicsConfig(),
            burn_uid=burn_uid,
        ),
        store,
    )
    return OwnAuditGate(
        auditor=auditor,
        store=store,
        # min_samples=0 so rate 0.0 samples no media (earning + weight only).
        policy=SamplePolicy(
            sample_rate=sample_rate, min_samples=0, all_items=all_items
        ),
        recomputer=recomputer,
        is_genesis_for=is_genesis_for,
        prior_log_for=prior_log_for,
        ledger=ledger,
        report_only=report_only,
    )


async def test_gate_clears_an_honest_burn_epoch(tmp_path) -> None:
    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        finalized = await a.finalize(
            epoch_id=1, close_block=3599, miners=[], items=None
        )
        verdict = _gate(a.store).review(finalized.log, now=NOW)
        assert verdict.ok is True
        assert verdict.status is AuditStatus.CLEAN
    finally:
        a.close()


async def test_gate_reports_disputed_burn_to_noncanonical_uid(tmp_path) -> None:
    """an internal review: the own-audit reviewer must DISPUTE a log that burns to a
    NON-CANONICAL uid — an untrusted authority anchoring an empty burn to a beneficiary it
    controls. The gate's audit resolves the canonical burn uid (999) from config, INDEPENDENTLY
    of the log, and reports a burn to any other uid as DISPUTED (never false CLEAN). The
    finding is report-only and does not block the authenticated authority submission."""
    a = AuthorityHarness(tmp_path, burn_uid=5)  # the authority burns to a non-canonical uid
    try:
        finalized = await a.finalize(
            epoch_id=1, close_block=3599, miners=[], items=None
        )
        assert finalized.log.burn_uid == 5
        # The gate's canonical burn uid is 999 (its configured value); 5 is NOT canonical.
        verdict = _gate(a.store, burn_uid=999).review(finalized.log, now=NOW)
        assert verdict.ok is False
        assert verdict.status is AuditStatus.DISPUTED
    finally:
        a.close()


async def test_gate_clears_honest_epoch_when_media_not_sampled(tmp_path) -> None:
    """sample_rate 0 => earning re-fold + weight re-derivation only (both PASS) => CLEAN."""
    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        items = [make_item(1, a.store), make_item(2, a.store)]
        finalized = await a.finalize(
            epoch_id=2, close_block=7199,
            miners=[make_miner(1, 0.9), make_miner(2, 0.9)], items=items,
        )
        verdict = _gate(a.store, sample_rate=0.0).review(finalized.log, now=NOW)
        assert verdict.ok is True
        assert verdict.status is AuditStatus.CLEAN
    finally:
        a.close()


async def test_gate_reports_inconclusive_when_media_cannot_recompute(tmp_path) -> None:
    """sample_rate 1.0 but the media backend cannot recompute => all SKIP => INCONCLUSIVE."""
    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        items = [make_item(1, a.store), make_item(2, a.store)]
        finalized = await a.finalize(
            epoch_id=3, close_block=10799,
            miners=[make_miner(1, 0.9), make_miner(2, 0.9)], items=items,
        )
        verdict = _gate(
            a.store, sample_rate=1.0, recomputer=_UnsupportedRecomputer()
        ).review(finalized.log, now=NOW)
        assert verdict.ok is False
        assert verdict.status is AuditStatus.INCONCLUSIVE
    finally:
        a.close()


async def test_all_items_submit_gate_recomputes_more_than_fifty_items(tmp_path) -> None:
    from vidaio.audit.recompute import StaticRecomputer

    class CountingRecomputer:
        def __init__(self) -> None:
            self.calls = 0
            self._inner = StaticRecomputer(
                {"compression_rate": 0.125, "vmaf": 93.42, "final_score": 0.8},
                SCORER,
                score=0.8,
            )

        def recompute(self, bundle, artifacts):
            self.calls += 1
            return self._inner.recompute(bundle, artifacts)

    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        uids = range(1, 52)
        recomputer = CountingRecomputer()
        finalized = await a.finalize(
            epoch_id=1,
            close_block=3599,
            miners=[make_miner(uid) for uid in uids],
            items=[make_item(uid, a.store) for uid in uids],
        )
        verdict = _gate(
            a.store,
            sample_rate=1.0,
            all_items=True,
            recomputer=recomputer,
        ).review(finalized.log, now=NOW)
        assert recomputer.calls == 51
        # Every packet is identity-bound to its archived output, and none of the
        # 51 may escape the full own-audit report through SamplePolicy's ordinary 50-item
        # cap. Placeholder proof fields can remain INCONCLUSIVE, but the fixture
        # must contain no contradictory evidence.
        assert verdict.status is not AuditStatus.DISPUTED
    finally:
        a.close()


async def test_gate_reports_non_genesis_log_with_missing_prior_digest(tmp_path) -> None:
    """an internal review: the own-audit reviewer decides genesis from an authenticated,
    operator-set genesis floor, never from a withholdable `_FINALIZED` absence. A NON-genesis epoch whose
    prior_log_digest is None (the omitted-digest earning-carry-in reset) must be reported
    DISPUTED, exactly as the dedicated auditor loop reports it. This does not block submission.
    The authenticated floor here is epoch 0, so epoch 5 is NON-genesis."""
    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        finalized = await a.finalize(
            epoch_id=5, close_block=21599,
            miners=[make_miner(1, 0.9)], items=[make_item(1, a.store)],
        )
        assert finalized.log.prior_log_digest is None  # the reset the gate must catch

        # genesis_floor = 0 (the authenticated, operator-configured floor) => epoch 5 is NOT
        # genesis => the missing prior_log_digest is a broken chain => DISPUTED report.
        held = _gate(a.store, is_genesis_for=lambda log: log.epoch_id == 0).review(
            finalized.log, now=NOW
        )
        assert held.ok is False
        assert held.status is AuditStatus.DISPUTED

        # Contrast: the SAME log IS the genuine genesis when the authenticated floor names it
        # (epoch_id == floor) — then a None prior_log_digest is legitimate and the gate clears.
        cleared = _gate(a.store, is_genesis_for=lambda log: log.epoch_id == 5).review(
            finalized.log, now=NOW
        )
        assert cleared.ok is True
        assert cleared.status is AuditStatus.CLEAN
    finally:
        a.close()


async def _finalize_chain(a):
    """A genesis epoch (1) and a CHAINED successor (2) whose nonzero carry-in equals
    epoch-1's stated accumulate for uid 1 (a self-consistent chain — the auditor's earning
    re-fold PASSES both). Returns (genesis_log, successor_log). Both use the SAME hotkey
    (hk1) so the (uid,hotkey) carry does not reset."""
    from vidaio.tokenomics import MinerSnapshot
    from vidaio.tokenomics.ewma import accumulate

    decay = TokenomicsConfig().ewma_decay
    prior_acc = accumulate(0.0, 0.8, decay)  # epoch-1 accumulate for uid 1 (== make_miner)
    succ_acc = accumulate(prior_acc, 0.8, decay)  # epoch-2 fold of the same cycle over it

    genesis = await a.finalize(
        epoch_id=1, close_block=3599, miners=[make_miner(1, 0.9)], items=[make_item(1, a.store)],
    )
    successor = await a.finalize(
        epoch_id=2, close_block=7199,
        miners=[
            MinerSnapshot(
                uid=1, hotkey="hk1", coldkey="ck1", ip="10.0.0.1",
                track="compression", accumulate_score=succ_acc,
            )
        ],
        # A NEW cycle at a strictly higher committed key (seq=1) — a genuine second earning,
        # not a re-fold of epoch-1's packet (the monotonic ordering_key invariant, round-22 #1).
        items=[make_item(1, a.store, seq=1)],
        prior_accumulate={1: prior_acc},
        prior_log_digest=genesis.log.log_digest(),
    )
    return genesis.log, successor.log


async def test_gate_reports_nonzero_carry_in_with_unaudited_predecessor(tmp_path) -> None:
    """an internal review (CRITICAL): a self-consistent epoch whose NONZERO carry-in chains
    to a predecessor the reviewer NEVER own-audited CLEAN must be INCONCLUSIVE. The earning re-fold PASSES
    (the number chains to the prior log), yet the predecessor could carry an INJECTED
    accumulator — so a fresh ledger cannot vouch for it. This audit status is reported, not
    enforced against weight submission."""
    from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger

    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        genesis_log, successor_log = await _finalize_chain(a)
        priors = {1: genesis_log, 2: successor_log}
        # Fresh ledger: the predecessor (epoch 1) was never recorded own-audited CLEAN.
        ledger = OwnAuditLedger.open(":memory:")
        gate = _gate(
            a.store,
            prior_log_for=lambda log: priors.get(log.epoch_id - 1),
            is_genesis_for=lambda log: log.epoch_id == 1,
            ledger=ledger,
        )
        verdict = gate.review(successor_log, now=NOW)
        assert verdict.ok is False
        assert verdict.status is AuditStatus.INCONCLUSIVE
    finally:
        a.close()


async def test_report_only_returns_auditor_result_without_legacy_ledger_downgrade(
    tmp_path,
) -> None:
    """Production reporting never lets legacy CLEAN-ledger state suppress an epoch.

    The underlying signed Auditor report is CLEAN because the successor's carry-in
    correctly chains to the supplied predecessor. A fresh legacy ledger downgrades that
    to INCONCLUSIVE, but report-only mode returns the actual report result directly and
    does not mutate the ledger.
    """
    from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger

    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        genesis_log, successor_log = await _finalize_chain(a)
        priors = {1: genesis_log, 2: successor_log}
        ledger = OwnAuditLedger.open(":memory:")
        verdict = _gate(
            a.store,
            prior_log_for=lambda log: priors.get(log.epoch_id - 1),
            is_genesis_for=lambda log: log.epoch_id == 1,
            ledger=ledger,
            report_only=True,
        ).review(successor_log, now=NOW)
        assert verdict.ok is True
        assert verdict.status is AuditStatus.CLEAN
        assert verdict.status is verdict.report.overall
        assert not ledger.is_clean(successor_log.epoch_id, successor_log.log_digest())
    finally:
        a.close()


async def test_gate_clears_chain_when_predecessor_recorded_clean(tmp_path) -> None:
    """an internal review: a proper chain — the gate clears the GENESIS epoch (recording it
    own-audited CLEAN), then clears the successor because its predecessor IS a recorded
    CLEAN digest. Genesis + zero carry-in also clears with no predecessor required."""
    from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger

    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        genesis_log, successor_log = await _finalize_chain(a)
        priors = {1: genesis_log, 2: successor_log}
        ledger = OwnAuditLedger.open(":memory:")
        gate = _gate(
            a.store,
            prior_log_for=lambda log: priors.get(log.epoch_id - 1),
            is_genesis_for=lambda log: log.epoch_id == 1,
            ledger=ledger,
        )
        # Genesis (zero carry-in) clears and is RECORDED CLEAN.
        g = gate.review(genesis_log, now=NOW)
        assert g.ok is True and g.status is AuditStatus.CLEAN
        assert ledger.is_clean(genesis_log.epoch_id, genesis_log.log_digest())
        # Successor's nonzero carry-in chains to the now-recorded predecessor ⇒ clears.
        s = gate.review(successor_log, now=NOW)
        assert s.ok is True and s.status is AuditStatus.CLEAN
        assert ledger.is_clean(successor_log.epoch_id, successor_log.log_digest())
    finally:
        a.close()


async def test_gate_clears_genesis_with_zero_carry_in(tmp_path) -> None:
    """an internal review: genesis + zero carry-in needs NO recorded predecessor — it clears
    on a fresh (empty) ledger and is recorded so the next epoch can chain onto it."""
    from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger

    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        finalized = await a.finalize(
            epoch_id=1, close_block=3599,
            miners=[make_miner(1, 0.9)], items=[make_item(1, a.store)],
        )
        ledger = OwnAuditLedger.open(":memory:")
        verdict = _gate(
            a.store, is_genesis_for=lambda log: log.epoch_id == 1, ledger=ledger
        ).review(finalized.log, now=NOW)
        assert verdict.ok is True
        assert verdict.status is AuditStatus.CLEAN
        assert ledger.is_clean(finalized.log.epoch_id, finalized.log.log_digest())
    finally:
        a.close()


#): the gate covers IMPLICIT positive carry-forward too ----


def _fold(prior: float, scores) -> float:
    from vidaio.tokenomics.ewma import accumulate

    decay = TokenomicsConfig().ewma_decay
    v = prior
    for s in scores:
        v = accumulate(v, s, decay)
    return v


def _persisted_bundle(store, uid: int, item_id: str, score: float):
    """A resolvable committed bundle in the STORE (so the gate's over_store auditor resolves it)."""
    from tests.auditor.fakes import make_fake_bundle, make_packet
    from vidaio.auditor.service import persist_bundle

    packet = make_packet(
        challenge_id="c1", item_id=item_id, miner_hotkey=f"hk{uid}", score=score,
        cycle_sequence=0, metrics={"compression_rate": 0.1, "vmaf": 93.0, "final_score": score},
    )
    b = make_fake_bundle(
        store, challenge_id="c1", item_id=item_id, miner_hotkey=f"hk{uid}",
        packet=packet, dispatch_ordering_key=0,
    )
    persist_bundle(store, b)
    return b


def _scored(b, uid: int, score: float):
    from vidaio.authority import ScoredItem

    return ScoredItem(
        uid=uid, hotkey=f"hk{uid}", challenge_id=b.challenge_id, item_id=b.item_id,
        bundle_digest=b.bundle_digest(), packet_digest=b.score_packet.digest,
        committed_track="compression", score=score, cycle_sequence=0,
    )


def _implicit_carry_logs(store):
    """(prior E1, current E2) where a ZERO-weight uid carries a POSITIVE accumulator forward
    IMPLICITLY — a positive `accumulate_score` with NO `EarningInput` this epoch — while every
    E2 EarningInput has a ZERO explicit carry-in (the E2 tops are NEW uids scored this epoch).

    So `_has_nonzero_carry_in(E2)` is driven ONLY by the implicit carry-forward (uid 6), the
    exact case round-12's earning_inputs-only check missed. Built with `top_n_per_track=5`, so
    the five fresh tops (uids 11..15 @0.9) fill the podium and the carried-forward uid 6 (@0.1)
    ranks below the cutoff => zero weight but a positive accumulator with no EarningInput."""
    from tests.auditor.fakes import BURN_UID, make_miner as fake_miner
    from vidaio.authority import build_audit_manifest
    from vidaio.authority.finalizer import EpochFinalizer

    fin = EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER)
    # E1 (genesis): uids 1..5 @0.9 + uid 6 @0.1 (a below-cutoff loser that ends E1 with a
    # positive accumulator). Each has an EarningInput folded from a zero carry-in.
    e1_items, e1_miners = [], []
    for uid in range(1, 6):
        b = _persisted_bundle(store, uid, f"e1i{uid}", 0.9)
        e1_items.append(_scored(b, uid, 0.9))
        e1_miners.append(fake_miner(uid, _fold(0.0, [0.9])))
    b6 = _persisted_bundle(store, 6, "e1i6", 0.1)
    e1_items.append(_scored(b6, 6, 0.1))
    carried = _fold(0.0, [0.1])
    e1_miners.append(fake_miner(6, carried))
    e1_manifest = build_audit_manifest(e1_items, store=store)
    e1 = fin.build_log(
        epoch_id=1, close_block=3599, snapshots=tuple(e1_miners), burn_uid=BURN_UID,
        audit_manifest=e1_manifest, now=NOW,
    )
    # E2: NEW tops uids 11..15 @0.9 (fresh, ZERO carry-in) + uid 6 carries its accumulator
    # forward with NO new item (implicit carry-forward, stays a zero-weight loser).
    e2_items, e2_miners = [], []
    for uid in range(11, 16):
        b = _persisted_bundle(store, uid, f"e2i{uid}", 0.9)
        e2_items.append(_scored(b, uid, 0.9))
        e2_miners.append(fake_miner(uid, _fold(0.0, [0.9])))
    e2_miners.append(fake_miner(6, carried))  # positive accumulator, NO EarningInput
    e2_manifest = build_audit_manifest(
        e2_items, store=store, prior_fold_cursors=e1_manifest.fold_cursors
    )
    e2 = fin.build_log(
        epoch_id=2, close_block=7199, snapshots=tuple(e2_miners), burn_uid=BURN_UID,
        audit_manifest=e2_manifest, now=NOW,
        prior_log_digest=e1.log_digest(),
    )
    return e1, e2


async def test_has_nonzero_carry_in_flags_implicit_carry_forward(tmp_path) -> None:
    """an internal review: the depends-on-predecessor predicate must be True for an IMPLICIT
    carry-forward (a positive accumulator with no EarningInput) even when EVERY EarningInput
    has a zero explicit carry-in — and False for a genuinely fresh epoch."""
    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        e1, e2 = _implicit_carry_logs(a.store)
        # E2: every EarningInput has a zero carry-in, so the round-12 earning_inputs-only check
        # would (wrongly) say False; the implicit uid-6 carry-forward makes the fixed check True.
        assert all(
            ei.prior_accumulate_score == 0.0
            for ei in e2.audit_manifest.earning_inputs.values()
        )
        assert e2.weight_shares.get(6, 0.0) == 0.0
        assert e2.audit_manifest.earning_for(6) is None
        assert OwnAuditGate._has_nonzero_carry_in(e2) is True
        # E1 (genesis) — uid 6 has an EarningInput folded THIS epoch (fresh), no carried value.
        assert OwnAuditGate._has_nonzero_carry_in(e1) is False
    finally:
        a.close()


async def test_gate_reports_implicit_carry_forward_with_unaudited_predecessor(tmp_path) -> None:
    """an internal review (CRITICAL): a zero-weight miner with a POSITIVE accumulator and NO
    EarningInput (an IMPLICIT carry-forward) depends on its predecessor. With a FRESH ledger
    (predecessor never own-audited CLEAN) the reviewer must return INCONCLUSIVE — the same
    conservative audit classification
    the explicit carry-in gets — so an unaudited predecessor cannot inject an accumulator that
    an implicit carry re-attributed past round-12's earning_inputs-only check. It remains
    report-only for weight-setting."""
    from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger

    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        e1, e2 = _implicit_carry_logs(a.store)
        priors = {1: e1, 2: e2}
        ledger = OwnAuditLedger.open(":memory:")  # E1 never recorded CLEAN
        gate = _gate(
            a.store,
            prior_log_for=lambda log: priors.get(log.epoch_id - 1),
            is_genesis_for=lambda log: log.epoch_id == 1,
            ledger=ledger,
        )
        verdict = gate.review(e2, now=NOW)
        assert verdict.ok is False
        assert verdict.status is AuditStatus.INCONCLUSIVE
    finally:
        a.close()


async def test_gate_clears_implicit_carry_forward_when_predecessor_recorded_clean(tmp_path) -> None:
    """an internal review: once the predecessor (E1) is own-audited CLEAN and recorded, the
    IMPLICIT carry-forward in E2 is trusted and the gate clears — no false permanent HOLD."""
    from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger

    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        e1, e2 = _implicit_carry_logs(a.store)
        priors = {1: e1, 2: e2}
        ledger = OwnAuditLedger.open(":memory:")
        gate = _gate(
            a.store,
            prior_log_for=lambda log: priors.get(log.epoch_id - 1),
            is_genesis_for=lambda log: log.epoch_id == 1,
            ledger=ledger,
        )
        g = gate.review(e1, now=NOW)  # genesis clears (zero carry-in) and is RECORDED
        assert g.ok is True and g.status is AuditStatus.CLEAN
        assert ledger.is_clean(e1.epoch_id, e1.log_digest())
        s = gate.review(e2, now=NOW)  # implicit carry chains to the recorded predecessor
        assert s.ok is True and s.status is AuditStatus.CLEAN
    finally:
        a.close()


async def test_gate_reports_disputed_substituted_weight(tmp_path) -> None:
    """A vector that does not follow from inputs is reported DISPUTED, never enforced."""
    a = AuthorityHarness(tmp_path, burn_uid=999)
    try:
        miners = (make_miner(1, 0.9), make_miner(2, 0.9))
        shares = {1: 1.0}  # build_weight_vector over {1,2} could NEVER emit this
        u16 = quantize_u16(shares)
        # uid 1 gets a real manifest entry (the log validator requires backing for a
        # nonzero-weight uid) AND committed window evidence (so the auditor re-derives the
        # windowed inputs and PROCEEDS to the weight compare); only the WEIGHTS are the lie.
        manifest = build_audit_manifest(
            [make_item(1, a.store)],
        )
        manifest = manifest.model_copy(
            update={"fold_cursors": {**manifest.fold_cursors, 2: None}}
        )
        # This case intentionally builds an inference-only epoch.
        log = EpochLog(
            epoch_id=4, close_block=14399, scorer_version=SCORER, created_at=NOW,
            burn_uid=None, miners=miners,
            miner_census=tuple(MinerCensusEntry.from_miner(m) for m in miners),
            weight_shares=shares, weight_u16=u16,
            weight_vector_digest=weight_vector_digest(u16),
            audit_manifest=manifest,
        )
        verdict = _gate(a.store).review(log, now=NOW)
        assert verdict.ok is False
        assert verdict.status is AuditStatus.DISPUTED
        assert verdict.report.weight_verdict.verdict is ItemVerdictKind.FAIL
    finally:
        a.close()
