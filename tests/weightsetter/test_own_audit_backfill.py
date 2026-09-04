"""The own-audit gate's CONTIGUOUS backfill + durable cursor.

Finding #2 (HIGH): the round-12 reviewer resolved only the authority's LATEST pointer. A SKIPPED
epoch (weightsetting ~30s while epochs are ~20s) then stranded the own-audited-CLEAN ledger
chain FOREVER — the next positive-carry audit was INCONCLUSIVE because its predecessor was never recorded,
and later passes only fetched still-newer epochs. A restart with an empty ledger mid-chain
wedged the same way.

The fix mirrors the public auditor loop: a durable contiguous cursor + sequential BACKFILL.
`review(latest)` walks `cursor+1 .. latest`, fetching each missed epoch via a `fetch_log_for`
seam (wired in production to the shared provider's `resolve_epoch`, itself `pointer_for` + the
three-leg verify), own-auditing it, recording it CLEAN, and advancing the cursor — so the
target's carry-in predecessor is present on an honest chain and the chain never gaps. A
withheld/unverifiable predecessor still stops the audit walk conservatively, but an honest
contiguous chain never wedges, and a restart resumes from the durable cursor. These outcomes
are report-only and never interrupt the authenticated authority weight submission.
"""

from __future__ import annotations

from vidaio.auditor import AuditorConfig, AuditStatus, SamplePolicy
from vidaio.tokenomics import MinerSnapshot, TokenomicsConfig
from vidaio.tokenomics.ewma import accumulate
from vidaio.weightsetter.own_audit import OwnAuditGate
from vidaio.weightsetter.own_audit_cursor import OwnAuditCursor
from vidaio.weightsetter.own_audit_ledger import OwnAuditLedger

from tests.weightsetter.weightsetter_support import (
    NOW,
    AuthorityHarness,
    make_item,
    make_miner,
)

_CYCLE = 0.8


async def _finalize_carry_chain(a: AuthorityHarness, n: int) -> dict[int, object]:
    """Finalize a self-consistent chain of `n` epochs (ids 1..n), each carrying uid 1's
    accumulate forward (same hotkey) so every epoch past genesis has a NONZERO explicit
    carry-in the auditor's earning re-fold verifies against its predecessor."""
    decay = TokenomicsConfig().ewma_decay
    logs: dict[int, object] = {}
    acc = 0.0
    prev_digest = None
    for k in range(1, n + 1):
        prior_acc = acc
        acc = accumulate(acc, _CYCLE, decay)
        if k == 1:
            miner = make_miner(1, 0.9)  # genesis: accumulate = fold(0, [0.8])
            prior_accumulate = None
        else:
            miner = MinerSnapshot(
                uid=1, hotkey="hk1", coldkey="ck1", ip="10.0.0.1",
                track="compression", accumulate_score=acc,
            )
            prior_accumulate = {1: prior_acc}
        finalized = await a.finalize(
            # Each epoch folds a NEW cycle at a strictly higher committed key (seq=k-1) — a
            # monotonic per-uid ordering, not a re-fold of an earlier packet (round-22 #1).
            epoch_id=k, close_block=(k + 1) * 3600 - 1,
            miners=[miner], items=[make_item(1, a.store, seq=k - 1)],
            prior_accumulate=prior_accumulate, prior_log_digest=prev_digest,
        )
        logs[k] = finalized.log
        prev_digest = finalized.log.log_digest()
    return logs


def _gate(store, logs, *, ledger, cursor, fetches: list[int] | None = None, floor: int = 1):
    from tests.auditor.fakes import MetagraphAuditor

    auditor = MetagraphAuditor.over_store(
        AuditorConfig(
            auditor_hotkey="v-own", backend="fake", tokenomics=TokenomicsConfig(),
            burn_uid=0,  # explicit report-mode fallback; production reads chain state
        ),
        store,
    )

    def fetch_log_for(epoch_id: int):
        if fetches is not None:
            fetches.append(epoch_id)
        return logs.get(epoch_id)  # a KeyError-free "unavailable" is None (HOLD)

    return OwnAuditGate(
        auditor=auditor,
        store=store,
        policy=SamplePolicy(sample_rate=0.0, min_samples=0),
        prior_log_for=lambda log: logs.get(log.epoch_id - 1),
        is_genesis_for=lambda log: log.epoch_id == floor,
        ledger=ledger,
        cursor=cursor,
        fetch_log_for=fetch_log_for,
        audit_floor=floor,
    )


async def test_skipped_epoch_is_backfilled_and_the_chain_clears(tmp_path) -> None:
    """The crux: reviewing the LATEST epoch (3) with epochs 1,2 NEVER reviewed does NOT
    permanently HOLD. The gate BACKFILLS 1 then 2 (fetch+own-audit+record), so 3's carry-in
    predecessor is present and it clears — the chain stays gap-free."""
    a = AuthorityHarness(tmp_path)
    try:
        logs = await _finalize_carry_chain(a, 3)
        ledger = OwnAuditLedger.open(":memory:")
        cursor = OwnAuditCursor.open(":memory:")
        fetches: list[int] = []
        gate = _gate(a.store, logs, ledger=ledger, cursor=cursor, fetches=fetches)

        # Jump straight to the latest (epoch 3) — the SKIP the round-12 gate wedged on.
        verdict = gate.review(logs[3], now=NOW)
        assert verdict.ok is True and verdict.status is AuditStatus.CLEAN
        # 1 and 2 were BACKFILLED (fetched), and the whole chain is recorded gap-free.
        assert fetches == [1, 2]
        for k in (1, 2, 3):
            assert ledger.is_clean(k, logs[k].log_digest())
        assert cursor.last_clean() == 3
    finally:
        a.close()


async def test_without_backfill_a_skipped_epoch_permanently_holds(tmp_path) -> None:
    """Contrast (the round-12 bug): the SINGLE-epoch gate (no cursor / no fetch seam) HOLDs the
    latest epoch forever when its predecessor was skipped — the wedge finding #2 fixes."""
    a = AuthorityHarness(tmp_path)
    try:
        logs = await _finalize_carry_chain(a, 3)
        from tests.auditor.fakes import MetagraphAuditor

        auditor = MetagraphAuditor.over_store(
            AuditorConfig(
                auditor_hotkey="v-own", backend="fake", tokenomics=TokenomicsConfig(),
                burn_uid=0,  # explicit report-mode fallback
            ),
            a.store,
        )
        gate = OwnAuditGate(  # no cursor, no fetch_log_for -> pure single-epoch gate
            auditor=auditor, store=a.store,
            policy=SamplePolicy(sample_rate=0.0, min_samples=0),
            prior_log_for=lambda log: logs.get(log.epoch_id - 1),
            is_genesis_for=lambda log: log.epoch_id == 1,
            ledger=OwnAuditLedger.open(":memory:"),
        )
        verdict = gate.review(logs[3], now=NOW)  # predecessor 2 never recorded
        assert verdict.ok is False and verdict.status is AuditStatus.INCONCLUSIVE
    finally:
        a.close()


async def test_restart_mid_chain_resumes_from_the_durable_cursor(tmp_path) -> None:
    """A restart (fresh gate over the SAME on-disk ledger + cursor) resumes at cursor+1 — it
    does NOT re-fetch / re-audit the epochs already recorded CLEAN, and the next epoch clears."""
    a = AuthorityHarness(tmp_path)
    try:
        logs = await _finalize_carry_chain(a, 4)
        ledger_path = tmp_path / "own-audit-clean.db"
        cursor_path = tmp_path / "own-audit-cursor.db"

        # Phase 1: a fresh gate backfills + clears up to epoch 3.
        gate1 = _gate(
            a.store, logs,
            ledger=OwnAuditLedger.open(ledger_path), cursor=OwnAuditCursor.open(cursor_path),
        )
        assert gate1.review(logs[3], now=NOW).ok is True

        # Phase 2 "restart": brand-new ledger + cursor handles over the SAME files.
        fetches: list[int] = []
        gate2 = _gate(
            a.store, logs, fetches=fetches,
            ledger=OwnAuditLedger.open(ledger_path), cursor=OwnAuditCursor.open(cursor_path),
        )
        cursor2 = OwnAuditCursor.open(cursor_path)
        assert cursor2.last_clean() == 3  # durable: resumes where phase 1 left off

        verdict = gate2.review(logs[4], now=NOW)  # only epoch 4 is new
        assert verdict.ok is True and verdict.status is AuditStatus.CLEAN
        assert fetches == []  # NO re-fetch of 1..3 — resumed from the cursor (start=4)
        assert OwnAuditCursor.open(cursor_path).last_clean() == 4
    finally:
        a.close()


async def test_zero_carry_burn_target_holds_when_a_predecessor_is_gapped(tmp_path) -> None:
    """an internal review: a ZERO-carry / burn TARGET must STILL HOLD (and NOT advance the cursor
    over the gap) when a predecessor in the backfill range was HELD/unavailable.

    The pre-fix gate ignored whether `_backfill_predecessors` stopped early: a burn/zero-carry
    target self-audits CLEAN with no carry-in to check, so it advanced the cursor OVER the gap and
    the skipped epoch was never revisited (the contiguous own-audited-CLEAN invariant broke). Now
    the backfill REPORTS non-contiguity and the target HOLDs regardless of its carry-in."""
    a = AuthorityHarness(tmp_path)
    try:
        logs = await _finalize_carry_chain(a, 2)  # 1,2 carry-chain (uid 1 forward)
        # Epoch 3 is a BURN epoch: no miners -> ZERO carry-in, yet it still chains the prior log.
        burn = await a.finalize(
            epoch_id=3, close_block=4 * 3600 - 1, miners=[], items=None,
            prior_log_digest=logs[2].log_digest(),
        )
        logs[3] = burn.log
        assert not OwnAuditGate._has_nonzero_carry_in(logs[3])  # the target truly carries nothing

        # Epoch 2 is WITHHELD from the fetch seam, so the backfill stops before the target's
        # immediate predecessor.
        withheld = {1: logs[1], 3: logs[3]}
        ledger = OwnAuditLedger.open(":memory:")
        cursor = OwnAuditCursor.open(":memory:")
        gate = _gate(a.store, withheld, ledger=ledger, cursor=cursor)

        verdict = gate.review(logs[3], now=NOW)
        assert verdict.ok is False and verdict.status is AuditStatus.INCONCLUSIVE
        # 1 recorded; the walk stopped at the withheld 2; the target 3 is NEITHER recorded NOR
        # advanced over — the cursor stays at 1 (no gap-skipping).
        assert ledger.is_clean(1, logs[1].log_digest())
        assert cursor.last_clean() == 1
        assert not ledger.is_clean(3, logs[3].log_digest())

        # Once the gap is fillable (epoch 2 available), the target proceeds and the chain closes.
        fillable = {1: logs[1], 2: logs[2], 3: logs[3]}
        gate2 = _gate(a.store, fillable, ledger=ledger, cursor=cursor)
        v2 = gate2.review(logs[3], now=NOW)
        assert v2.ok is True and v2.status is AuditStatus.CLEAN
        assert cursor.last_clean() == 3
        for k in (1, 2, 3):
            assert ledger.is_clean(k, logs[k].log_digest())
    finally:
        a.close()


class _FlakyLedger:
    """An OwnAuditLedger whose `record_clean` raises until `heal()`."""

    def __init__(self, inner: OwnAuditLedger) -> None:
        self._inner = inner
        self.fail = True

    def is_clean(self, epoch_id: int, log_digest: str) -> bool:
        return self._inner.is_clean(epoch_id, log_digest)

    def record_clean(self, epoch_id: int, log_digest: str) -> None:
        if self.fail:
            raise RuntimeError("injected ledger write failure")
        self._inner.record_clean(epoch_id, log_digest)

    def heal(self) -> None:
        self.fail = False


async def test_ledger_write_failure_holds_without_advancing_the_cursor(tmp_path) -> None:
    """an internal review: an injected `record_clean` failure must HOLD — the cursor does NOT
    advance (no CLEAN-without-ledger-entry), and a later successful pass records + advances.

    The pre-fix gate SWALLOWED the record_clean exception and still advanced the cursor, so the
    cursor could permanently pass an ABSENT ledger entry (the two are separate connections),
    stranding every future carry epoch. Now record_clean -> (on success) CLEAN -> advance is
    strictly ordered: the cursor only ever moves AFTER the entry is durably recorded."""
    a = AuthorityHarness(tmp_path)
    try:
        logs = await _finalize_carry_chain(a, 1)  # genesis (zero carry-in) — no predecessor
        real = OwnAuditLedger.open(":memory:")
        flaky = _FlakyLedger(real)
        cursor = OwnAuditCursor.open(":memory:")
        gate = _gate(a.store, logs, ledger=flaky, cursor=cursor)

        # The genesis epoch self-audits CLEAN, but the ledger write fails -> HOLD, cursor unmoved.
        verdict = gate.review(logs[1], now=NOW)
        assert verdict.ok is False and verdict.status is AuditStatus.INCONCLUSIVE
        assert cursor.last_clean() is None  # NOT advanced past an unrecorded entry
        assert not real.is_clean(1, logs[1].log_digest())

        # The ledger heals; the retry records + advances (no CLEAN was ever emitted without a
        # durable ledger entry, so nothing was stranded).
        flaky.heal()
        v2 = gate.review(logs[1], now=NOW)
        assert v2.ok is True and v2.status is AuditStatus.CLEAN
        assert cursor.last_clean() == 1
        assert real.is_clean(1, logs[1].log_digest())
    finally:
        a.close()


async def test_backfill_stops_at_a_withheld_predecessor_and_holds(tmp_path) -> None:
    """Fail closed: a predecessor the authority WITHHOLDS (fetch returns None) stops the walk,
    so nothing is recorded ahead of it and the target HOLDs on its now-missing predecessor —
    never a false CLEAN. (An honest, available chain clears; only the withheld one HOLDs.)"""
    a = AuthorityHarness(tmp_path)
    try:
        logs = await _finalize_carry_chain(a, 3)
        withheld = {1: logs[1], 3: logs[3]}  # epoch 2 is unavailable (withheld)
        ledger = OwnAuditLedger.open(":memory:")
        cursor = OwnAuditCursor.open(":memory:")
        gate = _gate(a.store, withheld, ledger=ledger, cursor=cursor)

        verdict = gate.review(logs[3], now=NOW)
        assert verdict.ok is False and verdict.status is AuditStatus.INCONCLUSIVE
        # 1 was recorded; the walk stopped at the withheld 2; 3 never cleared.
        assert ledger.is_clean(1, logs[1].log_digest())
        assert cursor.last_clean() == 1
        assert not ledger.is_clean(3, logs[3].log_digest())
    finally:
        a.close()
