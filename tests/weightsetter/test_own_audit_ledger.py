"""The legacy own-audited-CLEAN classifier ledger.

Production weight-setting uses report-only own-audit mode and bypasses this policy
state; these tests preserve the conservative audit classification behavior only.

The gate trusts a nonzero earning carry-in ONLY when the predecessor was itself recorded
own-audited CLEAN in this durable store. These tests pin the store's contract: exact
(epoch_id, log_digest) membership, idempotent re-record, an immutable-entry conflict, and
the in-database monotonic-forward guard (mirroring the auditor cursor) so the chain can
never be rewound to re-attribute an unaudited carry-in.
"""

from __future__ import annotations

import sqlite3

import pytest

from vidaio.weightsetter.own_audit_ledger import OwnAuditCleanConflict, OwnAuditLedger


def _ledger() -> OwnAuditLedger:
    return OwnAuditLedger.open(":memory:")


def test_is_clean_requires_exact_epoch_and_digest() -> None:
    led = _ledger()
    assert led.is_clean(1, "d1") is False  # empty ledger vouches for nothing
    led.record_clean(1, "d1")
    assert led.is_clean(1, "d1") is True
    assert led.is_clean(1, "d2") is False  # same epoch, WRONG digest
    assert led.is_clean(2, "d1") is False  # same digest, WRONG epoch


def test_record_clean_is_idempotent_for_same_entry() -> None:
    led = _ledger()
    led.record_clean(5, "dd")
    led.record_clean(5, "dd")  # no-op, no error
    assert led.is_clean(5, "dd") is True


def test_record_clean_conflict_on_different_digest_same_epoch() -> None:
    led = _ledger()
    led.record_clean(5, "dd")
    with pytest.raises(OwnAuditCleanConflict):
        led.record_clean(5, "OTHER")


def test_ledger_only_extends_forward() -> None:
    led = _ledger()
    led.record_clean(3, "d3")
    # A brand-new epoch at/below the highest recorded one is aborted in-database — the chain
    # never rewinds (which would re-open a window to trust an unaudited carry-in).
    with pytest.raises(sqlite3.IntegrityError):
        led.record_clean(2, "d2")
    led.record_clean(4, "d4")  # strictly forward is allowed
    assert led.is_clean(4, "d4") is True
