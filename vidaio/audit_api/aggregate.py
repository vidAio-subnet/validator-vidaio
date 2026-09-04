"""The aggregate honesty verdict — the investigation surface (the project design record §3.2).

``GET /audit/status`` does not return one auditor's opinion; it returns the AGGREGATE
across every auditor that reported an epoch: how many reported, how many CLEAN vs
DISPUTED, the union of disputed items + their reason codes, and one epoch verdict:

    UNAUDITED  — no auditor has reported this epoch yet
    DISPUTED   — at least one auditor reported a FAIL (any dispute is conclusive; a
                 single provable fault flips the epoch, the project design record §5)
    CLEAN      — at least one report and NONE disputed
    INCONCLUSIVE — reports exist but none achieved recompute coverage

This is what makes misreporting visible: an honest majority cannot out-vote one auditor's
provable FAIL. Conflicts (a divergent resubmission by one auditor for one epoch) are
carried alongside as their own signal — surfaced, not folded into the verdict.

Pure functions over ``StoredReport``s: the store reads the rows, these shape them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from vidaio.auditor.report import (
    AuditMode,
    AuditReport,
    AuditStatus,
    ItemVerdictKind,
    overall_status,
)

from vidaio.audit_api.store import StoredReport

#: The epoch-level verdicts the aggregate can assert.
EPOCH_CLEAN = "CLEAN"
EPOCH_DISPUTED = "DISPUTED"
EPOCH_UNAUDITED = "UNAUDITED"
#: Reports exist but NONE achieved recompute coverage (every one INCONCLUSIVE): the
#: epoch is neither proven honest nor disputed — a distinct NEEDS-ATTENTION state, not
#: washed to CLEAN (#8, the project design record §3.2 INCONCLUSIVE).
EPOCH_INCONCLUSIVE = "INCONCLUSIVE"

#: The synthetic "source" a weight-derivation dispute is reported under (it is an
#: epoch-level fault, not a per-item one, but belongs in the disputed set).
WEIGHT_SOURCE = "weight"


def _disputed_items_of(report: AuditReport) -> list[dict[str, Any]]:
    """Every provable FAIL in one report — item FAILs plus a weight FAIL."""
    out: list[dict[str, Any]] = []
    for v in report.failures():
        out.append(
            {
                "auditor_hotkey": report.auditor_hotkey,
                "audit_mode": report.audit_mode.value,
                "source": v.source,
                "challenge_id": v.challenge_id,
                "item_id": v.item_id,
                "miner_hotkey": v.miner_hotkey,
                "uid": v.uid,
                "bundle_digest": v.bundle_digest,
                "packet_digest": v.packet_digest,
                "code": v.code,
                "detail": v.detail,
            }
        )
    if report.weight_verdict.verdict is ItemVerdictKind.FAIL:
        out.append(
            {
                "auditor_hotkey": report.auditor_hotkey,
                "audit_mode": report.audit_mode.value,
                "source": WEIGHT_SOURCE,
                "challenge_id": "",
                "item_id": "weight_vector",
                "miner_hotkey": None,
                "uid": None,
                "bundle_digest": "",
                "packet_digest": "",
                "code": report.weight_verdict.code,
                "detail": report.weight_verdict.detail,
            }
        )
    return out


def _effective_verdict(report: AuditReport) -> AuditStatus:
    """One report's verdict RECOMPUTED from its item + weight verdicts.

    "One provable fault ⇒ DISPUTED" is enforced HERE, never trusting the report's
    self-reported ``overall``: a report claiming CLEAN while carrying a FAIL item
    aggregates as DISPUTED (the project design record §5). ``earning_verdicts`` is
    passed as the third channel so an unverifiable earning state aggregates INCONCLUSIVE
    (matching the report's own derived ``overall``), never washed to CLEAN.
    """
    return overall_status(
        report.item_verdicts, report.weight_verdict, report.earning_verdicts
    )


def epoch_status(
    epoch_id: int,
    reports: list[StoredReport],
    *,
    conflicts: int,
    disputed_conflicts: int = 0,
) -> dict[str, Any]:
    """The aggregate honesty status for investigation and manual remediation."""
    audit_reports = [s.report for s in reports]
    verdicts = [_effective_verdict(r) for r in audit_reports]
    clean = sum(1 for v in verdicts if v is AuditStatus.CLEAN)
    disputed = sum(1 for v in verdicts if v is AuditStatus.DISPUTED)
    inconclusive = sum(1 for v in verdicts if v is AuditStatus.INCONCLUSIVE)
    reports_by_mode = {
        mode.value: sum(1 for stored in reports if stored.audit_mode is mode)
        for mode in AuditMode
    }

    if not audit_reports:
        verdict = EPOCH_UNAUDITED
    # A DISPUTED persisted report OR a DISPUTED divergent (conflict) report is
    # conclusive — a single provable fault flips the epoch and cannot be out-voted
    # or buried by a CLEAN first report.
    elif disputed > 0 or disputed_conflicts > 0:
        verdict = EPOCH_DISPUTED
    # At least one auditor achieved recompute coverage and it was clean.
    elif clean > 0:
        verdict = EPOCH_CLEAN
    # Reports exist, none disputed, none clean -> every one was INCONCLUSIVE: the
    # epoch is un-audited (nothing recomputed), a NEEDS-ATTENTION state, NOT CLEAN (#8).
    else:
        verdict = EPOCH_INCONCLUSIVE

    disputed_items: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for r in audit_reports:
        for item in _disputed_items_of(r):
            disputed_items.append(item)
            if item["code"]:
                reason_counts[item["code"]] += 1

    # Distinct snapshot digests reported — a single value when auditors agree on
    # WHICH bytes they audited; more than one is itself worth surfacing.
    digests = sorted({s.snapshot_digest for s in reports})
    snapshot_digest = digests[0] if len(digests) == 1 else None

    return {
        "epoch_id": epoch_id,
        "verdict": verdict,
        # A validator may submit one beacon report and one own-audit report. It is
        # still one reporting validator, not two votes.
        "auditors_reporting": len({stored.auditor_hotkey for stored in reports}),
        "reports_received": len(reports),
        "reports_by_mode": reports_by_mode,
        "clean": clean,
        "disputed": disputed,
        "inconclusive": inconclusive,
        "conflicts": conflicts,
        "disputed_conflicts": disputed_conflicts,
        "snapshot_digest": snapshot_digest,
        "snapshot_digests": digests,
        "reason_counts": dict(sorted(reason_counts.items())),
        "disputed_items": disputed_items,
    }


def epoch_rollup(
    epoch_id: int,
    reports: list[StoredReport],
    *,
    conflicts: int,
    disputed_conflicts: int = 0,
) -> dict[str, Any]:
    """A compact per-epoch row for GET /audit/epochs (no per-item detail)."""
    status = epoch_status(
        epoch_id, reports, conflicts=conflicts, disputed_conflicts=disputed_conflicts
    )
    return {
        "epoch_id": status["epoch_id"],
        "verdict": status["verdict"],
        "auditors_reporting": status["auditors_reporting"],
        "reports_received": status["reports_received"],
        "reports_by_mode": status["reports_by_mode"],
        "clean": status["clean"],
        "disputed": status["disputed"],
        "inconclusive": status["inconclusive"],
        "conflicts": status["conflicts"],
        "disputed_conflicts": status["disputed_conflicts"],
        "snapshot_digest": status["snapshot_digest"],
        "reason_counts": status["reason_counts"],
    }


def feed_entry(stored: StoredReport) -> dict[str, Any]:
    """One report as a dashboard-feed row (summary, not the full report bytes)."""
    report = stored.report
    return {
        "report_id": stored.report_id,
        "auditor_hotkey": stored.auditor_hotkey,
        "epoch_id": stored.epoch_id,
        "audit_mode": stored.audit_mode.value,
        "snapshot_digest": stored.snapshot_digest,
        "pipeline_version": stored.pipeline_version,
        "overall": stored.overall,
        "competition_n": stored.competition_n,
        "inference_n": stored.inference_n,
        "sampled_at": stored.sampled_at,
        "received_at": stored.received_at,
        "failures": [
            {
                "source": v.source,
                "challenge_id": v.challenge_id,
                "item_id": v.item_id,
                "code": v.code,
            }
            for v in report.failures()
        ],
        "weight_verdict": report.weight_verdict.verdict.value,
    }
