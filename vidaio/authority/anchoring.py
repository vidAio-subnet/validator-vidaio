"""Anchoring — bind a finalized epoch's log_digest on chain (tamper-evidence root).

After the finalizer writes a `FinalizedEpoch` to the object store, the authority
ANCHORS its `log_digest` on chain via an injected `ChainAdapter`. The anchored
digest is the root a validator verifies its mirrored bytes against
(`sha256(bytes) == snapshot_digest == on-chain anchored digest`,
the project design record §4/§5, build-wave 4). In report/chainsim mode the
`InMemoryChain`/chainsim simply records the payload. Bittensor anchors persist
the signed hash before one-shot dispatch, then confirm that same hash across
bounded retries and process restarts.

The commitment payload is a small, domain-tagged, <=128-byte value over the
log_digest (the commitments-module payload style — a domain tag + a digest —
without pulling in the merkle/ledger machinery, which is for score-packet sets, not
a single epoch-log root).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from vidaio.authority.api import AnchorRecord, anchor_from_record
from vidaio.authority.finalizer import FinalizedEpoch
from vidaio.authority.index import AnchorSubmissionRecord, EpochIndex
from vidaio.chain.adapter import ChainAdapter
from vidaio.chain.anchor_receipt import (
    DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS,
    DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS,
    wait_for_finalized_anchor_receipt,
)
from vidaio.chain.anchor_writer import (
    anchor_writer_lock, clear_anchor_writer_pending, mark_anchor_writer_pending,
    read_anchor_writer_pending,
)
from vidaio.services.commitment_capacity import require_commitment_capacity

#: Versioned domain tag for the epoch-log anchor payload. Bump on any change to the
#: payload byte contract (it would change every anchored value).
ANCHOR_DOMAIN = "vidaio.epoch.anchor.v1"


class AnchorSubmissionHold(RuntimeError):
    """An expected anchor hold that must not cause another uncertain dispatch."""


def _restore_marker_submission(index: EpochIndex, marker: dict) -> AnchorSubmissionRecord:
    recorded = index.anchor_submission_by_hash(marker["extrinsic_hash"])
    if recorded is not None:
        if (recorded.epoch_id != marker.get("epoch_id")
                or recorded.netuid != marker.get("netuid")
                or recorded.log_digest != marker.get("log_digest")
                or recorded.submission != marker.get("submission")):
            raise AnchorSubmissionHold("pending writer marker contradicts durable submission evidence")
        return recorded
    submission = marker.get("submission")
    if not isinstance(submission, dict) or submission.get("extrinsic_hash") != marker["extrinsic_hash"]:
        raise AnchorSubmissionHold("pending writer marker lacks recoverable signed evidence")
    return index.record_anchor_submission(
        marker["epoch_id"], netuid=marker["netuid"],
        log_digest=marker["log_digest"], submission=submission,
    )


async def clear_confirmed_anchor_writer(
    *, index: EpochIndex, netuid: int | None = None,
    writer_lock_path: Path | None = None, writer_lock_timeout_seconds: float = 30.0,
) -> bool:
    """Recover a crash between durable terminal indexing and fence removal."""
    marker = read_anchor_writer_pending(writer_lock_path)
    if marker is None:
        return False
    txid = marker["extrinsic_hash"]
    async with anchor_writer_lock(
        writer_lock_path, timeout_seconds=writer_lock_timeout_seconds,
        pending_extrinsic_hash=txid,
    ):
        submission = _restore_marker_submission(index, marker)
        if netuid is not None and submission.netuid != netuid:
            raise AnchorSubmissionHold("pending authority marker belongs to a different subnet")
        record = index.get(submission.epoch_id)
        if submission.outcome is None and record is not None and record.anchored:
            if record.anchor_txid != txid:
                raise AnchorSubmissionHold("indexed anchor differs from pending signed evidence")
            submission = index.finish_anchor_submission(
                txid, outcome="confirmed", outcome_at=datetime.now(timezone.utc).isoformat(),
                evidence={"block": record.anchor_block},
            )
        if submission.outcome == "confirmed" and (
            record is None or record.anchor_txid != txid
        ):
            raise AnchorSubmissionHold("confirmed evidence has no matching indexed anchor")
        if submission.outcome is None:
            return False
        clear_anchor_writer_pending(writer_lock_path, extrinsic_hash=txid)
        return True


async def acknowledge_anchor_submission(
    *, index: EpochIndex, extrinsic_hash: str, writer_lock_path: Path | None = None,
    writer_lock_timeout_seconds: float = 30.0, now: datetime | None = None,
) -> AnchorSubmissionRecord:
    """Apply the explicit operator acknowledgement without issuing a transaction."""
    async with anchor_writer_lock(
        writer_lock_path, timeout_seconds=writer_lock_timeout_seconds,
        pending_extrinsic_hash=extrinsic_hash,
    ):
        submission = index.anchor_submission_by_hash(extrinsic_hash)
        if submission is None:
            marker = read_anchor_writer_pending(writer_lock_path)
            if marker is not None and marker["extrinsic_hash"] == extrinsic_hash:
                submission = _restore_marker_submission(index, marker)
        if submission is None or submission.outcome is not None:
            raise ValueError("authority anchor acknowledgement must name a pending hash")
        updated = index.finish_anchor_submission(
            extrinsic_hash, outcome="operator_acknowledged",
            outcome_at=(now or datetime.now(timezone.utc)).isoformat(),
            evidence={"operator_ack": extrinsic_hash},
        )
        clear_anchor_writer_pending(writer_lock_path, extrinsic_hash=extrinsic_hash)
        logging.getLogger("authority-finalizer").error(
            "authority anchor submission acknowledged by operator",
            extra={"fields": {"extrinsic_hash": extrinsic_hash, "epoch_id": updated.epoch_id}},
        )
        return updated


async def _anchor_with_durable_submission(
    finalized: FinalizedEpoch, *, chain: ChainAdapter, index: EpochIndex,
    netuid: int, now: datetime, anchor_hotkey: str, writer_lock_path: Path | None,
    on_phase: Callable[[str], None], search_blocks: int, confirmation_depth: int,
    verification_timeout_seconds: float, verification_poll_seconds: float,
    on_expired: Callable[[], None] | None,
) -> AnchorRecord:
    submission = index.get_anchor_submission(finalized.epoch_id)
    if submission is not None and submission.outcome == "operator_acknowledged":
        raise AnchorSubmissionHold("operator acknowledged this anchor; normal epoch recovery is required")
    if submission is None or submission.outcome == "expired_absent":
        on_phase("compose")
        payload = anchor_payload(finalized.epoch_id, netuid, finalized.log_digest)
        await require_commitment_capacity(
            chain, netuid=netuid, hotkey=anchor_hotkey, payload=payload,
            operation=f"epoch {finalized.epoch_id} anchor",
        )
        prepared = await chain.prepare_authority_anchor(payload)
        prepared_head = int(prepared["submit_block"])
        if prepared_head >= finalized.close_block + confirmation_depth:
            raise AnchorSubmissionHold(
                f"epoch {finalized.epoch_id} anchor deadline elapsed before dispatch: "
                f"fresh prepared head {prepared_head} >= close {finalized.close_block} "
                f"+ K {confirmation_depth}; no transaction submitted"
            )
        on_phase("index")
        submission = index.record_anchor_submission(
            finalized.epoch_id, netuid=netuid, log_digest=finalized.log_digest,
            submission=prepared,
        )
        if submission.outcome is not None:
            raise AnchorSubmissionHold(
                "fresh authority preparation reused a terminal signed hash; refusing dispatch"
            )
        fresh = True
    else:
        fresh = False
    if submission.netuid != netuid or submission.log_digest != finalized.log_digest:
        raise AnchorSubmissionHold("durable authority submission does not match the indexed epoch")
    txid = submission.extrinsic_hash
    marker = {
        "extrinsic_hash": txid, "epoch_id": finalized.epoch_id, "netuid": netuid,
        "log_digest": finalized.log_digest,
        "submit_block": submission.submission["submit_block"],
        "pid": os.getpid(), "time": time.time(), "submission": submission.submission,
    }
    on_phase("index")
    mark_anchor_writer_pending(writer_lock_path, marker)
    if fresh:
        on_phase("anchor_submit")
        try:
            await chain.submit_authority_anchor(submission.submission)
        except Exception as exc:
            # A lost response is not proof of failed dispatch. Only read the known
            # signed hash from now on, including on the next tick/process restart.
            logging.getLogger("authority-finalizer").warning(
                "authority anchor dispatch uncertain; confirming durable hash",
                extra={"fields": {"epoch_id": finalized.epoch_id, "extrinsic_hash": txid,
                                  "error": str(exc)}},
            )
        finally:
            on_phase("anchor_confirm")
    else:
        on_phase("anchor_confirm")
    try:
        block = await chain.confirm_authority_anchor(
            submission.submission, search_blocks=search_blocks,
            timeout_seconds=verification_timeout_seconds,
            poll_seconds=verification_poll_seconds, confirmation_depth=confirmation_depth,
        )
        receipt = await wait_for_finalized_anchor_receipt(
            chain, netuid=netuid, anchor_id=finalized.epoch_id, domain=ANCHOR_DOMAIN,
            expected_digest=finalized.log_digest, operation=f"epoch {finalized.epoch_id} anchor",
            timeout_seconds=verification_timeout_seconds, poll_seconds=verification_poll_seconds,
            inclusion_block=block,
            archive_reader=getattr(chain, "read_authority_anchor_at", None),
        )
    except Exception as exc:
        from vidaio.chain.bittensor_adapter import AuthorityAnchorExpiredAbsent

        if isinstance(exc, AuthorityAnchorExpiredAbsent):
            on_phase("index")
            index.finish_anchor_submission(
                txid, outcome="expired_absent", outcome_at=now.isoformat(), evidence=exc.evidence,
            )
            clear_anchor_writer_pending(writer_lock_path, extrinsic_hash=txid)
            if on_expired is not None:
                on_expired()
            logging.getLogger("authority-finalizer").error(
                "authority anchor expired and absent from finalized era",
                extra={"fields": {"epoch_id": finalized.epoch_id, **exc.evidence}},
            )
        raise AnchorSubmissionHold(f"epoch {finalized.epoch_id} anchor {txid}: {exc}") from exc
    on_phase("index")
    updated = index.set_anchor(finalized.epoch_id, txid=txid, block=receipt.block)
    index.finish_anchor_submission(
        txid, outcome="confirmed", outcome_at=now.isoformat(), evidence={"block": receipt.block},
    )
    clear_anchor_writer_pending(writer_lock_path, extrinsic_hash=txid)
    return anchor_from_record(updated)


def anchor_payload(epoch_id: int, netuid: int, log_digest: str) -> bytes:
    """The <=128-byte, domain-tagged commitment bytes anchoring `log_digest`.

    `<domain>:<netuid>:<epoch_id>:<log_digest>` (ascii). Domain-separated so an
    epoch-log anchor can never be confused with any other commitment; carries the
    64-hex `log_digest` verbatim so a third party reads the anchored digest straight
    out of the on-chain bytes.
    """
    payload = f"{ANCHOR_DOMAIN}:{netuid}:{epoch_id}:{log_digest}".encode("ascii")
    if len(payload) > 128:  # pragma: no cover - fixed-width inputs keep this ~95 bytes
        raise ValueError(f"anchor payload is {len(payload)} bytes (> 128)")
    return payload


async def anchor_epoch(
    finalized: FinalizedEpoch,
    *,
    chain: ChainAdapter,
    index: EpochIndex,
    netuid: int,
    now: datetime,
    anchor_hotkey: str = "",
    writer_lock_path: Path | None = None,
    writer_lock_timeout_seconds: float = 30.0,
    verification_timeout_seconds: float = DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS,
    verification_poll_seconds: float = DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS,
    on_phase: Callable[[str], None] | None = None,
    search_blocks: int = 60,
    confirmation_depth: int = 20,
    on_expired: Callable[[], None] | None = None,
) -> AnchorRecord:
    """Anchor the finalized epoch's log_digest on chain and record the txid.

    Idempotent per epoch: if the epoch is already anchored in the index this is a
    NO-OP that returns the existing anchor (never a second on-chain write). Otherwise
    it builds the domain-tagged payload over `log_digest`. Production adapters
    prepare a signed transaction, persist its exact evidence and the shared
    writer fence, submit once, and confirm the same hash. Pending evidence always
    resumes confirmation; only a proven expired-absent transaction permits a
    fresh preparation, still subject to the unchanged epoch deadline. Small
    report/test adapters retain their `anchor_commitment` interface.
    """
    phase = on_phase or (lambda _name: None)
    phase("index")
    pending = index.get_anchor_submission(finalized.epoch_id)
    marker = read_anchor_writer_pending(writer_lock_path)
    txid = pending.extrinsic_hash if pending is not None else None
    if marker is not None and marker.get("epoch_id") == finalized.epoch_id:
        txid = marker["extrinsic_hash"]
    async with anchor_writer_lock(
        writer_lock_path, timeout_seconds=writer_lock_timeout_seconds,
        pending_extrinsic_hash=txid,
    ):
        if marker is not None and marker.get("epoch_id") == finalized.epoch_id:
            pending = _restore_marker_submission(index, marker)
            if pending.outcome in {"expired_absent", "operator_acknowledged"}:
                clear_anchor_writer_pending(writer_lock_path, extrinsic_hash=pending.extrinsic_hash)
        # Re-check inside the cross-process lane. Two local callers may both have
        # observed the pre-anchor row before either acquired it.
        record = index.get(finalized.epoch_id)
        if record is not None and record.anchored:
            if pending is not None:
                if pending.extrinsic_hash != record.anchor_txid:
                    raise AnchorSubmissionHold("pending signed hash differs from indexed anchor")
                index.finish_anchor_submission(
                    pending.extrinsic_hash, outcome="confirmed", outcome_at=now.isoformat(),
                    evidence={"block": record.anchor_block},
                )
                clear_anchor_writer_pending(writer_lock_path, extrinsic_hash=pending.extrinsic_hash)
            return anchor_from_record(record)

        split_methods = [getattr(chain, name, None) for name in (
            "prepare_authority_anchor", "submit_authority_anchor", "confirm_authority_anchor"
        )]
        if all(callable(method) for method in split_methods):
            return await _anchor_with_durable_submission(
                finalized, chain=chain, index=index, netuid=netuid, now=now,
                anchor_hotkey=anchor_hotkey, writer_lock_path=writer_lock_path,
                on_phase=phase, search_blocks=search_blocks, confirmation_depth=confirmation_depth,
                verification_timeout_seconds=verification_timeout_seconds,
                verification_poll_seconds=verification_poll_seconds, on_expired=on_expired,
            )
        if pending is not None or any(callable(method) for method in split_methods):
            raise AnchorSubmissionHold("chain adapter cannot safely resume durable anchor evidence")

        payload = anchor_payload(finalized.epoch_id, netuid, finalized.log_digest)
        await require_commitment_capacity(
            chain,
            netuid=netuid,
            hotkey=anchor_hotkey,
            payload=payload,
            operation=f"epoch {finalized.epoch_id} anchor",
        )
        phase("anchor_submit")
        try:
            txid = await chain.anchor_commitment(payload)
        finally:
            phase("anchor_confirm")
        # Capture the commitment record's exact inclusion block before another
        # process can overwrite the one-slot account. The adapter's nested lock is
        # re-entrant in this task, so this outer lane spans write + read-back. A
        # real adapter's independent read socket can trail the write socket's
        # finality notification by several blocks; poll that already-submitted
        # receipt here and NEVER resubmit merely because visibility is lagging.
        read_anchor_block = getattr(chain, "read_anchor_block", None)
        if callable(read_anchor_block):
            finalized_reader = getattr(chain, "finalized_block", None)
            archive_reader = getattr(chain, "read_anchor_at", None)
            if callable(finalized_reader) and callable(archive_reader):
                receipt = await wait_for_finalized_anchor_receipt(
                    chain,
                    netuid=netuid,
                    anchor_id=finalized.epoch_id,
                    domain=ANCHOR_DOMAIN,
                    expected_digest=finalized.log_digest,
                    operation=f"epoch {finalized.epoch_id} anchor",
                    timeout_seconds=verification_timeout_seconds,
                    poll_seconds=verification_poll_seconds,
                )
                block = receipt.block
            else:
                # Compatibility for small report/test adapters. Production guards
                # require finalized + historical archive seams and always take the
                # exact verifier above.
                block = await asyncio.to_thread(
                    read_anchor_block,
                    netuid=netuid,
                    epoch_id=finalized.epoch_id,
                    domain=ANCHOR_DOMAIN,
                )
                if block is None:
                    raise RuntimeError(
                        f"epoch {finalized.epoch_id} anchor write returned {txid!r}, "
                        "but its inclusion block is not readable; refusing an "
                        "unverifiable pointer"
                    )
        else:
            try:
                block = await asyncio.to_thread(chain.current_block)
            except Exception:  # pragma: no cover - legacy report adapter
                block = None
        phase("index")
        updated = index.set_anchor(finalized.epoch_id, txid=txid, block=block)
        return anchor_from_record(updated)
