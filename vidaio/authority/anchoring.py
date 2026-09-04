"""Anchoring — bind a finalized epoch's log_digest on chain (tamper-evidence root).

After the finalizer writes a `FinalizedEpoch` to the object store, the authority
ANCHORS its `log_digest` on chain via an injected `ChainAdapter`. The anchored
digest is the root a validator verifies its mirrored bytes against
(`sha256(bytes) == snapshot_digest == on-chain anchored digest`,
the project design record §4/§5, build-wave 4). In report/chainsim mode the
`InMemoryChain`/chainsim simply records the payload; in bittensor mode the real
adapter submits a `set_commitment` extrinsic — the SAME code path, only the
adapter behind the Protocol differs.

The commitment payload is a small, domain-tagged, <=128-byte value over the
log_digest (the commitments-module payload style — a domain tag + a digest —
without pulling in the merkle/ledger machinery, which is for score-packet sets, not
a single epoch-log root).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from vidaio.authority.api import AnchorRecord, anchor_from_record
from vidaio.authority.finalizer import FinalizedEpoch
from vidaio.authority.index import EpochIndex
from vidaio.chain.adapter import ChainAdapter
from vidaio.chain.anchor_receipt import (
    DEFAULT_ANCHOR_RECEIPT_POLL_SECONDS,
    DEFAULT_ANCHOR_RECEIPT_TIMEOUT_SECONDS,
    wait_for_finalized_anchor_receipt,
)
from vidaio.chain.anchor_writer import anchor_writer_lock
from vidaio.services.commitment_capacity import require_commitment_capacity

#: Versioned domain tag for the epoch-log anchor payload. Bump on any change to the
#: payload byte contract (it would change every anchored value).
ANCHOR_DOMAIN = "vidaio.epoch.anchor.v1"


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
) -> AnchorRecord:
    """Anchor the finalized epoch's log_digest on chain and record the txid.

    Idempotent per epoch: if the epoch is already anchored in the index this is a
    NO-OP that returns the existing anchor (never a second on-chain write). Otherwise
    it builds the domain-tagged payload over `log_digest`, calls
    `chain.anchor_commitment`, captures the block it landed at (when the adapter can
    report it), and records the anchor in the index. `now` is accepted for the
    finalize->anchor call contract; the finalized_at timestamp is recorded at
    finalize time.
    """
    del now  # reserved by the finalize->anchor contract; anchor time is the chain's
    async with anchor_writer_lock(
        writer_lock_path, timeout_seconds=writer_lock_timeout_seconds
    ):
        # Re-check inside the cross-process lane. Two local callers may both have
        # observed the pre-anchor row before either acquired it.
        record = index.get(finalized.epoch_id)
        if record is not None and record.anchored:
            return anchor_from_record(record)

        payload = anchor_payload(finalized.epoch_id, netuid, finalized.log_digest)
        await require_commitment_capacity(
            chain,
            netuid=netuid,
            hotkey=anchor_hotkey,
            payload=payload,
            operation=f"epoch {finalized.epoch_id} anchor",
        )
        txid = await chain.anchor_commitment(payload)
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
        updated = index.set_anchor(finalized.epoch_id, txid=txid, block=block)
        return anchor_from_record(updated)
