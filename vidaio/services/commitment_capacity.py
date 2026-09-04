"""Fail-closed Commitments-pallet capacity gates for authority writers.

The current Bittensor Commitments pallet meters bytes per ``(netuid, account,
subnet epoch)``.  Every write consumes ``max(100, payload_bytes)`` from the
runtime's mutable ``MaxSpace`` budget.  The generic Subtensor transaction-rate
limit is not this gate.

Production Bittensor adapters expose ``commitment_capacity``.  Report adapters
predate that read seam and are deliberately feature-detected so local/chainsim
tests keep exercising the common write path without inventing runtime state.
The production adapter contract independently requires the seam, so a real
deployment can never take that compatibility branch.
"""

from __future__ import annotations

import asyncio

from vidaio.chain.adapter import ChainAdapter, CommitmentCapacity


# Epoch-log anchors are the audit root and have the tight close+K deadline.  Every
# lower-priority authority writer leaves enough room for the largest legal epoch
# anchor payload, even though today's canonical payload is usually smaller.
EPOCH_ANCHOR_CAPACITY_RESERVE_BYTES = 128


class CommitmentCapacityError(RuntimeError):
    """A commitment write cannot safely fit in the current subnet epoch."""


async def require_commitment_capacity(
    chain: ChainAdapter,
    *,
    netuid: int,
    hotkey: str,
    payload: bytes,
    operation: str,
    reserve_payload_bytes: int = 0,
) -> CommitmentCapacity | None:
    """Prove that ``payload`` fits, optionally retaining another write's charge.

    Returns the block-pinned capacity snapshot for logging/metrics, or ``None``
    only for a legacy/report adapter without the read seam.  Any malformed read,
    identity mismatch, or insufficient budget raises before the extrinsic is
    attempted.
    """
    reader = getattr(chain, "commitment_capacity", None)
    if not callable(reader):
        return None
    signer = hotkey.strip()
    if not signer:
        raise CommitmentCapacityError(
            f"{operation}: commitment signer hotkey is empty; cannot read its "
            "per-epoch capacity"
        )
    if reserve_payload_bytes < 0:
        raise ValueError("reserve_payload_bytes must be non-negative")
    try:
        capacity = await asyncio.to_thread(
            reader,
            netuid=netuid,
            hotkey=signer,
        )
    except Exception as exc:
        raise CommitmentCapacityError(
            f"{operation}: cannot prove Commitments-pallet capacity for "
            f"{signer!r} on subnet {netuid}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(capacity, CommitmentCapacity):
        raise CommitmentCapacityError(
            f"{operation}: commitment_capacity returned "
            f"{type(capacity).__name__}, expected CommitmentCapacity"
        )
    if capacity.netuid != netuid or capacity.hotkey != signer:
        raise CommitmentCapacityError(
            f"{operation}: capacity snapshot identity mismatch "
            f"(requested netuid={netuid}, hotkey={signer!r}; got "
            f"netuid={capacity.netuid}, hotkey={capacity.hotkey!r})"
        )

    required = capacity.required_space(len(payload))
    reserved = (
        capacity.required_space(reserve_payload_bytes) if reserve_payload_bytes else 0
    )
    if capacity.remaining_space < required + reserved:
        raise CommitmentCapacityError(
            f"{operation}: insufficient Commitments-pallet capacity at block "
            f"{capacity.block} in subnet epoch {capacity.current_epoch}: "
            f"used={capacity.used_space}/{capacity.max_space}, "
            f"remaining={capacity.remaining_space}, write_charge={required}, "
            f"reserved_for_epoch_anchor={reserved}"
        )
    return capacity


__all__ = [
    "CommitmentCapacityError",
    "EPOCH_ANCHOR_CAPACITY_RESERVE_BYTES",
    "require_commitment_capacity",
]
