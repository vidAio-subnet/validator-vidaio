from __future__ import annotations

import pytest

from vidaio.chain import CommitmentCapacity, InMemoryChain
from vidaio.services.commitment_capacity import (
    CommitmentCapacityError,
    require_commitment_capacity,
)


class CapacityChain(InMemoryChain):
    def __init__(self, capacity: CommitmentCapacity) -> None:
        super().__init__()
        self.capacity = capacity
        self.requests: list[tuple[int, str]] = []

    def commitment_capacity(self, netuid: int, hotkey: str) -> CommitmentCapacity:
        self.requests.append((netuid, hotkey))
        return self.capacity


def _capacity(*, maximum: int, used: int = 0) -> CommitmentCapacity:
    return CommitmentCapacity(
        netuid=85,
        hotkey="authority-hotkey",
        block=123,
        current_epoch=7,
        usage_epoch=7,
        max_space=maximum,
        reported_used_space=used,
        used_space=used,
    )


@pytest.mark.asyncio
async def test_capacity_gate_accounts_for_pallet_minimum_and_reserve() -> None:
    chain = CapacityChain(_capacity(maximum=228))

    observed = await require_commitment_capacity(
        chain,
        netuid=85,
        hotkey="authority-hotkey",
        payload=b"small",
        operation="challenge anchor",
        reserve_payload_bytes=128,
    )

    assert observed is chain.capacity
    assert chain.requests == [(85, "authority-hotkey")]


@pytest.mark.asyncio
async def test_capacity_gate_refuses_to_spend_epoch_anchor_reserve() -> None:
    chain = CapacityChain(_capacity(maximum=227))

    with pytest.raises(CommitmentCapacityError, match="write_charge=100") as caught:
        await require_commitment_capacity(
            chain,
            netuid=85,
            hotkey="authority-hotkey",
            payload=b"small",
            operation="competition anchor",
            reserve_payload_bytes=128,
        )

    assert "reserved_for_epoch_anchor=128" in str(caught.value)


@pytest.mark.asyncio
async def test_report_adapter_without_capacity_seam_remains_supported() -> None:
    assert (
        await require_commitment_capacity(
            InMemoryChain(),
            netuid=85,
            hotkey="",
            payload=b"small",
            operation="report anchor",
        )
        is None
    )
