"""AT MOST ONE ON-CHAIN COMMITMENT PER COMPETITION.

The chain write used to happen BEFORE the guarded DB transition, so two
concurrent anchor requests carrying DIFFERENT payloads could both reach the chain
while only whichever returned first was recorded — leaving a second, valid,
completely untracked commitment on chain for the same competition.

The anchoring right is now CLAIMED in the DB first (BEGIN IMMEDIATE: still
SCHEDULED, no commitment_root, no in-flight claim), carrying the EXACT payload
digest about to be written. A second request fails the claim and is refused before
touching the chain. A crash mid-anchor leaves a re-checkable claim rather than an
unknown: once stale, the identical payload may only be archive-verified again
without a write; a DIFFERENT/new write needs explicit operator resolution.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest

from vidaio.chain import ChainCommitmentRecord, CommitmentCapacity
from vidaio.audit.commitments import reward_parameter_digest
from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import AnchorClaimRefused, AnchorError
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.services.commitment_capacity import CommitmentCapacityError

from orchestrator_support import (
    BASELINE,
    START,
    T0,
    FakeRunner,
    RecordingChain,
    build_manifest,
    phase,
    seed_items,
)

TOKEN = "control-token-for-tests"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BASELINE_IMAGE_DIGEST = FakeRunner.digest_for(BASELINE["tree_sha"])
CONFIG_A = TokenomicsConfig()
CONFIG_B = TokenomicsConfig(minimum_payout_score=0.11)
REWARD_A = reward_parameter_digest(CONFIG_A)
REWARD_B = reward_parameter_digest(CONFIG_B)


class Clock:
    def __init__(self, value=T0) -> None:
        self.value = value

    def __call__(self):
        return self.value


class SlowChain(RecordingChain):
    """Records the anchor, then yields — so a concurrent request gets to run."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def anchor_commitment(self, payload: bytes) -> str:
        self.entered.set()
        await self.release.wait()
        return await super().anchor_commitment(payload)


class AlmostFullChain(RecordingChain):
    def commitment_capacity(self, netuid: int, hotkey: str) -> CommitmentCapacity:
        return CommitmentCapacity(
            netuid=netuid,
            hotkey=hotkey,
            block=10,
            current_epoch=3,
            usage_epoch=3,
            max_space=3_100,
            reported_used_space=2_873,
            used_space=2_873,
        )


def _make(orchestrator_factory, fixture_repos, chain, clock=None):
    orch = orchestrator_factory(
        repos=fixture_repos, chain=chain, clock=clock or Clock(), control_token=TOKEN
    )
    manifest = build_manifest(baseline=BASELINE)
    orch.create_competition(manifest, T0)
    return orch, manifest.competition_id


async def test_capacity_refusal_happens_before_claim_or_chain_write(
    orchestrator_factory, fixture_repos
):
    chain = AlmostFullChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)

    with pytest.raises(CommitmentCapacityError, match="reserved_for_epoch_anchor=128"):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )

    assert chain.anchor_calls == []
    assert pers.open_anchor_claim(orch.conn, cid) is None


async def test_control_api_reports_capacity_refusal_as_retryable_503(
    orchestrator_factory, fixture_repos, tmp_path
):
    chain = AlmostFullChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)
    seed_items(orch, cid, tmp_path / "capacity-items", n=1)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app),
        base_url="http://control",
    ) as client:
        response = await client.post(
            f"/competitions/{cid}/anchor",
            headers=AUTH,
            json={
                "baseline_image_digest": BASELINE_IMAGE_DIGEST,
                "reward_param_digest": REWARD_A,
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "commitment_capacity_unavailable"
    assert chain.anchor_calls == []
    assert pers.open_anchor_claim(orch.conn, cid) is None


# ---- the race ---------------------------------------------------------------------


async def test_concurrent_anchors_with_different_digests_write_the_chain_once(
    orchestrator_factory, fixture_repos
):
    """THE regression. Two requests, two different reward-param digests, one loop.

    Exactly one reaches the chain; the other is refused with a machine-readable
    code and never anchors anything.
    """
    chain = SlowChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)

    first = asyncio.create_task(
        orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    )
    await chain.entered.wait()  # the first request is inside the chain write

    # Model an operator changing the active reward policy while the first exact
    # payload is already in flight. The second request is internally valid but
    # different, and still must be stopped by the pre-chain claim.
    orch.tokenomics = CONFIG_B
    with pytest.raises(AnchorClaimRefused) as refused:
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_B,  # a DIFFERENT payload
            now=T0,
        )
    assert refused.value.code == "anchor_in_progress"

    chain.release.set()
    result = await first

    assert len(chain.anchor_calls) == 1
    assert chain.anchored == [result.payload]
    assert result.anchor_block == 1
    assert result.anchor_block_hash == chain.block_hash(1)
    assert result.finalized_block >= result.anchor_block
    receipt = pers.latest_verified_anchor_receipt(orch.conn, cid)
    assert receipt is not None
    assert receipt["anchor_netuid"] == 85
    assert receipt["payload_hex"] == result.payload.hex()
    assert receipt["anchor_block"] == result.anchor_block
    assert receipt["anchor_block_hash"] == result.anchor_block_hash
    assert receipt["archive_verified"] is True
    comp = repo.get_competition(orch.conn, cid)
    assert comp.commitment_root == result.root
    # The claim is resolved by completion: a later anchor is refused for the
    # right reason (already anchored), not because a claim is stuck open.
    assert pers.open_anchor_claim(orch.conn, cid) is None


async def test_the_same_payload_concurrently_is_also_refused(
    orchestrator_factory, fixture_repos
):
    """Even identical requests must not both hit the chain: the commitment would
    be duplicated on chain, and a duplicate extrinsic is still an extrinsic."""
    chain = SlowChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)

    first = asyncio.create_task(
        orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    )
    await chain.entered.wait()
    with pytest.raises(AnchorClaimRefused) as refused:
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    assert refused.value.code == "anchor_in_progress"
    chain.release.set()
    await first
    assert len(chain.anchor_calls) == 1


async def test_the_guards_are_checked_before_the_chain_not_after(
    orchestrator_factory, fixture_repos
):
    """Already anchored / past SCHEDULED must not produce another chain write."""
    chain = RecordingChain()
    clock = Clock()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain, clock)

    await orch.anchor_competition(
        cid, baseline_image_digest=BASELINE_IMAGE_DIGEST, reward_param_digest=REWARD_A, now=T0
    )
    assert len(chain.anchor_calls) == 1

    with pytest.raises(AnchorClaimRefused) as already:
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    assert already.value.code == "already_anchored"
    assert len(chain.anchor_calls) == 1  # NOT re-written

    clock.value = START
    await orch.step(START)
    assert phase(orch, cid) is Phase.ENROLLING
    with pytest.raises(AnchorClaimRefused):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=START,
        )
    assert len(chain.anchor_calls) == 1


async def test_an_unknown_competition_never_reaches_the_chain(
    orchestrator_factory, fixture_repos
):
    chain = RecordingChain()
    orch, _cid = _make(orchestrator_factory, fixture_repos, chain)
    # The manifest read already refuses it; the claim's own unknown_competition
    # branch is the belt-and-braces guard behind that. Either way: no chain write.
    with pytest.raises((AnchorClaimRefused, KeyError)):
        await orch.anchor_competition(
            "no-such-competition",
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    assert chain.anchor_calls == []


# ---- crash recovery ---------------------------------------------------------------


class BrokenChain(RecordingChain):
    async def anchor_commitment(self, payload: bytes) -> str:
        self.anchor_calls.append(bytes(payload))
        raise OSError("substrate node unreachable")


class LostResponseChain(RecordingChain):
    async def anchor_commitment(self, payload: bytes) -> str:
        await super().anchor_commitment(payload)
        raise OSError("write response was lost after inclusion")


async def test_lost_write_response_is_recovered_from_exact_archive_receipt(
    orchestrator_factory, fixture_repos
):
    chain = LostResponseChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)

    result = await orch.anchor_competition(
        cid,
        baseline_image_digest=BASELINE_IMAGE_DIGEST,
        reward_param_digest=REWARD_A,
        now=T0,
    )

    assert result.recorded is True
    assert result.tx_id is None
    assert result.write_response_recovered is True
    assert len(chain.anchor_calls) == 1
    receipt = pers.latest_verified_anchor_receipt(orch.conn, cid)
    assert receipt is not None and receipt["write_response_recovered"] is True


async def test_archive_mismatch_never_marks_or_resubmits(
    orchestrator_factory, fixture_repos
):
    class ArchiveMismatchChain(RecordingChain):
        def read_commitment_record(self, *, netuid, block_number=None):
            record = super().read_commitment_record(
                netuid=netuid, block_number=block_number
            )
            if record is not None and block_number is not None:
                return ChainCommitmentRecord(payload=b"different", block=record.block)
            return record

    chain = ArchiveMismatchChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)

    with pytest.raises(AnchorError, match="finalized archive state"):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )

    assert len(chain.anchor_calls) == 1
    assert repo.get_competition(orch.conn, cid).commitment_root is None
    assert pers.latest_verified_anchor_receipt(orch.conn, cid) is None
    assert pers.open_anchor_claim(orch.conn, cid) is not None


async def test_lagging_finality_is_polled_without_resubmitting(
    orchestrator_factory, fixture_repos
):
    class LaggingFinalityChain(RecordingChain):
        finality_reads = 0

        def finalized_block(self):
            self.finality_reads += 1
            return 0 if self.finality_reads == 1 else self.current_block()

    chain = LaggingFinalityChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)

    result = await orch.anchor_competition(
        cid,
        baseline_image_digest=BASELINE_IMAGE_DIGEST,
        reward_param_digest=REWARD_A,
        now=T0,
    )

    assert result.recorded is True
    assert chain.finality_reads == 2
    assert len(chain.anchor_calls) == 1


async def test_a_failed_write_leaves_an_ambiguous_claim_that_blocks_a_new_payload(
    orchestrator_factory, fixture_repos
):
    """A timed-out extrinsic may still have landed. Anchoring a DIFFERENT payload
    over it would be exactly the double-commitment this protocol prevents."""
    chain = BrokenChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)
    with pytest.raises(AnchorError):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    assert len(chain.anchor_calls) == 1  # ambiguous writes are never blind-retried
    assert repo.get_competition(orch.conn, cid).commitment_root is None

    claim = pers.open_anchor_claim(orch.conn, cid)
    assert claim is not None and claim["payload_digest"]
    assert pers.EVENT_ANCHOR_FAILED in {
        e["event_type"] for e in repo.list_events(orch.conn, cid)
    }

    # Fresh claim: even the identical payload waits.
    with pytest.raises(AnchorClaimRefused) as fresh:
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    assert fresh.value.code == "anchor_in_progress"

    # Stale claim + a DIFFERENT payload: refused as ambiguous, not silently retried.
    stale = T0 + timedelta(seconds=orch.cfg.anchor_claim_stale_seconds + 1)
    orch.tokenomics = CONFIG_B
    with pytest.raises(AnchorClaimRefused) as ambiguous:
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_B,
            now=stale,
        )
    assert ambiguous.value.code == "anchor_ambiguous"
    assert "release_anchor_claim" in str(ambiguous.value)
    assert len(chain.anchor_calls) == 1  # nothing new was written


async def test_a_stale_claim_is_recovered_by_readback_without_resubmitting(
    orchestrator_factory, fixture_repos
):
    """Crash recovery verifies the already-landed bytes and consumes no capacity."""

    class TemporarilyUnreadableChain(RecordingChain):
        readable = False

        def read_commitment_record(self, **kwargs):
            if not self.readable:
                raise OSError("independent archive socket unavailable")
            return super().read_commitment_record(**kwargs)

    chain = TemporarilyUnreadableChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)
    with pytest.raises(AnchorError):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    claim = pers.open_anchor_claim(orch.conn, cid)
    assert claim is not None

    assert len(chain.anchor_calls) == 1
    chain.readable = True  # the independent archive socket came back
    stale = T0 + timedelta(seconds=orch.cfg.anchor_claim_stale_seconds + 1)
    result = await orch.anchor_competition(
        cid,
        baseline_image_digest=BASELINE_IMAGE_DIGEST,
        reward_param_digest=REWARD_A,
        now=stale,
    )
    assert result.recorded is True
    assert result.tx_id is None
    assert result.write_response_recovered is True
    assert len(chain.anchor_calls) == 1
    assert repo.get_competition(orch.conn, cid).commitment_root == result.root
    assert pers.open_anchor_claim(orch.conn, cid) is None


async def test_an_operator_can_resolve_an_ambiguous_claim(
    orchestrator_factory, fixture_repos
):
    """The documented escape hatch, deliberately manual (like clear_halt)."""
    chain = BrokenChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)
    with pytest.raises(AnchorError):
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    assert pers.open_anchor_claim(orch.conn, cid) is not None

    assert (
        orch.release_anchor_claim(
            cid, "ops@vidaio", "checked the chain: nothing landed", T0
        )
        is True
    )
    assert pers.open_anchor_claim(orch.conn, cid) is None
    # A second release is a no-op, not an error.
    assert orch.release_anchor_claim(cid, "ops@vidaio", "again", T0) is False

    healthy = RecordingChain()
    orch.chain = healthy
    orch.tokenomics = CONFIG_B
    result = await orch.anchor_competition(
        cid, baseline_image_digest=BASELINE_IMAGE_DIGEST, reward_param_digest=REWARD_B, now=T0
    )
    assert result.recorded is True
    assert len(healthy.anchor_calls) == 1


async def test_a_crash_after_the_chain_write_recovers_from_the_recorded_root(
    orchestrator_factory, fixture_repos
):
    """The root is checked BEFORE the claim, so a claim left open by a crash after
    the root landed never blocks anything — the competition is simply anchored."""
    chain = RecordingChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)
    result = await orch.anchor_competition(
        cid, baseline_image_digest=BASELINE_IMAGE_DIGEST, reward_param_digest=REWARD_A, now=T0
    )
    # Simulate the crash window: a stray, unresolved claim after a landed anchor.
    with pers.txn(orch.conn):
        pers.record_anchor_claim(
            orch.conn, cid, payload_digest="ab" * 32, root=result.root, now=T0
        )
    assert pers.open_anchor_claim(orch.conn, cid) is not None

    with pytest.raises(AnchorClaimRefused) as refused:
        await orch.anchor_competition(
            cid,
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            reward_param_digest=REWARD_A,
            now=T0,
        )
    assert refused.value.code == "already_anchored"
    assert len(chain.anchor_calls) == 1


# ---- through the control API ------------------------------------------------------


async def test_the_control_api_reports_409_and_exposes_the_claim(
    orchestrator_factory, fixture_repos, tmp_path
):
    chain = SlowChain()
    orch, cid = _make(orchestrator_factory, fixture_repos, chain)
    seed_items(orch, cid, tmp_path / "anchor-control-items", n=1)
    body_a = {"baseline_image_digest": BASELINE_IMAGE_DIGEST, "reward_param_digest": REWARD_A}
    body_b = {"baseline_image_digest": BASELINE_IMAGE_DIGEST, "reward_param_digest": REWARD_B}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app), base_url="http://control"
    ) as client:
        first = asyncio.create_task(
            client.post(f"/competitions/{cid}/anchor", headers=AUTH, json=body_a)
        )
        await asyncio.wait_for(chain.entered.wait(), timeout=2.0)
        orch.tokenomics = CONFIG_B
        second = await client.post(
            f"/competitions/{cid}/anchor", headers=AUTH, json=body_b
        )
        assert second.status_code == 409
        assert second.json()["detail"]["code"] == "anchor_in_progress"

        # The in-flight claim is visible to an operator on the status route.
        status = (await client.get(f"/competitions/{cid}", headers=AUTH)).json()
        assert status["anchor_claim"]["payload_digest"]

        chain.release.set()
        assert (await first).status_code == 200

        # Once anchored, a repeat is refused BEFORE the chain (it used to write).
        orch.tokenomics = CONFIG_A
        again = await client.post(
            f"/competitions/{cid}/anchor", headers=AUTH, json=body_a
        )
        assert again.status_code == 409
        assert again.json()["detail"]["code"] == "already_anchored"
        assert len(chain.anchor_calls) == 1

        released = await client.post(
            f"/competitions/{cid}/anchor/release",
            headers=AUTH,
            json={"operator": "ops", "reason": "nothing to resolve"},
        )
        assert released.status_code == 200
        assert released.json()["released"] is False  # no open claim

        assert (
            await client.post(f"/competitions/{cid}/anchor/release", json={})
        ).status_code == 401
