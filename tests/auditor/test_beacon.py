"""The chain-derived sampling beacon seam (round-3 #10; UN-GRINDABLE round-6 #2).

Round-6 #2: the beacon is a FUTURE FINALIZED BLOCK HASH — ``block_hash(close_block + K)``
— fixed by the epoch's ``close_block`` (recorded in the anchored log). Because it depends
ONLY on close_block (not on when/whether the authority (re-)anchors), RE-ANCHORING the same
payload in a later block CANNOT reroll it (that closes the round-5 inclusion-block hole,
where each re-anchor minted a fresh beacon to grind). The auditor also REFUSES a backdated
close_block and an anchor committed after the beacon block was knowable, and HOLDS until
the beacon block is finalized.
"""

from __future__ import annotations

import asyncio

import pytest

from vidaio.auditor import (
    BeaconGrindRisk,
    BeaconUnavailable,
    SamplePolicy,
    chain_beacon,
    manifest_items,
    sample_items,
)
from vidaio.authority.anchoring import ANCHOR_DOMAIN, anchor_payload
from vidaio.chain import InMemoryChain
from vidaio.chain.adapter import synthetic_block_hash

from tests.auditor.fakes import make_fake_bundle, refs_for
from vidaio.audit.store import LocalFsStore
from vidaio.epoch.log import AuditManifest

NETUID = 85
EPOCH = 42
BPE = 100
#: The canonical close block for EPOCH under BPE: (EPOCH + 1) * BPE - 1.
CLOSE = (EPOCH + 1) * BPE - 1  # 4299
K = 2
BEACON_BLOCK = CLOSE + K  # 4301
LOG_DIGEST = "d" * 64


def _chain(*, anchor_block: int, head: int, digest: str = LOG_DIGEST) -> InMemoryChain:
    """An InMemoryChain that anchored EPOCH at `anchor_block`, head advanced to `head`.

    Anchoring goes through `anchor_commitment` so BOTH `anchored` (the payload) and
    `_anchor_blocks` (the inclusion block) advance together — the same path production uses.
    """
    chain = InMemoryChain(tempo=BPE)
    if anchor_block > 1:
        chain.advance_blocks(anchor_block - 1)
    asyncio.run(chain.anchor_commitment(anchor_payload(EPOCH, NETUID, digest)))
    assert chain.current_block() == anchor_block
    if head > anchor_block:
        chain.advance_blocks(head - anchor_block)
    return chain


def _beacon(chain: InMemoryChain, *, close_block: int = CLOSE) -> str:
    return chain_beacon(
        chain, netuid=NETUID, epoch_id=EPOCH, domain=ANCHOR_DOMAIN,
        close_block=close_block, confirmation_depth=K, blocks_per_epoch=BPE,
        current_block=chain.current_block(),
    )


class _FailingBlockChain:
    """A chain whose read_anchor_block RAISES — an unreadable chain, never 'no anchor'."""

    def read_anchor_block(self, *, netuid, epoch_id, domain):
        raise RuntimeError("commitments pallet query timed out")

    def block_hash(self, n):  # pragma: no cover - not reached (read fails first)
        return synthetic_block_hash(n)


class _DigestOnlyChain:
    """A LEGACY adapter exposing only `read_anchor` — no round-6 beacon seams.

    Must NOT be silently accepted: `chain_beacon` fails CLOSED rather than fall back to
    the authority-grindable digest (round-6 #2)."""

    def read_anchor(self, *, netuid, epoch_id, domain):
        return LOG_DIGEST

    def current_block(self) -> int:
        return BEACON_BLOCK


def test_beacon_is_the_future_finalized_block_hash() -> None:
    chain = _chain(anchor_block=CLOSE, head=BEACON_BLOCK)
    beacon = _beacon(chain)
    # The beacon is the hash of the FUTURE block close_block + K, not the log digest.
    assert beacon == synthetic_block_hash(BEACON_BLOCK)
    assert beacon != LOG_DIGEST
    # read_anchor still returns the anchored digest (the finding-#4 tamper leg) — a
    # DIFFERENT value than the beacon.
    assert chain.read_anchor(netuid=NETUID, epoch_id=EPOCH, domain=ANCHOR_DOMAIN) == LOG_DIGEST


def test_re_anchoring_the_same_payload_does_not_reroll_the_beacon() -> None:
    """THE CRUX of round-6 #2: the repeat-anchor grind is closed.

    The old inclusion-block beacon changed every time the authority re-anchored the same
    payload in a later block — each re-anchor minted a fresh beacon to grind. The round-6
    beacon = block_hash(close_block + K) depends ONLY on the log's fixed close_block, so a
    chain that anchored once and one that RE-ANCHORED the same payload in later blocks
    (within the confirmation window) yield the IDENTICAL beacon."""
    once = _chain(anchor_block=CLOSE, head=BEACON_BLOCK)  # anchored once at close_block

    reanchored = _chain(anchor_block=CLOSE, head=CLOSE)  # anchored at close_block
    reanchored.advance_blocks(1)  # block CLOSE + 1
    asyncio.run(reanchored.anchor_commitment(anchor_payload(EPOCH, NETUID, LOG_DIGEST)))
    reanchored.advance_blocks(1)  # block CLOSE + 2 == BEACON_BLOCK (still <= it)
    asyncio.run(reanchored.anchor_commitment(anchor_payload(EPOCH, NETUID, LOG_DIGEST)))

    assert _beacon(once) == _beacon(reanchored) == synthetic_block_hash(BEACON_BLOCK)


def test_idempotent_reanchor_PAST_the_beacon_is_NOT_a_grind() -> None:
    """an internal review: a crash-recovery RE-ANCHOR of the SAME payload — after a crash between
    the chain write and the index update — must NOT be mis-read as a grind.

    The runner re-anchors the identical payload; the chain records a DUPLICATE identical anchor at
    a LATER block (here PAST the beacon block). The anchor-before-beacon check now reads the
    EARLIEST block of that committed payload (the ORIGINAL, timely anchor at close_block), so the
    honest epoch audits normally instead of being permanently invalidated (BeaconGrindRisk). A
    re-anchor of identical bytes is not a new commitment."""
    chain = _chain(anchor_block=CLOSE, head=CLOSE)  # original anchor at close_block (<= beacon)
    chain.advance_blocks(BEACON_BLOCK - CLOSE + 1)  # head past the beacon block
    asyncio.run(chain.anchor_commitment(anchor_payload(EPOCH, NETUID, LOG_DIGEST)))  # late re-anchor
    # EARLIEST matching-payload block = CLOSE (4299) <= BEACON_BLOCK (4301) => proceeds, real beacon.
    assert _beacon(chain) == synthetic_block_hash(BEACON_BLOCK)


def test_codex_earliest_le_beacon_is_not_a_grind_but_first_late_is_refused() -> None:
    """an internal review reproduction (anchor blocks 19 & 22, beacon block 21).

    Read the inclusion block DIRECTLY off the adapter (no canonical-close gate): an idempotent
    re-anchor of the SAME payload at blocks 19 then 22 reports the EARLIEST (19) <= beacon 21 (not
    a grind), while a genuine FIRST anchor at 22 alone reports 22 > 21 (a real late anchor)."""
    reanchored = InMemoryChain(tempo=BPE)
    reanchored.advance_blocks(18)  # -> block 19
    asyncio.run(reanchored.anchor_commitment(anchor_payload(EPOCH, NETUID, LOG_DIGEST)))
    reanchored.advance_blocks(3)  # -> block 22
    asyncio.run(reanchored.anchor_commitment(anchor_payload(EPOCH, NETUID, LOG_DIGEST)))  # same bytes
    assert (
        reanchored.read_anchor_block(netuid=NETUID, epoch_id=EPOCH, domain=ANCHOR_DOMAIN) == 19
    )  # earliest of the identical re-anchor; 19 <= 21 => NOT a grind

    late = InMemoryChain(tempo=BPE)
    late.advance_blocks(21)  # -> block 22
    asyncio.run(late.anchor_commitment(anchor_payload(EPOCH, NETUID, LOG_DIGEST)))  # first & only
    assert (
        late.read_anchor_block(netuid=NETUID, epoch_id=EPOCH, domain=ANCHOR_DOMAIN) == 22
    )  # genuinely-late first anchor; 22 > 21 => still refused


def test_a_first_anchor_genuinely_past_the_beacon_is_still_refused() -> None:
    """The genuine late-anchor grind is still caught: a FIRST/only anchor committed AFTER the
    beacon block was knowable is refused (the item set could have been chosen to spare)."""
    chain = _chain(anchor_block=BEACON_BLOCK + 1, head=BEACON_BLOCK + 1)  # first anchor past beacon
    with pytest.raises(BeaconGrindRisk):
        _beacon(chain)


def test_a_DIFFERENT_payload_anchored_past_the_beacon_is_still_refused() -> None:
    """A re-anchor of a DIFFERENT payload (not idempotent — a genuinely new commitment) past the
    beacon block is still refused: the committed (last-matching) payload's OWN earliest block is
    the late one, so choosing earliest does NOT re-attribute a content change through recovery."""
    chain = _chain(anchor_block=CLOSE, head=CLOSE, digest="a" * 64)  # earlier, DIFFERENT payload
    chain.advance_blocks(BEACON_BLOCK - CLOSE + 1)  # head past the beacon block
    asyncio.run(
        chain.anchor_commitment(anchor_payload(EPOCH, NETUID, LOG_DIGEST))  # NEW payload, late
    )
    with pytest.raises(BeaconGrindRisk):
        _beacon(chain)


def test_holds_until_the_beacon_block_is_finalized_then_proceeds() -> None:
    """HOLD when current_block < close_block + K; a later pass, head advanced past it,
    PROCEEDS with the real beacon."""
    # Head at close_block + 1 (< beacon block) -> the beacon block is not finalized.
    chain = _chain(anchor_block=CLOSE, head=CLOSE + 1)
    with pytest.raises(BeaconUnavailable):
        _beacon(chain)
    # Advance past the beacon block -> it is finalized -> the real beacon.
    chain.advance_blocks(BEACON_BLOCK - chain.current_block())
    assert _beacon(chain) == synthetic_block_hash(BEACON_BLOCK)


def test_refuses_when_the_anchor_landed_after_the_beacon_block() -> None:
    """An anchor committed AFTER close_block + K => the item set was committed after the
    beacon was knowable => REFUSE (grind risk), never sample media off it."""
    chain = _chain(anchor_block=BEACON_BLOCK + 1, head=BEACON_BLOCK + 1)
    with pytest.raises(BeaconGrindRisk):
        _beacon(chain)


def test_refuses_a_backdated_close_block() -> None:
    """A close_block that is not canonical for its epoch under the live tempo (an authority
    backdating it to a block whose hash it already knows) => REFUSE (grind risk)."""
    chain = _chain(anchor_block=CLOSE, head=BEACON_BLOCK)
    # close_block=5 is nowhere near (EPOCH+1)*BPE-1 -> not canonical.
    with pytest.raises(BeaconGrindRisk):
        _beacon(chain, close_block=5)


class _ChainNativeEpoch:
    def __init__(self, *, close: int, finalized: int, anchor: int | None = None):
        self.close = close
        self.finalized = finalized
        self.anchor = close + 1 if anchor is None else anchor
        self.close_calls = []

    def epoch_close_block(self, *, netuid, epoch_id):
        self.close_calls.append((netuid, epoch_id))
        return self.close

    def finalized_block(self):
        return self.finalized

    def read_anchor_block(self, *, netuid, epoch_id, domain):
        return self.anchor

    def block_hash(self, block_number):
        return synthetic_block_hash(block_number)


def test_production_beacon_uses_chain_native_close_not_tempo_grid() -> None:
    native_close = 7_777  # intentionally unrelated to epoch 42 / tempo 100
    chain = _ChainNativeEpoch(close=native_close, finalized=native_close + 20)
    assert chain_beacon(
        chain,
        netuid=NETUID,
        epoch_id=EPOCH,
        domain=ANCHOR_DOMAIN,
        close_block=native_close,
        confirmation_depth=20,
        blocks_per_epoch=None,
        current_block=10**9,  # ignored when GRANDPA finality seam is present
    ) == synthetic_block_hash(native_close + 20)
    assert chain.close_calls == [(NETUID, EPOCH)]


def test_production_beacon_holds_on_finalized_head_not_best_head() -> None:
    native_close = 7_777
    chain = _ChainNativeEpoch(close=native_close, finalized=native_close + 19)
    with pytest.raises(BeaconUnavailable, match="finalized_block"):
        chain_beacon(
            chain,
            netuid=NETUID,
            epoch_id=EPOCH,
            domain=ANCHOR_DOMAIN,
            close_block=native_close,
            confirmation_depth=20,
            blocks_per_epoch=None,
            current_block=native_close + 1_000,  # best head is not finality
        )


def test_production_beacon_refuses_a_non_runtime_close() -> None:
    chain = _ChainNativeEpoch(close=7_777, finalized=8_000)
    with pytest.raises(BeaconGrindRisk, match="archive-proven"):
        chain_beacon(
            chain,
            netuid=NETUID,
            epoch_id=EPOCH,
            domain=ANCHOR_DOMAIN,
            close_block=7_776,
            confirmation_depth=20,
            blocks_per_epoch=None,
            current_block=8_000,
        )


def test_beacon_fails_closed_without_the_round6_seams() -> None:
    """An adapter with only `read_anchor` (the grindable digest) => HOLD, never a fallback."""
    with pytest.raises(BeaconUnavailable):
        _beacon(_DigestOnlyChain())


def test_beacon_unavailable_when_chain_holds_no_anchor() -> None:
    """No anchor yet => HOLD, never an authority value."""
    chain = InMemoryChain(tempo=BPE)
    chain.advance_blocks(BEACON_BLOCK)  # head is finalized, but nothing anchored
    with pytest.raises(BeaconUnavailable):
        _beacon(chain)


def test_beacon_unavailable_when_the_anchor_block_read_fails() -> None:
    """An unreadable chain HOLDS — a failed read is never 'no anchor' nor a fallback."""
    with pytest.raises(BeaconUnavailable):
        chain_beacon(
            _FailingBlockChain(), netuid=NETUID, epoch_id=EPOCH, domain=ANCHOR_DOMAIN,
            close_block=CLOSE, confirmation_depth=K, blocks_per_epoch=BPE,
            current_block=BEACON_BLOCK,
        )


def test_chain_beacon_actually_steers_the_sample(tmp_path) -> None:
    """The chain-read beacon flows into sampling and moves the draw off NO_BEACON."""
    store = LocalFsStore(tmp_path / "s")
    per_uid = {}
    for uid in range(1, 21):
        bundle = make_fake_bundle(
            store, challenge_id="c1", item_id=f"i{uid}", miner_hotkey=f"hk{uid}"
        )
        per_uid[uid] = refs_for(bundle, source="inference")
    manifest = AuditManifest(per_uid=per_uid)
    policy = SamplePolicy(sample_rate=0.25, min_samples=1, max_samples=50)

    beacon = _beacon(_chain(anchor_block=CLOSE, head=BEACON_BLOCK))
    def keys(items):
        return {item.key() for item in items}
    seeded = keys(sample_items(manifest, epoch_id=EPOCH, auditor_hotkey="hkA", policy=policy, beacon=beacon))
    unseeded = keys(sample_items(manifest, epoch_id=EPOCH, auditor_hotkey="hkA", policy=policy))
    repeat = keys(sample_items(manifest, epoch_id=EPOCH, auditor_hotkey="hkA", policy=policy, beacon=beacon))

    assert seeded != unseeded  # the on-chain beacon moves the draw
    assert seeded == repeat  # reproducible given the same chain-read beacon
    assert manifest_items(manifest)  # sanity: the manifest is well-formed
