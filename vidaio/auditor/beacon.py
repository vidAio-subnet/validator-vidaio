"""Chain-derived sampling beacon (round-3 #10; round-5 #2; UN-GRINDABLE round-6 #2).

The sample seed mixes in an UNPREDICTABLE beacon so the authority cannot steer
which items get audited when it builds the manifest (see ``vidaio.auditor.sampling``).
That guarantee only holds if the beacon is obtained INDEPENDENTLY of the authority
AND the authority cannot PREDICT/GRIND it when it builds — and anchors — the epoch log.

The evolution of this seam is the story of closing successive grind holes:

* round-3 #10: the runner seeded from ``pointer.anchor.txid`` — an AUTHORITY-supplied
  pointer field, not bound into the anchored log. Trivially chosen.
* round-5 #2 attempt 1: derive from ``read_anchor`` (the anchored LOG DIGEST). Still
  grindable — the authority CONSTRUCTS that digest before anchoring (it controls
  grindable fields like ``created_at``) and can search for a digest whose beacon lands
  its misreporting outside every auditor's sample.
* round-5 #2 attempt 2: derive from the anchor's INCLUSION-BLOCK entropy. Better, but
  STILL rerollable: the authority does not choose which block its extrinsic lands in,
  but it DOES choose WHEN to (re-)anchor. Anchoring is idempotent on the payload, so it
  can RE-ANCHOR the same epoch log in a later block — each re-anchor yields a fresh
  inclusion-block beacon — and repeat until the fresh beacon's sample misses its misreporting.

* round-6 #2 (this module): the beacon is a FUTURE FINALIZED BLOCK HASH

      beacon = block_hash(close_block + K)

  where ``close_block`` is the epoch-close block recorded IN the committed/anchored epoch
  log (a fixed field of the log the auditor is verifying) and ``K`` is a fixed
  confirmation depth (``beacon_confirmation_depth``, production minimum 20). Because
  the beacon block depends ONLY on the epoch's ``close_block`` — NOT on when or
  whether the authority (re-)anchors — re-anchoring the same payload in a later
  block CANNOT reroll it. The
  hash of a not-yet-produced block is unknowable at log-build time, so the sample cannot
  be pre-steered even by a grinding authority.

Two residual grind vectors this module still closes explicitly:

1. BACKDATING ``close_block`` — the authority could try to set ``close_block`` to an
   already-produced past block whose hash it already knows, choosing an epoch whose
   beacon it can precompute. In production, ``close_block`` is verified against the exact
   archive-proven ``SubnetEpochIndex`` transition (see ``chain_beacon``), not a synthetic
   tempo grid. Report-mode adapters retain their deterministic fixed-grid check.
2. ANCHORING LATE — the authority could commit the item set AFTER the beacon block is
   already known, then pick items to spare. So the anchor's inclusion block must be
   ``<= close_block + K`` (the set was committed BEFORE the beacon could be known); an
   anchor block past the beacon block is a :class:`BeaconGrindRisk` (REFUSE).

Fail-CLOSED (never fall back to any authority-chosen value):
- an adapter missing the ``block_hash`` / ``read_anchor_block`` seams => BeaconUnavailable
  (HOLD; we NEVER fall back to ``read_anchor`` / the log digest — the fallback is the hole);
- the chain holds no anchor yet (``read_anchor_block`` -> None) => BeaconUnavailable
  ("no anchor yet" — HOLD, retry);
- the beacon block is not finalized yet (``finalized_block < close_block + K``, or
  ``block_hash`` -> None) => BeaconUnavailable (HOLD until it is finalized — the finding-#3
  cursor retries later);
- any read fails (raises) => BeaconUnavailable (an unreadable chain HOLDS, never mistaken
  for "no anchor").

The seam is PURE: the runner passes the concrete adapter and compatibility inputs.
Production adapters supply ``epoch_close_block`` and ``finalized_block``; report mode
falls back to ``blocks_per_epoch`` and its deterministic current head.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BeaconUnavailable(RuntimeError):
    """The sampling beacon could not be established this pass — HOLD and retry.

    A benign, RECOVERABLE condition: the chain holds no anchor yet, the beacon block
    is not finalized yet, or a read failed. The auditor HOLDS the epoch (submits
    nothing, does not advance the finding-#3 cursor) and retries next pass. NEVER a
    signal to sample with an authority-chosen value — no beacon => no honest sample =>
    hold (#10, the project design record §4/§5).
    """


class BeaconGrindRisk(RuntimeError):
    """The anchored log is trying to STEER the beacon — REFUSE loudly, do not retry.

    A POSITIVE tamper/grind signal, not a transient unavailability: the log's
    ``close_block`` is not the exact archive-proven runtime transition for its
    ``epoch_id`` (a backdating attempt), or the anchor's inclusion block is AFTER the
    beacon block (the item set was committed after the beacon was knowable). Either
    lets the authority pick a beacon it can precompute, so the auditor must not sample
    media off this log. Distinct from :class:`BeaconUnavailable` because retrying
    cannot fix it — the loud REFUSE blocks the finding-#3 cursor as an alarm.
    """


@runtime_checkable
class AnchorReadable(Protocol):
    """The tamper-evidence read seam: a ChainAdapter that reads the anchored digest.

    Mirrors ``vidaio.chain.adapter.EpochAnchorReadable``: returns the 64-hex LOG
    DIGEST anchored for ``(netuid, epoch_id)``, None to say POSITIVELY the chain holds
    no such anchor, and RAISES on a read/transport failure. Kept as the third
    verification leg (``sha256(bytes)==pointer==anchor``, finding #4); it is NOT the
    beacon source. Structural, so any adapter satisfies it without an import dependency.
    """

    def read_anchor(self, *, netuid: int, epoch_id: int, domain: str) -> str | None: ...


@runtime_checkable
class AnchorBlockReadable(Protocol):
    """The anchor-INCLUSION-BLOCK read seam (round-6 #2, step 3).

    Mirrors ``vidaio.chain.adapter.EpochAnchorBlockReadable``: returns the block NUMBER
    the epoch's anchor commitment landed at, None to say POSITIVELY the chain holds no
    anchor yet, and RAISES on a read/transport failure (never substitutes None). Used to
    confirm the item set was committed BEFORE the beacon block could be known
    (``anchor_block <= close_block + K``). Structural, so any adapter satisfies it.
    """

    def read_anchor_block(
        self, *, netuid: int, epoch_id: int, domain: str
    ) -> int | None: ...


@runtime_checkable
class BlockHashReadable(Protocol):
    """The BEACON read seam: a block HASH by height (round-6 #2).

    Mirrors ``vidaio.chain.adapter.BlockHashReadable``: returns the hash of a PRODUCED
    block (64-hex), None to say POSITIVELY the block is not yet produced (the beacon is
    not finalized — HOLD, retry), and RAISES on a read/transport failure. This is what
    the authority cannot predict when it builds the log — the hash of a future finalized
    block fixed by the epoch's ``close_block``. Structural, so any adapter satisfies it.
    """

    def block_hash(self, block_number: int) -> str | None: ...


def _canonical_close_block(epoch_id: int, blocks_per_epoch: int) -> int:
    """Report/sim's deterministic fixed-grid close (never used for Bittensor)."""
    return (epoch_id + 1) * blocks_per_epoch - 1


def chain_beacon(
    chain: object,
    *,
    netuid: int,
    epoch_id: int,
    domain: str,
    close_block: int,
    confirmation_depth: int,
    blocks_per_epoch: int | None,
    current_block: int,
    anchor_block: int | None = None,
) -> str:
    """The round-6 sampling beacon: ``block_hash(close_block + K)`` (UN-GRINDABLE).

    The beacon block is ``close_block + confirmation_depth`` — a FUTURE FINALIZED block
    fixed by the epoch's ``close_block`` (a field of the anchored log). Its hash is
    unknowable at log-build time and does NOT change when the authority re-anchors, so a
    grinding/re-anchoring authority cannot steer which items are sampled.

    The acceptance sequence:

    1. ``beacon_block = close_block + K``.
    2. ``close_block`` MUST equal the chain-native, archive-proven close for
       ``epoch_id`` when the adapter exposes ``epoch_close_block``. Report/sim uses
       its deterministic fixed grid. A mismatch is :class:`BeaconGrindRisk`.
    3. the epoch MUST be anchored AND its anchor inclusion block ``<= beacon_block`` (the
       item set was committed BEFORE the beacon could be known). No anchor yet ->
       :class:`BeaconUnavailable` (HOLD); anchor block AFTER the beacon block ->
       :class:`BeaconGrindRisk` (REFUSE).
    4. the beacon block MUST be finalized. Production reads GRANDPA finality through
       ``finalized_block``; report mode treats its deterministic head as final.
    5. ``beacon = block_hash(beacon_block)``; a None (not produced) or a read failure ->
       :class:`BeaconUnavailable` (HOLD).

    NEVER falls back to ``read_anchor`` / the log digest / any authority-supplied value —
    that fallback is exactly the grind hole. Pure and side-effect free.
    """
    beacon_block = close_block + confirmation_depth

    # Step 2 — reject a BACKDATED close_block (a grind attempt on the beacon block).
    # Bittensor's schedule is stateful: tempo changes reset LastEpochBlock, owners
    # may trigger early and capacity may defer a due epoch. Its production adapter
    # therefore proves the exact historical SubnetEpochIndex transition. Only the
    # deterministic report/sim adapters use the fixed-grid compatibility formula.
    read_epoch_close = getattr(chain, "epoch_close_block", None)
    if callable(read_epoch_close):
        try:
            expected_close = read_epoch_close(netuid=netuid, epoch_id=epoch_id)
        except Exception as exc:  # noqa: BLE001 - unreadable archive history => HOLD
            raise BeaconUnavailable(
                f"could not verify subnet {netuid} epoch {epoch_id}'s chain-native close "
                f"from historical epoch transitions: {type(exc).__name__}: {exc}"
            ) from exc
        if expected_close is None:
            raise BeaconUnavailable(
                f"subnet {netuid} epoch {epoch_id} has not closed at the finalized head"
            )
        close_basis = "the archive-proven SubnetEpochIndex transition"
    else:
        if blocks_per_epoch is None:
            raise BeaconUnavailable(
                "the chain adapter exposes neither epoch_close_block nor a report-mode "
                "blocks_per_epoch fallback"
            )
        expected_close = _canonical_close_block(epoch_id, blocks_per_epoch)
        close_basis = (
            f"the report-mode fixed grid ((epoch_id + 1) * {blocks_per_epoch} - 1)"
        )
    if close_block != expected_close:
        raise BeaconGrindRisk(
            f"epoch {epoch_id}: close_block {close_block} is not canonical under "
            f"{close_basis} (expected {expected_close})"
            " — the authority may have backdated close_block to a block whose hash it"
            " already knows; REFUSING to sample media off it"
        )

    # Step 3 — the item set must be COMMITTED (anchored) before the beacon is knowable.
    if anchor_block is None:
        read_anchor_block = getattr(chain, "read_anchor_block", None)
        if not callable(read_anchor_block):
            raise BeaconUnavailable(
                f"the wired chain adapter does not implement read_anchor_block, so the anchor"
                f" inclusion block for epoch {epoch_id} cannot be checked against the beacon"
                " block; the auditor HOLDS rather than fall back to an authority-grindable"
                " value"
            )
        try:
            anchor_block = read_anchor_block(
                netuid=netuid, epoch_id=epoch_id, domain=domain
            )
        except BeaconUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any read/transport failure => HOLD
            raise BeaconUnavailable(
                f"could not read the anchor inclusion block for epoch {epoch_id}; the sampling"
                f" beacon is unavailable and the auditor HOLDS: {type(exc).__name__}: {exc}"
            ) from exc
    if anchor_block is None:
        raise BeaconUnavailable(
            f"the chain holds NO anchor for epoch {epoch_id} yet — the sampling beacon is"
            " unavailable; the auditor HOLDS rather than seed its sample from an"
            " authority-supplied value"
        )
    if anchor_block > beacon_block:
        raise BeaconGrindRisk(
            f"epoch {epoch_id}: the anchor landed at block {anchor_block}, AFTER the beacon"
            f" block {beacon_block} (= close_block {close_block} + K {confirmation_depth})"
            " — the item set was committed after the beacon was knowable, so the authority"
            " could have chosen items to spare; REFUSING to sample media off it (round-6 #2)"
        )

    # Step 4 — the beacon block must be finalized already. A Bittensor adapter
    # MUST supply GRANDPA finality; its cached best head is not a substitute.
    read_finalized = getattr(chain, "finalized_block", None)
    if callable(read_finalized):
        try:
            finalized_height = int(read_finalized())
        except Exception as exc:  # noqa: BLE001 - unknown finality => HOLD
            raise BeaconUnavailable(
                f"could not read the finalized block for epoch {epoch_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        finalized_height = current_block  # deterministic report/sim compatibility
    if finalized_height < beacon_block:
        raise BeaconUnavailable(
            f"epoch {epoch_id}: the beacon block {beacon_block} is not finalized yet"
            f" (finalized_block {finalized_height} < {beacon_block}) — the beacon is not"
            " knowable yet; the auditor HOLDS and retries once it is"
        )

    # Step 5 — the beacon IS the future-finalized block hash.
    block_hash = getattr(chain, "block_hash", None)
    if not callable(block_hash):
        raise BeaconUnavailable(
            f"the wired chain adapter does not implement block_hash, so the round-6"
            f" beacon block_hash(close_block + K) for epoch {epoch_id} cannot be derived;"
            " the auditor HOLDS rather than fall back to an authority-grindable value"
        )
    try:
        beacon = block_hash(beacon_block)
    except BeaconUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - any read/transport failure => HOLD
        raise BeaconUnavailable(
            f"could not read block_hash({beacon_block}) for epoch {epoch_id}; the sampling"
            f" beacon is unavailable and the auditor HOLDS: {type(exc).__name__}: {exc}"
        ) from exc
    if not beacon:
        raise BeaconUnavailable(
            f"epoch {epoch_id}: block {beacon_block} is not produced yet (block_hash ->"
            " None) — the beacon block is not finalized; the auditor HOLDS and retries"
        )
    return beacon
