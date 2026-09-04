"""Media-free builders for the auditor logic tests (sampling, weights, aggregation).

Bundles here carry arbitrary bytes and an ItemScore-shaped packet; a
``StaticRecomputer`` stands in for the scoring engine, so PASS/FAIL is driven by
whether the packet matches the fixed recompute — no ffmpeg required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from vidaio.audit.bundle import AuditBundle, LifecycleStage, build_bundle
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.auditor import Auditor
from vidaio.authority import EpochFinalizer, ScoredItem
from vidaio.chain.adapter import ChainNeuron, InMemoryChain
from vidaio.epoch.log import (
    AuditFileKind,
    AuditFileRef,
    AuditManifest,
    EpochLog,
    MinerCensusEntry,
)
from vidaio.tokenomics import (
    MinerSnapshot,
    TokenomicsConfig,
)
from vidaio.tokenomics.ewma import accumulate

NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
SCORER = "scoring-1.0.0+abc123def456"
BACKENDS = {"vmaf": "3.0.0", "ffmpeg": "7.1"}
HONEST_METRICS = {"compression_rate": 0.125, "vmaf": 93.42, "final_score": 0.81}
HONEST_SCORE = HONEST_METRICS["final_score"]
BURN_UID = 999
DECAY = TokenomicsConfig().ewma_decay
#: The default close_block honest_log pins (also the default metagraph_chain uses it so the
#: metagraph identities agree with the log by construction).
CLOSE_BLOCK = 360_000


def fold(prior: float, scores) -> float:
    """EWMA-fold ``scores`` over ``prior`` at the default decay (test convenience)."""
    v = prior
    for s in scores:
        v = accumulate(v, s, DECAY)
    return v


def make_packet(
    *, challenge_id: str, item_id: str, miner_hotkey: str, score: float | None = None,
    metrics: dict[str, float] | None = None, cycle_sequence: int = 0,
    excluded: bool = False, **overrides: Any,
) -> bytes:
    metrics = metrics or dict(HONEST_METRICS)
    final = metrics.get("final_score", 0.0)
    packet: dict[str, Any] = {
        "item_id": item_id,
        "challenge_id": challenge_id,
        "track": "compression",
        "score": final if score is None else score,
        "gate_passed": True,
        "violations": [],
        "skips": [],
        "miner_hotkey": miner_hotkey,
        "content_digest": sha256_hex(f"out-{item_id}".encode()),
        "breakdown": {"kind": "compression", "final": final, "vmaf_threshold": 90.0},
        "metrics": metrics,
        "scorer_version": SCORER,
        "backend_versions": BACKENDS,
        "pieapp_start_frame": None,
        "scoring_config_digest": sha256_hex(b"scoring config"),
        "canonicalization_plan_digest": sha256_hex(b"plan"),
        # committed evidence-bound earning fields: the monotonic fold
        # ORDER anchor and the exclusion marker the auditor re-reads from the packet.
        "cycle_sequence": cycle_sequence,
        "excluded": excluded,
    }
    packet.update(overrides)
    return canonical_json_bytes(packet)


def make_fake_bundle(
    store: LocalFsStore,
    *,
    challenge_id: str,
    item_id: str,
    miner_hotkey: str,
    packet: bytes | None = None,
    committed_track: str = "compression",
    dispatch_ordering_key: int = 0,
) -> AuditBundle:
    """Store a full artifact set (arbitrary bytes) and return its bundle.

    The DAG_REVEAL artifact is a REAL challenge-commitment preimage carrying the
    pre-dispatch committed ``track`` + ``dispatch_ordering_key``, so
    the auditor's earning path can read the committed fold order/track back from the
    anchored commitment (not the score packet). ``dispatch_ordering_key`` is kept a
    SEPARATE knob from the packet's ``cycle_sequence`` precisely so a reordered-fold
    case (packet sequence reassigned, but the committed dispatch order fixed) can be
    exercised.
    """
    from vidaio.challenge.commitment import ChallengeCommitment

    dag_bytes = ChallengeCommitment.preimage_payload(
        f"asset-{item_id}",
        sha256_hex(b"dag-" + item_id.encode()),
        1,
        SCORER,
        committed_track,
        dispatch_ordering_key,
    )
    if packet is None:
        packet = make_packet(
            challenge_id=challenge_id, item_id=item_id, miner_hotkey=miner_hotkey,
            cycle_sequence=dispatch_ordering_key,
        )
    return build_bundle(
        challenge_id=challenge_id,
        item_id=item_id,
        miner_hotkey=miner_hotkey,
        commitment_hash=sha256_hex(dag_bytes),
        stage=LifecycleStage.POST_RETIREMENT,
        challenge_input=store.put(f"in-{item_id}".encode(), ArtifactKind.CHALLENGE_INPUT),
        miner_output=store.put(f"out-{item_id}".encode(), ArtifactKind.MINER_OUTPUT),
        manifest=store.put(canonical_json_bytes({"item": item_id}), ArtifactKind.MANIFEST),
        score_packet=store.put(packet, ArtifactKind.SCORE_PACKET),
        reference_original=store.put(f"ref-{item_id}".encode(), ArtifactKind.REFERENCE_ORIGINAL),
        dag_reveal=store.put(dag_bytes, ArtifactKind.DAG_REVEAL),
        scorer_version=SCORER,
        backend_versions=BACKENDS,
        created_at="2026-08-21T12:00:00+00:00",
    )


def refs_for(
    bundle: AuditBundle, *, source: str = "inference", committed_track: str = "compression"
) -> tuple[AuditFileRef, AuditFileRef]:
    """The manifest's two refs for a bundle: its AUDIT_BUNDLE binding + SCORE_PACKET."""
    common = dict(challenge_id=bundle.challenge_id, item_id=bundle.item_id, source=source)
    return (
        AuditFileRef(
            kind=AuditFileKind.AUDIT_BUNDLE, digest=bundle.bundle_digest(), **common
        ),
        AuditFileRef(
            kind=AuditFileKind.SCORE_PACKET, digest=bundle.score_packet.digest,
            committed_track=committed_track, **common,
        ),
    )


def scored_item(
    bundle: AuditBundle,
    uid: int,
    *,
    score: float = HONEST_SCORE,
    seq: int = 0,
    source: str = "inference",
    committed_track: str = "compression",
) -> ScoredItem:
    """A ScoredItem bound to ``bundle``'s committed packet (score + fold-order seq).

    The committed packet ``make_fake_bundle`` stored records ``score`` at
    ``cycle_sequence=seq``; this attests the SAME value/order so the auditor's
    evidence-bound earning re-fold verifies.
    """
    return ScoredItem(
        uid=uid,
        hotkey=f"hk{uid}",
        challenge_id=bundle.challenge_id,
        item_id=bundle.item_id,
        bundle_digest=bundle.bundle_digest(),
        packet_digest=bundle.score_packet.digest,
        committed_track=committed_track,
        source=source,
        score=score,
        cycle_sequence=seq,
    )


def folded_miner(uid: int, score: float = HONEST_SCORE, track: str = "compression") -> MinerSnapshot:
    """A miner whose accumulate_score is the genesis fold of a single ``score`` cycle."""
    return make_miner(uid, fold(0.0, [score]), track=track)


def make_miner(
    uid: int,
    score: float,
    track: str = "compression",
) -> MinerSnapshot:
    # (The retention-window kwargs were REMOVED with the retention multiplier for v1 —
    # retention removed — owner decision; an internal review.)
    return MinerSnapshot(
        uid=uid,
        hotkey=f"hk{uid}",
        coldkey=f"ck{uid}",
        ip=f"10.0.0.{uid}",
        track=track,
        accumulate_score=score,
    )


# (The committed windowed-evidence helpers — _reg_block / _end_stake / window_input /
# window_inputs_for / with_windows — were REMOVED with the retention multiplier for v1
# — retention removed — owner decision; an internal review.)


def metagraph_chain(
    miners, *, close_block: int = CLOSE_BLOCK, close_block_time: datetime = NOW
) -> InMemoryChain:
    """An InMemoryChain metagraph holding ``miners``' identities.

    The auditor reads this INDEPENDENTLY of the authority log to bind each nonzero-weight
    uid's uid->hotkey/coldkey/ip and to re-derive the IP/coldkey dedup. Honest fixtures build
    it from the SAME miner list the log was built from, so the identities match and the epoch
    stays CLEAN; adversarial tests build it from the TRUE identities while the log lies. (The
    committed-window endpoint cross-check was REMOVED with the retention multiplier for v1 —
    retention removed — owner decision; an internal review.)

    an internal review: the block clock is anchored so ``block_time(close_block) ==
    close_block_time`` — honest fixtures pass ``close_block_time = log.created_at`` (via
    ``MetagraphAuditor``), so the created_at binding PASSES; a backdated created_at disagrees.
    """
    return InMemoryChain(
        _neurons=[
            ChainNeuron(
                uid=m.uid, hotkey=m.hotkey, coldkey=m.coldkey, ip=m.ip,
                alpha_stake=0.0, emission=0.0,
            )
            for m in miners
        ],
        block_time_anchor=(close_block, close_block_time),
    )


class MetagraphAuditor(Auditor):
    """Test Auditor that auto-wires its close-block metagraph from the audited log's OWN
    honest identities when no explicit ``chain`` is injected.

    Honest fixtures thus get a metagraph consistent with ``log.miner_census`` with no call-site
    churn — exactly "supply a metagraph consistent with the log" — so the snapshot binding
    PASSES and honest epochs stay CLEAN. Adversarial snapshot tests inject an EXPLICIT
    (tampered/true) ``chain`` instead, which takes precedence, exercising the real seam.
    """

    def audit_epoch(self, epoch_log, *args, **kwargs):
        if self._chain is not None:  # an explicit (adversarial/true) chain was injected
            return super().audit_epoch(epoch_log, *args, **kwargs)
        # Rebuild the metagraph from THIS log's complete registered census each call (multi-epoch
        # runners audit different sets — never cache the first epoch's identities), then restore. The
        # metagraph is pinned to the log's OWN close_block so the 1b window endpoints agree.
        # Pin the block clock so block_time(close_block) == the log's created_at, so the honest
        # created_at binding PASSES; the window endpoints agree via close_block.
        self._chain = metagraph_chain(
            epoch_log.miner_census,
            close_block=epoch_log.close_block,
            close_block_time=epoch_log.created_at,
        )
        try:
            return super().audit_epoch(epoch_log, *args, **kwargs)
        finally:
            self._chain = None


# Shared schema-v14 competition-evidence builders live in the dedicated competition
# fixtures; this module keeps the inference audit helpers small.


def rebuild_log(log: EpochLog, **overrides) -> EpochLog:
    """Rebuild an EpochLog from ``log``'s fields with ``overrides`` (bypasses build_log).

    Lets adversarial tests tamper ONE field (audit_manifest / weight_shares) while keeping an
    internally-consistent (constructible) EpochLog — the auditor's independent re-derivation,
    not the finalizer's producer-side refusal, is what must catch the tamper.
    """
    fields = dict(
        schema_version=log.schema_version,
        epoch_id=log.epoch_id,
        close_block=log.close_block,
        scorer_version=log.scorer_version,
        created_at=log.created_at,
        prior_log_digest=log.prior_log_digest,
        burn_uid=log.burn_uid,
        competition_result=log.competition_result,
        reward_window_state=log.reward_window_state,
        miners=log.miners,
        miner_census=log.miner_census,
        weight_shares=log.weight_shares,
        weight_u16=log.weight_u16,
        weight_vector_digest=log.weight_vector_digest,
        audit_manifest=log.audit_manifest,
    )
    if "miners" in overrides and "miner_census" not in overrides:
        # Keep census-only registrations, but make any deliberately-tampered economic
        # identity self-consistent with the schema so the auditor can test it against the
        # independent metagraph rather than failing at construction.
        census = {entry.uid: entry for entry in log.miner_census}
        for miner in overrides["miners"]:
            census[miner.uid] = MinerCensusEntry.from_miner(miner)
        overrides["miner_census"] = tuple(census.values())
    if "weight_shares" in overrides and "burn_uid" not in overrides:
        # Fixed per-track / competition pools make the canonical sink coexist
        # with ordinary earners.  A tampered vector that removes that sink must
        # also clear the marker or it is locally malformed before the auditor can
        # exercise the intended independent check.
        candidate = fields["burn_uid"]
        if candidate is not None and overrides["weight_shares"].get(candidate, 0.0) <= 0.0:
            overrides["burn_uid"] = None
    fields.update(overrides)
    return EpochLog(**fields)


def honest_log(
    miners: list[MinerSnapshot],
    manifest: AuditManifest,
    *,
    epoch_id: int = 100,
    close_block: int = CLOSE_BLOCK,
) -> EpochLog:
    """An HONEST epoch log whose weights follow from its inputs (finalizer path).

    (The committed-window-evidence attachment was REMOVED with the retention multiplier for
    v1 — retention removed — owner decision; an internal review.
    Competition arguments default absent so legacy callers build inference-only logs.)
    """
    finalizer = EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER)
    return finalizer.build_log(
        epoch_id=epoch_id,
        close_block=close_block,
        snapshots=tuple(miners),
        burn_uid=BURN_UID,
        audit_manifest=manifest,
        now=NOW,
    )
