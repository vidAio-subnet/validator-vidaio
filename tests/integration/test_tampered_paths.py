"""Integrity properties across module seams: every tampering path must fail loudly.

Each test starts from a private golden world (fresh_world) and spoofs exactly one
thing, then asserts the seam that catches it and the stable failure code it emits.
Bundle verifications run STRICT with every anchor supplied — including the anchors
an adversary would control in the worst case — so each test isolates the one check
that still catches the tampering.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import timedelta

import pytest

from vidaio.audit import (
    ArtifactKind,
    LifecycleStage,
    MERKLE_EXCLUSION,
    PACKET_INCONSISTENT,
    REVEAL_INVALID,
    SCORE_MISMATCH,
    build_bundle,
    merkle_proof,
    merkle_root,
    sha256_hex,
    verify_bundle,
)
from vidaio.audit import COMMITMENT_MISMATCH
from vidaio.challenge import (
    RevealBeforeResolutionError,
    RevealBeforeRetireError,
    RevealedCommitment,
    WeakSeedError,
    checkout_asset,
    build_dag,
    dag_rng_from_seed,
    get_asset,
    make_challenge,
    record_challenge,
    resolve_challenge,
    retire_asset,
    reveal_commitment,
    verify_reveal,
)
from vidaio.challenge import canonical_json_dumps as challenge_canonical
from vidaio.competition import Phase, ScorePacketError
from vidaio.competition import repository as comp_repo
from vidaio.scoring import TRACK_COMPRESSION
from vidaio.tokenomics import (
    EXCLUDED_SCORE,
    MinerSnapshot,
    build_weight_vector,
)

from integration_support import (
    COMPETITION_ID,
    NOW_POST,
    PRIVATE_SEED,
    SCORER_VERSION,
    START,
    T0,
    GoldenWorld,
    build_manifest,
    deep_reveal_verifier,
    fleet,
)


def _tampered_bundle(world: GoldenWorld, *, score_packet=None, dag_reveal=None, commitment_hash=None):
    """The honest hk-a bundle with exactly one piece swapped for a tampered value."""
    honest = world.bundles["hk-a"]
    return build_bundle(
        challenge_id=honest.challenge_id,
        item_id=honest.item_id,
        miner_hotkey=honest.miner_hotkey,
        commitment_hash=commitment_hash or honest.commitment_hash,
        stage=LifecycleStage.POST_RETIREMENT,
        challenge_input=honest.challenge_input,
        miner_output=honest.miner_output,
        manifest=honest.manifest,
        score_packet=score_packet or honest.score_packet,
        reference_original=honest.reference_original,
        dag_reveal=dag_reveal or honest.dag_reveal,
        scorer_version=honest.scorer_version,
        backend_versions=honest.backend_versions,
        created_at=honest.created_at,
    )


def _adversary_anchors(world: GoldenWorld, bundle, extra_digest: str | None = None) -> dict:
    """Worst-case anchors: the adversary publishes their own bundle digest and (when
    `extra_digest` is given) a merkle root that INCLUDES the tampered packet. The
    honest reveal verifier stays wired. Whatever still fails under these anchors is
    caught by recompute alone — the check no adversary can satisfy."""
    digests = sorted(world.packet_digests + ([extra_digest] if extra_digest else []))
    return {
        "expected_bundle_digest": bundle.bundle_digest(),
        "published_root": merkle_root(digests),
        "inclusion_proof": merkle_proof(digests, bundle.score_packet.digest),
        "reveal_verifier": deep_reveal_verifier,
    }


# ---- score tampering: recompute is the check an adversary cannot satisfy -----------


def test_inflated_score_packet_fails_score_mismatch(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    packet = json.loads(world.packets["hk-a"])
    packet["metrics"]["final_score"] = 0.999  # inflate the recorded score
    packet["score"] = 0.999
    tampered_bytes = json.dumps(packet).encode()
    tampered_ref = world.store.put(tampered_bytes, ArtifactKind.SCORE_PACKET)
    bundle = _tampered_bundle(world, score_packet=tampered_ref)

    # even with adversary-controlled digest + merkle anchors, recompute catches it
    report = verify_bundle(
        bundle, world.store, world.recomputer(),
        **_adversary_anchors(world, bundle, tampered_ref.digest),
    )
    assert not report.passed
    failed = {c.name: c.code for c in report.failures()}
    assert failed == {
        "score_recompute:final_score": SCORE_MISMATCH,
        "score_recompute:score": SCORE_MISMATCH,
    }


def test_edited_top_level_score_with_honest_metrics_fails(fresh_world: GoldenWorld) -> None:
    """The review probe: metrics agree with recompute, only the top-level score — the
    value rankings consume — was edited. Must fail, not slip through."""
    world = fresh_world
    packet = json.loads(world.packets["hk-a"])
    packet["score"] = 0.999  # metrics stay honest
    tampered_ref = world.store.put(json.dumps(packet).encode(), ArtifactKind.SCORE_PACKET)
    bundle = _tampered_bundle(world, score_packet=tampered_ref)

    report = verify_bundle(
        bundle, world.store, world.recomputer(),
        **_adversary_anchors(world, bundle, tampered_ref.digest),
    )
    assert not report.passed
    failed = {c.name: c.code for c in report.failures()}
    assert failed == {"score_recompute:score": SCORE_MISMATCH}


def test_gate_failed_packet_with_nonzero_score_is_inconsistent(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    packet = json.loads(world.packets["hk-a"])
    packet["gate_passed"] = False
    packet["score"] = 0.999  # gates-first requires 0.0
    tampered_ref = world.store.put(json.dumps(packet).encode(), ArtifactKind.SCORE_PACKET)
    bundle = _tampered_bundle(world, score_packet=tampered_ref)

    report = verify_bundle(
        bundle, world.store, world.recomputer(),
        **_adversary_anchors(world, bundle, tampered_ref.digest),
    )
    assert not report.passed
    failed = {c.name: c.code for c in report.failures()}
    assert failed["packet_consistency"] == PACKET_INCONSISTENT


# ---- reveal tampering ---------------------------------------------------------------


def test_reveal_with_wrong_seed_fails_verification(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    honest = world.revealed
    assert verify_reveal(honest)  # sanity: the true preimage verifies
    tampered = RevealedCommitment(
        clean_asset_id=honest.clean_asset_id,
        dag_digest=honest.dag_digest,
        seed=honest.seed + 1,  # different seed than committed
        scorer_version=honest.scorer_version,
        track=honest.track,
        dispatch_ordering_key=honest.dispatch_ordering_key,
        commit_hash=honest.commit_hash,
        revealed_at=honest.revealed_at,
    )
    assert not verify_reveal(tampered)
    # a different DAG than committed fails the same way
    tampered_dag = RevealedCommitment(
        clean_asset_id=honest.clean_asset_id,
        dag_digest=sha256_hex(b"some-other-dag"),
        seed=honest.seed,
        scorer_version=honest.scorer_version,
        track=honest.track,
        dispatch_ordering_key=honest.dispatch_ordering_key,
        commit_hash=honest.commit_hash,
        revealed_at=honest.revealed_at,
    )
    assert not verify_reveal(tampered_dag)


def test_tampered_reveal_artifact_fails_commitment_mismatch(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    # a reveal artifact whose preimage carries a different seed than was committed
    c = world.challenge.commitment
    tampered_seed = next(
        seed
        for seed in range(c.seed + 1, c.seed + 1_000)
        if build_dag(c.track, dag_rng_from_seed(seed)).canonical_digest()
        != c.dag_digest
    )
    tampered_preimage = challenge_canonical(
        {
            "asset_id": c.clean_asset_id,
            "dag_digest": c.dag_digest,
            "dispatch_ordering_key": c.dispatch_ordering_key,
            "scorer_version": c.scorer_version,
            "seed": tampered_seed,
            "track": c.track,
        }
    ).encode()
    assert tampered_preimage != c.preimage_bytes()
    tampered_ref = world.store.put(tampered_preimage, ArtifactKind.DAG_REVEAL)
    bundle = _tampered_bundle(world, dag_reveal=tampered_ref)

    report = verify_bundle(
        bundle, world.store, world.recomputer(), **_adversary_anchors(world, bundle)
    )
    assert not report.passed
    failed = {c.name: c.code for c in report.failures()}
    assert failed["commitment_reveal"] == COMMITMENT_MISMATCH
    # The deep verifier independently rejects a seed known to regenerate another DAG.
    assert failed["dag_reveal_generation"] == REVEAL_INVALID


def test_handpicked_dag_merely_hashed_into_commitment_fails_deep_reveal(
    fresh_world: GoldenWorld,
) -> None:
    """A validator that hand-picks a corruption and commits its hash (instead of
    deriving it from the seed) produces a self-consistent commitment — hash check
    passes — but the injected deep verifier proves the DAG never regenerates."""
    world = fresh_world
    c = world.challenge.commitment
    picked_preimage = challenge_canonical(
        {
            "asset_id": c.clean_asset_id,
            "dag_digest": sha256_hex(b"hand-picked-corruption"),  # not seed-derived
            "dispatch_ordering_key": c.dispatch_ordering_key,
            "scorer_version": c.scorer_version,
            "seed": c.seed,
            "track": c.track,
        }
    ).encode()
    picked_ref = world.store.put(picked_preimage, ArtifactKind.DAG_REVEAL)
    bundle = _tampered_bundle(
        world, dag_reveal=picked_ref, commitment_hash=sha256_hex(picked_preimage)
    )

    report = verify_bundle(
        bundle, world.store, world.recomputer(), **_adversary_anchors(world, bundle)
    )
    assert not report.passed
    failed = {c.name: c.code for c in report.failures()}
    assert failed == {"dag_reveal_generation": REVEAL_INVALID}


def test_reveal_ordering_is_enforced(fresh_world: GoldenWorld) -> None:
    """Reveal needs the asset retired AND every challenge on it resolved — each
    precondition alone is insufficient."""
    world = fresh_world
    # issue a SECOND challenge from the remaining fresh challenge-split asset
    rng = random.Random(1)
    asset = checkout_asset(world.challenge_conn, rng, T0.isoformat())
    challenge = make_challenge(TRACK_COMPRESSION, asset, PRIVATE_SEED + 1, SCORER_VERSION)
    record_challenge(world.challenge_conn, challenge, T0.isoformat())
    commit_hash = challenge.commitment.commit_hash

    # live asset: no reveal
    with pytest.raises(RevealBeforeRetireError):
        reveal_commitment(world.challenge_conn, commit_hash, T0.isoformat())
    # force-retired but the challenge is still dispatched: still no reveal
    retire_asset(world.challenge_conn, asset.id, T0.isoformat(), force=True)
    with pytest.raises(RevealBeforeResolutionError):
        reveal_commitment(world.challenge_conn, commit_hash, T0.isoformat())
    # resolved + retired: reveal opens and verifies
    resolve_challenge(world.challenge_conn, challenge.challenge_id, T0.isoformat())
    revealed = reveal_commitment(world.challenge_conn, commit_hash, T0.isoformat())
    assert verify_reveal(revealed)


def test_weak_private_seed_is_rejected(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    asset = get_asset(world.challenge_conn, world.asset_id)
    with pytest.raises(WeakSeedError):
        make_challenge(TRACK_COMPRESSION, asset, 20260901, SCORER_VERSION)  # 25-bit seed


# ---- publication tampering ---------------------------------------------------------


def test_unpublished_packet_fails_merkle_exclusion(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    # mint a packet OUTSIDE the committed set (never a merkle leaf): byte-different
    # via an ignored extra key, so ONLY the merkle check isolates the tampering (the
    # packet's identity/score stay honest and every other check passes)
    packet = json.loads(world.packets["hk-a"])
    packet["_injected"] = True
    injected_ref = world.store.put(json.dumps(packet).encode(), ArtifactKind.SCORE_PACKET)
    assert injected_ref.digest not in world.packet_digests
    # no inclusion proof can exist for it under the HONEST published root
    with pytest.raises(ValueError):
        merkle_proof(world.packet_digests, injected_ref.digest)
    # presenting someone else's proof does not help
    bundle = _tampered_bundle(world, score_packet=injected_ref)
    stolen_proof = merkle_proof(world.packet_digests, world.packet_refs["hk-b"].digest)
    report = verify_bundle(
        bundle,
        world.store,
        world.recomputer(),
        expected_bundle_digest=bundle.bundle_digest(),
        published_root=world.published_root,
        inclusion_proof=stolen_proof,
        reveal_verifier=deep_reveal_verifier,
    )
    assert not report.passed
    failed = {c.name: c.code for c in report.failures()}
    assert failed == {"merkle_inclusion": MERKLE_EXCLUSION}


# ---- competition seams -------------------------------------------------------------


def test_gate_failed_nonzero_packet_rejected_at_record_time(fresh_world: GoldenWorld) -> None:
    """The old bypass — persisting a caller-chosen score alongside a gate-failed
    packet — is now impossible: the packet is the only score source and its
    gates-first invariant is enforced when it is recorded."""
    world = fresh_world
    packet = json.loads(world.packets["hk-b"])  # honest gate-failed packet
    assert packet["gate_passed"] is False
    packet["score"] = 0.35  # the tampered non-zero score the old API accepted
    with pytest.raises(ScorePacketError, match="gates-first"):
        comp_repo.record_item_score(
            world.comp_conn,
            COMPETITION_ID,
            contender_id=world.contender_ids["hk-b"],
            item_id=world.item_ids[0],
            packet_bytes=json.dumps(packet).encode(),
            now=NOW_POST,
        )


def test_packet_identity_must_match_contender_and_item(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    # hk-a's packet cannot be recorded for hk-b's contender row
    with pytest.raises(ScorePacketError, match="miner_hotkey"):
        comp_repo.record_item_score(
            world.comp_conn,
            COMPETITION_ID,
            contender_id=world.contender_ids["hk-b"],
            item_id=world.item_ids[0],
            packet_bytes=world.packets["hk-a"],
            now=NOW_POST,
        )
    # a packet minted for another challenge cannot be recorded against this item
    packet = json.loads(world.packets["hk-a"])
    packet["challenge_id"] = "some-other-challenge"
    with pytest.raises(ScorePacketError, match="does not match"):
        comp_repo.record_item_score(
            world.comp_conn,
            COMPETITION_ID,
            contender_id=world.contender_ids["hk-a"],
            item_id=world.item_ids[0],
            packet_bytes=json.dumps(packet).encode(),
            now=NOW_POST,
        )


def test_calibration_contender_can_never_hold_a_rank(fresh_world: GoldenWorld) -> None:
    world = fresh_world
    with pytest.raises(sqlite3.IntegrityError):
        world.comp_conn.execute(
            "UPDATE contenders SET final_rank = 1 WHERE contender_id = ?",
            (world.baseline_contender_id,),
        )


def test_enrollment_never_opens_without_anchored_commitment(fresh_world: GoldenWorld) -> None:
    """review #4 structurally: the SCHEDULED -> ENROLLING tick defers until the
    pre-commitment root is anchored — no anchor, no enrollment, ever."""
    world = fresh_world
    manifest = build_manifest(competition_id="comp-integ-02")
    world.engine.create_competition(world.comp_conn, manifest, T0)

    world.engine.tick(world.comp_conn, START)  # start_time passed, but no anchor
    comp = comp_repo.get_competition(world.comp_conn, "comp-integ-02")
    assert comp is not None and comp.status is Phase.SCHEDULED

    world.engine.mark_commitment_anchored(
        world.comp_conn, "comp-integ-02", "d" * 64, START + timedelta(minutes=1)
    )
    world.engine.tick(world.comp_conn, START + timedelta(minutes=2))
    comp = comp_repo.get_competition(world.comp_conn, "comp-integ-02")
    assert comp is not None and comp.status is Phase.ENROLLING


# ---- tokenomics seams --------------------------------------------------------------


def test_substituted_miner_gets_no_weight(fresh_world: GoldenWorld) -> None:
    # This inference-path check proves a miner with no genuine score, or the exclusion
    # sentinel, takes zero. Competition substitution is covered by schema-v13 tests.
    world = fresh_world
    substituted = [
        MinerSnapshot(
            uid=66, hotkey="hk-fake", coldkey="ck-fake", ip="10.0.0.66",
            track="compression", accumulate_score=0.0,  # never earned a genuine score,
        ),
        MinerSnapshot(
            uid=67, hotkey="hk-excl", coldkey="ck-excl", ip="10.0.0.67",
            track="compression", accumulate_score=EXCLUDED_SCORE,  # exclusion sentinel,
        ),
    ]
    vector = build_weight_vector(
        world.tokenomics_config, fleet() + substituted, burn_uid=0
    )
    assert vector[66] == 0.0
    assert vector[67] == 0.0
    assert sum(vector.values()) == pytest.approx(1.0)
