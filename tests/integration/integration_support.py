"""Shared world-builder for the cross-module integration suite.

Builds one deterministic "golden" run that threads all five foundation modules
together end to end, entirely on fakes / in-memory stores:

  challenge  -> asset pool (ingest planned + confirmed), commit-before-dispatch
                challenge from a CSPRNG-strength seed (SQLite)
  scoring    -> gates + formula + dedup over three fake miner responses and the
                baseline calibration run (real compose_item_score packets)
  competition-> full lifecycle walk with a fake clock: the pre-commitment is
                anchored while SCHEDULED (before enrollment opens
                chronologically genuine), scores recorded from packet bytes only,
                completion gated on FULL audit linkage (the tick observably defers
                until the baseline calibration row's bundle digest is linked too)
  audit      -> content-addressed store (explicit plaintext-holdout opt-in),
                post-retirement bundles for EVERY row — miners AND the
                unattributed baseline calibration run (miner_hotkey=None) — built and
                linked BEFORE the completion tick, verified STRICT with all
                anchors and a deep reveal verifier, merkle publication (every
                packet a leaf), commitment ledger
  tokenomics -> this fixture's inference-only vector; separate schema-v13 integration
                coverage proves earning competition/crown composition and audit

Everything is a pure function of the constants below — no wall clock, no
randomness outside seeded `random.Random`, no network, no subprocess. Event
timestamps advance monotonically along the real causal order (nothing is
anchored or backdated after the fact).
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vidaio.audit import (
    ArtifactKind,
    ArtifactRef,
    AuditBundle,
    AuditConfig,
    CommitmentLedger,
    CommitmentPayload,
    CommitmentStatus,
    CompetitionCommitment,
    LifecycleStage,
    PublicationRecord,
    RecomputedScore,
    build_bundle,
    build_competition_commitment,
    build_publication_record,
    canonical_json_bytes,
    make_store,
    merkle_root,
    pin_git_sha,
    reward_parameter_digest,
    sha256_hex,
)
from vidaio.challenge import (
    MIGRATIONS_DIR as CHALLENGE_MIGRATIONS_DIR,
    Challenge,
    ChallengeConfig,
    RevealedCommitment,
    checkout_asset,
    confirm_ingest_step,
    make_challenge,
    record_challenge,
    register_asset,
    release_asset,
    resolve_challenge,
    reveal_commitment,
    verify_reveal_deep,
)
from vidaio.competition import CompetitionManifest, LifecycleEngine, Phase, migrate
from vidaio.competition import repository as comp_repo
from vidaio.competition.orchestrator.persistence import record_submission_archived
from vidaio.core import apply_migrations, connect
from vidaio.chain.adapter import InMemoryChain
from vidaio.scoring import (
    TRACK_COMPRESSION,
    CompressionBreakdown,
    DedupEntry,
    DedupVerdict,
    DeterministicFakeBackend,
    GateContext,
    ItemScore,
    MediaInfo,
    ReasonCode,
    ScoringConfig,
    ValidityViolation,
    compose_item_score,
    default_pipeline,
    dedup_responses,
    score_compression,
)
from vidaio.tokenomics import (
    MinerSnapshot,
    TokenomicsConfig,
    build_weight_vector,
)

# --- fake clock (mirrors the tests/competition support conventions) -----------------

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
INGESTED_AT = T0 - timedelta(days=1)
INGEST_CONFIRMED_AT = T0 - timedelta(hours=23)
CHECKOUT_AT = T0 - timedelta(hours=2)
ANCHOR_AT = T0 + timedelta(minutes=5)  # while SCHEDULED, before enrollment opens
START = T0 + timedelta(hours=1)
ENROLL_DEADLINE = T0 + timedelta(hours=2)
FINALIZATION = T0 + timedelta(hours=3)
SCORES_AT = FINALIZATION + timedelta(hours=1)
RESOLVED_AT = SCORES_AT + timedelta(minutes=1)  # scoring finished -> challenge resolved
RETIRED_AT = SCORES_AT + timedelta(minutes=2)
# Reveal + bundling happen after retirement but BEFORE end_time: the completion
# tick is gated on full audit linkage (require_audit_linkage), so every bundle —
# the baseline calibration bundle included — must exist and be linked first.
REVEALED_AT = RETIRED_AT + timedelta(minutes=10)
BUNDLED_AT = REVEALED_AT + timedelta(hours=1)
# end_time sits past the 24h human-review window opened at SCORES_AT, so the
# AWAITING_END_TIME -> COMPLETED guard (max of end_time and review deadline) is
# governed by end_time and the review window is never truncated.
END = T0 + timedelta(hours=48)
STALL_TICK_AT = END  # first tick: the baseline row is not yet linked -> completion defers
COMPLETED_AT = END + timedelta(minutes=1)  # baseline linked -> the re-tick completes
PUBLICATION_RECORDED_AT = END + timedelta(hours=1)
PUBLICATION_ANCHORED_AT = END + timedelta(minutes=90)
PUBLISHED_AT = END + timedelta(minutes=100)
NOW_REIGN = END + timedelta(hours=2)  # inside the 24h reign from COMPLETED_AT
NOW_POST = END + timedelta(hours=25)  # after the reign lapsed

# --- deterministic identities --------------------------------------------------------

#: Fixed 255-bit constant standing in for a CSPRNG draw (secrets.randbits(256)) —
#: make_challenge rejects anything under 128 bits with WeakSeedError.
PRIVATE_SEED = 0x7D5A9C4E2B8F16A3D90E7C5B4A382F1E6D0C9B8A7F6E5D4C3B2A190887766554
SCORER_VERSION = "scorer-v1"
COMPETITION_ID = "comp-integ-01"
SEED_COMMITMENT = "a" * 64

BASELINE_ARCHIVE_BYTES = b"reference-baseline-archive-v0"
BASELINE_PROVENANCE_BYTES = b"reference-baseline-provenance-v0"
BASELINE = {
    "version": 0,
    "artifact_digest": sha256_hex(BASELINE_ARCHIVE_BYTES),
    "artifact_bytes": len(BASELINE_ARCHIVE_BYTES),
    "image_digest": sha256_hex(b"baseline-image-v1"),
    "provenance_digest": sha256_hex(BASELINE_PROVENANCE_BYTES),
    "provenance_bytes": len(BASELINE_PROVENANCE_BYTES),
    "repo_url": "https://github.com/vidaio/reference-baseline",
    "commit_sha": "b" * 40,
    "tree_sha": "c" * 40,
}
#: The baseline container image digest the owner pins BEFORE the competition — part of
#: the pre-enrollment commitment. There is no baseline hotkey/uid anywhere: the baseline is
#: a non-earning calibration baseline with no identity at any seam.
BASELINE_IMAGE_DIGEST = sha256_hex(b"baseline-image-v1")

# Fake media payloads. Byte LENGTH is the scored quantity (compression rate), and the
# bytes themselves are the content-addressed artifacts, so scoring stays recomputable
# from the audit store alone.
REF_DATA = b"REFCLIP!" * 1250  # 10_000 bytes: pristine reference / compression input
GOOD_DATA = b"MINERA!!" * 625  # 5_000 bytes -> rate 0.50 (passes)
BAD_DATA = b"MINERB!!" * 1125  # 9_000 bytes -> rate 0.90 >= 0.80 (gate-failed)
DUP_DATA = GOOD_DATA  # exact replay of hk-a's output (dedup zeroes it)
BASELINE_DATA = b"BASELINE" * 700  # 5_600 bytes -> rate 0.56 (calibration run)

VMAF = {"good": 93.0, "bad": 92.0, "baseline": 91.0}

#: The deterministic submission transcript: (miner_hotkey, output bytes, order_key).
#: order_key encodes submission precedence for the dedup pass; the transcript is
#: itself part of what an auditor holds (all outputs are stored content-addressed).
RESPONSES: list[tuple[str, bytes, str]] = [
    ("hk-a", GOOD_DATA, "2026-09-01T03:10:01"),
    ("hk-b", BAD_DATA, "2026-09-01T03:10:02"),
    ("hk-e", DUP_DATA, "2026-09-01T03:10:03"),  # replay of hk-a, received later
]

#: (creator, source) pairs precomputed against the default split salt: the first two
#: land in the "challenge" split, the third in "holdout" (never issued).
CHALLENGE_SOURCES = [("studio-alpha", "archive-a"), ("studio-gamma", "archive-a")]
HOLDOUT_SOURCE = ("studio-zeta", "archive-a")

#: item identity: the evaluation item's scoring_item_id defaults to the sealed
#: input's sha256 — the content-addressed identity every score packet must carry.
ITEM_SHA256 = sha256_hex(REF_DATA)


def _media_info(codec: str, byte_size: int) -> MediaInfo:
    return MediaInfo(
        codec=codec,
        width=1920,
        height=1080,
        fps=24.0,
        frame_count=240,
        duration=10.0,
        byte_size=byte_size,
    )


def packet_metrics(breakdown: CompressionBreakdown) -> dict[str, float | int]:
    """The audit-facing metric set. Must stay numeric-only (the audit packet parser
    only recomputes numeric values) and must exactly match what
    CompressionRecomputer reproduces from the raw artifacts."""
    return {
        "vmaf": breakdown.vmaf,
        "compression_rate": breakdown.compression_rate,
        "candidate_bytes": breakdown.candidate_bytes,
        "reference_bytes": breakdown.reference_bytes,
        "final_score": breakdown.final,
    }


def make_backend() -> DeterministicFakeBackend:
    """Fake metric backend keyed by CONTENT DIGESTS (not paths) so both the live
    scoring pass and the audit recompute address metrics the same way."""
    ref_digest = sha256_hex(REF_DATA)
    return DeterministicFakeBackend(
        vmaf={
            (ref_digest, sha256_hex(GOOD_DATA)): VMAF["good"],
            (ref_digest, sha256_hex(BAD_DATA)): VMAF["bad"],
            (ref_digest, sha256_hex(BASELINE_DATA)): VMAF["baseline"],
        },
    )


def _score_one(
    *,
    challenge_id: str,
    miner_hotkey: str | None,
    data: bytes,
    ref_data: bytes,
    backend: DeterministicFakeBackend,
    config: ScoringConfig,
    verdict: DedupVerdict | None,
) -> ItemScore:
    """Gate + score one response (miner or the hotkey-less baseline calibration run).

    Contexts supply BOTH VMAF model runs (require_secondary_vmaf fails closed) —
    the fake backend is deterministic, so the second run reproduces the first
    (model delta 0). The dedup verdict, when supplied and negative, zeroes the
    item with REPLAY_DUPLICATE via the gates-first invariant.
    """
    ref_digest = sha256_hex(ref_data)
    cand_digest = sha256_hex(data)
    vmaf = backend.compute(ref_digest, cand_digest)
    ctx = GateContext(
        track=TRACK_COMPRESSION,
        config=config,
        reference_info=_media_info("ffv1", len(ref_data)),
        candidate_info=_media_info("h264", len(data)),
        reference_path=ref_digest,
        candidate_path=cand_digest,
        vmaf_primary=vmaf,
        vmaf_secondary=vmaf,  # second pinned model run; delta 0 passes the gate
    )
    passed, violations = default_pipeline(backend).run(ctx)
    if verdict is not None and not verdict.kept:
        passed = False
        violations = violations + [
            ValidityViolation(code=ReasonCode.REPLAY_DUPLICATE, detail=verdict.detail)
        ]
    breakdown = score_compression(
        candidate_bytes=len(data),
        reference_bytes=len(ref_data),
        vmaf=vmaf,
        config=config,
    )
    return compose_item_score(
        item_id=ITEM_SHA256,
        challenge_id=challenge_id,
        track=TRACK_COMPRESSION,
        gate_passed=passed,
        violations=violations,
        skips=ctx.skips,  # config-disabled checks become part of the audit packet
        breakdown=breakdown,
        config=config,
        miner_hotkey=miner_hotkey,
        content_digest=cand_digest,
        metrics=packet_metrics(breakdown),
        backend_versions=backend.versions(),
        scorer_version=SCORER_VERSION,
    )


def dedup_verdicts(
    responses: list[tuple[str, bytes, str]],
    _backend: DeterministicFakeBackend,
    _config: ScoringConfig,
) -> dict[str, DedupVerdict]:
    return dedup_responses(
        [
            DedupEntry(key=hk, content_digest=sha256_hex(data), order_key=order)
            for hk, data, order in responses
        ]
    )


def score_batch(
    challenge: Challenge,
    ref_data: bytes,
    responses: list[tuple[str, bytes, str]],
    backend: DeterministicFakeBackend,
    config: ScoringConfig,
) -> dict[str, ItemScore]:
    """Gate + score + dedup one batch of miner responses for one challenge item."""
    verdicts = dedup_verdicts(responses, backend, config)
    return {
        hk: _score_one(
            challenge_id=challenge.challenge_id,
            miner_hotkey=hk,
            data=data,
            ref_data=ref_data,
            backend=backend,
            config=config,
            verdict=verdicts[hk],
        )
        for hk, data, _order in responses
    }


def score_calibration(
    challenge: Challenge,
    ref_data: bytes,
    data: bytes,
    backend: DeterministicFakeBackend,
    config: ScoringConfig,
) -> ItemScore:
    """Score the baseline calibration run: identical pipeline, NO identity (hotkey null),
    outside the cross-miner dedup pool."""
    return _score_one(
        challenge_id=challenge.challenge_id,
        miner_hotkey=None,
        data=data,
        ref_data=ref_data,
        backend=backend,
        config=config,
        verdict=None,
    )


class CompressionRecomputer:
    """ScoreRecomputer over the REAL scoring module.

    Recomputes every packet metric AND the authoritative top-level outcome
    (score + gate_passed) strictly from the integrity-verified artifact bytes
    (challenge input + miner output), the pinned metric backend, and the
    published submission transcript (all outputs are content-addressed in the
    store, so the dedup pass is independently reproducible). The packet is read
    ONLY for its claimed identity (miner_hotkey) — every value it records is
    re-derived, never trusted.
    """

    def __init__(
        self,
        backend: DeterministicFakeBackend,
        config: ScoringConfig,
        responses: list[tuple[str, bytes, str]],
        scorer_version: str = SCORER_VERSION,
    ) -> None:
        self._backend = backend
        self._config = config
        self._responses = responses
        self._scorer_version = scorer_version

    def recompute(self, bundle: AuditBundle, artifacts) -> RecomputedScore:
        ref_payload = artifacts[ArtifactKind.CHALLENGE_INPUT]
        cand_payload = artifacts[ArtifactKind.MINER_OUTPUT]
        ref = ref_payload.read_bytes() if isinstance(ref_payload, Path) else ref_payload
        cand = (
            cand_payload.read_bytes()
            if isinstance(cand_payload, Path)
            else cand_payload
        )
        claimed_hotkey = json.loads(artifacts[ArtifactKind.SCORE_PACKET]).get(
            "miner_hotkey"
        )
        verdicts = dedup_verdicts(self._responses, self._backend, self._config)
        fresh = _score_one(
            challenge_id=bundle.challenge_id,
            miner_hotkey=claimed_hotkey,
            data=cand,
            ref_data=ref,
            backend=self._backend,
            config=self._config,
            verdict=verdicts.get(claimed_hotkey)
            if claimed_hotkey is not None
            else None,
        )
        assert fresh.breakdown is not None
        return RecomputedScore(
            metrics={k: float(v) for k, v in packet_metrics(fresh.breakdown).items()},
            scorer_version=self._scorer_version,
            backend_versions=self._backend.versions(),
            score=fresh.score,
            gate_passed=fresh.gate_passed,
        )


def deep_reveal_verifier(dag_bytes: bytes) -> bool:
    """The reveal verifier the audit layer expects to be injected.

    Parses the DAG_REVEAL artifact bytes — the challenge commitment's canonical
    preimage JSON ({asset_id, dag_digest, seed, scorer_version}) — back into a
    RevealedCommitment and runs the challenge module's deep check: hash match AND
    the committed DAG must genuinely regenerate from the revealed seed. Anything
    unparseable or non-canonical fails closed (the raised exception is itself a
    REVEAL_INVALID finding in verify_bundle).
    """
    doc = json.loads(dag_bytes)
    revealed = RevealedCommitment(
        clean_asset_id=doc["asset_id"],
        dag_digest=doc["dag_digest"],
        seed=doc["seed"],
        scorer_version=doc["scorer_version"],
        track=doc["track"],
        dispatch_ordering_key=doc["dispatch_ordering_key"],
        # canonical preimage bytes hash to the commit hash by construction; any
        # non-canonical re-serialization changes the hash and fails verify_reveal.
        commit_hash=sha256_hex(dag_bytes),
        revealed_at="",
    )
    return verify_reveal_deep(revealed)


def build_manifest(**overrides) -> CompetitionManifest:
    data = {
        "competition_id": COMPETITION_ID,
        "track": "compression",
        "start_time": START,
        "enrollment_deadline": ENROLL_DEADLINE,
        "finalization_time": FINALIZATION,
        "end_time": END,
        "minimum_alpha_stake": 500.0,
        "scoring_factors": {
            "quality": 0.6,
            "cost_efficiency": 0.0,
            "length_coverage": 0.4,
        },
        "vmaf_threshold": 90.0,
        "sealed_vmaf_variants": [85.0, 89.0, 93.0],
        "allowed_gpus": ["L4", "L40S"],
        "evaluation_batch_size": {"min": 1, "max": 5},
        "scoring_seed_commitment": SEED_COMMITMENT,
        "container_size_limit_gb": 25.0,
        "scoring_version": SCORER_VERSION,
        "baseline": BASELINE,
    }
    data.update(overrides)
    return CompetitionManifest.model_validate(data)


def fleet() -> list[MinerSnapshot]:
    """Small inference fleet: both tracks, one miner without a full retention
    window (uid 12), one duplicate-IP miner (uid 14 shares uid 11's IP). The baseline
    has NO snapshot: it never enrolls, holds no hotkey, and no identity for it
    exists at the tokenomics seam at all."""
    mk = MinerSnapshot
    return [
        mk(
            uid=10,
            hotkey="hk-a",
            coldkey="ck-a",
            ip="10.0.0.1",
            track="compression",
            accumulate_score=0.9,
        ),
        mk(
            uid=11,
            hotkey="hk-b",
            coldkey="ck-b",
            ip="10.0.0.2",
            track="compression",
            accumulate_score=0.8,
        ),
        mk(
            uid=12,
            hotkey="hk-c",
            coldkey="ck-c",
            ip="10.0.0.3",
            track="upscaling",
            accumulate_score=0.7,
        ),  # zero retention bonus until a full window
        mk(
            uid=13,
            hotkey="hk-d",
            coldkey="ck-d",
            ip="10.0.0.4",
            track="upscaling",
            accumulate_score=0.6,
        ),
        mk(
            uid=14,
            hotkey="hk-e",
            coldkey="ck-e",
            ip="10.0.0.2",
            track="compression",
            accumulate_score=0.85,
        ),  # dup IP of uid 11 -> deduped out
    ]


@dataclass
class GoldenWorld:
    # configs
    scoring_config: ScoringConfig
    challenge_config: ChallengeConfig
    tokenomics_config: TokenomicsConfig
    audit_config: AuditConfig
    manifest: CompetitionManifest
    # challenge
    challenge_conn: sqlite3.Connection
    challenge: Challenge
    asset_id: str
    holdout_asset_id: str
    revealed: RevealedCommitment
    # scoring
    backend: DeterministicFakeBackend
    item_scores: dict[str, ItemScore]
    baseline_item_score: ItemScore
    packets: dict[str, bytes]
    baseline_packet: bytes
    # audit
    store: object  # AuditStore (LocalFsStore behind make_store)
    input_ref: ArtifactRef
    reference_ref: ArtifactRef
    manifest_ref: ArtifactRef
    dag_reveal_ref: ArtifactRef
    packet_refs: dict[str, ArtifactRef]
    baseline_packet_ref: ArtifactRef
    output_refs: dict[str, ArtifactRef]
    baseline_output_ref: ArtifactRef
    bundles: dict[str, AuditBundle]
    bundle_digests: dict[str, str]  # the anchored/published per-bundle digests
    baseline_bundle: AuditBundle  # POST_RETIREMENT, miner_hotkey=None
    baseline_bundle_digest: str
    weight_vector_ref: ArtifactRef
    packet_digests: list[str]
    published_root: str
    publication: CommitmentPayload
    pre_commitment: CommitmentPayload
    ledger: CommitmentLedger
    pre_commitment_id: int
    anchor_chain: InMemoryChain
    publication_id: int
    backup_ref: str
    # competition
    comp_conn: sqlite3.Connection
    engine: LifecycleEngine
    contender_ids: dict[str, int]
    baseline_contender_id: int
    item_ids: list[int]
    performance_ids: dict[str, int]  # hotkey (or "baseline") -> performance_history id
    baseline_final_score: float
    # audit-linkage completion gate observables (the tick before the baseline bundle
    # was linked): the open gap, the deferred (empty) tick result, and the phase
    # the competition was held in while stalled.
    stall_gaps: list[tuple[int, int]]
    stalled_transitions: list[tuple[str, Phase, Phase]]
    phase_during_stall: Phase
    # tokenomics (this legacy fixture has no earning competition input; dedicated
    # schema-v13 fixtures cover the competition/crown vector)
    miners: list[MinerSnapshot]
    weights: dict[int, float]

    def recomputer(self) -> CompressionRecomputer:
        return CompressionRecomputer(self.backend, self.scoring_config, RESPONSES)

    def verify_kwargs(self, hk: str) -> dict:
        """Every anchor STRICT verification needs, for the honest bundle of `hk`
        (`"baseline"` selects the unattributed baseline calibration bundle)."""
        from vidaio.audit import merkle_proof

        if hk == "baseline":
            bundle, digest = self.baseline_bundle, self.baseline_bundle_digest
        else:
            bundle, digest = self.bundles[hk], self.bundle_digests[hk]
        return {
            "expected_bundle_digest": digest,
            "published_root": self.published_root,
            "inclusion_proof": merkle_proof(
                self.packet_digests, bundle.score_packet.digest
            ),
            "reveal_verifier": deep_reveal_verifier,
        }


def build_golden_world(root: Path) -> GoldenWorld:
    scoring_config = ScoringConfig()
    challenge_config = ChallengeConfig()
    tokenomics_config = TokenomicsConfig()
    audit_config = AuditConfig(
        backend="local",
        local_root=root / "audit-store",
        # Test opt-in: no key-managed Envelope exists in this suite, and make_store
        # refuses a plaintext holdout at rest unless the config says so explicitly.
        allow_plaintext_holdout=True,
    )
    manifest = build_manifest()
    store = make_store(audit_config)
    baseline_archive_ref = store.put(
        BASELINE_ARCHIVE_BYTES, ArtifactKind.SUBMISSION_ARCHIVE
    )
    baseline_provenance_ref = store.put(
        BASELINE_PROVENANCE_BYTES, ArtifactKind.MANIFEST
    )
    assert baseline_archive_ref.digest == manifest.baseline.artifact_digest
    assert baseline_archive_ref.byte_size == manifest.baseline.artifact_bytes
    assert baseline_provenance_ref.digest == manifest.baseline.provenance_digest
    assert baseline_provenance_ref.byte_size == manifest.baseline.provenance_bytes

    # ---- (a) CHALLENGE: ingest (planned + confirmed), checkout, commit-first --------
    challenge_conn = connect(":memory:")
    apply_migrations(challenge_conn, CHALLENGE_MIGRATIONS_DIR)

    def _register(creator: str, source: str, payload: bytes) -> str:
        result = register_asset(
            challenge_conn,
            challenge_config,
            source_url=f"https://content.example/{creator}/{source}/clip.mp4",
            license_basis="CC-BY-4.0",
            creator=creator,
            source=source,
            content_digest=sha256_hex(payload),
            perceptual_fingerprint=f"fp-{creator}-{source}",
            resolution_tag="1080p",
            motion_tag="medium",
            content_type_tag="sports",
            ingested_at=INGESTED_AT.isoformat(),
        )
        # register_asset records PLANS only; the executor's completion facts (incl.
        # the metadata strip, which the transcode step implies) enter here.
        for step in ("fetch", "transcode", "segment"):
            confirm_ingest_step(
                challenge_conn, result.asset.id, step, INGEST_CONFIRMED_AT.isoformat()
            )
        return result.asset.id

    challenge_asset_ids = [
        _register(creator, source, REF_DATA + creator.encode())
        for creator, source in CHALLENGE_SOURCES
    ]
    holdout_asset_id = _register(*HOLDOUT_SOURCE, b"HOLDOUT!" * 1250)

    rng = random.Random(20260901)
    asset = checkout_asset(challenge_conn, rng, CHECKOUT_AT.isoformat())
    assert asset.id in challenge_asset_ids  # the holdout split is never issued
    challenge = make_challenge(
        TRACK_COMPRESSION,
        asset,
        PRIVATE_SEED,
        SCORER_VERSION,
        dag_version=challenge_config.dag_version,
    )
    record_challenge(challenge_conn, challenge, CHECKOUT_AT.isoformat())

    # ---- (b) COMPETITION: create + anchor the pre-commitment BEFORE enrollment ------
    comp_conn = connect(":memory:")
    migrate(comp_conn)
    engine = LifecycleEngine()
    engine.create_competition(comp_conn, manifest, T0)

    # The REAL pre-enrollment commitment, built at the honest point in time: the
    # manifest digest, the pinned baseline code/image identity, the dataset-selection
    # seed commitment and the reward parameters are all known before anyone enrolls.
    pre_commitment = build_competition_commitment(
        CompetitionCommitment(
            manifest_digest=manifest.manifest_digest(),
            baseline_version=manifest.baseline.version,
            baseline_artifact_digest=manifest.baseline.artifact_digest,
            baseline_provenance_digest=manifest.baseline.provenance_digest,
            baseline_tree_digest=pin_git_sha(BASELINE["tree_sha"]),
            baseline_image_digest=BASELINE_IMAGE_DIGEST,
            dataset_selection_seed_commitment=manifest.scoring_seed_commitment,
            reward_param_digest=reward_parameter_digest(
                TokenomicsConfig(competition_emissions_enabled=True)
            ),
        )
    )
    commitment_preimage_ref = store.put(
        pre_commitment.canonical_json, ArtifactKind.MANIFEST
    )
    assert commitment_preimage_ref.digest == pre_commitment.root
    ledger = CommitmentLedger.open(":memory:")
    pre_commitment_id = ledger.record(pre_commitment, T0.isoformat())
    ledger.advance(pre_commitment_id, CommitmentStatus.ANCHORED, ANCHOR_AT.isoformat())
    anchor_block = 10
    anchor_chain = InMemoryChain(
        _block=10_000,
        anchored=[pre_commitment.payload],
        _anchor_blocks=[anchor_block],
        block_time_anchor=(anchor_block, ANCHOR_AT),
    )
    anchor_block_hash = anchor_chain.block_hash(anchor_block)
    assert anchor_block_hash is not None
    engine.mark_commitment_anchored(
        comp_conn,
        COMPETITION_ID,
        pre_commitment.root,
        ANCHOR_AT,
        onchain_evidence={
            "root": pre_commitment.root,
            "anchor_netuid": 85,
            "payload_hex": pre_commitment.payload.hex(),
            "payload_digest": sha256_hex(pre_commitment.payload),
            "anchor_block": anchor_block,
            "anchor_block_hash": anchor_block_hash,
            "finalized_block": anchor_block,
            "archive_verified": True,
        },
    )

    engine.tick(comp_conn, START)  # SCHEDULED -> ENROLLING (anchored, so it opens)
    contender_ids = {
        hk: comp_repo.enroll_contender(
            comp_conn,
            COMPETITION_ID,
            hotkey=hk,
            repo_url=f"https://github.com/miners/{hk}",
            commit_sha="d" * 40,
            tree_sha="e" * 40,
            stake=1000.0,
            now=START + timedelta(minutes=5),
        )
        for hk in ("hk-a", "hk-b")
    }
    engine.tick(comp_conn, FINALIZATION)  # -> FINALIZING_SUBMISSIONS + baseline injection
    for c in comp_repo.list_contenders(comp_conn, COMPETITION_ID):
        if c.status == "ENROLLED":
            comp_repo.set_contender_status(
                comp_conn, c.contender_id, "ACCEPTED", FINALIZATION
            )
    t = FINALIZATION + timedelta(minutes=1)

    # Preserve every exact source archive before the combined backup guard. The
    # baseline archive is the active registry artifact; contender archives bind
    # their pinned source identities and remain sealed unless crowned.
    for contender in comp_repo.list_contenders(comp_conn, COMPETITION_ID):
        if contender.is_calibration:
            source_ref = baseline_archive_ref
        else:
            source_ref = store.put(
                canonical_json_bytes(
                    {
                        "repo_url": contender.repo_url,
                        "commit_sha": contender.commit_sha,
                        "tree_sha": contender.tree_sha,
                    }
                ),
                ArtifactKind.SUBMISSION_ARCHIVE,
            )
        record_submission_archived(
            comp_conn,
            COMPETITION_ID,
            contender.contender_id,
            source_ref.digest,
            source_ref.byte_size,
            t,
        )

    # Evidence-carrying backup guard: the submission set is preserved as an
    # audit-store artifact and its digest is the backup reference.
    backup_ref = store.put(
        canonical_json_bytes(
            {
                "competition_id": COMPETITION_ID,
                "submissions": [
                    {
                        "hotkey": c.hotkey,
                        "commit_sha": c.commit_sha,
                        "tree_sha": c.tree_sha,
                    }
                    for c in comp_repo.list_contenders(comp_conn, COMPETITION_ID)
                ],
            }
        ),
        ArtifactKind.MANIFEST,
    ).digest
    engine.mark_submissions_backed_up(comp_conn, COMPETITION_ID, backup_ref, t)
    engine.mark_validation_complete(comp_conn, COMPETITION_ID, t + timedelta(minutes=1))
    contenders = comp_repo.list_contenders(comp_conn, COMPETITION_ID)
    for c in contenders:
        comp_repo.set_contender_image_digest(
            comp_conn,
            c.contender_id,
            BASELINE_IMAGE_DIGEST if c.is_calibration else "1" * 64,
            t,
        )
    engine.mark_builds_complete(
        comp_conn, COMPETITION_ID, len(contenders), t + timedelta(minutes=2)
    )
    item_ids = [
        comp_repo.add_evaluation_item(
            comp_conn,
            COMPETITION_ID,
            item_index=0,
            input_sha256=ITEM_SHA256,  # scoring_item_id defaults to this
            input_bytes=len(REF_DATA),
            # The economic bridge requires the persisted bundle commitment to
            # equal the evaluation item's committed scoring-policy digest.
            threshold_commitment=challenge.commitment.commit_hash,
            challenge_id=challenge.challenge_id,
            length_seconds=10.0,
            now=t + timedelta(minutes=3),
        )
    ]
    engine.mark_evaluation_complete(comp_conn, COMPETITION_ID, t + timedelta(minutes=4))

    # ---- (c) SCORING: real packets for miners + the hotkey-less baseline run ------------
    backend = make_backend()
    item_scores = score_batch(challenge, REF_DATA, RESPONSES, backend, scoring_config)
    baseline_item_score = score_calibration(
        challenge, REF_DATA, BASELINE_DATA, backend, scoring_config
    )
    packets = {hk: score.to_json().encode() for hk, score in item_scores.items()}
    baseline_packet = baseline_item_score.to_json().encode()

    baseline_row = next(
        c
        for c in comp_repo.list_contenders(comp_conn, COMPETITION_ID)
        if c.is_calibration
    )
    outputs = {hk: data for hk, data, _ in RESPONSES}
    performance_ids: dict[str, int] = {}
    score_t = t + timedelta(minutes=5)
    for hk in ("hk-a", "hk-b"):  # the enrolled contenders
        performance_ids[hk] = comp_repo.record_item_score(
            comp_conn,
            COMPETITION_ID,
            contender_id=contender_ids[hk],
            item_id=item_ids[0],
            packet_bytes=packets[hk],
            now=score_t,
            output_bytes=len(outputs[hk]),
        )
    performance_ids["baseline"] = comp_repo.record_item_score(
        comp_conn,
        COMPETITION_ID,
        contender_id=baseline_row.contender_id,
        item_id=item_ids[0],
        packet_bytes=baseline_packet,  # miner_hotkey null, matching calibration
        now=score_t,
        output_bytes=len(BASELINE_DATA),
    )
    engine.mark_scores_persisted(comp_conn, COMPETITION_ID, SCORES_AT)

    # Scoring finished -> the challenge resolves; the single-use asset retires.
    resolve_challenge(challenge_conn, challenge.challenge_id, RESOLVED_AT.isoformat())
    asset = release_asset(
        challenge_conn,
        asset.id,
        challenge_config.retire_after_uses,
        RETIRED_AT.isoformat(),
    )
    assert asset.status == "retired"

    # ---- (d) AUDIT: reveal + bundles for EVERY row BEFORE the completion tick -------
    # (require_audit_linkage: the run cannot complete until every performance row,
    # the baseline calibration row included, carries its audit bundle digest.)
    revealed = reveal_commitment(
        challenge_conn, challenge.commitment.commit_hash, REVEALED_AT.isoformat()
    )

    input_ref = store.put(REF_DATA, ArtifactKind.CHALLENGE_INPUT)
    reference_ref = store.put(REF_DATA, ArtifactKind.REFERENCE_ORIGINAL)
    manifest_ref = store.put(manifest.canonical_json().encode(), ArtifactKind.MANIFEST)
    # The DAG_REVEAL artifact is the commitment's canonical preimage bytes — the
    # only bytes that both hash to the commitment AND deep-verify via the seed.
    dag_reveal_ref = store.put(
        challenge.commitment.preimage_bytes(), ArtifactKind.DAG_REVEAL
    )
    output_refs = {
        hk: store.put(data, ArtifactKind.MINER_OUTPUT) for hk, data, _ in RESPONSES
    }
    # The baseline's output is stored content-addressed exactly like a miner's, so the
    # calibration score is independently recomputable from the audit store too.
    baseline_output_ref = store.put(BASELINE_DATA, ArtifactKind.MINER_OUTPUT)
    packet_refs = {
        hk: store.put(packet, ArtifactKind.SCORE_PACKET)
        for hk, packet in packets.items()
    }
    baseline_packet_ref = store.put(baseline_packet, ArtifactKind.SCORE_PACKET)

    def _post_retirement_bundle(
        hotkey: str | None, output_ref: ArtifactRef, packet_ref: ArtifactRef
    ) -> AuditBundle:
        return build_bundle(
            challenge_id=challenge.challenge_id,
            item_id=ITEM_SHA256,
            miner_hotkey=hotkey,  # None = unattributed (the baseline has no identity)
            commitment_hash=challenge.commitment.commit_hash,
            stage=LifecycleStage.POST_RETIREMENT,
            challenge_input=input_ref,
            miner_output=output_ref,
            manifest=manifest_ref,
            score_packet=packet_ref,
            reference_original=reference_ref,
            dag_reveal=dag_reveal_ref,
            execution_image_digest=(
                BASELINE_IMAGE_DIGEST if hotkey is None else "1" * 64
            ),
            scorer_version=SCORER_VERSION,
            backend_versions=backend.versions(),
            created_at=BUNDLED_AT.isoformat(),
        )

    bundles = {
        hk: _post_retirement_bundle(hk, output_refs[hk], packet_refs[hk])
        for hk in packets
    }
    bundle_digests = {hk: bundle.bundle_digest() for hk, bundle in bundles.items()}
    baseline_bundle = _post_retirement_bundle(
        None, baseline_output_ref, baseline_packet_ref
    )
    baseline_bundle_digest = baseline_bundle.bundle_digest()
    # Production epoch manifests resolve bundle digests through the shared object
    # store.  Keep the golden world faithful: an event-log digest without the
    # canonical AUDIT_BUNDLE object is not independently fetchable by an auditor.
    for bundle in (*bundles.values(), baseline_bundle):
        ref = store.put(
            canonical_json_bytes(bundle.model_dump(mode="json")),
            ArtifactKind.AUDIT_BUNDLE,
        )
        assert ref.digest == bundle.bundle_digest()

    # ---- (e) COMPLETION, gated on audit linkage: link the miner rows, observe the
    # gate hold the run open while the baseline row is unlinked, link the baseline, complete.
    for hk in ("hk-a", "hk-b"):
        comp_repo.set_audit_bundle_digest(
            comp_conn, performance_ids[hk], bundle_digests[hk]
        )
    stall_gaps = engine.audit_linkage_gaps(comp_conn, COMPETITION_ID)
    # Review window (SCORES_AT + 24h) elapsed untouched and end_time passed, but the
    # baseline calibration row is still unlinked: the tick defers (reason=audit_linkage_gaps).
    stalled_transitions = engine.tick(comp_conn, STALL_TICK_AT)
    stalled_comp = comp_repo.get_competition(comp_conn, COMPETITION_ID)
    assert stalled_comp is not None
    phase_during_stall = stalled_comp.status
    comp_repo.set_audit_bundle_digest(
        comp_conn, performance_ids["baseline"], baseline_bundle_digest
    )
    engine.tick(comp_conn, COMPLETED_AT)  # AWAITING_END_TIME -> COMPLETED
    baseline_row = next(
        c
        for c in comp_repo.list_contenders(comp_conn, COMPETITION_ID)
        if c.is_calibration
    )
    assert baseline_row.final_score is not None
    baseline_final_score = baseline_row.final_score

    # ---- (f) TOKENOMICS: this fixture's inference-only weight vector ------------------
    # No earning CompetitionInput is supplied here. Dedicated schema-v13 integration
    # tests cover the packet-economic result, crown, and combined vector. The baseline still
    # holds no identity at this seam.
    miners = fleet()
    weights = build_weight_vector(tokenomics_config, miners, burn_uid=0)

    # ---- (g) PUBLICATION: merkle root over every packet + ledger --------------------
    weight_vector_ref = store.put(
        canonical_json_bytes({"weights": {str(uid): w for uid, w in weights.items()}}),
        ArtifactKind.WEIGHT_VECTOR,
    )
    # The committed packet set covers EVERY packet of the evaluation, the baseline
    # calibration packet included — nothing scored stays outside the merkle root.
    packet_digests = sorted(
        [ref.digest for ref in packet_refs.values()] + [baseline_packet_ref.digest]
    )
    published_root = merkle_root(packet_digests)
    publication = build_publication_record(
        PublicationRecord(
            score_packet_merkle_root=published_root,
            weight_vector_digest=weight_vector_ref.digest,
        )
    )
    publication_id = ledger.record(publication, PUBLICATION_RECORDED_AT.isoformat())
    ledger.advance(
        publication_id, CommitmentStatus.ANCHORED, PUBLICATION_ANCHORED_AT.isoformat()
    )
    ledger.advance(publication_id, CommitmentStatus.PUBLISHED, PUBLISHED_AT.isoformat())

    return GoldenWorld(
        scoring_config=scoring_config,
        challenge_config=challenge_config,
        tokenomics_config=tokenomics_config,
        audit_config=audit_config,
        manifest=manifest,
        challenge_conn=challenge_conn,
        challenge=challenge,
        asset_id=asset.id,
        holdout_asset_id=holdout_asset_id,
        revealed=revealed,
        backend=backend,
        item_scores=item_scores,
        baseline_item_score=baseline_item_score,
        packets=packets,
        baseline_packet=baseline_packet,
        store=store,
        input_ref=input_ref,
        reference_ref=reference_ref,
        manifest_ref=manifest_ref,
        dag_reveal_ref=dag_reveal_ref,
        packet_refs=packet_refs,
        baseline_packet_ref=baseline_packet_ref,
        output_refs=output_refs,
        baseline_output_ref=baseline_output_ref,
        bundles=bundles,
        bundle_digests=bundle_digests,
        baseline_bundle=baseline_bundle,
        baseline_bundle_digest=baseline_bundle_digest,
        weight_vector_ref=weight_vector_ref,
        packet_digests=packet_digests,
        published_root=published_root,
        publication=publication,
        pre_commitment=pre_commitment,
        ledger=ledger,
        pre_commitment_id=pre_commitment_id,
        anchor_chain=anchor_chain,
        publication_id=publication_id,
        backup_ref=backup_ref,
        comp_conn=comp_conn,
        engine=engine,
        contender_ids=contender_ids,
        baseline_contender_id=baseline_row.contender_id,
        item_ids=item_ids,
        performance_ids=performance_ids,
        baseline_final_score=baseline_final_score,
        stall_gaps=stall_gaps,
        stalled_transitions=stalled_transitions,
        phase_during_stall=phase_during_stall,
        miners=miners,
        weights=weights,
    )
