"""ORCHESTRATOR-MINTED ZEROS ARE ATTRIBUTED TO THE ORCHESTRATOR (review round 2,
new-3).

The orchestrator legitimately mints one kind of packet: a gate-failed ZERO for an
item with no measurable bytes. It used to stamp `manifest.scoring_version` on it,
so a SCORE_PACKET artifact claimed the scoring WORKER had produced bytes the
worker never saw — an audit could not tell a measurement from bookkeeping.

The convention (vidaio/competition/orchestrator/zero_packets.py) gives those
packets a reserved, distinct identity of the same shape:
``orchestrator-zero/1+<digest12>``. These tests pin down that

  - a locally minted zero never carries the worker's identity,
  - it is recognisable as orchestrator-minted, and its audit bundle agrees with
    the packet (so the recompute cross-check passes) while the bundle's manifest
    still names the committed worker,
  - MEASURED packets are unaffected,
  - and impersonation is impossible in BOTH directions: the orchestrator never
    mints under the worker's identity, and a manifest/worker/packet claiming the
    reserved namespace halts the pipeline instead of being accepted.
"""

from __future__ import annotations

import json
import logging

import pytest

from vidaio.audit import (
    ArtifactKind,
    AuditBundle,
    CompetitionCommitment,
    build_competition_commitment,
    merkle_proof,
    merkle_root,
    pin_git_sha,
    reward_parameter_digest,
    verify_bundle,
)
from vidaio.audit.recompute import CompetitionAuditContext
from vidaio.auditor import RealScoreRecomputer
from vidaio.chain.adapter import InMemoryChain
from vidaio.competition import repository as repo
from vidaio.competition.epoch_evidence import build_competition_epoch_evidence
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.zero_packets import (
    ORCHESTRATOR_ZERO_PREFIX,
    ReservedScorerIdentity,
    assert_not_reserved,
    is_orchestrator_zero_identity,
    orchestrator_zero_identity,
)
from vidaio.epoch import MinerCensusEntry
from vidaio.competition.states import Phase
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring import DeterministicFakeBackend
from vidaio.scoring_worker import ScoringBackends, ScoringWorkerConfig

from orchestrator_support import (
    CONTENDER_SHAS,
    END,
    FINALIZATION,
    M,
    START,
    T0,
    ContenderFaultRunner,
    FakeScoringClient,
    build_manifest,
    materialize_baseline,
    enroll,
    events_of,
    phase,
    seed_items,
    start_and_enroll,
)

WORKER_IDENTITY = FakeScoringClient.IDENTITY


# ---- the convention itself -------------------------------------------------------


def test_the_identity_has_the_worker_identity_shape_but_a_reserved_name():
    identity = orchestrator_zero_identity(
        committed_scoring_version=WORKER_IDENTITY, track="compression"
    )
    name, _, digest = identity.partition("+")
    assert name == "orchestrator-zero/1"
    assert len(digest) == 12 and int(digest, 16) >= 0
    assert is_orchestrator_zero_identity(identity)
    assert not is_orchestrator_zero_identity(WORKER_IDENTITY)


def test_the_identity_is_deterministic_and_moves_with_what_it_depends_on():
    base = dict(committed_scoring_version=WORKER_IDENTITY, track="compression")
    assert orchestrator_zero_identity(**base) == orchestrator_zero_identity(**base)
    # A different committed worker, track, or scoring config -> different identity.
    assert orchestrator_zero_identity(
        committed_scoring_version="other/1+aaaaaaaaaaaa", track="compression"
    ) != orchestrator_zero_identity(**base)
    assert orchestrator_zero_identity(
        committed_scoring_version=WORKER_IDENTITY, track="upscaling"
    ) != orchestrator_zero_identity(**base)
    moved = ScoringConfig(compression_rate_max=0.5)
    assert orchestrator_zero_identity(**base, config=moved) != orchestrator_zero_identity(
        **base
    )


def test_the_reserved_namespace_is_refused_for_anyone_else():
    assert_not_reserved(WORKER_IDENTITY, what="a worker")  # no raise
    assert_not_reserved(None, what="nothing")
    with pytest.raises(ReservedScorerIdentity):
        assert_not_reserved("orchestrator-zero/1+deadbeefcafe", what="a worker")
    with pytest.raises(ReservedScorerIdentity):
        # Any version of the namespace, not just the current one.
        assert_not_reserved("orchestrator-zero/9+deadbeefcafe", what="a worker")


# ---- end to end ------------------------------------------------------------------


async def _run_with_a_silent_contender(
    orchestrator_factory,
    fixture_repos,
    tmp_path,
    *,
    with_baseline=False,
    scoring_client=None,
):
    runner = ContenderFaultRunner(tmp_path / "work" / "outputs")
    runner.silent_for.add(CONTENDER_SHAS["hk-silent"][1])
    orch = orchestrator_factory(
        runner=runner,
        repos=fixture_repos,
        scoring_client=scoring_client,
        chain=InMemoryChain() if with_baseline else None,
    )
    manifest = build_manifest(
        scoring_version=WORKER_IDENTITY,
        baseline=(materialize_baseline(orch, fixture_repos) if with_baseline else None),
    )
    if with_baseline:
        cid = manifest.competition_id
        orch.create_competition(manifest, T0)
        assert manifest.baseline is not None
        commitment = build_competition_commitment(
            CompetitionCommitment(
                manifest_digest=manifest.manifest_digest(),
                baseline_version=manifest.baseline.version,
                baseline_artifact_digest=manifest.baseline.artifact_digest,
                baseline_provenance_digest=manifest.baseline.provenance_digest,
                baseline_tree_digest=pin_git_sha(manifest.baseline.tree_sha),
                baseline_image_digest=runner.digest_for(manifest.baseline.tree_sha),
                dataset_selection_seed_commitment=manifest.scoring_seed_commitment,
                reward_param_digest=reward_parameter_digest(orch.tokenomics),
            )
        )
        commitment_ref = orch.store.put(
            commitment.canonical_json, ArtifactKind.MANIFEST
        )
        assert commitment_ref.digest == commitment.root
        anchored = await orch.anchor_competition(
            cid,
            baseline_image_digest=runner.digest_for(manifest.baseline.tree_sha),
            reward_param_digest=reward_parameter_digest(orch.tokenomics),
            baseline_tree_digest=pin_git_sha(manifest.baseline.tree_sha),
            now=T0,
        )
        assert anchored.recorded is True
        assert anchored.root == commitment.root
        await orch.step(START)
        for hotkey in ("hk-a", "hk-silent"):
            enroll(orch, cid, hotkey)
    else:
        cid = await start_and_enroll(orch, manifest, ["hk-a", "hk-silent"])
    seed_items(orch, cid, tmp_path / "item-src")
    for at in (
        FINALIZATION,
        FINALIZATION + 2 * M,
        FINALIZATION + 3 * M,
        FINALIZATION + 4 * M,
        FINALIZATION + 10 * M,
        FINALIZATION + 15 * M,
    ):
        await orch.step(at)
    return orch, cid, manifest


def _archived_rows(orch, cid):
    """[(contender_id, bundle, packet)] read from the REAL archived artifacts.

    Packets live in the audit store, not the DB (the DB keeps their digest), so
    this reads exactly what an auditor would: the SCORE_PACKET bytes the bundle
    points at.
    """
    from vidaio.audit import ArtifactRef

    by_performance = {
        r["performance_id"]: r["contender_id"]
        for r in orch.conn.execute(
            "SELECT performance_id, contender_id FROM performance_history"
            " WHERE competition_id = ?",
            (cid,),
        ).fetchall()
    }
    out = []
    for event in events_of(orch, cid, "audit_bundle_built"):
        payload = json.loads(event["payload_json"])
        bundle = payload["bundle"]
        packet = json.loads(
            orch.store.get(ArtifactRef.model_validate(bundle["score_packet"]))
        )
        out.append((by_performance[payload["performance_id"]], bundle, packet))
    return out


def _packets(orch, cid, contender_id):
    return [p for cid_, _b, p in _archived_rows(orch, cid) if cid_ == contender_id]


def _cpu_recomputer(tmp_path):
    fake = DeterministicFakeBackend()
    return RealScoreRecomputer(
        ScoringWorkerConfig(work_dir=tmp_path / "zero-audit-work"),
        ScoringBackends(
            probe=fake,
            vmaf_primary=fake,
            vmaf_secondary=fake,
            pieapp=fake,
            perceptual=fake,
            canonicalizer=None,
            versions={},
        ),
        scoring_config=ScoringConfig(),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )


def _competition_context(orch, cid, manifest, bundle):
    row = orch.conn.execute(
        "SELECT * FROM evaluation_items WHERE competition_id = ? "
        "AND challenge_id = ? AND scoring_item_id = ?",
        (cid, bundle.challenge_id, bundle.item_id),
    ).fetchone()
    assert row is not None
    return CompetitionAuditContext(
        competition_id=cid,
        track=manifest.track,
        manifest_digest=manifest.manifest_digest(),
        threshold_commitment=row["threshold_commitment"],
    )


async def test_a_locally_minted_zero_never_claims_the_workers_identity(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch, cid, manifest = await _run_with_a_silent_contender(
        orchestrator_factory, fixture_repos, tmp_path
    )
    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    expected = orchestrator_zero_identity(
        committed_scoring_version=manifest.scoring_version, track=manifest.track
    )

    zeros = _packets(orch, cid, by_hotkey["hk-silent"].contender_id)
    assert len(zeros) == 3
    for packet in zeros:
        assert packet["scorer_version"] == expected
        assert is_orchestrator_zero_identity(packet["scorer_version"])
        # THE POINT: it is not the worker's identity, in any form.
        assert packet["scorer_version"] != manifest.scoring_version
        assert WORKER_IDENTITY not in packet["scorer_version"]
        # ... and it asserts no measurement.
        assert packet["gate_passed"] is False and packet["score"] == 0.0
        assert packet["violations"][0]["code"] == "METRIC_MISSING"

    # MEASURED packets are untouched: they still carry the worker's own stamp.
    measured = _packets(orch, cid, by_hotkey["hk-a"].contender_id)
    assert len(measured) == 3
    assert all(p["scorer_version"] == WORKER_IDENTITY for p in measured)
    assert all(not is_orchestrator_zero_identity(p["scorer_version"]) for p in measured)


async def test_the_audit_bundle_agrees_with_the_packet_it_links(
    orchestrator_factory, fixture_repos, tmp_path
):
    """audit/recompute.py cross-checks packet.scorer_version == bundle.scorer_version.

    A zero row must therefore carry the orchestrator identity on BOTH — while the
    bundle's manifest artifact still names the committed worker, so the bundle
    reads: "this competition committed to X; here there were no bytes to measure".
    """
    measured_backends = {
        "ffmpeg": "ffmpeg/test-7.1",
        "libvmaf": "libvmaf/test-3.0.0",
    }
    orch, cid, manifest = await _run_with_a_silent_contender(
        orchestrator_factory,
        fixture_repos,
        tmp_path,
        scoring_client=FakeScoringClient(backend_versions=measured_backends),
    )
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    silent_id = by_hotkey["hk-silent"].contender_id
    honest_id = by_hotkey["hk-a"].contender_id

    rows = _archived_rows(orch, cid)
    assert len(rows) == 6
    manifest_refs = set()
    seen = {"zero": 0, "measured": 0}
    for contender_id, bundle, packet in rows:
        # The invariant the recompute path checks.
        assert bundle["scorer_version"] == packet["scorer_version"]
        assert bundle["backend_versions"] == packet["backend_versions"]
        manifest_refs.add(bundle["manifest"]["digest"])
        if contender_id == silent_id:
            assert is_orchestrator_zero_identity(bundle["scorer_version"])
            assert bundle["backend_versions"] == {}
            seen["zero"] += 1
        else:
            assert contender_id == honest_id
            assert bundle["scorer_version"] == manifest.scoring_version
            assert bundle["backend_versions"] == measured_backends
            seen["measured"] += 1
    assert seen == {"zero": 3, "measured": 3}
    # Both kinds of row point at the SAME committed manifest artifact: the
    # difference is only in who minted the packet, never in what was committed to.
    assert len(manifest_refs) == 1

    # Every zero row is audit-linked exactly like a measured one.
    assert orch.engine.audit_linkage_gaps(orch.conn, cid) == []


async def test_orchestrator_zero_strictly_recomputes_on_cpu_end_to_end(
    orchestrator_factory, fixture_repos, tmp_path
):
    """A silent GPU/container result remains exact economic evidence on CPU."""
    orch, cid, manifest = await _run_with_a_silent_contender(
        orchestrator_factory, fixture_repos, tmp_path, with_baseline=True
    )
    archived = _archived_rows(orch, cid)
    bundles = [AuditBundle.model_validate(raw) for _contender, raw, _packet in archived]
    leaves = [bundle.score_packet.digest for bundle in bundles]
    zero_bundle = next(
        bundle
        for bundle in bundles
        if bundle.miner_hotkey == "hk-silent"
    )
    packet = json.loads(orch.store.get(zero_bundle.score_packet))
    assert packet["metrics"] == {}

    report = verify_bundle(
        zero_bundle,
        orch.store,
        _cpu_recomputer(tmp_path),
        expected_bundle_digest=zero_bundle.bundle_digest(),
        expected_miner_hotkey="hk-silent",
        require_expected_miner=True,
        published_root=merkle_root(leaves),
        inclusion_proof=merkle_proof(leaves, zero_bundle.score_packet.digest),
        strict=True,
        competition_context=_competition_context(
            orch, cid, manifest, zero_bundle
        ),
    )
    assert report.passed, [
        (failure.name, failure.code, failure.reason)
        for failure in report.failures()
    ]

    # Completion does not drop the silent contender after seeing its zeros. The
    # exact zero packets remain in the economic mean/result worklist.
    await orch.step(END + M)
    assert phase(orch, cid) is Phase.COMPLETED
    evidence = build_competition_epoch_evidence(
        orch.conn,
        competition_id=cid,
        census_by_hotkey={
            "hk-a": MinerCensusEntry(
                uid=10, hotkey="hk-a", coldkey="ck-a", ip="203.0.113.10"
            ),
            "hk-silent": MinerCensusEntry(
                uid=11,
                hotkey="hk-silent",
                coldkey="ck-silent",
                ip="203.0.113.11",
            ),
        },
        store=orch.store,
        through_time=END + M,
    )
    assert evidence is not None
    silent = next(
        subject
        for subject in evidence.competition_input.subjects
        if subject.hotkey == "hk-silent"
    )
    assert set(silent.packet_digests).issubset(evidence.packet_scores)
    assert all(evidence.packet_scores[digest] == 0.0 for digest in silent.packet_digests)


@pytest.mark.parametrize("tamper", ["metric", "nonempty_output", "scoring_config"])
async def test_orchestrator_zero_cpu_audit_rejects_measurement_or_output_injection(
    orchestrator_factory, fixture_repos, tmp_path, tamper
):
    orch, cid, manifest = await _run_with_a_silent_contender(
        orchestrator_factory, fixture_repos, tmp_path
    )
    archived = _archived_rows(orch, cid)
    zero_raw = next(
        raw
        for _contender, raw, packet in archived
        if packet["miner_hotkey"] == "hk-silent"
    )
    zero_bundle = AuditBundle.model_validate(zero_raw)
    packet = json.loads(orch.store.get(zero_bundle.score_packet))
    update = {}
    if tamper == "metric":
        packet["metrics"] = {"claimed_measurement": 1.0}
    elif tamper == "nonempty_output":
        output_ref = orch.store.put(b"not-empty", ArtifactKind.MINER_OUTPUT)
        packet["content_digest"] = output_ref.digest
        update["miner_output"] = output_ref
    else:
        moved_config = ScoringConfig(compression_rate_max=0.5)
        moved_identity = orchestrator_zero_identity(
            committed_scoring_version=manifest.scoring_version,
            track=manifest.track,
            config=moved_config,
        )
        from vidaio.scoring.result import config_digest

        packet["scoring_config_digest"] = config_digest(moved_config)
        packet["scorer_version"] = moved_identity
        update["scorer_version"] = moved_identity
    packet_ref = orch.store.put(
        json.dumps(packet, separators=(",", ":")).encode(), ArtifactKind.SCORE_PACKET
    )
    update["score_packet"] = packet_ref
    tampered_bundle = zero_bundle.model_copy(update=update)

    report = verify_bundle(
        tampered_bundle,
        orch.store,
        _cpu_recomputer(tmp_path),
        expected_bundle_digest=tampered_bundle.bundle_digest(),
        expected_miner_hotkey="hk-silent",
        require_expected_miner=True,
        published_root=merkle_root([packet_ref.digest]),
        inclusion_proof=merkle_proof([packet_ref.digest], packet_ref.digest),
        strict=True,
        competition_context=_competition_context(
            orch, cid, manifest, tampered_bundle
        ),
    )
    assert not report.passed
    assert any(
        failure.name == "score_recompute" and failure.code == "RECOMPUTE_ERROR"
        for failure in report.failures()
    )


async def test_the_event_log_records_the_zero_as_an_orchestrator_attribution(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch, cid, manifest = await _run_with_a_silent_contender(
        orchestrator_factory, fixture_repos, tmp_path
    )
    zeroed = events_of(orch, cid, pers.EVENT_ITEM_ZEROED)
    assert len(zeroed) == 3
    payload = json.loads(zeroed[0]["payload_json"])
    assert is_orchestrator_zero_identity(payload["minted_by"])
    assert payload["committed_scoring_version"] == manifest.scoring_version
    assert payload["code"] == "METRIC_MISSING"


# ---- impersonation, both directions ----------------------------------------------


async def test_a_manifest_committing_to_the_reserved_namespace_halts(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """A competition may not commit to a scorer named like our zero records."""
    orch = orchestrator_factory(repos=fixture_repos)
    manifest = build_manifest(scoring_version=f"{ORCHESTRATOR_ZERO_PREFIX}deadbeefcafe")
    cid = await start_and_enroll(orch, manifest, ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")

    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION)

    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.FINALIZING_SUBMISSIONS  # halted, never failed
    reason = events_of(orch, cid, pers.EVENT_HALTED)[0]["payload_json"]
    assert "reserved scorer namespace" in reason
    assert orch.scoring_client.calls == []


async def test_a_worker_advertising_the_reserved_namespace_halts(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    orch = orchestrator_factory(repos=fixture_repos)
    orch.scoring_client.identity = f"{ORCHESTRATOR_ZERO_PREFIX}0123456789ab"
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")

    with caplog.at_level(logging.CRITICAL):
        await orch.step(FINALIZATION)

    assert pers.is_halted(orch.conn, cid)
    reason = events_of(orch, cid, pers.EVENT_HALTED)[0]["payload_json"]
    assert "reserved scorer namespace" in reason
    assert orch.scoring_client.calls == []


async def test_a_worker_packet_stamped_with_the_reserved_identity_is_never_persisted(
    orchestrator_factory, fixture_repos, tmp_path, caplog
):
    """The last impersonation route: an honest /healthz, a tampered packet stamp.

    A packet claiming the orchestrator's identity would read as "no bytes to
    measure" while carrying whatever score the worker chose. It halts instead.
    """
    from orchestrator_support import client_conn
    from vidaio.competition.interfaces import ScorePacket

    class ImpersonatingClient(FakeScoringClient):
        def score_item(self, competition_id, contender_id, item, output):
            packet = super().score_item(competition_id, contender_id, item, output)
            tampered = json.loads(packet.packet_bytes)
            tampered["scorer_version"] = f"{ORCHESTRATOR_ZERO_PREFIX}0123456789ab"
            return ScorePacket(
                item_id=packet.item_id,
                contender_id=packet.contender_id,
                packet_bytes=json.dumps(tampered).encode("utf-8"),
            )

    client = ImpersonatingClient()
    orch = orchestrator_factory(scoring_client=client, repos=fixture_repos)
    client.conn = client_conn(orch.core.db_path)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a"])
    seed_items(orch, cid, tmp_path / "item-src")
    with caplog.at_level(logging.CRITICAL):
        for at in (
            FINALIZATION,
            FINALIZATION + 2 * M,
            FINALIZATION + 3 * M,
            FINALIZATION + 4 * M,
            FINALIZATION + 10 * M,
            FINALIZATION + 15 * M,
        ):
            await orch.step(at)

    assert pers.is_halted(orch.conn, cid)
    assert phase(orch, cid) is Phase.SCORING  # halted, never failed
    # Not one tampered packet reached the database.
    assert (
        orch.conn.execute(
            "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?",
            (cid,),
        ).fetchone()["n"]
        == 0
    )
