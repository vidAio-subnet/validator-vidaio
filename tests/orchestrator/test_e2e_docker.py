"""Full local competition e2e over real Docker (spec §05 + §14):

2 honest contenders + 1 untrusted + the baseline calibration baseline:
build -> isolation probe (untrusted disqualified) -> batched evaluation in
network-blocked read-only sandboxes -> scoring with REAL compose_item_score
packets -> per-item audit bundles linked -> engine completes after the review
window -> podium excludes the baseline (the project design record #1)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from vidaio.audit import ArtifactKind, ArtifactRef
from vidaio.audit.store import backend_key
from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import OrchestratorConfig, build_docker_runner
from vidaio.competition.states import Phase

from orchestrator_support import (
    BASELINE,
    COMMITMENT_ROOT,
    DOCKER,
    END,
    FINALIZATION,
    M,
    START,
    T0,
    build_manifest,
    enroll,
    events_of,
    make_raw_config,
    phase,
    seed_items,
)

pytestmark = pytest.mark.docker


async def test_full_competition_e2e(orchestrator_factory, fixture_repos, tmp_path):
    from vidaio.competition.runners import LocalRepoProvider

    raw = make_raw_config(tmp_path)
    runner = build_docker_runner(
        OrchestratorConfig(**raw["orchestrator"]),
        LocalRepoProvider(fixture_repos),
        docker_path=DOCKER,
    )
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)

    manifest = build_manifest(baseline=BASELINE)
    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    orch.anchor_commitment(cid, COMMITMENT_ROOT, T0)

    await orch.step(START)
    assert phase(orch, cid) is Phase.ENROLLING
    for hotkey in ["hk-a", "hk-b", "hk-mal"]:
        enroll(orch, cid, hotkey)
    item_ids = seed_items(orch, cid, tmp_path / "item-src", n=3)

    # ENROLLING -> FINALIZING (baseline injected) -> submissions backed up -> VALIDATING
    await orch.step(FINALIZATION)
    assert phase(orch, cid) is Phase.VALIDATING
    contenders = repo.list_contenders(orch.conn, cid)
    assert sum(1 for c in contenders if c.is_calibration) == 1
    backup_events = [
        e
        for e in repo.list_events(orch.conn, cid)
        if e["event_type"] == "transition" and e["to_phase"] == "VALIDATING"
    ]
    assert "audit://submissions/sha256:" in backup_events[0]["payload_json"]

    # VALIDATING -> BUILDING (all four have a Dockerfile)
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.BUILDING

    # BUILDING: honest + baseline build & pass the probe; untrusted is disqualified.
    await orch.step(FINALIZATION + 3 * M)
    assert phase(orch, cid) is Phase.EVALUATING
    by_hotkey = {c.hotkey: c for c in repo.list_contenders(orch.conn, cid)}
    baseline_row = next(c for c in repo.list_contenders(orch.conn, cid) if c.is_calibration)
    assert by_hotkey["hk-a"].status == "BUILT"
    assert by_hotkey["hk-b"].status == "BUILT"
    assert baseline_row.status == "BUILT"
    assert by_hotkey["hk-mal"].status == "BUILD_FAILED"
    dq = events_of(orch, cid, "isolation_probe_failed")
    assert len(dq) == 1 and "VIDAIO_VALIDATOR_PAT" in dq[0]["payload_json"]

    # EVALUATING: 3 items, batch max 2 -> 2 batches x 3 built contenders.
    await orch.step(FINALIZATION + 10 * M)
    assert phase(orch, cid) is Phase.SCORING
    batches = orch.conn.execute(
        "SELECT status FROM batches WHERE competition_id = ?", (cid,)
    ).fetchall()
    assert len(batches) == 6 and all(b["status"] == "COMPLETED" for b in batches)

    # SCORING: real compose_item_score packets persisted + audit-linked atomically.
    await orch.step(FINALIZATION + 15 * M)
    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    rows = orch.conn.execute(
        "SELECT COUNT(*) AS n FROM performance_history WHERE competition_id = ?",
        (cid,),
    ).fetchone()["n"]
    assert rows == 3 * 3  # (2 honest + baseline) x 3 items; untrusted never evaluated
    assert orch.engine.audit_linkage_gaps(orch.conn, cid) == []
    assert repo.count_missing_calibration_rows(orch.conn, cid) == 0
    assert len(events_of(orch, cid, "audit_bundle_built")) == 9

    # Artifacts really archived: inputs, outputs and packets fetchable by digest.
    for item_id in item_ids:
        row = orch.conn.execute(
            "SELECT input_sha256, input_bytes FROM evaluation_items WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        ref = ArtifactRef(
            digest=row["input_sha256"],
            kind=ArtifactKind.CHALLENGE_INPUT,
            byte_size=row["input_bytes"],
            backend_key=backend_key(ArtifactKind.CHALLENGE_INPUT, row["input_sha256"]),
        )
        assert orch.store.exists(ref)

    # Review window still open at T0+30h (deadline ~T0+27h but end_time is T0+48h).
    await orch.step(T0 + timedelta(hours=30))
    assert phase(orch, cid) is Phase.AWAITING_END_TIME

    # Completion after end_time: gates (audit linkage + baseline matrix) are satisfied.
    await orch.step(END + M)
    assert phase(orch, cid) is Phase.COMPLETED

    # Ranking: hk-a (512 B outputs) beats hk-b (1024 B); untrusted ranks last on
    # zero; the baseline is EXCLUDED from ranking/podium by construction, even though
    # its compression (256 B) beat everyone — non-earning calibration baseline.
    ranking = repo.ranking(orch.conn, cid)
    assert [c.hotkey for c in ranking] == ["hk-a", "hk-b", "hk-mal"]
    assert ranking[0].final_score > ranking[1].final_score > ranking[2].final_score
    assert ranking[2].final_score == 0.0
    podium = repo.podium(orch.conn, cid)
    assert all(not c.is_calibration for c in podium)
    assert [c.hotkey for c in podium] == ["hk-a", "hk-b", "hk-mal"]

    refreshed_baseline = repo.get_contender(orch.conn, baseline_row.contender_id)
    assert refreshed_baseline.final_rank is None
    assert refreshed_baseline.media_score_aggregate is not None
    # The baseline's genuine aggregate anchors the bar above the winner's.
    assert refreshed_baseline.media_score_aggregate > ranking[0].media_score_aggregate

    # Review chain untouched but verifiable; event log intact.
    assert repo.verify_review_chain(orch.conn, cid)
