"""Fresh Modal process-restart fencing and full-matrix recovery.

No test imports Modal or contacts the cloud. ``RuntimeRunner`` models the
important SDK property: live image handles are process-scoped, while an exact
immutable Image object id can be rehydrated into a new process after its binding
to the pinned source and evidence digest was durably recorded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pytest

from vidaio.competition import repository as repo
from vidaio.competition.interfaces import (
    BatchItem,
    BatchOutput,
    ContenderSpec,
    IsolationProbeReport,
    logical_build_identity,
)
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.service import reward_parameter_digest
from vidaio.competition.states import Phase
from vidaio.tokenomics.config import TokenomicsConfig

from orchestrator_support import (
    BASELINE,
    FINALIZATION,
    M,
    T0,
    FakeRunner,
    RecordingChain,
    SimulatedCrash,
    build_manifest,
    events_of,
    phase,
    seed_items,
    start_and_enroll,
)


class RuntimeRunner(FakeRunner):
    """Fake with process-scoped image handles and runtime-distinct outputs."""

    def __init__(
        self,
        outputs_dir: Path,
        marker: str,
        *,
        digest_drift: bool = False,
        restore_drift: bool = False,
        fail_reprobe: bool = False,
        crash_on_build_call: int | None = None,
        crash_on_batch_call: int | None = None,
    ) -> None:
        super().__init__(outputs_dir)
        self.runtime_session_id = marker * 64
        self.runtime_label = f"vidaio-next-runtime-{marker * 8}"
        self.digest_drift = digest_drift
        self.restore_drift = restore_drift
        self.fail_reprobe = fail_reprobe
        self.crash_on_build_call = crash_on_build_call
        self.crash_on_batch_call = crash_on_batch_call
        self.live_images: set[str] = set()
        self.image_ids: dict[str, str] = {}
        self.spec_digests: dict[tuple[str, str, str], str] = {}

    def has_live_image(self, image_digest: str) -> bool:
        return image_digest in self.live_images

    def image_object_id(self, image_digest: str) -> str | None:
        return self.image_ids.get(image_digest)

    def restore_image(self, contender, image_digest: str, image_object_id: str) -> str:
        restored = image_digest
        if self.restore_drift:
            restored = hashlib.sha256(
                f"restore-drift:{image_digest}".encode()
            ).hexdigest()
        self.live_images.add(restored)
        self.image_ids[restored] = image_object_id
        self.spec_digests[
            (contender.repo_url, contender.commit_sha, contender.tree_sha)
        ] = restored
        return restored

    def build(self, contender) -> str:
        spec_key = (contender.repo_url, contender.commit_sha, contender.tree_sha)
        cached = self.spec_digests.get(spec_key)
        if cached is not None and cached in self.live_images:
            return cached
        if (
            self.crash_on_build_call is not None
            and len(self.build_calls) + 1 >= self.crash_on_build_call
        ):
            raise SimulatedCrash("process died mid-build")
        super().build(contender)
        object_id = f"im-{self.runtime_session_id[:8]}-{contender.tree_sha[:16]}"
        digest = logical_build_identity(
            repo_url=contender.repo_url,
            commit_sha=contender.commit_sha,
            tree_sha=contender.tree_sha,
        )
        if self.digest_drift:
            digest = hashlib.sha256(f"drift:{digest}".encode()).hexdigest()
        self.live_images.add(digest)
        self.image_ids[digest] = object_id
        self.spec_digests[spec_key] = digest
        return digest

    def isolation_probe(self, image_digest: str) -> IsolationProbeReport:
        assert image_digest in self.live_images
        self.probe_calls.append(image_digest)
        return IsolationProbeReport(
            network_blocked=not self.fail_reprobe,
            secrets_absent=True,
            reference_mounts_absent=True,
            index_leak_absent=True,
            details=f"runtime {self.runtime_session_id[:8]}",
        )

    def run_batch(
        self, image_digest: str, items: Sequence[BatchItem], batch_index: int
    ) -> Sequence[BatchOutput]:
        assert image_digest in self.live_images, "persisted digest is not a live handle"
        self.batch_calls.append((image_digest, batch_index))
        if (
            self.crash_on_batch_call is not None
            and len(self.batch_calls) >= self.crash_on_batch_call
        ):
            raise SimulatedCrash("process died mid-evaluation")
        outputs: list[BatchOutput] = []
        for item in items:
            data = (
                f"runtime:{self.runtime_session_id[:8]}:{image_digest}:"
                f"{item.input_sha256}"
            ).encode()
            digest = hashlib.sha256(data).hexdigest()
            (self.outputs_dir / digest).write_bytes(data)
            outputs.append(
                BatchOutput(
                    item_id=item.item_id,
                    output_sha256=digest,
                    output_bytes=len(data),
                    wall_seconds=0.01,
                )
            )
        return outputs


async def _drive_to_building(orch, tmp_path: Path, hotkeys: list[str]) -> str:
    cid = await start_and_enroll(orch, build_manifest(), hotkeys)
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    assert phase(orch, cid) is Phase.BUILDING
    return cid


async def _crash_mid_evaluation(orchestrator_factory, fixture_repos, tmp_path: Path):
    runner_a = RuntimeRunner(tmp_path / "work" / "outputs", "a", crash_on_batch_call=2)
    orch_a = orchestrator_factory(runner=runner_a, repos=fixture_repos)
    cid = await _drive_to_building(orch_a, tmp_path, ["hk-a"])
    await orch_a.step(FINALIZATION + 3 * M)
    assert phase(orch_a, cid) is Phase.EVALUATING
    with pytest.raises(SimulatedCrash, match="mid-evaluation"):
        await orch_a.step(FINALIZATION + 4 * M)
    assert len(events_of(orch_a, cid, pers.EVENT_BATCH_OUTPUTS)) == 1
    statuses = [
        row["status"]
        for row in orch_a.conn.execute(
            "SELECT status FROM batches WHERE competition_id = ? ORDER BY batch_index",
            (cid,),
        )
    ]
    assert statuses == ["COMPLETED", "RUNNING"]
    return orch_a, cid


async def test_earning_baseline_prebuild_is_runtime_bound_before_anchor(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    runner = RuntimeRunner(tmp_path / "work" / "outputs", "a")
    chain = RecordingChain()
    orch = orchestrator_factory(runner=runner, repos=fixture_repos, chain=chain)
    config = TokenomicsConfig(competition_emissions_enabled=True)
    orch.tokenomics = config
    manifest = build_manifest("runtime-bound-baseline", baseline=BASELINE)
    orch.create_competition(manifest, T0)

    anchored = await orch.anchor_competition(
        manifest.competition_id,
        reward_param_digest=reward_parameter_digest(config),
        now=T0,
    )

    events = repo.list_events(orch.conn, manifest.competition_id)
    types = [event["event_type"] for event in events]
    assert types.index(pers.EVENT_MODAL_RUNTIME_BOUND) < types.index(
        pers.EVENT_COMMITMENT_ANCHORED
    )
    assert types.index(pers.EVENT_MODAL_IMAGE_BOUND) < types.index(
        pers.EVENT_COMMITMENT_ANCHORED
    )
    assert runner.has_live_image(anchored.baseline_image_digest)
    assert runner.build_calls == [0]


async def test_latest_exact_provider_binding_wins_for_one_stable_logical_digest(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    cid = "stable-logical-two-provider-objects"
    runner_a = RuntimeRunner(tmp_path / "work" / "outputs", "a")
    orch_a = orchestrator_factory(runner=runner_a, repos=fixture_repos)
    orch_a.create_competition(build_manifest(cid), T0)
    spec = ContenderSpec(
        contender_id=1,
        repo_url="local://hk-a",
        commit_sha="1a" * 20,
        tree_sha="2a" * 20,
    )

    digest_a = runner_a.build(spec)
    object_a = runner_a.image_object_id(digest_a)
    assert object_a is not None
    with pers.txn(orch_a.conn):
        pers.record_modal_image_binding(
            orch_a.conn,
            cid,
            contender_id=spec.contender_id,
            is_calibration=False,
            repo_url=spec.repo_url,
            commit_sha=spec.commit_sha,
            tree_sha=spec.tree_sha,
            image_digest=digest_a,
            image_object_id=object_a,
            runtime_session_id=runner_a.runtime_session_id,
            runtime_label=runner_a.runtime_label,
            now=T0,
        )

    runner_b = RuntimeRunner(tmp_path / "work" / "outputs", "b")
    digest_b = runner_b.build(spec)
    object_b = runner_b.image_object_id(digest_b)
    assert digest_b == digest_a
    assert object_b is not None and object_b != object_a
    with pers.txn(orch_a.conn):
        pers.record_modal_image_binding(
            orch_a.conn,
            cid,
            contender_id=spec.contender_id,
            is_calibration=False,
            repo_url=spec.repo_url,
            commit_sha=spec.commit_sha,
            tree_sha=spec.tree_sha,
            image_digest=digest_b,
            image_object_id=object_b,
            runtime_session_id=runner_b.runtime_session_id,
            runtime_label=runner_b.runtime_label,
            now=T0 + M,
        )

    latest = pers.latest_modal_image_binding(
        orch_a.conn, cid, digest_a, is_calibration=False
    )
    assert latest is not None
    assert latest["image_object_id"] == object_b
    assert latest["build_identity_scheme"] == ("vidaio.competition.logical-build.v1")
    assert (
        orch_a.conn.execute(
            "SELECT COUNT(*) FROM modal_image_bindings WHERE competition_id = ?",
            (cid,),
        ).fetchone()[0]
        == 2
    )

    runner_c = RuntimeRunner(tmp_path / "work" / "outputs", "c")
    orch_c = orchestrator_factory(runner=runner_c, repos=fixture_repos)
    restored = await orch_c._restore_bound_modal_image(
        cid, spec, digest_a, is_calibration=False
    )
    assert restored == digest_a
    assert runner_c.image_object_id(digest_a) == object_b


async def test_legacy_json_only_image_binding_is_ignored_and_restore_fails_closed(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    cid = "legacy-object-scoped-binding"
    runner = RuntimeRunner(tmp_path / "work" / "outputs", "a")
    orch = orchestrator_factory(runner=runner, repos=fixture_repos)
    orch.create_competition(build_manifest(cid), T0)
    spec = ContenderSpec(
        contender_id=1,
        repo_url="local://hk-a",
        commit_sha="1a" * 20,
        tree_sha="2a" * 20,
    )
    logical_digest = logical_build_identity(
        repo_url=spec.repo_url,
        commit_sha=spec.commit_sha,
        tree_sha=spec.tree_sha,
    )
    # This is the pre-0005 shape: append-only chronology only, with no typed
    # scheme/provider ownership row. Migration cannot prove its old digest
    # semantics and deliberately does not backfill it.
    with pers.txn(orch.conn):
        repo.record_event(
            orch.conn,
            cid,
            pers.EVENT_MODAL_IMAGE_BOUND,
            T0,
            payload={
                "contender_id": spec.contender_id,
                "is_calibration": False,
                "repo_url": spec.repo_url,
                "commit_sha": spec.commit_sha,
                "tree_sha": spec.tree_sha,
                "image_digest": "f" * 64,
                "image_object_id": "im-legacy-object",
                "runtime_session_id": "0" * 64,
                "runtime_label": "vidaio-next-legacy-runtime",
            },
        )

    assert (
        pers.latest_modal_image_binding(
            orch.conn, cid, logical_digest, is_calibration=False
        )
        is None
    )
    with pytest.raises(ValueError, match="no append-only.*binding"):
        await orch._restore_bound_modal_image(
            cid, spec, logical_digest, is_calibration=False
        )


async def test_earning_baseline_restart_restores_the_exact_anchored_owned_image(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    runner_a = RuntimeRunner(tmp_path / "work" / "outputs", "a")
    chain = RecordingChain()
    orch_a = orchestrator_factory(runner=runner_a, repos=fixture_repos, chain=chain)
    config = TokenomicsConfig(competition_emissions_enabled=True)
    orch_a.tokenomics = config
    manifest = build_manifest("runtime-restored-baseline", baseline=BASELINE)
    cid = manifest.competition_id
    orch_a.create_competition(manifest, T0)
    anchored = await orch_a.anchor_competition(
        cid,
        reward_param_digest=reward_parameter_digest(config),
        now=T0,
    )
    anchored_object_id = runner_a.image_object_id(anchored.baseline_image_digest)
    assert anchored_object_id is not None

    await orch_a.step(manifest.start_time)
    orch_a.enroll_contender(
        cid,
        hotkey="hk-a",
        repo_url="local://hk-a",
        commit_sha="1a" * 20,
        tree_sha="2a" * 20,
        stake=1000.0,
        now=manifest.start_time + M,
    )
    seed_items(orch_a, cid, tmp_path / "restart-baseline-items")
    await orch_a.step(FINALIZATION)
    await orch_a.step(FINALIZATION + 2 * M)
    assert phase(orch_a, cid) is Phase.BUILDING

    runner_b = RuntimeRunner(tmp_path / "work" / "outputs", "b")
    orch_b = orchestrator_factory(runner=runner_b, repos=fixture_repos, chain=chain)
    orch_b.tokenomics = config
    await orch_b.step(FINALIZATION + 3 * M)

    halts = events_of(orch_b, cid, pers.EVENT_HALTED)
    assert not pers.is_halted(orch_b.conn, cid), [
        json.loads(event["payload_json"])["reason"] for event in halts
    ]
    assert phase(orch_b, cid) is Phase.EVALUATING
    baseline = next(
        contender
        for contender in repo.list_contenders(orch_b.conn, cid)
        if contender.is_calibration
    )
    assert baseline.image_digest == anchored.baseline_image_digest
    assert runner_b.image_object_id(baseline.image_digest) == anchored_object_id
    # Only the ordinary contender gets a new image. The baseline is restored by its
    # exact competition-owned id; no old Sandbox/instance is reused.
    ordinary = next(
        contender
        for contender in repo.list_contenders(orch_b.conn, cid)
        if not contender.is_calibration
    )
    assert runner_b.build_calls == [ordinary.contender_id]


async def test_building_restart_rebinds_built_images_before_new_builds(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    runner_a = RuntimeRunner(tmp_path / "work" / "outputs", "a", crash_on_build_call=2)
    orch_a = orchestrator_factory(runner=runner_a, repos=fixture_repos)
    cid = await _drive_to_building(orch_a, tmp_path, ["hk-a", "hk-b"])
    with pytest.raises(SimulatedCrash, match="mid-build"):
        await orch_a.step(FINALIZATION + 3 * M)
    built_before = [
        contender
        for contender in repo.list_contenders(orch_a.conn, cid)
        if contender.status == "BUILT"
    ]
    assert len(built_before) == 1

    runner_b = RuntimeRunner(tmp_path / "work" / "outputs", "b")
    orch_b = orchestrator_factory(runner=runner_b, repos=fixture_repos)
    await orch_b.step(FINALIZATION + 4 * M)

    assert phase(orch_b, cid) is Phase.EVALUATING
    assert (
        len(runner_b.build_calls) == 1
    )  # old image restored; only pending image builds
    assert all(
        contender.image_digest in runner_b.live_images
        for contender in repo.list_contenders(orch_b.conn, cid)
        if contender.status == "BUILT"
    )
    bindings = events_of(orch_b, cid, pers.EVENT_MODAL_RUNTIME_BOUND)
    assert len(bindings) == 2
    rebound = json.loads(bindings[-1]["payload_json"])
    assert rebound["previous_runtime_session_id"] == "a" * 64
    assert rebound["runtime_session_id"] == "b" * 64
    assert len(rebound["rebound_images"]) == 1
    assert not events_of(orch_b, cid, pers.EVENT_MODAL_EVALUATION_RESET)


async def test_evaluating_restart_reruns_whole_matrix_without_mixing_outputs(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    orch_a, cid = await _crash_mid_evaluation(
        orchestrator_factory, fixture_repos, tmp_path
    )
    old_event = events_of(orch_a, cid, pers.EVENT_BATCH_OUTPUTS)[0]
    old_digest = json.loads(old_event["payload_json"])["outputs"][0][1]

    runner_b = RuntimeRunner(tmp_path / "work" / "outputs", "b")
    orch_b = orchestrator_factory(runner=runner_b, repos=fixture_repos)
    await orch_b.step(FINALIZATION + 5 * M)

    assert phase(orch_b, cid) is Phase.SCORING
    # Batch 0 was completed by A, but B reruns BOTH batches behind the reset
    # fence. It never resumes from only the stale RUNNING batch 1.
    assert [batch_index for _digest, batch_index in runner_b.batch_calls] == [0, 1]
    resets = events_of(orch_b, cid, pers.EVENT_MODAL_EVALUATION_RESET)
    assert len(resets) == 1
    reset = json.loads(resets[0]["payload_json"])
    assert reset["previous_runtime_session_id"] == "a" * 64
    assert reset["runtime_session_id"] == "b" * 64
    assert reset["policy"] == "discard_prior_effective_batches_and_rerun_full_matrix"

    contender = repo.list_contenders(orch_b.conn, cid)[0]
    effective = pers.outputs_for_contender(orch_b.conn, cid, contender.contender_id)
    assert len(effective) == 3
    assert old_digest not in {digest for digest, _size in effective.values()}
    # Old evidence remains append-only and inspectable; it is just below the
    # effective-output fence and cannot flow into scoring.
    assert len(events_of(orch_b, cid, pers.EVENT_BATCH_OUTPUTS)) == 3
    assert len(events_of(orch_b, cid, pers.EVENT_MODAL_RUNTIME_BOUND)) == 2


async def test_restart_digest_drift_halts_before_replacement_runtime_executes(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    _orch_a, cid = await _crash_mid_evaluation(
        orchestrator_factory, fixture_repos, tmp_path
    )
    runner_b = RuntimeRunner(tmp_path / "work" / "outputs", "b", restore_drift=True)
    orch_b = orchestrator_factory(runner=runner_b, repos=fixture_repos)

    await orch_b.step(FINALIZATION + 5 * M)

    assert phase(orch_b, cid) is Phase.EVALUATING
    assert pers.is_halted(orch_b.conn, cid)
    assert runner_b.batch_calls == []
    assert not events_of(orch_b, cid, pers.EVENT_MODAL_EVALUATION_RESET)
    assert len(events_of(orch_b, cid, pers.EVENT_MODAL_RUNTIME_BOUND)) == 1
    halt = json.loads(events_of(orch_b, cid, pers.EVENT_HALTED)[0]["payload_json"])
    assert "could not restore persisted contender" in halt["reason"]


async def test_restart_probe_failure_halts_with_failed_probe_evidence(
    orchestrator_factory, fixture_repos, tmp_path
) -> None:
    _orch_a, cid = await _crash_mid_evaluation(
        orchestrator_factory, fixture_repos, tmp_path
    )
    runner_b = RuntimeRunner(tmp_path / "work" / "outputs", "b", fail_reprobe=True)
    orch_b = orchestrator_factory(runner=runner_b, repos=fixture_repos)

    await orch_b.step(FINALIZATION + 5 * M)

    assert phase(orch_b, cid) is Phase.EVALUATING
    assert pers.is_halted(orch_b.conn, cid)
    assert runner_b.batch_calls == []
    assert not events_of(orch_b, cid, pers.EVENT_MODAL_EVALUATION_RESET)
    failed = orch_b.conn.execute(
        "SELECT isolation_probe_json FROM sandboxes"
        " WHERE competition_id = ? AND status = 'FAILED' ORDER BY sandbox_id DESC",
        (cid,),
    ).fetchone()
    assert failed is not None
    assert json.loads(failed["isolation_probe_json"])["passed"] is False
