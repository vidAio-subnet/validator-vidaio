"""DockerSandboxRunner against the real local Docker daemon (spec §05).

Exercises the isolation contract for real: network-blocked runs, read-only
mounts, env attestation, containment of an untrusted solution, and the typed
error surface. Images are tiny (alpine + a shell script)."""

from __future__ import annotations

import hashlib

import pytest

from vidaio.competition.interfaces import BatchItem, ContenderSpec
from vidaio.competition.runners import (
    DockerSandboxRunner,
    LocalRepoProvider,
    RunnerUnavailableError,
    UnknownImageError,
)
from vidaio.competition.runners.errors import InputStagingError

from orchestrator_support import BASELINE_URL, CONTENDER_SHAS, DOCKER, repo_url

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def runner_env(tmp_path_factory, fixture_repos):
    root = tmp_path_factory.mktemp("docker-runner")
    runner = DockerSandboxRunner(
        LocalRepoProvider(fixture_repos),
        inputs_dir=root / "inputs",
        outputs_dir=root / "outputs",
        scratch_dir=root / "scratch",
        docker_path=DOCKER,
        build_timeout=600.0,
        batch_timeout=180.0,
        probe_timeout=90.0,
    )
    return root, runner


def _stage_input(runner: DockerSandboxRunner, data: bytes) -> BatchItem:
    digest = hashlib.sha256(data).hexdigest()
    (runner.inputs_dir / digest).write_bytes(data)
    return BatchItem(
        item_id=1, item_index=0, input_sha256=digest, input_bytes=len(data)
    )


def _spec(hotkey: str, contender_id: int = 1) -> ContenderSpec:
    commit_sha, tree_sha = CONTENDER_SHAS[hotkey]
    return ContenderSpec(
        contender_id=contender_id,
        repo_url=repo_url(hotkey),
        commit_sha=commit_sha,
        tree_sha=tree_sha,
    )


def test_construction_fails_fast_without_docker(tmp_path, fixture_repos):
    with pytest.raises(RunnerUnavailableError):
        DockerSandboxRunner(
            LocalRepoProvider(fixture_repos),
            inputs_dir=tmp_path / "in",
            outputs_dir=tmp_path / "out",
            scratch_dir=tmp_path / "scratch",
            docker_path="/nonexistent/docker",
        )


def test_build_probe_and_batch_on_honest_solution(runner_env):
    _, runner = runner_env
    image_digest = runner.build(_spec("hk-a"))
    assert len(image_digest) == 64 and int(image_digest, 16) >= 0

    report = runner.isolation_probe(image_digest)
    assert report.network_blocked, report.details
    assert report.secrets_absent, report.details
    assert report.reference_mounts_absent, report.details
    assert report.index_leak_absent, report.details
    assert report.passed

    data = bytes(range(256)) * 16  # 4096 bytes
    item = _stage_input(runner, data)
    outputs = runner.run_batch(image_digest, [item], batch_index=0)
    assert len(outputs) == 1
    out = outputs[0]
    assert out.item_id == item.item_id
    assert out.output_bytes == 512  # hk-a truncates to 512 bytes ("compression")
    pooled = runner.outputs_dir / out.output_sha256
    produced = pooled.read_bytes()
    assert produced == data[:512]
    assert hashlib.sha256(produced).hexdigest() == out.output_sha256


def test_build_is_resumable_across_runner_instances(runner_env, fixture_repos):
    root, runner = runner_env
    image_digest = runner.build(_spec("hk-a"))
    # A fresh runner (new process after a crash) resolves the digest-derived tag
    # without rebuilding — run_batch works immediately.
    runner2 = DockerSandboxRunner(
        LocalRepoProvider(fixture_repos),
        inputs_dir=root / "inputs",
        outputs_dir=root / "outputs",
        scratch_dir=root / "scratch",
        docker_path=DOCKER,
    )
    item = _stage_input(runner2, b"resume-me" * 512)
    outputs = runner2.run_batch(image_digest, [item], batch_index=0)
    assert len(outputs) == 1


def test_probe_disqualifies_secret_shaped_env(runner_env):
    _, runner = runner_env
    image_digest = runner.build(_spec("hk-mal", contender_id=3))
    report = runner.isolation_probe(image_digest)
    assert not report.secrets_absent, report.details
    assert "VIDAIO_VALIDATOR_PAT" in report.details
    assert not report.passed  # ANY probe failure disqualifies
    assert report.network_blocked  # the sandbox itself still blocked egress


def test_probe_detects_unblocked_network(runner_env, fixture_repos):
    """Fault injection: a runner misconfigured with a real network must be caught
    by the probe (this is the probe's whole reason to exist).

    No internet is required and none is assumed: the verdict is host-observed
    (`docker inspect` says NetworkMode=bridge), so this holds on an air-gapped
    machine too — see the trust model in docker_runner and the lying-image
    regression in test_sandbox_safety.py."""
    root, runner = runner_env
    image_digest = runner.build(_spec("hk-a"))
    compromised = DockerSandboxRunner(
        LocalRepoProvider(fixture_repos),
        inputs_dir=root / "inputs",
        outputs_dir=root / "outputs",
        scratch_dir=root / "scratch",
        docker_path=DOCKER,
        network_mode="bridge",  # test-only knob simulating a misconfigured runner
    )
    report = compromised.isolation_probe(image_digest)
    assert not report.network_blocked, report.details
    assert not report.passed


def test_untrusted_solution_is_contained_at_runtime(runner_env):
    """The untrusted run.sh tries egress and writes outside /output; the sandbox
    must block both while the contracted output path still works."""
    _, runner = runner_env
    image_digest = runner.build(_spec("hk-mal", contender_id=3))
    data = b"\xab" * 4096
    item = _stage_input(runner, data)
    pool_before = sorted(p.name for p in runner.inputs_dir.iterdir())
    outputs = runner.run_batch(image_digest, [item], batch_index=0)
    assert len(outputs) == 1
    # Its own output records that the network attempt failed inside the sandbox.
    produced = (runner.outputs_dir / outputs[0].output_sha256).read_bytes()
    assert produced == b"NO-NETWORK"
    # The sealed input pool is untouched: no 'hack' file, input bytes intact.
    assert sorted(p.name for p in runner.inputs_dir.iterdir()) == pool_before
    assert (runner.inputs_dir / item.input_sha256).read_bytes() == data


def test_run_batch_unknown_image_raises_typed_error(runner_env):
    _, runner = runner_env
    item = _stage_input(runner, b"z" * 128)
    with pytest.raises(UnknownImageError):
        runner.run_batch("f" * 64, [item], batch_index=0)


def test_missing_sealed_input_is_an_infra_error(runner_env):
    _, runner = runner_env
    image_digest = runner.build(_spec("hk-a"))
    ghost = BatchItem(item_id=9, item_index=0, input_sha256="9" * 64, input_bytes=10)
    with pytest.raises(InputStagingError):
        runner.run_batch(image_digest, [ghost], batch_index=0)


def test_baseline_solution_builds_and_runs(runner_env):
    _, runner = runner_env
    baseline_spec = ContenderSpec(
        contender_id=99, repo_url=BASELINE_URL, commit_sha="b" * 40, tree_sha="0b" * 20
    )
    image_digest = runner.build(baseline_spec)
    assert runner.isolation_probe(image_digest).passed
    data = b"\x01\x02" * 2048
    item = _stage_input(runner, data)
    outputs = runner.run_batch(image_digest, [item], batch_index=0)
    assert outputs[0].output_bytes == 256  # baseline compresses hardest
