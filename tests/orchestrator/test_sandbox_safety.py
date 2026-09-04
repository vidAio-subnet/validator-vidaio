"""Untrusted sandbox behaviour against the REAL local Docker daemon.

Covers the three sandbox findings of the review service review:
  #3  a container that emits its output as a symlink to a host file,
  #12 an image that lies to the isolation probe (fake `wget`), caught by the
      host-observed verdict even when the runner is misconfigured,
  #13 a container that fills /output, killed by the host-side byte watchdog.

All three must be CONTENDER faults: typed, attributable, and never a halt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vidaio.competition.interfaces import BatchItem, ContenderSpec
from vidaio.competition.runners import (
    DockerSandboxRunner,
    LocalRepoProvider,
    OutputRejectedError,
    OversizeOutputError,
    SolutionExitError,
)
from vidaio.competition.runners.errors import ContenderFaultError
from vidaio.competition.orchestrator.failures import Fault, classify_failure

from orchestrator_support import CONTENDER_SHAS, DOCKER, repo_url

pytestmark = pytest.mark.docker

HOST_SECRET = Path("/etc/passwd")


def _runner(root: Path, fixture_repos, **overrides):
    kwargs = dict(
        inputs_dir=root / "inputs",
        outputs_dir=root / "outputs",
        scratch_dir=root / "scratch",
        docker_path=DOCKER,
        build_timeout=600.0,
        batch_timeout=180.0,
        probe_timeout=90.0,
    )
    kwargs.update(overrides)
    return DockerSandboxRunner(LocalRepoProvider(fixture_repos), **kwargs)


@pytest.fixture(scope="module")
def safety_env(tmp_path_factory, fixture_repos):
    root = tmp_path_factory.mktemp("sandbox-safety")
    return root, _runner(root, fixture_repos)


def _spec(hotkey: str, contender_id: int = 1) -> ContenderSpec:
    commit_sha, tree_sha = CONTENDER_SHAS[hotkey]
    return ContenderSpec(
        contender_id=contender_id,
        repo_url=repo_url(hotkey),
        commit_sha=commit_sha,
        tree_sha=tree_sha,
    )


def _stage(runner: DockerSandboxRunner, data: bytes) -> BatchItem:
    digest = hashlib.sha256(data).hexdigest()
    (runner.inputs_dir / digest).write_bytes(data)
    return BatchItem(item_id=1, item_index=0, input_sha256=digest, input_bytes=len(data))


# ---- #3 symlinked output ---------------------------------------------------------


def test_symlinked_output_is_rejected_and_the_host_file_is_not_archived(safety_env):
    """`<expected-name> -> /etc/passwd` must never be followed, hashed or pooled."""
    _root, runner = safety_env
    image_digest = runner.build(_spec("hk-sym", contender_id=11))
    assert runner.isolation_probe(image_digest).passed
    item = _stage(runner, b"\x11" * 4096)
    pooled_before = {p.name for p in runner.outputs_dir.iterdir()}

    with pytest.raises(OutputRejectedError) as excinfo:
        runner.run_batch(image_digest, [item], batch_index=0)
    assert "not a regular file" in str(excinfo.value)

    # It is a CONTENDER fault: the orchestrator zeroes it, never halts.
    assert isinstance(excinfo.value, ContenderFaultError)
    assert classify_failure(excinfo.value) is Fault.CONTENDER

    # Nothing new entered the pool, and no pooled blob holds the host secret.
    assert {p.name for p in runner.outputs_dir.iterdir()} == pooled_before
    secret_head = HOST_SECRET.read_bytes()[:32]
    for blob in runner.outputs_dir.iterdir():
        if blob.is_file():
            assert secret_head not in blob.read_bytes()


# ---- #13 unbounded output --------------------------------------------------------


def test_output_flood_is_killed_by_the_host_watchdog(tmp_path, fixture_repos):
    """A contender filling /output is killed mid-run and zeroed, not tolerated."""
    runner = _runner(
        tmp_path / "flood",
        fixture_repos,
        max_output_bytes=1 << 20,  # 1 MiB
        max_batch_output_bytes=2 << 20,  # 2 MiB
        output_poll_seconds=0.05,
        batch_timeout=120.0,
    )
    image_digest = runner.build(_spec("hk-flood", contender_id=12))
    item = _stage(runner, b"\x22" * 4096)
    with pytest.raises(OversizeOutputError):
        runner.run_batch(image_digest, [item], batch_index=0)
    # The host disk never accumulated the flood: the scratch run dir is gone and
    # nothing oversized reached the content-addressed pool.
    assert all(
        p.stat().st_size <= (1 << 20) for p in runner.outputs_dir.iterdir() if p.is_file()
    )
    assert not list((tmp_path / "flood" / "scratch").glob("run-*"))


def test_a_fast_log_flood_is_caught_even_though_it_exits_between_polls(
    tmp_path, fixture_repos
):
    """review #13 (round 2): the watchdog missed FAST writers.

    Log size was only measured while `proc.poll()` was None, the loop broke out on
    the exit BEFORE measuring, and the post-exit check covered /output only. A
    container that floods stdout and exits inside one poll interval therefore blew
    the cap unobserved. `hk-logflood` writes ~1 MiB against a 64 KiB cap with the
    poll interval set to 30s, so no mid-run poll can possibly see it — only the
    exit-time measurement and the final check can, and one of them must.
    """
    runner = _runner(
        tmp_path / "logflood",
        fixture_repos,
        max_log_bytes=64 * 1024,
        output_poll_seconds=30.0,  # guarantees zero useful mid-run polls
        batch_timeout=120.0,
    )
    image_digest = runner.build(_spec("hk-logflood", contender_id=19))
    item = _stage(runner, b"\x55" * 4096)
    with pytest.raises(OversizeOutputError) as excinfo:
        runner.run_batch(image_digest, [item], batch_index=0)
    assert "container logs" in str(excinfo.value)
    # Same fault class as any other cap breach: that contender is zeroed, not a halt.
    assert classify_failure(excinfo.value) is Fault.CONTENDER
    # Nothing of the flood survived on the host.
    assert not list((tmp_path / "logflood" / "scratch").glob("run-*"))


def test_a_process_that_writes_and_exits_immediately_is_still_bounded(
    tmp_path, fixture_repos
):
    """The watchdog itself, without docker: a child that floods and exits at once.

    Whether the breach is caught by the exit-iteration measurement or by the final
    post-loop check is an implementation detail; the CONTRACT is that a writer
    cannot escape the cap by being fast.
    """
    runner = _runner(
        tmp_path / "fastwriter",
        fixture_repos,
        max_log_bytes=4096,
        output_poll_seconds=30.0,
    )
    run_dir = tmp_path / "fastwriter" / "run"
    watch_dir = run_dir / "out"
    watch_dir.mkdir(parents=True)
    with pytest.raises(OversizeOutputError) as excinfo:
        runner._run_container_watched(
            ["/bin/sh", "-c", "yes vidaio | head -c 200000"],
            "not-a-container",
            run_dir=run_dir,
            timeout=60.0,
            watch_dir=watch_dir,
            byte_cap=1 << 30,
            what="fast writer",
        )
    assert "container logs" in str(excinfo.value)
    assert classify_failure(excinfo.value) is Fault.CONTENDER


def test_a_well_behaved_short_run_is_not_flagged(tmp_path, fixture_repos):
    """The bound must not fire on ordinary output — no false contender faults."""
    runner = _runner(
        tmp_path / "quiet", fixture_repos, max_log_bytes=4096, output_poll_seconds=30.0
    )
    run_dir = tmp_path / "quiet" / "run"
    watch_dir = run_dir / "out"
    watch_dir.mkdir(parents=True)
    returncode, stdout_path, _stderr = runner._run_container_watched(
        ["/bin/sh", "-c", "echo hello"],
        "not-a-container",
        run_dir=run_dir,
        timeout=60.0,
        watch_dir=watch_dir,
        byte_cap=1 << 30,
        what="quiet run",
    )
    assert returncode == 0
    assert stdout_path.read_bytes() == b"hello\n"


def test_per_output_cap_is_enforced_after_the_run(tmp_path, fixture_repos):
    """A modest output over the per-output cap is rejected even if the batch cap
    was never crossed mid-run."""
    runner = _runner(
        tmp_path / "cap",
        fixture_repos,
        max_output_bytes=128,  # hk-a truncates to 512 bytes
        max_batch_output_bytes=1 << 30,
        output_poll_seconds=0.05,
    )
    image_digest = runner.build(_spec("hk-a", contender_id=13))
    item = _stage(runner, bytes(range(256)) * 16)
    with pytest.raises(OversizeOutputError) as excinfo:
        runner.run_batch(image_digest, [item], batch_index=0)
    assert "per-output cap" in str(excinfo.value)
    assert classify_failure(excinfo.value) is Fault.CONTENDER


# ---- #14 contender exit codes ----------------------------------------------------


def test_solution_exit_one_is_a_typed_contender_fault(safety_env):
    _root, runner = safety_env
    image_digest = runner.build(_spec("hk-exit", contender_id=14))
    item = _stage(runner, b"\x33" * 4096)
    with pytest.raises(SolutionExitError) as excinfo:
        runner.run_batch(image_digest, [item], batch_index=0)
    assert "refuses to work" in str(excinfo.value)  # stderr tail is carried
    assert classify_failure(excinfo.value) is Fault.CONTENDER


def test_solution_that_writes_nothing_yields_no_outputs_not_an_error(safety_env):
    """Absent outputs are the scorer's call — the runner never substitutes one."""
    _root, runner = safety_env
    image_digest = runner.build(_spec("hk-silent", contender_id=15))
    item = _stage(runner, b"\x44" * 4096)
    assert runner.run_batch(image_digest, [item], batch_index=0) == []


# ---- #12 spoofable probe ---------------------------------------------------------


def test_probe_verdict_comes_from_the_host_not_the_image(tmp_path, fixture_repos):
    """The regression this exists for.

    `hk-lie` ships a fake `wget` that always exits 1, so the in-container script
    reports NETWORK_ATTEMPT=0 ("egress blocked") no matter what. Run it through a
    runner regressed to `--network bridge` and the probe must STILL fail, purely
    from `docker inspect`. No internet access is needed to prove it.
    """
    root = tmp_path / "lying-probe"
    honest_runner = _runner(root, fixture_repos)
    image_digest = honest_runner.build(_spec("hk-lie", contender_id=16))

    # Sanity: correctly configured, the same image passes.
    good = honest_runner.isolation_probe(image_digest)
    assert good.passed, good.details
    good_detail = json.loads(good.details)
    assert good_detail["host"]["network_mode"] == "none"
    assert good_detail["host"]["networks"] == ["none"]

    regressed = _runner(root, fixture_repos, network_mode="bridge")
    report = regressed.isolation_probe(image_digest)
    detail = json.loads(report.details)

    # The image's own answer is the LIE that used to make this pass.
    assert detail["container"]["completed"] is True
    assert detail["container"]["network_attempt"] == "0"  # "blocked", it claims
    # The host disagrees, and the host wins.
    assert detail["host"]["network_mode"] == "bridge"
    assert detail["host"]["network_isolated"] is False
    assert not report.network_blocked
    assert not report.passed
    assert "host-observed facts are authoritative" in detail["trust"]


def test_batch_run_is_also_host_verified_not_just_the_probe(tmp_path, fixture_repos):
    """A runner regression is caught on the container that ACTUALLY ran a batch —
    and it is an INFRA fault (our bug), so it halts rather than zeroing anyone."""
    from vidaio.competition.runners.errors import SandboxIsolationError

    root = tmp_path / "regressed-batch"
    honest = _runner(root, fixture_repos)
    image_digest = honest.build(_spec("hk-a", contender_id=17))
    regressed = _runner(root, fixture_repos, network_mode="bridge")
    item = _stage(regressed, bytes(range(256)) * 16)
    with pytest.raises(SandboxIsolationError) as excinfo:
        regressed.run_batch(image_digest, [item], batch_index=0)
    assert "network_not_isolated" in str(excinfo.value)
    assert classify_failure(excinfo.value) is Fault.INFRA


def test_probe_records_the_full_host_fact_set(safety_env):
    """The attestation is evidence, not a boolean: the persisted details carry the
    host's mount/privilege/env observations for the audit trail (spec §05)."""
    _root, runner = safety_env
    image_digest = runner.build(_spec("hk-a", contender_id=18))
    detail = json.loads(runner.isolation_probe(image_digest).details)["host"]
    assert detail["readonly_rootfs"] is True
    assert detail["cap_drop"] == ["ALL"]
    assert detail["cap_add"] == []
    assert any(o.startswith("no-new-privileges") for o in detail["security_opt"])
    assert detail["tmpfs"] == ["/tmp"]
    assert sorted(detail["mounts"]) == ["/evaluation-inputs", "/vidaio-probe"]
    assert detail["mounts"]["/evaluation-inputs"]["rw"] is False
    assert detail["mount_problems"] == []
    assert detail["env_leaked"] == []
