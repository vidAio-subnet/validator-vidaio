"""RemoteDockerSandboxRunner against a scripted docker CLI (no daemon needed).

The fake CLI keeps named volumes as local directories and emulates just enough of
`volume`, `create`, `cp`, `run`, `inspect` and `rm` to drive the runner end to end:
inputs travel into a volume through the trusted helper, the contender runs with
`--gpus` and the unchanged isolation flags, the host facts are checked against the
volume mountpoints, outputs come back through `docker cp`, and nothing is left behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from vidaio.competition.interfaces import BatchItem
from vidaio.competition.runners import LocalRepoProvider
from vidaio.competition.runners.errors import (
    OutputRejectedError,
    RunnerUnavailableError,
    SandboxIsolationError,
)
from vidaio.competition.runners.remote_docker_runner import (
    RemoteDockerSandboxRunner,
    normalize_gpu_name,
)

HELPER = "python:3.13-slim-bookworm@sha256:" + "0" * 64

FAKE_DOCKER = r'''#!PYTHON
import json, os, shutil, sys
from pathlib import Path
STATE = Path(os.environ["FAKE_DOCKER_STATE"])
STATE.mkdir(parents=True, exist_ok=True)
log = STATE / "calls.jsonl"
args = sys.argv[1:]
if args[:1] == ["-H"]:
    host, args = args[1], args[2:]
with log.open("a") as fh:
    fh.write(json.dumps(args) + "\n")
vols = STATE / "volumes"; vols.mkdir(exist_ok=True)
cons = STATE / "containers"; cons.mkdir(exist_ok=True)
behaviour = os.environ.get("FAKE_BEHAVIOUR", "honest")
def vol(name): return vols / name / "_data"
def mounts_of(argv):
    out = []
    for i, a in enumerate(argv):
        if a == "-v":
            src, dest, mode = argv[i + 1].split(":")
            out.append((src, dest, mode))
    return out
cmd = args[0]
if cmd == "version":
    print("27.0"); sys.exit(0)
if cmd == "info":
    print("daemon-" + os.environ.get("FAKE_DAEMON", "1")); sys.exit(0)
if cmd == "ps" or (cmd == "volume" and args[1] == "ls"):
    sys.exit(0)
if cmd == "image":
    if "{{json .Config.Env}}" in args:
        print("[]")
    else:
        print("sha256:" + "a" * 64)
    sys.exit(0)
if cmd == "volume":
    if args[1] == "create":
        vol(args[-1]).mkdir(parents=True, exist_ok=True); print(args[-1]); sys.exit(0)
    if args[1] == "inspect":
        print(str(vol(args[2]))); sys.exit(0)
    if args[1] == "rm":
        shutil.rmtree(vols / args[-1], ignore_errors=True); sys.exit(0)
if cmd == "create":
    name = args[args.index("--name") + 1]
    (cons / name).write_text(json.dumps({"argv": args}))
    sys.exit(0)
if cmd == "rm":
    (cons / args[-1]).unlink(missing_ok=True); sys.exit(0)
if cmd == "cp":
    src, dst = args[1], args[2]
    if ":" in dst.split("/")[0]:
        name, path = dst.split(":", 1)
        info = json.loads((cons / name).read_text())
        m = {d: s for s, d, _ in mounts_of(info["argv"])}
        target = vol(m[path.rstrip("/")])
        shutil.copytree(src.rstrip(".").rstrip("/"), target, dirs_exist_ok=True)
    else:
        name, path = src.split(":", 1)
        info = json.loads((cons / name).read_text())
        m = {d: s for s, d, _ in mounts_of(info["argv"])}
        mount, _, rel = path.lstrip("/").partition("/")
        source = vol(m["/" + mount]) / rel
        target = Path(dst) / source.name
        if source.is_symlink():
            os.symlink(os.readlink(source), target)
        elif source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target)
    sys.exit(0)
if cmd == "container" and args[1] == "inspect":
    info = json.loads((cons / args[2]).read_text())
    argv = info["argv"]
    mounts = [
        {"Type": "volume", "Source": str(vol(s)), "Destination": d, "RW": mode == "rw"}
        for s, d, mode in mounts_of(argv)
    ]
    readonly = "--read-only" in argv and behaviour != "writable_root"
    print(json.dumps({
        "HostConfig": {
            "NetworkMode": "none", "ReadonlyRootfs": readonly, "CapDrop": ["ALL"],
            "CapAdd": None, "SecurityOpt": ["no-new-privileges"], "Privileged": False,
            "Tmpfs": {"/tmp": "rw"},
            "DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}] if "--gpus" in argv else None,
        },
        "NetworkSettings": {"Networks": {}},
        "Config": {"Env": []},
        "Mounts": mounts,
    }))
    sys.exit(0)
if cmd == "run":
    if "nvidia-smi" in args:
        print(os.environ.get("FAKE_GPU", "NVIDIA L40S")); sys.exit(0)
    if "--rm" in args:  # trusted helper
        tail = args[args.index(os.environ["FAKE_HELPER"]) + 1:]
        entry = args[args.index("--entrypoint") + 1]
        m = {d: s for s, d, _ in mounts_of(args)}
        if entry == "sh" and "/v" in m:
            root = vol(m["/v"])
            entries = sum(1 for _ in root.rglob("*"))
            total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
            print(entries); print(total)
        elif entry == "sh" and "/collect" in m:
            import subprocess as sp
            script = args[args.index("-c") + 1].replace("/collect", str(vol(m["/collect"])))
            done = sp.run(["sh", "-c", script], capture_output=True, text=True)
            sys.stdout.write(done.stdout)
            sys.exit(done.returncode)
        sys.exit(0)
    name = args[args.index("--name") + 1]
    (cons / name).write_text(json.dumps({"argv": args}))
    m = {d: s for s, d, _ in mounts_of(args)}
    if "/vidaio-probe" in m:
        print("NETWORK_ATTEMPT=0\nINPUT_WRITE=0\nROOT_WRITE=0\nREF_MOUNTS=0\nINDEX_LEAK=0\nENV_BEGIN\nENV_END\nPROBE_DONE=1")
        sys.exit(0)
    inputs, outputs = vol(m["/evaluation-inputs"]), vol(m["/output"])
    for item in sorted(inputs.iterdir()):
        target = outputs / item.name
        if behaviour == "silent":
            continue
        if behaviour == "symlink":
            os.symlink("/etc/passwd", target)
        elif behaviour == "flood":
            for i in range(50):
                (outputs / f"junk-{i}").write_bytes(b"x")
            target.write_bytes(b"repaired:" + item.read_bytes())
        else:
            target.write_bytes(b"repaired:" + item.read_bytes())
    sys.exit(0)
print("unsupported fake docker call: " + json.dumps(args), file=sys.stderr)
sys.exit(2)
'''


class _Tunnel:
    port = 23999

    def __init__(self) -> None:
        self.ensured = 0

    def ensure(self) -> None:
        self.ensured += 1

    def close(self) -> None:
        pass


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "docker"
    script.write_text(FAKE_DOCKER.replace("#!PYTHON", "#!" + sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(tmp_path / "docker-state"))
    monkeypatch.setenv("FAKE_HELPER", HELPER)
    return script


def _runner(tmp_path: Path, docker: Path, **overrides) -> RemoteDockerSandboxRunner:
    kwargs = dict(
        inputs_dir=tmp_path / "inputs",
        outputs_dir=tmp_path / "outputs",
        scratch_dir=tmp_path / "scratch",
        ssh_target="root@gpu.example",
        ssh_key_path=tmp_path / "key",
        ssh_known_hosts_path=tmp_path / "known_hosts",
        helper_image=HELPER,
        expected_gpu="L40S",
        tunnel=_Tunnel(),
        docker_path=str(docker),
        volume_poll_seconds=1.0,
    )
    kwargs.update(overrides)
    return RemoteDockerSandboxRunner(LocalRepoProvider({}), **kwargs)


def _item(runner: RemoteDockerSandboxRunner, data: bytes, item_id: int = 1) -> BatchItem:
    digest = hashlib.sha256(data).hexdigest()
    (runner.inputs_dir / digest).write_bytes(data)
    return BatchItem(item_id=item_id, item_index=item_id - 1, input_sha256=digest, input_bytes=len(data))


def _calls(tmp_path: Path) -> list[list[str]]:
    lines = (tmp_path / "docker-state" / "calls.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_gpu_names_normalize_to_manifest_tokens() -> None:
    assert normalize_gpu_name("NVIDIA A100-SXM4-80GB") == "A100"
    assert normalize_gpu_name("NVIDIA H100 80GB HBM3") == "H100"
    assert normalize_gpu_name("L40S") == "L40S"


def test_batch_runs_on_the_remote_host_with_gpus_and_volumes(tmp_path, fake_docker) -> None:
    runner = _runner(tmp_path, fake_docker)
    assert runner.gpu == "L40S"
    assert len(runner.runtime_session_id) == 64
    assert runner.runtime_label.startswith("vidaio-next-remote-")
    items = [_item(runner, b"clip-one", 1), _item(runner, b"clip-two", 2)]
    image = "e" * 64

    outputs = runner.run_batch(image, items, 0)

    assert len(outputs) == 2
    for item, out in zip(items, outputs):
        pooled = runner.outputs_dir / out.output_sha256
        assert pooled.read_bytes().startswith(b"repaired:")
    contender_runs = [c for c in _calls(tmp_path) if c[0] == "run" and "--rm" not in c]
    [run] = contender_runs
    assert "--gpus" in run and run[run.index("--gpus") + 1] == "all"
    assert "--network=none" in run and "--read-only" in run
    assert any(v.endswith(":/evaluation-inputs:ro") for v in run)
    assert any(v.endswith(":/output:rw") for v in run)
    # every sandbox volume was removed again
    volumes = tmp_path / "docker-state" / "volumes"
    assert not any(volumes.iterdir())


def test_same_host_keeps_its_session_and_another_gpu_host_does_not(tmp_path, fake_docker, monkeypatch) -> None:
    first = _runner(tmp_path, fake_docker)
    again = _runner(tmp_path, fake_docker)
    assert first.runtime_session_id == again.runtime_session_id
    monkeypatch.setenv("FAKE_DAEMON", "2")
    replaced = _runner(tmp_path, fake_docker)
    assert replaced.runtime_session_id != first.runtime_session_id


def test_wrong_gpu_model_is_refused_at_construction(tmp_path, fake_docker, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_GPU", "NVIDIA A100-SXM4-80GB")
    with pytest.raises(RunnerUnavailableError, match="configured 'L40S'"):
        _runner(tmp_path, fake_docker)


def test_symlinked_output_is_a_contender_fault(tmp_path, fake_docker, monkeypatch) -> None:
    runner = _runner(tmp_path, fake_docker)
    monkeypatch.setenv("FAKE_BEHAVIOUR", "symlink")
    with pytest.raises(OutputRejectedError):
        runner.run_batch("e" * 64, [_item(runner, b"clip")], 0)


def test_isolation_breach_observed_on_the_host_is_infra(tmp_path, fake_docker, monkeypatch) -> None:
    runner = _runner(tmp_path, fake_docker)
    monkeypatch.setenv("FAKE_BEHAVIOUR", "writable_root")
    with pytest.raises(SandboxIsolationError):
        runner.run_batch("e" * 64, [_item(runner, b"clip")], 0)


def test_isolation_probe_passes_on_host_facts(tmp_path, fake_docker) -> None:
    runner = _runner(tmp_path, fake_docker)
    report = runner.isolation_probe("e" * 64)
    assert report.passed, report.details
    assert json.loads(report.details)["host"]["mounts_ok"]


def test_helper_image_must_be_digest_pinned(tmp_path, fake_docker) -> None:
    with pytest.raises(ValueError, match="pinned by digest"):
        _runner(tmp_path, fake_docker, helper_image="python:3.13-slim")


def test_an_output_entry_flood_is_a_contender_fault(tmp_path, fake_docker, monkeypatch) -> None:
    from vidaio.competition.runners.errors import OversizeOutputError

    runner = _runner(tmp_path, fake_docker, max_output_entries=10)
    monkeypatch.setenv("FAKE_BEHAVIOUR", "flood")
    with pytest.raises(OversizeOutputError):
        runner.run_batch("e" * 64, [_item(runner, b"clip")], 0)


def test_a_run_without_outputs_collects_nothing_and_is_not_infra(tmp_path, fake_docker, monkeypatch) -> None:
    runner = _runner(tmp_path, fake_docker)
    monkeypatch.setenv("FAKE_BEHAVIOUR", "silent")
    assert runner.run_batch("e" * 64, [_item(runner, b"clip")], 0) == []
