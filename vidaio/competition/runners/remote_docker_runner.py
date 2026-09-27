"""SandboxRunner on an operator-owned GPU host, reached through an SSH-forwarded daemon.

This is the fallback sandbox for a running competition when the managed GPU backend
(Modal) is unavailable.  It is the local :class:`DockerSandboxRunner` with the same
isolation contract and the same host-observed attestation, pointed at a remote
Docker daemon:

* the daemon is reached ONLY through an SSH local forward of its unix socket to a
  private unix socket of the orchestrator (``ssh -N -L <0700 dir>/docker.sock:/var/
  run/docker.sock``), with a dedicated key and a pinned known_hosts file.  Nothing on
  the GPU host listens on the network for this;
* ``docker build`` streams the pinned checkout from the orchestrator, exactly as
  locally, so the contender repository never has to be fetched on the GPU host;
* batch inputs and outputs travel as named Docker volumes filled and drained with
  ``docker cp`` through a pinned, trusted helper image (never the contender image);
  outputs are then collected by the unchanged symlink-safe local path;
* every run gets ``--gpus`` on top of the unchanged isolation flags (no network,
  read-only root, all capabilities dropped, no-new-privileges, pids/memory/cpu
  bounds, a bounded /tmp), and ``docker inspect`` of the container that actually
  ran is checked against the same contract as the local runner.

Restart fence.  The runner exposes ``runtime_session_id``/``runtime_label``/
``has_live_image`` like the Modal runner.  The session id is derived from the SSH
target, the remote daemon's own id and the GPU model, so restarting the orchestrator
against the SAME host keeps built images and completed batches, while switching
from Modal (or to another GPU host) presents a new session: the orchestrator then
rebuilds every BUILT contender on the new host, re-probes it and, during
EVALUATING, reruns the whole matrix there, so every subject is scored from one
runtime.  ``gpu`` reports the detected model for the manifest's ``allowed_gpus``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from vidaio.core.logging import get_logger, log_fields
from vidaio.competition.interfaces import BatchItem, BatchOutput, IsolationProbeReport
from vidaio.competition.runners.docker_runner import (
    _LABEL,
    _PROBE_SCRIPT,
    DockerSandboxRunner,
    _read_tail,
    host_isolation_facts,
)
from vidaio.competition.runners.errors import (
    BatchExecutionError,
    BatchTimeout,
    InputStagingError,
    OversizeOutputError,
    RunnerUnavailableError,
    SandboxProbeUnavailableError,
    SolutionExitError,
    UnknownImageError,
)
from vidaio.competition.runners.repo import RepoProvider

logger = get_logger("vidaio.competition.runners.remote_docker")

#: Volumes we create carry this label so a crashed orchestrator's leftovers are ours.
_VOLUME_LABEL = "vidaio.sandbox.volume=1"
_SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9.:-]+$")
_GPU_TOKEN_RE = re.compile(r"[^A-Z0-9]+")


def normalize_gpu_name(name: str) -> str:
    """``NVIDIA A100-SXM4-80GB`` -> ``A100``; ``NVIDIA L40S`` -> ``L40S``;
    ``NVIDIA H100 80GB HBM3`` -> ``H100``; ``NVIDIA RTX 6000 Ada Generation`` ->
    ``RTX6000ADA``.  Comparable with the manifest's ``allowed_gpus`` entries."""
    text = name.upper().replace("NVIDIA", " ").strip()
    first = re.split(r"[-\s]+(?=\d+GB|SXM|PCIE|HBM|NVL)", text)[0]
    return _GPU_TOKEN_RE.sub("", first.replace("GENERATION", ""))


class SshDockerTunnel:
    """One ``ssh -N -L`` forward of the remote Docker socket to a private unix socket."""

    def __init__(
        self,
        *,
        target: str,
        key_path: str | Path,
        known_hosts_path: str | Path,
        port: int,
        ssh_path: str = "ssh",
        remote_socket: str = "/var/run/docker.sock",
        connect_timeout: float = 20.0,
    ) -> None:
        if not _SSH_TARGET_RE.fullmatch(target):
            raise ValueError(f"ssh target must be user@host, got {target!r}")
        self.target = target
        self.port = int(port)
        #: The forward ends in a unix socket inside a private 0700 directory, so only
        #: this process's user can reach the remote daemon (a loopback TCP port would
        #: be reachable by every process sharing the network namespace).
        self._socket_dir = Path(tempfile.mkdtemp(prefix="vidaio-remote-docker-"))
        os.chmod(self._socket_dir, 0o700)
        self.socket_path = self._socket_dir / "docker.sock"
        self._argv = [
            ssh_path,
            "-N",
            "-T",
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts_path}",
            "-o", "IdentitiesOnly=yes",
            "-o", f"ConnectTimeout={int(connect_timeout)}",
            "-o", "StreamLocalBindUnlink=yes",
            "-o", "StreamLocalBindMask=0177",
            "-i", str(key_path),
            "-L", f"{self.socket_path}:{remote_socket}",
            target,
        ]
        self._connect_timeout = connect_timeout
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    @property
    def docker_host(self) -> str:
        return f"unix://{self.socket_path}"

    def _port_open(self) -> bool:
        if not self.socket_path.exists():
            return False
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(2.0)
        try:
            probe.connect(str(self.socket_path))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    def ensure(self) -> None:
        """Start (or restart) the forward and wait until the port accepts."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._port_open():
                return
            self.close_locked()
            try:
                self._proc = subprocess.Popen(
                    self._argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
            except OSError as exc:
                raise RunnerUnavailableError(f"cannot start ssh: {exc}") from exc
            deadline = time.monotonic() + self._connect_timeout + 10.0
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    err = (self._proc.stderr.read() if self._proc.stderr else b"")[-500:]
                    raise RunnerUnavailableError(
                        f"ssh forward to {self.target} exited "
                        f"{self._proc.returncode}: {err.decode('utf-8', 'replace')}"
                    )
                if self._port_open():
                    return
                time.sleep(0.2)
            self.close_locked()
            raise RunnerUnavailableError(
                f"ssh forward to {self.target} did not open {self.socket_path}"
            )

    def close_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def close(self) -> None:
        with self._lock:
            self.close_locked()


class RemoteDockerSandboxRunner(DockerSandboxRunner):
    """:class:`DockerSandboxRunner` on a remote GPU daemon (see the module doc)."""

    def __init__(
        self,
        repo_provider: RepoProvider,
        *,
        inputs_dir: str | Path,
        outputs_dir: str | Path,
        scratch_dir: str | Path,
        ssh_target: str,
        ssh_key_path: str | Path,
        ssh_known_hosts_path: str | Path,
        helper_image: str,
        tunnel_port: int = 23750,
        gpus: str = "all",
        expected_gpu: str | None = None,
        ssh_path: str = "ssh",
        volume_poll_seconds: float = 5.0,
        max_output_entries: int = 4096,
        tunnel: SshDockerTunnel | None = None,
        **docker_kwargs: Any,
    ) -> None:
        if "@sha256:" not in helper_image:
            raise ValueError("helper_image must be pinned by digest (name@sha256:...)")
        self._tunnel = tunnel or SshDockerTunnel(
            target=ssh_target,
            key_path=ssh_key_path,
            known_hosts_path=ssh_known_hosts_path,
            port=tunnel_port,
            ssh_path=ssh_path,
        )
        self._ssh_target = ssh_target
        self._helper_image = helper_image
        self._gpus = gpus
        self._volume_poll_seconds = max(1.0, float(volume_poll_seconds))
        self._max_output_entries = int(max_output_entries)
        #: local watch path -> (remote output volume, last poll time, last bytes)
        self._watched_volumes: dict[str, list[Any]] = {}
        try:
            self._tunnel.ensure()
        except RunnerUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RunnerUnavailableError(f"ssh forward failed: {exc}") from exc
        super().__init__(
            repo_provider,
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            scratch_dir=scratch_dir,
            **docker_kwargs,
        )
        try:
            self._daemon_id = self._run_docker(
                ["info", "--format", "{{.ID}}"], timeout=30.0, err_cls=RunnerUnavailableError
            ).strip()
            self._ensure_helper_image()
            raw_gpu = self._detect_gpu()
        except RunnerUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RunnerUnavailableError(f"remote GPU host is not usable: {exc}") from exc
        self.gpu_model = raw_gpu
        self.gpu = normalize_gpu_name(raw_gpu)
        if expected_gpu is not None and self.gpu != normalize_gpu_name(expected_gpu):
            raise RunnerUnavailableError(
                f"remote GPU is {raw_gpu!r} ({self.gpu}); configured {expected_gpu!r}"
            )
        self.runtime_session_id = hashlib.sha256(
            json.dumps(
                {
                    "scheme": "vidaio.competition.remote-docker-runtime.v1",
                    "target": ssh_target,
                    "daemon_id": self._daemon_id,
                    "gpu": self.gpu,
                    "gpus": gpus,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        host = re.sub(r"[^a-z0-9-]+", "-", ssh_target.split("@", 1)[1].lower())
        self.runtime_label = f"vidaio-next-remote-{host}-{self.runtime_session_id[:12]}"
        self._reap_stale_volumes()
        logger.info(
            "remote GPU sandbox ready",
            extra=log_fields(
                target=ssh_target,
                gpu=raw_gpu,
                runtime_label=self.runtime_label,
            ),
        )

    # ---- restart fence ------------------------------------------------------------

    def has_live_image(self, image_digest: str) -> bool:
        try:
            self._resolve_image(image_digest)
            return True
        except (UnknownImageError, BatchExecutionError):
            return False

    def available(self) -> bool:
        try:
            self._tunnel.ensure()
        except Exception:  # noqa: BLE001
            return False
        return super().available()

    # ---- docker plumbing ------------------------------------------------------------

    def _cli(self) -> list[str]:
        host = getattr(self._tunnel, "docker_host", None) or f"tcp://127.0.0.1:{self._tunnel.port}"
        return [self._docker, "-H", host]

    def _isolation_flags(self) -> list[str]:
        flags = super()._isolation_flags()
        if self._gpus:
            flags += ["--gpus", self._gpus]
        return flags

    def _ensure_helper_image(self) -> None:
        try:
            self._run_docker(
                ["image", "inspect", self._helper_image, "--format", "{{.Id}}"],
                timeout=30.0,
                err_cls=BatchExecutionError,
            )
        except BatchExecutionError:
            self._run_docker(
                ["pull", self._helper_image], timeout=900.0, err_cls=RunnerUnavailableError
            )

    def _detect_gpu(self) -> str:
        if not self._gpus:
            return "none"
        out = self._run_docker(
            [
                "run", "--rm", "--network", "none", "--gpus", self._gpus,
                "-e", "NVIDIA_DRIVER_CAPABILITIES=utility",
                "--entrypoint", "nvidia-smi", self._helper_image,
                "--query-gpu=name", "--format=csv,noheader",
            ],
            timeout=120.0,
            err_cls=RunnerUnavailableError,
        )
        names = [line.strip() for line in out.splitlines() if line.strip()]
        if not names:
            raise RunnerUnavailableError("nvidia-smi reported no GPU on the remote host")
        if len({normalize_gpu_name(n) for n in names}) != 1:
            raise RunnerUnavailableError(f"remote host mixes GPU models: {names}")
        return names[0]

    def _volume_create(self, name: str) -> str:
        self._run_docker(
            ["volume", "create", "--label", _VOLUME_LABEL, name],
            timeout=30.0,
            err_cls=BatchExecutionError,
        )
        return self._run_docker(
            ["volume", "inspect", name, "--format", "{{.Mountpoint}}"],
            timeout=30.0,
            err_cls=BatchExecutionError,
        ).strip()

    def _volume_remove(self, name: str) -> None:
        try:
            subprocess.run(
                [*self._cli(), "volume", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=60.0,
            )
        except Exception:  # noqa: BLE001 - cleanup never masks the primary error
            pass

    def _helper(self, args: list[str], mounts: list[tuple[str, str, bool]], *, timeout: float = 300.0) -> str:
        """Run the trusted helper image (no network) with the given volume mounts."""
        argv = ["run", "--rm", "--network", "none", "--label", _LABEL]
        for volume, dest, rw in mounts:
            argv += ["-v", f"{volume}:{dest}:{'rw' if rw else 'ro'}"]
        argv += ["--entrypoint", args[0], self._helper_image, *args[1:]]
        return self._run_docker(argv, timeout=timeout, err_cls=BatchExecutionError)

    def _copy_into_volume(self, volume: str, source_dir: Path) -> None:
        name = f"vidaio-sbx-stage-{uuid.uuid4().hex[:12]}"
        try:
            self._run_docker(
                ["create", "--name", name, "--label", _LABEL, "--network", "none",
                 "-v", f"{volume}:/stage:rw", "--entrypoint", "true", self._helper_image],
                timeout=60.0,
                err_cls=InputStagingError,
            )
            self._run_docker(
                ["cp", f"{source_dir}/.", f"{name}:/stage/"],
                timeout=1800.0,
                err_cls=InputStagingError,
            )
        finally:
            self._force_remove(name)
        # contender images may run as any uid: inputs readable, never writable
        self._helper(["chmod", "-R", "a+rX,go-w", "/stage"], [(volume, "/stage", True)])

    def _copy_out_of_volume(
        self, volume: str, target_dir: Path, items: Sequence[BatchItem]
    ) -> None:
        """Copy back ONLY the expected per-item outputs (never the whole tree)."""
        listing = self._helper(
            # an empty output directory is a normal result (every item then scores
            # zero on its own); the listing must succeed either way
            ["sh", "-c", "cd /collect && for f in *; do [ -e \"$f\" ] && echo \"$f\"; done; exit 0"],
            [(volume, "/collect", False)],
            timeout=120.0,
        )
        present = set(listing.split())
        name = f"vidaio-sbx-collect-{uuid.uuid4().hex[:12]}"
        try:
            self._run_docker(
                ["create", "--name", name, "--label", _LABEL, "--network", "none",
                 "-v", f"{volume}:/collect:ro", "--entrypoint", "true", self._helper_image],
                timeout=60.0,
                err_cls=BatchExecutionError,
            )
            for item in items:
                if item.input_sha256 not in present:
                    continue  # no output for this item: the scorer zero-scores it
                # docker cp copies a symbolic link as a link and a directory as a
                # directory; the local collector refuses both as contender faults
                self._run_docker(
                    ["cp", f"{name}:/collect/{item.input_sha256}", f"{target_dir}/"],
                    timeout=600.0,
                    err_cls=BatchExecutionError,
                )
        finally:
            self._force_remove(name)

    def _volume_usage(self, volume: str) -> tuple[int, int]:
        """(bytes, entries) of a volume; entries are counted up to the cap + 1 so a
        contender that floods the output cannot make the measurement itself slow."""
        cap = self._max_output_entries + 1
        script = (
            f"find /v -mindepth 1 | head -n {cap} | wc -l; "
            "du -sb /v | cut -f1"
        )
        out = self._helper(["sh", "-c", script], [(volume, "/v", False)], timeout=120.0)
        try:
            entries, total = (int(x) for x in out.split()[:2])
        except ValueError as exc:
            raise BatchExecutionError(f"cannot size volume {volume}: {out!r}") from exc
        return total, entries

    def _watched_bytes(self, watch_dir: Path, *, final: bool = False) -> int:
        entry = self._watched_volumes.get(str(watch_dir))
        if entry is None:
            return super()._watched_bytes(watch_dir, final=final)
        volume, polled_at, last = entry
        now = time.monotonic()
        if final or now - polled_at >= self._volume_poll_seconds:
            try:
                last, entries = self._volume_usage(volume)
            except BatchExecutionError:
                if final:
                    raise
            else:
                if entries > self._max_output_entries:
                    # a contender fault: reported as an oversize output, never INFRA
                    return self._max_batch_output_bytes + 1
            entry[1], entry[2] = now, last
        return int(last)

    def _reap_stale_volumes(self) -> None:
        try:
            names = self._run_docker(
                ["volume", "ls", "-q", "--filter", f"label={_VOLUME_LABEL}"],
                timeout=30.0,
                err_cls=BatchExecutionError,
            ).split()
        except Exception:  # noqa: BLE001
            return
        for name in names:
            self._volume_remove(name)

    # ---- SandboxRunner: run_batch ----------------------------------------------------

    def run_batch(
        self, image_digest: str, items: Sequence[BatchItem], batch_index: int
    ) -> Sequence[BatchOutput]:
        self._tunnel.ensure()
        image = self._resolve_image(image_digest)
        tag = uuid.uuid4().hex[:12]
        run_dir = self._scratch_dir / f"run-{image_digest[:16]}-b{batch_index}-{tag}"
        in_dir = run_dir / "inputs"
        out_dir = run_dir / "out"
        watch_dir = run_dir / "watch"
        for d in (in_dir, out_dir, watch_dir):
            d.mkdir(parents=True)
        vin, vout = f"vidaio-sbx-in-{tag}", f"vidaio-sbx-out-{tag}"
        name = f"vidaio-sbx-{tag}"
        try:
            for item in items:
                self._stage_input(item, in_dir)
            in_mount = self._volume_create(vin)
            out_mount = self._volume_create(vout)
            self._copy_into_volume(vin, in_dir)
            self._helper(["chmod", "1777", "/out"], [(vout, "/out", True)])
            self._watched_volumes[str(watch_dir)] = [vout, 0.0, 0]
            argv = [
                *self._cli(),
                "run",
                "--name", name,
                "--label", _LABEL,
                *self._isolation_flags(),
                "-v", f"{vin}:/evaluation-inputs:ro",
                "-v", f"{vout}:/output:rw",
                "--entrypoint", "/bin/sh",
                image,
                "/app/run.sh", "/evaluation-inputs", "/output",
            ]
            started = time.monotonic()
            try:
                returncode, _stdout, stderr_path = self._run_container_watched(
                    argv,
                    name,
                    run_dir=run_dir,
                    timeout=self._batch_timeout,
                    watch_dir=watch_dir,
                    byte_cap=self._max_batch_output_bytes,
                    what=f"batch {batch_index} ({image_digest[:16]})",
                )
                elapsed = time.monotonic() - started
                self._assert_host_isolation(
                    name,
                    image,
                    expected_mounts={
                        "/evaluation-inputs": (False, in_mount),
                        "/output": (True, out_mount),
                    },
                    what=f"batch {batch_index}",
                )
            finally:
                self._force_remove(name)
            if returncode != 0:
                raise SolutionExitError(
                    f"batch {batch_index} ({image_digest[:16]}) solution exited "
                    f"{returncode}: {_read_tail(stderr_path)}"
                )
            self._copy_out_of_volume(vout, out_dir, items)
            outputs = self._collect_outputs(out_dir, items, elapsed)
            logger.info(
                "remote batch executed",
                extra=log_fields(
                    image_digest=image_digest,
                    batch_index=batch_index,
                    items=len(items),
                    outputs=len(outputs),
                    wall_seconds=round(elapsed, 3),
                    runtime_label=self.runtime_label,
                ),
            )
            return outputs
        finally:
            self._watched_volumes.pop(str(watch_dir), None)
            self._volume_remove(vin)
            self._volume_remove(vout)
            shutil.rmtree(run_dir, ignore_errors=True)

    # ---- SandboxRunner: isolation_probe ------------------------------------------------

    def isolation_probe(self, image_digest: str) -> IsolationProbeReport:
        """Same verdict rules as the local runner (host facts decide), on the GPU host."""
        try:
            self._tunnel.ensure()
            image = self._resolve_image(image_digest)
            declared_env = self._image_declared_env(image)
        except (BatchExecutionError, RunnerUnavailableError) as exc:
            raise SandboxProbeUnavailableError(
                f"isolation probe could not prepare image {image_digest}: {exc}"
            ) from exc
        tag = uuid.uuid4().hex[:12]
        probe_dir = self._scratch_dir / f"probe-{tag}"
        script_dir = probe_dir / "script"
        watch_dir = probe_dir / "watch"
        script_dir.mkdir(parents=True)
        watch_dir.mkdir(parents=True)
        (script_dir / "probe.sh").write_text(_PROBE_SCRIPT)
        vin, vscript = f"vidaio-sbx-probe-in-{tag}", f"vidaio-sbx-probe-sh-{tag}"
        name = f"vidaio-probe-{tag}"
        container_stdout = ""
        try:
            try:
                in_mount = self._volume_create(vin)
                script_mount = self._volume_create(vscript)
                self._copy_into_volume(vscript, script_dir)
            except Exception as exc:  # noqa: BLE001
                raise SandboxProbeUnavailableError(
                    f"isolation probe volumes could not be prepared: {exc}"
                ) from exc
            argv = [
                *self._cli(),
                "run",
                "--name", name,
                "--label", _LABEL,
                *self._isolation_flags(),
                "-v", f"{vin}:/evaluation-inputs:ro",
                "-v", f"{vscript}:/vidaio-probe:ro",
                "--entrypoint", "/bin/sh",
                image,
                "/vidaio-probe/probe.sh",
            ]
            try:
                returncode, stdout_path, _stderr = self._run_container_watched(
                    argv,
                    name,
                    run_dir=probe_dir,
                    timeout=self._probe_timeout,
                    watch_dir=watch_dir,
                    byte_cap=self._max_output_bytes,
                    what="isolation probe",
                )
                container_stdout = _read_tail(stdout_path, limit=64 * 1024)
                container_note = "" if returncode == 0 else f"probe script exited {returncode}"
            except (BatchTimeout, OversizeOutputError) as exc:
                container_note = f"probe script aborted: {exc}"
            except BatchExecutionError as exc:
                raise SandboxProbeUnavailableError(
                    f"isolation probe container could not be launched: {exc}"
                ) from exc
            try:
                info = self._inspect_container(name)
            except BatchExecutionError as exc:
                raise SandboxProbeUnavailableError(
                    f"host inspection of the probe container failed: {exc}"
                ) from exc
            facts = host_isolation_facts(
                info,
                expected_mounts={
                    "/evaluation-inputs": (False, in_mount),
                    "/vidaio-probe": (False, script_mount),
                },
                declared_env=declared_env,
            )
            return self._compose_report(facts, container_stdout, container_note)
        finally:
            self._force_remove(name)
            self._volume_remove(vin)
            self._volume_remove(vscript)
            shutil.rmtree(probe_dir, ignore_errors=True)

    def close(self) -> None:
        self._tunnel.close()


__all__ = ["RemoteDockerSandboxRunner", "SshDockerTunnel", "normalize_gpu_name"]
