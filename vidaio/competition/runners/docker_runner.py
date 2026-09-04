"""DockerSandboxRunner — the local-first implementation of interfaces.SandboxRunner.

Spec: design spec §05 (sandbox isolation contract) executed with local Docker instead of
Modal (the project design record workflow rule 7: everything runs and is tested locally,
in Docker, before any deployment). ModalSandboxRunner (modal_runner.py) documents
the flag-for-flag parity mapping for the deploy-time swap.

Isolation contract enforced on every solution run:
- ``--network none``            no network of any kind
- ``--read-only``               immutable rootfs; only /tmp (bounded tmpfs) and
                                /output are writable
- ``--cap-drop ALL``            no capabilities
- ``--security-opt no-new-privileges``
- ``--pids-limit / --memory / --cpus``  bounded resources (config/manifest)
- inputs mounted READ-ONLY at /evaluation-inputs — a per-batch staging subdir
  holding ONLY that batch's items (never the whole sealed pool, never index.json)
- per-contender scratch output dir mounted READ-WRITE at /output, with a HOST-side
  byte watchdog (see "Bounded output" below)
- no env injected beyond what the image itself declares (no secrets, no OIDC)

Solution image contract (local-first; documented for miners):
    The image must contain a POSIX shell script at ``/app/run.sh``. It is invoked as

        /bin/sh /app/run.sh <input_dir> <output_dir>

    with ``<input_dir> = /evaluation-inputs`` (read-only; one media file per item,
    named by the item's input sha256 hex digest) and ``<output_dir> = /output``
    (writable). Upscaling inputs also have a hidden
    ``.vidaio-next-upscale-task-<input_sha256>`` sidecar containing canonical JSON
    with factor and exact target width/height; it exposes no reference identity
    or bytes. For every media
    input the solution must write the processed result to
    ``<output_dir>/<same filename>`` as a PLAIN REGULAR FILE. Anything else on disk
    is read-only except /tmp. There is no network. Exit code 0 on success.

Build binding (spec §05): ``image_digest`` is the versioned logical build identity
over the exact repository/commit/tree source coordinates. The digest-derived local tag
(``vidaio-sbx:<digest[:32]>``) makes builds resumable across orchestrator restarts
(the docker image store, like the DB, survives a crash; nothing is kept in memory
that cannot be re-derived).

--------------------------------------------------------------------------------
ISOLATION TRUST MODEL
--------------------------------------------------------------------------------
The probe used to derive its whole verdict from commands executed by the IMAGE
(`/bin/sh`, `wget`, `env`). An untrusted image can ship a fake `wget` that always
"fails" and print whatever probe markers it likes, so that verdict attested
nothing. The rule now is:

  HOST-OBSERVED FACTS ARE AUTHORITATIVE. Only ``docker inspect`` of the container
  we actually launched can make a probe PASS: network mode, attached networks,
  ReadonlyRootfs, the exact mount set (destination, RW flag and source path),
  tmpfs destinations, Privileged/CapAdd/CapDrop/SecurityOpt, and the container's
  effective ``Config.Env``.

  CONTAINER-REPORTED FACTS ARE ADVISORY-NEGATIVE-ONLY. The in-container script is
  still run and still parsed, but its answers can only ever turn a True into a
  False (e.g. it managed to reach the network, or it wrote to the input mount).
  It can never turn a False into a True, and a probe script that does not run at
  all no longer changes the verdict — the host already knows the answer.

The same host verification runs on every EVALUATION container too (not only on
the probe): a run whose isolation flags did not take effect produces tainted
outputs, so it raises ``SandboxIsolationError`` (an INFRA fault: our bug, not the
contender's) instead of being scored.

--------------------------------------------------------------------------------
BOUNDED OUTPUT
--------------------------------------------------------------------------------
``/output`` is a host directory, so an unbounded contender could fill the
validator's disk. It is bounded by a HOST-side watchdog: the ``docker run`` child
is polled while it executes and the on-disk size of the output directory (counted
with ``lstat``, links included) plus the captured container logs are compared
against ``max_output_bytes`` / ``max_batch_output_bytes``. Crossing the cap
force-removes the container and raises ``OversizeOutputError`` — a CONTENDER
fault: that contender is zero-scored, the competition continues. The caps are
re-checked after the run, per output and per batch.

FAST WRITERS: polling alone cannot bound a
process that finishes between two polls. A container that floods stdout and exits
in 20 ms used to slip past every check, because the loop broke out on ``poll()``
BEFORE measuring and the post-exit check only looked at /output. Two fixes, both
here: the watchdog now MEASURES ON THE SAME ITERATION IT OBSERVES THE EXIT (sizes
first, break second), and a FINAL check of both /output and the captured logs runs
after the loop. Either breach is the same CONTENDER fault as a mid-run breach —
the bound is on what was written, never on how long the writer stayed alive.

Why not a tmpfs at /output (the obvious alternative): a tmpfs mount lives and dies
with the container, so the results would have to be extracted before it exits —
which means asking the untrusted image to cooperate (a wrapper running the image's
own shell, or a sentinel-wait loop it can simply skip). A bound that depends on
the thing being bounded is not a bound. The watchdog keeps the durable bind mount
and puts the enforcement entirely on the host. MODAL PARITY: the deploy-time
runner gets the same guarantee natively from the sandbox's ephemeral-disk/volume
quota plus the same post-run per-output and per-batch caps (modal_runner.py).

Every failure raised from ``run_batch`` is typed by FAULT CLASS (errors.py): a
solution's own ``exit 1``, timeout, unsafe output or oversize output is a
``ContenderFaultError``; docker/image/input problems stay INFRA.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from vidaio.core.logging import get_logger, log_fields
from vidaio.competition.interfaces import (
    BatchItem,
    BatchOutput,
    ContenderSpec,
    IsolationProbeReport,
    logical_build_identity,
    upscale_task_sidecar_bytes,
    upscale_task_sidecar_name,
)
from vidaio.competition.runners import safeio
from vidaio.competition.runners.errors import (
    BatchExecutionError,
    BatchTimeout,
    BuildError,
    BuildTimeout,
    ContenderBuildError,
    InputStagingError,
    OutputRejectedError,
    OversizeOutputError,
    RunnerUnavailableError,
    SandboxIsolationError,
    SandboxProbeUnavailableError,
    SolutionExitError,
    UnknownImageError,
    UnsafePathError,
)
from vidaio.competition.runners.repo import (
    RepoProvider,
    checkout_pinned,
    release_checkout,
)

logger = get_logger("vidaio.competition.runners.docker")

_CHUNK = safeio.CHUNK

#: Label stamped on every container we create, so stale ones (orchestrator killed
#: between `docker run` and `docker rm`) can be reaped at construction.
_LABEL = "vidaio.sandbox=1"

#: Env var names docker/sh define on their own inside any container — allowed on
#: top of the image's self-declared env. Everything else is a leak.
_RUNTIME_ENV_ALLOWED = frozenset(
    {"PATH", "HOSTNAME", "HOME", "PWD", "OLDPWD", "SHLVL", "TERM", "container", "_"}
)

#: Env var NAMES that look credential-bearing. Matched against EVERY var in the
#: probe container — including the image's own: an image that ships (or fakes)
#: secret-shaped env can never be attested secret-free, so it fails the probe.
_SECRET_NAME_RE = re.compile(
    r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|PAT)(_|$)"
    r"|API_KEY|PRIVATE_KEY|ACCESS_KEY|SESSION_KEY|AUTH_TOKEN",
)

#: Mount destinations that must never exist in ANY sandbox container.
_FORBIDDEN_MOUNT_DESTS = (
    "/reference",
    "/references",
    "/holdout",
    "/evaluation-reference",
)

_PROBE_SCRIPT = """\
#!/bin/sh
# vidaio isolation probe (spec S05). No `set -e`: every check must report, never abort.
# ADVISORY-NEGATIVE ONLY: nothing printed here can make the probe pass — the host's
# `docker inspect` verdict is authoritative (see the module trust model).
net=2
if command -v wget >/dev/null 2>&1; then
  wget -q -T 3 -t 1 -O /dev/null http://example.com/ >/dev/null 2>&1 && net=1 || net=0
elif command -v python3 >/dev/null 2>&1; then
  python3 -c 'import socket;socket.create_connection(("1.1.1.1",80),3)' \
    >/dev/null 2>&1 && net=1 || net=0
elif command -v getent >/dev/null 2>&1; then
  getent hosts example.com >/dev/null 2>&1 && net=1 || net=0
elif command -v nc >/dev/null 2>&1; then
  nc -z -w 3 1.1.1.1 80 >/dev/null 2>&1 && net=1 || net=0
fi
echo "NETWORK_ATTEMPT=$net"
w=0; ( : > /evaluation-inputs/.probe-write ) 2>/dev/null && w=1
echo "INPUT_WRITE=$w"
r=0; ( : > /probe-rootfs-write ) 2>/dev/null && r=1
echo "ROOT_WRITE=$r"
m=0
for d in /reference /references /holdout /evaluation-reference; do
  [ -e "$d" ] && m=1
done
echo "REF_MOUNTS=$m"
i=0
for f in /evaluation-inputs/index.json /index.json /app/index.json; do
  [ -e "$f" ] && i=1
done
echo "INDEX_LEAK=$i"
echo "ENV_BEGIN"
env
echo "ENV_END"
echo "PROBE_DONE=1"
"""


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _read_tail(path: Path, limit: int = 2000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return _tail(f.read().decode("utf-8", "replace"), limit)
    except OSError:
        return ""


class HostIsolationFacts:
    """What the HOST observed about a container we launched (docker inspect).

    These are the only facts that may make an isolation probe pass; see the module
    trust model. ``ok`` is the conjunction of every boundary this class asserts.
    """

    def __init__(self, detail: dict[str, Any]) -> None:
        self.detail = detail

    @property
    def network_isolated(self) -> bool:
        return bool(self.detail["network_isolated"])

    @property
    def mounts_ok(self) -> bool:
        return bool(self.detail["mounts_ok"])

    @property
    def privileges_ok(self) -> bool:
        return bool(self.detail["privileges_ok"])

    @property
    def index_leak_absent(self) -> bool:
        return bool(self.detail["index_leak_absent"])

    @property
    def env_leaked(self) -> list[str]:
        return list(self.detail["env_leaked"])

    @property
    def env_secret_shaped(self) -> list[str]:
        return list(self.detail["env_secret_shaped"])

    @property
    def secrets_absent(self) -> bool:
        return not self.env_leaked and not self.env_secret_shaped

    @property
    def ok(self) -> bool:
        """The RUN-TIME contract: did OUR boundary hold on this container?

        Deliberately excludes ``env_secret_shaped``: an image shipping its own
        credential-shaped env is a PROBE disqualification (the build never gets to
        run), not evidence that the sandbox leaked something. What matters here is
        ``env_leaked`` — variables present beyond the image's own declared set,
        i.e. something WE injected.
        """
        return (
            self.network_isolated
            and self.mounts_ok
            and self.privileges_ok
            and self.index_leak_absent
            and not self.env_leaked
        )

    def violations(self) -> list[str]:
        out = []
        if not self.network_isolated:
            out.append("network_not_isolated")
        if not self.mounts_ok:
            out.append("mount_discipline_broken")
        if not self.privileges_ok:
            out.append("privilege_discipline_broken")
        if not self.index_leak_absent:
            out.append("index_leak")
        if self.env_leaked:
            out.append("env_injected")
        if self.env_secret_shaped:
            out.append("image_env_secret_shaped")
        return out


def host_isolation_facts(
    info: dict[str, Any],
    *,
    expected_mounts: dict[str, tuple[bool, str]],
    declared_env: set[str],
) -> HostIsolationFacts:
    """Derive the authoritative isolation verdict from ``docker inspect`` output.

    ``expected_mounts`` maps destination -> (read_write, host_source_path). A
    container whose mount set differs in ANY way (extra mount, missing mount,
    wrong RW flag, unexpected source) fails the mount discipline.
    """
    host_config = info.get("HostConfig") or {}
    net_settings = info.get("NetworkSettings") or {}
    config = info.get("Config") or {}

    network_mode = str(host_config.get("NetworkMode") or "")
    networks = sorted((net_settings.get("Networks") or {}).keys())
    network_isolated = network_mode == "none" and networks in ([], ["none"])

    mounts = info.get("Mounts") or []
    seen: dict[str, tuple[bool, str, str]] = {}
    tmpfs_dests: set[str] = set()
    for mount in mounts:
        dest = str(mount.get("Destination") or "")
        if str(mount.get("Type") or "") == "tmpfs":
            tmpfs_dests.add(dest)
            continue
        seen[dest] = (
            bool(mount.get("RW")),
            str(mount.get("Source") or ""),
            str(mount.get("Type") or ""),
        )
    tmpfs_dests |= {str(d) for d in (host_config.get("Tmpfs") or {})}

    mount_problems: list[str] = []
    if set(seen) != set(expected_mounts):
        mount_problems.append(
            f"mount set {sorted(seen)} != expected {sorted(expected_mounts)}"
        )
    for dest, (want_rw, want_source) in expected_mounts.items():
        if dest not in seen:
            continue
        got_rw, got_source, _kind = seen[dest]
        if got_rw != want_rw:
            mount_problems.append(f"{dest} RW={got_rw}, expected {want_rw}")
        if os.path.realpath(got_source) != os.path.realpath(want_source):
            mount_problems.append(f"{dest} source {got_source!r} != {want_source!r}")
    for dest in seen:
        if dest in _FORBIDDEN_MOUNT_DESTS:
            mount_problems.append(f"forbidden reference mount {dest}")
    if not tmpfs_dests <= {"/tmp"}:
        mount_problems.append(
            f"unexpected tmpfs mounts {sorted(tmpfs_dests - {'/tmp'})}"
        )
    readonly_rootfs = bool(host_config.get("ReadonlyRootfs"))
    if not readonly_rootfs:
        mount_problems.append("rootfs is writable (ReadonlyRootfs=false)")

    cap_drop = {str(c).upper() for c in (host_config.get("CapDrop") or [])}
    cap_add = [str(c) for c in (host_config.get("CapAdd") or [])]
    security_opt = [str(o) for o in (host_config.get("SecurityOpt") or [])]
    privilege_problems: list[str] = []
    if host_config.get("Privileged"):
        privilege_problems.append("container is privileged")
    if "ALL" not in cap_drop:
        privilege_problems.append(f"CapDrop={sorted(cap_drop)} does not drop ALL")
    if cap_add:
        privilege_problems.append(f"CapAdd={cap_add}")
    if not any(o.startswith("no-new-privileges") for o in security_opt):
        privilege_problems.append("no-new-privileges not set")

    env_names = [
        entry.split("=", 1)[0]
        for entry in (config.get("Env") or [])
        if "=" in str(entry)
    ]
    env_leaked = sorted(
        {
            n
            for n in env_names
            if n not in declared_env and n not in _RUNTIME_ENV_ALLOWED
        }
    )
    env_secret_shaped = sorted({n for n in env_names if _SECRET_NAME_RE.search(n)})

    index_leak_absent = not any(
        Path(source).name == "index.json" for _rw, source, _k in seen.values()
    )

    return HostIsolationFacts(
        {
            "network_mode": network_mode,
            "networks": networks,
            "network_isolated": network_isolated,
            "mounts": {
                d: {"rw": rw, "source": s} for d, (rw, s, _k) in sorted(seen.items())
            },
            "tmpfs": sorted(tmpfs_dests),
            "readonly_rootfs": readonly_rootfs,
            "mount_problems": mount_problems,
            "mounts_ok": not mount_problems,
            "cap_drop": sorted(cap_drop),
            "cap_add": cap_add,
            "security_opt": security_opt,
            "privilege_problems": privilege_problems,
            "privileges_ok": not privilege_problems,
            "env_leaked": env_leaked,
            "env_secret_shaped": env_secret_shaped,
            "index_leak_absent": index_leak_absent,
        }
    )


class DockerSandboxRunner:
    """interfaces.SandboxRunner over a local Docker daemon.

    All docker invocations are argv subprocesses (never shell strings) with bounded
    timeouts; stderr is captured into the typed errors of runners.errors. Docker
    being unavailable raises RunnerUnavailableError at CONSTRUCTION.

    ``inputs_dir`` is the sealed local input pool: one file per evaluation item,
    named by its sha256 hex digest (the orchestrator stages it). ``outputs_dir``
    is the collected output pool, content-addressed the same way — a BatchOutput's
    bytes are always at ``outputs_dir/<output_sha256>``.

    ``network_mode`` is a TEST/fault-injection knob only (defaults to the contract
    value "none"); production must never override it — the host-observed isolation
    verdict exists precisely to catch a runner whose network is not actually
    blocked, and it catches this knob too.
    """

    def __init__(
        self,
        repo_provider: RepoProvider,
        *,
        inputs_dir: str | Path,
        outputs_dir: str | Path,
        scratch_dir: str | Path,
        docker_path: str = "docker",
        build_timeout: float = 600.0,
        batch_timeout: float = 900.0,
        probe_timeout: float = 120.0,
        memory: str = "2g",
        cpus: float = 1.0,
        tmpfs_size: str = "256m",
        pids_limit: int = 256,
        network_mode: str = "none",
        max_output_bytes: int = 512 * 1024 * 1024,
        max_batch_output_bytes: int = 2 * 1024 * 1024 * 1024,
        max_log_bytes: int = 8 * 1024 * 1024,
        output_poll_seconds: float = 0.25,
    ) -> None:
        self._repos = repo_provider
        self._inputs_dir = Path(inputs_dir)
        self._outputs_dir = Path(outputs_dir)
        self._scratch_dir = Path(scratch_dir)
        self._docker = docker_path
        self._build_timeout = build_timeout
        self._batch_timeout = batch_timeout
        self._probe_timeout = probe_timeout
        self._memory = memory
        self._cpus = cpus
        self._tmpfs_size = tmpfs_size
        self._pids_limit = pids_limit
        self._network_mode = network_mode
        self._max_output_bytes = max_output_bytes
        self._max_batch_output_bytes = max_batch_output_bytes
        self._max_log_bytes = max_log_bytes
        self._output_poll_seconds = max(0.01, output_poll_seconds)
        for d in (self._inputs_dir, self._outputs_dir, self._scratch_dir):
            d.mkdir(parents=True, exist_ok=True)
        try:
            self._run_docker(
                ["version", "--format", "{{.Server.Version}}"],
                timeout=20.0,
                err_cls=BuildError,
            )
        except Exception as exc:
            raise RunnerUnavailableError(
                f"docker is not usable via {self._docker!r}: {exc} — the sandbox "
                "runner refuses to construct without a live container runtime"
            ) from exc
        self._reap_stale_containers()

    # ---- introspection ---------------------------------------------------------

    @property
    def inputs_dir(self) -> Path:
        return self._inputs_dir

    @property
    def outputs_dir(self) -> Path:
        return self._outputs_dir

    def available(self) -> bool:
        """Health-check hook: is the docker daemon still reachable?"""
        try:
            self._run_docker(
                ["version", "--format", "{{.Server.Version}}"],
                timeout=10.0,
                err_cls=BuildError,
            )
            return True
        except Exception:
            return False

    # ---- SandboxRunner: build --------------------------------------------------

    def build(self, contender: ContenderSpec) -> str:
        """Build the contender image from its local checkout; killable subprocess,
        bounded timeout. Returns the stable logical pinned-source identity.

        FAULT CLASSES: everything the CONTENDER controls —
        a missing/unsafe Dockerfile, a `docker build` that exits non-zero, a build
        that blows its bounded timeout — raises ``ContenderBuildError`` (that
        contender is BUILD_FAILED, the competition continues). Everything WE
        control — the docker CLI being unusable, `image inspect`/`tag` failing on
        our own daemon after a successful build — raises ``BuildError``, which is
        INFRA and takes the retry/halt path instead of blaming a submission.
        """
        checkout = checkout_pinned(
            self._repos,
            contender.repo_url,
            contender.commit_sha,
            contender.tree_sha,
        )
        try:
            return self._build_checkout(contender, checkout)
        finally:
            # docker build has returned (successfully or otherwise), so the CLI
            # has finished consuming its local context and the fresh clone can go.
            release_checkout(self._repos, checkout)

    def _build_checkout(self, contender: ContenderSpec, checkout: Path) -> str:
        dockerfile = checkout / "Dockerfile"
        try:
            safeio.lstat_regular(dockerfile, what="Dockerfile")
        except (FileNotFoundError, UnsafePathError) as exc:
            raise ContenderBuildError(
                f"contender {contender.contender_id}: checkout {checkout} has no usable "
                f"Dockerfile ({exc})"
            ) from exc
        build_tag = (
            f"vidaio-sbx-build-{contender.contender_id}-{contender.tree_sha[:12]}"
        )
        started = time.monotonic()
        self._run_docker(
            ["build", "--pull=false", "-t", build_tag, str(checkout)],
            timeout=self._build_timeout,
            err_cls=ContenderBuildError,  # the build said no to THIS submission
            timeout_cls=BuildTimeout,
            exec_err_cls=BuildError,  # ... but a missing docker CLI is ours
        )
        image_id = self._run_docker(
            ["image", "inspect", build_tag, "--format", "{{.Id}}"],
            timeout=30.0,
            err_cls=BuildError,  # our daemon lost an image it just built: INFRA
        ).strip()
        image_digest = logical_build_identity(
            repo_url=contender.repo_url,
            commit_sha=contender.commit_sha,
            tree_sha=contender.tree_sha,
        )
        # Digest-derived tag: resolvable after a process restart (resumable builds).
        self._run_docker(
            ["tag", image_id, self._digest_tag(image_digest)],
            timeout=30.0,
            err_cls=BuildError,
        )
        logger.info(
            "contender image built",
            extra=log_fields(
                contender_id=contender.contender_id,
                tree_sha=contender.tree_sha,
                image_id=image_id,
                image_digest=image_digest,
                build_seconds=round(time.monotonic() - started, 3),
            ),
        )
        return image_digest

    # ---- SandboxRunner: run_batch ----------------------------------------------

    def run_batch(
        self, image_digest: str, items: Sequence[BatchItem], batch_index: int
    ) -> Sequence[BatchOutput]:
        """Run one batch inside the isolated sandbox; collect outputs by digest.

        Inputs are staged (and digest-verified) into a per-batch subdir mounted RO;
        outputs the solution wrote to /output are collected SYMLINK-SAFELY (see
        safeio) and streamed into the content-addressed output pool. An item the
        solution produced no output for simply has no BatchOutput — zero-scoring
        absent outputs is the scorer's call, never substituted here.

        Raises (fault classes per errors.py):
        - SolutionExitError / BatchTimeout / OutputRejectedError /
          OversizeOutputError — CONTENDER faults; the orchestrator zero-scores this
          contender's items and the competition continues.
        - InputStagingError / UnknownImageError / BatchExecutionError /
          SandboxIsolationError — INFRA faults; requeue then halt.
        """
        image = self._resolve_image(image_digest)
        run_dir = self._scratch_dir / (
            f"run-{image_digest[:16]}-b{batch_index}-{uuid.uuid4().hex[:8]}"
        )
        in_dir = run_dir / "inputs"
        out_dir = run_dir / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        out_dir.chmod(0o777)  # image may run as any uid
        try:
            for item in items:
                self._stage_input(item, in_dir)
            name = f"vidaio-sbx-{uuid.uuid4().hex[:12]}"
            argv = [
                self._docker,
                "run",
                "--name",
                name,
                "--label",
                _LABEL,
                *self._isolation_flags(),
                "-v",
                f"{in_dir}:/evaluation-inputs:ro",
                "-v",
                f"{out_dir}:/output:rw",
                "--entrypoint",
                "/bin/sh",
                image,
                "/app/run.sh",
                "/evaluation-inputs",
                "/output",
            ]
            started = time.monotonic()
            try:
                returncode, _stdout_path, stderr_path = self._run_container_watched(
                    argv,
                    name,
                    run_dir=run_dir,
                    timeout=self._batch_timeout,
                    watch_dir=out_dir,
                    byte_cap=self._max_batch_output_bytes,
                    what=f"batch {batch_index} ({image_digest[:16]})",
                )
                elapsed = time.monotonic() - started
                # The isolation contract is verified on the container that ACTUALLY
                # ran this batch, from the host — not just on the probe image.
                self._assert_host_isolation(
                    name,
                    image,
                    expected_mounts={
                        "/evaluation-inputs": (False, str(in_dir)),
                        "/output": (True, str(out_dir)),
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
            outputs = self._collect_outputs(out_dir, items, elapsed)
            logger.info(
                "batch executed",
                extra=log_fields(
                    image_digest=image_digest,
                    batch_index=batch_index,
                    items=len(items),
                    outputs=len(outputs),
                    wall_seconds=round(elapsed, 3),
                ),
            )
            return outputs
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def _collect_outputs(
        self, out_dir: Path, items: Sequence[BatchItem], elapsed: float
    ) -> list[BatchOutput]:
        """Symlink-safe, size-bounded collection of the solution's outputs.

        Every expected entry is ``lstat``-ed (never ``stat``), required to be a
        plain regular file resolving inside ``out_dir``, opened with O_NOFOLLOW and
        streamed into the pool in a single hash-and-write pass. An entry that is
        anything else is REJECTED — the host never reads, hashes or archives it
.
        """
        outputs: list[BatchOutput] = []
        total = 0
        for item in items:
            produced = out_dir / item.input_sha256
            try:
                st = safeio.lstat_regular(produced, what="sandbox output")
            except FileNotFoundError:
                continue  # no output for this item: the scorer zero-scores it
            except UnsafePathError as exc:
                raise OutputRejectedError(f"item {item.item_id}: {exc}") from exc
            safeio.assert_within(produced, out_dir, what="sandbox output")
            if st.st_size > self._max_output_bytes:
                raise OversizeOutputError(
                    f"item {item.item_id}: output is {st.st_size} bytes, over the "
                    f"per-output cap of {self._max_output_bytes}"
                )
            total += st.st_size
            if total > self._max_batch_output_bytes:
                raise OversizeOutputError(
                    f"batch outputs total {total} bytes, over the per-batch cap of "
                    f"{self._max_batch_output_bytes}"
                )
            try:
                digest, size = safeio.hash_into_pool(
                    produced,
                    st,
                    self._outputs_dir,
                    max_bytes=self._max_output_bytes,
                    what="sandbox output",
                )
            except UnsafePathError as exc:
                raise OutputRejectedError(f"item {item.item_id}: {exc}") from exc
            outputs.append(
                BatchOutput(
                    item_id=item.item_id,
                    output_sha256=digest,
                    output_bytes=size,
                    wall_seconds=elapsed,  # batch-level wall clock (documented)
                )
            )
        return outputs

    # ---- SandboxRunner: isolation_probe ----------------------------------------

    def isolation_probe(self, image_digest: str) -> IsolationProbeReport:
        """Attest the isolation boundary of a container built from this image.

        The verdict comes from the HOST (``docker inspect`` of the probe container
        we launched); the in-container script is a secondary, advisory-negative
        signal only. See the module-level trust model.

        Fields (spec §05):
        - network_blocked: host says NetworkMode=none with no network attached.
          Downgraded to False if the container nonetheless reached the network.
        - secrets_absent: the container's effective ``Config.Env`` holds nothing
          beyond the image's own declared env plus the docker/sh defaults, and NO
          var (image-declared included) has a credential-shaped name.
        - reference_mounts_absent: host MOUNT AND PRIVILEGE discipline — exactly
          the expected mounts with the expected RW flags and sources, read-only
          rootfs, tmpfs only at /tmp, not privileged, all caps dropped,
          no-new-privileges. Downgraded if the container wrote to the input mount
          or the rootfs, or saw a reference mount.
        - index_leak_absent: no index.json reachable through any mount; downgraded
          if the container found one.

        A probe that RAN and observed a violated boundary returns a failing report
        and disqualifies that contender. A probe that could NOT RUN raises
        ``SandboxProbeUnavailableError`` instead (review service-review #14, round
        2): image unresolvable, `docker run` refused, host inspection failed. Those
        are OUR outages, and the all-False report they used to return was
        indistinguishable from an attested escape — it disqualified innocent
        contenders on our own infrastructure failures. The orchestrator classifies
        the raise as INFRA and retries/halts."""
        try:
            image = self._resolve_image(image_digest)
        except BatchExecutionError as exc:
            raise SandboxProbeUnavailableError(
                f"isolation probe could not resolve image {image_digest}: {exc}"
            ) from exc
        try:
            declared_env = self._image_declared_env(image)
        except BatchExecutionError as exc:
            raise SandboxProbeUnavailableError(
                f"isolation probe could not inspect the env of image {image_digest}: {exc}"
            ) from exc
        probe_dir = self._scratch_dir / f"probe-{uuid.uuid4().hex[:8]}"
        empty_inputs = probe_dir / "inputs"
        script_dir = probe_dir / "script"
        empty_inputs.mkdir(parents=True)
        script_dir.mkdir(parents=True)
        (script_dir / "probe.sh").write_text(_PROBE_SCRIPT)
        name = f"vidaio-probe-{uuid.uuid4().hex[:12]}"
        argv = [
            self._docker,
            "run",
            "--name",
            name,
            "--label",
            _LABEL,
            *self._isolation_flags(),
            "-v",
            f"{empty_inputs}:/evaluation-inputs:ro",
            "-v",
            f"{script_dir}:/vidaio-probe:ro",
            "--entrypoint",
            "/bin/sh",
            image,
            "/vidaio-probe/probe.sh",
        ]
        container_stdout = ""
        try:
            returncode, stdout_path, _stderr_path = self._run_container_watched(
                argv,
                name,
                run_dir=probe_dir,
                timeout=self._probe_timeout,
                watch_dir=empty_inputs,
                byte_cap=self._max_output_bytes,
                what="isolation probe",
            )
            container_stdout = _read_tail(stdout_path, limit=64 * 1024)
            container_note = (
                "" if returncode == 0 else f"probe script exited {returncode}"
            )
        except (BatchTimeout, OversizeOutputError) as exc:
            # The script misbehaved; the host verdict below still stands on its own.
            container_note = f"probe script aborted: {exc}"
        except BatchExecutionError as exc:
            self._force_remove(name)
            shutil.rmtree(probe_dir, ignore_errors=True)
            raise SandboxProbeUnavailableError(
                f"isolation probe container could not be launched: {exc}"
            ) from exc
        try:
            info = self._inspect_container(name)
        except BatchExecutionError as exc:
            self._force_remove(name)
            shutil.rmtree(probe_dir, ignore_errors=True)
            # No host facts means NOTHING was attested. Refusing to produce a
            # verdict is the honest outcome; an all-False report here would read
            # exactly like a contender that escaped the sandbox.
            raise SandboxProbeUnavailableError(
                f"host inspection of the probe container failed: {exc}"
            ) from exc
        finally:
            self._force_remove(name)
        facts = host_isolation_facts(
            info,
            expected_mounts={
                "/evaluation-inputs": (False, str(empty_inputs)),
                "/vidaio-probe": (False, str(script_dir)),
            },
            declared_env=declared_env,
        )
        shutil.rmtree(probe_dir, ignore_errors=True)
        return self._compose_report(facts, container_stdout, container_note)

    def _compose_report(
        self, facts: HostIsolationFacts, container_stdout: str, container_note: str
    ) -> IsolationProbeReport:
        """Host facts decide PASS; container-reported facts may only FAIL."""
        advisory = _parse_probe_output(container_stdout)
        advisory_env = advisory.get("env_names", [])
        host_flagged = set(facts.env_leaked) | set(facts.env_secret_shaped)
        # Advisory-negative only: an in-container env var with a credential-shaped
        # name that the host did not already see can still FAIL the probe.
        extra_leaked = sorted(
            {n for n in advisory_env if _SECRET_NAME_RE.search(n)} - host_flagged
        )
        network_blocked = (
            facts.network_isolated and advisory.get("NETWORK_ATTEMPT") != "1"
        )
        secrets_absent = facts.secrets_absent and not extra_leaked
        reference_mounts_absent = (
            facts.mounts_ok
            and facts.privileges_ok
            and advisory.get("REF_MOUNTS") != "1"
            and advisory.get("INPUT_WRITE") != "1"
            and advisory.get("ROOT_WRITE") != "1"
        )
        index_leak_absent = (
            facts.index_leak_absent and advisory.get("INDEX_LEAK") != "1"
        )
        details = json.dumps(
            {
                "trust": "host-observed facts are authoritative; container-reported "
                "facts are advisory-negative-only",
                "host": facts.detail,
                "container": {
                    "completed": advisory.get("PROBE_DONE") == "1",
                    "note": container_note,
                    "network_attempt": advisory.get("NETWORK_ATTEMPT"),
                    "input_write": advisory.get("INPUT_WRITE"),
                    "root_write": advisory.get("ROOT_WRITE"),
                    "ref_mounts": advisory.get("REF_MOUNTS"),
                    "index_leak": advisory.get("INDEX_LEAK"),
                    "env_secret_shaped_extra": extra_leaked,
                },
            },
            sort_keys=True,
            default=str,
        )
        report = IsolationProbeReport(
            network_blocked=network_blocked,
            secrets_absent=secrets_absent,
            reference_mounts_absent=reference_mounts_absent,
            index_leak_absent=index_leak_absent,
            details=details,
        )
        logger.info(
            "isolation probe finished",
            extra=log_fields(
                passed=report.passed,
                host_violations=facts.violations(),
                probe=details,
            ),
        )
        return report

    # ---- internals -------------------------------------------------------------

    def _isolation_flags(self) -> list[str]:
        return [
            f"--network={self._network_mode}",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            f"--pids-limit={self._pids_limit}",
            f"--memory={self._memory}",
            f"--cpus={self._cpus}",
            "--tmpfs",
            f"/tmp:rw,size={self._tmpfs_size}",
        ]

    def _digest_tag(self, image_digest: str) -> str:
        return f"vidaio-sbx:{image_digest[:32]}"

    def _resolve_image(self, image_digest: str) -> str:
        tag = self._digest_tag(image_digest)
        try:
            self._run_docker(
                ["image", "inspect", tag, "--format", "{{.Id}}"],
                timeout=30.0,
                err_cls=UnknownImageError,
            )
        except UnknownImageError as exc:
            raise UnknownImageError(
                f"no local image for image_digest {image_digest} (tag {tag}); "
                f"build it first — {exc}"
            ) from exc
        return tag

    def _image_declared_env(self, image: str) -> set[str]:
        raw = self._run_docker(
            ["image", "inspect", image, "--format", "{{json .Config.Env}}"],
            timeout=30.0,
            err_cls=BatchExecutionError,
        ).strip()
        entries = json.loads(raw) or []
        return {entry.split("=", 1)[0] for entry in entries}

    def _inspect_container(self, name: str) -> dict[str, Any]:
        raw = self._run_docker(
            ["container", "inspect", name, "--format", "{{json .}}"],
            timeout=30.0,
            err_cls=BatchExecutionError,
        ).strip()
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BatchExecutionError(
                f"docker inspect returned non-JSON: {exc}"
            ) from exc
        if not isinstance(info, dict):
            raise BatchExecutionError("docker inspect returned an unexpected shape")
        return info

    def _assert_host_isolation(
        self,
        name: str,
        image: str,
        *,
        expected_mounts: dict[str, tuple[bool, str]],
        what: str,
    ) -> None:
        """Verify the isolation contract HELD on the container that just ran.

        A violation is an INFRA fault (our flags did not take effect), never a
        contender fault: the outputs are tainted and must not be scored.
        """
        info = self._inspect_container(name)
        declared_env = self._image_declared_env(image)
        facts = host_isolation_facts(
            info, expected_mounts=expected_mounts, declared_env=declared_env
        )
        if facts.ok:
            return
        raise SandboxIsolationError(
            f"{what}: the sandbox isolation contract did not hold on container "
            f"{name} — {facts.violations()}; host facts: "
            f"{json.dumps(facts.detail, sort_keys=True, default=str)}"
        )

    def _run_container_watched(
        self,
        argv: list[str],
        name: str,
        *,
        run_dir: Path,
        timeout: float,
        watch_dir: Path,
        byte_cap: int,
        what: str,
    ) -> tuple[int, Path, Path]:
        """Run ``docker run`` as a polled child process with a HOST-side byte cap.

        Container stdout/stderr go to files under our own scratch dir (never a
        pipe: a pipe would deadlock while we poll, and an unbounded pipe is another
        way to exhaust the host). Returns (returncode, stdout_path, stderr_path).
        The container is force-removed before either bounded-failure error is
        raised, so nothing keeps running past its budget.
        """
        stdout_path = run_dir / "container.stdout"
        stderr_path = run_dir / "container.stderr"
        deadline = time.monotonic() + timeout
        with open(stdout_path, "wb") as out_fh, open(stderr_path, "wb") as err_fh:
            try:
                proc = subprocess.Popen(argv, stdout=out_fh, stderr=err_fh)
            except OSError as exc:
                raise BatchExecutionError(
                    f"cannot execute {self._docker!r}: {exc}"
                ) from exc
            while True:
                # MEASURE FIRST, then decide whether the child is done (an internal review,
                # round 2): breaking on poll() before measuring let a fast writer
                # blow either cap and exit unobserved between two polls.
                returncode = proc.poll()
                used = safeio.tree_bytes(watch_dir)
                logs = _file_size(stdout_path) + _file_size(stderr_path)
                if used > byte_cap:
                    self._force_remove(name)
                    _terminate(proc)
                    raise OversizeOutputError(
                        f"{what} wrote {used} bytes to /output, over the per-batch cap "
                        f"of {byte_cap} — the container was killed mid-run"
                    )
                if logs > self._max_log_bytes:
                    self._force_remove(name)
                    _terminate(proc)
                    raise OversizeOutputError(
                        f"{what} produced {logs} bytes of container logs, over the cap "
                        f"of {self._max_log_bytes} — the container was killed mid-run"
                    )
                if returncode is not None:
                    break
                if time.monotonic() > deadline:
                    self._force_remove(name)
                    _terminate(proc)
                    raise BatchTimeout(f"{what} exceeded {timeout}s and was killed")
                time.sleep(self._output_poll_seconds)
        # FINAL check, after the child is reaped and its fds are closed: a process
        # that wrote everything and exited within a single poll interval is bounded
        # by exactly the same caps as one that lingered. Logs included — the
        # post-exit check used to cover /output only.
        used = safeio.tree_bytes(watch_dir)
        if used > byte_cap:
            raise OversizeOutputError(
                f"{what} left {used} bytes in /output, over the per-batch cap of {byte_cap}"
            )
        logs = _file_size(stdout_path) + _file_size(stderr_path)
        if logs > self._max_log_bytes:
            raise OversizeOutputError(
                f"{what} left {logs} bytes of container logs, over the cap of "
                f"{self._max_log_bytes}"
            )
        return returncode, stdout_path, stderr_path

    def _stage_input(self, item: BatchItem, in_dir: Path) -> None:
        src = self._inputs_dir / item.input_sha256
        if not src.is_file():
            raise InputStagingError(
                f"sealed input {item.input_sha256} (item {item.item_id}) missing "
                f"from input pool {self._inputs_dir}"
            )
        digest, size = _sha256_file(src)
        if digest != item.input_sha256 or size != item.input_bytes:
            raise InputStagingError(
                f"sealed input for item {item.item_id} failed verification: "
                f"expected {item.input_sha256}/{item.input_bytes}B, got {digest}/{size}B"
            )
        shutil.copy2(src, in_dir / item.input_sha256)
        if item.upscale_factor is not None:
            try:
                if item.target_width is None or item.target_height is None:
                    raise ValueError("target geometry is missing")
                sidecar = in_dir / upscale_task_sidecar_name(item.input_sha256)
                payload = upscale_task_sidecar_bytes(
                    item.upscale_factor,
                    item.target_width,
                    item.target_height,
                )
                fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                with os.fdopen(fd, "wb") as handle:
                    os.fchmod(handle.fileno(), 0o444)
                    handle.write(payload)
            except (OSError, ValueError) as exc:
                raise InputStagingError(
                    f"item {item.item_id}: could not stage committed upscale task: "
                    f"{exc}"
                ) from exc

    def _reap_stale_containers(self) -> None:
        """Best-effort cleanup of our own containers left by a killed orchestrator.

        Containers are created WITHOUT ``--rm`` (their host-observed isolation
        facts must survive the run long enough to be inspected), so a hard crash
        between `docker run` and `docker rm` can leak one; the label makes them
        unambiguously ours.

        Only NON-RUNNING containers are reaped: another orchestrator (or another
        test) may be mid-batch on the same daemon, and reaping must never kill a
        live sandbox out from under it.
        """
        try:
            ids = self._run_docker(
                [
                    "ps",
                    "-aq",
                    "--filter",
                    f"label={_LABEL}",
                    "--filter",
                    "status=exited",
                    "--filter",
                    "status=created",
                    "--filter",
                    "status=dead",
                ],
                timeout=20.0,
                err_cls=BuildError,
            ).split()
        except Exception:
            return
        for container_id in ids:
            self._force_remove(container_id)
        if ids:
            logger.info(
                "reaped stale sandbox containers", extra=log_fields(count=len(ids))
            )

    def _force_remove(self, name: str) -> None:
        """Best-effort cleanup of a (possibly still running) container."""
        try:
            subprocess.run(
                [self._docker, "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=30.0,
            )
        except Exception:  # cleanup must never mask the primary error
            pass

    def _run_docker(
        self,
        args: list[str],
        *,
        timeout: float,
        err_cls: type[Exception],
        timeout_cls: type[Exception] | None = None,
        exec_err_cls: type[Exception] | None = None,
    ) -> str:
        """Run one docker CLI command.

        ``err_cls`` types a NON-ZERO EXIT (the command ran and said no).
        ``exec_err_cls`` types "we could not even execute the CLI", which is a
        different fault class whenever err_cls is contender-attributable: the
        contender's Dockerfile is not to blame for our docker binary being gone
. Defaults to err_cls.
        """
        argv = [self._docker, *args]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise (timeout_cls or err_cls)(
                f"docker {args[0]} timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise (exec_err_cls or err_cls)(
                f"cannot execute {self._docker!r}: {exc}"
            ) from exc
        if proc.returncode != 0:
            raise err_cls(
                f"docker {args[0]} failed (exit {proc.returncode}): {_tail(proc.stderr)}"
            )
        return proc.stdout


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _file_size(path: Path) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _parse_probe_output(stdout: str) -> dict[str, Any]:
    """Parse the in-container probe markers. ADVISORY ONLY — see the trust model."""
    values: dict[str, Any] = {}
    env_names: list[str] = []
    in_env = False
    for line in stdout.splitlines():
        if line == "ENV_BEGIN":
            in_env = True
            continue
        if line == "ENV_END":
            in_env = False
            continue
        if in_env:
            if "=" in line:
                env_names.append(line.split("=", 1)[0])
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    values["env_names"] = env_names
    return values
