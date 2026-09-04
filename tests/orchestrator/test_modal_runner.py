"""Create-only Modal runner contract tests (all provider calls are mocked)."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import pytest

from vidaio.competition.interfaces import (
    BatchItem,
    ContenderSpec,
    logical_build_identity,
    upscale_task_sidecar_name,
)
from vidaio.competition.runners.errors import (
    BuildError,
    BuildTimeout,
    InputStagingError,
    OutputRejectedError,
    OversizeOutputError,
    RunnerUnavailableError,
    SandboxIsolationError,
    UnknownImageError,
)
from vidaio.competition.runners.modal_runner import (
    FRESH_CREATION_CONFIRMATION,
    ModalRunnerConfig,
    ModalSandboxRunner,
    ModalSdkRuntime,
    RemoteFile,
    SandboxRequestAttestation,
)
from vidaio.competition.runners.repo import LocalRepoProvider


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu", float("nan")),
        ("cpu", float("inf")),
        ("build_timeout_seconds", float("nan")),
        ("batch_timeout_seconds", float("inf")),
        ("probe_timeout_seconds", float("nan")),
        ("output_poll_seconds", float("inf")),
    ],
)
def test_modal_runner_config_rejects_nonfinite_provider_limits(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        ModalRunnerConfig(**{field: value})


@dataclass
class _Image:
    files: dict[str, tuple[str, bytes]]


class _Process:
    def __init__(self, stdout: bytes, *, returncode: int = 0) -> None:
        self.stdout = [stdout]
        self.stderr: list[bytes] = []
        self._returncode = returncode

    def poll(self) -> int:
        return self._returncode


class _Lease:
    def __init__(
        self,
        runtime: _Runtime,
        object_id: str,
        request: SandboxRequestAttestation,
    ) -> None:
        self.runtime = runtime
        self.object_id = object_id
        self.request = request
        self.mounts: list[str] = []
        self.files: dict[str, tuple[str, bytes]] = {}
        self.directories: set[str] = {"/", "/tmp"}
        self.terminated = 0
        self.detached = 0

    def attestation(self) -> SandboxRequestAttestation:
        return SandboxRequestAttestation(
            **{**self.request.__dict__, "image_mounts": tuple(self.mounts)}
        )

    def mount_image(self, path: str, image: object) -> None:
        assert isinstance(image, _Image)
        self.mounts.append(path)
        self.directories.add(path)
        for relative, value in image.files.items():
            self.files[f"{path}/{relative.lstrip('/')}"] = value

    def make_directory(self, path: str) -> None:
        self.directories.add(path)

    def write_text(self, path: str, value: str) -> None:
        self.files[path] = ("file", value.encode())

    def exec(self, args: Sequence[str], *, timeout_seconds: float) -> _Process:
        assert timeout_seconds > 0
        if "isolation-probe" in " ".join(args):
            return _Process(
                (
                    "NETWORK_ATTEMPT=0\n"
                    f"INPUT_WRITE={self.runtime.probe_input_write}\n"
                    "REF_MOUNTS=0\nINDEX_LEAK=0\nENV_BEGIN\nPATH=/usr/bin\n"
                    "ENV_END\nPROBE_DONE=1\n"
                ).encode()
            )
        if "input-overlay-check" in " ".join(args):
            return _Process(
                (
                    f"INPUT_BASE_MUTATED={self.runtime.probe_overlay_mutated}\n"
                    "OVERLAY_CHECK_DONE=1\n"
                ).encode()
            )
        assert list(args) == [
            "/bin/sh",
            "/app/run.sh",
            "/evaluation-inputs",
            "/output",
        ]
        for path, (kind, data) in list(self.files.items()):
            if PurePosixPath(path).parent != PurePosixPath("/evaluation-inputs"):
                continue
            if PurePosixPath(path).name.startswith("."):
                continue
            output_path = f"/output/{PurePosixPath(path).name}"
            if self.runtime.output_kind == "symlink":
                self.files[output_path] = ("symlink", b"/etc/passwd")
            else:
                self.files[output_path] = (
                    "file",
                    data[: self.runtime.output_bytes],
                )
        return _Process(self.runtime.solution_log, returncode=self.runtime.returncode)

    def list_files(self, path: str) -> Sequence[RemoteFile]:
        root = PurePosixPath(path)
        entries: list[RemoteFile] = []
        for remote, (kind, data) in sorted(self.files.items()):
            remote_path = PurePosixPath(remote)
            if remote_path.parent == root:
                entries.append(RemoteFile(remote, kind, len(data)))
        for directory in sorted(self.directories):
            remote_path = PurePosixPath(directory)
            if remote_path != root and remote_path.parent == root:
                entries.append(RemoteFile(directory, "directory", 0))
        return entries

    def stat(self, path: str) -> RemoteFile:
        value = self.files.get(path)
        if value is None:
            raise FileNotFoundError(path)
        kind, data = value
        return RemoteFile(path, kind, len(data))

    def copy_to_local(self, remote_path: str, local_path: Path) -> None:
        kind, data = self.files[remote_path]
        assert kind == "file"
        local_path.write_bytes(data)

    def snapshot_directory(self, path: str, *, ttl_seconds: int) -> object:
        assert 0 < ttl_seconds <= 3600
        root = PurePosixPath(path)
        return _Image(
            {
                str(PurePosixPath(remote).relative_to(root)): value
                for remote, value in self.files.items()
                if root in PurePosixPath(remote).parents
            }
        )

    def terminate(self) -> None:
        self.terminated += 1

    def detach(self) -> None:
        self.detached += 1


class _Runtime:
    run_label = "vidaio-next-test-run-abcdef12"

    def __init__(self) -> None:
        self.closed = False
        self.leases: list[_Lease] = []
        self.create_calls: list[dict[str, object]] = []
        self.input_images: list[_Image] = []
        self.output_bytes = 8
        self.output_kind = "file"
        self.solution_log = b"solution complete\n"
        self.returncode = 0
        self.bad_network_attestation = False
        self.restored_image_ids: list[str] = []
        self.probe_input_write = "1"
        self.probe_overlay_mutated = "0"

    def available(self) -> bool:
        return not self.closed

    def build_contender_image(
        self, checkout: Path, dockerfile: Path
    ) -> tuple[object, str]:
        assert dockerfile == checkout / "Dockerfile"
        return object(), "im-fresh-0001"

    def restore_contender_image(self, image_object_id: str) -> tuple[object, str]:
        self.restored_image_ids.append(image_object_id)
        return object(), image_object_id

    def build_input_image(self, staged_inputs: Path) -> object:
        files = {
            path.name: ("file", path.read_bytes())
            for path in sorted(staged_inputs.iterdir())
        }
        image = _Image(files)
        self.input_images.append(image)
        return image

    def create_sandbox(self, **kwargs: object) -> _Lease:
        self.create_calls.append(dict(kwargs))
        request = SandboxRequestAttestation(
            name=str(kwargs["name"]),
            role=str(kwargs["role"]),
            block_network=not self.bad_network_attestation,
            secret_count=0,
            env_keys=(),
            include_oidc_identity_token=False,
            volume_mounts=(),
            network_filesystem_mounts=(),
            ports=(),
            gpu=kwargs["gpu"] if isinstance(kwargs["gpu"], str) else None,
        )
        lease = _Lease(self, f"sb-fresh-{len(self.leases) + 1}", request)
        self.leases.append(lease)
        return lease

    def create_collector_sandbox(self, **kwargs: object) -> _Lease:
        expanded = dict(kwargs)
        expanded.update(
            image=object(),
            role="collector",
            gpu=None,
            cpu=0.25,
            memory_mb=512,
            idle_timeout_seconds=60,
        )
        return self.create_sandbox(**expanded)

    def close(self) -> None:
        self.closed = True


def _make_runner(tmp_path: Path, runtime: _Runtime, **cfg: object):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "Dockerfile").write_text("FROM scratch\n")
    work = tmp_path / "work"
    runner = ModalSandboxRunner(
        LocalRepoProvider({"local://contender": checkout}),
        runtime,
        inputs_dir=work / "inputs",
        outputs_dir=work / "outputs",
        scratch_dir=work / "scratch",
        config=ModalRunnerConfig(**cfg),
    )
    spec = ContenderSpec(
        contender_id=1,
        repo_url="local://contender",
        commit_sha="a" * 40,
        tree_sha="b" * 40,
    )
    return runner, spec


def _stage(
    runner: ModalSandboxRunner,
    data: bytes,
    *,
    item_id: int = 1,
    upscale_factor: int | None = None,
) -> BatchItem:
    digest = hashlib.sha256(data).hexdigest()
    (runner.inputs_dir / digest).write_bytes(data)
    return BatchItem(
        item_id=item_id,
        item_index=item_id - 1,
        input_sha256=digest,
        input_bytes=len(data),
        upscale_factor=upscale_factor,
        target_width=None if upscale_factor is None else 1920,
        target_height=None if upscale_factor is None else 1080,
    )


def test_build_probe_run_rolls_fresh_sandboxes_and_collects_bytes(tmp_path: Path):
    runtime = _Runtime()
    runner, spec = _make_runner(tmp_path, runtime, output_poll_seconds=0.001)
    assert runner.gpu == "L4"
    image_digest = runner.build(spec)
    assert image_digest == logical_build_identity(
        repo_url=spec.repo_url,
        commit_sha=spec.commit_sha,
        tree_sha=spec.tree_sha,
    )
    assert runner.isolation_probe(image_digest).passed

    item = _stage(runner, b"0123456789abcdef")
    first = runner.run_batch(image_digest, [item], 0)
    second = runner.run_batch(image_digest, [item], 1)
    assert len(first) == len(second) == 1
    assert first[0].output_bytes == 8
    assert (runner.outputs_dir / first[0].output_sha256).read_bytes() == b"01234567"

    roles = [str(call["role"]) for call in runtime.create_calls]
    assert roles == [
        "probe",
        "probe-overlay-check",
        "contender",
        "collector",
        "contender",
        "collector",
    ]
    contender_ids = [
        lease.object_id for lease in runtime.leases if lease.request.role == "contender"
    ]
    assert len(set(contender_ids)) == 2  # forced per-batch rollover, never warm reuse
    assert all(call["name"].startswith("vidaio-next-") for call in runtime.create_calls)
    assert all(
        lease.terminated >= 1 and lease.detached >= 1 for lease in runtime.leases
    )
    assert all(
        call["gpu"] == "L4"
        for call in runtime.create_calls
        if call["role"] in {"probe", "contender"}
    )
    assert all(
        call["gpu"] is None
        for call in runtime.create_calls
        if call["role"] in {"probe-overlay-check", "collector"}
    )
    # Probe got an empty image; each batch got only its digest-named sealed file.
    assert runtime.input_images[0].files == {}
    assert all(
        set(image.files) == {item.input_sha256} for image in runtime.input_images[1:]
    )


def test_repeated_spec_reuses_exact_image_from_same_fresh_modal_run(
    tmp_path: Path,
) -> None:
    class ChangingIdRuntime(_Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.builds = 0

        def build_contender_image(
            self, checkout: Path, dockerfile: Path
        ) -> tuple[object, str]:
            self.builds += 1
            return object(), f"im-fresh-changing-{self.builds}"

    runtime = ChangingIdRuntime()
    runner, spec = _make_runner(tmp_path, runtime)

    anchored_digest = runner.build(spec)
    lifecycle_digest = runner.build(
        spec.__class__(
            contender_id=99,
            repo_url=spec.repo_url,
            commit_sha=spec.commit_sha,
            tree_sha=spec.tree_sha,
        )
    )

    assert lifecycle_digest == anchored_digest
    assert runtime.builds == 1


@pytest.mark.parametrize("image_id", ["", "im-", "not-an-image", "im-bad.id", None])
def test_build_rejects_malformed_provider_image_object_id(
    tmp_path: Path, image_id: object
) -> None:
    class MalformedIdRuntime(_Runtime):
        def build_contender_image(
            self, checkout: Path, dockerfile: Path
        ) -> tuple[object, object]:
            assert dockerfile == checkout / "Dockerfile"
            return object(), image_id

    runner, spec = _make_runner(tmp_path, MalformedIdRuntime())

    with pytest.raises(BuildError, match="malformed image object id"):
        runner.build(spec)

    assert not runner.has_live_image("f" * 64)


def test_runtime_session_fence_is_per_runner_even_when_label_is_reused(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    runner_a, spec_a = _make_runner(first_root, _Runtime())
    runner_b, _spec_b = _make_runner(second_root, _Runtime())

    assert runner_a.runtime_label == runner_b.runtime_label
    assert runner_a.runtime_session_id != runner_b.runtime_session_id
    assert len(runner_a.runtime_session_id) == 64
    assert not runner_a.has_live_image("f" * 64)
    digest = runner_a.build(spec_a)
    assert runner_a.has_live_image(digest)
    assert not runner_b.has_live_image(digest)


def test_exact_competition_owned_image_can_be_rehydrated_without_sandbox_reuse(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-owned"
    second_root = tmp_path / "second-owned"
    first_root.mkdir()
    second_root.mkdir()
    runtime_a = _Runtime()
    runner_a, spec = _make_runner(first_root, runtime_a)
    digest = runner_a.build(spec)
    object_id = runner_a.image_object_id(digest)
    assert object_id == "im-fresh-0001"

    runtime_b = _Runtime()
    runner_b, spec_b = _make_runner(second_root, runtime_b)
    assert runner_b.restore_image(spec_b, digest, object_id) == digest
    assert runner_b.has_live_image(digest)
    assert runner_b.image_object_id(digest) == object_id
    assert runtime_b.restored_image_ids == [object_id]
    assert runtime_b.create_calls == []  # no Sandbox/instance was attached


def test_fresh_builds_share_logical_identity_but_keep_distinct_provider_ids(
    tmp_path: Path,
) -> None:
    class SecondFreshRuntime(_Runtime):
        def build_contender_image(
            self, checkout: Path, dockerfile: Path
        ) -> tuple[object, str]:
            assert dockerfile == checkout / "Dockerfile"
            return object(), "im-fresh-0042"

    first_root = tmp_path / "first-fresh-build"
    second_root = tmp_path / "second-fresh-build"
    first_root.mkdir()
    second_root.mkdir()
    runtime_a = _Runtime()
    runtime_b = SecondFreshRuntime()
    runner_a, spec_a = _make_runner(first_root, runtime_a)
    runner_b, spec_b = _make_runner(second_root, runtime_b)

    digest_a = runner_a.build(spec_a)
    digest_b = runner_b.build(spec_b)

    assert (
        digest_a
        == digest_b
        == logical_build_identity(
            repo_url=spec_a.repo_url,
            commit_sha=spec_a.commit_sha,
            tree_sha=spec_a.tree_sha,
        )
    )
    assert runner_a.image_object_id(digest_a) == "im-fresh-0001"
    assert runner_b.image_object_id(digest_b) == "im-fresh-0042"


def test_restore_rejects_logical_identity_for_a_different_pinned_source(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "source-a"
    second_root = tmp_path / "source-b"
    first_root.mkdir()
    second_root.mkdir()
    runner_a, spec = _make_runner(first_root, _Runtime())
    digest = runner_a.build(spec)
    object_id = runner_a.image_object_id(digest)
    assert object_id is not None
    runner_b, _ = _make_runner(second_root, _Runtime())
    different = ContenderSpec(
        contender_id=spec.contender_id,
        repo_url=spec.repo_url,
        commit_sha="c" * 40,
        tree_sha=spec.tree_sha,
    )

    with pytest.raises(UnknownImageError, match="does not match the pinned source"):
        runner_b.restore_image(different, digest, object_id)


def test_mixed_upscale_factors_are_digest_bound_without_reference_material(
    tmp_path: Path,
) -> None:
    runtime = _Runtime()
    runner, spec = _make_runner(tmp_path, runtime, output_poll_seconds=0.001)
    image_digest = runner.build(spec)
    assert runner.isolation_probe(image_digest).passed
    two_x = _stage(runner, b"low-resolution-two", item_id=1, upscale_factor=2)
    four_x = _stage(runner, b"low-resolution-four", item_id=2, upscale_factor=4)

    outputs = runner.run_batch(image_digest, [two_x, four_x], 0)

    assert {output.item_id for output in outputs} == {1, 2}
    image = runtime.input_images[-1]
    sidecar_two = upscale_task_sidecar_name(two_x.input_sha256)
    sidecar_four = upscale_task_sidecar_name(four_x.input_sha256)
    assert set(image.files) == {
        two_x.input_sha256,
        four_x.input_sha256,
        sidecar_two,
        sidecar_four,
    }
    assert image.files[sidecar_two] == (
        "file",
        b'{"target_height":1080,"target_width":1920,"upscale_factor":2}\n',
    )
    assert image.files[sidecar_four] == (
        "file",
        b'{"target_height":1080,"target_width":1920,"upscale_factor":4}\n',
    )
    assert all("reference" not in name for name in image.files)
    private_reference_digest = "f" * 64
    assert private_reference_digest not in image.files
    assert all(
        private_reference_digest.encode("ascii") not in payload
        for _kind, payload in image.files.values()
    )


def test_invalid_upscale_factor_is_rejected_before_a_gpu_batch(tmp_path: Path) -> None:
    runtime = _Runtime()
    runner, spec = _make_runner(tmp_path, runtime)
    image_digest = runner.build(spec)
    assert runner.isolation_probe(image_digest).passed
    item = _stage(runner, b"input", upscale_factor=3)

    with pytest.raises(InputStagingError, match="upscale_factor"):
        runner.run_batch(image_digest, [item], 0)

    assert [call["role"] for call in runtime.create_calls] == [
        "probe",
        "probe-overlay-check",
    ]


def test_build_releases_fresh_checkout_only_after_modal_consumes_context(
    tmp_path: Path,
):
    checkout = tmp_path / "fresh-checkout"
    checkout.mkdir()
    (checkout / "Dockerfile").write_text("FROM scratch\n")
    events: list[str] = []

    class Provider:
        def checkout(self, repo_url: str, commit_sha: str) -> Path:
            events.append("checkout")
            return checkout

        def release(self, path: str | Path) -> None:
            assert Path(path) == checkout
            events.append("release")

    class Runtime(_Runtime):
        def build_contender_image(
            self, got_checkout: Path, dockerfile: Path
        ) -> tuple[object, str]:
            assert got_checkout == checkout
            assert dockerfile.read_text() == "FROM scratch\n"
            events.append("modal-build-complete")
            return object(), "im-fresh-release-test"

    runtime = Runtime()
    work = tmp_path / "work-release"
    runner = ModalSandboxRunner(
        Provider(),
        runtime,
        inputs_dir=work / "inputs",
        outputs_dir=work / "outputs",
        scratch_dir=work / "scratch",
    )
    runner.build(
        ContenderSpec(
            contender_id=1,
            repo_url="https://example.invalid/solution.git",
            commit_sha="a" * 40,
            tree_sha="b" * 40,
        )
    )
    assert events == ["checkout", "modal-build-complete", "release"]


def test_request_isolation_mismatch_fails_closed_and_terminates(tmp_path: Path):
    runtime = _Runtime()
    runtime.bad_network_attestation = True
    runner, spec = _make_runner(tmp_path, runtime)
    digest = runner.build(spec)
    with pytest.raises(SandboxIsolationError, match="not isolated"):
        runner.isolation_probe(digest)
    assert runtime.leases[0].terminated == 1
    assert runtime.leases[0].detached == 1


@pytest.mark.parametrize(
    ("input_write", "input_base_mutated"),
    [("2", "0"), ("1", "1")],
)
def test_probe_fails_closed_on_invalid_write_signal_or_persistent_overlay(
    tmp_path: Path,
    input_write: str,
    input_base_mutated: str,
) -> None:
    runtime = _Runtime()
    runtime.probe_input_write = input_write
    runtime.probe_overlay_mutated = input_base_mutated
    runner, spec = _make_runner(tmp_path, runtime)
    digest = runner.build(spec)

    report = runner.isolation_probe(digest)

    assert not report.passed
    assert not report.reference_mounts_absent
    assert all(
        lease.terminated == 1 and lease.detached == 1 for lease in runtime.leases
    )


def test_batch_requires_live_owned_image_and_passed_probe(tmp_path: Path):
    runtime = _Runtime()
    runner, spec = _make_runner(tmp_path, runtime)
    item = _stage(runner, b"input")
    with pytest.raises(UnknownImageError):
        runner.run_batch("f" * 64, [item], 0)
    digest = runner.build(spec)
    with pytest.raises(SandboxIsolationError, match="has not passed"):
        runner.run_batch(digest, [item], 0)


def test_symlink_output_is_rejected_after_frozen_snapshot(tmp_path: Path):
    runtime = _Runtime()
    runtime.output_kind = "symlink"
    runner, spec = _make_runner(tmp_path, runtime, output_poll_seconds=0.001)
    digest = runner.build(spec)
    assert runner.isolation_probe(digest).passed
    item = _stage(runner, b"untrusted")
    with pytest.raises(OutputRejectedError, match="symlink"):
        runner.run_batch(digest, [item], 0)
    assert not any(runner.outputs_dir.iterdir())


def test_live_output_cap_terminates_contender(tmp_path: Path):
    runtime = _Runtime()
    runtime.output_bytes = 8
    runner, spec = _make_runner(
        tmp_path,
        runtime,
        max_output_bytes=4,
        max_batch_output_bytes=4,
        output_poll_seconds=0.001,
    )
    digest = runner.build(spec)
    assert runner.isolation_probe(digest).passed
    item = _stage(runner, b"0123456789")
    with pytest.raises(OversizeOutputError, match="over the cap"):
        runner.run_batch(digest, [item], 0)
    contender = next(x for x in runtime.leases if x.request.role == "contender")
    assert contender.terminated >= 1


def test_output_watchdog_rescans_atomic_rename_and_counts_final_bytes(
    tmp_path: Path,
) -> None:
    class AtomicRenameLease:
        def __init__(self) -> None:
            self.root_scans = 0

        def list_files(self, path: str) -> Sequence[RemoteFile]:
            if path == "/output":
                self.root_scans += 1
                if self.root_scans == 1:
                    return [RemoteFile("/output/.temporary", "directory", 0)]
                return [RemoteFile("/output/final.mkv", "file", 321)]
            if path == "/output/.temporary":
                raise FileNotFoundError(path)
            raise AssertionError(f"unexpected path: {path}")

    runner, _spec = _make_runner(tmp_path, _Runtime())
    lease = AtomicRenameLease()

    total, entries = runner._remote_tree_usage(lease, "/output")  # noqa: SLF001

    assert (total, entries) == (321, 1)
    assert lease.root_scans == 2


def test_output_watchdog_fails_closed_on_persistent_atomic_rename_churn(
    tmp_path: Path,
) -> None:
    class ChurningLease:
        def __init__(self) -> None:
            self.root_scans = 0

        def list_files(self, path: str) -> Sequence[RemoteFile]:
            if path == "/output":
                self.root_scans += 1
                return [
                    RemoteFile(f"/output/.temporary-{self.root_scans}", "directory", 0)
                ]
            if path.startswith("/output/.temporary-"):
                raise FileNotFoundError(path)
            raise AssertionError(f"unexpected path: {path}")

    runner, _spec = _make_runner(tmp_path, _Runtime())
    lease = ChurningLease()

    with pytest.raises(OutputRejectedError, match="kept changing"):
        runner._remote_tree_usage(lease, "/output")  # noqa: SLF001

    assert 1 < lease.root_scans < 10


def test_output_watchdog_fails_closed_when_output_root_disappears(
    tmp_path: Path,
) -> None:
    class MissingRootLease:
        def list_files(self, path: str) -> Sequence[RemoteFile]:
            assert path == "/output"
            raise FileNotFoundError(path)

    runner, _spec = _make_runner(tmp_path, _Runtime())

    with pytest.raises(FileNotFoundError, match="/output"):
        runner._remote_tree_usage(MissingRootLease(), "/output")  # noqa: SLF001


def test_input_digest_mismatch_is_infra_and_creates_no_gpu_batch(tmp_path: Path):
    runtime = _Runtime()
    runner, spec = _make_runner(tmp_path, runtime)
    digest = runner.build(spec)
    assert runner.isolation_probe(digest).passed
    item = _stage(runner, b"original")
    (runner.inputs_dir / item.input_sha256).write_bytes(b"tampered")
    with pytest.raises(InputStagingError):
        runner.run_batch(digest, [item], 0)
    assert [call["role"] for call in runtime.create_calls] == [
        "probe",
        "probe-overlay-check",
    ]


def test_build_timeout_closes_fresh_app_and_poisoned_runner(tmp_path: Path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingRuntime(_Runtime):
        def build_contender_image(
            self, checkout: Path, dockerfile: Path
        ) -> tuple[object, str]:
            assert dockerfile == checkout / "Dockerfile"
            entered.set()
            assert release.wait(2.0)
            return object(), "im-too-late"

    runtime = BlockingRuntime()
    runner, spec = _make_runner(tmp_path, runtime, build_timeout_seconds=0.01)
    try:
        with pytest.raises(BuildTimeout):
            runner.build(spec)
        assert entered.wait(1.0)
    finally:
        release.set()
    assert runtime.closed
    assert not runner.available()


def test_sdk_entrypoint_requires_explicit_fresh_creation_confirmation():
    with pytest.raises(RunnerUnavailableError, match="creation is disabled"):
        ModalSdkRuntime.start_fresh(
            environment_name="vidaio-next-env-abcdef12",
            app_name="vidaio-next-app-abcdef12",
            run_label="vidaio-next-run-abcdef12",
            confirmation="NO",
        )
    assert FRESH_CREATION_CONFIRMATION.startswith("CREATE_FRESH_")


def test_sdk_entrypoint_requires_three_distinct_fresh_names():
    with pytest.raises(RunnerUnavailableError, match="must be distinct"):
        ModalSdkRuntime.start_fresh(
            environment_name="vidaio-next-same-abcdef12",
            app_name="vidaio-next-same-abcdef12",
            run_label="vidaio-next-run-abcdef12",
            confirmation=FRESH_CREATION_CONFIRMATION,
        )


def test_sdk_entrypoint_closes_partially_entered_fresh_context(
    monkeypatch: pytest.MonkeyPatch,
):
    import vidaio.competition.runners.modal_runner as modal_runner_module

    events: list[str] = []

    class FakeContext:
        def __enter__(self):
            events.append("enter")
            raise RuntimeError("synthetic create failure")

        def __exit__(self, exc_type, exc, traceback):
            assert exc_type is RuntimeError
            assert str(exc) == "synthetic create failure"
            assert traceback is not None
            events.append("exit")

    class FakeApp:
        def __init__(self, name, *, tags):
            assert name.startswith("vidaio-next-")
            assert tags["vidaio-resource"] == "vidaio-next"
            events.append("app")

        def run(self, **kwargs):
            assert kwargs["detach"] is False
            events.append("run")
            return FakeContext()

    class FakeModal:
        App = FakeApp

    real_import = modal_runner_module.importlib.import_module
    monkeypatch.setattr(
        modal_runner_module.importlib,
        "import_module",
        lambda name: FakeModal if name == "modal" else real_import(name),
    )

    with pytest.raises(
        RunnerUnavailableError, match="could not create fresh Modal App"
    ):
        ModalSdkRuntime.start_fresh(
            environment_name="vidaio-next-env-abcdef12",
            app_name="vidaio-next-app-abcdef12",
            run_label="vidaio-next-run-abcdef12",
            confirmation=FRESH_CREATION_CONFIRMATION,
        )

    assert events == ["app", "run", "enter", "exit"]


def test_sdk_sandbox_uses_trusted_keepalive_instead_of_contender_cmd():
    observed: dict[str, object] = {}

    class RawSandbox:
        object_id = "sb-fresh-owned"

    class FakeSandboxApi:
        @staticmethod
        def create(*args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return RawSandbox()

    class FakeModal:
        Sandbox = FakeSandboxApi

    runtime = object.__new__(ModalSdkRuntime)
    runtime.run_label = "vidaio-next-run-abcdef12"
    runtime._modal = FakeModal
    runtime._environment_name = "vidaio-next-env-abcdef12"
    runtime._app = object()
    runtime._active = True
    runtime._leases = []
    lease = runtime.create_sandbox(
        image=object(),
        name="vidaio-next-sandbox-abcdef12",
        role="contender",
        tags={"vidaio-resource": "vidaio-next"},
        gpu="L4",
        cpu=2.0,
        memory_mb=8192,
        timeout_seconds=600,
        idle_timeout_seconds=120,
    )

    assert lease.object_id == "sb-fresh-owned"
    assert observed["args"] == (
        "/bin/sh",
        "-c",
        "while :; do sleep 3600; done",
    )
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["block_network"] is True
    assert kwargs["gpu"] == "L4"


def test_sdk_list_files_translates_modal_missing_path() -> None:
    class SandboxFilesystemNotFoundError(Exception):
        pass

    class RawFilesystem:
        @staticmethod
        def list_files(path: str):
            raise SandboxFilesystemNotFoundError(path)

    class RawSandbox:
        object_id = "sb-fresh-owned"
        filesystem = RawFilesystem()

    class FakeSandboxApi:
        @staticmethod
        def create(*args, **kwargs):
            return RawSandbox()

    class FakeModalException:
        pass

    FakeModalException.SandboxFilesystemNotFoundError = SandboxFilesystemNotFoundError

    class FakeModal:
        Sandbox = FakeSandboxApi
        exception = FakeModalException

    runtime = object.__new__(ModalSdkRuntime)
    runtime.run_label = "vidaio-next-run-abcdef12"
    runtime._modal = FakeModal
    runtime._environment_name = "vidaio-next-env-abcdef12"
    runtime._app = object()
    runtime._active = True
    runtime._leases = []
    lease = runtime.create_sandbox(
        image=object(),
        name="vidaio-next-sandbox-abcdef12",
        role="contender",
        tags={"vidaio-resource": "vidaio-next"},
        gpu="L4",
        cpu=2.0,
        memory_mb=8192,
        timeout_seconds=600,
        idle_timeout_seconds=120,
    )

    with pytest.raises(FileNotFoundError, match="/output/.temporary"):
        lease.list_files("/output/.temporary")


def test_sdk_input_image_explicitly_includes_hidden_factor_sidecars(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    class FakeImageBuilder:
        def add_local_dir(self, local_path, remote_path, **kwargs):
            observed.update(
                local_path=local_path, remote_path=remote_path, kwargs=kwargs
            )
            return self

        def build(self, app):
            observed["app"] = app

    class FakeImageApi:
        @staticmethod
        def from_scratch(*, force_build):
            assert force_build is True
            return FakeImageBuilder()

    class FakeModal:
        Image = FakeImageApi

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / f".vidaio-next-upscale-factor-{'a' * 64}").write_text("2\n")
    runtime = object.__new__(ModalSdkRuntime)
    runtime._modal = FakeModal
    runtime._app = object()
    runtime._active = True

    runtime.build_input_image(staged)

    assert observed["local_path"] == staged
    assert observed["remote_path"] == "/"
    assert observed["kwargs"] == {"copy": True, "ignore": []}
    assert observed["app"] is runtime._app


def test_sdk_restore_uses_only_the_exact_owned_immutable_image_id() -> None:
    observed: list[str] = []

    class Handle:
        object_id = "im-owned-abcdef12"

    class FakeImageApi:
        @staticmethod
        def from_id(image_object_id: str):
            observed.append(image_object_id)
            return Handle()

    class FakeModal:
        Image = FakeImageApi

    runtime = object.__new__(ModalSdkRuntime)
    runtime._modal = FakeModal
    runtime._active = True

    image, image_id = runtime.restore_contender_image("im-owned-abcdef12")

    assert isinstance(image, Handle)
    assert image_id == "im-owned-abcdef12"
    assert observed == ["im-owned-abcdef12"]
    with pytest.raises(ValueError, match="invalid Modal image object id"):
        runtime.restore_contender_image("sb-not-an-image")


def test_modal_adapter_source_has_no_inventory_discovery_or_sandbox_restore():
    source = Path("vidaio/competition/runners/modal_runner.py").read_text()
    forbidden = (
        "App.lookup",
        "Sandbox.from_id",
        "Sandbox.from_name",
        "Sandbox.list",
        "Image.from_name",
        "Volume.from_name",
        "Secret.from_name",
        "Function.from_name",
    )
    assert not [needle for needle in forbidden if needle in source]
    # The sole recovery exception is an exact immutable Image id previously
    # persisted by this competition. Sandboxes/instances remain create-only.
    assert "def restore_contender_image" in source
    assert "self._modal.Image.from_id(image_object_id)" in source
    assert "force_build=True" in source
    assert ".entrypoint([])" in source
    assert ".cmd(list(_KEEPALIVE_COMMAND))" in source
    assert '"while :; do sleep 3600; done"' in source
    assert "block_network=True" in source
    assert "include_oidc_identity_token=False" in source
    assert "secrets=[]" in source
    assert "volumes={}" in source
    assert "gpu=None" in source  # collector/build; contender GPU is configured
