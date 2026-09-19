"""The sandbox runtime exists only while needed and is replaced when it dies."""

from __future__ import annotations

import pytest

from vidaio.competition.runners.errors import RunnerUnavailableError
from vidaio.competition.runners.renewable import (
    RenewableRunner,
    generation_name,
    is_dead_runtime_error,
)
from vidaio.competition.states import Phase

from orchestrator_support import FakeRunner, build_manifest, phase, start_and_enroll


class _Inner:
    def __init__(self, sequence: int, log: list[str]) -> None:
        self.sequence = sequence
        self.runtime_session_id = f"{sequence:064x}"
        self.runtime_label = f"vidaio-next-test-g{sequence:03d}"
        self._log = log
        self.alive = True
        self.closed = False
        self.fail_with: BaseException | None = None

    def available(self) -> bool:
        return self.alive and not self.closed

    def build(self, contender) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self._log.append(f"build:{self.sequence}")
        return f"digest-{self.sequence}"

    def has_live_image(self, digest: str) -> bool:
        return True

    def close(self) -> None:
        self.closed = True
        self._log.append(f"close:{self.sequence}")


def _runner(**kwargs):
    log: list[str] = []
    made: list[_Inner] = []

    def factory(sequence: int, resources=None) -> _Inner:
        inner = _Inner(sequence, log)
        inner.resources = resources
        made.append(inner)
        return inner

    now = [0.0]
    runner = RenewableRunner(
        factory, static={"gpu": "L4"}, clock=lambda: now[0], **kwargs
    )
    return runner, made, log, now


def test_nothing_remote_exists_until_acquired() -> None:
    runner, made, _log, _now = _runner()
    assert made == [] and not runner.active
    assert runner.available()  # idle is healthy: there is nothing that can die
    assert runner.gpu == "L4"  # manifest policy check works while idle
    assert getattr(runner, "runtime_session_id", None) is None
    with pytest.raises(RunnerUnavailableError):
        runner.build(object())


def test_acquire_is_idempotent_and_release_respects_grace_and_in_flight() -> None:
    runner, made, log, now = _runner(idle_grace_seconds=120.0)
    runner.acquire()
    runner.acquire()
    assert len(made) == 1 and runner.runtime_session_id == made[0].runtime_session_id
    runner.release()  # inside the grace window: a tick must not close it
    assert runner.active
    now[0] = 121.0
    runner.release()
    assert not runner.active and log == ["close:1"]
    runner.acquire()
    assert [inner.sequence for inner in made] == [1, 2]  # never the same generation


def test_dead_remote_runtime_is_replaced_on_next_acquire() -> None:
    runner, made, log, _now = _runner()
    runner.acquire()
    made[0].fail_with = RuntimeError("ConflictError: App state is APP_STATE_STOPPED")
    with pytest.raises(RuntimeError):
        runner.build(object())
    assert not runner.active and "close:1" in log
    runner.acquire()
    assert runner.build(object()) == "digest-2"
    assert runner.runtime_session_id == made[1].runtime_session_id


def test_ordinary_contender_failure_keeps_the_runtime() -> None:
    runner, made, _log, _now = _runner()
    runner.acquire()
    made[0].fail_with = ValueError("Dockerfile is invalid")
    with pytest.raises(ValueError):
        runner.build(object())
    assert runner.active and len(made) == 1


def test_unavailable_inner_runner_is_replaced() -> None:
    runner, made, _log, _now = _runner()
    runner.acquire()
    made[0].alive = False
    assert not runner.available()
    runner.acquire()
    assert len(made) == 2 and runner.available()


def test_factory_failure_propagates_and_next_acquire_retries() -> None:
    calls = []

    def factory(sequence: int, resources=None):
        calls.append(sequence)
        if sequence == 1:
            raise RunnerUnavailableError("could not create fresh Modal App")
        return _Inner(sequence, [])

    runner = RenewableRunner(factory)
    with pytest.raises(RunnerUnavailableError):
        runner.acquire()
    runner.acquire()
    assert calls == [1, 2] and runner.active


def test_generation_names_are_unique_bounded_and_keep_the_prefix() -> None:
    base = "vidaio-next-m1-comp-env-20260904-1428"
    first = generation_name(base, 1, stamp="0919T120000")
    second = generation_name(base, 2, stamp="0919T120001")
    assert first != second and first.startswith("vidaio-next-")
    long_base = "vidaio-next-" + "x" * 80
    assert len(generation_name(long_base, 7, stamp="0919T120000")) <= 63
    assert generation_name(long_base, 7, stamp="0919T120000").endswith("-g007-0919T120000")


@pytest.mark.parametrize(
    ("text", "dead"),
    [
        ("modal.exception.ConflictError: App state is APP_STATE_STOPPED", True),
        ("the owned ephemeral App was closed and this runner must be replaced", True),
        ("Dockerfile parse error", False),
    ],
)
def test_dead_runtime_detection(text: str, dead: bool) -> None:
    assert is_dead_runtime_error(RuntimeError(text)) is dead


async def test_orchestrator_acquires_for_building_and_releases_when_idle(
    orchestrator_factory, fixture_repos
) -> None:
    from orchestrator_support import FINALIZATION, M

    made: list[FakeRunner] = []
    holder: dict = {}

    def factory(_sequence: int, _resources=None) -> FakeRunner:
        inner = FakeRunner(holder["outputs"])
        made.append(inner)
        return inner

    renewable = RenewableRunner(factory, idle_grace_seconds=0.0)
    orch = orchestrator_factory(repos=fixture_repos, runner=renewable)
    holder["outputs"] = orch.outputs_dir
    cid = await start_and_enroll(orch, build_manifest("comp-renew"), ["hk-a"])
    assert made == []  # enrollment holds no sandbox runtime
    offset = 0
    while phase(orch, cid) is not Phase.BUILDING and offset < 20:
        await orch.step(FINALIZATION + offset * M)
        offset += 1
    assert phase(orch, cid) is Phase.BUILDING and made == []
    await orch.step(FINALIZATION + offset * M)  # the BUILDING stage itself runs
    assert len(made) == 1 and renewable.active


def test_changed_compute_envelope_forces_a_new_generation() -> None:
    runner, made, log, _now = _runner()
    small = {"cpu": 2.0, "memory_mb": 8192, "batch_timeout_seconds": 900}
    big = {"cpu": 32.0, "memory_mb": 65536, "batch_timeout_seconds": 3600}
    runner.configure(small)
    runner.acquire()
    runner.acquire()
    assert len(made) == 1 and made[0].resources == small
    runner.configure(big)
    runner.acquire()
    assert len(made) == 2 and made[1].resources == big and "close:1" in log


def test_manifest_sandbox_resources_are_anchored_and_bounded() -> None:
    from vidaio.competition.config import CompetitionConfig
    from vidaio.competition.manifest import (
        CompetitionManifest,
        ManifestBoundsError,
        validate_against_config,
    )

    plain = build_manifest("comp-res-plain")
    assert "sandbox_resources" not in plain.canonical_json()
    sized = build_manifest(
        "comp-res-plain",
        sandbox_resources={"cpu": 32, "memory_mb": 65536, "batch_timeout_seconds": 1800},
    )
    assert sized.manifest_digest() != plain.manifest_digest()
    assert (
        CompetitionManifest.model_validate_json(sized.canonical_json()).manifest_digest()
        == sized.manifest_digest()
    )
    validate_against_config(sized, CompetitionConfig())
    with pytest.raises(ManifestBoundsError, match="cpu"):
        validate_against_config(sized, CompetitionConfig(sandbox_cpu_max=16))
    with pytest.raises(ValueError):
        build_manifest("comp-res-bad", sandbox_resources={"cpu": 0, "memory_mb": 1, "batch_timeout_seconds": 1})


async def test_operator_abort_frees_the_running_slot(orchestrator_factory, fixture_repos):
    import httpx

    from orchestrator_support import T0

    token = "abort-token"
    orch = orchestrator_factory(repos=fixture_repos, control_token=token)
    cid = await start_and_enroll(orch, build_manifest("comp-abort"), ["hk-a"])
    auth = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app), base_url="http://control"
    ) as client:
        refused = await client.post(
            f"/competitions/{cid}/abort", headers=auth, json={"operator": " ", "reason": "x"}
        )
        assert refused.status_code == 422
        done = await client.post(
            f"/competitions/{cid}/abort",
            headers=auth,
            json={"operator": "ops", "reason": "hidden media leaked"},
        )
        assert done.status_code == 200 and done.json()["status"] == "CANCELLED"
        again = await client.post(
            f"/competitions/{cid}/abort", headers=auth,
            json={"operator": "ops", "reason": "twice"},
        )
        assert again.status_code == 409  # terminal phases have no outgoing edge
    assert phase(orch, cid) is Phase.CANCELLED
    from vidaio.competition import repository as repo

    assert repo.running_competition_id(orch.conn) is None
