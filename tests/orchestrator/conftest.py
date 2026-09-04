from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

# pytest 9 uses importlib import mode; make the shared support module importable.
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from orchestrator_support import (
    BASELINE_URL,
    DOCKER,
    DOCKERFILE,
    EXIT_ONE_RUN_SH,
    FLOOD_RUN_SH,
    HONEST_RUN_SH,
    LOG_FLOOD_RUN_SH,
    LYING_PROBE_DOCKERFILE,
    UNTRUSTED_RUN_SH,
    NO_OUTPUT_RUN_SH,
    SYMLINK_RUN_SH,
    FakeRunner,
    FakeScoringClient,
    client_conn,
    make_raw_config,
    repo_url,
    write_solution,
)


def _docker_available() -> bool:
    try:
        proc = subprocess.run(
            [DOCKER, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=20,
        )
        return proc.returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "docker: exercises the real local Docker daemon (spec §05 sandbox)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if DOCKER_AVAILABLE:
        return  # docker-marked tests run by default (local-first: docker is up)
    skip = pytest.mark.skip(reason="docker daemon not available")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixture_repos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """repo_url -> local checkout dir. Compression quality: baseline (256 B) beats
    hk-a (512 B) beats hk-b (1024 B); hk-mal is the untrusted fixture (network
    attempt + out-of-mount writes + a secret-shaped env var baked into the image)."""
    root = tmp_path_factory.mktemp("solution-repos")
    repos = {
        repo_url("hk-a"): write_solution(
            root / "contender-a", HONEST_RUN_SH.format(head=512)
        ),
        repo_url("hk-b"): write_solution(
            root / "contender-b", HONEST_RUN_SH.format(head=1024)
        ),
        BASELINE_URL: write_solution(root / "baseline", HONEST_RUN_SH.format(head=256)),
        repo_url("hk-mal"): write_solution(
            root / "untrusted",
            UNTRUSTED_RUN_SH,
            DOCKERFILE + "ENV VIDAIO_VALIDATOR_PAT=exfiltrated\n",
        ),
        #
        repo_url("hk-sym"): write_solution(root / "symlink-output", SYMLINK_RUN_SH),
        repo_url("hk-flood"): write_solution(root / "flood-output", FLOOD_RUN_SH),
        repo_url("hk-exit"): write_solution(root / "exit-one", EXIT_ONE_RUN_SH),
        repo_url("hk-silent"): write_solution(root / "no-output", NO_OUTPUT_RUN_SH),
        repo_url("hk-lie"): write_solution(
            root / "lying-probe", HONEST_RUN_SH.format(head=512), LYING_PROBE_DOCKERFILE
        ),
        repo_url("hk-logflood"): write_solution(root / "log-flood", LOG_FLOOD_RUN_SH),
    }
    # A submission tree carrying a symlink to a host file: the backup tarball must
    # never follow it and validation must reject the contender.
    link_repo = write_solution(root / "symlink-repo", HONEST_RUN_SH.format(head=512))
    link = link_repo / "stolen-credentials"
    if not link.exists():
        link.symlink_to("/etc/passwd")
    repos[repo_url("hk-link")] = link_repo
    return repos


@pytest.fixture
def orchestrator_factory(tmp_path: Path):
    """Build orchestrators sharing tmp_path state (DB, work dir, audit store) so
    re-instantiation models a process restart over the same persisted state."""
    from vidaio.competition.orchestrator import Orchestrator
    from vidaio.competition.runners import LocalRepoProvider

    created: list[Any] = []
    extra_conns: list[Any] = []

    def factory(
        *,
        runner: Any = None,
        scoring_client: Any = None,
        repos: dict[str, Path] | None = None,
        repo_provider: Any = None,
        chain: Any = None,
        clock: Any = None,
        **overrides: Any,
    ):
        raw = make_raw_config(tmp_path, **overrides)
        work = Path(raw["orchestrator"]["work_dir"])
        runner = runner if runner is not None else FakeRunner(work / "outputs")
        client = scoring_client if scoring_client is not None else FakeScoringClient()
        # `repo_provider` lets a test inject a failing/flaky provider (review
        # round 2, new-5); otherwise the plain fixture mapping is used.
        provider = (
            repo_provider if repo_provider is not None else LocalRepoProvider(repos or {})
        )
        orch = Orchestrator(
            raw,
            runner=runner,
            scoring_client=client,
            repo_provider=provider,
            chain=chain,
            clock=clock,
        )
        if hasattr(client, "conn") and getattr(client, "conn", None) is None:
            client.conn = client_conn(orch.core.db_path)
            extra_conns.append(client.conn)
        created.append(orch)
        return orch

    yield factory
    for conn in extra_conns:
        conn.close()
    for orch in created:
        orch.conn.close()
