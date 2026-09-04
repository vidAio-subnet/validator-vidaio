from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pytest

# pytest runs with --import-mode=importlib, which does not add test dirs to sys.path;
# make the local helper module (weightsetter_support) importable. The name is unique
# across the test tree on purpose (see tests/integration/conftest.py).
sys.path.insert(0, str(Path(__file__).parent))

from weightsetter_support import Clock, FakeSnapshots

from vidaio.audit import CommitmentLedger, LocalFsStore
from vidaio.chain import InMemoryChain
from vidaio.core.db import connect
from vidaio.tokenomics import CompetitionResult, ContenderResult, MinerSnapshot
from vidaio.weightsetter import WeightSetter, migrate

T0 = Clock().now  # shared deterministic epoch (weightsetter_support.T0)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def chain() -> InMemoryChain:
    return InMemoryChain()


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    # FILE-backed on purpose: the health checks must be answerable from the
    # HealthServer's own thread with their own connection, which is
    # impossible for a ':memory:' database.
    c = connect(tmp_path / "weightsetter.db")
    migrate(c)
    return c


@pytest.fixture
def store(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path / "audit")


@pytest.fixture
def ledger(tmp_path: Path) -> CommitmentLedger:
    return CommitmentLedger.open(tmp_path / "ledger.db")


@pytest.fixture
def raw_config() -> dict[str, Any]:
    # metrics_port 0 -> ephemeral bind if a health server ever starts in a test;
    # tight chain timeout/backoff keeps the retry tests fast.
    return {
        "core": {"metrics_port": 0},
        "weightsetter": {
            "metrics_port": 0,
            "chain_timeout_seconds": 0.2,
            "chain_retry_attempts": 3,
            "chain_retry_base_delay_seconds": 0.01,
            # Dependency-free report-mode canonical sink. Production resolves
            # the subnet-owner uid from its live chain adapter instead.
            "burn_uid": 999,
        },
    }


@pytest.fixture
def mk_miner():
    def _mk(
        uid: int,
        *,
        track: str = "compression",
        score: float = 0.5,
        hotkey: str | None = None,
    ) -> MinerSnapshot:
        return MinerSnapshot(
            uid=uid,
            hotkey=hotkey if hotkey is not None else f"hk{uid}",
            coldkey=f"ck{uid}",
            ip=f"10.0.0.{uid}",
            track=track,
            accumulate_score=score,
        )

    return _mk


@pytest.fixture
def mk_result():
    def _mk(
        cycle: int = 1,
        applied_at: datetime = T0,
        contenders: Sequence[ContenderResult] = (),
        baseline_score: float | None = 0.5,
        competition_id: str | None = None,
        track: str = "compression",
        baseline_version: int = 1,
        baseline_artifact_digest: str = "a" * 64,
    ) -> CompetitionResult:
        return CompetitionResult(
            competition_id=competition_id or f"competition-{cycle}",
            track=track,
            cycle=cycle,
            applied_at=applied_at,
            contenders=tuple(contenders),
            baseline_score=baseline_score,
            baseline_version=baseline_version,
            baseline_artifact_digest=baseline_artifact_digest,
        )

    return _mk


@pytest.fixture
def make_setter(raw_config, chain, conn, store, ledger, clock):
    """Build a WeightSetter with fakes; keyword overrides patch the weightsetter section."""

    def _mk(
        miners: Sequence[MinerSnapshot],
        *,
        chain_override=None,
        snapshots_override=None,
        publication_inputs=None,
        wall_clock=None,
        **overrides: Any,
    ) -> WeightSetter:
        raw = {
            **raw_config,
            "weightsetter": {**raw_config["weightsetter"], **overrides},
        }
        kwargs: dict[str, Any] = {}
        if wall_clock is not None:
            kwargs["wall_clock"] = wall_clock
        return WeightSetter(
            raw,
            chain=chain_override if chain_override is not None else chain,
            snapshots=(
                snapshots_override
                if snapshots_override is not None
                else FakeSnapshots(miners)
            ),
            conn=conn,
            store=store,
            ledger=ledger,
            publication_inputs=publication_inputs,
            clock=clock,
            **kwargs,
        )

    return _mk
