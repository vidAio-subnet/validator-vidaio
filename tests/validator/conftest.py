"""Fixtures for the validator suite: in-memory DB, InMemoryChain, all-fake clients.

A full InferenceValidator round runs deterministically in-process: the fake
challenge client serves real files (digests computed), the fake miner client
writes real output files, the fake scoring client returns genuine ItemScore
packets. No network, no subprocess, no wall-clock dependence beyond the tiny
timeouts configured here.
"""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

# pytest runs with --import-mode=importlib, which does not add test dirs to
# sys.path; make the local helper module (validator_support — uniquely named to
# avoid sys.modules collisions across suites) importable.
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from vidaio.audit import LocalFsStore
from vidaio.chain.adapter import InMemoryChain
from vidaio.core import apply_migrations, connect
from vidaio.validator import MIGRATIONS_DIR, InferenceValidator

from validator_support import (
    VALIDATOR_IDENTITY,
    FakeChallengeClient,
    FakeMinerClient,
    FakeScoringClient,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A FILE-backed validator DB.

    Deliberately not ':memory:': the health checks must be answerable from the
    HealthServer's own thread with their own connection, and the
    weight-setter reads this same file as a separate process.
    """
    return tmp_path / "validator.db"


@pytest.fixture
def conn(db_path: Path) -> sqlite3.Connection:
    connection = connect(db_path)
    apply_migrations(connection, MIGRATIONS_DIR)
    yield connection
    connection.close()


@pytest.fixture
def raw_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "core": {"data_dir": str(tmp_path / "data")},
        "validator": {
            # WHO this validator is: stamped on every /challenge/next and the
            # ownership boundary of the orphan sweep.
            "identity": VALIDATOR_IDENTITY,
            "cycle_sleep_min_seconds": 0.01,
            "cycle_sleep_max_seconds": 0.02,
            "metagraph_refresh_seconds": 0.0,
            "warrant_probe_timeout_seconds": 0.2,
            "miner_request_timeout_seconds": 0.2,
            "challenge_request_timeout_seconds": 0.5,
            "scoring_request_timeout_seconds": 0.5,
        },
        "tokenomics": {},
    }


@pytest.fixture
def chain() -> InMemoryChain:
    return InMemoryChain()


@pytest.fixture
def challenge_client(tmp_path: Path) -> FakeChallengeClient:
    return FakeChallengeClient(tmp_path)


@pytest.fixture
def miner_client(tmp_path: Path) -> FakeMinerClient:
    outdir = tmp_path / "outputs"
    outdir.mkdir()
    return FakeMinerClient(outdir)


@pytest.fixture
def scoring_client() -> FakeScoringClient:
    return FakeScoringClient()


@pytest.fixture
def store(tmp_path: Path) -> LocalFsStore:
    """Audit store for SCORE_PACKET evidence."""
    return LocalFsStore(tmp_path / "audit")


@pytest.fixture
def make_validator(
    raw_config: dict[str, Any],
    chain: InMemoryChain,
    challenge_client: FakeChallengeClient,
    miner_client: FakeMinerClient,
    scoring_client: FakeScoringClient,
    conn: sqlite3.Connection,
    store: LocalFsStore,
):
    """Build an InferenceValidator over the shared fakes, with keyword overrides."""

    def _mk(**overrides: Any) -> InferenceValidator:
        kwargs: dict[str, Any] = {
            "chain": chain,
            "challenge_client": challenge_client,
            "miner_client": miner_client,
            "scoring_client": scoring_client,
            "conn": conn,
            "store": store,
            "rng": random.Random(85),
        }
        config_overrides = overrides.pop("config", {})
        kwargs.update(overrides)
        raw = {
            **raw_config,
            "validator": {**raw_config["validator"], **config_overrides},
        }
        return InferenceValidator(raw, **kwargs)

    return _mk


@pytest.fixture
def validator(make_validator) -> InferenceValidator:
    return make_validator()
