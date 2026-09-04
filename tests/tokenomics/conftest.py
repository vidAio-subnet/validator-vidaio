from __future__ import annotations

from datetime import UTC, datetime
from typing import Sequence

import pytest

from vidaio.tokenomics import (
    CompetitionResult,
    ContenderResult,
    MinerSnapshot,
    TokenomicsConfig,
)

T0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
BASELINE_DIGEST = "ab" * 32


@pytest.fixture
def cfg() -> TokenomicsConfig:
    return TokenomicsConfig()


@pytest.fixture
def live_cfg() -> TokenomicsConfig:
    return TokenomicsConfig(competition_emissions_enabled=True)


@pytest.fixture
def mk_miner():
    def _mk(
        uid: int,
        *,
        track: str = "compression",
        score: float = 0.5,
        ip: str | None = None,
        coldkey: str | None = None,
        hotkey: str | None = None,
        excluded: bool = False,
    ) -> MinerSnapshot:
        return MinerSnapshot(
            uid=uid,
            hotkey=hotkey or f"hk{uid}",
            coldkey=coldkey or f"ck{uid}",
            ip=ip or f"10.0.0.{uid}",
            track=track,
            accumulate_score=score,
            excluded=excluded,
        )

    return _mk


@pytest.fixture
def mk_result():
    def _mk(
        cycle: int = 1,
        applied_at: datetime = T0,
        scores: Sequence[float] = (0.51,),
        baseline_score: float | None = 0.5,
        contenders: Sequence[ContenderResult] | None = None,
        start_uid: int = 100,
        competition_id: str | None = None,
        track: str = "compression",
        baseline_version: int = 0,
        baseline_artifact_digest: str = BASELINE_DIGEST,
    ) -> CompetitionResult:
        if contenders is None:
            contenders = tuple(
                ContenderResult(
                    hotkey=f"comp{start_uid + i}", uid=start_uid + i, score=score
                )
                for i, score in enumerate(scores)
            )
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
def mk_podium_miners(mk_miner):
    def _mk(*uids: int) -> list[MinerSnapshot]:
        return [mk_miner(uid, hotkey=f"comp{uid}", score=0.0) for uid in uids]

    return _mk
