"""Tokenomics v3: the per-competition payout policy survives manifest -> evidence -> epoch log."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from vidaio.competition.manifest import ResultRules
from vidaio.epoch.log import CompetitionRulesInput, _reward_window_from_obj, _reward_window_obj
from vidaio.tokenomics import EmissionState, RewardWindowState, TokenomicsConfig, resolve_reward_window
from vidaio.tokenomics.state import CompetitionResult, CompetitionRules, ContenderResult

T0 = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
DIGEST = "ab" * 32


def test_manifest_rules_validate_the_payout_policy() -> None:
    rules = ResultRules(
        crown_margin=0.05, crown_min_score=0.869, podium_min_score=0.86,
        crown_competition_share=1.0, podium_competition_share=0.8,
        crown_split=[0.9, 0.04, 0.03, 0.02, 0.01], podium_split=[0.5, 0.24, 0.13, 0.08, 0.05],
        redistribute_empty_places=True, no_qualifier_closes_window=True,
    )
    assert rules.crown_split == (0.9, 0.04, 0.03, 0.02, 0.01)
    for bad in (
        {"crown_split": [0.9, 0.2]},
        {"podium_split": []},
        {"podium_split": [0.2] * 6},
        {"crown_competition_share": 1.5},
    ):
        with pytest.raises(ValueError):
            ResultRules(crown_margin=0.05, **bad)


def test_rules_input_mirrors_the_manifest_and_omits_absent_policy_fields() -> None:
    manifest = ResultRules(crown_margin=0.05, podium_split=[0.6, 0.4])
    committed = CompetitionRulesInput(**manifest.model_dump())
    assert committed.podium_split == (0.6, 0.4)
    obj = committed._canonical_obj()
    assert obj["podium_split"] == [0.6, 0.4]
    assert "crown_split" not in obj and "crown_competition_share" not in obj
    plain = CompetitionRulesInput(crown_margin=0.05)._canonical_obj()
    assert set(plain) == {"crown_margin", "crown_min_score", "podium_min_margin", "podium_min_score"}


def test_window_policy_roundtrips_through_the_epoch_log_and_pays_from_the_window() -> None:
    config = TokenomicsConfig(competition_emissions_enabled=True)
    rules = CompetitionRules(crown_margin=0.05, crown_min_score=0.95, podium_min_score=0.86, podium_competition_share=0.75,
                             podium_split=(0.6, 0.4))
    result = CompetitionResult(
        competition_id="c", track="compression", cycle=1, applied_at=T0,
        contenders=(ContenderResult("a", 1, 0.87), ContenderResult("b", 2, 0.865), ContenderResult("c", 3, 0.861)),
        baseline_score=0.3, baseline_version=0, baseline_artifact_digest=DIGEST, rules=rules,
    )
    state = resolve_reward_window(config, RewardWindowState(), result)
    assert state.kind is EmissionState.PODIUM
    assert state.podium_hotkeys == ("a", "b") and state.place_shares == (0.6, 0.4)
    assert state.competition_share == 0.75
    obj = _reward_window_obj(state, schema_version=18)
    assert obj["competition_share"] == 0.75 and obj["place_shares"] == [0.6, 0.4]
    back = _reward_window_from_obj(json.loads(json.dumps(obj)))
    assert back == state
    legacy = _reward_window_obj(state, schema_version=17)
    assert "place_shares" not in legacy and "competition_share" not in legacy
    # weights follow the window, not the (different) live defaults
    from vidaio.tokenomics import MinerSnapshot, build_weight_vector

    miners = [MinerSnapshot(uid=u, hotkey=h, coldkey=f"ck{u}", ip=f"10.0.0.{u}", track="compression", accumulate_score=0.0)
              for u, h in ((1, "a"), (2, "b"), (3, "c"))]
    vector = build_weight_vector(config, miners, burn_uid=0, reward_state=state, now=T0 + timedelta(hours=1))
    assert vector == pytest.approx({1: 0.45, 2: 0.30, 3: 0.0, 0: 0.25})
