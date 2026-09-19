"""Per-competition crown/podium rules anchored in the manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from vidaio.competition.epoch_evidence import build_competition_epoch_evidence
from vidaio.competition.manifest import CompetitionManifest, ResultRules
from vidaio.epoch import MinerCensusEntry
from vidaio.epoch.log import CompetitionRulesInput
from vidaio.tokenomics.breakthrough import (
    qualifies_for_crown,
    qualifies_for_podium,
    resolve_reward_window,
)
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.state import CompetitionRules, EmissionState, RewardWindowState

from integration_support import COMPLETED_AT, build_golden_world, build_manifest

CONFIG = TokenomicsConfig(competition_emissions_enabled=True)
# Golden world: baseline mean 0.51875, hk-a 0.56161 (+8.26 %), hk-b 0.0.


def _census() -> dict[str, MinerCensusEntry]:
    return {
        hotkey: MinerCensusEntry(
            uid=index + 1, hotkey=hotkey, coldkey=f"ck-{hotkey}", ip=f"10.0.0.{index + 1}"
        )
        for index, hotkey in enumerate(("hk-a", "hk-b"))
    }


def _evidence(tmp_path: Path, rules: dict | None):
    world = build_golden_world(
        tmp_path, manifest_overrides=None if rules is None else {"result_rules": rules}
    )
    return build_competition_epoch_evidence(
        world.comp_conn,
        census_by_hotkey=_census(),
        store=world.store,
        tokenomics=CONFIG,
        through_time=COMPLETED_AT,
    )


def test_manifest_without_rules_keeps_its_anchored_digest() -> None:
    manifest = build_manifest()
    assert manifest.result_rules is None
    assert "result_rules" not in manifest.canonical_json()
    reloaded = CompetitionManifest.model_validate_json(manifest.canonical_json())
    assert reloaded.manifest_digest() == manifest.manifest_digest()


def test_rules_change_the_manifest_digest_and_round_trip() -> None:
    plain = build_manifest()
    ruled = build_manifest(result_rules={"crown_margin": 0.1, "crown_min_score": 0.6})
    assert ruled.manifest_digest() != plain.manifest_digest()
    reloaded = CompetitionManifest.model_validate_json(ruled.canonical_json())
    assert reloaded.result_rules == ResultRules(crown_margin=0.1, crown_min_score=0.6)
    assert reloaded.manifest_digest() == ruled.manifest_digest()


@pytest.mark.parametrize(
    "bad",
    [
        {"crown_margin": 0.0},
        {"crown_margin": 0.05, "crown_min_score": 1.5},
        {"crown_margin": 0.05, "podium_min_margin": 0.2},
        {"crown_margin": 0.05, "winner_hotkey": "x"},
    ],
)
def test_invalid_rules_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        build_manifest(result_rules=bad)


def test_default_rules_crown_the_golden_winner(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, None)
    assert evidence is not None and evidence.competition_input.rules is None
    window = resolve_reward_window(CONFIG, RewardWindowState(), evidence.result)
    assert window.kind is EmissionState.CROWN


def test_higher_crown_margin_turns_the_same_result_into_a_podium(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, {"crown_margin": 0.10})
    assert evidence is not None
    assert evidence.competition_input.rules == CompetitionRulesInput(crown_margin=0.10)
    assert evidence.result.rules == CompetitionRules(crown_margin=0.10)
    window = resolve_reward_window(CONFIG, RewardWindowState(), evidence.result)
    assert window.kind is EmissionState.PODIUM
    assert window.podium_hotkeys == ("hk-a", "hk-b")


def test_absolute_crown_floor_replaces_a_mutable_baseline(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, {"crown_margin": 0.05, "crown_min_score": 0.60})
    assert evidence is not None
    window = resolve_reward_window(CONFIG, RewardWindowState(), evidence.result)
    assert window.kind is EmissionState.PODIUM  # +8.26 % but below the 0.60 bar


def test_podium_condition_removes_non_qualifying_contenders(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, {"crown_margin": 0.05, "podium_min_score": 0.10})
    assert evidence is not None
    # The result keeps every committed contender (the epoch log binds that identity
    # set); the anchored condition only decides who holds a PAID rank.
    assert [c.hotkey for c in evidence.result.contenders] == ["hk-a", "hk-b"]
    window = resolve_reward_window(CONFIG, RewardWindowState(), evidence.result)
    assert window.podium_hotkeys == ("hk-a",)


def test_no_qualifying_contender_means_no_result(tmp_path: Path) -> None:
    assert _evidence(tmp_path, {"crown_margin": 0.05, "podium_min_score": 0.90}) is None


def test_rule_predicates_are_inclusive() -> None:
    rules = CompetitionRules(crown_margin=0.10, crown_min_score=0.55, podium_min_margin=0.0)
    assert qualifies_for_crown(CONFIG, 0.5, 0.55, rules)
    assert not qualifies_for_crown(CONFIG, 0.5, 0.5499, rules)
    assert qualifies_for_crown(CONFIG, 0.5, 0.525)  # protocol default 5 %
    assert qualifies_for_podium(rules, 0.5, 0.5)
    assert not qualifies_for_podium(rules, 0.5, 0.4999)
    assert qualifies_for_podium(None, 0.5, 0.0)


def _finalized_log(tmp_path: Path, rules: dict | None, **extra_overrides):
    from vidaio.auditor import Auditor, AuditorConfig, InMemoryBundleSource
    from vidaio.authority.finalizer import EpochFinalizer, build_audit_manifest
    from vidaio.tokenomics.state import MinerSnapshot

    overrides = dict(extra_overrides)
    if rules is not None:
        overrides["result_rules"] = rules
    world = build_golden_world(tmp_path, manifest_overrides=overrides)
    evidence = build_competition_epoch_evidence(
        world.comp_conn, census_by_hotkey=_census(), store=world.store,
        tokenomics=CONFIG, through_time=COMPLETED_AT,
    )
    assert evidence is not None

    class _NoInferenceCommitments:
        def __getattr__(self, name):  # pragma: no cover - must never be consulted
            raise AssertionError("competition items have no inference commitment")

    manifest = build_audit_manifest(
        evidence.scored_items, store=world.store,
        competition_input=evidence.competition_input,
        commitment_source=_NoInferenceCommitments(),
    )
    census = _census()
    snapshots = tuple(
        MinerSnapshot(
            uid=entry.uid, hotkey=entry.hotkey, coldkey=entry.coldkey, ip=entry.ip,
            track="compression", accumulate_score=0.0,
        )
        for entry in census.values()
    )
    log = EpochFinalizer(CONFIG, scorer_version="scorer-v1").build_log(
        epoch_id=2, close_block=719, snapshots=snapshots, burn_uid=94,
        audit_manifest=manifest, now=COMPLETED_AT,
        competition_result=evidence.result,
        competition_packet_scores=evidence.packet_scores,
    )
    bundle_source = InMemoryBundleSource()
    for bundle in (*world.bundles.values(), world.baseline_bundle):
        bundle_source.add(bundle)
    auditor = Auditor(
        AuditorConfig(auditor_hotkey="auditor-test", tokenomics=CONFIG, burn_uid=94),
        bundle_source, chain=world.anchor_chain,
    )
    return world, evidence, log, auditor


def test_ruled_result_finalizes_round_trips_and_audits_clean(tmp_path: Path) -> None:
    from vidaio.auditor.report import ItemVerdictKind
    from vidaio.epoch import EpochLog

    world, evidence, log, auditor = _finalized_log(tmp_path, {"crown_margin": 0.10})
    assert log.reward_window_state.kind is EmissionState.PODIUM
    reparsed = EpochLog.from_json(log.to_json())
    assert reparsed.to_json() == log.to_json()
    assert reparsed.competition_result == evidence.result
    assert reparsed.audit_manifest.competition_input.rules == CompetitionRulesInput(
        crown_margin=0.10
    )
    derived, _window, verdicts = auditor._competition_verdicts(
        reparsed, world.store, None, True
    )
    assert derived == evidence.result
    assert verdicts and all(v.verdict is ItemVerdictKind.PASS for v in verdicts), [
        (v.verdict, v.code, v.detail) for v in verdicts
    ]


def test_auditor_rejects_rules_that_differ_from_the_anchored_manifest(
    tmp_path: Path,
) -> None:
    from vidaio.auditor.report import ItemVerdictKind

    world, _evidence_, log, auditor = _finalized_log(tmp_path, {"crown_margin": 0.10})
    forged_input = log.audit_manifest.competition_input.model_copy(
        update={"rules": CompetitionRulesInput(crown_margin=0.05)}
    )
    forged = log.model_copy(
        update={
            "audit_manifest": log.audit_manifest.model_copy(
                update={"competition_input": forged_input}
            )
        }
    )
    _derived, _window, verdicts = auditor._competition_verdicts(
        forged, world.store, None, True
    )
    assert any(v.verdict is not ItemVerdictKind.PASS for v in verdicts)
    assert any("anchored manifest" in (v.detail or "") for v in verdicts)


def test_podium_rule_with_a_zero_scoring_contender_finalizes_and_audits(tmp_path: Path) -> None:
    """Regression from the first full rehearsal: a podium condition that excludes a
    committed contender must not make the epoch log unpublishable."""
    from vidaio.auditor.report import ItemVerdictKind

    world, evidence, log, auditor = _finalized_log(
        tmp_path, {"crown_margin": 0.05, "podium_min_margin": 0.0}
    )
    assert [c.hotkey for c in log.competition_result.contenders] == ["hk-a", "hk-b"]
    assert log.reward_window_state.podium_hotkeys == ("hk-a",)  # hk-b scored 0.0
    assert log.reward_window_state.kind is EmissionState.CROWN
    _derived, _window, verdicts = auditor._competition_verdicts(log, world.store, None, True)
    assert verdicts and all(v.verdict is ItemVerdictKind.PASS for v in verdicts), [
        (v.verdict, v.code, v.detail) for v in verdicts
    ]


def test_dry_run_reports_what_finalize_would_refuse(tmp_path: Path) -> None:
    from dataclasses import replace

    from vidaio.authority.finalizer import EpochFinalizer, build_audit_manifest
    from vidaio.epoch import EpochLogInvalid
    from vidaio.tokenomics.state import MinerSnapshot

    world = build_golden_world(tmp_path)
    evidence = build_competition_epoch_evidence(
        world.comp_conn, census_by_hotkey=_census(), store=world.store,
        tokenomics=CONFIG, through_time=COMPLETED_AT,
    )
    assert evidence is not None

    class _NoInference:
        def __getattr__(self, name):  # pragma: no cover
            raise AssertionError("no inference commitment is consulted")

    manifest = build_audit_manifest(
        evidence.scored_items, store=world.store,
        competition_input=evidence.competition_input, commitment_source=_NoInference(),
    )
    snapshots = tuple(
        MinerSnapshot(uid=e.uid, hotkey=e.hotkey, coldkey=e.coldkey, ip=e.ip,
                      track="compression", accumulate_score=0.0)
        for e in _census().values()
    )
    finalizer = EpochFinalizer(CONFIG, scorer_version="scorer-v1")
    kwargs = dict(
        epoch_id=2, close_block=719, snapshots=snapshots, burn_uid=94,
        audit_manifest=manifest, now=COMPLETED_AT,
        competition_packet_scores=evidence.packet_scores,
    )
    # A result that drops a committed contender is what the rehearsal produced.
    broken = replace(evidence.result, contenders=evidence.result.contenders[:1])
    with pytest.raises(EpochLogInvalid):
        finalizer.dry_run(competition_result=broken, **kwargs)
