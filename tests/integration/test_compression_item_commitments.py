"""Hidden compression clips are committed in the anchored manifest and re-proved."""

from __future__ import annotations

from pathlib import Path

import pytest

from vidaio.competition.config import CompetitionConfig
from vidaio.competition.epoch_evidence import build_competition_epoch_evidence
from vidaio.competition.item_commitment import compression_item_commitment
from vidaio.competition.manifest import ManifestBoundsError, validate_against_config
from vidaio.epoch import MinerCensusEntry
from vidaio.tokenomics.config import TokenomicsConfig

from integration_support import (
    COMPLETED_AT,
    ITEM_SHA256,
    build_golden_world,
    build_manifest,
)

CONFIG = TokenomicsConfig(competition_emissions_enabled=True)


def _commitment(manifest_id: str, index: int = 0, digest: str = ITEM_SHA256) -> str:
    return compression_item_commitment(
        competition_id=manifest_id, item_index=index, input_sha256=digest
    )


def _census() -> dict[str, MinerCensusEntry]:
    return {
        hotkey: MinerCensusEntry(
            uid=index + 1, hotkey=hotkey, coldkey=f"ck-{hotkey}", ip=f"10.0.0.{index + 1}"
        )
        for index, hotkey in enumerate(("hk-a", "hk-b"))
    }


def test_commitment_is_deterministic_and_position_bound() -> None:
    first = _commitment("comp-x", 0)
    assert first == _commitment("comp-x", 0)
    assert first != _commitment("comp-x", 1)
    assert first != _commitment("comp-y", 0)
    assert first != _commitment("comp-x", 0, "ab" * 32)
    with pytest.raises(ValueError):
        compression_item_commitment(competition_id="c", item_index=-1, input_sha256=ITEM_SHA256)


def test_production_envelope_requires_committed_compression_items() -> None:
    plain = build_manifest()
    validate_against_config(plain, CompetitionConfig())
    with pytest.raises(ManifestBoundsError, match="precommit"):
        validate_against_config(plain, CompetitionConfig(require_item_commitments=True))
    committed = build_manifest(
        evaluation_item_commitments=[_commitment(plain.competition_id)]
    )
    validate_against_config(committed, CompetitionConfig(require_item_commitments=True))
    with pytest.raises(ValueError):
        build_manifest(evaluation_item_commitments=[])


def test_committed_clip_flows_into_the_epoch_input(tmp_path: Path) -> None:
    competition_id = build_manifest().competition_id
    world = build_golden_world(
        tmp_path,
        manifest_overrides={
            "evaluation_item_commitments": [_commitment(competition_id)]
        },
    )
    evidence = build_competition_epoch_evidence(
        world.comp_conn, census_by_hotkey=_census(), store=world.store,
        tokenomics=CONFIG, through_time=COMPLETED_AT,
    )
    assert evidence is not None
    (item,) = evidence.competition_input.items
    assert item.input_sha256 == ITEM_SHA256
    assert item.item_commitment == _commitment(competition_id)


def test_a_swapped_clip_cannot_be_ingested(tmp_path: Path) -> None:
    competition_id = build_manifest().competition_id
    with pytest.raises(ValueError, match="do not match the manifest commitment"):
        build_golden_world(
            tmp_path,
            manifest_overrides={
                "evaluation_item_commitments": [
                    _commitment(competition_id, 0, "cd" * 32)
                ]
            },
        )


def test_uncommitted_extra_clip_is_refused(tmp_path: Path) -> None:
    from vidaio.competition import repository as comp_repo

    competition_id = build_manifest().competition_id
    world = build_golden_world(
        tmp_path,
        manifest_overrides={
            "evaluation_item_commitments": [_commitment(competition_id)]
        },
    )
    with pytest.raises(Exception, match="no precommitted manifest entry|not|COMPLETED"):
        comp_repo.add_evaluation_item(
            world.comp_conn, competition_id, item_index=1,
            input_sha256="ef" * 32, input_bytes=10, length_seconds=10.0,
            threshold_commitment="12" * 32, challenge_id="extra", now=COMPLETED_AT,
        )


def test_auditor_reopens_the_anchored_clip_commitment(tmp_path: Path) -> None:
    from vidaio.auditor.report import ItemVerdictKind

    from test_competition_result_rules import _finalized_log

    competition_id = build_manifest().competition_id
    world, _evidence, log, auditor = _finalized_log(
        tmp_path, None, evaluation_item_commitments=[_commitment(competition_id)]
    )
    _derived, _window, verdicts = auditor._competition_verdicts(
        log, world.store, None, True
    )
    assert verdicts and all(v.verdict is ItemVerdictKind.PASS for v in verdicts), [
        (v.verdict, v.code, v.detail) for v in verdicts
    ]

    comp_input = log.audit_manifest.competition_input
    forged_item = comp_input.items[0].model_copy(update={"item_commitment": "ab" * 32})
    forged = log.model_copy(
        update={
            "audit_manifest": log.audit_manifest.model_copy(
                update={
                    "competition_input": comp_input.model_copy(
                        update={"items": (forged_item,)}
                    )
                }
            )
        }
    )
    _derived, _window, verdicts = auditor._competition_verdicts(
        forged, world.store, None, True
    )
    assert any(v.verdict is not ItemVerdictKind.PASS for v in verdicts)
    assert any("anchored manifest commitment" in (v.detail or "") for v in verdicts)
