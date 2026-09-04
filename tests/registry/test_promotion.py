"""The retired schema-v13 promotion path is a fail-closed tombstone."""

from __future__ import annotations

import sqlite3

import pytest

from vidaio.audit.store import ArtifactKind, ArtifactRef, LocalFsStore
from vidaio.registry.competition_source import SqliteCompetitionSource
from vidaio.registry.promotion import PromotionPipeline
from vidaio.registry.registry import LegacyRegistryWriteDisabledError

from registry_support import NOW


def test_legacy_pipeline_cannot_activate_executable_state(
    conn: sqlite3.Connection,
    comp_conn: sqlite3.Connection,
    store: LocalFsStore,
) -> None:
    pipeline = PromotionPipeline(store, SqliteCompetitionSource(comp_conn))
    submission = ArtifactRef(
        digest="ab" * 32,
        kind=ArtifactKind.SUBMISSION_ARCHIVE,
        byte_size=1,
        backend_key="unused",
    )

    with pytest.raises(LegacyRegistryWriteDisabledError, match="schema-v13"):
        pipeline.evaluate_and_promote(
            conn,
            competition_id="caller-selected",
            artifact_ref=submission,
            audit_bundle_ref=None,
            now=NOW,
        )

    assert conn.execute("SELECT COUNT(*) FROM champions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0] == 0


def test_legacy_pipeline_is_not_exported_from_package() -> None:
    import vidaio.registry as public

    assert not hasattr(public, "PromotionPipeline")
    assert "PromotionPipeline" not in public.__all__
