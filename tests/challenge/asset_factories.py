"""Test helpers: asset construction for the challenge-pool tests."""

import sqlite3

from vidaio.challenge import Asset, add_asset


def mk_asset(i: int = 0, split: str = "challenge", **overrides) -> Asset:
    fields = dict(
        id=f"asset_{i:04d}",
        content_digest=f"{i:064x}",
        perceptual_fingerprint=f"fp_{i:04d}",
        source_url=f"https://example.com/video_{i}.mp4",
        license_basis="cc0",
        ingest_date="2026-08-20T00:00:00Z",
        creator="alice",
        source="shoot-1",
        subject="street",
        scene=f"scene-{i}",
        resolution_tag="1080p",
        motion_tag="medium",
        content_type_tag="sports",
        metadata_stripped=True,
        split=split,
        status="fresh",
        use_count=0,
    )
    fields.update(overrides)
    return Asset.model_validate(fields)


def add_assets(conn: sqlite3.Connection, *assets: Asset) -> None:
    for a in assets:
        add_asset(conn, a)
