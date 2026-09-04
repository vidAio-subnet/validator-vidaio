"""Authenticated, bounded competition-item ingest through the control API."""

from __future__ import annotations

import hashlib

import httpx
import pytest

from vidaio.audit.commitments import reward_parameter_digest
from vidaio.competition import evaluation_item_commitment
from vidaio.tokenomics.config import TokenomicsConfig

from orchestrator_support import (
    BASELINE,
    T0,
    RecordingChain,
    build_manifest,
)

TOKEN = "item-ingest-control-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
THRESHOLD_COMMITMENT = "f" * 64
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


class Clock:
    def __init__(self, value=T0) -> None:
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def item_control(orchestrator_factory):
    orch = orchestrator_factory(
        chain=RecordingChain(), clock=Clock(), control_token=TOKEN
    )
    assert orch.control_app is not None
    return orch, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch.control_app),
        base_url="http://control",
    )


def _body(input_name: str, *, item_index: int = 0, **overrides):
    return {
        "input_name": input_name,
        "item_index": item_index,
        "threshold_commitment": THRESHOLD_COMMITMENT,
        "challenge_id": "chal-control-ingest",
        "length_seconds": 1.0,
        **overrides,
    }


def _item_count(orch, competition_id: str) -> int:
    row = orch.conn.execute(
        "SELECT COUNT(*) AS n FROM evaluation_items WHERE competition_id = ?",
        (competition_id,),
    ).fetchone()
    assert row is not None
    return int(row["n"])


async def test_item_route_authenticates_before_competition_or_body_validation(
    item_control,
) -> None:
    _orch, client = item_control
    async with client:
        response = await client.post(
            "/competitions/does-not-exist/items", json={}
        )
    assert response.status_code == 401


async def test_compression_item_ingests_one_basename_before_anchor(item_control) -> None:
    orch, client = item_control
    manifest = build_manifest("comp-control-compression")
    orch.create_competition(manifest, T0)
    payload = b"compression-evaluation-input"
    (orch.ingest_dir / "input.bin").write_bytes(payload)

    async with client:
        response = await client.post(
            f"/competitions/{manifest.competition_id}/items",
            headers=AUTH,
            json=_body("input.bin"),
        )
    assert response.status_code == 201, response.text
    digest = hashlib.sha256(payload).hexdigest()
    assert response.json() == {
        "item_id": response.json()["item_id"],
        "item_index": 0,
        "input_sha256": digest,
        "reference_sha256": digest,
        "upscale_factor": None,
        "target_width": None,
        "target_height": None,
        "item_commitment": None,
    }
    row = orch.conn.execute(
        "SELECT * FROM evaluation_items WHERE item_id = ?",
        (response.json()["item_id"],),
    ).fetchone()
    assert row is not None
    assert row["input_sha256"] == row["reference_sha256"] == digest
    assert (orch.inputs_dir / digest).read_bytes() == payload


async def test_upscaling_item_ingests_distinct_committed_reference_and_factor(
    item_control,
) -> None:
    orch, client = item_control
    competition_id = "comp-control-upscaling"
    miner_input = b"low-resolution-input"
    reference = b"pristine-high-resolution-reference"
    input_digest = hashlib.sha256(miner_input).hexdigest()
    reference_digest = hashlib.sha256(reference).hexdigest()
    commitment = evaluation_item_commitment(
        competition_id=competition_id,
        item_index=0,
        reference_sha256=reference_digest,
        input_sha256=input_digest,
        upscale_factor=4,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    manifest = build_manifest(
        competition_id,
        track="upscaling",
        allowed_upscale_factors=[2, 4],
        evaluation_item_commitments=[commitment],
    )
    orch.create_competition(manifest, T0)
    (orch.ingest_dir / "input-low.bin").write_bytes(miner_input)
    (orch.ingest_dir / "reference-high.bin").write_bytes(reference)

    async with client:
        response = await client.post(
            f"/competitions/{competition_id}/items",
            headers=AUTH,
            json=_body(
                "input-low.bin",
                reference_name="reference-high.bin",
                upscale_factor=4,
                target_width=TARGET_WIDTH,
                target_height=TARGET_HEIGHT,
            ),
        )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["input_sha256"] == input_digest
    assert body["reference_sha256"] == reference_digest
    assert body["input_sha256"] != body["reference_sha256"]
    assert body["upscale_factor"] == 4
    assert body["target_width"] == TARGET_WIDTH
    assert body["target_height"] == TARGET_HEIGHT
    assert body["item_commitment"] == commitment
    assert (orch.inputs_dir / input_digest).read_bytes() == miner_input
    assert (orch.inputs_dir / reference_digest).read_bytes() == reference


@pytest.mark.parametrize("unsafe_name", ["../outside.bin", "nested/input.bin", ".", ".."])
async def test_item_ingest_rejects_non_basename_paths(
    item_control, tmp_path, unsafe_name: str
) -> None:
    orch, client = item_control
    manifest = build_manifest("comp-control-traversal")
    orch.create_competition(manifest, T0)
    (tmp_path / "outside.bin").write_bytes(b"must-never-be-read")

    async with client:
        response = await client.post(
            f"/competitions/{manifest.competition_id}/items",
            headers=AUTH,
            json=_body(unsafe_name),
        )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "item_ingest_refused"
    assert _item_count(orch, manifest.competition_id) == 0


async def test_item_ingest_rejects_symlink_without_reading_target(
    item_control, tmp_path
) -> None:
    orch, client = item_control
    manifest = build_manifest("comp-control-symlink")
    orch.create_competition(manifest, T0)
    target = tmp_path / "outside-secret.bin"
    target.write_bytes(b"secret-outside-ingest-root")
    (orch.ingest_dir / "input.bin").symlink_to(target)

    async with client:
        response = await client.post(
            f"/competitions/{manifest.competition_id}/items",
            headers=AUTH,
            json=_body("input.bin"),
        )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "item_ingest_refused"
    assert _item_count(orch, manifest.competition_id) == 0
    assert target.read_bytes() == b"secret-outside-ingest-root"


async def test_item_ingest_rejects_oversize_regular_file(
    item_control, monkeypatch
) -> None:
    # Keep the test tiny: exercise the real bounded stream with a five-byte file
    # and a four-byte cap instead of creating/reading a multi-gigabyte fixture.
    import vidaio.competition.orchestrator.service as service

    monkeypatch.setattr(service, "MAX_COMPETITION_INGEST_BYTES", 4)
    orch, client = item_control
    manifest = build_manifest("comp-control-oversize")
    orch.create_competition(manifest, T0)
    (orch.ingest_dir / "input.bin").write_bytes(b"12345")

    async with client:
        response = await client.post(
            f"/competitions/{manifest.competition_id}/items",
            headers=AUTH,
            json=_body("input.bin"),
        )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "item_ingest_refused"
    assert _item_count(orch, manifest.competition_id) == 0


async def test_item_ingest_is_immutable_after_anchor(item_control) -> None:
    orch, client = item_control
    manifest = build_manifest("comp-control-anchored", baseline=BASELINE)
    orch.create_competition(manifest, T0)
    (orch.ingest_dir / "first.bin").write_bytes(b"first-input")
    (orch.ingest_dir / "late.bin").write_bytes(b"late-input")

    async with client:
        seeded = await client.post(
            f"/competitions/{manifest.competition_id}/items",
            headers=AUTH,
            json=_body("first.bin"),
        )
        assert seeded.status_code == 201, seeded.text
        anchored = await client.post(
            f"/competitions/{manifest.competition_id}/anchor",
                headers=AUTH,
                json={
                    "reward_param_digest": reward_parameter_digest(TokenomicsConfig()),
                },
        )
        assert anchored.status_code == 200, anchored.text
        late = await client.post(
            f"/competitions/{manifest.competition_id}/items",
            headers=AUTH,
            json=_body("late.bin", item_index=1),
        )
    assert late.status_code == 422, late.text
    assert late.json()["detail"]["code"] == "item_ingest_refused"
    assert _item_count(orch, manifest.competition_id) == 1
